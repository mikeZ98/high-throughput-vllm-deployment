"""Benchmark aggregation: per-request records -> typed :class:`BenchmarkResult`.

The collector is append-only: ``record()`` is a single ``list.append``, which
is atomic under the GIL and between gevent/asyncio switches, so asyncio tasks,
Locust greenlets and threads can share one instance without locks. All
aggregation happens once, in ``result()``.

Percentiles use linear interpolation between closest ranks (NumPy's default
``linear`` method, as used by vLLM's ``benchmark_serving``) and are clamped to
the bracketing samples, so ``p50 <= p90 <= p99 <= max`` holds exactly even
under floating-point rounding.
"""

from __future__ import annotations

import math
import socket
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from high_throughput_vllm_deployment.core.errors import (
    EngineHTTPError,
    EngineProtocolError,
    EngineRequestError,
    EngineTimeoutError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.schemas.enums import ErrorCategory, ModelBackend
from high_throughput_vllm_deployment.schemas.telemetry import BenchmarkResult, LatencyPercentiles

Clock = Callable[[], float]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an ascending, non-empty sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    rank = (len(sorted_values) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    lo_value, hi_value = sorted_values[low], sorted_values[high]
    value = lo_value + (hi_value - lo_value) * (rank - low)
    return min(max(value, lo_value), hi_value)


def summarize(samples_s: Sequence[float]) -> LatencyPercentiles | None:
    """Summarise latencies given in seconds as milliseconds; ``None`` without samples."""
    if not samples_s:
        return None
    values = sorted(sample * 1000.0 for sample in samples_s)
    mean = min(max(math.fsum(values) / len(values), values[0]), values[-1])
    return LatencyPercentiles(
        mean_ms=mean,
        p50_ms=percentile(values, 50),
        p90_ms=percentile(values, 90),
        p99_ms=percentile(values, 99),
        max_ms=values[-1],
    )


# --------------------------------------------------------------------------- #
# Error taxonomy                                                              #
# --------------------------------------------------------------------------- #


def categorize_status(status_code: int) -> ErrorCategory:
    """Map an HTTP status (0 = no response) to an :class:`ErrorCategory`."""
    if status_code == 0 or status_code in (502, 503, 504):
        return ErrorCategory.UNAVAILABLE
    if 400 <= status_code < 500:
        return ErrorCategory.CLIENT_ERROR
    if 500 <= status_code < 600:
        return ErrorCategory.SERVER_ERROR
    return ErrorCategory.OTHER


def categorize_exception(exc: BaseException) -> ErrorCategory:
    """Map engine, socket and HTTP-library exceptions to an :class:`ErrorCategory`.

    Third-party timeouts (``requests``, ``urllib3``, ``gevent``) are recognised
    by class name so this module stays free of those imports.
    """
    if isinstance(exc, EngineRequestError):
        return ErrorCategory.CLIENT_ERROR
    if isinstance(exc, EngineHTTPError):
        return categorize_status(exc.status_code)
    if isinstance(exc, EngineTimeoutError | TimeoutError | socket.timeout) or any(
        "Timeout" in cls.__name__ for cls in type(exc).__mro__
    ):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, EngineUnavailableError | ConnectionError) or any(
        cls.__name__ in {"ConnectionError", "HTTPConnectionClosed"} for cls in type(exc).__mro__
    ):
        return ErrorCategory.UNAVAILABLE
    if isinstance(exc, EngineProtocolError | ValueError | RecursionError):
        return ErrorCategory.PROTOCOL
    return ErrorCategory.OTHER


# --------------------------------------------------------------------------- #
# Records and collector                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """Timing and token accounting for one request (times from one monotonic clock)."""

    start_s: float
    end_s: float
    success: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float | None = None
    itl_s: tuple[float, ...] = ()
    error: ErrorCategory | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        if self.success == (self.error is not None):
            raise ValueError("a record carries an error category iff it failed")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        if self.ttft_s is not None and not 0.0 <= self.ttft_s <= self.e2e_s:
            raise ValueError("ttft_s must lie within [0, e2e_s]")
        if any(gap < 0 for gap in self.itl_s):
            raise ValueError("inter-token gaps must be >= 0")

    @property
    def e2e_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first (vLLM's TPOT)."""
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)

    @classmethod
    def failure(
        cls,
        start_s: float,
        end_s: float,
        error: ErrorCategory,
        detail: str = "",
        *,
        input_tokens: int = 0,
    ) -> RequestRecord:
        return cls(
            start_s=start_s,
            end_s=end_s,
            success=False,
            error=error,
            detail=detail,
            input_tokens=input_tokens,
        )


class MetricsCollector:
    """Lock-free, append-only accumulator of :class:`RequestRecord` objects."""

    def __init__(self, *, backend: ModelBackend, model: str, clock: Clock = time.perf_counter):
        self.backend = backend
        self.model = model
        self._clock = clock
        self._records: list[RequestRecord] = []
        self._began: float | None = None
        self._ended: float | None = None

    def begin(self) -> None:
        """Open the measurement window (defaults to the span of the records)."""
        self._began = self._clock()
        self._ended = None

    def end(self) -> None:
        self._ended = self._clock()

    def record(self, record: RequestRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> tuple[RequestRecord, ...]:
        return tuple(self._records)

    def duration_s(self) -> float:
        if self._began is not None:
            ended = self._ended if self._ended is not None else self._clock()
            return max(0.0, ended - self._began)
        if not self._records:
            return 0.0
        return max(r.end_s for r in self._records) - min(r.start_s for r in self._records)

    def result(self) -> BenchmarkResult:
        records = self.records
        ok = [r for r in records if r.success]
        errors = Counter(r.error for r in records if r.error is not None)
        return BenchmarkResult(
            backend=self.backend,
            model=self.model,
            total_requests=len(records),
            failed_requests=len(records) - len(ok),
            duration_s=self.duration_s(),
            total_input_tokens=sum(r.input_tokens for r in ok),
            total_output_tokens=sum(r.output_tokens for r in ok),
            ttft=summarize([r.ttft_s for r in ok if r.ttft_s is not None]),
            tpot=summarize([tpot for r in ok if (tpot := r.tpot_s) is not None]),
            itl=summarize([gap for r in ok for gap in r.itl_s]),
            e2e_latency=summarize([r.e2e_s for r in ok]),
            errors=dict(errors),
        )
