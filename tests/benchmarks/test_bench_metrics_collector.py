"""Aggregator: exact percentiles, monotonic summaries, zero-safety, error taxonomy."""

from __future__ import annotations

import math
import socket
import statistics
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from high_throughput_vllm_deployment.benchmarks.metrics_collector import (
    MetricsCollector,
    RequestRecord,
    categorize_exception,
    categorize_status,
    percentile,
    summarize,
)
from high_throughput_vllm_deployment.core.errors import (
    EngineHTTPError,
    EngineProtocolError,
    EngineRequestError,
    EngineTimeoutError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.schemas.enums import ErrorCategory, ModelBackend
from high_throughput_vllm_deployment.schemas.telemetry import BenchmarkResult

latencies = st.lists(
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=200,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _collector(clock: FakeClock | None = None) -> MetricsCollector:
    return MetricsCollector(backend=ModelBackend.VLLM, model="m", clock=clock or FakeClock())


# --------------------------------------------------------------------------- #
# Percentiles                                                                 #
# --------------------------------------------------------------------------- #


def test_known_stream_yields_numpy_linear_percentiles() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 90) == pytest.approx(90.1)
    assert percentile(values, 99) == pytest.approx(99.01)
    assert percentile(values, 0) == 1.0 and percentile(values, 100) == 100.0
    assert percentile([42.0], 99) == 42.0


def test_summary_of_known_stream_in_milliseconds() -> None:
    summary = summarize([v / 1000 for v in range(1, 101)])  # 1..100 ms
    assert summary is not None
    assert summary.mean_ms == pytest.approx(50.5)
    assert (summary.p50_ms, summary.p90_ms, summary.p99_ms) == pytest.approx((50.5, 90.1, 99.01))
    assert summary.max_ms == pytest.approx(100.0)


@pytest.mark.parametrize(("values", "q"), [([], 50.0), ([1.0], -1.0), ([1.0], 100.5)])
def test_percentile_rejects_invalid_input(values: list[float], q: float) -> None:
    with pytest.raises(ValueError):
        percentile(values, q)


def test_summarize_without_samples_is_none() -> None:
    assert summarize([]) is None


def test_mean_never_exceeds_max_under_rounding() -> None:
    # Three 0.1 ms samples: fsum([0.1] * 3) / 3 == 0.10000000000000002 > max, which
    # the LatencyPercentiles contract would reject without the clamp.
    summary = summarize([0.0001] * 3)
    assert summary is not None
    assert summary.mean_ms == summary.max_ms == 0.1


@given(latencies)
def test_summaries_are_monotonic_and_match_the_statistics_oracle(samples: list[float]) -> None:
    summary = summarize(samples)
    assert summary is not None
    assert summary.p50_ms <= summary.p90_ms <= summary.p99_ms <= summary.max_ms
    ms = sorted(s * 1000 for s in samples)
    assert ms[0] <= summary.mean_ms <= ms[-1]
    if len(ms) >= 2:
        cuts = statistics.quantiles(ms, n=100, method="inclusive")
        for got, q in ((summary.p50_ms, 50), (summary.p90_ms, 90), (summary.p99_ms, 99)):
            assert got == pytest.approx(cuts[q - 1], rel=1e-9, abs=1e-9)


@given(latencies, st.floats(0, 100), st.floats(0, 100))
def test_percentile_is_monotonic_in_q(samples: list[float], q1: float, q2: float) -> None:
    ordered = sorted(samples)
    low, high = sorted((q1, q2))
    assert percentile(ordered, low) <= percentile(ordered, high)
    assert ordered[0] <= percentile(ordered, low) <= ordered[-1]


# --------------------------------------------------------------------------- #
# Error taxonomy                                                              #
# --------------------------------------------------------------------------- #


class ReadTimeout(Exception):  # stands in for requests/urllib3 timeouts
    pass


class HTTPConnectionClosed(Exception):  # geventhttpclient
    pass


class ConnectionError(OSError):  # noqa: A001 - mimics requests.exceptions.ConnectionError
    pass


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (0, ErrorCategory.UNAVAILABLE),
        (400, ErrorCategory.CLIENT_ERROR),
        (429, ErrorCategory.CLIENT_ERROR),
        (500, ErrorCategory.SERVER_ERROR),
        (502, ErrorCategory.UNAVAILABLE),
        (503, ErrorCategory.UNAVAILABLE),
        (504, ErrorCategory.UNAVAILABLE),
        (302, ErrorCategory.OTHER),
    ],
)
def test_status_categories(status: int, category: ErrorCategory) -> None:
    assert categorize_status(status) is category


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (EngineRequestError("bad", status_code=404), ErrorCategory.CLIENT_ERROR),
        (EngineHTTPError("boom", status_code=500), ErrorCategory.SERVER_ERROR),
        (EngineHTTPError("busy", status_code=503), ErrorCategory.UNAVAILABLE),
        (EngineTimeoutError("slow"), ErrorCategory.TIMEOUT),
        (TimeoutError(), ErrorCategory.TIMEOUT),
        (socket.timeout(), ErrorCategory.TIMEOUT),
        (ReadTimeout(), ErrorCategory.TIMEOUT),
        (EngineUnavailableError("down"), ErrorCategory.UNAVAILABLE),
        (ConnectionRefusedError(), ErrorCategory.UNAVAILABLE),
        (ConnectionError(), ErrorCategory.UNAVAILABLE),
        (HTTPConnectionClosed(), ErrorCategory.UNAVAILABLE),
        (EngineProtocolError("garbled"), ErrorCategory.PROTOCOL),
        (ValueError("bad json"), ErrorCategory.PROTOCOL),
        (RecursionError("absurdly deep JSON"), ErrorCategory.PROTOCOL),
        (RuntimeError("?"), ErrorCategory.OTHER),
    ],
)
def test_exception_categories(exc: BaseException, category: ErrorCategory) -> None:
    assert categorize_exception(exc) is category


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start_s": 2.0, "end_s": 1.0}, "end_s"),
        ({"success": False}, "error category"),
        ({"error": ErrorCategory.OTHER}, "error category"),
        ({"output_tokens": -1}, "token counts"),
        ({"ttft_s": 2.0}, "ttft_s"),
        ({"ttft_s": -0.1}, "ttft_s"),
        ({"itl_s": (0.1, -0.1)}, "gaps"),
    ],
)
def test_record_validation(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {"start_s": 0.0, "end_s": 1.0}
    with pytest.raises(ValueError, match=match):
        RequestRecord(**(base | kwargs))  # type: ignore[arg-type]


def test_tpot_is_defined_only_for_streams_with_two_or_more_tokens() -> None:
    assert RequestRecord(0.0, 1.0, output_tokens=5, ttft_s=0.2).tpot_s == pytest.approx(0.2)
    assert RequestRecord(0.0, 1.0, output_tokens=1, ttft_s=0.2).tpot_s is None
    assert RequestRecord(0.0, 1.0, output_tokens=5).tpot_s is None


# --------------------------------------------------------------------------- #
# Collector                                                                   #
# --------------------------------------------------------------------------- #


def test_exact_aggregation_of_a_mixed_run() -> None:
    clock = FakeClock(10.0)
    collector = _collector(clock)
    collector.begin()
    collector.record(
        RequestRecord(10.0, 11.0, input_tokens=4, output_tokens=5, ttft_s=0.2, itl_s=(0.2,) * 4)
    )
    collector.record(RequestRecord(10.5, 12.5, input_tokens=6, output_tokens=3))
    collector.record(RequestRecord.failure(11.0, 11.5, ErrorCategory.TIMEOUT, input_tokens=9))
    clock.now = 14.0
    collector.end()
    clock.now = 99.0  # later reads must not move the closed window

    result = collector.result()
    BenchmarkResult.model_validate_json(result.model_dump_json())
    assert (result.total_requests, result.failed_requests, result.duration_s) == (3, 1, 4.0)
    assert (result.total_input_tokens, result.total_output_tokens) == (10, 8)
    assert result.errors == {ErrorCategory.TIMEOUT: 1}
    assert result.ttft is not None and result.ttft.max_ms == pytest.approx(200.0)
    assert result.tpot is not None and result.tpot.mean_ms == pytest.approx(200.0)
    assert result.itl is not None and result.itl.p99_ms == pytest.approx(200.0)
    assert result.e2e_latency is not None
    assert (result.e2e_latency.p50_ms, result.e2e_latency.max_ms) == pytest.approx((1500, 2000))
    assert result.request_throughput == pytest.approx(0.5)
    assert result.output_token_throughput == pytest.approx(2.0)


def test_zero_requests_is_zero_safe() -> None:
    result = _collector().result()
    assert (result.total_requests, result.duration_s, result.failure_rate) == (0, 0.0, 0.0)
    assert result.request_throughput == result.output_token_throughput == 0.0
    assert result.e2e_latency is None and result.ttft is None


def test_total_failure_is_zero_safe() -> None:
    collector = _collector()
    for i in range(5):
        collector.record(RequestRecord.failure(float(i), i + 0.5, ErrorCategory.UNAVAILABLE))
    result = collector.result()
    assert result.failure_rate == 1.0
    assert result.successful_requests == 0
    assert result.request_throughput == 0.0 and result.output_token_throughput == 0.0
    assert result.e2e_latency is None and result.errors == {ErrorCategory.UNAVAILABLE: 5}
    assert result.duration_s == pytest.approx(4.5)  # span of the records


def test_open_window_is_measured_up_to_now() -> None:
    clock = FakeClock(1.0)
    collector = _collector(clock)
    collector.begin()
    clock.now = 3.5
    assert collector.duration_s() == 2.5
    assert collector.records == ()


def test_concurrent_appends_from_threads_are_all_kept() -> None:
    collector = _collector()

    def worker() -> None:
        for _ in range(2000):
            collector.record(RequestRecord(0.0, 1.0, output_tokens=1))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result = collector.result()
    assert result.total_requests == 16_000
    assert result.total_output_tokens == 16_000
    assert math.isclose(result.duration_s, 1.0)
