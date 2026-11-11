"""Engine telemetry snapshots and Locust/benchmark result contracts.

Units are explicit in field names: ``_pct`` is a percentage in ``[0, 100]``,
``_ms`` is milliseconds, ``_s`` is seconds.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from high_throughput_vllm_deployment.schemas._base import (
    ContractModel,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    Percentage,
    PositiveInt,
)
from high_throughput_vllm_deployment.schemas.enums import (
    LaxErrorCategory,
    LaxModelBackend,
    SchedulerState,
)

# --------------------------------------------------------------------------- #
# Cluster telemetry                                                           #
# --------------------------------------------------------------------------- #


class KvCacheBlocks(ContractModel):
    """Paged-attention KV cache block allocation on GPU and CPU (swap space)."""

    block_size: PositiveInt = Field(description="Tokens stored per KV cache block.")
    total_gpu_blocks: NonNegativeInt
    used_gpu_blocks: NonNegativeInt
    total_cpu_blocks: NonNegativeInt
    used_cpu_blocks: NonNegativeInt

    @model_validator(mode="after")
    def _used_within_total(self) -> Self:
        if self.used_gpu_blocks > self.total_gpu_blocks:
            raise ValueError("used_gpu_blocks cannot exceed total_gpu_blocks")
        if self.used_cpu_blocks > self.total_cpu_blocks:
            raise ValueError("used_cpu_blocks cannot exceed total_cpu_blocks")
        return self

    @property
    def free_gpu_blocks(self) -> int:
        return self.total_gpu_blocks - self.used_gpu_blocks

    @property
    def gpu_token_capacity(self) -> int:
        """Maximum number of tokens the GPU KV cache can hold."""
        return self.total_gpu_blocks * self.block_size


class ClusterTelemetry(ContractModel):
    """Point-in-time scheduler and KV cache snapshot of a vLLM engine."""

    active_requests: NonNegativeInt = Field(description="Requests in the RUNNING queue.")
    waiting_requests: NonNegativeInt = Field(description="Requests in the WAITING queue.")
    swapped_requests: NonNegativeInt = Field(
        default=0, description="Requests preempted to CPU (SWAPPED queue)."
    )
    gpu_cache_usage_pct: Percentage
    cpu_cache_usage_pct: Percentage
    kv_cache: KvCacheBlocks

    @property
    def requests_by_state(self) -> dict[SchedulerState, int]:
        return {
            SchedulerState.WAITING: self.waiting_requests,
            SchedulerState.RUNNING: self.active_requests,
            SchedulerState.SWAPPED: self.swapped_requests,
        }

    @property
    def total_requests(self) -> int:
        return self.active_requests + self.waiting_requests + self.swapped_requests


# --------------------------------------------------------------------------- #
# Benchmark results                                                           #
# --------------------------------------------------------------------------- #


class LatencyPercentiles(ContractModel):
    """Latency distribution summary in milliseconds.

    Order statistics must be monotonic (``p50 <= p90 <= p99 <= max``) and the
    mean cannot exceed the maximum.
    """

    mean_ms: NonNegativeFloat
    p50_ms: NonNegativeFloat
    p90_ms: NonNegativeFloat
    p99_ms: NonNegativeFloat
    max_ms: NonNegativeFloat

    @model_validator(mode="after")
    def _monotonic(self) -> Self:
        if not self.p50_ms <= self.p90_ms <= self.p99_ms <= self.max_ms:
            raise ValueError("percentiles must satisfy p50 <= p90 <= p99 <= max")
        if self.mean_ms > self.max_ms:
            raise ValueError("mean cannot exceed max")
        return self

    def dominates(self, other: LatencyPercentiles) -> bool:
        """True if every statistic is >= the corresponding one in *other*."""
        return (
            self.mean_ms >= other.mean_ms
            and self.p50_ms >= other.p50_ms
            and self.p90_ms >= other.p90_ms
            and self.p99_ms >= other.p99_ms
            and self.max_ms >= other.max_ms
        )


class BenchmarkResult(ContractModel):
    """Aggregate outcome of one load-test run against a serving backend.

    Latency summaries are computed over successful requests and are ``None``
    when there is no sample to summarise:

    * ``ttft`` - time to first token (streaming requests only);
    * ``tpot`` - time per output token after the first (streaming, >= 2 tokens);
    * ``itl``  - gaps between consecutive streamed tokens;
    * ``e2e_latency`` - full request latency (present iff any request succeeded).

    Per request, TTFT <= end-to-end latency. The summaries are deliberately not
    cross-checked: TTFT covers streaming requests only while ``e2e_latency``
    covers every success, so in mixed workloads they summarise different
    populations (use :meth:`LatencyPercentiles.dominates` for same-population checks).
    ``errors`` breaks ``failed_requests`` down by category and must sum to it.
    Derived rates are zero-safe: an empty or instantaneous run reports ``0.0``.
    """

    backend: LaxModelBackend
    model: NonEmptyStr
    total_requests: NonNegativeInt
    failed_requests: NonNegativeInt
    duration_s: NonNegativeFloat
    total_input_tokens: NonNegativeInt
    total_output_tokens: NonNegativeInt
    ttft: LatencyPercentiles | None = None
    tpot: LatencyPercentiles | None = None
    itl: LatencyPercentiles | None = None
    e2e_latency: LatencyPercentiles | None = None
    errors: dict[LaxErrorCategory, NonNegativeInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.failed_requests > self.total_requests:
            raise ValueError("failed_requests cannot exceed total_requests")
        if sum(self.errors.values()) != self.failed_requests:
            raise ValueError("errors must account for exactly failed_requests")
        if (self.e2e_latency is None) != (self.successful_requests == 0):
            raise ValueError("e2e_latency must be present iff some request succeeded")
        return self

    @property
    def successful_requests(self) -> int:
        return self.total_requests - self.failed_requests

    @property
    def failure_rate(self) -> float:
        return self.failed_requests / self.total_requests if self.total_requests else 0.0

    @property
    def request_throughput(self) -> float:
        """Successful requests per second."""
        return self.successful_requests / self.duration_s if self.duration_s else 0.0

    @property
    def output_token_throughput(self) -> float:
        """Generated tokens per second."""
        return self.total_output_tokens / self.duration_s if self.duration_s else 0.0
