"""Engine telemetry snapshots and Locust/benchmark result contracts.

Units are explicit in field names: ``_pct`` is a percentage in ``[0, 100]``,
``_ms`` is milliseconds, ``_s`` is seconds.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from high_throughput_vllm_deployment.schemas._base import (
    ContractModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    Percentage,
    PositiveInt,
)
from high_throughput_vllm_deployment.schemas.enums import LaxModelBackend, SchedulerState

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
    """Latency distribution summary in milliseconds; percentiles must be monotonic."""

    p50_ms: NonNegativeFloat
    p90_ms: NonNegativeFloat
    p99_ms: NonNegativeFloat

    @model_validator(mode="after")
    def _monotonic(self) -> Self:
        if not self.p50_ms <= self.p90_ms <= self.p99_ms:
            raise ValueError("percentiles must satisfy p50 <= p90 <= p99")
        return self

    def dominates(self, other: LatencyPercentiles) -> bool:
        """True if every percentile is >= the corresponding one in *other*."""
        return (
            self.p50_ms >= other.p50_ms
            and self.p90_ms >= other.p90_ms
            and self.p99_ms >= other.p99_ms
        )


class BenchmarkResult(ContractModel):
    """Aggregate outcome of one load-test run against a serving backend.

    Latency percentiles are computed over successful requests:

    * ``ttft`` - time to first token;
    * ``tpot`` - time per output token, excluding the first;
    * ``itl``  - inter-token latency between consecutive streamed tokens;
    * ``e2e_latency`` - full request latency.

    Since every request's end-to-end latency is at least its TTFT, each
    ``e2e_latency`` percentile must be >= the matching ``ttft`` percentile.
    """

    backend: LaxModelBackend
    model: NonEmptyStr
    total_requests: PositiveInt
    failed_requests: NonNegativeInt
    duration_s: Annotated[FiniteFloat, Field(gt=0.0)]
    total_input_tokens: NonNegativeInt
    total_output_tokens: NonNegativeInt
    ttft: LatencyPercentiles
    tpot: LatencyPercentiles
    itl: LatencyPercentiles
    e2e_latency: LatencyPercentiles

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.failed_requests > self.total_requests:
            raise ValueError("failed_requests cannot exceed total_requests")
        if not self.e2e_latency.dominates(self.ttft):
            raise ValueError("e2e_latency percentiles must be >= ttft percentiles")
        return self

    @property
    def successful_requests(self) -> int:
        return self.total_requests - self.failed_requests

    @property
    def failure_rate(self) -> float:
        return self.failed_requests / self.total_requests

    @property
    def request_throughput(self) -> float:
        """Successful requests per second."""
        return self.successful_requests / self.duration_s

    @property
    def output_token_throughput(self) -> float:
        """Generated tokens per second."""
        return self.total_output_tokens / self.duration_s
