"""Cluster telemetry and benchmark result contracts."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from high_throughput_vllm_deployment.schemas.enums import ModelBackend, SchedulerState
from high_throughput_vllm_deployment.schemas.telemetry import (
    BenchmarkResult,
    ClusterTelemetry,
    KvCacheBlocks,
    LatencyPercentiles,
)

KV: dict[str, int] = {
    "block_size": 16,
    "total_gpu_blocks": 1000,
    "used_gpu_blocks": 250,
    "total_cpu_blocks": 512,
    "used_cpu_blocks": 0,
}


def _telemetry(**overrides: Any) -> ClusterTelemetry:
    payload: dict[str, Any] = {
        "active_requests": 12,
        "waiting_requests": 3,
        "gpu_cache_usage_pct": 25.0,
        "cpu_cache_usage_pct": 0.0,
        "kv_cache": KV,
    }
    return ClusterTelemetry.model_validate(payload | overrides)


# --------------------------------------------------------------------------- #
# ClusterTelemetry / KvCacheBlocks                                            #
# --------------------------------------------------------------------------- #


def test_cluster_telemetry_aggregates_scheduler_queues() -> None:
    telemetry = _telemetry(swapped_requests=2)
    assert telemetry.requests_by_state == {
        SchedulerState.WAITING: 3,
        SchedulerState.RUNNING: 12,
        SchedulerState.SWAPPED: 2,
    }
    assert telemetry.total_requests == 17
    assert _telemetry().swapped_requests == 0


def test_kv_cache_derived_capacity() -> None:
    blocks = KvCacheBlocks.model_validate(KV)
    assert blocks.free_gpu_blocks == 750
    assert blocks.gpu_token_capacity == 16_000


@pytest.mark.parametrize(
    "overrides",
    [
        {"active_requests": -1},
        {"waiting_requests": 1.0},
        {"gpu_cache_usage_pct": 100.01},
        {"gpu_cache_usage_pct": float("nan")},
        {"cpu_cache_usage_pct": -0.1},
        {"kv_cache": KV | {"used_gpu_blocks": 1001}},
        {"kv_cache": KV | {"used_cpu_blocks": 513}},
        {"kv_cache": KV | {"block_size": 0}},
        {"kv_cache": KV | {"num_free_blocks": 1}},
    ],
)
def test_cluster_telemetry_rejects_invalid_snapshots(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _telemetry(**overrides)


@pytest.mark.parametrize("pct", [0.0, 100.0, 100])
def test_cache_usage_percentage_bounds_are_inclusive(pct: float) -> None:
    assert _telemetry(gpu_cache_usage_pct=pct).gpu_cache_usage_pct == pct


@given(
    total=st.integers(min_value=0, max_value=10**6),
    data=st.data(),
)
def test_any_in_range_block_usage_is_valid(total: int, data: st.DataObject) -> None:
    used = data.draw(st.integers(min_value=0, max_value=total))
    blocks = KvCacheBlocks(
        block_size=16,
        total_gpu_blocks=total,
        used_gpu_blocks=used,
        total_cpu_blocks=0,
        used_cpu_blocks=0,
    )
    assert blocks.free_gpu_blocks == total - used >= 0


# --------------------------------------------------------------------------- #
# LatencyPercentiles                                                          #
# --------------------------------------------------------------------------- #

finite_ms = st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False)


@given(st.lists(finite_ms, min_size=3, max_size=3))
def test_percentiles_accept_exactly_the_monotonic_triples(values: list[float]) -> None:
    p50, p90, p99 = values
    if p50 <= p90 <= p99:
        assert LatencyPercentiles(p50_ms=p50, p90_ms=p90, p99_ms=p99).p99_ms == p99
    else:
        with pytest.raises(ValidationError, match="p50 <= p90 <= p99"):
            LatencyPercentiles(p50_ms=p50, p90_ms=p90, p99_ms=p99)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_percentiles_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValidationError):
        LatencyPercentiles(p50_ms=bad, p90_ms=1.0, p99_ms=2.0)


def test_dominance() -> None:
    low = LatencyPercentiles(p50_ms=1.0, p90_ms=2.0, p99_ms=3.0)
    high = LatencyPercentiles(p50_ms=1.0, p90_ms=5.0, p99_ms=9.0)
    assert high.dominates(low)
    assert high.dominates(high)
    assert not low.dominates(high)


# --------------------------------------------------------------------------- #
# BenchmarkResult                                                             #
# --------------------------------------------------------------------------- #


def _pct(p50: float, p90: float, p99: float) -> dict[str, float]:
    return {"p50_ms": p50, "p90_ms": p90, "p99_ms": p99}


def _result_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": "vllm",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "total_requests": 1000,
        "failed_requests": 10,
        "duration_s": 60.0,
        "total_input_tokens": 512_000,
        "total_output_tokens": 128_000,
        "ttft": _pct(45.0, 120.0, 310.0),
        "tpot": _pct(12.0, 18.0, 30.0),
        "itl": _pct(11.0, 17.0, 45.0),
        "e2e_latency": _pct(1600.0, 2400.0, 4100.0),
    }
    return payload | overrides


def test_benchmark_result_derived_metrics() -> None:
    result = BenchmarkResult.model_validate(_result_payload())
    assert result.backend is ModelBackend.VLLM
    assert result.successful_requests == 990
    assert result.failure_rate == pytest.approx(0.01)
    assert result.request_throughput == pytest.approx(16.5)
    assert result.output_token_throughput == pytest.approx(128_000 / 60)


def test_benchmark_result_round_trips_through_json() -> None:
    result = BenchmarkResult.model_validate(_result_payload())
    dumped = result.model_dump_json()
    assert BenchmarkResult.model_validate_json(dumped) == result
    assert "request_throughput" not in dumped  # derived values never hit the wire


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"failed_requests": 1001}, "failed_requests cannot exceed"),
        ({"total_requests": 0}, "total_requests"),
        ({"duration_s": 0.0}, "duration_s"),
        ({"duration_s": float("inf")}, "duration_s"),
        ({"backend": "triton"}, "backend"),
        ({"model": " "}, "model"),
        ({"ttft": _pct(300.0, 200.0, 100.0)}, "p50 <= p90 <= p99"),
        ({"e2e_latency": _pct(40.0, 2400.0, 4100.0)}, "e2e_latency percentiles"),
        ({"rps": 16.5}, "rps"),
    ],
)
def test_benchmark_result_rejects_inconsistent_runs(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        BenchmarkResult.model_validate(_result_payload(**overrides))


def test_all_requests_failed_is_representable() -> None:
    result = BenchmarkResult.model_validate(_result_payload(failed_requests=1000))
    assert result.failure_rate == 1.0
    assert result.request_throughput == 0.0
