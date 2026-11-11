"""TelemetryCollector: periodic sampling, failure accounting, summaries."""

from __future__ import annotations

import asyncio

import pytest

from high_throughput_vllm_deployment.core import instant_sleep
from high_throughput_vllm_deployment.core.collector import TelemetryCollector, TelemetrySummary
from high_throughput_vllm_deployment.core.errors import EngineUnavailableError
from high_throughput_vllm_deployment.core.mock import MockEngineConfig, MockVllmEngine
from high_throughput_vllm_deployment.schemas.inference import CompletionRequest
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry


class _FlakyEngine(MockVllmEngine):
    """Fails every other telemetry scrape."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def get_telemetry(self) -> ClusterTelemetry:
        self.calls += 1
        if self.calls % 2 == 0:
            raise EngineUnavailableError("scrape failed")
        return await super().get_telemetry()


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"interval_s": 0.0}, "interval_s"), ({"max_samples": 0}, "max_samples")]
)
def test_invalid_settings_are_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TelemetryCollector(MockVllmEngine(), **kwargs)  # type: ignore[arg-type]


def test_empty_summary() -> None:
    assert TelemetryCollector(MockVllmEngine()).summary() == TelemetrySummary(
        0, 0, 0, 0, 0, 0.0, 0.0, 0
    )


@pytest.mark.asyncio
async def test_scrape_failures_are_counted_not_raised() -> None:
    collector = TelemetryCollector(_FlakyEngine())
    assert await collector.scrape() is not None
    assert await collector.scrape() is None
    assert (len(collector.samples), collector.failed_scrapes) == (1, 1)


@pytest.mark.asyncio
async def test_buffer_is_bounded() -> None:
    collector = TelemetryCollector(MockVllmEngine(), max_samples=3)
    for _ in range(5):
        await collector.scrape()
    assert len(collector.samples) == 3


@pytest.mark.asyncio
async def test_collector_observes_load_on_the_mock_engine() -> None:
    config = MockEngineConfig(
        block_size=4, num_gpu_blocks=16, max_model_len=64, max_num_batched_tokens=64
    )
    async with MockVllmEngine(config, sleep=instant_sleep) as engine:
        requests = [
            CompletionRequest(model="mock-vllm", prompt=f"load {i} " * 3, max_tokens=24)
            for i in range(6)
        ]
        async with TelemetryCollector(engine, interval_s=0.01, sleep=instant_sleep) as collector:
            await asyncio.gather(*(engine.complete(request) for request in requests))
        summary = collector.summary()

    assert summary.samples == len(collector.samples) >= 2
    assert summary.failed_scrapes == 0
    assert summary.peak_active_requests >= 1
    assert 0.0 < summary.peak_gpu_cache_usage_pct <= 100.0
    assert 0.0 <= summary.mean_gpu_cache_usage_pct <= summary.peak_gpu_cache_usage_pct
    assert 0 < summary.peak_gpu_blocks_used <= 16
    # stop() records the drained end state.
    assert collector.samples[-1].kv_cache.used_gpu_blocks == 0


@pytest.mark.asyncio
async def test_start_twice_is_rejected_and_stop_is_idempotent() -> None:
    collector = TelemetryCollector(MockVllmEngine(), sleep=instant_sleep)
    collector.start()
    with pytest.raises(RuntimeError, match="already running"):
        collector.start()
    await collector.stop()
    samples = len(collector.samples)
    await collector.stop()
    assert len(collector.samples) == samples
