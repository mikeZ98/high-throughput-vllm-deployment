"""Periodic telemetry collection from any :class:`InferenceEnginePort`."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Self

from high_throughput_vllm_deployment.core._clock import Sleep
from high_throughput_vllm_deployment.core.errors import EngineError
from high_throughput_vllm_deployment.core.ports import InferenceEnginePort
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry


@dataclass(frozen=True, slots=True)
class TelemetrySummary:
    samples: int
    failed_scrapes: int
    peak_active_requests: int
    peak_waiting_requests: int
    peak_swapped_requests: int
    peak_gpu_cache_usage_pct: float
    mean_gpu_cache_usage_pct: float
    peak_gpu_blocks_used: int


class TelemetryCollector:
    """Samples ``engine.get_telemetry()`` every ``interval_s`` into a bounded buffer.

    Scrape failures are counted, not raised, so a flaky metrics endpoint never
    aborts the load test it is observing. Use as an async context manager::

        async with TelemetryCollector(engine, interval_s=1.0) as collector:
            ...  # drive load
        summary = collector.summary()
    """

    def __init__(
        self,
        engine: InferenceEnginePort,
        *,
        interval_s: float = 1.0,
        max_samples: int = 10_000,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        self._engine = engine
        self.interval_s = interval_s
        self._sleep = sleep
        self._samples: deque[ClusterTelemetry] = deque(maxlen=max_samples)
        self._failed = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def samples(self) -> tuple[ClusterTelemetry, ...]:
        return tuple(self._samples)

    @property
    def failed_scrapes(self) -> int:
        return self._failed

    async def scrape(self) -> ClusterTelemetry | None:
        try:
            sample = await self._engine.get_telemetry()
        except EngineError:
            self._failed += 1
            return None
        self._samples.append(sample)
        return sample

    async def _loop(self) -> None:
        while True:
            await self.scrape()
            await self._sleep(self.interval_s)

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("collector is already running")
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        """Stop sampling and take one final sample of the end state."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        await self.scrape()

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.stop()

    def summary(self) -> TelemetrySummary:
        samples = self._samples
        if not samples:
            return TelemetrySummary(0, self._failed, 0, 0, 0, 0.0, 0.0, 0)
        gpu = [sample.gpu_cache_usage_pct for sample in samples]
        return TelemetrySummary(
            samples=len(samples),
            failed_scrapes=self._failed,
            peak_active_requests=max(sample.active_requests for sample in samples),
            peak_waiting_requests=max(sample.waiting_requests for sample in samples),
            peak_swapped_requests=max(sample.swapped_requests for sample in samples),
            peak_gpu_cache_usage_pct=max(gpu),
            mean_gpu_cache_usage_pct=sum(gpu) / len(gpu),
            peak_gpu_blocks_used=max(sample.kv_cache.used_gpu_blocks for sample in samples),
        )
