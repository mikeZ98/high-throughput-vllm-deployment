"""In-process async benchmark harness for any :class:`InferenceEnginePort`.

Drives a generated workload against ``VllmClient`` (real HTTP server) or
``MockVllmEngine`` (direct, zero-network), honouring the workload's arrival
schedule and an optional concurrency cap, and reports results in the layout
of vLLM's ``benchmark_serving`` CLI (or as Markdown / JSON).

Examples::

    uv run python -m high_throughput_vllm_deployment.benchmarks.runner \\
        --target mock --num-requests 500 --request-rate 50 --virtual-time
    uv run python -m high_throughput_vllm_deployment.benchmarks.runner \\
        --target http --base-url http://127.0.0.1:8000 --model mistralai/Mistral-7B-Instruct-v0.2
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from high_throughput_vllm_deployment.benchmarks.metrics_collector import (
    Clock,
    MetricsCollector,
    RequestRecord,
    categorize_exception,
)
from high_throughput_vllm_deployment.benchmarks.observer import StreamObserver
from high_throughput_vllm_deployment.benchmarks.workload import (
    ArrivalPattern,
    PromptTier,
    WorkItem,
    WorkloadProfile,
    generate_workload,
)
from high_throughput_vllm_deployment.core._clock import Sleep, instant_sleep
from high_throughput_vllm_deployment.core.client import VllmClient
from high_throughput_vllm_deployment.core.errors import EngineError
from high_throughput_vllm_deployment.core.mock import MockEngineConfig, MockVllmEngine
from high_throughput_vllm_deployment.core.ports import InferenceEnginePort
from high_throughput_vllm_deployment.schemas._base import reject_host_path_escape
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import ModelBackend
from high_throughput_vllm_deployment.schemas.inference import ChatCompletionRequest
from high_throughput_vllm_deployment.schemas.telemetry import BenchmarkResult, LatencyPercentiles

WIDTH: Final = 50

# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


async def _execute(
    engine: InferenceEnginePort,
    item: WorkItem,
    collector: MetricsCollector,
    clock: Clock,
    limit: asyncio.Semaphore | None,
) -> None:
    async with limit if limit is not None else contextlib.nullcontext():
        start = clock()
        request = item.request
        try:
            if isinstance(request, ChatCompletionRequest) and item.stream:
                observer = StreamObserver(
                    clock=clock, start_s=start, prompt_tokens=item.prompt_tokens
                )
                async for chunk in engine.stream_chat(request):
                    observer.observe(chunk)
                record = observer.finish()
            else:
                if isinstance(request, ChatCompletionRequest):
                    usage = (await engine.chat_complete(request)).usage
                else:
                    usage = (await engine.complete(request)).usage
                record = RequestRecord(
                    start_s=start,
                    end_s=clock(),
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
        except EngineError as exc:
            record = RequestRecord.failure(
                start, clock(), categorize_exception(exc), str(exc), input_tokens=item.prompt_tokens
            )
        collector.record(record)


class VirtualTimeline:
    """Simulated-time clock/sleep pair for driving a :class:`MockVllmEngine`.

    While the engine is busy its own scheduler steps advance simulated time,
    so ``sleep`` just yields until the engine clock reaches the deadline; once it
    is quiescent (idle, all results delivered), ``sleep`` jumps the clock
    forward. Arrival schedules are thus honoured exactly (to step granularity)
    without any real waiting.
    """

    def __init__(self, engine: MockVllmEngine) -> None:
        self.engine = engine

    def now(self) -> float:
        return self.engine.clock_s

    async def sleep(self, seconds: float) -> None:
        deadline = self.engine.clock_s + seconds
        while True:
            await asyncio.sleep(0)  # let freshly dispatched requests reach the engine
            remaining = deadline - self.engine.clock_s
            if remaining <= 0:
                return
            if self.engine.is_quiescent:
                self.engine.advance_idle_clock(remaining)
                return


async def run_workload(
    engine: InferenceEnginePort,
    items: Sequence[WorkItem],
    *,
    backend: ModelBackend,
    model: str,
    max_concurrency: int | None = None,
    clock: Clock = time.perf_counter,
    sleep: Sleep = asyncio.sleep,
) -> BenchmarkResult:
    """Dispatch *items* at their arrival offsets and aggregate the outcome."""
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    limit = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
    collector = MetricsCollector(backend=backend, model=model, clock=clock)
    collector.begin()
    origin = clock()
    async with asyncio.TaskGroup() as group:
        for item in items:
            delay = item.arrival_s - (clock() - origin)
            if delay > 0:
                await sleep(delay)
            group.create_task(_execute(engine, item, collector, clock, limit))
    collector.end()
    return collector.result()


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

_SECTIONS: Final = (
    ("ttft", "Time to First Token", "TTFT"),
    ("tpot", "Time per Output Token (excl. 1st token)", "TPOT"),
    ("itl", "Inter-token Latency", "ITL"),
    ("e2e_latency", "End-to-end Latency", "E2EL"),
)


def _latencies(result: BenchmarkResult) -> list[tuple[str, str, LatencyPercentiles]]:
    rows = []
    for field, title, short in _SECTIONS:
        stats = getattr(result, field)
        if stats is not None:
            rows.append((title, short, stats))
    return rows


def format_text(result: BenchmarkResult) -> str:
    """Render in the layout of vLLM's ``benchmark_serving`` summary."""
    lines = [
        "{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=WIDTH, c="="),
        f"{'Backend:':<40} {result.backend.value}",
        f"{'Successful requests:':<40} {result.successful_requests}",
        f"{'Failed requests:':<40} {result.failed_requests}",
        f"{'Benchmark duration (s):':<40} {result.duration_s:.2f}",
        f"{'Total input tokens:':<40} {result.total_input_tokens}",
        f"{'Total generated tokens:':<40} {result.total_output_tokens}",
        f"{'Request throughput (req/s):':<40} {result.request_throughput:.2f}",
        f"{'Output token throughput (tok/s):':<40} {result.output_token_throughput:.2f}",
    ]
    for title, short, stats in _latencies(result):
        lines.append("{s:{c}^{n}}".format(s=title, n=WIDTH, c="-"))
        for label, value in (
            ("Mean", stats.mean_ms),
            ("Median", stats.p50_ms),
            ("P90", stats.p90_ms),
            ("P99", stats.p99_ms),
            ("Max", stats.max_ms),
        ):
            lines.append(f"{f'{label} {short} (ms):':<40} {value:.2f}")
    if result.errors:
        lines.append("{s:{c}^{n}}".format(s="Errors", n=WIDTH, c="-"))
        for category, count in sorted(result.errors.items()):
            lines.append(f"{f'{category.value}:':<40} {count}")
    lines.append("=" * WIDTH)
    return "\n".join(lines) + "\n"


def format_markdown(result: BenchmarkResult) -> str:
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Backend | {result.backend.value} |",
        f"| Successful requests | {result.successful_requests} |",
        f"| Failed requests | {result.failed_requests} |",
        f"| Benchmark duration (s) | {result.duration_s:.2f} |",
        f"| Total input tokens | {result.total_input_tokens} |",
        f"| Total generated tokens | {result.total_output_tokens} |",
        f"| Request throughput (req/s) | {result.request_throughput:.2f} |",
        f"| Output token throughput (tok/s) | {result.output_token_throughput:.2f} |",
    ]
    lines += [f"| Errors: {c.value} | {n} |" for c, n in sorted(result.errors.items())]
    latencies = _latencies(result)
    if latencies:
        lines += [
            "",
            "| Latency (ms) | Mean | P50 | P90 | P99 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, short, s in latencies:
            lines.append(
                f"| {short} | {s.mean_ms:.2f} | {s.p50_ms:.2f} | {s.p90_ms:.2f} "
                f"| {s.p99_ms:.2f} | {s.max_ms:.2f} |"
            )
    return "\n".join(lines) + "\n"


def render(result: BenchmarkResult, fmt: str) -> str:
    if fmt == "markdown":
        return format_markdown(result)
    if fmt == "json":
        return result.model_dump_json(indent=2) + "\n"
    return format_text(result)


def contained_output_path(raw: str) -> Path:
    """Resolve a report path that must stay inside the current working directory."""
    reject_host_path_escape(raw)
    root = Path.cwd().resolve()
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"output path {raw!r} resolves outside the project root")
    return path


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _prompt_mix(raw: str) -> dict[PromptTier, float]:
    mix: dict[PromptTier, float] = {}
    for part in raw.split(","):
        tier, sep, weight = part.partition("=")
        try:
            mix[PromptTier(tier.strip())] = float(weight) if sep else 1.0
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid prompt-mix entry {part!r}") from None
    return mix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a vLLM deployment or the mock engine.")
    parser.add_argument("--target", choices=("mock", "http"), default="mock")
    parser.add_argument("--base-url", help="http target: server URL (default: from --env-file)")
    parser.add_argument("--env-file", default=None, help="http target: project-relative .env")
    parser.add_argument("--model", help="model name (default: mock-vllm / VLLM_MODEL)")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=20.0)
    parser.add_argument("--arrival", choices=[p.value for p in ArrivalPattern], default="poisson")
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--prompt-mix", type=_prompt_mix, default={PromptTier.SHORT: 1.0})
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--chat-fraction", type=float, default=1.0)
    parser.add_argument("--stream-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--mock-decode-step-s", type=float, default=0.02)
    parser.add_argument("--mock-prefill-s-per-token", type=float, default=1e-4)
    parser.add_argument("--mock-num-gpu-blocks", type=int, default=1024)
    parser.add_argument(
        "--virtual-time",
        action="store_true",
        help="mock target: simulated clock, no real waiting (fast, fully deterministic)",
    )
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    parser.add_argument("--output", help="also write the JSON result to this project-relative path")
    return parser


async def _run(args: argparse.Namespace) -> BenchmarkResult:
    if args.target == "http":
        config = (
            VllmDeploymentConfig.from_env_file(args.env_file)
            if args.env_file
            else VllmDeploymentConfig(model=args.model)
        )
        model = args.model or config.model
        engine: InferenceEnginePort = VllmClient(config, base_url=args.base_url)
        backend = ModelBackend.VLLM
        clock: Clock = time.perf_counter
        sleep: Sleep = asyncio.sleep
    else:
        model = args.model or MockEngineConfig().model_name
        mock = MockVllmEngine(
            MockEngineConfig(
                model_name=model,
                decode_step_s=args.mock_decode_step_s,
                prefill_s_per_token=args.mock_prefill_s_per_token,
                num_gpu_blocks=args.mock_num_gpu_blocks,
                ignore_eos=True,
            ),
            sleep=instant_sleep if args.virtual_time else asyncio.sleep,
        )
        engine, backend = mock, ModelBackend.VLLM
        if args.virtual_time:
            timeline = VirtualTimeline(mock)
            clock, sleep = timeline.now, timeline.sleep
        else:
            clock, sleep = time.perf_counter, asyncio.sleep

    profile = WorkloadProfile(
        model=model,
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        arrival=args.arrival,
        burst_size=args.burst_size,
        prompt_mix=args.prompt_mix,
        max_tokens=args.max_tokens,
        chat_fraction=args.chat_fraction,
        stream_fraction=args.stream_fraction,
        seed=args.seed,
    )
    try:
        return await run_workload(
            engine,
            generate_workload(profile),
            backend=backend,
            model=model,
            max_concurrency=args.max_concurrency,
            clock=clock,
            sleep=sleep,
        )
    finally:
        await engine.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target == "http" and not (args.env_file or args.model):
        parser.error("--target http needs --model or --env-file")
    output = contained_output_path(args.output) if args.output else None
    result = asyncio.run(_run(args))
    rendered = render(result, args.format)
    sys.stdout.write(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
