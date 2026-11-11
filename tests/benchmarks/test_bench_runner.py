"""Async benchmark runner: virtual time, concurrency, failures, reports, CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from high_throughput_vllm_deployment.benchmarks.mock_server import ThreadedMockServer
from high_throughput_vllm_deployment.benchmarks.runner import (
    VirtualTimeline,
    contained_output_path,
    format_markdown,
    format_text,
    main,
    render,
    run_workload,
)
from high_throughput_vllm_deployment.benchmarks.workload import (
    WorkItem,
    WorkloadProfile,
    generate_workload,
)
from high_throughput_vllm_deployment.core import instant_sleep
from high_throughput_vllm_deployment.core.mock import MockEngineConfig, MockVllmEngine
from high_throughput_vllm_deployment.schemas.enums import ErrorCategory, ModelBackend
from high_throughput_vllm_deployment.schemas.telemetry import BenchmarkResult

MODEL = "mock-vllm"
ENGINE = MockEngineConfig(decode_step_s=0.02, prefill_s_per_token=1e-4, ignore_eos=True)


def _items(**overrides: object) -> list[WorkItem]:
    profile = WorkloadProfile.model_validate(
        {"model": MODEL, "num_requests": 40, "request_rate": 5.0, "max_tokens": 16} | overrides
    )
    return generate_workload(profile)


async def _virtual_run(
    items: list[WorkItem],
    *,
    max_concurrency: int | None = None,
    model: str = MODEL,
    config: MockEngineConfig = ENGINE,
) -> tuple[BenchmarkResult, MockVllmEngine]:
    engine = MockVllmEngine(config, sleep=instant_sleep)
    timeline = VirtualTimeline(engine)
    async with engine:
        result = await run_workload(
            engine,
            items,
            backend=ModelBackend.VLLM,
            model=model,
            max_concurrency=max_concurrency,
            clock=timeline.now,
            sleep=timeline.sleep,
        )
    return result, engine


@pytest.mark.asyncio
async def test_virtual_time_honours_the_arrival_schedule() -> None:
    items = _items(arrival="uniform", request_rate=2.0, num_requests=10)
    result, engine = await _virtual_run(items)
    # Light load: each request finishes before the next arrives, so the run lasts
    # ~ last arrival (4.5 s) + one request's service time, and nothing queues.
    service = 16 * ENGINE.decode_step_s
    assert result.duration_s == pytest.approx(4.5 + service, abs=0.1)
    assert result.ttft is not None and result.ttft.max_ms < 50
    assert engine.stats.peak_running == 1
    assert result.successful_requests == 10 and result.total_output_tokens == 160


@pytest.mark.asyncio
async def test_overload_shows_up_as_queueing_latency() -> None:
    light, _ = await _virtual_run(_items(arrival="uniform", request_rate=1.0, num_requests=20))
    burst = _items(arrival="burst", burst_size=20, request_rate=1000.0, num_requests=20)
    batched, _ = await _virtual_run(burst)
    saturated, engine = await _virtual_run(
        burst, config=ENGINE.model_copy(update={"max_num_seqs": 2})
    )
    assert light.e2e_latency is not None and batched.e2e_latency is not None
    assert saturated.e2e_latency is not None and saturated.ttft is not None
    # Continuous batching absorbs a 20-request burst: one combined prefill step
    # (~20 x 128 tokens) then shared decode steps, far below 20 serial services.
    assert batched.e2e_latency.p99_ms < 2 * light.e2e_latency.p99_ms
    # ...but with only 2 sequence slots the tail queues for ~9 service times.
    assert saturated.e2e_latency.p99_ms > 8 * light.e2e_latency.p99_ms
    assert saturated.ttft.p99_ms > saturated.ttft.p50_ms > light.ttft.p99_ms  # type: ignore[union-attr]
    assert engine.stats.peak_waiting == 18

    client_capped, _ = await _virtual_run(burst, max_concurrency=2)
    # A client-side cap delays *sending*; per-request (server) latency stays low.
    assert client_capped.e2e_latency is not None
    assert client_capped.e2e_latency.p99_ms < 1.5 * light.e2e_latency.p99_ms
    # 20 requests through 2 client slots = 10 sequential waves of one pair each.
    wave_s = client_capped.e2e_latency.mean_ms / 1000
    assert client_capped.duration_s == pytest.approx(10 * wave_s, rel=0.05)


@pytest.mark.asyncio
async def test_mixed_workload_accounting() -> None:
    items = _items(chat_fraction=0.5, stream_fraction=0.5, num_requests=60)
    result, _ = await _virtual_run(items)
    streamed = sum(item.stream for item in items)
    assert result.successful_requests == 60
    assert result.total_input_tokens >= sum(item.prompt_tokens for item in items)
    assert result.ttft is not None and result.e2e_latency is not None
    assert 0 < streamed < 60  # TTFT covers only the streamed subset


@pytest.mark.asyncio
async def test_engine_failures_are_recorded_by_category() -> None:
    result, _ = await _virtual_run(_items(num_requests=5, model="not-served"), model="x")
    assert result.failed_requests == 5
    assert result.errors == {ErrorCategory.CLIENT_ERROR: 5}
    assert result.e2e_latency is None and result.request_throughput == 0.0


@pytest.mark.asyncio
async def test_invalid_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        await run_workload(
            MockVllmEngine(), [], backend=ModelBackend.VLLM, model=MODEL, max_concurrency=0
        )


@pytest.mark.asyncio
async def test_virtual_sleep_waits_for_a_busy_engine() -> None:
    engine = MockVllmEngine(ENGINE, sleep=instant_sleep)
    timeline = VirtualTimeline(engine)
    async with engine:
        await timeline.sleep(1.5)  # idle: jumps
        assert engine.clock_s == 1.5
        task = asyncio.ensure_future(
            run_workload(
                engine,
                _items(num_requests=1, max_tokens=50),
                backend=ModelBackend.VLLM,
                model=MODEL,
                clock=timeline.now,
                sleep=timeline.sleep,
            )
        )
        await asyncio.sleep(0)
        await timeline.sleep(0.1)  # busy: advanced by engine steps, not jumped
        assert 1.6 <= engine.clock_s < 1.6 + 2 * ENGINE.decode_step_s + 0.01
        await timeline.sleep(0.0)
        await task


def test_idle_clock_rules() -> None:
    engine = MockVllmEngine()
    with pytest.raises(ValueError, match="backwards"):
        engine.advance_idle_clock(-1.0)
    engine._waiting.append(object())  # type: ignore[arg-type]
    assert not engine.is_idle
    with pytest.raises(RuntimeError, match="only advances by itself"):
        engine.advance_idle_clock(1.0)


# --------------------------------------------------------------------------- #
# Reports                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reports_follow_the_vllm_layout() -> None:
    result, _ = await _virtual_run(_items(num_requests=10))
    text = format_text(result)
    assert text.startswith("============ Serving Benchmark Result ============\n")
    for label in ("Successful requests:", "Mean TTFT (ms):", "P99 ITL (ms):", "Max E2EL (ms):"):
        assert label in text
    assert "Errors" not in text
    assert all(len(line) == 50 for line in text.splitlines() if set(line) <= {"=", "-"} and line)
    markdown = format_markdown(result)
    assert "| TTFT |" in markdown and "| Latency (ms) | Mean | P50 | P90 | P99 | Max |" in markdown
    assert render(result, "json").startswith("{") and render(result, "markdown") == markdown
    assert render(result, "text") == text


def test_reports_of_failed_runs_list_errors_and_skip_latency() -> None:
    failed = BenchmarkResult(
        backend=ModelBackend.VLLM,
        model=MODEL,
        total_requests=3,
        failed_requests=3,
        duration_s=1.0,
        total_input_tokens=0,
        total_output_tokens=0,
        errors={ErrorCategory.TIMEOUT: 2, ErrorCategory.UNAVAILABLE: 1},
    )
    text = format_text(failed)
    assert "timeout:" in text and "unavailable:" in text and "TTFT" not in text
    markdown = format_markdown(failed)
    assert "| Errors: timeout | 2 |" in markdown and "Latency" not in markdown


def test_output_paths_must_stay_inside_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    assert contained_output_path("reports/run.json") == project.resolve() / "reports/run.json"
    for bad in ("/etc/run.json", "../run.json", "~root/run.json"):
        with pytest.raises(ValueError):
            contained_output_path(bad)
    (project / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="outside the project root"):
        contained_output_path("escape/run.json")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in [n for n in __import__("os").environ if n.upper().startswith("VLLM_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_mock_virtual_time_writes_json(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--target",
        "mock",
        "--virtual-time",
        "--num-requests",
        "20",
        "--request-rate",
        "50",
        "--prompt-mix",
        "short=0.7,medium",
        "--arrival",
        "burst",
        "--format",
        "json",
        "--output",
        "out/result.json",
    ]
    assert main(argv) == 0
    stdout = capsys.readouterr().out
    written = BenchmarkResult.model_validate_json((project / "out/result.json").read_text())
    assert BenchmarkResult.model_validate_json(stdout) == written
    assert written.successful_requests == 20


def test_cli_mock_real_time(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--num-requests",
        "3",
        "--request-rate",
        "1000",
        "--max-tokens",
        "2",
        "--mock-decode-step-s",
        "0.0005",
        "--mock-prefill-s-per-token",
        "0",
        "--format",
        "markdown",
    ]
    assert main(argv) == 0
    assert "| Successful requests | 3 |" in capsys.readouterr().out


def test_cli_http_target_against_the_mock_server(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fast = MockEngineConfig(decode_step_s=0.001, prefill_s_per_token=0.0, ignore_eos=True)
    with ThreadedMockServer(fast) as base_url:
        assert (
            main(
                [
                    "--target",
                    "http",
                    "--base-url",
                    base_url,
                    "--model",
                    MODEL,
                    "--num-requests",
                    "4",
                    "--request-rate",
                    "1000",
                    "--max-tokens",
                    "3",
                ]
            )
            == 0
        )
        (project / ".env").write_text(f"VLLM_MODEL={MODEL}\n", encoding="utf-8")
        assert (
            main(
                [
                    "--target",
                    "http",
                    "--base-url",
                    base_url,
                    "--env-file",
                    ".env",
                    "--num-requests",
                    "2",
                    "--request-rate",
                    "1000",
                    "--max-tokens",
                    "2",
                    "--format",
                    "markdown",
                ]
            )
            == 0
        )
    out = capsys.readouterr().out
    assert "Successful requests:                     4" in out
    assert "| Successful requests | 2 |" in out


@pytest.mark.parametrize(
    "argv",
    [["--target", "http"], ["--prompt-mix", "huge=1"], ["--prompt-mix", "short=x"]],
)
def test_cli_rejects_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_json_result_is_parseable_by_downstream_tools(project: Path) -> None:
    assert main(["--virtual-time", "--num-requests", "5", "--output", "r.json"]) == 0
    data = json.loads((project / "r.json").read_text())
    assert set(data) >= {"ttft", "itl", "e2e_latency", "errors"}
