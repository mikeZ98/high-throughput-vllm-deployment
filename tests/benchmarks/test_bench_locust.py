"""Locust harness: event hooks, both user classes, TTFT capture, error categories, headless CLI.

In-process tests import the locustfile with ``LOCUST_SKIP_MONKEY_PATCH=1`` (set
by ``tests/conftest.py``) so gevent never patches this interpreter. The
end-to-end test runs the real ``locust --headless`` CLI in a subprocess whose
environment drops that flag and confines ``HOME`` to ``./.cache``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gevent.monkey  # type: ignore[import-untyped]
import pytest
from gevent._hub_local import get_hub_if_exists  # type: ignore[import-untyped]
from locust.env import Environment
from locust.event import Events

from high_throughput_vllm_deployment.benchmarks import locustfile
from high_throughput_vllm_deployment.benchmarks.mock_server import ThreadedMockServer
from high_throughput_vllm_deployment.core.mock import MockEngineConfig
from high_throughput_vllm_deployment.schemas.enums import ErrorCategory
from high_throughput_vllm_deployment.schemas.inference import ChatCompletionRequest
from high_throughput_vllm_deployment.schemas.telemetry import BenchmarkResult

FAST = MockEngineConfig(decode_step_s=0.001, prefill_s_per_token=0.0, ignore_eos=True)
LOCUSTFILE = Path(locustfile.__file__)


def _options(**overrides: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    locustfile.add_arguments(parser)
    options = parser.parse_args(["--vllm-max-tokens", "8"])
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(locustfile, "_state", None)
    yield
    # FastHttpUser lazily creates this thread's gevent hub (an event loop that
    # owns a pipe pair); destroy it so the resource ledger sees no leak.
    hub = get_hub_if_exists()
    if hub is not None:
        hub.destroy(destroy_loop=True)


@pytest.fixture
def captured() -> list[dict[str, Any]]:
    return []


def _environment(base_url: str, captured: list[dict[str, Any]], **options: Any) -> Environment:
    # Locust's runner normally copies --host onto the user classes.
    for user_class in (locustfile.VllmChatUser, locustfile.VllmFastChatUser):
        user_class.host = base_url
    environment = Environment(
        user_classes=[locustfile.VllmChatUser, locustfile.VllmFastChatUser],
        host=base_url,
        events=Events(),
        parsed_options=_options(**options),
    )
    environment.events.request.add_listener(lambda **event: captured.append(event))
    return environment


@contextlib.contextmanager
def _user(user_class: type[Any], environment: Environment) -> Iterator[Any]:
    """Instantiate a user and close its connection pool afterwards (Locust never does)."""
    user = user_class(environment)
    try:
        yield user
    finally:
        client = user.client  # requests.Session, or FastHttpSession wrapping a UserAgent
        getattr(client, "client", client).close()


def test_locust_was_imported_without_monkey_patching() -> None:
    assert not gevent.monkey.is_module_patched("socket")
    assert not gevent.monkey.is_module_patched("threading")


def test_command_line_options() -> None:
    options = _options()
    assert (options.vllm_model, options.vllm_max_tokens, options.vllm_prompt_tier) == (
        "mock-vllm",
        8,
        "short",
    )
    assert options.vllm_benchmark_output == ""


def test_state_requires_test_start() -> None:
    with pytest.raises(RuntimeError, match="test_start"):
        locustfile.current_state()
    with pytest.raises(RuntimeError, match="locust CLI"):
        locustfile.on_test_start(Environment(events=Events()))


def test_chat_body_toggles_streaming_fields() -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream_options": {}}
    )
    assert locustfile.chat_body(request, stream=True)["stream_options"] == {"include_usage": True}
    blocking = locustfile.chat_body(request, stream=False)
    assert blocking["stream"] is False and "stream_options" not in blocking


@pytest.mark.parametrize("user_class", [locustfile.VllmChatUser, locustfile.VllmFastChatUser])
def test_users_measure_streaming_and_blocking_chats(
    user_class: type[Any],
    captured: list[dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with ThreadedMockServer(FAST) as base_url:
        environment = _environment(base_url, captured, vllm_benchmark_output="out/run.json")
        locustfile.on_test_start(environment)
        with _user(user_class, environment) as user:
            for _ in range(3):
                user.chat_stream()
            user.chat_blocking()
        locustfile.on_test_stop(environment)

    by_type: dict[str, list[dict[str, Any]]] = {}
    for event in captured:
        by_type.setdefault(event["request_type"], []).append(event)
    assert len(by_type["POST"]) == 4 and all(e["exception"] is None for e in by_type["POST"])
    assert len(by_type["TTFT"]) == 3, "TTFT must be captured for every streamed request"
    assert len(by_type["TPOT"]) == 3
    assert all(e["response_length"] == 8 and e["response_time"] >= 0 for e in by_type["TTFT"])

    result = BenchmarkResult.model_validate_json((tmp_path / "out/run.json").read_text())
    assert (result.total_requests, result.failed_requests) == (4, 0)
    assert result.total_output_tokens == 32
    assert result.ttft is not None and result.itl is not None and result.e2e_latency is not None


@pytest.mark.parametrize("user_class", [locustfile.VllmChatUser, locustfile.VllmFastChatUser])
def test_failures_are_categorised_and_reported_to_locust(
    user_class: type[Any], captured: list[dict[str, Any]]
) -> None:
    with ThreadedMockServer(FAST, fault_every=2) as base_url:
        environment = _environment(base_url, captured)
        locustfile.on_test_start(environment)
        with _user(user_class, environment) as user:
            for _ in range(2):
                user.chat_stream()
                user.chat_blocking()
        locustfile.on_test_stop(environment)  # no output path configured: log only

    state = locustfile.current_state()
    failures = [record for record in state.collector.records if not record.success]
    assert len(failures) == 2
    assert {record.error for record in failures} == {ErrorCategory.UNAVAILABLE}
    failed_events = [e for e in captured if e["request_type"] == "POST" and e["exception"]]
    assert len(failed_events) == 2
    assert all("unavailable" in str(e["exception"]) for e in failed_events)
    assert not any(e["request_type"] == "TTFT" and e["exception"] for e in captured)


def test_unreachable_server_is_categorised_as_unavailable(captured: list[dict[str, Any]]) -> None:
    with ThreadedMockServer(FAST) as base_url:
        pass  # server stopped: its port now refuses connections
    environment = _environment(base_url, captured)
    locustfile.on_test_start(environment)
    with _user(locustfile.VllmChatUser, environment) as user:
        user.chat_blocking()
    (record,) = locustfile.current_state().collector.records
    assert record.error is ErrorCategory.UNAVAILABLE


def test_user_classes_declare_a_3_to_1_stream_mix() -> None:
    for user_class in (locustfile.VllmChatUser, locustfile.VllmFastChatUser):
        names = [getattr(task, "__name__", "") for task in user_class.tasks]
        assert names.count("chat_stream") == 3 and names.count("chat_blocking") == 1


# --------------------------------------------------------------------------- #
# Headless CLI, end to end                                                    #
# --------------------------------------------------------------------------- #


def test_headless_cli_with_50_users_against_the_mock_server(repo_root: Path) -> None:
    run_dir = repo_root / ".cache" / "tmp" / "locust-e2e"
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    output = run_dir / "result.json"
    output.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "LOCUST_SKIP_MONKEY_PATCH"}
    env["HOME"] = str(home)

    with ThreadedMockServer(
        MockEngineConfig(decode_step_s=0.002, ignore_eos=True), fault_every=20
    ) as base_url:
        completed = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
            [
                sys.executable,
                "-m",
                "locust",
                "-f",
                str(LOCUSTFILE),
                "--headless",
                "-u",
                "50",
                "-r",
                "50",
                "--run-time",
                "4s",
                "--host",
                base_url,
                "--vllm-max-tokens",
                "16",
                "--vllm-benchmark-output",
                str(output.relative_to(repo_root)),
                "--only-summary",
                "--exit-code-on-error",
                "0",
                "--stop-timeout",
                "10",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

    logs = completed.stdout + completed.stderr
    assert completed.returncode == 0, logs[-3000:]
    for row in ("POST     /v1/chat/completions [stream]", "TTFT     ", "TPOT     "):
        assert row in logs, f"missing stats row {row!r}"
    assert "Serving Benchmark Result" in logs

    result = BenchmarkResult.model_validate_json(output.read_text())
    assert result.total_requests > 100
    assert result.failed_requests > 0 and result.errors == {
        ErrorCategory.UNAVAILABLE: result.failed_requests
    }
    assert result.ttft is not None and result.ttft.p50_ms > 0
    assert result.e2e_latency is not None
    assert result.ttft.p50_ms <= result.e2e_latency.max_ms
    assert not any(home.iterdir()), "locust must not write to HOME"
