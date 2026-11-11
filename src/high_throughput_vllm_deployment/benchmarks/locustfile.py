"""Locust scenarios for vLLM's OpenAI-compatible ``/v1/chat/completions`` endpoint.

Two user classes share one scenario (3:1 streaming vs non-streaming chats):

* :class:`VllmChatUser` (``HttpUser``/requests) - urllib3 yields each HTTP
  chunk as it arrives, so streamed TTFT/ITL are measured accurately;
* :class:`VllmFastChatUser` (``FastHttpUser``/geventhttpclient) - much higher
  request throughput per worker, but its reader returns only once
  ``STREAM_READ_CHUNK_BYTES`` are buffered, so streamed timings can lag by up
  to about one SSE event. Prefer it for throughput, ``VllmChatUser`` for latency.

Beyond Locust's own HTTP stats, every success fires custom ``TTFT`` and
``TPOT`` request events (``response_length`` = generated tokens), failures are
labelled with an :class:`ErrorCategory`, and at test stop the run is
aggregated into a validated :class:`BenchmarkResult` (logged in vLLM's summary
layout and optionally written as JSON).

Headless run against a server (mock or real)::

    uv run locust -f src/high_throughput_vllm_deployment/benchmarks/locustfile.py \\
        --headless -u 50 -r 10 --run-time 60s --host http://127.0.0.1:8000 \\
        --vllm-model mock-vllm --vllm-benchmark-output .cache/benchmarks/locust.json

Only local (single-process) runs aggregate a ``BenchmarkResult``; with
distributed workers each worker would hold a partial collector.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Final

from locust import FastHttpUser, HttpUser, TaskSet, events
from locust.env import Environment

from high_throughput_vllm_deployment.benchmarks.metrics_collector import MetricsCollector
from high_throughput_vllm_deployment.benchmarks.observer import (
    custom_metric_events,
    record_from_response,
)
from high_throughput_vllm_deployment.benchmarks.runner import contained_output_path, format_text
from high_throughput_vllm_deployment.benchmarks.workload import (
    PromptTier,
    WorkItem,
    WorkloadProfile,
    generate_workload,
)
from high_throughput_vllm_deployment.schemas.enums import ModelBackend
from high_throughput_vllm_deployment.schemas.inference import ChatCompletionRequest

CHAT_PATH: Final = "/v1/chat/completions"
WORKLOAD_SIZE: Final = 512
logger = logging.getLogger(__name__)


@events.init_command_line_parser.add_listener
def add_arguments(parser: argparse.ArgumentParser, **_kwargs: Any) -> None:
    group = parser.add_argument_group("vLLM benchmark")
    group.add_argument("--vllm-model", default="mock-vllm", help="served model name")
    group.add_argument("--vllm-max-tokens", type=int, default=128)
    group.add_argument(
        "--vllm-prompt-tier",
        choices=[tier.value for tier in PromptTier],
        default=PromptTier.SHORT.value,
    )
    group.add_argument("--vllm-seed", type=int, default=0)
    group.add_argument(
        "--vllm-benchmark-output", default="", help="project-relative JSON path for the result"
    )


@dataclass(slots=True)
class BenchmarkState:
    collector: MetricsCollector
    items: Iterator[WorkItem]
    output: str
    lock: threading.Lock

    def next_item(self) -> WorkItem:
        with self.lock:  # greenlets never block here; guards real threads too
            return next(self.items)


_state: BenchmarkState | None = None


def current_state() -> BenchmarkState:
    if _state is None:
        raise RuntimeError("benchmark state is initialised by the test_start event")
    return _state


@events.test_start.add_listener
def on_test_start(environment: Environment, **_kwargs: Any) -> None:
    global _state
    options = environment.parsed_options
    if options is None:
        raise RuntimeError("run through the locust CLI so that options are parsed")
    profile = WorkloadProfile(
        model=options.vllm_model,
        num_requests=WORKLOAD_SIZE,
        prompt_mix={PromptTier(options.vllm_prompt_tier): 1.0},
        max_tokens=options.vllm_max_tokens,
        seed=options.vllm_seed,
    )
    collector = MetricsCollector(backend=ModelBackend.VLLM, model=options.vllm_model)
    collector.begin()
    _state = BenchmarkState(
        collector=collector,
        items=itertools.cycle(generate_workload(profile)),
        output=options.vllm_benchmark_output,
        lock=threading.Lock(),
    )


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **_kwargs: Any) -> None:
    state = current_state()
    state.collector.end()
    result = state.collector.result()
    logger.info("vLLM benchmark summary\n%s", format_text(result))
    if state.output:
        path = contained_output_path(state.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def chat_body(request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
    body = request.model_dump(mode="json", exclude_none=True)
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True}
    else:
        body.pop("stream_options", None)
    return body


class ChatScenario:
    """Task bodies shared by both user classes (``self.client`` differs).

    Locust only collects ``@task`` methods from ``User`` subclasses, so each
    user class declares :data:`CHAT_TASKS` explicitly.
    """

    client: Any
    environment: Environment

    def chat_stream(self) -> None:
        self.chat(stream=True)

    def chat_blocking(self) -> None:
        self.chat(stream=False)

    def chat(self, *, stream: bool) -> None:
        state = current_state()
        item = state.next_item()
        assert isinstance(item.request, ChatCompletionRequest)  # noqa: S101 - chat-only workload
        name = f"{CHAT_PATH} [{'stream' if stream else 'json'}]"
        start = time.perf_counter()
        with self.client.post(
            CHAT_PATH,
            json=chat_body(item.request, stream=stream),
            stream=stream,
            catch_response=True,
            name=name,
        ) as response:
            record = record_from_response(
                response, stream=stream, start_s=start, prompt_tokens=item.prompt_tokens
            )
            if record.success:
                response.success()
            else:
                response.failure(f"{record.error}: {record.detail}")
        state.collector.record(record)
        for event in custom_metric_events(record, name):
            self.environment.events.request.fire(**event)


# 3:1 streaming vs non-streaming (Locust weights list entries by repetition).
CHAT_TASKS: Final[list[TaskSet | Callable[..., Any]]] = [
    *[ChatScenario.chat_stream] * 3,
    ChatScenario.chat_blocking,
]


class VllmChatUser(ChatScenario, HttpUser):
    """requests-based user: accurate streaming timings."""

    weight = 1
    tasks = CHAT_TASKS


class VllmFastChatUser(ChatScenario, FastHttpUser):
    """geventhttpclient-based user: maximum request throughput."""

    weight = 1
    tasks = CHAT_TASKS
