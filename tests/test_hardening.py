"""State-space hardening: scheduler FSM fuzzing, fairness, hostile input, isolation proofs.

* FSM fuzzing: random arrival streams with cancellations and mid-stream client
  disconnects (in-process and over HTTP) must always drain to zero KV blocks,
  with exact block accounting at every scheduler step.
* Fairness: FCFS admission plus latest-arrival preemption gives a provable
  liveness bound - while request *i* is unfinished, every scheduler step emits
  a token for some request that arrived no later than *i*, so *i* completes
  within ``sum(tokens of requests 0..i)`` steps of its arrival. No starvation.
* Hostile input: SSE framing, Unicode extremes, deep JSON nesting, Prometheus
  text and raw HTTP/1.1 garbage only ever produce the documented exceptions or
  HTTP errors, with every buffer bounded.
* Isolation: the resource ledger (tests/conftest.py) is proven to catch leaks,
  and every cache/temp location is proven to sit under ``./.cache``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import CACHE_ROOT, REPO_ROOT, TMP_ROOT
from engine_invariants import assert_scheduler_invariants
from high_throughput_vllm_deployment.benchmarks.mock_server import MockVllmServer
from high_throughput_vllm_deployment.benchmarks.observer import StreamObserver
from high_throughput_vllm_deployment.core.client import (
    RetryPolicy,
    VllmClient,
    decode_sse_event,
)
from high_throughput_vllm_deployment.core.errors import EngineError, EngineProtocolError
from high_throughput_vllm_deployment.core.metrics import telemetry_from_prometheus
from high_throughput_vllm_deployment.core.mock import MockEngineConfig, MockVllmEngine
from high_throughput_vllm_deployment.core.sse import LineDecoder, SseDecoder
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import FinishReason
from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    StreamChunk,
)

MODEL = "mock-vllm"
FUZZ = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _chat(words: int, max_tokens: int) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL,
        messages=[ChatMessage(role="user", content=" ".join(["w"] * words))],
        max_tokens=max_tokens,
    )


async def _yield_until(predicate: Any, *, limit: int = 20_000) -> None:
    """Bounded cooperative wait: fails instead of hanging if *predicate* never holds."""
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached within the step budget")


# --------------------------------------------------------------------------- #
# FSM fuzzing: arrivals, cancellations, disconnects -> zero block leaks       #
# --------------------------------------------------------------------------- #

operations = st.lists(
    st.tuples(
        st.sampled_from(["complete", "stream", "cancel", "disconnect"]),
        st.integers(min_value=1, max_value=12),  # prompt words (+6 chat-template tokens)
        st.integers(min_value=1, max_value=20),  # max tokens: 12 + 6 + 20 <= max_model_len
        st.integers(min_value=0, max_value=30),  # cancel/disconnect point (steps / chunks)
    ),
    min_size=1,
    max_size=14,
)


@FUZZ
@given(
    ops=operations,
    block_size=st.sampled_from([1, 2, 4]),
    extra_blocks=st.integers(min_value=0, max_value=12),
    num_cpu_blocks=st.integers(min_value=0, max_value=16),
    max_num_seqs=st.integers(min_value=1, max_value=6),
)
def test_random_workloads_with_aborts_always_reclaim_every_block(
    ops: list[tuple[str, int, int, int]],
    block_size: int,
    extra_blocks: int,
    num_cpu_blocks: int,
    max_num_seqs: int,
) -> None:
    config = MockEngineConfig(
        block_size=block_size,
        num_gpu_blocks=-(-40 // block_size) + extra_blocks,
        num_cpu_blocks=num_cpu_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=40,
        max_num_batched_tokens=40,
        max_natural_tokens=16,
    )

    async def scenario() -> None:
        cancellations: dict[int, list[asyncio.Task[Any]]] = {}

        async def checked_sleep(_duration: float) -> None:
            assert_scheduler_invariants(engine)
            for task in cancellations.pop(engine.stats.steps, []):
                task.cancel()
            await asyncio.sleep(0)

        engine = MockVllmEngine(config, sleep=checked_sleep)

        async def disconnecting_stream(request: ChatCompletionRequest, after: int) -> None:
            stream = engine.stream_chat(request)
            try:
                for _ in range(after):
                    await anext(stream)
            except StopAsyncIteration:
                return
            finally:
                await stream.aclose()  # type: ignore[attr-defined]

        tasks: list[asyncio.Task[Any]] = []
        for kind, words, max_tokens, point in ops:
            request = _chat(words, max_tokens)
            if kind == "complete":
                tasks.append(asyncio.ensure_future(engine.chat_complete(request)))
            elif kind == "stream":
                tasks.append(asyncio.ensure_future(_drain(engine.stream_chat(request))))
            elif kind == "disconnect":
                tasks.append(asyncio.ensure_future(disconnecting_stream(request, point)))
            else:  # cancel a blocking request after `point` scheduler steps
                task = asyncio.ensure_future(engine.chat_complete(request))
                cancellations.setdefault(point, []).append(task)
                tasks.append(task)
            await asyncio.sleep(0)

        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30)
        await _yield_until(lambda: engine.is_quiescent)
        final = engine.snapshot()
        assert final.total_requests == 0
        assert final.kv_cache.used_gpu_blocks == 0 and final.kv_cache.used_cpu_blocks == 0
        submitted = next(engine._seq_ids)  # sequences ever created
        assert engine.stats.completed_sequences + engine.stats.aborted_sequences == submitted
        for task in tasks:
            if not task.cancelled() and (error := task.exception()) is not None:
                raise error
        await engine.aclose()

    asyncio.run(scenario())


async def _drain(stream: AsyncIterator[StreamChunk]) -> int:
    return sum([1 async for _ in stream])


@settings(max_examples=15, deadline=None)
@given(
    disconnects=st.lists(st.integers(min_value=0, max_value=1500), min_size=1, max_size=6),
)
def test_http_clients_disconnecting_mid_stream_release_their_blocks(
    disconnects: list[int],
) -> None:
    async def scenario() -> None:
        engine = MockVllmEngine(MockEngineConfig(decode_step_s=0.0005, ignore_eos=True))
        server = MockVllmServer(engine)
        host, port = (await server.start()).removeprefix("http://").split(":")
        body = json.dumps(_chat(4, 400).model_dump(mode="json") | {"stream": True}).encode()
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        for nbytes in disconnects:
            reader, writer = await asyncio.open_connection(host, int(port))
            writer.write(request)
            await writer.drain()
            if nbytes:
                await reader.read(nbytes)  # read part of the stream, then vanish
            writer.close()
            await writer.wait_closed()
        try:
            await asyncio.wait_for(_until_settled(server, len(disconnects)), timeout=30)
            assert engine.snapshot().kv_cache.used_gpu_blocks == 0
            settled = engine.stats.aborted_sequences + engine.stats.completed_sequences
            assert settled == len(disconnects)
        finally:
            await server.aclose()

    asyncio.run(scenario())


async def _until_settled(server: MockVllmServer, requests: int) -> None:
    """All *requests* were dispatched, every handler finished, nothing is in flight.

    Counting dispatched requests matters: a client that writes and leaves at
    once may not even have been accepted yet when we start waiting.
    """
    while server.api_requests < requests or server._connections or not server.engine.is_quiescent:
        await asyncio.sleep(0.001)


# --------------------------------------------------------------------------- #
# Fairness: bounded arrivals never starve a request                           #
# --------------------------------------------------------------------------- #


@FUZZ
@given(
    jobs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=6),  # arrival gap, in scheduler steps
            st.integers(min_value=1, max_value=10),  # prompt words
            st.integers(min_value=1, max_value=20),  # output tokens
        ),
        min_size=1,
        max_size=24,
    ),
    num_gpu_blocks=st.integers(min_value=10, max_value=30),
    num_cpu_blocks=st.integers(min_value=0, max_value=10),
    max_num_seqs=st.integers(min_value=1, max_value=5),
)
def test_fcfs_scheduling_has_a_liveness_bound(
    jobs: list[tuple[int, int, int]],
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    max_num_seqs: int,
) -> None:
    config = MockEngineConfig(
        block_size=4,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=40,
        max_num_batched_tokens=40,
        ignore_eos=True,
    )
    tokens = [max_tokens for _, _, max_tokens in jobs]
    arrival: list[int] = []
    first_token: dict[int, int] = {}
    done: dict[int, int] = {}

    async def scenario() -> None:
        async def checked_sleep(_duration: float) -> None:
            assert_scheduler_invariants(engine)
            await asyncio.sleep(0)

        engine = MockVllmEngine(config, sleep=checked_sleep)

        async def client(index: int, request: ChatCompletionRequest) -> None:
            async for chunk in engine.stream_chat(request):
                choice = chunk.choices[0]
                if choice.delta.content and index not in first_token:
                    first_token[index] = engine.stats.steps
                if choice.finish_reason is not None:
                    assert choice.finish_reason is FinishReason.LENGTH
                    done[index] = engine.stats.steps

        tasks = []
        for index, (gap, words, max_tokens) in enumerate(jobs):
            target = engine.stats.steps + gap
            await _yield_until(lambda t=target: engine.stats.steps >= t or engine.is_idle)
            arrival.append(engine.stats.steps)
            tasks.append(asyncio.ensure_future(client(index, _chat(words, max_tokens))))
            await asyncio.sleep(0)
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
        assert engine.stats.steps <= sum(tokens)  # every step emits >= 1 token
        await engine.aclose()

    asyncio.run(scenario())

    assert len(done) == len(jobs), "a request starved"
    # Clients read the step counter when they *consume* an event; the engine
    # yields to them once per step, so an observation trails by at most 1 step.
    observation_lag = 1
    for i in range(len(jobs)):
        bound = sum(tokens[: i + 1]) + observation_lag
        assert done[i] - arrival[i] <= bound, f"request {i} exceeded its liveness bound"
    order = [first_token[i] for i in range(len(jobs))]
    assert order == sorted(order), "a later arrival was served before an earlier one"


# --------------------------------------------------------------------------- #
# Hostile input: framing, Unicode, nesting, Prometheus, HTTP                  #
# --------------------------------------------------------------------------- #

any_text = st.text(alphabet=st.characters(codec=None, categories=None), max_size=200)


@given(chunks=st.lists(st.binary(max_size=300), max_size=20))
def test_line_and_event_buffers_stay_bounded(chunks: list[bytes]) -> None:
    lines, events = LineDecoder(max_line_chars=64), SseDecoder(max_event_bytes=256)
    for chunk in chunks:
        try:
            for line in lines.feed(chunk):
                events.feed(line)
        except EngineProtocolError:
            lines, events = LineDecoder(max_line_chars=64), SseDecoder(max_event_bytes=256)
        assert lines.pending_chars <= 64
        assert events._size <= 256


@given(chunks=st.lists(st.one_of(st.binary(max_size=200), any_text), max_size=12))
def test_stream_observer_raises_only_engine_errors(chunks: list[bytes | str]) -> None:
    observer = StreamObserver(clock=lambda: 0.0, start_s=0.0)
    try:
        for chunk in chunks:
            observer.feed(chunk)
        observer.finish()
    except EngineError:
        pass


@given(
    payload=st.one_of(
        any_text,
        st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=8),
            lambda inner: (
                st.lists(inner, max_size=4)
                | st.dictionaries(st.text(max_size=6), inner, max_size=4)
            ),
            max_leaves=30,
        ).map(json.dumps),
    )
)
def test_sse_event_decoding_raises_only_engine_errors(payload: str) -> None:
    try:
        assert isinstance(decode_sse_event(payload), StreamChunk)
    except EngineError:
        pass


@pytest.mark.parametrize("depth", [50, 900, 990, 5_000, 200_000])
@pytest.mark.parametrize("kind", ["list", "dict"])
def test_deeply_nested_json_is_a_protocol_error_not_a_crash(depth: int, kind: str) -> None:
    payload = "[" * depth + "]" * depth if kind == "list" else '{"a":' * depth + "1" + "}" * depth
    with pytest.raises(EngineProtocolError):
        decode_sse_event(payload)


@given(text=any_text)
def test_prometheus_parser_raises_only_protocol_errors(text: str) -> None:
    try:
        telemetry_from_prometheus(text)
    except EngineProtocolError:
        pass


@pytest.mark.asyncio
async def test_client_refuses_to_buffer_an_endless_line() -> None:
    sent = 0

    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal sent
            while sent < 50 * 1024 * 1024:  # would be 50 MiB without the bound
                sent += 64 * 1024
                yield b"x" * (64 * 1024)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Endless())

    client = VllmClient(
        VllmDeploymentConfig(model=MODEL),
        transport=httpx.MockTransport(handler),
        retry=RetryPolicy(max_attempts=1),
        max_sse_event_bytes=256 * 1024,
    )
    async with client:
        with pytest.raises(EngineProtocolError, match="without a newline"):
            [chunk async for chunk in client.stream_chat(_chat(1, 1))]
    assert sent <= 512 * 1024, "the client kept reading after the bound was exceeded"


async def _raw_exchange(server: MockVllmServer, payload: bytes) -> bytes:
    host, port = server.base_url.removeprefix("http://").split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    try:
        with contextlib.suppress(ConnectionError):  # server may stop reading early
            writer.write(payload)
            await writer.drain()
            writer.write_eof()
        try:
            return await asyncio.wait_for(reader.read(), timeout=10)
        except ConnectionError:
            return b""
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (b"GET /" + b"a" * 100_000 + b" HTTP/1.1\r\n\r\n", b"414"),
        (b"GET /health HTTP/1.1\r\nX-Big: " + b"b" * 100_000 + b"\r\n\r\n", b"431"),
        (
            b'POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 6\r\n\r\n{"a":[',
            b"400",
        ),
    ],
)
async def test_server_maps_oversized_and_malformed_input_to_http_errors(
    payload: bytes, status: bytes
) -> None:
    server = MockVllmServer(MockVllmEngine())
    await server.start()
    try:
        response = await _raw_exchange(server, payload)
    finally:
        await server.aclose()
    assert response.startswith(b"HTTP/1.1 " + status)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        '{"model": "mock-vllm", "messages": [{"role": "user", "content": "\\ud800"}]}',
        '{"model": "mock-vllm", "messages": ' + "[" * 5000 + "]" * 5000 + "}",
        '{"model": "mock-vllm", "messages": [{"role": "user", "content": "'
        + "\U0010ffff" * 2000
        + '"}]}',
    ],
    ids=["lone-surrogate", "deep-nesting", "astral-plane"],
)
async def test_server_handles_unicode_and_nesting_extremes(body: str) -> None:
    server = MockVllmServer(MockVllmEngine(MockEngineConfig(decode_step_s=0.0)))
    await server.start()
    async with httpx.AsyncClient(base_url=server.base_url) as http:
        response = await http.post(
            "/v1/chat/completions",
            content=body.encode("utf-8", "surrogatepass"),
            headers={"content-type": "application/json"},
        )
    await server.aclose()
    assert response.status_code in (200, 400)
    assert response.json()  # always a well-formed JSON answer


@settings(max_examples=40, deadline=None)
@given(head=st.binary(min_size=1, max_size=2048))
def test_server_survives_arbitrary_request_bytes(head: bytes) -> None:
    async def scenario() -> None:
        failures: list[dict[str, Any]] = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: failures.append(ctx))
        server = MockVllmServer(MockVllmEngine(MockEngineConfig(decode_step_s=0.0)))
        await server.start()
        try:
            response = await _raw_exchange(server, head)
        finally:
            await server.aclose()
        assert not failures, failures
        assert response == b"" or response.startswith(b"HTTP/1.1 ")

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Isolation proofs                                                            #
# --------------------------------------------------------------------------- #

LEAKY_TESTS = textwrap.dedent(
    """
    import os, socket, threading
    _keep = []

    def test_leaks_a_file_descriptor():
        _keep.append(os.open(os.devnull, os.O_RDONLY))

    def test_leaks_a_socket():
        _keep.append(socket.socket())

    def test_leaks_a_thread():
        stop = threading.Event()
        thread = threading.Thread(target=stop.wait, name="leaky", daemon=True)
        thread.start()
        _keep.append(thread)

    def test_clean():
        pass
    """
)


def test_resource_ledger_catches_leaks(tmp_path: Path) -> None:
    """Meta-test: the conftest ledger must flag leaks, or 'zero leaks' proves nothing."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")  # isolate from the repo's config
    (tmp_path / "conftest.py").write_text((REPO_ROOT / "tests" / "conftest.py").read_text())
    (tmp_path / "test_leaky.py").write_text(LEAKY_TESTS)
    completed = subprocess.run(  # noqa: S603 - fixed argv
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--capture=sys",
            "test_leaky.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert "4 passed, 3 errors" in output, output
    for test in ("file_descriptor", "socket", "thread"):
        assert f"resource leak after test_leaky.py::test_leaks_a_{test}" in output
    assert "test_clean" not in output.split("short test summary")[-1]
    assert "SESSION RESOURCE LEAK" in output


def test_every_cache_and_temp_location_is_anchored(pytestconfig: pytest.Config) -> None:
    def inside_cache(path: str | Path) -> bool:
        return Path(path).resolve().is_relative_to(CACHE_ROOT.resolve())

    assert pytestconfig.getoption("capture") == "sys"  # fd capture would use the host temp dir
    assert inside_cache(tempfile.gettempdir()) and inside_cache(TMP_ROOT)
    assert inside_cache(pytestconfig.cache.mkdir("probe"))
    stray = [
        path.relative_to(REPO_ROOT).as_posix()
        for name in (
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".hypothesis",
            ".coverage",
            "htmlcov",
        )
        for path in REPO_ROOT.glob(name)
    ]
    assert stray == []
