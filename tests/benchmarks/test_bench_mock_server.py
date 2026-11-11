"""MockVllmServer: HTTP/1.1 framing, OpenAI routes, SSE, metrics, faults, threaded mode, CLI."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from high_throughput_vllm_deployment.benchmarks.mock_server import (
    MockVllmServer,
    ThreadedMockServer,
    main,
    render_vllm_metrics,
)
from high_throughput_vllm_deployment.core import instant_sleep
from high_throughput_vllm_deployment.core.client import RetryPolicy, VllmClient
from high_throughput_vllm_deployment.core.errors import (
    EngineHTTPError,
    EngineRequestError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.core.metrics import telemetry_from_prometheus
from high_throughput_vllm_deployment.core.mock import MockEngineConfig, MockVllmEngine
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import FinishReason
from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    StreamOptions,
)
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry, KvCacheBlocks

MODEL = "mock-vllm"
FAST = MockEngineConfig(decode_step_s=0.0, prefill_s_per_token=0.0, ignore_eos=True)
CHAT = ChatCompletionRequest(
    model=MODEL, messages=[ChatMessage(role="user", content="hello there")], max_tokens=4
)


async def _server(
    *, fault_every: int = 0, config: MockEngineConfig = FAST, **engine_kwargs: Any
) -> MockVllmServer:
    server = MockVllmServer(MockVllmEngine(config, **engine_kwargs), fault_every=fault_every)
    await server.start()
    return server


def _client(server: MockVllmServer, **kwargs: Any) -> VllmClient:
    return VllmClient(
        VllmDeploymentConfig(model=MODEL),
        base_url=server.base_url,
        retry=RetryPolicy(max_attempts=1),
        **kwargs,
    )


async def _raw(server: MockVllmServer, payload: bytes, *, read_all: bool = True) -> bytes:
    host, port = server.base_url.removeprefix("http://").split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(payload)
    await writer.drain()
    data = await reader.read() if read_all else await reader.read(4096)
    writer.close()
    await writer.wait_closed()
    return data


# --------------------------------------------------------------------------- #
# OpenAI surface via VllmClient                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_client_round_trips_every_route() -> None:
    server = await _server()
    async with _client(server) as client:
        assert await client.is_healthy()
        chat = await client.chat_complete(CHAT)
        completion = await client.complete(
            CompletionRequest(model=MODEL, prompt="hi", max_tokens=3)
        )
        stream_request = CHAT.model_copy(
            update={"stream_options": StreamOptions(include_usage=True)}
        )
        chunks = [chunk async for chunk in client.stream_chat(stream_request)]
        telemetry = await client.get_telemetry()
    await server.aclose()
    await server.aclose()  # idempotent

    assert chat.choices[0].finish_reason is FinishReason.LENGTH
    assert chat.usage.completion_tokens == 4
    assert completion.usage.completion_tokens == 3
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 4
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == chat.choices[0].message.content
    assert telemetry.kv_cache.total_gpu_blocks == FAST.num_gpu_blocks
    assert server.api_requests == 3


@pytest.mark.asyncio
async def test_engine_errors_map_to_http_statuses() -> None:
    server = await _server()
    wrong_model = CHAT.model_copy(update={"model": "other"})
    async with _client(server) as client:
        with pytest.raises(EngineRequestError, match="HTTP 404"):
            await client.chat_complete(wrong_model)
        with pytest.raises(EngineRequestError, match="HTTP 404"):
            [chunk async for chunk in client.stream_chat(wrong_model)]
        with pytest.raises(EngineRequestError, match="does not stream"):
            await client._send(
                "POST", "/v1/completions", body={"model": MODEL, "prompt": "x", "stream": True}
            )
        with pytest.raises(EngineRequestError, match="validation error"):
            await client._send("POST", "/v1/chat/completions", body={"model": MODEL})
        await server.engine.aclose()  # closed engine -> 503
        with pytest.raises(EngineUnavailableError, match="HTTP 503"):
            await client.chat_complete(CHAT)
        with pytest.raises(EngineUnavailableError, match="HTTP 503"):
            [chunk async for chunk in client.stream_chat(CHAT)]
    await server.aclose()


@pytest.mark.asyncio
async def test_fault_injection_is_deterministic() -> None:
    server = await _server(fault_every=3)
    statuses = []
    async with httpx.AsyncClient(base_url=server.base_url) as http:
        for _ in range(6):
            body = CHAT.model_dump(mode="json")
            statuses.append((await http.post("/v1/chat/completions", json=body)).status_code)
        assert (await http.get("/health")).status_code == 200  # not counted as API traffic
    await server.aclose()
    assert statuses == [200, 200, 503, 200, 200, 503]


@pytest.mark.asyncio
async def test_mid_stream_engine_failure_is_sent_as_error_event() -> None:
    steps = 0

    async def failing_sleep(_duration: float) -> None:
        nonlocal steps
        steps += 1
        if steps == 3:
            raise RuntimeError("device lost")
        await asyncio.sleep(0)

    server = await _server(sleep=failing_sleep)
    async with httpx.AsyncClient(base_url=server.base_url) as http:
        body = CHAT.model_dump(mode="json") | {"stream": True, "max_tokens": 10}
        response = await http.post("/v1/chat/completions", json=body)
    await server.aclose()
    events = [line for line in response.text.split("\n\n") if line]
    assert '"error"' in events[-2] and "device lost" in events[-2]
    assert events[-1] == "data: [DONE]"


# --------------------------------------------------------------------------- #
# HTTP/1.1 framing                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (b"NONSENSE\r\n\r\n", b"400"),
        (b"GET /health SPDY/3\r\n\r\n", b"400"),
        (b"GET /health HTTP/1.1\r\nbroken header\r\n\r\n", b"400"),
        (b"GET /health HTTP/1.1\r\n" + b"X-A: b\r\n" * 101 + b"\r\n", b"400"),
        (b"POST /v1/completions HTTP/1.1\r\nContent-Length: nan\r\n\r\n", b"400"),
        (b"POST /v1/completions HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n", b"413"),
        (b"GET /nowhere HTTP/1.1\r\nConnection: close\r\n\r\n", b"404"),
        (b"GET /v1/completions HTTP/1.1\r\nConnection: close\r\n\r\n", b"405"),
        (b"POST /v1/chat/completions HTTP/1.1\r\nConnection: close\r\n\r\n", b"400"),
    ],
)
async def test_malformed_or_unroutable_requests(payload: bytes, status: bytes) -> None:
    server = await _server()
    response = await _raw(server, payload)
    await server.aclose()
    assert response.startswith(b"HTTP/1.1 " + status)
    assert b'"object": "error"' in response


@pytest.mark.asyncio
async def test_keep_alive_and_connection_semantics() -> None:
    server = await _server()
    pipelined = (
        b"GET /health HTTP/1.1\r\n\r\n"
        b"GET /metrics?verbose=1 HTTP/1.1\r\n\r\n"
        b"GET /health HTTP/1.0\r\n\r\n"  # HTTP/1.0 without keep-alive closes
        b"GET /health HTTP/1.1\r\n\r\n"  # never answered
    )
    response = await _raw(server, pipelined)
    legacy_keepalive = await _raw(
        server, b"GET /health HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", read_all=False
    )
    await server.aclose()
    assert response.count(b"HTTP/1.1 200 OK") == 3
    assert b"vllm:num_requests_running" in response
    assert response.rstrip().endswith(b"Connection: close")
    assert b"Connection: keep-alive" in legacy_keepalive


@pytest.mark.asyncio
async def test_truncated_body_and_idle_connections_are_cleaned_up() -> None:
    server = await _server()
    host, port = server.base_url.removeprefix("http://").split(":")
    _, writer = await asyncio.open_connection(host, int(port))
    writer.write(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 50\r\n\r\n{}")
    await writer.drain()
    writer.close()  # half a body, then gone: IncompleteReadError path
    idle_reader, idle_writer = await asyncio.open_connection(host, int(port))
    await asyncio.sleep(0.01)
    await server.aclose()  # cancels the idle keep-alive connection
    assert await idle_reader.read() == b""
    idle_writer.close()


def test_negative_fault_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="fault_every"):
        MockVllmServer(MockVllmEngine(), fault_every=-1)


# --------------------------------------------------------------------------- #
# Metrics exposition                                                          #
# --------------------------------------------------------------------------- #


@given(
    total=st.integers(min_value=1, max_value=4096),
    used=st.data(),
    cpu_total=st.integers(min_value=0, max_value=512),
    running=st.integers(min_value=0, max_value=256),
    waiting=st.integers(min_value=0, max_value=256),
    swapped=st.integers(min_value=0, max_value=64),
)
def test_rendered_metrics_round_trip_through_the_vllm_parser(
    total: int, used: st.DataObject, cpu_total: int, running: int, waiting: int, swapped: int
) -> None:
    used_gpu = used.draw(st.integers(min_value=0, max_value=total))
    used_cpu = used.draw(st.integers(min_value=0, max_value=cpu_total))
    telemetry = ClusterTelemetry(
        active_requests=running,
        waiting_requests=waiting,
        swapped_requests=swapped,
        gpu_cache_usage_pct=100.0 * used_gpu / total,
        cpu_cache_usage_pct=100.0 * used_cpu / cpu_total if cpu_total else 0.0,
        kv_cache=KvCacheBlocks(
            block_size=16,
            total_gpu_blocks=total,
            used_gpu_blocks=used_gpu,
            total_cpu_blocks=cpu_total,
            used_cpu_blocks=used_cpu,
        ),
    )
    parsed = telemetry_from_prometheus(render_vllm_metrics(telemetry, 'model "quoted"'))
    assert parsed.kv_cache == telemetry.kv_cache
    assert parsed.requests_by_state == telemetry.requests_by_state
    assert parsed.gpu_cache_usage_pct == pytest.approx(telemetry.gpu_cache_usage_pct)


# --------------------------------------------------------------------------- #
# Threaded mode and CLI                                                       #
# --------------------------------------------------------------------------- #


def test_threaded_server_serves_synchronous_clients() -> None:
    with ThreadedMockServer(FAST, sleep=instant_sleep) as base_url:
        with httpx.Client(base_url=base_url) as http:
            response = http.post("/v1/chat/completions", json=CHAT.model_dump(mode="json"))
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 4


def test_threaded_server_reports_startup_failure() -> None:
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match="did not start"):
            with ThreadedMockServer(FAST, port=port):
                pass  # pragma: no cover - never entered
    finally:
        blocker.close()


def test_cli_serves_for_a_bounded_time(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--port", "0", "--run-seconds", "0.05", "--ignore-eos", "--fault-every", "2"]) == 0
    assert "mock vLLM server listening on http://127.0.0.1:" in capsys.readouterr().out


def test_mid_stream_http_error_is_surfaced_by_the_client() -> None:
    async def scenario() -> None:
        steps = 0

        async def failing_sleep(_duration: float) -> None:
            nonlocal steps
            steps += 1
            if steps == 3:
                raise RuntimeError("device lost")
            await asyncio.sleep(0)

        server = await _server(sleep=failing_sleep)
        async with _client(server) as client:
            with pytest.raises(EngineHTTPError, match="mid-stream error"):
                [
                    chunk
                    async for chunk in client.stream_chat(
                        CHAT.model_copy(update={"max_tokens": 10})
                    )
                ]
        await server.aclose()

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_server_as_async_context_manager() -> None:
    async with MockVllmServer(MockVllmEngine(FAST)) as server:
        async with httpx.AsyncClient(base_url=server.base_url) as http:
            assert (await http.get("/health")).status_code == 200
    assert not await server.engine.is_healthy()  # engine closed with the server


def test_threaded_server_exit_without_start_is_a_no_op() -> None:
    ThreadedMockServer(FAST).__exit__(None, None, None)


def test_cli_serves_until_sigterm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os
    import signal

    original_start = MockVllmServer.start

    async def start_then_signal(
        self: MockVllmServer, host: str = "127.0.0.1", port: int = 0
    ) -> str:
        url = await original_start(self, host, port)
        # Runs on the next loop turn, i.e. only after serve() installed its handlers.
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
        return url

    monkeypatch.setattr(MockVllmServer, "start", start_then_signal)
    assert main(["--port", "0"]) == 0  # no --run-seconds: runs until signalled
    assert "listening on" in capsys.readouterr().out
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL  # handler removed again
