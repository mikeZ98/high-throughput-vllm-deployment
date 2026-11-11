"""VllmClient over httpx.MockTransport: retries, timeouts, error mapping, SSE, metrics."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from test_core_metrics import VLLM_V0_METRICS

from high_throughput_vllm_deployment.core.client import (
    ClientTimeouts,
    ConnectionLimits,
    RetryPolicy,
    VllmClient,
    drop_nulls,
)
from high_throughput_vllm_deployment.core.errors import (
    EngineHTTPError,
    EngineProtocolError,
    EngineRequestError,
    EngineTimeoutError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.core.ports import InferenceEnginePort
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import FinishReason
from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    StreamChunk,
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CONFIG = VllmDeploymentConfig(model=MODEL, host="127.0.0.1", port=8000)
CHAT = ChatCompletionRequest(model=MODEL, messages=[ChatMessage(role="user", content="hi")])
COMPLETION = CompletionRequest(model=MODEL, prompt="Hello", max_tokens=4)

Handler = Callable[[httpx.Request], httpx.Response]

# Shapes captured from vLLM's OpenAI server, including its null-valued extensions.
VLLM_COMPLETION: dict[str, Any] = {
    "id": "cmpl-abc",
    "object": "text_completion",
    "created": 1_700_000_000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "text": " world",
            "logprobs": None,
            "finish_reason": "length",
            "stop_reason": None,
            "prompt_logprobs": None,
        }
    ],
    "usage": {
        "prompt_tokens": 2,
        "total_tokens": 6,
        "completion_tokens": 4,
        "prompt_tokens_details": None,
    },
    "system_fingerprint": None,
    "service_tier": None,
    "kv_transfer_params": None,
}
VLLM_CHAT: dict[str, Any] = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": None,
                "content": "Hi!",
                "tool_calls": [],
            },
            "logprobs": None,
            "finish_reason": "stop",
            "stop_reason": None,
        }
    ],
    "usage": {"prompt_tokens": 9, "total_tokens": 12, "completion_tokens": 3},
    "prompt_logprobs": None,
    "system_fingerprint": "fp-1",
}


def _chunk(content: str | None, *, role: str | None = None, finish: str | None = None) -> str:
    delta: dict[str, Any] = {"content": content}
    if role is not None:
        delta["role"] = role
    payload = {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


SSE_BODY = (
    _chunk("", role="assistant")
    + ": keep-alive\n\n"
    + _chunk("Hel")
    + _chunk("lo", finish="stop")
    + "data: [DONE]\n\n"
)


class _Recorder:
    """MockTransport handler that replays a scripted sequence of outcomes."""

    def __init__(self, *outcomes: httpx.Response | Exception | Handler) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []
        self._served: set[int] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx.Response):
            if id(outcome) in self._served:  # a response object can only be sent once
                return httpx.Response(
                    outcome.status_code, headers=outcome.headers, content=outcome.content
                )
            self._served.add(id(outcome))
            return outcome
        return outcome(request)


class _SleepLog:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _client(
    recorder: _Recorder, *, retry: RetryPolicy | None = None, sleep: _SleepLog | None = None
) -> VllmClient:
    return VllmClient(
        CONFIG,
        retry=retry or RetryPolicy(max_attempts=3, initial_backoff_s=0.1, max_backoff_s=1.0),
        transport=httpx.MockTransport(recorder),
        sleep=sleep or _SleepLog(),
    )


def _sse(body: str | bytes, status: int = 200) -> httpx.Response:
    content = body.encode() if isinstance(body, str) else body
    return httpx.Response(status, content=content, headers={"content-type": "text/event-stream"})


# --------------------------------------------------------------------------- #
# Policies and helpers                                                        #
# --------------------------------------------------------------------------- #


def test_client_satisfies_the_engine_port() -> None:
    port: InferenceEnginePort = VllmClient(CONFIG)
    assert isinstance(port, InferenceEnginePort)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"initial_backoff_s": -1.0}, "initial_backoff_s"),
        ({"initial_backoff_s": 2.0, "max_backoff_s": 1.0}, "initial_backoff_s"),
        ({"multiplier": 0.5}, "multiplier"),
    ],
)
def test_retry_policy_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RetryPolicy(**kwargs)


def test_backoff_is_exponential_capped_and_honours_retry_after() -> None:
    policy = RetryPolicy(initial_backoff_s=0.1, multiplier=2.0, max_backoff_s=0.5)
    assert [policy.backoff_s(n) for n in (1, 2, 3, 4)] == pytest.approx([0.1, 0.2, 0.4, 0.5])
    assert policy.backoff_s(1, "0.3") == pytest.approx(0.3)
    assert policy.backoff_s(1, "60") == 0.5
    assert policy.backoff_s(2, "Wed, 21 Oct 2015 07:28:00 GMT") == pytest.approx(0.2)


def test_timeouts_and_limits_map_onto_httpx() -> None:
    timeouts = ClientTimeouts(connect_s=1.0, read_s=2.0, write_s=3.0, pool_s=4.0)
    client = VllmClient(CONFIG, timeouts=timeouts, limits=ConnectionLimits(max_connections=7))
    assert client._http.timeout == httpx.Timeout(connect=1.0, read=2.0, write=3.0, pool=4.0)
    assert ConnectionLimits(max_connections=7).to_httpx().max_connections == 7


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("0.0.0.0", "http://127.0.0.1:8000"),  # noqa: S104 - bind wildcard mapped to loopback
        ("::", "http://[::1]:8000"),
        ("::1", "http://[::1]:8000"),
        ("vllm.internal", "http://vllm.internal:8000"),
    ],
)
def test_base_url_targets_a_connectable_host(host: str, url: str) -> None:
    config = VllmDeploymentConfig(model=MODEL, host=host)
    assert VllmClient(config).base_url == url


def test_explicit_base_url_wins() -> None:
    assert VllmClient(CONFIG, base_url="http://gateway:9000").base_url == "http://gateway:9000"


def test_drop_nulls_is_recursive_and_keeps_falsy_values() -> None:
    raw = {"a": None, "b": [{"c": None, "d": 0}, None, ""], "e": {"f": False, "g": None}}
    assert drop_nulls(raw) == {"b": [{"d": 0}, None, ""], "e": {"f": False}}


# --------------------------------------------------------------------------- #
# Non-streaming calls                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_complete_posts_contract_body_and_parses_real_vllm_shape() -> None:
    recorder = _Recorder(httpx.Response(200, json=VLLM_COMPLETION))
    async with _client(recorder) as client:
        response = await client.complete(COMPLETION.model_copy(update={"stream": True}))
    request = recorder.requests[0]
    assert (request.method, request.url.path) == ("POST", "/v1/completions")
    body = json.loads(request.content)
    assert body == {
        "model": MODEL,
        "prompt": "Hello",
        "max_tokens": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
    }
    assert response.choices[0].finish_reason is FinishReason.LENGTH
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_chat_complete_parses_real_vllm_shape() -> None:
    async with _client(_Recorder(httpx.Response(200, json=VLLM_CHAT))) as client:
        response = await client.chat_complete(CHAT)
    assert response.choices[0].message.content == "Hi!"
    assert response.choices[0].message.tool_calls == []
    assert response.system_fingerprint == "fp-1"


@pytest.mark.asyncio
async def test_non_null_unknown_fields_are_still_rejected() -> None:
    payload = VLLM_CHAT | {"unexpected": {"x": 1}}
    async with _client(_Recorder(httpx.Response(200, json=payload))) as client:
        with pytest.raises(EngineProtocolError, match="ChatCompletionResponse"):
            await client.chat_complete(CHAT)


@pytest.mark.asyncio
async def test_non_json_body_is_a_protocol_error() -> None:
    async with _client(_Recorder(httpx.Response(200, text="<html>"))) as client:
        with pytest.raises(EngineProtocolError, match="CompletionResponse"):
            await client.complete(COMPLETION)


# --------------------------------------------------------------------------- #
# Retries and error mapping                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [502, 503, 504])
async def test_transient_statuses_are_retried_with_backoff(status: int) -> None:
    sleep = _SleepLog()
    recorder = _Recorder(
        httpx.Response(status), httpx.Response(status), httpx.Response(200, json=VLLM_COMPLETION)
    )
    async with _client(recorder, sleep=sleep) as client:
        response = await client.complete(COMPLETION)
    assert response.id == "cmpl-abc"
    assert len(recorder.requests) == 3
    assert sleep.delays == pytest.approx([0.1, 0.2])


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured() -> None:
    sleep = _SleepLog()
    recorder = _Recorder(
        httpx.Response(503, headers={"Retry-After": "0.75"}),
        httpx.Response(200, json=VLLM_COMPLETION),
    )
    async with _client(recorder, sleep=sleep) as client:
        await client.complete(COMPLETION)
    assert sleep.delays == [0.75]


@pytest.mark.asyncio
async def test_exhausted_retries_raise_unavailable_with_detail() -> None:
    sleep = _SleepLog()
    recorder = _Recorder(httpx.Response(503, text="engine overloaded"))
    async with _client(recorder, sleep=sleep) as client:
        with pytest.raises(
            EngineUnavailableError, match=r"HTTP 503 after 3 attempt\(s\): engine overloaded"
        ):
            await client.complete(COMPLETION)
    assert len(recorder.requests) == 3
    assert len(sleep.delays) == 2


@pytest.mark.asyncio
async def test_connect_errors_are_retried_then_succeed() -> None:
    recorder = _Recorder(httpx.ConnectError("refused"), httpx.Response(200, json=VLLM_COMPLETION))
    async with _client(recorder) as client:
        assert (await client.complete(COMPLETION)).id == "cmpl-abc"
    assert len(recorder.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "error", "attempts"),
    [
        (httpx.ConnectError("refused"), EngineUnavailableError, 3),
        (httpx.ConnectTimeout("connect timed out"), EngineTimeoutError, 3),
        (httpx.PoolTimeout("pool exhausted"), EngineTimeoutError, 3),
        (httpx.ReadTimeout("read timed out"), EngineTimeoutError, 1),
        (httpx.WriteTimeout("write timed out"), EngineTimeoutError, 1),
        (httpx.RemoteProtocolError("peer closed"), EngineUnavailableError, 1),
    ],
)
async def test_transport_faults_map_to_engine_errors(
    fault: Exception, error: type[Exception], attempts: int
) -> None:
    recorder = _Recorder(fault)
    async with _client(recorder) as client:
        with pytest.raises(error, match=rf"{type(fault).__name__} after {attempts} attempt"):
            await client.complete(COMPLETION)
    assert len(recorder.requests) == attempts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, EngineRequestError),
        (404, EngineRequestError),
        (422, EngineRequestError),
        (500, EngineHTTPError),
    ],
)
async def test_non_retryable_statuses_fail_fast(status: int, error: type[EngineHTTPError]) -> None:
    recorder = _Recorder(httpx.Response(status, json={"message": "nope"}))
    async with _client(recorder) as client:
        with pytest.raises(error, match=f"HTTP {status}") as info:
            await client.complete(COMPLETION)
    assert info.value.status_code == status
    assert len(recorder.requests) == 1


@pytest.mark.asyncio
async def test_empty_error_body_falls_back_to_reason_phrase() -> None:
    async with _client(_Recorder(httpx.Response(500))) as client:
        with pytest.raises(EngineHTTPError, match="Internal Server Error"):
            await client.complete(COMPLETION)


# --------------------------------------------------------------------------- #
# Streaming                                                                   #
# --------------------------------------------------------------------------- #


async def _collect(client: VllmClient) -> list[StreamChunk]:
    return [chunk async for chunk in client.stream_chat(CHAT)]


@pytest.mark.asyncio
async def test_stream_yields_chunks_until_done() -> None:
    recorder = _Recorder(_sse(SSE_BODY))
    async with _client(recorder) as client:
        chunks = await _collect(client)
    assert json.loads(recorder.requests[0].content)["stream"] is True
    assert [c.choices[0].delta.content for c in chunks] == ["", "Hel", "lo"]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason is FinishReason.STOP


@pytest.mark.asyncio
async def test_stream_stops_at_done_and_ignores_trailing_bytes() -> None:
    async with _client(_Recorder(_sse(SSE_BODY + _chunk("ignored")))) as client:
        assert len(await _collect(client)) == 3


@pytest.mark.asyncio
async def test_stream_accepts_done_without_trailing_blank_line() -> None:
    async with _client(_Recorder(_sse(_chunk("a") + "data: [DONE]"))) as client:
        assert len(await _collect(client)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [_chunk("a"), _chunk("a").rstrip("\n")], ids=["terminated", "unterminated"]
)
async def test_stream_without_done_is_a_protocol_error(body: str) -> None:
    seen: list[StreamChunk] = []
    async with _client(_Recorder(_sse(body))) as client:
        with pytest.raises(EngineProtocolError, match=r"ended before \[DONE\]"):
            async for chunk in client.stream_chat(CHAT):
                seen.append(chunk)
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "error", "match"),
    [
        ("data: {not json\n\n", EngineProtocolError, "not JSON"),
        ('data: {"id": "x"}\n\n', EngineProtocolError, "StreamChunk"),
        ('data: {"error": {"message": "OOM", "code": 500}}\n\n', EngineHTTPError, "OOM"),
    ],
)
async def test_stream_rejects_bad_events(body: str, error: type[Exception], match: str) -> None:
    async with _client(_Recorder(_sse(body))) as client:
        with pytest.raises(error, match=match):
            await _collect(client)


@pytest.mark.asyncio
async def test_stream_retries_before_the_first_byte() -> None:
    recorder = _Recorder(httpx.Response(503), _sse(SSE_BODY))
    async with _client(recorder) as client:
        assert len(await _collect(client)) == 3
    assert len(recorder.requests) == 2


class _StallingStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _chunk("partial").encode()
        raise httpx.ReadTimeout("stalled")


@pytest.mark.asyncio
async def test_mid_stream_timeout_maps_to_engine_timeout_and_is_not_retried() -> None:
    recorder = _Recorder(httpx.Response(200, stream=_StallingStream()))
    seen: list[StreamChunk] = []
    async with _client(recorder) as client:
        with pytest.raises(EngineTimeoutError, match="ReadTimeout"):
            async for chunk in client.stream_chat(CHAT):
                seen.append(chunk)
    assert len(seen) == 1
    assert len(recorder.requests) == 1


@pytest.mark.asyncio
async def test_oversized_sse_event_is_rejected() -> None:
    client = VllmClient(
        CONFIG,
        transport=httpx.MockTransport(_Recorder(_sse("data: " + "x" * 64 + "\n\n"))),
        max_sse_event_bytes=32,
    )
    async with client:
        with pytest.raises(EngineProtocolError, match="exceeds 32 bytes"):
            await _collect(client)


@pytest.mark.asyncio
async def test_closing_the_stream_early_releases_the_response() -> None:
    closed: list[bool] = []

    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield SSE_BODY.encode()

        async def aclose(self) -> None:
            closed.append(True)

    async with _client(_Recorder(httpx.Response(200, stream=_Body()))) as client:
        stream = client.stream_chat(CHAT)
        first = await anext(stream)
        await stream.aclose()  # type: ignore[attr-defined]
    assert first.choices[0].delta.role == "assistant"
    assert closed


# --------------------------------------------------------------------------- #
# Telemetry and health                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_telemetry_scrapes_metrics() -> None:
    recorder = _Recorder(httpx.Response(200, text=VLLM_V0_METRICS))
    async with _client(recorder) as client:
        telemetry = await client.get_telemetry()
    assert recorder.requests[0].url.path == "/metrics"
    assert telemetry.active_requests == 12
    assert telemetry.kv_cache.used_gpu_blocks == 512


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "healthy"),
    [
        (httpx.Response(200), True),
        (httpx.Response(503), False),
        (httpx.ConnectError("refused"), False),
    ],
)
async def test_is_healthy_never_raises_and_never_retries(
    outcome: httpx.Response | Exception, healthy: bool
) -> None:
    recorder = _Recorder(outcome)
    async with _client(recorder) as client:
        assert await client.is_healthy() is healthy
    assert [r.url.path for r in recorder.requests] == ["/health"]
