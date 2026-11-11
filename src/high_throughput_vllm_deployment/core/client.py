"""Production async adapter for the vLLM OpenAI-compatible HTTP server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Self, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from high_throughput_vllm_deployment.core._clock import Sleep
from high_throughput_vllm_deployment.core.errors import (
    EngineError,
    EngineHTTPError,
    EngineProtocolError,
    EngineRequestError,
    EngineTimeoutError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.core.metrics import telemetry_from_prometheus
from high_throughput_vllm_deployment.core.sse import (
    DEFAULT_MAX_EVENT_BYTES,
    LineDecoder,
    SseDecoder,
)
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.inference import (
    SSE_DONE_SENTINEL,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    SamplingParams,
    StreamChunk,
)
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry

_ModelT = TypeVar("_ModelT", bound=BaseModel)


COMPLETIONS_PATH: Final = "/v1/completions"
CHAT_COMPLETIONS_PATH: Final = "/v1/chat/completions"
METRICS_PATH: Final = "/metrics"
HEALTH_PATH: Final = "/health"

_WILDCARD_TO_LOOPBACK: Final = {"0.0.0.0": "127.0.0.1", "::": "::1"}  # noqa: S104
_ERROR_BODY_LIMIT: Final = 512

# Faults where the request provably never reached the engine: safe to retry
# even though completion requests are not idempotent.
_RETRYABLE_EXCEPTIONS: Final = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff for transient faults.

    Retried: HTTP 502/503/504 (overloaded/restarting engine or proxy) and
    connect/pool failures. Never retried: read/write timeouts and dropped
    connections, where the engine may already be generating.
    """

    max_attempts: int = 4
    initial_backoff_s: float = 0.1
    multiplier: float = 2.0
    max_backoff_s: float = 2.0
    retry_statuses: frozenset[int] = frozenset({502, 503, 504})

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_backoff_s < 0 or self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("require 0 <= initial_backoff_s <= max_backoff_s")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def backoff_s(self, retry_number: int, retry_after: str | None = None) -> float:
        """Delay before retry *retry_number* (1-based), honouring a numeric Retry-After."""
        delay = self.initial_backoff_s * self.multiplier ** (retry_number - 1)
        if retry_after is not None:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        return min(delay, self.max_backoff_s)


@dataclass(frozen=True, slots=True)
class ClientTimeouts:
    """Per-phase timeouts, mapped 1:1 onto :class:`httpx.Timeout`.

    ``read_s`` bounds the gap between bytes, so for streams it is the maximum
    tolerated inter-token stall, not the total generation time.
    """

    connect_s: float = 5.0
    read_s: float = 120.0
    write_s: float = 10.0
    pool_s: float = 5.0

    def to_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_s, read=self.read_s, write=self.write_s, pool=self.pool_s
        )


@dataclass(frozen=True, slots=True)
class ConnectionLimits:
    max_connections: int = 512
    max_keepalive_connections: int = 128
    keepalive_expiry_s: float = 30.0

    def to_httpx(self) -> httpx.Limits:
        return httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry_s,
        )


def drop_nulls(value: Any) -> Any:
    """Recursively remove ``null``-valued object members.

    vLLM serialises every optional extension field it knows about (often as
    ``null``), and the set differs between releases. A null member carries no
    information and every optional contract field already defaults to
    ``None``, so pruning is lossless for known fields and lets the contracts
    keep ``extra="forbid"``: any *non-null* unknown field is still rejected.
    """
    if isinstance(value, dict):
        return {key: drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value]
    return value


def decode_sse_event(data: str) -> StreamChunk:
    """Decode one SSE ``data`` payload (not ``[DONE]``) into a :class:`StreamChunk`.

    A ``{"error": ...}`` payload (vLLM's mid-stream failure event) raises
    :class:`EngineHTTPError`; anything else that is not a valid chunk raises
    :class:`EngineProtocolError`.
    """
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EngineProtocolError(f"SSE event is not JSON: {data[:80]!r}") from exc
    if isinstance(payload, dict) and "error" in payload:
        raise EngineHTTPError(
            f"engine reported a mid-stream error: {payload['error']}", status_code=500
        )
    try:
        return StreamChunk.model_validate(drop_nulls(payload))
    except (ValidationError, RecursionError) as exc:  # RecursionError: absurdly deep JSON
        raise EngineProtocolError(f"invalid StreamChunk payload: {exc}") from exc


def _base_url(config: VllmDeploymentConfig) -> str:
    host = _WILDCARD_TO_LOOPBACK.get(config.host, config.host)
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{config.port}"


def _request_body(request: SamplingParams, *, stream: bool) -> dict[str, Any]:
    body = request.model_dump(mode="json", exclude_none=True)
    body["stream"] = stream
    if not stream:
        body.pop("stream_options", None)  # vLLM rejects stream_options without stream
    return body


def _error_detail(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:_ERROR_BODY_LIMIT] or response.reason_phrase


class VllmClient:
    """:class:`InferenceEnginePort` over a pooled :class:`httpx.AsyncClient`."""

    def __init__(
        self,
        config: VllmDeploymentConfig,
        *,
        base_url: str | None = None,
        retry: RetryPolicy | None = None,
        timeouts: ClientTimeouts | None = None,
        limits: ConnectionLimits | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
        max_sse_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    ) -> None:
        self.config = config
        self.retry = retry or RetryPolicy()
        self.timeouts = timeouts or ClientTimeouts()
        self._sleep = sleep
        self._max_sse_event_bytes = max_sse_event_bytes
        self._http = httpx.AsyncClient(
            base_url=base_url or _base_url(config),
            timeout=self.timeouts.to_httpx(),
            limits=(limits or ConnectionLimits()).to_httpx(),
            transport=transport,
            headers={"Accept": "application/json"},
        )

    @property
    def base_url(self) -> str:
        return str(self._http.base_url)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # Transport                                                          #
    # ------------------------------------------------------------------ #

    async def _send(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        stream: bool = False,
        retry: bool = True,
    ) -> httpx.Response:
        """Send with retries; return a 2xx response (unread when *stream*)."""
        max_attempts = self.retry.max_attempts if retry else 1
        for attempt in range(1, max_attempts + 1):
            last = attempt == max_attempts
            request = self._http.build_request(method, path, json=body)
            try:
                response = await self._http.send(request, stream=stream)
            except _RETRYABLE_EXCEPTIONS as exc:
                if last:
                    raise self._map_transport_error(exc, attempt) from exc
                await self._sleep(self.retry.backoff_s(attempt))
                continue
            except httpx.HTTPError as exc:
                raise self._map_transport_error(exc, attempt) from exc

            if response.is_success:
                return response
            await response.aread()
            await response.aclose()
            status = response.status_code
            if status in self.retry.retry_statuses:
                if last:
                    raise EngineUnavailableError(
                        f"{method} {path}: HTTP {status} after {attempt} attempt(s): "
                        f"{_error_detail(response)}"
                    )
                await self._sleep(
                    self.retry.backoff_s(attempt, response.headers.get("Retry-After"))
                )
                continue
            error_cls = EngineRequestError if 400 <= status < 500 else EngineHTTPError
            raise error_cls(
                f"{method} {path}: HTTP {status}: {_error_detail(response)}", status_code=status
            )
        raise AssertionError("unreachable")  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _map_transport_error(exc: httpx.HTTPError, attempts: int) -> EngineError:
        suffix = f" after {attempts} attempt(s)"
        if isinstance(exc, httpx.TimeoutException):
            return EngineTimeoutError(f"{type(exc).__name__}{suffix}: {exc}")
        return EngineUnavailableError(f"{type(exc).__name__}{suffix}: {exc}")

    @staticmethod
    def _parse(model: type[_ModelT], response: httpx.Response) -> _ModelT:
        try:
            return model.model_validate(drop_nulls(response.json()))
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, RecursionError) as exc:
            raise EngineProtocolError(f"invalid {model.__name__} payload: {exc}") from exc

    # ------------------------------------------------------------------ #
    # InferenceEnginePort                                                #
    # ------------------------------------------------------------------ #

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        response = await self._send(
            "POST", COMPLETIONS_PATH, body=_request_body(request, stream=False)
        )
        return self._parse(CompletionResponse, response)

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = await self._send(
            "POST", CHAT_COMPLETIONS_PATH, body=_request_body(request, stream=False)
        )
        return self._parse(ChatCompletionResponse, response)

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        response = await self._send(
            "POST", CHAT_COMPLETIONS_PATH, body=_request_body(request, stream=True), stream=True
        )
        lines = LineDecoder(max_line_chars=self._max_sse_event_bytes)
        decoder = SseDecoder(max_event_bytes=self._max_sse_event_bytes)
        try:
            async for raw in response.aiter_bytes():
                for line in lines.feed(raw):
                    data = decoder.feed(line)
                    if data is None:
                        continue
                    if data == SSE_DONE_SENTINEL:
                        return
                    yield decode_sse_event(data)
            decoder.feed(lines.flush())
            tail = decoder.flush()
            if tail == SSE_DONE_SENTINEL:
                return
            if tail is not None:
                yield decode_sse_event(tail)
            raise EngineProtocolError("stream ended before [DONE]")
        except httpx.HTTPError as exc:
            # Mid-stream faults are never retried: tokens were already delivered.
            raise self._map_transport_error(exc, 1) from exc
        finally:
            await response.aclose()

    async def get_telemetry(self) -> ClusterTelemetry:
        response = await self._send("GET", METRICS_PATH)
        return telemetry_from_prometheus(response.text)

    async def is_healthy(self) -> bool:
        try:
            await self._send("GET", HEALTH_PATH, retry=False)
        except EngineError:
            return False
        return True
