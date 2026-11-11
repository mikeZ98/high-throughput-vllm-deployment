"""Client-side token timing for OpenAI-compatible streams.

:class:`StreamObserver` timestamps content-bearing chunks, whether they come
as decoded :class:`StreamChunk` objects (in-process ports) or as raw body
bytes (Locust's HTTP clients). Token counts prefer the server's usage chunk
(``stream_options.include_usage``) and fall back to counting content chunks.

:func:`record_from_response` turns a finished Locust/requests-style response
into a :class:`RequestRecord`; it is duck-typed so it stays importable (and
testable) without Locust, whose import monkey-patches the process.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterable
from typing import Any, Final, Protocol

from pydantic import ValidationError

from high_throughput_vllm_deployment.benchmarks.metrics_collector import (
    Clock,
    RequestRecord,
    categorize_exception,
    categorize_status,
)
from high_throughput_vllm_deployment.core.client import decode_sse_event, drop_nulls
from high_throughput_vllm_deployment.core.errors import EngineError, EngineProtocolError
from high_throughput_vllm_deployment.core.sse import LineDecoder, SseDecoder
from high_throughput_vllm_deployment.schemas.inference import (
    SSE_DONE_SENTINEL,
    ChatCompletionResponse,
    StreamChunk,
    Usage,
)

STREAM_READ_CHUNK_BYTES: Final = 256


class StreamObserver:
    def __init__(
        self,
        *,
        clock: Clock = time.perf_counter,
        start_s: float | None = None,
        prompt_tokens: int = 0,
    ) -> None:
        self._clock = clock
        self.start_s = clock() if start_s is None else start_s
        self.prompt_tokens = prompt_tokens
        self._lines = LineDecoder()
        self._sse = SseDecoder()
        self._raw = False
        self.done = False
        self._first: float | None = None
        self._last = 0.0
        self._gaps: list[float] = []
        self._content_chunks = 0
        self._usage: Usage | None = None

    # -- decoded chunks ---------------------------------------------------- #

    def observe(self, chunk: StreamChunk, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        if chunk.usage is not None:
            self._usage = chunk.usage
        if not any(choice.delta.content for choice in chunk.choices):
            return
        if self._first is None:
            self._first = now
        else:
            self._gaps.append(now - self._last)
        self._last = now
        self._content_chunks += 1

    # -- raw SSE bytes ----------------------------------------------------- #

    def feed(self, data: bytes | str) -> None:
        """Consume a slice of the response body exactly as received."""
        self._raw = True
        for line in self._lines.feed(data):
            self._feed_line(line)

    def _feed_line(self, line: str) -> None:
        payload = self._sse.feed(line)
        if payload is None or self.done:
            return
        if payload == SSE_DONE_SENTINEL:
            self.done = True
            return
        self.observe(decode_sse_event(payload))

    # -- completion -------------------------------------------------------- #

    def finish(self, now: float | None = None) -> RequestRecord:
        """Close the stream and return its record; raw streams must have ended with [DONE]."""
        if self._raw:
            self._feed_line(self._lines.flush())
            self._feed_line("")
            if not self.done:
                raise EngineProtocolError("stream ended before [DONE]")
        end = self._clock() if now is None else now
        usage = self._usage
        return RequestRecord(
            start_s=self.start_s,
            end_s=end,
            input_tokens=usage.prompt_tokens if usage else self.prompt_tokens,
            output_tokens=usage.completion_tokens if usage else self._content_chunks,
            ttft_s=None if self._first is None else self._first - self.start_s,
            itl_s=tuple(self._gaps),
        )


class HttpResponseLike(Protocol):
    """The subset of ``requests.Response`` / Locust ``FastResponse`` that is used."""

    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes | None: ...

    def iter_content(self, chunk_size: int) -> Iterable[bytes | str]: ...


def release(response: HttpResponseLike) -> None:
    """Best-effort drain of an unread streamed body so its pooled connection is reused.

    Leaving a ``stream=True`` body unread pins (and eventually leaks) the
    keep-alive connection; failures here are irrelevant to the measurement.
    """
    with contextlib.suppress(Exception):
        for _ in response.iter_content(STREAM_READ_CHUNK_BYTES):
            pass


def record_from_response(
    response: HttpResponseLike,
    *,
    stream: bool,
    start_s: float,
    prompt_tokens: int,
    clock: Clock = time.perf_counter,
) -> RequestRecord:
    """Measure one chat-completion exchange; never raises for engine-side faults."""
    status = response.status_code
    if status != 200:
        if stream:
            release(response)
        error = getattr(response, "error", None)
        if status == 0 and isinstance(error, BaseException):  # no HTTP response at all
            return RequestRecord.failure(start_s, clock(), categorize_exception(error), repr(error))
        return RequestRecord.failure(start_s, clock(), categorize_status(status), f"HTTP {status}")
    try:
        if stream:
            observer = StreamObserver(clock=clock, start_s=start_s, prompt_tokens=prompt_tokens)
            for data in response.iter_content(STREAM_READ_CHUNK_BYTES):
                observer.feed(data)
            return observer.finish()
        body = json.loads(response.content or b"")
        usage = ChatCompletionResponse.model_validate(drop_nulls(body)).usage
        return RequestRecord(
            start_s=start_s,
            end_s=clock(),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )
    except (EngineError, ValidationError, ValueError, OSError, RecursionError) as exc:
        if stream:
            release(response)
        return RequestRecord.failure(start_s, clock(), categorize_exception(exc), repr(exc))


def custom_metric_events(record: RequestRecord, name: str) -> list[dict[str, Any]]:
    """Locust ``request`` event payloads for the token-level metrics of a success.

    ``TTFT`` entries carry time-to-first-token; ``TPOT`` entries carry time per
    output token, with the output token count as ``response_length`` so Locust's
    stats also report generated tokens per request.
    """
    events: list[dict[str, Any]] = []
    if not record.success:
        return events
    for request_type, value in (("TTFT", record.ttft_s), ("TPOT", record.tpot_s)):
        if value is not None:
            events.append(
                {
                    "request_type": request_type,
                    "name": name,
                    "response_time": value * 1000.0,
                    "response_length": record.output_tokens,
                    "exception": None,
                    "context": {},
                }
            )
    return events
