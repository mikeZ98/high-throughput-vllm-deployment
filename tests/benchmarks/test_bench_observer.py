"""StreamObserver and record_from_response: token timing, chunk boundaries, failure mapping."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from high_throughput_vllm_deployment.benchmarks.metrics_collector import RequestRecord
from high_throughput_vllm_deployment.benchmarks.observer import (
    StreamObserver,
    custom_metric_events,
    record_from_response,
)
from high_throughput_vllm_deployment.core.errors import EngineHTTPError, EngineProtocolError
from high_throughput_vllm_deployment.schemas.enums import ErrorCategory
from high_throughput_vllm_deployment.schemas.inference import (
    DeltaMessage,
    StreamChoice,
    StreamChunk,
    Usage,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _event(
    content: str | None = None, *, role: str | None = None, usage: dict[str, int] | None = None
) -> str:
    choices = (
        []
        if usage
        else [{"index": 0, "delta": {"content": content, "role": role}, "finish_reason": None}]
    )
    payload: dict[str, Any] = {"id": "c", "created": 0, "model": "m", "choices": choices}
    if usage:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
BODY = (
    _event("", role="assistant")
    + _event("Hel")
    + ": ping\n\n"
    + _event("lo")
    + _event("!")
    + _event(usage=USAGE)
    + "data: [DONE]\n\n"
)


def _chunk(content: str | None) -> StreamChunk:
    return StreamChunk(
        id="c",
        created=0,
        model="m",
        choices=[StreamChoice(index=0, delta=DeltaMessage(content=content))],
    )


# --------------------------------------------------------------------------- #
# StreamObserver                                                              #
# --------------------------------------------------------------------------- #


def test_raw_stream_timing_and_usage() -> None:
    clock = Clock()
    observer = StreamObserver(clock=clock, start_s=0.0, prompt_tokens=99)
    for offset, part in zip((0.1, 0.3, 0.6, 0.7, 0.8), BODY.split("\n\n")[:5], strict=True):
        clock.now = offset
        observer.feed(part + "\n\n")
    clock.now = 0.9
    observer.feed(BODY.split("\n\n")[5] + "\n\n")  # usage-only chunk
    clock.now = 1.0
    observer.feed("data: [DONE]\n\n")
    record = observer.finish()
    # role-only chunk (t=0.1) carries no content; first token at 0.3, then 0.6 (ping skipped)...
    assert record.ttft_s == pytest.approx(0.3)
    assert record.itl_s == pytest.approx((0.4, 0.1))
    assert (record.input_tokens, record.output_tokens) == (7, 3)  # server usage wins
    assert record.end_s == 1.0 and record.success


@given(st.lists(st.integers(min_value=0, max_value=len(BODY.encode())), max_size=12))
def test_arbitrary_chunk_boundaries_give_the_same_record(cuts: list[int]) -> None:
    data = BODY.replace("Hel", "Hé✓").encode()  # multi-byte characters split across reads
    points = sorted({0, len(data), *cuts})
    observer = StreamObserver(clock=Clock(), start_s=0.0)
    for start, end in zip(points, points[1:], strict=False):
        observer.feed(data[start:end])
    record = observer.finish()
    assert record.output_tokens == 3 and record.ttft_s == 0.0


def test_without_usage_tokens_are_counted_from_content_chunks() -> None:
    observer = StreamObserver(clock=Clock(), start_s=0.0, prompt_tokens=5)
    for content in ("", "a", None, "b"):
        observer.observe(_chunk(content))
    record = observer.finish()
    assert (record.input_tokens, record.output_tokens) == (5, 2)


def test_usage_chunk_via_observe() -> None:
    observer = StreamObserver(clock=Clock(), start_s=0.0)
    observer.observe(StreamChunk(id="c", created=0, model="m", choices=[], usage=Usage(**USAGE)))
    assert observer.finish().output_tokens == 3


def test_start_defaults_to_clock() -> None:
    clock = Clock()
    clock.now = 5.0
    assert StreamObserver(clock=clock).start_s == 5.0


def test_done_without_trailing_blank_line_and_events_after_done() -> None:
    observer = StreamObserver(clock=Clock(), start_s=0.0)
    observer.feed(_event("x") + "data: [DONE]\n\n" + _event("ignored"))
    observer.feed("data: [DONE]")
    assert observer.finish().output_tokens == 1
    tail = StreamObserver(clock=Clock(), start_s=0.0)
    tail.feed(_event("x") + "data: [DONE]")
    assert tail.finish().output_tokens == 1


def test_stream_without_done_is_a_protocol_error() -> None:
    observer = StreamObserver(clock=Clock(), start_s=0.0)
    observer.feed(_event("x"))
    with pytest.raises(EngineProtocolError, match=r"\[DONE\]"):
        observer.finish()


def test_error_event_raises_http_error() -> None:
    observer = StreamObserver(clock=Clock(), start_s=0.0)
    with pytest.raises(EngineHTTPError, match="OOM"):
        observer.feed('data: {"error": {"message": "OOM"}}\n\n')


def test_chunk_mode_does_not_require_done() -> None:
    observer = StreamObserver(clock=Clock(), start_s=0.0)
    observer.observe(_chunk("a"))
    assert observer.finish(now=2.0).end_s == 2.0


# --------------------------------------------------------------------------- #
# record_from_response                                                        #
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        *,
        content: bytes | None = None,
        parts: Iterable[bytes | str] = (),
        error: BaseException | None = None,
        fail_after: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self._parts = list(parts)
        self.fail_after = fail_after
        self.reads = 0
        if error is not None:
            self.error = error

    def iter_content(self, chunk_size: int) -> Iterator[bytes | str]:
        assert chunk_size > 0
        self.reads += 1
        parts, self._parts = self._parts, []  # a body can be consumed only once
        yield from parts
        if self.fail_after is not None:
            raise self.fail_after


CHAT_JSON = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "logprobs": None}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}


def _record(response: FakeResponse, *, stream: bool) -> RequestRecord:
    return record_from_response(
        response, stream=stream, start_s=0.0, prompt_tokens=4, clock=Clock()
    )


def test_successful_stream_and_json_responses() -> None:
    streamed = _record(FakeResponse(parts=[BODY.encode()[:50], BODY.encode()[50:]]), stream=True)
    assert streamed.success and streamed.output_tokens == 3
    blocking = _record(FakeResponse(content=json.dumps(CHAT_JSON).encode()), stream=False)
    assert blocking.success and (blocking.input_tokens, blocking.output_tokens) == (4, 2)
    assert blocking.ttft_s is None


class ConnectTimeout(OSError):  # requests.exceptions.ConnectTimeout lookalike
    pass


@pytest.mark.parametrize(
    ("response", "stream", "category"),
    [
        (FakeResponse(503), True, ErrorCategory.UNAVAILABLE),
        (FakeResponse(400, error=RuntimeError("bad status")), False, ErrorCategory.CLIENT_ERROR),
        (FakeResponse(0, error=ConnectTimeout()), False, ErrorCategory.TIMEOUT),
        (FakeResponse(0, error=ConnectionRefusedError()), True, ErrorCategory.UNAVAILABLE),
        (FakeResponse(0), True, ErrorCategory.UNAVAILABLE),
        (FakeResponse(content=b"<html>"), False, ErrorCategory.PROTOCOL),
        (FakeResponse(content=None), False, ErrorCategory.PROTOCOL),
        (FakeResponse(content=b'{"id": "x"}'), False, ErrorCategory.PROTOCOL),
        (FakeResponse(parts=[_event("x")]), True, ErrorCategory.PROTOCOL),
        (FakeResponse(parts=["data: {broken\n\n"]), True, ErrorCategory.PROTOCOL),
        (
            FakeResponse(parts=['data: {"error": {"message": "x"}}\n\n']),
            True,
            ErrorCategory.SERVER_ERROR,
        ),
        (
            FakeResponse(parts=[_event("x")], fail_after=ConnectionResetError()),
            True,
            ErrorCategory.UNAVAILABLE,
        ),
    ],
)
def test_failures_are_categorised(
    response: FakeResponse, stream: bool, category: ErrorCategory
) -> None:
    record = _record(response, stream=stream)
    assert not record.success
    assert record.error is category
    assert record.detail


def test_custom_metric_events() -> None:
    streamed = RequestRecord(0.0, 1.0, output_tokens=5, ttft_s=0.2)
    events = custom_metric_events(streamed, "chat")
    assert [(e["request_type"], e["response_time"]) for e in events] == [
        ("TTFT", pytest.approx(200.0)),
        ("TPOT", pytest.approx(200.0)),
    ]
    assert all(e["response_length"] == 5 and e["exception"] is None for e in events)
    assert custom_metric_events(RequestRecord(0.0, 1.0, output_tokens=5), "chat") == []
    failed = RequestRecord.failure(0.0, 1.0, ErrorCategory.TIMEOUT)
    assert custom_metric_events(failed, "chat") == []


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(503, parts=[b'{"object": "error"}']),
        FakeResponse(parts=["data: {broken\n\n", _event("unread")]),
        FakeResponse(parts=[_event("x")], fail_after=ConnectionResetError()),
    ],
)
def test_failed_streams_are_drained_to_release_the_connection(response: FakeResponse) -> None:
    record = _record(response, stream=True)
    assert not record.success
    assert response._parts == []  # nothing left unread
    assert response.reads >= 1


def test_blocking_failures_do_not_touch_the_stream() -> None:
    response = FakeResponse(503, parts=[b"x"])
    _record(response, stream=False)
    assert response.reads == 0
