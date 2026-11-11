"""SSE decoder: event framing, ignored fields, bounded buffering."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from high_throughput_vllm_deployment.core.errors import EngineProtocolError
from high_throughput_vllm_deployment.core.sse import SseDecoder


def _decode(lines: list[str], *, max_event_bytes: int = 1 << 20) -> list[str]:
    decoder = SseDecoder(max_event_bytes=max_event_bytes)
    events = [data for line in lines if (data := decoder.feed(line)) is not None]
    if (tail := decoder.flush()) is not None:
        events.append(tail)
    return events


def test_events_dispatch_on_blank_lines() -> None:
    assert _decode(["data: one", "", "data: two", ""]) == ["one", "two"]


def test_multi_line_data_is_joined_with_newlines() -> None:
    assert _decode(["data: a", "data:b", "data", ""]) == ["a\nb\n"]


def test_comments_and_other_fields_are_ignored() -> None:
    lines = [": keep-alive", "event: message", "id: 7", "retry: 100", "data: x", "", ""]
    assert _decode(lines) == ["x"]


def test_only_one_leading_space_is_stripped_and_crlf_is_tolerated() -> None:
    assert _decode(["data:  padded\r\n", "\r\n"]) == [" padded"]


def test_unterminated_trailing_event_is_returned_by_flush() -> None:
    assert _decode(["data: tail"]) == ["tail"]


def test_flush_without_buffered_data_returns_none() -> None:
    assert SseDecoder().flush() is None


def test_oversized_events_are_rejected() -> None:
    decoder = SseDecoder(max_event_bytes=8)
    assert decoder.feed("data: 1234") is None
    with pytest.raises(EngineProtocolError, match="exceeds 8 bytes"):
        decoder.feed("data: 5678")


def test_decoder_resets_size_after_dispatch() -> None:
    decoder = SseDecoder(max_event_bytes=8)
    for _ in range(5):
        assert decoder.feed("data: 123456") is None
        assert decoder.feed("") == "123456"


def test_invalid_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        SseDecoder(max_event_bytes=0)


@given(st.lists(st.text(alphabet=st.characters(blacklist_characters="\r\n")), max_size=20))
def test_any_payload_sequence_round_trips(payloads: list[str]) -> None:
    lines = [line for payload in payloads for line in (f"data: {payload}", "")]
    assert _decode(lines) == list(payloads)


def test_line_decoder_limit_must_be_positive() -> None:
    from high_throughput_vllm_deployment.core.sse import LineDecoder

    with pytest.raises(ValueError, match=">= 1"):
        LineDecoder(max_line_chars=0)
