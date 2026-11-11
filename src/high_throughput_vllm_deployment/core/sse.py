"""Incremental, bounded decoder for ``text/event-stream`` bodies.

Implements the subset of the WHATWG server-sent-events grammar needed for
OpenAI-compatible streaming: ``data:`` fields are accumulated (multi-line data
is joined with ``\\n``) and dispatched on a blank line; comments and the
``event``/``id``/``retry`` fields are ignored. The buffered event size is
capped so a misbehaving server cannot make the client buffer without bound.
"""

from __future__ import annotations

import codecs
from typing import Final

from high_throughput_vllm_deployment.core.errors import EngineProtocolError

DEFAULT_MAX_EVENT_BYTES: Final[int] = 1 << 20


class LineDecoder:
    """Incremental body -> lines splitter with a hard cap on the unterminated tail.

    ``httpx``'s ``aiter_lines`` (and naive ``split`` loops) buffer a line until
    its newline arrives, so a peer that never sends ``\n`` can grow client
    memory without bound. This decoder raises instead. Bytes are decoded as
    UTF-8 incrementally (multi-byte characters may straddle reads); invalid
    sequences become U+FFFD rather than failing.
    """

    def __init__(self, *, max_line_chars: int = DEFAULT_MAX_EVENT_BYTES) -> None:
        if max_line_chars < 1:
            raise ValueError("max_line_chars must be >= 1")
        self._max = max_line_chars
        self._utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""

    @property
    def pending_chars(self) -> int:
        return len(self._pending)

    def feed(self, data: bytes | str) -> list[str]:
        text = self._utf8.decode(data) if isinstance(data, bytes) else data
        *lines, self._pending = (self._pending + text).split("\n")
        if len(self._pending) > self._max:
            self._pending = ""
            raise EngineProtocolError(f"line exceeds {self._max} characters without a newline")
        return lines

    def flush(self) -> str:
        """Return (and clear) the unterminated tail at end of stream."""
        tail, self._pending = self._pending + self._utf8.decode(b"", final=True), ""
        return tail


class SseDecoder:
    def __init__(self, *, max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES) -> None:
        if max_event_bytes < 1:
            raise ValueError("max_event_bytes must be >= 1")
        self._max_event_bytes = max_event_bytes
        self._data: list[str] = []
        self._size = 0

    def feed(self, line: str) -> str | None:
        """Consume one line (without terminator); return a completed event's data, if any."""
        line = line.rstrip("\r\n")
        if not line:
            return self.flush()
        if line.startswith(":"):
            return None
        field, sep, value = line.partition(":")
        if field != "data":
            return None
        if sep and value.startswith(" "):
            value = value[1:]
        # surrogatepass: sizing must never fail on a str, whatever it contains.
        self._size += len(value.encode("utf-8", "surrogatepass")) + 1
        if self._size > self._max_event_bytes:
            raise EngineProtocolError(
                f"SSE event exceeds {self._max_event_bytes} bytes without a terminator"
            )
        self._data.append(value)
        return None

    def flush(self) -> str | None:
        """Dispatch the buffered event (end of event or end of stream)."""
        if not self._data:
            return None
        data = "\n".join(self._data)
        self._data.clear()
        self._size = 0
        return data
