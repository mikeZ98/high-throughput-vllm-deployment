"""OpenAI-compatible HTTP front-end for :class:`MockVllmEngine` (loopback load-test target).

A dependency-free asyncio HTTP/1.1 server exposing the vLLM API surface that
the benchmarks use, so Locust (or ``VllmClient``) can be pointed at a fully
simulated engine:

* ``POST /v1/chat/completions`` - JSON, or SSE (``Transfer-Encoding: chunked``)
  when ``stream=true``; mid-stream failures are sent as vLLM does, as a
  ``data: {"error": ...}`` event followed by ``[DONE]``;
* ``POST /v1/completions`` - JSON only;
* ``GET /metrics`` - Prometheus text with vLLM's metric names, parseable by
  :func:`~high_throughput_vllm_deployment.core.metrics.telemetry_from_prometheus`;
* ``GET /health``.

``fault_every=N`` answers every N-th API request with HTTP 503, deterministically,
to exercise retry and error-categorisation paths. Connections are keep-alive.

Run standalone::

    uv run python -m high_throughput_vllm_deployment.benchmarks.mock_server --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, Final, Self, cast

from pydantic import BaseModel, ValidationError

from high_throughput_vllm_deployment.core._clock import Sleep
from high_throughput_vllm_deployment.core.errors import EngineError, EngineHTTPError
from high_throughput_vllm_deployment.core.mock import MockEngineConfig, MockVllmEngine
from high_throughput_vllm_deployment.schemas.inference import (
    SSE_DONE_EVENT,
    ChatCompletionRequest,
    CompletionRequest,
)
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry

CHAT_PATH: Final = "/v1/chat/completions"
COMPLETIONS_PATH: Final = "/v1/completions"
_ROUTES: Final = {
    CHAT_PATH: "POST",
    COMPLETIONS_PATH: "POST",
    "/metrics": "GET",
    "/health": "GET",
}
_REASONS: Final = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    414: "URI Too Long",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}
_MAX_HEADER_LINES: Final = 100
_LINGER_BYTES: Final = 1 << 20
_LINGER_SECONDS: Final = 1.0


async def _readline(reader: asyncio.StreamReader, status: int) -> bytes:
    """``readline`` that maps asyncio's line-length limit (64 KiB) to an HTTP error.

    Without this an oversized request/header line raises ``ValueError`` out of
    the connection task instead of producing a response.
    """
    try:
        return await reader.readline()
    except ValueError:  # LimitOverrunError surfaces as ValueError from readline()
        raise _HttpError(status, "request line or header field too large") from None


async def _linger(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Lingering close after rejecting a request we did not read in full.

    Closing a socket with unread input makes the kernel send RST, which can
    destroy the error response before the client reads it. Half-close our side
    and discard (bounded) input first, as production servers do.
    """
    if writer.can_write_eof():
        writer.write_eof()

    async def discard() -> None:
        received = 0
        while received < _LINGER_BYTES and (chunk := await reader.read(65536)):
            received += len(chunk)

    with contextlib.suppress(TimeoutError, ConnectionError):
        await asyncio.wait_for(discard(), _LINGER_SECONDS)


class _HttpError(Exception):
    def __init__(self, status: int, message: str, *, close: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.close = close


def render_vllm_metrics(telemetry: ClusterTelemetry, model_name: str) -> str:
    """Render telemetry using vLLM's (V0) Prometheus metric names and label layout."""
    label = json.dumps(model_name)
    kv = telemetry.kv_cache
    gauges = (
        ("vllm:num_requests_running", telemetry.active_requests),
        ("vllm:num_requests_waiting", telemetry.waiting_requests),
        ("vllm:num_requests_swapped", telemetry.swapped_requests),
        ("vllm:gpu_cache_usage_perc", telemetry.gpu_cache_usage_pct / 100.0),
        ("vllm:cpu_cache_usage_perc", telemetry.cpu_cache_usage_pct / 100.0),
    )
    lines: list[str] = []
    for name, value in gauges:
        lines += [f"# TYPE {name} gauge", f"{name}{{model_name={label}}} {float(value)!r}"]
    lines += [
        "# TYPE vllm:cache_config_info gauge",
        f'vllm:cache_config_info{{block_size="{kv.block_size}",'
        f'num_cpu_blocks="{kv.total_cpu_blocks}",num_gpu_blocks="{kv.total_gpu_blocks}"}} 1.0',
    ]
    return "\n".join(lines) + "\n"


def _error_body(status: int, message: str) -> dict[str, Any]:
    kind = "BadRequestError" if status < 500 else "ServiceUnavailableError"
    return {"object": "error", "message": message, "type": kind, "param": None, "code": status}


class MockVllmServer:
    """Serves one :class:`MockVllmEngine` over loopback HTTP."""

    def __init__(
        self,
        engine: MockVllmEngine,
        *,
        fault_every: int = 0,
        max_body_bytes: int = 1 << 20,
    ) -> None:
        if fault_every < 0:
            raise ValueError("fault_every must be >= 0")
        self.engine = engine
        self.fault_every = fault_every
        self.max_body_bytes = max_body_bytes
        self.api_requests = 0
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self.base_url = ""

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        self._server = await asyncio.start_server(self._serve_connection, host, port)
        bound_host, bound_port = self._server.sockets[0].getsockname()[:2]
        self.base_url = f"http://{bound_host}:{bound_port}"
        return self.base_url

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            for task in list(self._connections):
                task.cancel()
            await asyncio.gather(*self._connections, return_exceptions=True)
            await self._server.wait_closed()
            self._server = None
        await self.engine.aclose()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # HTTP/1.1                                                           #
    # ------------------------------------------------------------------ #

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Always set inside a connection callback, which runs as its own task.
        task = cast("asyncio.Task[None]", asyncio.current_task())
        self._connections.add(task)
        try:
            while await self._serve_one(reader, writer):
                pass
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # client went away
        finally:
            self._connections.discard(task)
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def _serve_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        """Serve one request; returns whether the connection stays open."""
        try:
            request_line = await _readline(reader, 414)
            if not request_line:
                return False
            method, path, keep_alive, body = await self._read_request(request_line, reader)
            await self._dispatch(method, path, body, writer, keep_alive)
        except _HttpError as exc:
            await self._write_json(writer, exc.status, _error_body(exc.status, str(exc)), False)
            await _linger(reader, writer)
            return False
        return keep_alive

    async def _read_request(
        self, request_line: bytes, reader: asyncio.StreamReader
    ) -> tuple[str, str, bool, bytes]:
        parts = request_line.decode("latin-1").split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            raise _HttpError(400, "malformed request line")
        method, target, version = parts
        headers: dict[str, str] = {}
        for _ in range(_MAX_HEADER_LINES):
            line = (await _readline(reader, 431)).decode("latin-1").strip()
            if not line:
                break
            name, sep, value = line.partition(":")
            if not sep:
                raise _HttpError(400, f"malformed header line {line!r}")
            headers[name.strip().lower()] = value.strip()
        else:
            raise _HttpError(400, "too many header lines")
        connection = headers.get("connection", "").lower()
        keep_alive = connection != "close" and (version == "HTTP/1.1" or connection == "keep-alive")
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            raise _HttpError(400, "invalid Content-Length") from None
        if length < 0 or length > self.max_body_bytes:
            raise _HttpError(413, f"body exceeds {self.max_body_bytes} bytes")
        body = await reader.readexactly(length) if length else b""
        return method, target.split("?", 1)[0], keep_alive, body

    async def _dispatch(
        self, method: str, path: str, body: bytes, writer: asyncio.StreamWriter, keep_alive: bool
    ) -> None:
        expected = _ROUTES.get(path)
        if expected is None:
            await self._write_json(writer, 404, _error_body(404, f"no route {path}"), keep_alive)
            return
        if method != expected:
            error = _error_body(405, f"{method} not allowed on {path}")
            await self._write_json(writer, 405, error, keep_alive)
            return
        if path == "/health":
            await self._write(writer, 200, b"", "text/plain", keep_alive)
            return
        if path == "/metrics":
            text = render_vllm_metrics(self.engine.snapshot(), self.engine.config.model_name)
            await self._write(writer, 200, text.encode(), "text/plain; version=0.0.4", keep_alive)
            return

        self.api_requests += 1
        if self.fault_every and self.api_requests % self.fault_every == 0:
            error = _error_body(503, "injected fault: engine overloaded")
            await self._write_json(writer, 503, error, keep_alive)
            return
        try:
            if path == COMPLETIONS_PATH:
                completion = self._parse(CompletionRequest, body)
                if completion.stream:
                    raise _HttpError(400, "the mock server does not stream /v1/completions")
                response: BaseModel = await self.engine.complete(completion)
            else:
                chat = self._parse(ChatCompletionRequest, body)
                if chat.stream:
                    await self._stream(writer, chat, keep_alive)
                    return
                response = await self.engine.chat_complete(chat)
        except _HttpError as exc:
            error = _error_body(exc.status, str(exc))
            await self._write_json(writer, exc.status, error, keep_alive)
            return
        except EngineError as exc:
            status = exc.status_code if isinstance(exc, EngineHTTPError) else 503
            await self._write_json(writer, status, _error_body(status, str(exc)), keep_alive)
            return
        await self._write(
            writer, 200, response.model_dump_json().encode(), "application/json", keep_alive
        )

    @staticmethod
    def _parse[ModelT: BaseModel](model: type[ModelT], body: bytes) -> ModelT:
        try:
            return model.model_validate_json(body)
        except ValidationError as exc:
            raise _HttpError(
                400, f"invalid request: {exc.error_count()} validation error(s)"
            ) from exc

    async def _stream(
        self, writer: asyncio.StreamWriter, request: ChatCompletionRequest, keep_alive: bool
    ) -> None:
        stream: AsyncIterator[Any] = self.engine.stream_chat(request)
        try:
            first = await anext(stream)
        except EngineError as exc:  # rejected before any byte was sent: plain HTTP error
            status = exc.status_code if isinstance(exc, EngineHTTPError) else 503
            await self._write_json(writer, status, _error_body(status, str(exc)), keep_alive)
            return
        connection = "keep-alive" if keep_alive else "close"
        head = (
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
            f"Transfer-Encoding: chunked\r\nConnection: {connection}\r\n\r\n"
        )
        writer.write(head.encode())
        try:
            await self._write_chunk(writer, first.to_sse())
            try:
                async for chunk in stream:
                    await self._write_chunk(writer, chunk.to_sse())
            except EngineError as exc:
                error = {"error": {"message": str(exc), "type": type(exc).__name__, "code": 503}}
                await self._write_chunk(writer, f"data: {json.dumps(error)}\n\n")
            await self._write_chunk(writer, SSE_DONE_EVENT)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            await stream.aclose()  # type: ignore[attr-defined]

    @staticmethod
    async def _write_chunk(writer: asyncio.StreamWriter, text: str) -> None:
        data = text.encode()
        writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        await writer.drain()

    async def _write_json(
        self, writer: asyncio.StreamWriter, status: int, payload: dict[str, Any], keep_alive: bool
    ) -> None:
        await self._write(
            writer, status, json.dumps(payload).encode(), "application/json", keep_alive
        )

    @staticmethod
    async def _write(
        writer: asyncio.StreamWriter, status: int, body: bytes, content_type: str, keep_alive: bool
    ) -> None:
        connection = "keep-alive" if keep_alive else "close"
        head = (
            f"HTTP/1.1 {status} {_REASONS[status]}\r\nContent-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: {connection}\r\n\r\n"
        )
        writer.write(head.encode() + body)
        await writer.drain()


class ThreadedMockServer:
    """Runs a :class:`MockVllmServer` on its own event loop in a daemon thread.

    For synchronous load generators (Locust, ``requests``) that must not share
    the server's event loop::

        with ThreadedMockServer(MockEngineConfig()) as base_url:
            ...
    """

    def __init__(
        self,
        config: MockEngineConfig | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        fault_every: int = 0,
        sleep: Sleep = asyncio.sleep,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self._config = config or MockEngineConfig()
        self._host = host
        self._port = port
        self._fault_every = fault_every
        self._sleep = sleep
        self._timeout = startup_timeout_s
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # surfaced to the caller by __enter__
            self._error = exc
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = MockVllmServer(
            MockVllmEngine(self._config, sleep=self._sleep), fault_every=self._fault_every
        )
        try:
            self.base_url = await server.start(self._host, self._port)
            self._ready.set()
            await self._stop.wait()
        finally:
            await server.aclose()

    def __enter__(self) -> str:
        self._thread = threading.Thread(target=self._run, name="mock-vllm-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._timeout) or self._error is not None:
            raise RuntimeError("mock vLLM server did not start") from self._error
        return self.base_url

    def __exit__(self, *_exc_info: object) -> None:
        loop, stop, thread = self._loop, self._stop, self._thread
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
        if thread is not None:
            thread.join(self._timeout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve MockVllmEngine over OpenAI-compatible HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=MockEngineConfig().model_name)
    parser.add_argument("--decode-step-s", type=float, default=0.02)
    parser.add_argument("--prefill-s-per-token", type=float, default=1e-4)
    parser.add_argument("--num-gpu-blocks", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--fault-every", type=int, default=0)
    parser.add_argument("--run-seconds", type=float, default=0.0, help="0 = serve until killed")
    return parser


async def serve(args: argparse.Namespace) -> None:
    config = MockEngineConfig(
        model_name=args.model,
        decode_step_s=args.decode_step_s,
        prefill_s_per_token=args.prefill_s_per_token,
        num_gpu_blocks=args.num_gpu_blocks,
        max_num_seqs=args.max_num_seqs,
        ignore_eos=args.ignore_eos,
    )
    server = MockVllmServer(MockVllmEngine(config), fault_every=args.fault_every)
    await server.start(args.host, args.port)
    # As a container's PID 1, Python ignores SIGTERM unless a handler is
    # installed; without this `docker stop` would wait and then SIGKILL.
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    signals = (signal.SIGTERM, signal.SIGINT)
    for sig in signals:
        loop.add_signal_handler(sig, stop.set)
    if args.run_seconds > 0:
        loop.call_later(args.run_seconds, stop.set)
    try:
        sys.stdout.write(f"mock vLLM server listening on {server.base_url}\n")
        sys.stdout.flush()
        await stop.wait()
    finally:
        for sig in signals:
            loop.remove_signal_handler(sig)
        await server.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    asyncio.run(serve(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
