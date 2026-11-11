"""Hexagonal port every inference engine adapter implements."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry


@runtime_checkable
class InferenceEnginePort(Protocol):
    """Async interface to an OpenAI-compatible, continuously batching engine.

    Implementations raise only :class:`~.errors.EngineError` subclasses.
    """

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run a non-streaming text completion."""

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run a non-streaming chat completion."""

    def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion chunk by chunk, ending after the final chunk.

        Closing the iterator early aborts the request.
        """

    async def get_telemetry(self) -> ClusterTelemetry:
        """Return a point-in-time scheduler and KV cache snapshot."""

    async def is_healthy(self) -> bool:
        """Return whether the engine is ready to serve; never raises."""
