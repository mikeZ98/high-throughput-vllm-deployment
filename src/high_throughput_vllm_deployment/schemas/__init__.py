"""Pydantic v2 data contracts: engine config, OpenAI-compatible inference, telemetry."""

from high_throughput_vllm_deployment.schemas._base import ContractModel
from high_throughput_vllm_deployment.schemas.config import ConfigPathError, VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import (
    FinishReason,
    ModelBackend,
    QuantizationMethod,
    SchedulerState,
)
from high_throughput_vllm_deployment.schemas.inference import (
    SSE_DONE_EVENT,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    SamplingParams,
    StreamChoice,
    StreamChunk,
    Usage,
)
from high_throughput_vllm_deployment.schemas.telemetry import (
    BenchmarkResult,
    ClusterTelemetry,
    KvCacheBlocks,
    LatencyPercentiles,
)

__all__ = [
    "SSE_DONE_EVENT",
    "BenchmarkResult",
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "ClusterTelemetry",
    "CompletionChoice",
    "CompletionRequest",
    "CompletionResponse",
    "ConfigPathError",
    "ContractModel",
    "DeltaMessage",
    "FinishReason",
    "KvCacheBlocks",
    "LatencyPercentiles",
    "ModelBackend",
    "QuantizationMethod",
    "SamplingParams",
    "SchedulerState",
    "StreamChoice",
    "StreamChunk",
    "Usage",
    "VllmDeploymentConfig",
]
