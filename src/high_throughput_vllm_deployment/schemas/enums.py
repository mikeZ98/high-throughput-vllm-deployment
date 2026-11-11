"""Closed vocabularies shared across configuration, inference and telemetry contracts.

Every enum is a :class:`~enum.StrEnum` so members serialise to their plain
string value on the wire.
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Annotated

from pydantic import Strict


@unique
class ModelBackend(StrEnum):
    """Serving backend under test (mirrors vLLM ``benchmark_serving`` backends)."""

    VLLM = "vllm"
    OPENAI = "openai"
    OPENAI_CHAT = "openai-chat"
    TGI = "tgi"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt-llm"


@unique
class SchedulerState(StrEnum):
    """Scheduler queue a request is in."""

    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"


@unique
class FinishReason(StrEnum):
    """Why generation stopped for a choice."""

    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@unique
class QuantizationMethod(StrEnum):
    """Weight quantization scheme.

    Unquantized models are represented by ``None`` at the field level, not by
    an enum member, so ``quantization: QuantizationMethod | None``.
    """

    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
    FP8 = "fp8"


# Contract models are strict, under which enums only accept member instances.
# These aliases re-enable value lookup so JSON-decoded dicts validate.
LaxModelBackend = Annotated[ModelBackend, Strict(False)]
LaxSchedulerState = Annotated[SchedulerState, Strict(False)]
LaxFinishReason = Annotated[FinishReason, Strict(False)]
LaxQuantizationMethod = Annotated[QuantizationMethod, Strict(False)]
