"""Closed vocabularies: exact members, string serialisation, lax lookup in strict models."""

from __future__ import annotations

from enum import StrEnum

import pytest
from pydantic import ValidationError

from high_throughput_vllm_deployment.schemas._base import ContractModel
from high_throughput_vllm_deployment.schemas.enums import (
    FinishReason,
    LaxFinishReason,
    ModelBackend,
    QuantizationMethod,
    SchedulerState,
)


@pytest.mark.parametrize(
    ("enum", "values"),
    [
        (SchedulerState, {"WAITING": "waiting", "RUNNING": "running", "SWAPPED": "swapped"}),
        (FinishReason, {"STOP": "stop", "LENGTH": "length", "ABORT": "abort"}),
        (
            QuantizationMethod,
            {"AWQ": "awq", "GPTQ": "gptq", "SQUEEZELLM": "squeezellm", "FP8": "fp8"},
        ),
        (
            ModelBackend,
            {
                "VLLM": "vllm",
                "OPENAI": "openai",
                "OPENAI_CHAT": "openai-chat",
                "TGI": "tgi",
                "SGLANG": "sglang",
                "TENSORRT_LLM": "tensorrt-llm",
            },
        ),
    ],
)
def test_enum_members_are_exact(enum: type[StrEnum], values: dict[str, str]) -> None:
    assert {member.name: member.value for member in enum} == values
    assert all(str(member) == member.value for member in enum)


def test_unquantized_is_not_an_enum_member() -> None:
    assert "none" not in {member.value for member in QuantizationMethod}


class _Holder(ContractModel):
    reason: LaxFinishReason


def test_lax_enum_accepts_values_and_members_inside_strict_models() -> None:
    assert _Holder.model_validate({"reason": "length"}).reason is FinishReason.LENGTH
    assert _Holder(reason=FinishReason.STOP).reason is FinishReason.STOP
    assert _Holder(reason=FinishReason.ABORT).model_dump(mode="json") == {"reason": "abort"}


@pytest.mark.parametrize("bad", ["STOP", "eos", 1, None])
def test_lax_enum_still_rejects_unknown_values(bad: object) -> None:
    with pytest.raises(ValidationError):
        _Holder.model_validate({"reason": bad})
