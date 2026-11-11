"""Shared contract base: frozen, closed, strict, finite; path-escape rejection."""

from __future__ import annotations

import inspect
import math
import os
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from high_throughput_vllm_deployment import schemas
from high_throughput_vllm_deployment.schemas._base import (
    ContractModel,
    reject_host_path_escape,
    require_non_blank,
)

CONTRACT_MODELS = sorted(
    (
        obj
        for obj in vars(schemas).values()
        if inspect.isclass(obj) and issubclass(obj, ContractModel) and obj is not ContractModel
    ),
    key=lambda cls: cls.__name__,
)


class _Probe(ContractModel):
    count: int
    ratio: float
    label: str


def test_every_exported_contract_inherits_the_shared_config() -> None:
    assert len(CONTRACT_MODELS) >= 15
    for model in CONTRACT_MODELS:
        config = model.model_config
        assert config.get("frozen") is True, model
        assert config.get("extra") == "forbid", model
        assert config.get("strict") is True, model
        assert config.get("allow_inf_nan") is False, model


def test_instances_are_frozen_and_hashable() -> None:
    probe = _Probe(count=1, ratio=0.5, label="x")
    with pytest.raises(ValidationError, match="frozen"):
        probe.count = 2
    assert hash(probe) == hash(_Probe(count=1, ratio=0.5, label="x"))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        _Probe.model_validate({"count": 1, "ratio": 0.5, "label": "x", "surprise": True})


@pytest.mark.parametrize(
    "payload",
    [
        {"count": "1", "ratio": 0.5, "label": "x"},
        {"count": True, "ratio": 0.5, "label": "x"},
        {"count": 1.0, "ratio": 0.5, "label": "x"},
        {"count": 1, "ratio": "0.5", "label": "x"},
        {"count": 1, "ratio": False, "label": "x"},
        {"count": 1, "ratio": 0.5, "label": 7},
    ],
)
def test_scalars_are_never_coerced(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _Probe.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        _Probe(count=1, ratio=value, label="x")


@pytest.mark.parametrize(
    "raw", ['{"count":1,"ratio":NaN,"label":"x"}', '{"count":1,"ratio":Infinity,"label":"x"}']
)
def test_non_finite_floats_are_rejected_from_json(raw: str) -> None:
    with pytest.raises(ValidationError):
        _Probe.model_validate_json(raw)


def test_int_is_accepted_for_float_fields() -> None:
    assert _Probe(count=1, ratio=1, label="x").ratio == 1.0


# --------------------------------------------------------------------------- #
# require_non_blank                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_blank_strings_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        require_non_blank(value)


def test_non_blank_strings_are_preserved_verbatim() -> None:
    assert require_non_blank("  padded  ") == "  padded  "


# --------------------------------------------------------------------------- #
# reject_host_path_escape                                                     #
# --------------------------------------------------------------------------- #

HOME_RELATIVE = f"~{os.sep}models"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("models/\x00evil", "NUL"),
        (HOME_RELATIVE, "home-relative"),
        ("~root/models", "home-relative"),
        ("/opt/models/llama", "absolute"),
        ("C:\\models\\llama", "absolute"),
        ("C:models", "absolute"),
        ("\\\\server\\share\\llama", "absolute"),
        ("\\models", "absolute"),
        ("../models", "traversal"),
        ("models/../../escape", "traversal"),
        ("models\\..\\..\\escape", "traversal"),
    ],
)
def test_host_path_escapes_are_rejected(value: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        reject_host_path_escape(value)


@pytest.mark.parametrize(
    "value",
    ["mistralai/Mistral-7B-Instruct-v0.2", "./models/llama", "models/llama", "facebook/opt-125m"],
)
def test_relative_identifiers_are_accepted(value: str) -> None:
    assert reject_host_path_escape(value) == value


@given(st.lists(st.from_regex(r"\A[A-Za-z0-9_.-]{1,12}\Z"), min_size=1, max_size=5))
def test_absolute_variants_of_any_path_are_rejected(segments: list[str]) -> None:
    relative = "/".join(segments)
    with pytest.raises(ValueError):
        reject_host_path_escape("/" + relative)
    with pytest.raises(ValueError):
        reject_host_path_escape("D:\\" + relative.replace("/", "\\"))
    with pytest.raises(ValueError):
        reject_host_path_escape(relative + "/../../..")
