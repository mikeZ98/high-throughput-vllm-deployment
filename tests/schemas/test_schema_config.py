"""VllmDeploymentConfig: defaults, bounds, strict env parsing, .env path containment."""

from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from high_throughput_vllm_deployment.schemas.config import ConfigPathError, VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import QuantizationMethod

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Run each test in an empty project dir with no inherited VLLM_* variables."""
    for name in list(os.environ):
        if name.upper().startswith("VLLM_"):
            monkeypatch.delenv(name)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    yield project


def _write_env(path: Path, **values: str) -> Path:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return path


def _validate(**overrides: Any) -> VllmDeploymentConfig:
    """Validate field values directly, bypassing env/.env sources."""
    return VllmDeploymentConfig.model_validate({"model": MODEL_ID, **overrides})


# --------------------------------------------------------------------------- #
# Defaults and derived values                                                 #
# --------------------------------------------------------------------------- #


def test_defaults() -> None:
    config = VllmDeploymentConfig(model=MODEL_ID)
    assert config.model_dump() == {
        "model": MODEL_ID,
        "host": "0.0.0.0",  # noqa: S104 - asserting the documented default
        "port": 8000,
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 8192,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "quantization": None,
    }
    assert config.world_size == 1


def test_world_size_is_tensor_times_pipeline() -> None:
    assert _validate(tensor_parallel_size=4, pipeline_parallel_size=2).world_size == 8


def test_model_is_required() -> None:
    with pytest.raises(ValidationError, match="model"):
        VllmDeploymentConfig()


def test_config_is_frozen() -> None:
    config = _validate()
    with pytest.raises(ValidationError, match="frozen"):
        config.port = 9000  # type: ignore[misc]


def test_unknown_init_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        _validate(max_num_seqs=256)


# --------------------------------------------------------------------------- #
# Field constraints                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", 0),
        ("port", 65536),
        ("gpu_memory_utilization", 0.0),
        ("gpu_memory_utilization", 1.0000001),
        ("gpu_memory_utilization", float("nan")),
        ("gpu_memory_utilization", float("inf")),
        ("max_num_batched_tokens", 0),
        ("max_model_len", 0),
        ("tensor_parallel_size", 0),
        ("pipeline_parallel_size", -1),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _validate(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("port", 1), ("port", 65535), ("gpu_memory_utilization", 1.0), ("gpu_memory_utilization", 1)],
)
def test_boundary_values_are_accepted(field: str, value: object) -> None:
    assert getattr(_validate(**{field: value}), field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", True),
        ("port", 8000.0),
        ("port", "8_000"),
        ("port", "8000.0"),
        ("port", "1e3"),
        ("port", " 8000"),
        ("port", Decimal(8000)),
        ("max_model_len", "4096 tokens"),
        ("gpu_memory_utilization", False),
        ("gpu_memory_utilization", "nan"),
        ("gpu_memory_utilization", "inf"),
        ("gpu_memory_utilization", "0,9"),
        ("gpu_memory_utilization", "0.9_0"),
        ("gpu_memory_utilization", Decimal("0.9")),
    ],
)
def test_non_canonical_numbers_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _validate(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("port", "8080", 8080),
        ("port", "+8080", 8080),
        ("gpu_memory_utilization", "0.85", 0.85),
        ("gpu_memory_utilization", ".5", 0.5),
        ("gpu_memory_utilization", "1", 1.0),
        ("gpu_memory_utilization", "5e-1", 0.5),
    ],
)
def test_canonical_numeric_strings_are_parsed(field: str, value: str, expected: object) -> None:
    assert getattr(_validate(**{field: value}), field) == expected


@given(port=st.integers(min_value=1, max_value=65535))
def test_any_valid_port_string_round_trips(port: int) -> None:
    assert _validate(port=str(port)).port == port


@given(fraction=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, allow_nan=False))
def test_any_valid_gpu_fraction_round_trips_through_repr(fraction: float) -> None:
    assert _validate(gpu_memory_utilization=repr(fraction)).gpu_memory_utilization == fraction


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "127.0.0.1", "::", "::1", "localhost", "vllm-0.svc.cluster.local", "a."],  # noqa: S104
)
def test_valid_hosts_are_accepted(host: str) -> None:
    assert _validate(host=host).host == host


@pytest.mark.parametrize(
    "host",
    [
        "",
        "   ",
        ".",
        "-bad.example",
        "bad-.example",
        "under_score",
        "a" * 64,
        ("a" * 63 + ".") * 4 + "a",
    ],
)
def test_invalid_hosts_are_rejected(host: str) -> None:
    with pytest.raises(ValidationError, match="host"):
        _validate(host=host)


@pytest.mark.parametrize(
    "model", ["", "  ", "/opt/models/llama", "../models", "C:\\models", "~root/m"]
)
def test_model_rejects_blank_and_host_paths(model: str) -> None:
    with pytest.raises(ValidationError, match="model"):
        VllmDeploymentConfig.model_validate({"model": model})


@pytest.mark.parametrize("model", [MODEL_ID, "./models/llama-3-8b"])
def test_model_accepts_hub_ids_and_relative_paths(model: str) -> None:
    assert VllmDeploymentConfig.model_validate({"model": model}).model == model


@pytest.mark.parametrize(
    ("value", "expected"),
    [("awq", QuantizationMethod.AWQ), ("fp8", QuantizationMethod.FP8), (None, None)],
)
def test_quantization(value: str | None, expected: QuantizationMethod | None) -> None:
    assert _validate(quantization=value).quantization is expected


def test_quantization_rejects_unknown_methods() -> None:
    with pytest.raises(ValidationError, match="quantization"):
        _validate(quantization="int4")


# --------------------------------------------------------------------------- #
# Environment and .env sources                                                #
# --------------------------------------------------------------------------- #


def test_process_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", MODEL_ID)
    monkeypatch.setenv("VLLM_PORT", "9000")
    monkeypatch.setenv("vllm_max_model_len", "2048")
    monkeypatch.setenv("VLLM_QUANTIZATION", "none")
    monkeypatch.setenv("VLLM_TENSOR_PARALLEL_SIZE", "")
    config = VllmDeploymentConfig()
    assert (config.port, config.max_model_len, config.quantization) == (9000, 2048, None)
    assert config.tensor_parallel_size == 1


def test_process_env_rejects_non_canonical_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", MODEL_ID)
    monkeypatch.setenv("VLLM_PORT", "true")
    with pytest.raises(ValidationError, match="port"):
        VllmDeploymentConfig()


def test_unrelated_vllm_process_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # vLLM itself reads many VLLM_* variables; they must not break config loading.
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    assert VllmDeploymentConfig(model=MODEL_ID).model == MODEL_ID


def test_implicit_dotenv_in_cwd_is_read(isolated_env: Path) -> None:
    _write_env(isolated_env / ".env", VLLM_MODEL=MODEL_ID, VLLM_GPU_MEMORY_UTILIZATION="0.75")
    assert VllmDeploymentConfig().gpu_memory_utilization == 0.75


def test_env_vars_override_dotenv(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(isolated_env / ".env", VLLM_MODEL=MODEL_ID, VLLM_PORT="8001")
    monkeypatch.setenv("VLLM_PORT", "8002")
    assert VllmDeploymentConfig.from_env_file(".env").port == 8002


def test_shared_dotenv_ignores_other_tools_keys(isolated_env: Path) -> None:
    _write_env(
        isolated_env / "deploy.env",
        VLLM_MODEL=MODEL_ID,
        VLLM_PORT="8000",
        LOCUST_HOST="http://127.0.0.1:8000",
        LOCUST_USERS="50",
    )
    config = VllmDeploymentConfig.from_env_file("deploy.env")
    assert (config.model, config.port) == (MODEL_ID, 8000)


def test_unknown_vllm_keys_in_dotenv_are_rejected(isolated_env: Path) -> None:
    _write_env(isolated_env / ".env", VLLM_MODEL=MODEL_ID, VLLM_GPU_MEMORY_UTILISATION="0.5")
    with pytest.raises(ValidationError, match="gpu_memory_utilisation"):
        VllmDeploymentConfig.from_env_file(".env")


def test_dotenv_values_are_strictly_parsed(isolated_env: Path) -> None:
    _write_env(isolated_env / ".env", VLLM_MODEL=MODEL_ID, VLLM_GPU_MEMORY_UTILIZATION="nan")
    with pytest.raises(ValidationError, match="gpu_memory_utilization"):
        VllmDeploymentConfig.from_env_file(".env")


def test_repository_env_example_is_loadable(isolated_env: Path, repo_root: Path) -> None:
    example = (repo_root / ".env.example").read_text(encoding="utf-8")
    (isolated_env / ".env").write_text(example, encoding="utf-8")
    config = VllmDeploymentConfig.from_env_file()
    assert config.model == MODEL_ID
    assert config.max_num_batched_tokens == 8192


def test_env_file_accepts_pathlike(isolated_env: Path) -> None:
    (isolated_env / "conf").mkdir()
    _write_env(isolated_env / "conf" / "vllm.env", VLLM_MODEL=MODEL_ID)
    assert VllmDeploymentConfig.from_env_file(Path("conf") / "vllm.env").model == MODEL_ID


@pytest.mark.parametrize(
    ("env_file", "reason"),
    [
        ("", "non-whitespace"),
        ("/etc/vllm.env", "absolute"),
        ("C:\\deploy\\.env", "absolute"),
        ("../.env", "traversal"),
        ("conf/../../.env", "traversal"),
        ("~root/.env", "home-relative"),
        ("missing.env", "does not exist"),
        (".", "does not exist"),
    ],
)
def test_env_file_path_escapes_are_rejected(env_file: str, reason: str) -> None:
    with pytest.raises(ConfigPathError, match=reason):
        VllmDeploymentConfig.from_env_file(env_file)


def test_env_file_symlink_escape_is_rejected(isolated_env: Path) -> None:
    outside = _write_env(isolated_env.parent / "outside.env", VLLM_MODEL=MODEL_ID)
    (isolated_env / "link.env").symlink_to(outside)
    with pytest.raises(ConfigPathError, match="outside the project root"):
        VllmDeploymentConfig.from_env_file("link.env")


def test_config_path_error_is_a_value_error() -> None:
    assert issubclass(ConfigPathError, ValueError)
