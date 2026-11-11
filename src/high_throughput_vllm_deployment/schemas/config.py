"""vLLM engine deployment configuration, loaded from ``VLLM_*`` env vars and ``.env``."""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Annotated, Any, Final, Self

from pydantic import AfterValidator, BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from high_throughput_vllm_deployment.schemas._base import (
    RelativePathStr,
    reject_host_path_escape,
    require_non_blank,
)
from high_throughput_vllm_deployment.schemas.enums import LaxQuantizationMethod

DEFAULT_ENV_FILE: Final[str] = ".env"

_INT_LITERAL: Final = re.compile(r"[+-]?[0-9]+")
_FLOAT_LITERAL: Final = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_HOSTNAME_LABEL: Final = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)")


class ConfigPathError(ValueError):
    """Raised when an env-file path is absolute, escapes the project, or is missing."""


# Environment values always arrive as strings, so settings fields cannot use
# pydantic strict mode. These validators restore strictness: only canonical
# numeric literals are accepted (no booleans, no "8_000", no "1e3" for ints,
# no "nan"/"inf"), and Python callers may not pass floats for integer fields.
def _strict_int(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("booleans are not accepted for integer settings")
    if isinstance(value, str):
        if not _INT_LITERAL.fullmatch(value):
            raise ValueError(f"{value!r} is not a canonical base-10 integer")
        return int(value)
    if isinstance(value, int):
        return value
    raise ValueError(f"expected an integer, got {type(value).__name__}")


def _strict_float(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("booleans are not accepted for float settings")
    if isinstance(value, str):
        if not _FLOAT_LITERAL.fullmatch(value):
            raise ValueError(f"{value!r} is not a canonical finite decimal number")
        return float(value)
    if isinstance(value, int | float):
        return value
    raise ValueError(f"expected a number, got {type(value).__name__}")


def _validate_host(value: str) -> str:
    """Accept an IPv4/IPv6 literal or an RFC 1123 hostname."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        name = value[:-1] if value.endswith(".") else value
        if not 0 < len(name) <= 253 or not all(
            _HOSTNAME_LABEL.fullmatch(label) for label in name.split(".")
        ):
            raise ValueError(f"{value!r} is neither an IP address nor a valid hostname") from None
    return value


StrictSettingInt = Annotated[int, BeforeValidator(_strict_int)]
StrictSettingFloat = Annotated[float, BeforeValidator(_strict_float)]
HostStr = Annotated[str, AfterValidator(require_non_blank), AfterValidator(_validate_host)]


class VllmDeploymentConfig(BaseSettings):
    """Engine and API-server settings for a single vLLM deployment.

    Values resolve from (highest priority first) init kwargs, ``VLLM_*``
    environment variables, then the ``.env`` file. Unknown ``VLLM_*`` keys in
    the ``.env`` file are rejected; unknown process env vars are ignored because
    vLLM itself reads many ``VLLM_*`` variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        # Only VLLM_* keys of a shared .env are considered; with extra="forbid" an
        # unknown VLLM_* key (e.g. a typo) is an error while LOCUST_* etc. are ignored.
        dotenv_filtering="match_prefix",
        env_ignore_empty=True,
        env_parse_none_str="none",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    model: RelativePathStr = Field(
        description="Hugging Face model id or project-relative local model path."
    )
    host: HostStr = Field(
        default="0.0.0.0",  # noqa: S104 - the API server must accept traffic from outside its container
        description="Interface the OpenAI-compatible API server binds to.",
    )
    port: StrictSettingInt = Field(
        default=8000,
        ge=1,
        le=65535,
        description=(
            "API server port. Read from VLLM_DEPLOYMENT_PORT, falling back to the legacy "
            "VLLM_PORT (which vLLM itself also reads as its internal IPC port)."
        ),
    )
    deployment_port: StrictSettingInt | None = Field(
        default=None,
        ge=1,
        le=65535,
        exclude=True,
        repr=False,
        description="Collision-free spelling of `port`; when set it takes precedence.",
    )
    gpu_memory_utilization: StrictSettingFloat = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory vLLM may reserve (weights, activations, KV cache).",
    )
    max_num_batched_tokens: StrictSettingInt = Field(default=8192, ge=1)
    max_model_len: StrictSettingInt = Field(default=4096, ge=1)
    tensor_parallel_size: StrictSettingInt = Field(default=1, ge=1)
    pipeline_parallel_size: StrictSettingInt = Field(default=1, ge=1)
    quantization: LaxQuantizationMethod | None = Field(
        default=None, description="Weight quantization method; None means unquantized."
    )

    @model_validator(mode="before")
    @classmethod
    def _prefer_deployment_port(cls, data: Any) -> Any:
        """Fold VLLM_DEPLOYMENT_PORT into `port`.

        The dedicated variable always wins, whatever source either value came
        from: a deployment that sets it may legitimately also export VLLM_PORT
        for vLLM's own inter-process communication.
        """
        if isinstance(data, dict) and data.get("deployment_port") is not None:
            return {**data, "port": data["deployment_port"]}
        return data

    @property
    def world_size(self) -> int:
        """Number of GPUs the deployment spans (tensor x pipeline parallelism)."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @classmethod
    def from_env_file(cls, env_file: str | os.PathLike[str] = DEFAULT_ENV_FILE) -> Self:
        """Load settings from an explicit, project-contained ``.env`` file.

        The path must be relative, must not traverse upwards, must resolve
        (after following symlinks) inside the current working directory, and
        must exist. Unlike the implicit default lookup, a missing file is an
        error rather than silently ignored.
        """
        raw = os.fspath(env_file)
        try:
            reject_host_path_escape(require_non_blank(raw))
        except ValueError as exc:
            raise ConfigPathError(f"env file {raw!r}: {exc}") from None
        root = Path.cwd().resolve()
        resolved = (root / raw).resolve()
        if not resolved.is_relative_to(root):
            raise ConfigPathError(f"env file {raw!r} resolves outside the project root")
        if not resolved.is_file():
            raise ConfigPathError(f"env file {raw!r} does not exist")
        return cls(_env_file=resolved)
