"""Shared contract primitives for every schema in the package.

All wire/data contracts inherit from :class:`ContractModel`, which is:

* **frozen** - instances are immutable and hashable;
* **closed** - unknown fields are rejected (``extra="forbid"``);
* **strict** - scalars are never coerced (``"1"`` is not an ``int``, ``True`` is
  not a number). Enum fields opt back into lax mode explicitly so that string
  values from JSON-decoded dicts are accepted;
* **finite** - ``NaN`` and ``+/-Infinity`` are rejected for every float.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Final

from pydantic import AfterValidator, AllowInfNan, BaseModel, ConfigDict, Field

CONTRACT_CONFIG: Final[ConfigDict] = ConfigDict(
    frozen=True,
    extra="forbid",
    strict=True,
    allow_inf_nan=False,
    validate_default=True,
    validate_return=True,
)


class ContractModel(BaseModel):
    """Frozen, closed, strict, finite base model for all data contracts."""

    model_config = CONTRACT_CONFIG


def require_non_blank(value: str) -> str:
    """Reject empty and whitespace-only strings (the value itself is preserved)."""
    if not value.strip():
        raise ValueError("must contain at least one non-whitespace character")
    return value


def reject_host_path_escape(value: str) -> str:
    """Reject values that would address the host filesystem outside the project.

    Absolute POSIX/Windows paths, drive- or UNC-qualified paths, home-relative
    (``~``) paths, parent-directory traversal and NUL bytes are all refused.
    Relative paths and Hugging Face style ``org/name`` identifiers pass.
    """
    if "\x00" in value:
        raise ValueError("path must not contain NUL bytes")
    if value.startswith("~"):
        raise ValueError("home-relative paths are not allowed")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        raise ValueError("absolute host paths are not allowed")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError("parent-directory traversal is not allowed")
    return value


NonEmptyStr = Annotated[str, AfterValidator(require_non_blank)]
"""A string with at least one non-whitespace character."""

RelativePathStr = Annotated[
    str, AfterValidator(require_non_blank), AfterValidator(reject_host_path_escape)
]
"""A non-blank string that cannot escape to an absolute or parent host path."""

FiniteFloat = Annotated[float, AllowInfNan(False)]
"""A float that is neither NaN nor infinite."""

NonNegativeFloat = Annotated[float, AllowInfNan(False), Field(ge=0.0)]
"""A finite float ``>= 0``."""

Percentage = Annotated[float, AllowInfNan(False), Field(ge=0.0, le=100.0)]
"""A finite percentage in ``[0, 100]``."""

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
