"""Baseline verification: tool configuration, import paths, storage anchoring, network isolation."""

from __future__ import annotations

import asyncio
import os
import socket
import stat
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase
from pytest_socket import SocketConnectBlockedError

import high_throughput_vllm_deployment as pkg
from conftest import (
    HYPOTHESIS_ROOT,
    LOOPBACK_V4,
    LOOPBACK_V6,
    REPO_ROOT,
    TMP_ROOT,
    DnsLeakError,
)

PYPROJECT: Final[dict[str, Any]] = tomllib.loads(
    (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)
TOOL: Final[dict[str, Any]] = PYPROJECT["tool"]

# pytest-socket emits a UserWarning alongside every block; with
# `filterwarnings = error` that warning would mask the real exception type.
ignore_socket_warning = pytest.mark.filterwarnings("ignore:A test tried to use socket")


def _is_within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


# --------------------------------------------------------------------------- #
# Tool configuration                                                          #
# --------------------------------------------------------------------------- #


def test_python_requirement_is_pinned_to_312() -> None:
    assert PYPROJECT["project"]["requires-python"] == ">=3.12,<3.13"


def test_runtime_dependencies() -> None:
    assert sorted(PYPROJECT["project"]["dependencies"]) == [
        "httpx>=0.27.0",
        "pydantic-settings>=2.16.2",
        "pydantic>=2.8.0",
    ]


def test_dev_dependencies() -> None:
    expected = {
        "ruff>=0.6.0",
        "mypy>=1.11.0",
        "pytest>=8.2.0",
        "pytest-asyncio>=0.24.0",
        "pytest-cov>=5.0.0",
        "pytest-socket>=0.7.0",
        "hypothesis>=6.100.0",
        "types-pyyaml>=6.0.0",
        "pyyaml>=6.0.0",
    }
    dev = PYPROJECT["dependency-groups"]["dev"]
    assert {d for d in dev if isinstance(d, str)} == expected
    assert {"include-group": "loadtest"} in dev
    assert PYPROJECT["dependency-groups"]["loadtest"] == ["locust>=2.31.0"]


@pytest.mark.parametrize(
    ("section", "key", "expected"),
    [
        ("uv", "cache-dir", "./.cache/uv"),
        ("ruff", "cache-dir", "./.cache/ruff"),
        ("mypy", "cache_dir", "./.cache/mypy"),
        ("pytest.ini_options", "cache_dir", "./.cache/pytest"),
        ("coverage.run", "data_file", "./.cache/coverage/.coverage"),
    ],
)
def test_tool_cache_routing(section: str, key: str, expected: str) -> None:
    table: dict[str, Any] = TOOL
    for part in section.split("."):
        table = table[part]
    value = table[key]
    assert value == expected
    assert not Path(value).is_absolute()
    assert _is_within(REPO_ROOT / value, REPO_ROOT / ".cache")


def test_ruff_configuration() -> None:
    ruff = TOOL["ruff"]
    assert ruff["target-version"] == "py312"
    assert {"E", "F", "B", "I", "S", "C4", "T201"} <= set(ruff["lint"]["select"])


def test_mypy_is_strict() -> None:
    assert TOOL["mypy"]["strict"] is True


def test_pytest_network_isolation_flags() -> None:
    addopts = TOOL["pytest"]["ini_options"]["addopts"].split()
    assert "--disable-socket" in addopts
    assert "--allow-hosts=127.0.0.1,localhost" in addopts


def test_coverage_measures_branches() -> None:
    assert TOOL["coverage"]["run"]["branch"] is True


# --------------------------------------------------------------------------- #
# Import paths                                                                #
# --------------------------------------------------------------------------- #


def test_package_imports_from_src_layout() -> None:
    assert pkg.__file__ is not None
    assert _is_within(Path(pkg.__file__), REPO_ROOT / "src" / "high_throughput_vllm_deployment")


def test_package_version_matches_pyproject() -> None:
    assert pkg.__version__ == PYPROJECT["project"]["version"]
    assert pkg.__all__ == ["__version__"]


def test_package_ships_py_typed_marker() -> None:
    assert (REPO_ROOT / "src" / "high_throughput_vllm_deployment" / "py.typed").is_file()


# --------------------------------------------------------------------------- #
# Storage anchoring                                                           #
# --------------------------------------------------------------------------- #


def test_tmp_path_is_anchored_under_cache(tmp_path: Path) -> None:
    assert _is_within(tmp_path, TMP_ROOT)


def test_tempfile_is_anchored_under_cache() -> None:
    assert _is_within(Path(tempfile.gettempdir()), TMP_ROOT)
    for var in ("TMPDIR", "TEMP", "TMP"):
        assert _is_within(Path(os.environ[var]), TMP_ROOT)
    with tempfile.NamedTemporaryFile() as handle:
        assert _is_within(Path(handle.name), TMP_ROOT)


def test_hypothesis_database_is_anchored_under_cache() -> None:
    database = settings().database
    assert isinstance(database, DirectoryBasedExampleDatabase)
    assert _is_within(Path(database.path), HYPOTHESIS_ROOT)


def test_pytest_cache_is_anchored_under_cache(pytestconfig: pytest.Config) -> None:
    assert pytestconfig.cache is not None
    assert _is_within(pytestconfig.cache.mkdir("probe"), REPO_ROOT / ".cache" / "pytest")


# --------------------------------------------------------------------------- #
# Network isolation                                                           #
# --------------------------------------------------------------------------- #


@ignore_socket_warning
def test_non_loopback_connect_is_blocked() -> None:
    # 192.0.2.0/24 is TEST-NET-1 (RFC 5737): never routable, never resolved.
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)


@pytest.mark.parametrize(
    "lookup",
    [
        lambda: socket.getaddrinfo("example.com", 443),
        lambda: socket.getaddrinfo(b"example.com", 443),
        lambda: socket.gethostbyname("example.com"),
        lambda: socket.gethostbyname_ex("example.com"),
        lambda: socket.gethostbyaddr("192.0.2.1"),
        lambda: socket.gethostbyname("::1"),
    ],
    ids=[
        "getaddrinfo",
        "getaddrinfo-bytes",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "gethostbyname-v6",
    ],
)
def test_dns_resolution_is_blocked(lookup: Any) -> None:
    with pytest.raises(DnsLeakError):
        lookup()


@given(
    host=st.from_regex(r"\A[a-z][a-z0-9-]{0,30}\.(com|net|org|io|ai|dev)\Z").filter(
        lambda h: not h.startswith("localhost.")
    )
)
def test_arbitrary_hostnames_never_reach_dns(host: str) -> None:
    with pytest.raises(DnsLeakError):
        socket.getaddrinfo(host, 80)


@pytest.mark.parametrize(
    ("host", "family", "expected"),
    [
        ("localhost", socket.AF_INET, LOOPBACK_V4),
        ("LOCALHOST.", socket.AF_INET, LOOPBACK_V4),
        ("localhost", socket.AF_INET6, LOOPBACK_V6),
        ("127.0.0.1", socket.AF_INET, LOOPBACK_V4),
    ],
)
def test_loopback_names_resolve_statically(host: str, family: int, expected: str) -> None:
    infos = socket.getaddrinfo(host, 80, family, socket.SOCK_STREAM)
    assert {info[4][0] for info in infos} == {expected}
    assert socket.gethostbyname("localhost") == LOOPBACK_V4
    assert socket.gethostbyname_ex("localhost") == ("localhost", [], [LOOPBACK_V4])
    assert socket.gethostbyaddr(LOOPBACK_V4) == ("localhost", [], [LOOPBACK_V4])
    assert socket.gethostbyaddr(LOOPBACK_V6) == ("localhost", [], [LOOPBACK_V6])


def test_loopback_round_trip_is_allowed() -> None:
    with socket.create_server((LOOPBACK_V4, 0)) as server:
        port = server.getsockname()[1]
        with socket.create_connection(("localhost", port), timeout=1.0) as client:
            conn, _ = server.accept()
            with conn:
                client.sendall(b"ping")
                assert conn.recv(4) == b"ping"


def test_httpx_cannot_reach_external_hosts() -> None:
    with httpx.Client(timeout=0.5) as client, pytest.raises(httpx.ConnectError):
        client.get("https://example.com/")


@pytest.mark.asyncio
async def test_async_resolution_is_blocked() -> None:
    loop = asyncio.get_running_loop()
    with pytest.raises(DnsLeakError):
        await loop.getaddrinfo("example.com", 443)


# --------------------------------------------------------------------------- #
# Audit scripts                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["audit_identity.sh", "audit_hygiene.sh"])
def test_audit_scripts_are_executable_bash(name: str) -> None:
    script = REPO_ROOT / "scripts" / name
    assert script.stat().st_mode & stat.S_IXUSR
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert 'if [ -z "${BASH_VERSION:-}" ]; then' in text
    assert "set -euo pipefail" in text
