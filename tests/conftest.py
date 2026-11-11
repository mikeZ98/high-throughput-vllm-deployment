"""Session-wide test isolation.

Two invariants are enforced for every test session:

1. **Storage anchoring** - every temporary and tool-cache directory that the
   test process may write to is rooted under ``./.cache`` inside the repository
   (``./.cache/tmp`` for ``tempfile``/``tmp_path``, ``./.cache/hypothesis`` for
   the Hypothesis example database). Nothing is written to the host home dir or
   the system temp dir.

2. **Network isolation without DNS leaks** - ``pytest-socket`` is configured
   (via ``addopts``) with ``--disable-socket --allow-hosts=127.0.0.1,localhost``.
   With an allow-list present, pytest-socket guards ``socket.socket.connect`` but
   leaves the resolver functions untouched, and it resolves ``localhost`` itself
   through the real ``getaddrinfo``. Name resolution happens in libc, outside
   Python's socket layer, so a lookup for an arbitrary hostname would still
   reach the system DNS resolver. The loopback resolver hook installed below
   closes that gap: loopback names resolve statically, IP literals are passed
   through in numeric-only mode, and every other lookup fails closed with
   :class:`DnsLeakError` *before* libc is consulted.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

if TYPE_CHECKING:
    from socket import _GetAddrInfoResult

# --------------------------------------------------------------------------- #
# Storage anchoring                                                           #
# --------------------------------------------------------------------------- #

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CACHE_ROOT: Final[Path] = REPO_ROOT / ".cache"
TMP_ROOT: Final[Path] = CACHE_ROOT / "tmp"
PYTEST_BASETEMP: Final[Path] = TMP_ROOT / "pytest"
HYPOTHESIS_ROOT: Final[Path] = CACHE_ROOT / "hypothesis"

_TMP_ENV_VARS: Final[tuple[str, ...]] = ("TMPDIR", "TEMP", "TMP")


def _anchor_storage() -> None:
    """Route tempfile, child-process temp dirs and Hypothesis under ``./.cache``."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    HYPOTHESIS_ROOT.mkdir(parents=True, exist_ok=True)

    for var in _TMP_ENV_VARS:
        os.environ[var] = str(TMP_ROOT)
    tempfile.tempdir = str(TMP_ROOT)

    os.environ["HYPOTHESIS_STORAGE_DIRECTORY"] = str(HYPOTHESIS_ROOT)
    set_hypothesis_home_dir(HYPOTHESIS_ROOT)
    settings.register_profile(
        "isolated",
        database=DirectoryBasedExampleDatabase(HYPOTHESIS_ROOT / "examples"),
    )
    settings.load_profile("isolated")


# Importing Locust normally gevent-monkey-patches the whole interpreter, which
# would corrupt asyncio, threading and pytest-socket for the rest of the session.
# Tests that import the locustfile get Locust *without* patching; the headless
# end-to-end test removes this variable from its subprocess environment.
os.environ["LOCUST_SKIP_MONKEY_PATCH"] = "1"

# Must run at import time: Hypothesis resolves its storage directory lazily but
# the profile has to be active before any `@given` test is collected.
_anchor_storage()


# --------------------------------------------------------------------------- #
# Loopback resolver hook                                                      #
# --------------------------------------------------------------------------- #

LOOPBACK_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
)
LOOPBACK_V4: Final[str] = "127.0.0.1"
LOOPBACK_V6: Final[str] = "::1"

_real_getaddrinfo: Final = socket.getaddrinfo
_real_gethostbyname: Final = socket.gethostbyname
_real_gethostbyname_ex: Final = socket.gethostbyname_ex
_real_gethostbyaddr: Final = socket.gethostbyaddr


class DnsLeakError(socket.gaierror):
    """Raised when a test attempts to resolve a non-loopback hostname."""

    def __init__(self, host: str) -> None:
        super().__init__(
            socket.EAI_NONAME,
            f"DNS resolution of {host!r} blocked by the test loopback resolver",
        )
        self.host = host


def _normalise(host: str | bytes) -> str:
    text = host.decode("idna") if isinstance(host, bytes) else host
    return text.strip().rstrip(".").lower()


def _ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None


def _static_target(host: str | bytes, family: int) -> str:
    """Map *host* to a numeric address without touching the system resolver."""
    name = _normalise(host)
    if name in LOOPBACK_HOSTNAMES:
        return LOOPBACK_V6 if family == socket.AF_INET6 else LOOPBACK_V4
    if _ip_literal(name) is not None:
        return name
    raise DnsLeakError(name)


def loopback_getaddrinfo(
    host: bytes | str | None,
    port: bytes | str | int | None,
    family: int = 0,
    type: int = 0,  # mirrors the stdlib signature
    proto: int = 0,
    flags: int = 0,
) -> _GetAddrInfoResult:
    if host is None:  # passive / wildcard lookups never hit DNS
        return _real_getaddrinfo(host, port, family, type, proto, flags)
    target = _static_target(host, family)
    return _real_getaddrinfo(target, port, family, type, proto, flags | socket.AI_NUMERICHOST)


def loopback_gethostbyname(hostname: str, /) -> str:
    target = _static_target(hostname, socket.AF_INET)
    ip = _ip_literal(target)
    if ip is None or ip.version != 4:
        raise DnsLeakError(hostname)
    return target


def loopback_gethostbyname_ex(hostname: str, /) -> tuple[str, list[str], list[str]]:
    return (_normalise(hostname), [], [loopback_gethostbyname(hostname)])


def loopback_gethostbyaddr(ip_address: str, /) -> tuple[str, list[str], list[str]]:
    """Reverse lookups are answered only for loopback addresses."""
    name = _normalise(ip_address)
    ip = _ip_literal(name)
    if name in LOOPBACK_HOSTNAMES or (ip is not None and ip.is_loopback):
        return (
            "localhost",
            [],
            [LOOPBACK_V6 if ip is not None and ip.version == 6 else LOOPBACK_V4],
        )
    raise DnsLeakError(name)


_resolver_patch: pytest.MonkeyPatch | None = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Pin pytest's basetemp under ./.cache/tmp and install the resolver hook.

    ``tryfirst`` guarantees this runs before the tmp_path plugin snapshots
    ``--basetemp`` and before pytest-socket resolves ``--allow-hosts``.
    """
    global _resolver_patch

    if config.option.basetemp is None:
        config.option.basetemp = str(PYTEST_BASETEMP)

    patch = pytest.MonkeyPatch()
    patch.setattr(socket, "getaddrinfo", loopback_getaddrinfo)
    patch.setattr(socket, "gethostbyname", loopback_gethostbyname)
    patch.setattr(socket, "gethostbyname_ex", loopback_gethostbyname_ex)
    patch.setattr(socket, "gethostbyaddr", loopback_gethostbyaddr)
    _resolver_patch = patch


def pytest_unconfigure(config: pytest.Config) -> None:
    global _resolver_patch
    if _resolver_patch is not None:
        _resolver_patch.undo()
        _resolver_patch = None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cache_root() -> Path:
    return CACHE_ROOT
