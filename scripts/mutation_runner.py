"""Strict, deterministic mutation-testing harness (standard library only).

Each mutant in the catalogue (``scripts/mutants.toml``) replaces one exact,
unique snippet of one file; the harness then runs the mutant's targeted tests
and classifies the outcome *fail-closed*:

* ``KILLED``    - pytest exited 1 and a ``FAILED`` test matches ``expect``;
* ``SURVIVED``  - the targeted tests passed;
* ``MISMATCH``  - tests failed, but none for the declared reason;
* ``CRASHED``   - collection/internal/usage errors, signals, or errors only;
* ``TIMEOUT``   - the run exceeded the mutant's time budget.

Only ``KILLED`` counts as a kill. Safety properties:

* preflight: clean working tree (no unstaged or untracked changes), a fully
  validated catalogue, and a passing baseline over every targeted test;
* byte-exact replacement; Python mutants must still parse and must change the
  AST (comment/whitespace-only edits are rejected as no-ops);
* atomic writes with SHA-256-verified rollback after every mutant, plus an
  on-disk journal and backup so that even a killed harness is repaired on the
  next start; original mtimes are restored so cached bytecode stays valid;
* tests run with ``PYTHONDONTWRITEBYTECODE=1`` (no stale mutant bytecode).

Exit codes: 0 = done (and, with ``--strict``, every mutant killed),
1 = ``--strict`` and at least one mutant not killed, 2 = harness/preflight error.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Final

SUITES: Final = ("schemas", "core", "benchmarks", "docker")
MUTANT_KEYS: Final = frozenset(
    {"id", "suite", "description", "file", "old", "new", "tests", "expect", "timeout_s"}
)
REQUIRED_KEYS: Final = MUTANT_KEYS - {"timeout_s"}
ID_PATTERN: Final = re.compile(r"[a-z]+(?:-[a-z0-9]+)+")
DEFAULT_TIMEOUT_S: Final = 180.0
STATE_DIR: Final = Path(".cache") / "mutations"
EXIT_OK, EXIT_SURVIVORS, EXIT_HARNESS = 0, 1, 2
KILLED: Final = "KILLED"


class HarnessError(Exception):
    """Preflight, catalogue or rollback failure: abort with exit code 2."""


@dataclass(frozen=True, slots=True)
class Mutant:
    id: str
    suite: str
    description: str
    file: str
    old: str
    new: str
    tests: tuple[str, ...]
    expect: str
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(frozen=True, slots=True)
class Outcome:
    mutant: str
    status: str
    seconds: float
    detail: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Catalogue                                                                   #
# --------------------------------------------------------------------------- #


def _ast_dump(source: str, where: str) -> str:
    try:
        return ast.dump(ast.parse(source))
    except SyntaxError as exc:
        raise HarnessError(f"{where}: does not parse: {exc}") from None


def _validate(entry: dict[str, Any], root: Path, seen: set[str]) -> Mutant:
    where = f"mutant {entry.get('id', '?')!r}"
    if missing := REQUIRED_KEYS - entry.keys():
        raise HarnessError(f"{where}: missing keys {sorted(missing)}")
    if unknown := entry.keys() - MUTANT_KEYS:
        raise HarnessError(f"{where}: unknown keys {sorted(unknown)}")
    strings = ("id", "suite", "description", "file", "old", "new", "expect")
    if not all(isinstance(entry[key], str) and entry[key] for key in strings):
        raise HarnessError(f"{where}: {', '.join(strings)} must be non-empty strings")
    mutant_id = entry["id"]
    if not ID_PATTERN.fullmatch(mutant_id):
        raise HarnessError(f"{where}: id must be kebab-case with a suite prefix")
    if mutant_id in seen:
        raise HarnessError(f"{where}: duplicate id")
    seen.add(mutant_id)
    if entry["suite"] not in SUITES:
        raise HarnessError(f"{where}: suite must be one of {SUITES}")
    if entry["old"] == entry["new"]:
        raise HarnessError(f"{where}: old and new are identical")
    try:
        re.compile(entry["expect"])
    except re.error as exc:
        raise HarnessError(f"{where}: invalid expect regex: {exc}") from None
    tests = entry["tests"]
    if not isinstance(tests, list) or not tests or not all(isinstance(t, str) for t in tests):
        raise HarnessError(f"{where}: tests must be a non-empty list of strings")
    timeout = entry.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(timeout, int | float) or isinstance(timeout, bool) or timeout <= 0:
        raise HarnessError(f"{where}: timeout_s must be a positive number")

    rel = Path(entry["file"])
    target = (root / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not target.is_relative_to(root.resolve()):
        raise HarnessError(f"{where}: file must be a project-relative path")
    if not target.is_file():
        raise HarnessError(f"{where}: no such file {rel}")
    for test in tests:
        if not (root / test.split("::", 1)[0]).exists():
            raise HarnessError(f"{where}: no such test path {test}")

    text = target.read_text(encoding="utf-8")
    if (count := text.count(entry["old"])) != 1:
        raise HarnessError(f"{where}: old snippet occurs {count} times in {rel} (need exactly 1)")
    if rel.suffix == ".py":
        mutated = text.replace(entry["old"], entry["new"], 1)
        if _ast_dump(text, f"{where} original") == _ast_dump(mutated, where):
            raise HarnessError(f"{where}: mutation does not change the AST (no-op)")
    return Mutant(
        id=mutant_id,
        suite=entry["suite"],
        description=entry["description"],
        file=rel.as_posix(),
        old=entry["old"],
        new=entry["new"],
        tests=tuple(tests),
        expect=entry["expect"],
        timeout_s=float(timeout),
    )


def load_catalogue(path: Path, root: Path) -> tuple[Mutant, ...]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HarnessError(f"cannot read catalogue {path}: {exc}") from None
    entries = data.get("mutant")
    if not isinstance(entries, list) or not entries:
        raise HarnessError("catalogue must define at least one [[mutant]]")
    seen: set[str] = set()
    mutants = tuple(_validate(entry, root, seen) for entry in entries)
    if missing := set(SUITES) - {m.suite for m in mutants}:
        raise HarnessError(f"catalogue has no mutants for suites {sorted(missing)}")
    return mutants


def select(mutants: Sequence[Mutant], filters: Sequence[str]) -> list[Mutant]:
    if not filters:
        return list(mutants)
    chosen = [
        m for m in mutants if any(f == m.suite or fnmatch.fnmatchcase(m.id, f) for f in filters)
    ]
    if not chosen:
        raise HarnessError(f"--filter {list(filters)} matches no mutant")
    return chosen


# --------------------------------------------------------------------------- #
# Preflight                                                                   #
# --------------------------------------------------------------------------- #


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed argv
        ["git", *args],  # noqa: S607 - git from PATH
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def dirty_paths(root: Path) -> list[str]:
    """Unstaged or untracked changes. Staged changes are fine: the index is the
    recovery source, so a clean *working tree* means worktree == index."""
    return [
        line
        for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line.startswith("??") or line[1] != " "
    ]


def assert_clean_tree(root: Path, mutants: Sequence[Mutant]) -> None:
    """Clean working tree, and every target file is tracked."""
    if dirty := dirty_paths(root):
        raise HarnessError("working tree is not clean:\n  " + "\n  ".join(dirty))
    tracked = set(_git(root, "ls-files").splitlines())
    if untracked := sorted({m.file for m in mutants} - tracked):
        raise HarnessError(f"mutation targets are not tracked by git: {untracked}")


# --------------------------------------------------------------------------- #
# Mutation with journaled, verified rollback                                  #
# --------------------------------------------------------------------------- #


def _atomic_write(target: Path, data: bytes, scratch: Path) -> None:
    temp = scratch / f"{target.name}.{os.getpid()}.tmp"
    temp.write_bytes(data)
    shutil.copymode(target, temp)
    os.replace(temp, target)  # same filesystem: scratch lives inside the project


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / STATE_DIR
        self.backups = self.state / "backup"
        self.journal = self.state / "journal.json"
        self.backups.mkdir(parents=True, exist_ok=True)

    def recover(self) -> str | None:
        """Undo a mutation left behind by a killed harness, if any."""
        if not self.journal.exists():
            return None
        entry = json.loads(self.journal.read_text(encoding="utf-8"))
        target, backup = self.root / entry["file"], Path(entry["backup"])
        original = backup.read_bytes()
        if sha256(original) != entry["sha256"]:
            raise HarnessError(f"backup for {entry['file']} is corrupt; restore it from git")
        _atomic_write(target, original, self.state)
        os.utime(target, ns=(entry["atime_ns"], entry["mtime_ns"]))
        self.journal.unlink()
        return str(entry["file"])

    @contextlib.contextmanager
    def mutated(self, mutant: Mutant) -> Iterator[None]:
        target = self.root / mutant.file
        original = target.read_bytes()
        digest, stat = sha256(original), target.stat()
        old, new = mutant.old.encode(), mutant.new.encode()
        if original.count(old) != 1:
            raise HarnessError(f"{mutant.id}: target changed since the catalogue was validated")
        backup = self.backups / digest
        backup.write_bytes(original)
        self.journal.write_text(
            json.dumps(
                {
                    "file": mutant.file,
                    "sha256": digest,
                    "backup": str(backup),
                    "atime_ns": stat.st_atime_ns,
                    "mtime_ns": stat.st_mtime_ns,
                }
            ),
            encoding="utf-8",
        )
        _atomic_write(target, original.replace(old, new, 1), self.state)
        try:
            yield
        finally:
            _atomic_write(target, original, self.state)
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            if sha256(target.read_bytes()) != digest:
                raise HarnessError(f"{mutant.id}: rollback verification failed for {mutant.file}")
            self.journal.unlink()


# --------------------------------------------------------------------------- #
# Test execution and classification                                           #
# --------------------------------------------------------------------------- #

_FAILED_PREFIX: Final = "FAILED "


def pytest_command(tests: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-rfE",
        "-o",
        f"cache_dir={STATE_DIR / 'pytest'}",
        *tests,
    ]


def run_tests(root: Path, tests: Sequence[str], timeout_s: float) -> tuple[int | None, str]:
    """Return (exit code or None on timeout, combined output)."""
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(root / ".cache" / "tmp"),
    }
    (root / ".cache" / "tmp").mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(  # noqa: S603 - fixed argv
        pytest_command(tests),
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return None, output
    return process.returncode, output


def classify(mutant: Mutant, code: int | None, output: str) -> tuple[str, str]:
    if code is None:
        return "TIMEOUT", f"exceeded {mutant.timeout_s:g}s"
    if code == 0:
        return "SURVIVED", "all targeted tests passed"
    # Whole "FAILED <node id> - <message>" lines: parametrized ids may contain spaces.
    failures = [
        line.removeprefix(_FAILED_PREFIX)
        for line in output.splitlines()
        if line.startswith(_FAILED_PREFIX)
    ]
    if code != 1 or not failures:
        return "CRASHED", f"pytest exit code {code} without assertion failures"
    if any(re.search(mutant.expect, failure) for failure in failures):
        return KILLED, failures[0]
    return "MISMATCH", f"failed, but not as expected ({mutant.expect!r}): {failures[0]}"


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _raise_on_sigterm(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt  # unwind through the rollback `finally`


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mutation catalogue against the tests.")
    parser.add_argument("--catalogue", default="scripts/mutants.toml")
    parser.add_argument("--root", default=".", help="project root (default: cwd)")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 unless every mutant is killed"
    )
    parser.add_argument(
        "--filter", action="append", default=[], help="suite name or id glob (repeatable)"
    )
    parser.add_argument("--verbose", action="store_true", help="print pytest output per mutant")
    parser.add_argument("--list", action="store_true", help="list selected mutants and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        workspace = Workspace(root)
        if recovered := workspace.recover():
            _write(f"recovered {recovered} from an interrupted run")
        mutants = select(load_catalogue(root / args.catalogue, root), args.filter)
        if args.list:
            for m in mutants:
                _write(f"{m.id:<40} {m.suite:<11} {m.file}: {m.description}")
            return EXIT_OK
        assert_clean_tree(root, mutants)

        targets = list(dict.fromkeys(t for m in mutants for t in m.tests))
        _write(f"baseline: {len(targets)} test target(s) for {len(mutants)} mutant(s)")
        code, output = run_tests(root, targets, max(m.timeout_s for m in mutants) * 2)
        if code != 0:
            raise HarnessError(f"baseline is not green (exit {code}):\n{output[-3000:]}")

        outcomes: list[Outcome] = []
        for mutant in mutants:
            started = time.monotonic()
            with workspace.mutated(mutant):
                code, output = run_tests(root, mutant.tests, mutant.timeout_s)
            status, detail = classify(mutant, code, output)
            outcomes.append(Outcome(mutant.id, status, time.monotonic() - started, detail))
            _write(f"{status:<9} {mutant.id:<40} {outcomes[-1].seconds:6.1f}s  {detail[:110]}")
            if args.verbose:
                _write(output)
        if changed := dirty_paths(root):
            raise HarnessError("working tree changed during the run: " + ", ".join(changed))
    except HarnessError as exc:
        _write(f"mutation harness error: {exc}")
        return EXIT_HARNESS

    killed = sum(o.status == KILLED for o in outcomes)
    rate = 100.0 * killed / len(outcomes)
    _write(f"kill rate: {killed}/{len(outcomes)} = {rate:.2f}%")
    report = workspace.state / "report.json"
    report.write_text(
        json.dumps({"kill_rate": rate, "outcomes": [asdict(o) for o in outcomes]}, indent=2),
        encoding="utf-8",
    )
    if args.strict and killed != len(outcomes):
        survivors = [o.mutant for o in outcomes if o.status != KILLED]
        _write(f"STRICT: not killed: {', '.join(survivors)}")
        return EXIT_SURVIVORS
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
