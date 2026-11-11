"""The mutation harness itself: catalogue validation, classification, rollback, CLI.

End-to-end runs use a miniature git repository under the test's tmp_path
(itself under ./.cache), never the real working tree.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mutation_runner as runner  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

CALC = textwrap.dedent(
    """\
    def add(a, b):
        return a + b


    def one():
        return 1


    def slow():
        return 2
    """
)
TESTS = textwrap.dedent(
    """\
    from pathlib import Path

    from calc import add, one, slow


    def test_add():
        assert add(2, 3) == 5


    def test_one():
        assert one() == 1


    def test_slow():
        assert slow() == 2


    def test_notes():
        assert Path("notes.txt").read_text() == "port 127.0.0.1\\n"
    """
)


def _mutant(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "schemas-add-subtracts",
        "suite": "schemas",
        "description": "add subtracts",
        "file": "calc.py",
        "old": "return a + b",
        "new": "return a - b",
        "tests": ["test_calc.py::test_add"],
        "expect": "test_add",
    }
    return base | overrides


def _toml(entries: list[dict[str, Any]]) -> str:
    def value(v: Any) -> str:
        if isinstance(v, list):
            return "[" + ", ".join(value(i) for i in v) + "]"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int | float):
            return repr(v)
        return "'''" + v + "'''" if "\n" in v or "'" not in v else repr(v)

    return "\n".join(
        "[[mutant]]\n" + "".join(f"{k} = {value(v)}\n" for k, v in entry.items())
        for entry in entries
    )


FULL_CATALOGUE = [
    _mutant(),  # KILLED
    _mutant(  # SURVIVED: semantically equivalent, but a real AST change
        id="core-one-equivalent",
        suite="core",
        old="return 1",
        new="return 1 if True else 0",
        tests=["test_calc.py::test_one"],
        expect="test_one",
    ),
    _mutant(  # MISMATCH: fails, but not in the declared test
        id="benchmarks-add-wrong-reason",
        suite="benchmarks",
        expect="test_one",
    ),
    _mutant(  # TIMEOUT
        id="docker-slow-hangs",
        suite="docker",
        old="return 2",
        new="return __import__('time').sleep(60)",
        tests=["test_calc.py::test_slow"],
        expect="test_slow",
        timeout_s=2,
    ),
    _mutant(  # CRASHED: collection error
        id="core-import-breaks",
        suite="core",
        old="def add(a, b):",
        new="import no_such_module_xyz\n\n\ndef add(a, b):",
        expect="test_add",
    ),
    _mutant(  # KILLED, non-Python target
        id="docker-notes-open-port",
        suite="docker",
        file="notes.txt",
        old="127.0.0.1",
        new="0.0.0.0",  # noqa: S104 - the mutated text, not a bind
        tests=["test_calc.py::test_notes"],
        expect="test_notes",
    ),
]


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)  # noqa: S603, S607


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "mini"
    root.mkdir()
    (root / "calc.py").write_text(CALC)
    (root / "test_calc.py").write_text(TESTS)
    (root / "notes.txt").write_text("port 127.0.0.1\n")
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / ".gitignore").write_text(".cache/\n__pycache__/\n")
    (root / "mutants.toml").write_text(_toml(FULL_CATALOGUE))
    _git(root, "-c", "init.defaultBranch=main", "init", "-q")
    _git(root, "add", "-A")
    return root


@pytest.fixture(autouse=True)
def restore_sigterm() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


def _main(project: Path, *args: str) -> int:
    return runner.main(["--root", str(project), "--catalogue", "mutants.toml", *args])


# --------------------------------------------------------------------------- #
# End to end                                                                  #
# --------------------------------------------------------------------------- #


def test_every_outcome_class_and_byte_exact_rollback(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.glob("*.*") if p.is_file()}
    assert _main(project) == runner.EXIT_OK  # report-only mode
    out = capsys.readouterr().out
    expected = {
        "schemas-add-subtracts": "KILLED",
        "core-one-equivalent": "SURVIVED",
        "benchmarks-add-wrong-reason": "MISMATCH",
        "docker-slow-hangs": "TIMEOUT",
        "core-import-breaks": "CRASHED",
        "docker-notes-open-port": "KILLED",
    }
    for mutant_id, status in expected.items():
        assert any(line.startswith(status) and mutant_id in line for line in out.splitlines()), (
            mutant_id,
            status,
            out,
        )
    assert "kill rate: 2/6 = 33.33%" in out
    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content, f"{path.name} was not restored"
        assert path.stat().st_mtime_ns == mtime, f"{path.name} mtime was not restored"
    assert not (project / runner.STATE_DIR / "journal.json").exists()
    assert (project / runner.STATE_DIR / "report.json").is_file()
    assert not list(project.rglob("__pycache__")), "mutant bytecode was written"


def test_strict_mode_fails_on_any_survivor_and_passes_when_all_killed(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project, "--strict", "--filter", "core") == runner.EXIT_SURVIVORS
    assert "STRICT: not killed: core-one-equivalent, core-import-breaks" in capsys.readouterr().out
    killed = ["--filter", "schemas-add-*", "--filter", "docker-notes-*"]
    assert _main(project, "--strict", *killed) == runner.EXIT_OK
    assert "kill rate: 2/2 = 100.00%" in capsys.readouterr().out


def test_list_mode(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(project, "--list", "--filter", "docker") == runner.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == ["docker-slow-hangs", "docker-notes-open-port"]


def test_verbose_mode_prints_pytest_output(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(project, "--verbose", "--filter", "schemas") == runner.EXIT_OK
    assert "AssertionError" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["unstaged", "untracked"])
def test_preflight_requires_a_clean_working_tree(
    project: Path, change: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if change == "unstaged":
        (project / "calc.py").write_text(CALC + "\n# edit\n")
    else:
        (project / "scratch.py").write_text("x = 1\n")
    assert _main(project, "--filter", "schemas") == runner.EXIT_HARNESS
    assert "working tree is not clean" in capsys.readouterr().out


def test_preflight_requires_tracked_targets(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(project, "rm", "-q", "--cached", "notes.txt")
    (project / ".gitignore").write_text(".cache/\n__pycache__/\nnotes.txt\n")
    _git(project, "add", ".gitignore")
    assert _main(project, "--filter", "docker-notes-*") == runner.EXIT_HARNESS
    assert "not tracked by git" in capsys.readouterr().out


def test_preflight_requires_a_green_baseline(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "test_calc.py").write_text(TESTS.replace("== 5", "== 6"))
    _git(project, "add", "-A")
    assert _main(project, "--filter", "schemas") == runner.EXIT_HARNESS
    assert "baseline is not green" in capsys.readouterr().out


def test_git_failures_are_harness_errors(project: Path) -> None:
    with pytest.raises(runner.HarnessError, match="git no-such-subcommand failed"):
        runner._git(project, "no-such-subcommand")


# --------------------------------------------------------------------------- #
# Journaled rollback                                                          #
# --------------------------------------------------------------------------- #


def _first(project: Path) -> runner.Mutant:
    return runner.load_catalogue(project / "mutants.toml", project)[0]


def _crash_mid_mutation(project: Path) -> None:
    """Apply the first mutant in a child process that dies without any cleanup."""
    code = textwrap.dedent(
        f"""
        import os, sys
        from pathlib import Path
        sys.path.insert(0, {str(REPO / "scripts")!r})
        import mutation_runner as r
        root = Path({str(project)!r})
        mutant = r.load_catalogue(root / "mutants.toml", root)[0]
        context = r.Workspace(root).mutated(mutant)  # keep alive: no GC finalizer
        context.__enter__()
        os._exit(9)  # like SIGKILL: no finally, no atexit
        """
    )
    completed = subprocess.run([sys.executable, "-B", "-c", code], check=False)  # noqa: S603
    assert completed.returncode == 9


def test_interrupted_run_is_recovered_on_next_start(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = (project / "calc.py").read_bytes()
    _crash_mid_mutation(project)
    assert (project / "calc.py").read_bytes() != original
    assert _main(project, "--list") == runner.EXIT_OK
    assert "recovered calc.py from an interrupted run" in capsys.readouterr().out
    assert (project / "calc.py").read_bytes() == original
    assert runner.Workspace(project).recover() is None


def test_corrupt_backup_is_refused(project: Path) -> None:
    _crash_mid_mutation(project)
    workspace = runner.Workspace(project)
    for backup in workspace.backups.iterdir():
        backup.write_bytes(b"tampered")
    with pytest.raises(runner.HarnessError, match="corrupt"):
        workspace.recover()


def test_rollback_is_verified(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digests = iter(["before", "after"])
    monkeypatch.setattr(runner, "sha256", lambda _data: next(digests))
    with pytest.raises(runner.HarnessError, match="rollback verification failed"):
        with runner.Workspace(project).mutated(_first(project)):
            pass


def test_target_drift_is_detected(project: Path) -> None:
    mutant = _first(project)
    (project / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    with pytest.raises(runner.HarnessError, match="target changed"):
        with runner.Workspace(project).mutated(mutant):
            pass  # pragma: no cover - never entered


def test_sigterm_unwinds_through_rollback() -> None:
    with pytest.raises(KeyboardInterrupt):
        runner._raise_on_sigterm(signal.SIGTERM, None)


# --------------------------------------------------------------------------- #
# Catalogue validation                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ([], "at least one"),
        ([{k: v for k, v in _mutant().items() if k != "expect"}], "missing keys"),
        ([_mutant(extra="x")], "unknown keys"),
        ([_mutant(description="")], "non-empty strings"),
        ([_mutant(id="NoSuitePrefix")], "kebab-case"),
        ([_mutant(), _mutant()], "duplicate id"),
        ([_mutant(suite="infra")], "suite must be one of"),
        ([_mutant(new="return a + b")], "identical"),
        ([_mutant(expect="(unclosed")], "invalid expect regex"),
        ([_mutant(tests=[])], "non-empty list"),
        ([_mutant(timeout_s=0)], "positive number"),
        ([_mutant(timeout_s=True)], "positive number"),
        ([_mutant(file="/etc/hosts")], "project-relative"),
        ([_mutant(file="../calc.py")], "project-relative"),
        ([_mutant(file="missing.py")], "no such file"),
        ([_mutant(tests=["test_missing.py::test_x"])], "no such test path"),
        ([_mutant(old="return a * b")], "occurs 0 times"),
        ([_mutant(old="return")], "occurs 3 times"),
        ([_mutant(old="def add(a, b):", new="def add(a, b):  # same AST")], "does not change"),
        ([_mutant(new="return a +")], "does not parse"),
    ],
)
def test_catalogue_errors(project: Path, entries: list[dict[str, Any]], match: str) -> None:
    (project / "bad.toml").write_text(_toml(entries))
    with pytest.raises(runner.HarnessError, match=match):
        runner.load_catalogue(project / "bad.toml", project)


def test_catalogue_must_cover_every_suite(project: Path) -> None:
    (project / "bad.toml").write_text(_toml([_mutant()]))
    with pytest.raises(runner.HarnessError, match="no mutants for suites"):
        runner.load_catalogue(project / "bad.toml", project)


def test_unreadable_catalogue(project: Path) -> None:
    (project / "bad.toml").write_text("[[mutant]\n")
    with pytest.raises(runner.HarnessError, match="cannot read catalogue"):
        runner.load_catalogue(project / "bad.toml", project)
    with pytest.raises(runner.HarnessError, match="cannot read catalogue"):
        runner.load_catalogue(project / "absent.toml", project)


def test_filter_without_matches(project: Path) -> None:
    mutants = runner.load_catalogue(project / "mutants.toml", project)
    with pytest.raises(runner.HarnessError, match="matches no mutant"):
        runner.select(mutants, ["nothing-*"])
    assert runner.select(mutants, []) == list(mutants)


# --------------------------------------------------------------------------- #
# Classification (fail-closed)                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("code", "output", "status"),
    [
        (None, "", "TIMEOUT"),
        (0, "1 passed", "SURVIVED"),
        (1, "FAILED t.py::test_add[a b] - AssertionError", "KILLED"),
        (1, "FAILED t.py::test_other - AssertionError", "MISMATCH"),
        (1, "ERROR t.py::test_add - fixture boom", "CRASHED"),
        (2, "FAILED t.py::test_add - collected anyway", "CRASHED"),
        (3, "INTERNALERROR", "CRASHED"),
        (-9, "", "CRASHED"),
    ],
)
def test_classification(code: int | None, output: str, status: str) -> None:
    mutant = runner.Mutant(
        id="core-x",
        suite="core",
        description="d",
        file="f.py",
        old="a",
        new="b",
        tests=("t.py",),
        expect="test_add",
    )
    assert runner.classify(mutant, code, output)[0] == status


# --------------------------------------------------------------------------- #
# The real catalogue                                                          #
# --------------------------------------------------------------------------- #

REQUIRED_CATEGORIES = {
    "schemas": ["base-accepts-nan", "negative-token", "percentile-order", "relaxed"],
    "core": ["batching-skips", "preemption-mode", "kv-block-leak", "fcfs", "prometheus", "sse"],
    "benchmarks": ["percentile", "zero-division", "poisson", "observer", "locust-weighting"],
    "docker": ["root", "unpinned", "all-interfaces", "package-managers"],
}


def test_real_catalogue_is_valid_and_covers_every_required_category() -> None:
    mutants = runner.load_catalogue(REPO / "scripts" / "mutants.toml", REPO)
    assert len(mutants) >= 50
    for suite, needles in REQUIRED_CATEGORIES.items():
        ids = [m.id for m in mutants if m.suite == suite]
        for needle in needles:
            assert any(needle in mutant_id for mutant_id in ids), f"{suite}: no {needle!r} mutant"


def test_wrapper_script_is_executable_bash_and_routes_state_into_cache() -> None:
    script = REPO / "scripts" / "run_mutations.sh"
    assert os.access(script, os.X_OK)
    text = script.read_text()
    assert 'if [ -z "${BASH_VERSION:-}" ]; then' in text and "set -euo pipefail" in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    assert 'export TMPDIR="$repo_root/.cache/tmp"' in text
    assert "python -B scripts/mutation_runner.py" in text
