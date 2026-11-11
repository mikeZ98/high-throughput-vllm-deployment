"""Daemonless verification of the container build and Compose cluster.

No Docker daemon, VM or registry is touched (nothing may be written to the
host home directory). Instead this module:

* parses the Dockerfile and checks grammar, digest pinning, stage layout,
  layer order, bytecode-only packaging, the non-root user and the absence of
  package managers in the runtime image;
* simulates Docker's ``.dockerignore`` matcher over the real file tree to
  prove exactly which files can enter the build context;
* loads ``compose.yaml`` with PyYAML, renders its ``${VAR:-default}``
  interpolation against ``.env.example`` and checks ports, health checks,
  GPU reservations, hardening and volume containment;
* executes the exact HEALTHCHECK command vectors (Dockerfile and Compose)
  against an in-process server, and the image's Locust entry shim through
  Locust's own loader.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from locust.util.load_locustfile import load_locustfile

from high_throughput_vllm_deployment.benchmarks.mock_server import ThreadedMockServer
from high_throughput_vllm_deployment.core import instant_sleep
from high_throughput_vllm_deployment.core.mock import MockEngineConfig

ROOT: Final = Path(__file__).resolve().parent.parent
DIGEST: Final = re.compile(r"@sha256:[0-9a-f]{64}$")
KNOWN_INSTRUCTIONS: Final = frozenset(
    "FROM RUN CMD LABEL EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG ONBUILD "
    "STOPSIGNAL HEALTHCHECK SHELL MAINTAINER".split()
)
JSON_FORM: Final = frozenset({"CMD", "ENTRYPOINT", "SHELL"})
DEFAULT_HEALTH_URL: Final = "http://127.0.0.1:8000/health"


# --------------------------------------------------------------------------- #
# Dockerfile model                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Instruction:
    keyword: str
    args: str
    lineno: int


@dataclass(frozen=True)
class Stage:
    base: str
    name: str
    instructions: tuple[Instruction, ...]

    def all(self, keyword: str) -> list[Instruction]:
        return [i for i in self.instructions if i.keyword == keyword]

    @property
    def run_script(self) -> str:
        return "\n".join(i.args for i in self.all("RUN"))


def parse_dockerfile(text: str) -> tuple[dict[str, str], list[Stage]]:
    """Return (global ARG defaults, stages) from Dockerfile text."""
    logical: list[tuple[int, str]] = []
    buffer, start = "", 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        if buffer and line.startswith("#"):
            continue  # comments inside continuations are dropped by Docker
        if not buffer:
            start = lineno
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        logical.append((start, buffer + line))
        buffer = ""
    assert not buffer, "dangling line continuation"

    global_args: dict[str, str] = {}
    stages: list[Stage] = []
    current: list[Instruction] = []
    base = name = ""
    for lineno, line in logical:
        keyword, _, args = line.partition(" ")
        keyword = keyword.upper()
        assert keyword in KNOWN_INSTRUCTIONS, f"line {lineno}: unknown instruction {keyword}"
        if keyword == "FROM":
            if base:
                stages.append(Stage(base, name, tuple(current)))
            parts = args.split()
            assert len(parts) == 3 and parts[1].upper() == "AS", f"line {lineno}: unnamed stage"
            base, name, current = parts[0], parts[2], []
            continue
        if not base:
            assert keyword == "ARG", f"line {lineno}: only ARG may precede FROM"
            key, _, value = args.partition("=")
            global_args[key] = value
            continue
        current.append(Instruction(keyword, args.strip(), lineno))
    stages.append(Stage(base, name, tuple(current)))
    return global_args, stages


@pytest.fixture(scope="module")
def dockerfile() -> tuple[dict[str, str], dict[str, Stage], list[Stage]]:
    global_args, stages = parse_dockerfile((ROOT / "Dockerfile").read_text(encoding="utf-8"))
    return global_args, {s.name: s for s in stages}, stages


def _resolve(ref: str, global_args: dict[str, str]) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: global_args[m.group(1)], ref)


def _healthcheck_argv(stage: Stage) -> list[str]:
    (check,) = stage.all("HEALTHCHECK")
    _, _, command = check.args.partition(" CMD ")
    argv = json.loads(command)
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    return argv


def test_grammar_and_stage_layout(dockerfile: Any) -> None:
    _, stages, ordered = dockerfile
    assert [s.name for s in ordered] == ["uv", "builder", "runtime-rootfs", "runtime"]
    for stage in ordered:
        for instruction in stage.instructions:
            if instruction.keyword in JSON_FORM:
                assert isinstance(json.loads(instruction.args), list), instruction
            if instruction.keyword == "COPY" and "--from=" in instruction.args:
                source = re.search(r"--from=(\S+)", instruction.args)
                assert source is not None and source.group(1) in stages
    assert not any(i.keyword in {"ADD", "MAINTAINER"} for s in ordered for i in s.instructions)


def test_base_images_are_pinned_by_digest(dockerfile: Any) -> None:
    global_args, _, ordered = dockerfile
    known = {s.name for s in ordered}
    for stage in ordered:
        image = _resolve(stage.base, global_args)
        if image == "scratch" or image in known:
            continue
        assert DIGEST.search(image), f"{stage.name}: {image} is not digest-pinned"
        assert ":latest" not in image
    assert _resolve(stages_base(ordered, "builder"), global_args).startswith(
        "python:3.12-slim-bookworm@sha256:"
    )
    # The pruned rootfs comes from the very same base as the builder (same Python ABI).
    assert stages_base(ordered, "runtime-rootfs") == stages_base(ordered, "builder")


def stages_base(ordered: list[Stage], name: str) -> str:
    return next(s.base for s in ordered if s.name == name)


def test_builder_installs_locked_dependencies_before_copying_source(dockerfile: Any) -> None:
    _, stages, _ = dockerfile
    builder = stages["builder"]
    sequence = [(i.keyword, i.args) for i in builder.instructions if i.keyword in {"COPY", "RUN"}]
    deps_copy = next(n for n, (k, a) in enumerate(sequence) if k == "COPY" and "uv.lock" in a)
    deps_sync = next(n for n, (k, a) in enumerate(sequence) if "--no-install-project" in a)
    src_copy = next(n for n, (k, a) in enumerate(sequence) if k == "COPY" and a.startswith("src "))
    assert deps_copy < deps_sync < src_copy
    assert "src" not in sequence[deps_copy][1].split()
    script = builder.run_script
    for flags in ("uv sync --frozen --no-dev", "--no-editable"):
        assert flags in script
    env = " ".join(i.args for i in builder.all("ENV"))
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in env and "UV_PYTHON_DOWNLOADS=never" in env
    assert "UV_NO_CACHE=1" in env  # no cache directory baked into layers


def test_application_ships_as_hash_checked_bytecode_only(dockerfile: Any) -> None:
    _, stages, _ = dockerfile
    script = stages["builder"].run_script
    assert "python -m compileall -q -b --invalidation-mode=checked-hash" in script
    assert "-name '*.py' -delete" in script
    assert "-name '__pycache__' -prune -exec rm -rf" in script
    # The stripped tree is import-checked at build time (fail closed).
    assert script.rstrip().endswith(
        "-c 'import high_throughput_vllm_deployment.benchmarks.mock_server'"
    )


def test_runtime_is_flattened_non_root_and_free_of_package_managers(dockerfile: Any) -> None:
    _, stages, ordered = dockerfile
    rootfs, runtime = stages["runtime-rootfs"], stages["runtime"]
    assert ordered[-1] is runtime and runtime.base == "scratch"
    assert [i.args for i in runtime.all("COPY")] == ["--from=runtime-rootfs / /"]

    script = rootfs.run_script
    assert "groupadd --system --gid 10001 vllmops" in script
    assert re.search(
        r"useradd --system --uid 10001 --gid vllmops .*--shell /usr/sbin/nologin vllmops", script
    )
    for removed in (
        "/usr/bin/apt-*",
        "/usr/bin/dpkg",
        "/var/lib/dpkg",
        "/usr/local/bin/pip",
        "/usr/local/lib/python3.12/site-packages/pip",
        "/usr/local/lib/python3.12/ensurepip",
    ):
        assert removed in script
    for tool in ("apt-get", "apt", "dpkg", "pip"):
        assert f"! command -v {tool}" in script  # build fails if any survived
    assert "chmod a-s" in script  # no setuid/setgid binaries
    for stage in (rootfs, runtime):
        assert not re.search(r"\b(apt-get|apt|pip|uv)\s+install\b", stage.run_script)

    (user,) = runtime.all("USER")
    assert user.args == "10001:10001"
    assert runtime.instructions.index(user) > max(
        runtime.instructions.index(i) for i in runtime.all("RUN")
    ), "USER must follow every RUN"
    assert [i.args for i in runtime.all("EXPOSE")] == ["8000"]
    entrypoint = json.loads(runtime.all("ENTRYPOINT")[0].args)
    assert entrypoint == ["python", "-m", "high_throughput_vllm_deployment.benchmarks.mock_server"]
    assert json.loads(runtime.all("CMD")[0].args) == ["--host", "0.0.0.0", "--port", "8000"]  # noqa: S104 - asserting the container's bind address
    env = " ".join(i.args for i in runtime.all("ENV"))
    assert "PATH=/opt/venv/bin:" in env and "PYTHONDONTWRITEBYTECODE=1" in env


def test_healthcheck_is_stdlib_only(dockerfile: Any) -> None:
    _, stages, _ = dockerfile
    (check,) = stages["runtime"].all("HEALTHCHECK")
    for option in ("--interval=10s", "--timeout=5s", "--retries=3"):
        assert option in check.args
    argv = _healthcheck_argv(stages["runtime"])
    assert argv[:2] == ["/opt/venv/bin/python", "-c"]
    assert DEFAULT_HEALTH_URL in argv[2] and "urllib.request" in argv[2]
    assert not re.search(r"\b(curl|wget)\b", (ROOT / "Dockerfile").read_text())


def test_no_secrets_or_host_paths_are_baked_in(dockerfile: Any) -> None:
    _, _, ordered = dockerfile
    for stage in ordered:
        for instruction in stage.all("ENV") + stage.all("ARG"):
            assert not re.search(r"(TOKEN|SECRET|PASSWORD|API_KEY)", instruction.args, re.I)
        for instruction in stage.instructions:
            assert not re.search(r"/(Users|home|Volumes)/|~/", instruction.args)


# --------------------------------------------------------------------------- #
# .dockerignore simulation                                                    #
# --------------------------------------------------------------------------- #


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def dockerignore_rules(text: str) -> list[tuple[bool, re.Pattern[str]]]:
    """(is_reinclusion, regex) in file order, cleaned like Docker's patternmatcher."""
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        cleaned = line.lstrip("!").strip().lstrip("/").removeprefix("./").rstrip("/")
        rules.append((negated, _pattern_regex(cleaned)))
    return rules


def in_build_context(path: str, rules: list[tuple[bool, re.Pattern[str]]]) -> bool:
    """Docker semantics: the last matching rule wins; a rule matches a path or any parent."""
    parts = path.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    excluded = False
    for negated, regex in rules:
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not negated
    return not excluded


@pytest.fixture(scope="module")
def ignore_rules() -> list[tuple[bool, re.Pattern[str]]]:
    return dockerignore_rules((ROOT / ".dockerignore").read_text(encoding="utf-8"))


def test_dockerignore_is_fail_closed(ignore_rules: list[tuple[bool, re.Pattern[str]]]) -> None:
    first_negated, first = ignore_rules[0]
    assert not first_negated and first.pattern == r"[^/]*\Z"


def _repository_files() -> Iterator[str]:
    skip = {".git", ".venv", ".cache"}
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.is_file() and not (set(rel.parts) & skip):
            yield rel.as_posix()


def test_build_context_contains_exactly_the_build_inputs(
    ignore_rules: list[tuple[bool, re.Pattern[str]]],
) -> None:
    context = {p for p in _repository_files() if in_build_context(p, ignore_rules)}
    allowed_roots = ("src/", "scripts/")
    allowed_files = {"pyproject.toml", "uv.lock", "README.md", ".env.example"}
    assert {"pyproject.toml", "uv.lock", "README.md", "scripts/locustfile.py"} <= context
    assert any(p.startswith("src/high_throughput_vllm_deployment/") for p in context)
    stray = {p for p in context if p not in allowed_files and not p.startswith(allowed_roots)}
    assert not stray, stray
    assert not any(p.endswith((".pyc", ".pyo")) or "__pycache__" in p for p in context)


@pytest.mark.parametrize(
    ("path", "included"),
    [
        ("pyproject.toml", True),
        ("src/high_throughput_vllm_deployment/__init__.py", True),
        ("scripts/audit_hygiene.sh", True),
        (".env.example", True),
        (".env", False),
        (".env.production", False),
        (".cache/uv/wheel.whl", False),
        (".venv/bin/python", False),
        (".git/HEAD", False),
        ("tests/test_docker.py", False),
        ("dist/pkg.whl", False),
        ("reports/locust.json", False),
        ("compose.yaml", False),
        ("Dockerfile", False),
        ("src/high_throughput_vllm_deployment/__pycache__/x.cpython-312.pyc", False),
        ("src/high_throughput_vllm_deployment/stale.pyc", False),
        ("scripts/.DS_Store", False),
    ],
)
def test_dockerignore_decisions(
    path: str, included: bool, ignore_rules: list[tuple[bool, re.Pattern[str]]]
) -> None:
    assert in_build_context(path, ignore_rules) is included


# --------------------------------------------------------------------------- #
# compose.yaml                                                                #
# --------------------------------------------------------------------------- #

_INTERPOLATION: Final = re.compile(r"\$\$|\$\{(\w+)(?:(:?)([-?])([^}]*))?\}")


def interpolate(value: str, env: dict[str, str]) -> str:
    """Compose-spec interpolation: ${V}, ${V:-d}, ${V-d}, ${V:?e}, ${V?e}, $$."""

    def replace(match: re.Match[str]) -> str:
        if match.group(0) == "$$":
            return "$"
        name, colon, op, arg = match.groups()
        current = env.get(name)
        unset = current is None or (colon == ":" and current == "")
        if op == "-" and unset:
            return arg
        if op == "?" and unset:
            raise ValueError(arg or f"{name} is required")
        return current or ""

    return _INTERPOLATION.sub(replace, value)


def render(node: Any, env: dict[str, str]) -> Any:
    if isinstance(node, str):
        return interpolate(node, env)
    if isinstance(node, list):
        return [render(item, env) for item in node]
    if isinstance(node, dict):
        return {key: render(value, env) for key, value in node.items()}
    return node


def env_example() -> dict[str, str]:
    pairs = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
    return pairs


@pytest.fixture(scope="module")
def compose_raw() -> dict[str, Any]:
    document = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


@pytest.fixture(scope="module", params=["defaults", "env-example"])
def compose(request: pytest.FixtureRequest, compose_raw: dict[str, Any]) -> dict[str, Any]:
    env = {} if request.param == "defaults" else env_example()
    rendered = render(compose_raw, env)
    assert isinstance(rendered, dict)
    return rendered


def test_interpolation_semantics() -> None:
    env = {"SET": "x", "EMPTY": ""}
    assert (
        interpolate("${SET:-d} ${EMPTY:-d} ${EMPTY-d} ${MISSING-d} ${SET} $$1", env)
        == "x d  d x $1"
    )
    with pytest.raises(ValueError, match="need it"):
        interpolate("${MISSING:?need it}", env)
    with pytest.raises(ValueError, match="EMPTY is required"):
        interpolate("${EMPTY:?}", env)
    assert interpolate("${MISSING}", env) == ""


def test_services_are_profile_gated(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert set(services) == {"vllm-server", "vllm-mock", "locust-loadgen"}
    profiles = {name: service.get("profiles") for name, service in services.items()}
    assert profiles == {
        "vllm-server": ["gpu"],
        "vllm-mock": ["mock"],
        "locust-loadgen": ["loadtest"],
    }


def test_published_ports_bind_to_loopback(compose: dict[str, Any]) -> None:
    published = 0
    for name, service in compose["services"].items():
        for mapping in service.get("ports", []):
            host_ip, host_port, container_port = str(mapping).split(":")
            assert host_ip == "127.0.0.1", f"{name} publishes on {host_ip}"
            assert 1 <= int(host_port) <= 65535 and container_port == "8000"
            published += 1
    assert published == 2
    host_ports = [s["ports"][0].split(":")[1] for s in compose["services"].values() if "ports" in s]
    assert len(set(host_ports)) == 2, "gpu and mock targets must be able to run side by side"


def test_vllm_server_arguments_mirror_env_example(compose_raw: dict[str, Any]) -> None:
    command = compose_raw["services"]["vllm-server"]["command"]
    flags = dict(zip(command[::2], command[1::2], strict=True))
    example = env_example()
    for flag, variable in (
        ("--model", "VLLM_MODEL"),
        ("--gpu-memory-utilization", "VLLM_GPU_MEMORY_UTILIZATION"),
        ("--max-num-batched-tokens", "VLLM_MAX_NUM_BATCHED_TOKENS"),
        ("--max-model-len", "VLLM_MAX_MODEL_LEN"),
        ("--tensor-parallel-size", "VLLM_TENSOR_PARALLEL_SIZE"),
    ):
        match = re.fullmatch(r"\$\{(\w+):-([^}]*)\}", flags[flag])
        assert match is not None, f"{flag} must be ${{VAR:-default}}"
        assert match.group(1) == variable
        assert match.group(2) == example[variable], (
            f"default of {variable} drifted from .env.example"
        )
    assert flags["--host"] == "0.0.0.0" and flags["--port"] == "8000"  # noqa: S104 - asserting the container's bind address


def test_vllm_server_reserves_gpus_and_avoids_host_mounts(compose: dict[str, Any]) -> None:
    server = compose["services"]["vllm-server"]
    assert server["image"].startswith("vllm/vllm-openai:")
    (device,) = server["deploy"]["resources"]["reservations"]["devices"]
    assert device["capabilities"] == ["gpu"] and device["driver"] == "nvidia"
    assert device["count"] == "all"
    assert server["volumes"] == ["hf-cache:/models/huggingface"]
    assert server["environment"]["HF_HOME"] == "/models/huggingface"
    assert "no-new-privileges:true" in server["security_opt"]


def test_vllm_server_healthcheck(compose: dict[str, Any], compose_raw: dict[str, Any]) -> None:
    check = compose["services"]["vllm-server"]["healthcheck"]
    assert (check["interval"], check["timeout"], check["retries"]) == ("10s", "5s", 3)
    assert check["start_period"] == "30s"
    assert check["test"][:3] == ["CMD", "python3", "-c"]
    assert "curl" not in check["test"][3] and "wget" not in check["test"][3]


def test_healthcheck_code_is_identical_in_dockerfile_and_compose(
    dockerfile: Any, compose_raw: dict[str, Any]
) -> None:
    _, stages, _ = dockerfile
    compose_code = compose_raw["services"]["vllm-server"]["healthcheck"]["test"][3]
    assert _healthcheck_argv(stages["runtime"])[2] == compose_code


def test_no_service_inherits_the_dotenv_file(compose_raw: dict[str, Any]) -> None:
    for name, service in compose_raw["services"].items():
        assert "env_file" not in service, name
        assert "VLLM_PORT" not in service.get("environment", {}), name


def test_app_services_are_hardened_and_built_from_the_runtime_stage(
    compose: dict[str, Any],
) -> None:
    for name in ("vllm-mock", "locust-loadgen"):
        service = compose["services"][name]
        assert service["build"] == {"context": ".", "dockerfile": "Dockerfile", "target": "runtime"}
        assert service["user"] == "10001:10001"
        assert service["read_only"] is True and service["init"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert service["tmpfs"] == ["/tmp:size=64m"]  # noqa: S108 - asserting the container tmpfs mount
        assert service["image"].startswith("high-throughput-vllm-deployment:")


def test_mock_and_loadgen_target_the_same_model(compose: dict[str, Any]) -> None:
    services = compose["services"]
    mock_command = services["vllm-mock"]["command"]
    mock_model = mock_command[mock_command.index("--model") + 1]
    load_args = dict(a.split("=", 1) for a in services["locust-loadgen"]["command"] if "=" in a)
    assert load_args["--vllm-model"] == mock_model == services["vllm-server"]["command"][1]


def test_loadgen_runs_headless_and_writes_only_to_reports(compose: dict[str, Any]) -> None:
    loadgen = compose["services"]["locust-loadgen"]
    assert loadgen["entrypoint"] == ["locust", "-f", "/app/locustfile.py"]
    command = loadgen["command"]
    assert command[0] == "--headless"
    args = dict(arg.split("=", 1) for arg in command[1:] if "=" in arg)
    assert (args["--users"], args["--spawn-rate"], args["--run-time"]) == ("50", "10", "60s")
    assert args["--host"] == "http://vllm-mock:8000"
    for key in ("--vllm-benchmark-output", "--csv", "--html"):
        assert args[key].startswith("reports/")
    assert loadgen["working_dir"] == "/app"
    assert loadgen["volumes"] == ["./reports:/app/reports:rw"]
    depends = loadgen["depends_on"]
    assert all(d == {"condition": "service_healthy", "required": False} for d in depends.values())


def test_volumes_are_contained(compose_raw: dict[str, Any]) -> None:
    named = set(compose_raw.get("volumes", {}))
    for name, service in compose_raw["services"].items():
        for volume in service.get("volumes", []):
            source = volume.split(":", 1)[0]
            if source.startswith((".", "/", "~", "$")):
                assert source.startswith("./"), f"{name}: bind source {source!r} must be relative"
                resolved = (ROOT / source).resolve()
                assert resolved.is_relative_to(ROOT), f"{name}: {source!r} escapes the project"
            else:
                assert source in named, f"{name}: undeclared named volume {source!r}"
    assert (ROOT / ".gitignore").read_text().count("\nreports/\n") == 1


# --------------------------------------------------------------------------- #
# Executable vectors                                                          #
# --------------------------------------------------------------------------- #


class _Unhealthy(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        self.send_response(503)
        self.end_headers()

    def log_message(self, *_args: Any) -> None:
        pass


def _run_vector(argv: list[str], url: str) -> int:
    command = [sys.executable, *argv[1:]]  # the image's interpreter -> ours
    env = os.environ | {"HEALTHCHECK_URL": url}
    completed = subprocess.run(command, env=env, capture_output=True, timeout=30, check=False)  # noqa: S603
    return completed.returncode


@pytest.fixture(scope="module")
def health_vectors(dockerfile: Any, compose_raw: dict[str, Any]) -> list[list[str]]:
    _, stages, _ = dockerfile
    compose_vector = compose_raw["services"]["vllm-server"]["healthcheck"]["test"][1:]
    return [_healthcheck_argv(stages["runtime"]), compose_vector]


def test_healthcheck_vectors_pass_against_a_healthy_server(health_vectors: list[list[str]]) -> None:
    fast = MockEngineConfig(decode_step_s=0.0, prefill_s_per_token=0.0)
    with ThreadedMockServer(fast, sleep=instant_sleep) as base_url:
        assert [_run_vector(v, f"{base_url}/health") for v in health_vectors] == [0, 0]


def test_healthcheck_vectors_fail_closed(health_vectors: list[list[str]]) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Unhealthy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        unhealthy = f"http://127.0.0.1:{server.server_address[1]}/health"
        assert all(_run_vector(v, unhealthy) != 0 for v in health_vectors)
    finally:
        server.shutdown()
        server.server_close()
    refused = f"http://127.0.0.1:{server.server_address[1]}/health"  # port now closed
    assert all(_run_vector(v, refused) != 0 for v in health_vectors)


def test_container_locust_shim_is_discovered_by_locust() -> None:
    user_classes, shapes = load_locustfile(str(ROOT / "scripts" / "locustfile.py"))
    assert set(user_classes) == {"VllmChatUser", "VllmFastChatUser"}
    assert shapes == []


def test_entrypoint_shuts_down_cleanly_on_sigterm(dockerfile: Any) -> None:
    """`docker stop` sends SIGTERM to PID 1; the server must exit promptly with 0."""
    _, stages, _ = dockerfile
    entrypoint = json.loads(stages["runtime"].all("ENTRYPOINT")[0].args)
    command = [sys.executable, *entrypoint[1:], "--port", "0"]
    process = subprocess.Popen(  # noqa: S603 - fixed argv
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout is not None
        banner = process.stdout.readline()
        assert "listening on http://127.0.0.1:" in banner, shlex.join(command)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:  # pragma: no cover - only on failure
            process.kill()
            process.wait()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
