# Multistage, reproducible OCI image for the benchmarking toolkit
# (MockVllmServer, standalone runner, Locust load generator).
#
#   docker build --target runtime -t high-throughput-vllm-deployment:local .
#
# Base images are pinned by digest (multi-arch OCI indexes: amd64 and arm64).
# Bump them deliberately; the pins are verified by tests/test_docker.py.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- #
# Stage 1: builder - locked dependencies + bytecode-only application          #
# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build

# Dependency layer first: it depends only on the lockfile inputs, so source
# edits never invalidate it.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group loadtest --no-install-project

# Application layer: install the project itself (non-editable), compile it to
# hash-checked bytecode (timestamp-free, hence reproducible) and strip sources.
COPY src ./src
RUN uv sync --frozen --no-dev --group loadtest --no-editable \
 && app="$(/opt/venv/bin/python -c 'import os, high_throughput_vllm_deployment as m; print(os.path.dirname(m.__file__))')" \
 && /opt/venv/bin/python -m compileall -q -b --invalidation-mode=checked-hash "$app" \
 && find "$app" -name '*.py' -delete \
 && find "$app" -name '__pycache__' -prune -exec rm -rf {} + \
 && /opt/venv/bin/python -c 'import high_throughput_vllm_deployment.benchmarks.mock_server'

# --------------------------------------------------------------------------- #
# Stage 2: runtime-rootfs - base filesystem without package managers          #
# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS runtime-rootfs

RUN groupadd --system --gid 10001 vllmops \
 && useradd --system --uid 10001 --gid vllmops --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin vllmops \
 && rm -rf /usr/bin/apt /usr/bin/apt-* /usr/bin/dpkg /usr/bin/dpkg-* /usr/lib/apt /usr/lib/dpkg \
        /usr/share/dpkg /var/lib/apt /var/lib/dpkg /var/cache/apt /var/cache/debconf /etc/apt \
        /etc/dpkg /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12 \
        /usr/local/lib/python3.12/site-packages/pip /usr/local/lib/python3.12/site-packages/pip-* \
        /usr/local/lib/python3.12/ensurepip /usr/share/doc /usr/share/man /usr/share/info \
 && find / -xdev -type f -perm /6000 -exec chmod a-s {} + \
 && ! command -v apt-get && ! command -v apt && ! command -v dpkg && ! command -v pip \
 && ! /usr/local/bin/python3.12 -c 'import pip' 2>/dev/null

COPY --from=builder /opt/venv /opt/venv
COPY scripts/locustfile.py /app/locustfile.py

# --------------------------------------------------------------------------- #
# Stage 3: runtime - flattened, non-root, no package managers                 #
# --------------------------------------------------------------------------- #
# Copying the pruned filesystem into `scratch` means the removed tools are
# physically absent (not merely hidden by whiteouts in a later layer).
FROM scratch AS runtime

COPY --from=runtime-rootfs / /

ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

LABEL org.opencontainers.image.title="high-throughput-vllm-deployment" \
      org.opencontainers.image.description="vLLM benchmarking toolkit: mock OpenAI server, runner, Locust load generator"

WORKDIR /app
RUN mkdir -p /app/reports && chown 10001:10001 /app/reports

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import os, sys, urllib.request; url = os.environ.get('HEALTHCHECK_URL', 'http://127.0.0.1:8000/health'); sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"]

ENTRYPOINT ["python", "-m", "high_throughput_vllm_deployment.benchmarks.mock_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
