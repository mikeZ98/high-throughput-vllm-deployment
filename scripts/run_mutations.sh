#!/usr/bin/env bash
# Mutation-testing entry point (see scripts/mutation_runner.py for semantics).
#
# Usage:
#   scripts/run_mutations.sh [--strict] [--filter <suite|id-glob>]... [--verbose] [--list]
#
#   --strict    exit 1 unless every mutant is KILLED (timeouts/crashes never count)
#   --filter    restrict to a suite (schemas|core|benchmarks|docker) or id glob
#   --verbose   print pytest output for every mutant
#   --list      list the selected mutants and exit
#
# Requires a clean working tree (stage your changes first). Exit codes:
# 0 = done, 1 = --strict and a mutant survived, 2 = harness/preflight error.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
# `sh script` runs bash in POSIX mode on macOS; leave it.
set +o posix

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a git work tree" >&2
  exit 2
}
cd "$repo_root"

# Everything the harness and its pytest runs write stays under ./.cache.
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-./.cache/uv}"
export TMPDIR="$repo_root/.cache/tmp"
mkdir -p "$TMPDIR"

exec uv run --frozen python -B scripts/mutation_runner.py "$@"
