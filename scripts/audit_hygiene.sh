#!/usr/bin/env bash
# Fail-closed repository hygiene audit.
#
# Checks:
#   1. Forbidden artifacts are not tracked/staged (bytecode, tool caches,
#      virtualenvs, build output, credentials, .env files, OS noise).
#   2. No file content embeds an absolute host path or a home-relative path.
#   3. No file content embeds a credential-shaped secret.
#   4. Cache routing invariants hold in pyproject.toml and .gitignore.
#   5. (--workspace) No stray default tool caches or bytecode exist on disk
#      outside ./.cache and ./.venv, and no cache env var points into $HOME.
#
# Usage:
#   scripts/audit_hygiene.sh [--tracked | --workspace]
#
#   --tracked     Audit files in the git index only (default; CI mode).
#   --workspace   Also audit untracked, non-ignored files and on-disk state.
#
# Exit codes: 0 = clean, 1 = violation(s) found, 2 = usage / environment error.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
# `sh script` on macOS runs bash in POSIX mode, which disables process
# substitution and other bashisms used below; leave POSIX mode in-process.
set +o posix

set -euo pipefail

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

mode="tracked"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tracked) mode="tracked" ;;
    --workspace) mode="workspace" ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a git work tree" >&2
  exit 2
}
cd "$repo_root"

violations=0
fail() {
  echo "FAIL: $*" >&2
  violations=$((violations + 1))
}

# Absolute host paths (macOS/Linux homes, mounted volumes, per-user temp) and
# home-relative paths. Written so the pattern text never matches itself.
readonly ABS_PATH_RE='/(Users|home|Volumes|private/var|var/folders|root)/|[A-Za-z]:[\\/]+Users[\\/]|(^|[[:space:]"'"'"'=(])~/|\$\{?HOME\}?/'
# Credential-shaped tokens.
readonly SECRET_RE='AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|(^|[^[:alnum:]_-])sk-(ant-)?[A-Za-z0-9_-]{20,}|(^|[^[:alnum:]_])hf_[A-Za-z0-9]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}'

# --------------------------------------------------------------------------- #
# Collect candidate files                                                     #
# --------------------------------------------------------------------------- #
files=()
if [ "$mode" = "workspace" ]; then
  list_cmd=(git ls-files -z --cached --others --exclude-standard)
else
  list_cmd=(git ls-files -z --cached)
fi
while IFS= read -r -d '' f; do
  files+=("$f")
done < <("${list_cmd[@]}")

if [ "${#files[@]}" -eq 0 ]; then
  echo "error: no files to audit in ${mode} mode (nothing tracked or staged?)" >&2
  exit 2
fi

# --------------------------------------------------------------------------- #
# 1. Forbidden artifacts                                                      #
# --------------------------------------------------------------------------- #
for f in "${files[@]}"; do
  base="${f##*/}"
  case "/$f" in
    */__pycache__/* | *.pyc | *.pyo | *.pyd)
      fail "bytecode artifact: $f" ;;
    /.cache/* | */.venv/* | /venv/* | */.mypy_cache/* | */.ruff_cache/* | */.pytest_cache/* \
      | */.hypothesis/* | */htmlcov/* | */.tox/* | */.nox/*)
      fail "tool cache / virtualenv artifact: $f" ;;
    */dist/* | */build/* | *.egg-info/* | *.whl)
      fail "build artifact: $f" ;;
  esac
  case "$base" in
    .coverage | .coverage.* | coverage.xml)
      fail "coverage artifact: $f" ;;
    .env.example) ;;
    .env | .env.*)
      fail "environment file with potential secrets: $f" ;;
    *.pem | *.key | *.p12 | *.pfx | *.jks | *.kdbx | id_rsa* | id_dsa* | id_ecdsa* | id_ed25519* \
      | .netrc | .pypirc | .npmrc | *credentials* | *secret*.json | *.keytab)
      fail "credential artifact: $f" ;;
    .DS_Store | Thumbs.db)
      fail "OS metadata artifact: $f" ;;
  esac
done

# --------------------------------------------------------------------------- #
# 2 + 3. Content scans                                                        #
# --------------------------------------------------------------------------- #
existing=()
for f in "${files[@]}"; do
  [ -f "$f" ] && existing+=("$f")
done

scan() {
  local label="$1" pattern="$2" out rc
  [ "${#existing[@]}" -gt 0 ] || return 0
  set +e
  out="$(grep -nIHE -e "$pattern" -- "${existing[@]}")"
  rc=$?
  set -e
  case "$rc" in
    0)
      while IFS= read -r line; do
        fail "${label}: ${line}"
      done <<<"$out"
      ;;
    1) ;;
    *) echo "error: grep failed (exit ${rc}) while scanning for ${label}" >&2; exit 2 ;;
  esac
}

scan "absolute/home host path" "$ABS_PATH_RE"
scan "credential-shaped secret" "$SECRET_RE"

# --------------------------------------------------------------------------- #
# 4. Cache routing invariants                                                 #
# --------------------------------------------------------------------------- #
# Prints the raw (unquoted) value of `key` inside TOML table `[section]`.
toml_get() {
  awk -v section="[$1]" -v key="$2" '
    /^[[:space:]]*\[/ { in_section = (index($0, section) == 1); next }
    in_section {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      if (index(line, key) == 1) {
        rest = substr(line, length(key) + 1)
        if (rest ~ /^[[:space:]]*=/) {
          sub(/^[[:space:]]*=[[:space:]]*/, "", rest)
          sub(/[[:space:]]*(#.*)?$/, "", rest)
          gsub(/^"|"$/, "", rest)
          print rest
          exit
        }
      }
    }
  ' pyproject.toml
}

if [ ! -f pyproject.toml ]; then
  fail "pyproject.toml missing"
else
  while IFS='|' read -r section key expected; do
    actual="$(toml_get "$section" "$key")"
    [ "$actual" = "$expected" ] \
      || fail "cache routing: [${section}] ${key} = '${actual}', expected '${expected}'"
  done <<'EOF'
tool.uv|cache-dir|./.cache/uv
tool.ruff|cache-dir|./.cache/ruff
tool.mypy|cache_dir|./.cache/mypy
tool.pytest.ini_options|cache_dir|./.cache/pytest
tool.coverage.run|data_file|./.cache/coverage/.coverage
EOF
fi

if [ ! -f .gitignore ] || ! grep -Eq '^/?\.cache/?$' .gitignore; then
  fail ".gitignore must ignore the routed cache root '.cache/'"
fi

# --------------------------------------------------------------------------- #
# 5. Workspace-only: on-disk state and environment routing                    #
# --------------------------------------------------------------------------- #
if [ "$mode" = "workspace" ]; then
  while IFS= read -r -d '' stray; do
    fail "stray cache/bytecode outside ./.cache: ${stray#./}"
  done < <(
    find . \
      \( -path ./.git -o -path ./.venv -o -path ./.cache \) -prune -o \
      \( -name .mypy_cache -o -name .ruff_cache -o -name .pytest_cache -o -name .hypothesis \
         -o -name htmlcov -o -name .coverage -o -name '.coverage.*' -o -name __pycache__ \
         -o -name '*.pyc' -o -name '*.pyo' \) -print0
  )

  home="${HOME%/}"
  under_home() {
    local p="$1"
    case "$p" in /*) ;; *) p="${repo_root}/${p}" ;; esac
    [ -n "$home" ] && case "$p/" in "$home"/*) return 0 ;; esac
    return 1
  }

  for var in UV_CACHE_DIR UV_PROJECT_ENVIRONMENT RUFF_CACHE_DIR MYPY_CACHE_DIR \
    PYTHONPYCACHEPREFIX XDG_CACHE_HOME HYPOTHESIS_STORAGE_DIRECTORY COVERAGE_FILE; do
    eval "val=\${${var}:-}"
    if [ -n "$val" ] && under_home "$val"; then
      fail "environment routing: ${var} points into \$HOME (${val})"
    fi
  done

  if command -v uv >/dev/null 2>&1; then
    uv_cache="$(uv cache dir 2>/dev/null || true)"
    if [ -z "$uv_cache" ]; then
      fail "cache routing: unable to determine effective uv cache dir"
    elif under_home "$uv_cache"; then
      fail "cache routing: effective uv cache dir is inside \$HOME (${uv_cache})"
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# Verdict                                                                     #
# --------------------------------------------------------------------------- #
if [ "$violations" -gt 0 ]; then
  echo "audit_hygiene: ${violations} violation(s) in ${mode} mode (${#files[@]} file(s))" >&2
  exit 1
fi
echo "audit_hygiene: OK (${mode} mode; ${#files[@]} file(s) audited)"
