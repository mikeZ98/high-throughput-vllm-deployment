#!/usr/bin/env bash
# Fail-closed commit identity audit.
#
# Asserts that every commit in the audited range was authored AND committed by
# the sole permitted identity, and that no commit message carries a trailer or
# an AI/co-author signature.
#
# Usage:
#   scripts/audit_identity.sh [--history-only] [--rev <range>]
#
#   --history-only   Audit commit history only; skip the check of the identity
#                    configured for *future* commits (git config user.*).
#   --rev <range>    Revision range passed to `git rev-list` (default: --all).
#
# Exit codes: 0 = clean, 1 = violation(s) found, 2 = usage / environment error.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
# `sh script` on macOS runs bash in POSIX mode, which disables process
# substitution and other bashisms used below; leave POSIX mode in-process.
set +o posix

set -euo pipefail

readonly EXPECTED_NAME="mikeZ98"
readonly EXPECTED_EMAIL="145975972+mikeZ98@users.noreply.github.com"

# Case-insensitive patterns that must never appear anywhere in a commit message.
readonly FORBIDDEN_MESSAGE_RE='co-authored-by|signed-off-by|claude|anthropic|generated (with|by)|🤖|(^|[^[:alnum:]])ai[- ](generated|assisted|authored)'

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

history_only=0
rev_range="--all"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --history-only) history_only=1 ;;
    --rev)
      [ "$#" -ge 2 ] || { echo "error: --rev requires a range argument" >&2; exit 2; }
      rev_range="$2"
      shift
      ;;
    --rev=*) rev_range="${1#--rev=}" ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$rev_range" ]; then
  echo "error: --rev range must not be empty" >&2
  exit 2
fi

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

# --------------------------------------------------------------------------- #
# 1. Configured identity for future commits                                   #
# --------------------------------------------------------------------------- #
if [ "$history_only" -eq 0 ]; then
  cfg_name="$(git config --get user.name || true)"
  cfg_email="$(git config --get user.email || true)"
  [ "$cfg_name" = "$EXPECTED_NAME" ] \
    || fail "git config user.name is '${cfg_name}', expected '${EXPECTED_NAME}'"
  [ "$cfg_email" = "$EXPECTED_EMAIL" ] \
    || fail "git config user.email is '${cfg_email}', expected '${EXPECTED_EMAIL}'"
  for env_var in GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL; do
    eval "env_val=\${${env_var}:-}"
    if [ -n "$env_val" ]; then
      case "$env_var" in
        *_NAME) expected="$EXPECTED_NAME" ;;
        *) expected="$EXPECTED_EMAIL" ;;
      esac
      [ "$env_val" = "$expected" ] || fail "${env_var}='${env_val}' overrides the permitted identity"
    fi
  done
  template="$(git config --get commit.template || true)"
  if [ -n "$template" ] && [ -f "$template" ] && grep -Eiq "$FORBIDDEN_MESSAGE_RE" "$template"; then
    fail "commit.template '${template}' contains a forbidden trailer/signature"
  fi
fi

# --------------------------------------------------------------------------- #
# 2. Commit history                                                           #
# --------------------------------------------------------------------------- #
commit_count=0
if [ "$rev_range" = "--all" ] && ! git rev-parse --verify --quiet HEAD >/dev/null \
  && [ -z "$(git for-each-ref --count=1 --format='%(refname)')" ]; then
  echo "note: repository has no commits yet; history audit is vacuously clean"
else
  # An explicit, unresolvable range is an error: never pass silently.
  if ! commits="$(git rev-list "$rev_range" --)"; then
    echo "error: cannot resolve revision range '${rev_range}'" >&2
    exit 2
  fi

  while IFS= read -r sha; do
    [ -n "$sha" ] || continue
    commit_count=$((commit_count + 1))
    short="${sha:0:12}"

    an="$(git show -s --format='%an' "$sha")"
    ae="$(git show -s --format='%ae' "$sha")"
    cn="$(git show -s --format='%cn' "$sha")"
    ce="$(git show -s --format='%ce' "$sha")"
    [ "$an" = "$EXPECTED_NAME" ] || fail "${short}: author name '${an}'"
    [ "$ae" = "$EXPECTED_EMAIL" ] || fail "${short}: author email '${ae}'"
    [ "$cn" = "$EXPECTED_NAME" ] || fail "${short}: committer name '${cn}'"
    [ "$ce" = "$EXPECTED_EMAIL" ] || fail "${short}: committer email '${ce}'"

    # Any git-recognised trailer at all is a violation.
    trailers="$(git show -s --format='%(trailers:unfold)' "$sha" | sed '/^[[:space:]]*$/d')"
    if [ -n "$trailers" ]; then
      while IFS= read -r line; do
        fail "${short}: trailer present: ${line}"
      done <<<"$trailers"
    fi

    # Signatures that may not be parsed as trailers (e.g. mid-body).
    body="$(git show -s --format='%B' "$sha")"
    if matches="$(printf '%s\n' "$body" | grep -Ein "$FORBIDDEN_MESSAGE_RE")"; then
      while IFS= read -r line; do
        fail "${short}: forbidden signature in message (line ${line%%:*}): ${line#*:}"
      done <<<"$matches"
    fi

    # Signed-off / GPG identities must not leak a different committer either.
    signer="$(git show -s --format='%GS' "$sha" 2>/dev/null || true)"
    if [ -n "$signer" ] && [ "$signer" != "${EXPECTED_NAME} <${EXPECTED_EMAIL}>" ]; then
      fail "${short}: signed by foreign identity '${signer}'"
    fi
  done <<<"$commits"
fi

# --------------------------------------------------------------------------- #
# Verdict                                                                     #
# --------------------------------------------------------------------------- #
if [ "$violations" -gt 0 ]; then
  echo "audit_identity: ${violations} violation(s) across ${commit_count} commit(s)" >&2
  exit 1
fi

scope="config + history"
[ "$history_only" -eq 1 ] && scope="history"
echo "audit_identity: OK (${scope}; ${commit_count} commit(s) in '${rev_range}'; identity ${EXPECTED_NAME} <${EXPECTED_EMAIL}>)"
