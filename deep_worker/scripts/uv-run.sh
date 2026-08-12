#!/usr/bin/env bash

################################################################################
# CLEANUP-SAFE `uv run` SEAM
################################################################################
#
# Usage:  deep_worker/scripts/uv-run.sh <uv-run-args...>
#         (run from the project root you want the command to execute in)
#
# WHY THIS EXISTS
# ---------------
# The pre-commit hook hard-blocks any `.venv` in the working tree. The
# rationale is a past incident: a stale `rpyc` installed inside a
# `.venv` shadowed a sibling project for every running agent, leaking 21 GB RSS
# and costing a day of debugging.
#
# But `ty check` / `pytest` without project dependencies are useless (170+
# unresolved-import diagnostics), and installing langchain systemwide is also
# forbidden. `uv run` needs a project venv. The compromise the repo already made
# for the pre-commit type check is: create the venv, then ALWAYS delete it.
#
# This script generalizes that compromise so every OTHER repository-provided
# command (both package Makefiles, and the advice printed to humans) can be
# cleanup-safe too. Before it existed, `make test` and `make type` leaked a
# `.venv` on every run — see the pre-commit hook, which names exactly that
# as a contributing cause of the incident.
#
# WHAT MAKES IT SAFE
# ------------------
#   1. Serialized per project root with flock, so two invocations cannot both
#      believe they own the same `.venv`.
#   2. A pre-existing `.venv` is NEVER deleted — it is a hard stop. We only
#      remove an environment this invocation created.
#   3. The payload runs in its own process group; on interrupt we KILL that
#      whole group unconditionally BEFORE cleaning, so no surviving grandchild
#      can recreate `.venv` after we are done.
#   4. Cleanup runs from a trap on EXIT/INT/TERM — every exit path.
#
# DO NOT remove the trap, the lock, or the group kill. Each one is load-bearing;
# see deep_worker/scripts/DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md for the failure
# each prevents.
#
# This is NOT a licence to use `uv run` elsewhere. Bare invocations remain
# forbidden and are caught by tests/test_no_bare_uv_invocations.py.
#
################################################################################

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

die() {
    echo -e "${RED}ERROR (uv-run.sh): $*${NC}" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [ "$#" -eq 0 ]; then
    die "no command given. Usage: uv-run.sh <uv-run-args...>"
fi

command -v uv >/dev/null 2>&1 || die "'uv' command not found in PATH"
command -v flock >/dev/null 2>&1 || die "'flock' command not found in PATH"

# ---------------------------------------------------------------------------
# Resolve the project root: the nearest ancestor with a pyproject.toml, which
# is the directory `uv` itself would treat as the project.
# ---------------------------------------------------------------------------
resolve_project_root() {
    local dir
    dir="$(pwd -P)"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/pyproject.toml" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_ROOT="$(resolve_project_root)" \
    || die "no pyproject.toml found in $(pwd -P) or any parent; not a uv project"
VENV_PATH="${PROJECT_ROOT}/.venv"

# ---------------------------------------------------------------------------
# Lock location.
#
# The lock must live OUTSIDE the project tree, or it would itself be caught by
# the `.venv` scan. A predictable path under a shared /tmp is a classic
# symlink/pre-creation target, so every component we create or open is validated
# for type, ownership and mode — not just the deepest one. `TMPDIR` is
# attacker-controllable and the per-UID component under it is predictable, so
# trusting the parent while checking only the child leaves the same hole one
# level up.
#
# Validation failure is a hard stop with the offending path named: we do not
# repair, chmod, or silently relocate. A foreign-owned lock path means something
# is wrong that a human must see.
# ---------------------------------------------------------------------------
EUID_NUM="$(id -u)"

# Returns 0 if $1 is a real directory (not a symlink), owned by us, with no
# group/world permission bits.
dir_is_private_and_ours() {
    local path="$1"
    [ -L "$path" ] && return 1
    [ -d "$path" ] || return 1
    local owner mode
    owner="$(stat -c '%u' "$path" 2>/dev/null)" || return 1
    mode="$(stat -c '%a' "$path" 2>/dev/null)" || return 1
    [ "$owner" = "$EUID_NUM" ] || return 1
    [ "$mode" = "700" ] || return 1
    return 0
}

# Create if absent, then validate. Aborts on anything suspicious.
ensure_private_dir() {
    local path="$1"
    if [ -e "$path" ] || [ -L "$path" ]; then
        if [ -L "$path" ]; then
            die "lock path is a symlink, refusing to use it: $path"
        fi
        if [ ! -d "$path" ]; then
            die "lock path exists but is not a directory: $path"
        fi
    else
        # Two wrappers can race here (distinct project roots share this
        # directory). `mkdir` losing the race is fine — what matters is that
        # the winner created it with mode 0700 and that we validate the result
        # below either way. We deliberately do NOT use `mkdir -p`: it would
        # silently accept a pre-existing directory without telling us whether
        # it came from us, and the validation is what has to be authoritative.
        mkdir -m 700 "$path" 2>/dev/null || true
        if [ ! -d "$path" ]; then
            die "cannot create lock directory: $path"
        fi
    fi
    dir_is_private_and_ours "$path" || die \
        "lock directory is not a private directory owned by uid ${EUID_NUM} (mode must be 0700): $path"
}

# XDG_RUNTIME_DIR is used only when it is absolute, an existing directory, ours,
# and not group/world-writable. Failing those checks is ORDINARY (no such dir on
# many systems), so we fall back rather than abort — unlike a hostile path we
# actually created, which is suspicious.
LOCK_BASE=""
if [ -n "${XDG_RUNTIME_DIR:-}" ] && [ "${XDG_RUNTIME_DIR:0:1}" = "/" ] \
    && [ -d "${XDG_RUNTIME_DIR}" ] && [ ! -L "${XDG_RUNTIME_DIR}" ]; then
    xdg_owner="$(stat -c '%u' "${XDG_RUNTIME_DIR}" 2>/dev/null || echo '')"
    xdg_mode="$(stat -c '%a' "${XDG_RUNTIME_DIR}" 2>/dev/null || echo '')"
    if [ "$xdg_owner" = "$EUID_NUM" ] && [ "$xdg_mode" = "700" ]; then
        LOCK_BASE="${XDG_RUNTIME_DIR}"
    fi
fi

if [ -z "$LOCK_BASE" ]; then
    LOCK_BASE="${TMPDIR:-/tmp}/deep-worker-${EUID_NUM}"
    ensure_private_dir "$LOCK_BASE"
fi

LOCK_DIR="${LOCK_BASE}/deep-worker-uv-locks"
ensure_private_dir "$LOCK_DIR"

# Full SHA-256 of the resolved root. A truncated digest could collide two
# unrelated projects onto one lock, silently serializing them against each
# other — a throughput bug that is very hard to diagnose.
ROOT_DIGEST="$(printf '%s' "$PROJECT_ROOT" | sha256sum | cut -d' ' -f1)"
LOCK_FILE="${LOCK_DIR}/${ROOT_DIGEST}"

# Acquire the lock via a helper that opens the file ATOMICALLY with O_NOFOLLOW,
# validates the OPENED FILE DESCRIPTION (fstat: regular file, our uid, mode
# 0600), takes an exclusive flock on that same fd, and only then re-executes
# this script with the lock already held.
#
# Why a helper rather than shell redirection: checking the pathname and then
# doing `exec 9>"$LOCK_FILE"` is a TOCTOU. Bash redirection follows symlinks, so
# between the check and the open another process can swap the path for a symlink
# pointing at any file we can write. Validation must happen on the fd we
# actually got, and the fd we validated must be the one we lock -- re-opening by
# name after validating re-introduces the same race. Bash cannot express
# O_NOFOLLOW, so the open, the fstat and the flock all happen in the helper, and
# the locked fd is inherited across the exec.
#
# The lock is kernel state tied to the open file description, so a persistent
# empty file across runs is expected and needs no cleanup.
# Whether we are already holding the lock is decided by INSPECTING FD 9, never
# by trusting an environment variable. An env marker is caller-controlled: with
# `_UV_RUN_LOCK_FD=9` set by hand (or simply inherited by the payload, since
# `uv` and everything it spawns inherit our environment), the resumed path would
# run `uv` with no lock at all -- and a nested wrapper for a DIFFERENT project
# would "resume" holding the wrong project's lock.
#
# So: ask the kernel whether fd 9 is the lock file for THIS project root and is
# actually flocked by us. Only that proves we may proceed.
LOCK_HELD=0
if { true >&9; } 2>/dev/null; then
    # fd 9 exists; ask the helper whether it is genuinely OUR lock on THIS file.
    if python3 "${SCRIPT_DIR}/_lock_and_exec.py" \
        --verify-held "$LOCK_FILE" "$EUID_NUM"; then
        LOCK_HELD=1
    fi
fi

if [ "$LOCK_HELD" -ne 1 ]; then
    exec python3 "${SCRIPT_DIR}/_lock_and_exec.py" \
        "$LOCK_FILE" "$EUID_NUM" "$0" "$@"
fi

# A `.venv` that predates us is not ours to delete. Fail loudly — the repo-wide
# ban exists to make exactly this visible to a human.
WE_CREATED_VENV=0
if [ -e "$VENV_PATH" ]; then
    echo -e "${RED}ERROR (uv-run.sh): a .venv already exists at ${VENV_PATH}${NC}" >&2
    echo -e "${YELLOW}This wrapper did not create it, so it will not delete it.${NC}" >&2
    echo -e "${YELLOW}Virtual environments are banned in this repository (see${NC}" >&2
    echo -e "${YELLOW}the pre-commit hook). Inspect it and remove it yourself:${NC}" >&2
    echo -e "${YELLOW}    rm -rf ${VENV_PATH}${NC}" >&2
    exit 1
fi

CHILD_PID=""
CHILD_PGID=""
CLEANED=0

# Kill the payload's ENTIRE process group, then remove the venv.
#
# Ordering matters: there is no readiness event for "every member of a process
# group has exited" (wait/pidfd observe only the direct child, and a leader can
# exit while a grandchild it spawned lives on and is reparented). So we do not
# try to detect group emptiness — we KILL the group unconditionally and only
# then clean up. ESRCH just means nobody was left, which is success.
terminate_child_group() {
    [ -n "$CHILD_PGID" ] || return 0
    kill -KILL -- "-${CHILD_PGID}" 2>/dev/null || true
    if [ -n "$CHILD_PID" ]; then
        wait "$CHILD_PID" 2>/dev/null || true
    fi
}

cleanup() {
    local rc=$?
    [ "$CLEANED" -eq 1 ] && return $rc
    CLEANED=1
    terminate_child_group
    if [ "$WE_CREATED_VENV" -eq 1 ] && [ -d "$VENV_PATH" ]; then
        rm -rf "$VENV_PATH" 2>/dev/null || true
    fi
    return $rc
}

# Forward the signal to the child group, allow a bounded grace period for a
# clean shutdown, then fall through to the unconditional group KILL in cleanup.
# The grace wait is a courtesy only; its outcome proves nothing about the rest
# of the group.
GRACE_SECONDS="${UV_RUN_GRACE_SECONDS:-5}"

on_signal() {
    local signame="$1" signo="$2"
    if [ -n "$CHILD_PGID" ]; then
        kill -"${signame}" -- "-${CHILD_PGID}" 2>/dev/null || true
        if [ -n "$CHILD_PID" ]; then
            # Bounded, poll-free: whichever finishes first wins.
            ( sleep "$GRACE_SECONDS" ) &
            local timer=$!
            wait -n "$CHILD_PID" "$timer" 2>/dev/null || true
            kill "$timer" 2>/dev/null || true
        fi
    fi
    cleanup
    # Re-raise so our exit status is the conventional 128+signo.
    trap - "$signame"
    kill -"${signame}" $$ 2>/dev/null || exit $((128 + signo))
}

trap cleanup EXIT INT TERM
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM

# `setsid` puts the payload in its own session/process group, which is what
# makes "everything this invocation started" addressable as a single unit.
WE_CREATED_VENV=1
set +e
setsid uv run "$@" &
CHILD_PID=$!
CHILD_PGID="$CHILD_PID"
wait "$CHILD_PID"
RC=$?
set -e
CHILD_PID=""

exit $RC
