#!/usr/bin/env bash

################################################################################
# MANUAL TYPE CHECK — the command it is safe to recommend to a human
################################################################################
#
# Usage:  deep_worker/scripts/type-check.sh [deep_worker_cli|deepagents]
#
#   no argument  -> check both packages
#   package name -> check only that package (fast iteration on one error)
#
# WHY THIS EXISTS
# ---------------
# When the pre-commit type check failed, it used to tell developers to run
#
#     cd .../deepagents && uv run --all-groups ty check deepagents
#
# That bare command creates a project `.venv` and leaves it behind, so the NEXT
# commit was rejected by the venv check rather than by the type error the
# developer was actually chasing. The developer was punished for following the
# tool's own printed instructions.
#
# This script does the same work through deep_worker/scripts/uv-run.sh, which
# owns the cleanup, so it is safe to recommend and safe to rerun.
#
# Exit status mirrors the type check: 0 if everything passed, non-zero otherwise.
#
################################################################################

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
UV_RUN="${SCRIPT_DIR}/uv-run.sh"
DEEP_WORKER_DIR="$(dirname "$SCRIPT_DIR")"

VALID_PACKAGES=("deep_worker_cli" "deepagents")

usage() {
    echo "Usage: $(basename "$0") [${VALID_PACKAGES[*]}]" >&2
}

if [ "$#" -gt 1 ]; then
    echo -e "${RED}ERROR: expected at most one package argument, got $#${NC}" >&2
    usage
    exit 2
fi

REQUESTED="${1:-}"
if [ -n "$REQUESTED" ]; then
    valid=0
    for pkg in "${VALID_PACKAGES[@]}"; do
        [ "$REQUESTED" = "$pkg" ] && valid=1
    done
    if [ "$valid" -ne 1 ]; then
        echo -e "${RED}ERROR: unknown package '${REQUESTED}'${NC}" >&2
        echo -e "${YELLOW}Valid packages: ${VALID_PACKAGES[*]}${NC}" >&2
        exit 2
    fi
    TARGETS=("$REQUESTED")
else
    TARGETS=("${VALID_PACKAGES[@]}")
fi

if [ ! -x "$UV_RUN" ]; then
    echo -e "${RED}ERROR: missing or non-executable ${UV_RUN}${NC}" >&2
    echo -e "${RED}That wrapper is what keeps this command from leaking .venv.${NC}" >&2
    exit 1
fi

ALL_PASSED=true
CHECKED_ANY=false

for pkg in "${TARGETS[@]}"; do
    pkg_dir="${DEEP_WORKER_DIR}/${pkg}"
    if [ ! -d "$pkg_dir" ]; then
        echo -e "${YELLOW}Skipping ${pkg} (directory not found: ${pkg_dir})${NC}"
        continue
    fi
    CHECKED_ANY=true

    echo -e "${YELLOW}Running type check on ${pkg}...${NC}"
    echo ""

    # NOTE: no `set -e`, and the failure is captured rather than propagated.
    # If the first package short-circuited the run, a developer would fix one
    # error, rerun, and only then discover a second one that was there all
    # along — one round trip per error. Check everything, aggregate at the end.
    if (cd "$pkg_dir" && "$UV_RUN" --all-groups ty check "$pkg"); then
        echo ""
        echo -e "${GREEN}✓ ${pkg} type check passed!${NC}"
    else
        echo ""
        echo -e "${RED}✗ ${pkg} TYPE CHECK FAILED${NC}"
        ALL_PASSED=false
    fi
    echo ""
done

if [ "$CHECKED_ANY" = false ]; then
    echo -e "${YELLOW}Warning: no packages found under ${DEEP_WORKER_DIR}${NC}" >&2
    echo -e "${YELLOW}Not a deep_worker repository? Nothing to check.${NC}" >&2
    exit 0
fi

if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✓ All type checks passed!${NC}"
    exit 0
fi

echo -e "${RED}✗ TYPE CHECK FAILED${NC}"
exit 1
