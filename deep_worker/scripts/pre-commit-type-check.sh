#!/usr/bin/env bash

################################################################################
# TYPE CHECKING PRE-COMMIT HOOK FOR deep_worker_cli AND deepagents
################################################################################
#
# ⚠️  CRITICAL: DO NOT DELETE, DISABLE, OR BYPASS THIS CHECK ⚠️
#
# This hook enforces strict type checking on both deep_worker_cli and deepagents
# packages using the 'ty' type checker. All code must pass type checking before commit.
#
# VIOLATION OF THIS POLICY IS STRICTLY PROHIBITED:
# - DO NOT use 'git commit --no-verify' to bypass this check
# - DO NOT delete or rename this hook
# - DO NOT modify this script to skip type checking
# - DO NOT add 'type: ignore' comments without proper justification
#
# Type safety is a fundamental requirement of this codebase. Any attempt
# to bypass type checking compromises code quality and maintainability.
#
# If type checking fails, FIX THE TYPE ERRORS - do not bypass the check.
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"
DEEP_WORKER_DIR="${REPO_ROOT}/deep_worker"
DEEP_WORKER_CLI_DIR="${DEEP_WORKER_DIR}/deep_worker_cli"
DEEPAGENTS_DIR="${DEEP_WORKER_DIR}/deepagents"

# Check if we're in the correct repository structure
if [ ! -d "${DEEP_WORKER_CLI_DIR}" ] && [ ! -d "${DEEPAGENTS_DIR}" ]; then
    echo -e "${YELLOW}Warning: neither deep_worker_cli nor deepagents directories found${NC}"
    echo -e "${YELLOW}Skipping type check (not a deep_worker repository)${NC}"
    exit 0
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}ERROR: 'uv' command not found${NC}"
    echo -e "${RED}Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    echo -e "${RED}Then add it to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    exit 1
fi

ALL_PASSED=true

# Check deep_worker_cli if it exists
if [ -d "${DEEP_WORKER_CLI_DIR}" ]; then
    echo -e "${YELLOW}Running type check on deep_worker_cli...${NC}"
    echo ""
    cd "${DEEP_WORKER_CLI_DIR}"

    if uv run --all-groups ty check deep_worker_cli; then
        echo ""
        echo -e "${GREEN}✓ deep_worker_cli type check passed!${NC}"
    else
        echo ""
        echo -e "${RED}✗ deep_worker_cli TYPE CHECK FAILED${NC}"
        ALL_PASSED=false
    fi
    echo ""
fi

# Check deepagents if it exists
if [ -d "${DEEPAGENTS_DIR}" ]; then
    echo -e "${YELLOW}Running type check on deepagents...${NC}"
    echo ""
    cd "${DEEPAGENTS_DIR}"

    if uv run --all-groups ty check deepagents; then
        echo ""
        echo -e "${GREEN}✓ deepagents type check passed!${NC}"
    else
        echo ""
        echo -e "${RED}✗ deepagents TYPE CHECK FAILED${NC}"
        ALL_PASSED=false
    fi
    echo ""
fi

# Final result
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✓ All type checks passed! All types are valid.${NC}"
    exit 0
else
    echo -e "${RED}✗ TYPE CHECK FAILED${NC}"
    echo ""
    echo -e "${RED}Your commit has been rejected due to type errors.${NC}"
    echo ""
    echo -e "${YELLOW}To fix this issue:${NC}"
    echo -e "  1. Review the type errors shown above"
    echo -e "  2. Fix the type errors in your code"
    echo -e "  3. Run type checks manually to verify:"
    if [ -d "${DEEP_WORKER_CLI_DIR}" ]; then
        echo -e "     - cd ${DEEP_WORKER_CLI_DIR} && uv run --all-groups ty check deep_worker_cli"
    fi
    if [ -d "${DEEPAGENTS_DIR}" ]; then
        echo -e "     - cd ${DEEPAGENTS_DIR} && uv run --all-groups ty check deepagents"
    fi
    echo -e "  4. Try committing again"
    echo ""
    echo -e "${RED}⚠️  DO NOT bypass this check with --no-verify${NC}"
    echo -e "${RED}⚠️  Type safety is mandatory for this codebase${NC}"
    echo ""
    exit 1
fi
