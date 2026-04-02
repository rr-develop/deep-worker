"""TDD test for LocalContext working directory bug.

This test verifies that LocalContextMiddleware uses working_dir from state
instead of Path.cwd(), which returns the process's current directory.

BUG: When agent is started with --dir /tmp, but the process's cwd is /,
LocalContext shows "/" instead of "/tmp".

ROOT CAUSE: LocalContextMiddleware._get_file_list() and other methods use
Path.cwd() instead of reading working_dir from runtime.state.

This test MUST FAIL before the fix and PASS after the fix.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401

# Try to import check_reply, but don't fail if claude_sdk is unavailable
try:
    from .check_reply import verify_screenshot_contains_sync
    CHECK_REPLY_AVAILABLE = True
except ImportError:
    CHECK_REPLY_AVAILABLE = False
    logging.warning("verify_screenshot_contains_sync not available")

logger = logging.getLogger(__name__)


def _check_dwcli_available() -> bool:
    """Check if dwcli command is available."""
    import subprocess
    try:
        result = subprocess.run(
            ["which", "dwcli"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_local_context_uses_working_dir_from_state(clean_agent):
    """TDD TEST: LocalContext MUST use working_dir from state, NOT Path.cwd().

    This test verifies the fix for the bug where LocalContext shows the process's
    cwd (usually "/") instead of the configured working_dir ("/tmp").

    CRITICAL REGRESSION PREVENTION:
    ================================
    LocalContextMiddleware MUST ALWAYS use working_dir from runtime.state,
    NEVER use Path.cwd() directly!

    Path.cwd() returns the process's current directory, which is often "/"
    for daemon processes, NOT the user's intended working directory.

    TEST APPROACH (TDD):
    ====================
    1. Start agent with working_dir=/tmp
    2. Ask agent about its current directory
    3. Check that agent reports "/tmp" NOT "/"

    PASS CRITERIA:
    - Agent's response includes "/tmp" as current directory
    - Agent does NOT report "/" as current directory

    FAIL CRITERIA:
    - Agent reports "/" instead of "/tmp"
    - Local context shows wrong directory
    """
    logger.info("=" * 80)
    logger.info("TDD TEST: LocalContext working_dir bug")
    logger.info("=" * 80)

    # Change to /tmp
    original_cwd = os.getcwd()
    os.chdir("/tmp")
    logger.info(f"Changed to: {os.getcwd()}")

    try:
        agent_name = "test_lc_wd"
        logger.info(f"Starting clean test agent: {agent_name}")

        # Start agent with working_dir=/tmp
        agent_process = clean_agent(agent_name, working_dir="/tmp")
        logger.info(f"✓ Agent started (PID: {agent_process.pid})")

        # Give agent time to initialize
        time.sleep(5)

        # Use dwcli to ask about directory (simpler than API)
        import subprocess
        logger.info("Asking agent about its current directory via dwcli...")

        result = subprocess.run(
            ["dwcli", agent_name, "-p", "What is your current working directory?"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/tmp",
        )

        response = result.stdout.strip()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Agent response:")
        logger.info(f"{'=' * 80}")
        logger.info(response)
        logger.info(f"{'=' * 80}\n")

        # CRITICAL ASSERTIONS
        # ===================

        # Check 1: Response should mention /tmp
        if "/tmp" not in response:
            logger.error("❌ FAIL: Agent does NOT report /tmp as current directory!")
            logger.error(f"   Response: {response}")
            pytest.fail(
                "LocalContext bug NOT fixed!\n"
                f"Expected agent to report /tmp as current directory.\n"
                f"Got: {response}\n\n"
                "ROOT CAUSE: LocalContextMiddleware still uses Path.cwd() instead of working_dir from state.\n"
                "FIX: Update LocalContextMiddleware.before_agent() to read working_dir from runtime.state"
            )

        # Check 2: If response mentions root directory "/", it should be part of "/tmp"
        # NOT standalone "/"
        if " / " in response or response.endswith(" /") or response.startswith("/ "):
            # Has standalone "/" - check if it's NOT referring to /tmp
            if response.count("/") > response.count("/tmp"):
                logger.error("❌ FAIL: Agent reports root directory / instead of /tmp!")
                logger.error(f"   Response: {response}")
                pytest.fail(
                    "LocalContext bug detected!\n"
                    f"Agent reports '/' (root) instead of '/tmp'\n"
                    f"Response: {response}\n\n"
                    "This indicates LocalContextMiddleware is using Path.cwd() (process cwd)\n"
                    "instead of working_dir from runtime.state"
                )

        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST PASSED: LocalContext correctly uses /tmp from state")
        logger.info("=" * 80)

    finally:
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {os.getcwd()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
