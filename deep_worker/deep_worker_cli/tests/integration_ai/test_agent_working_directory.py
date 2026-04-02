"""Test to verify that agent's working directory is correctly set to /tmp.

This test verifies that when a deep-worker session is started with --dir /tmp,
the agent's current working directory is actually /tmp, not some other location.

The test:
1. Changes to /tmp directory
2. Starts a clean test agent with working_dir=/tmp
3. Opens dwcli session
4. Asks agent "What is your current working directory?"
5. Uses AI screenshot validation to check if response mentions /tmp
6. Fails if agent reports different working directory

Run with:
    cd /tmp && pytest deep_worker/deep_worker_cli/tests/integration_ai/test_agent_working_directory.py -v -s
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401
from .terminal_emulator import (
    DwcliTerminalSession,
    create_terminal_snapshot,
)

# Try to import check_reply, but don't fail if claude_sdk is unavailable
try:
    from .check_reply import verify_screenshot_contains_sync
    CHECK_REPLY_AVAILABLE = True
except ImportError:
    CHECK_REPLY_AVAILABLE = False
    logging.warning("verify_screenshot_contains_sync not available (claude_sdk missing). AI validation will be skipped.")

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
@pytest.mark.skipif(
    not CHECK_REPLY_AVAILABLE,
    reason="claude_sdk not available (required for AI validation)",
)
def test_agent_working_directory_is_tmp(clean_agent):
    """Test that agent's working directory is /tmp when started with --dir /tmp.

    This test verifies CRITICAL behavior:
    1. Changes current directory to /tmp
    2. Starts a test agent with working_dir=/tmp
    3. Asks agent via dwcli: "What is your current working directory?"
    4. Uses AI validation to verify response contains "/tmp"
    5. CRITICAL: Verifies NO error messages are displayed in TUI
    6. CRITICAL: Verifies that tool call (execute("pwd")) is VISIBLE in TUI
    7. FAILS if agent reports different working directory
    8. FAILS if "Error" message appears in terminal output
    9. FAILS if tool call is NOT displayed in terminal output

    PASS CRITERIA:
    - Agent responds that its current working directory is /tmp
    - No "Error" messages displayed in TUI
    - Tool call display is visible (e.g., "execute", "pwd", or tool invocation marker)

    FAIL CRITERIA:
    - Agent responds with different directory (e.g., a sibling repo checkout)
    - Agent cannot determine its working directory
    - "Error" message found in terminal output
    - Tool call is NOT visible in terminal output (silent tool execution)
    """
    logger.info("=" * 80)
    logger.info("TEST: Agent working directory verification (/tmp)")
    logger.info("=" * 80)

    # Step 1: Change to /tmp directory
    original_cwd = os.getcwd()
    logger.info(f"Original working directory: {original_cwd}")

    os.chdir("/tmp")
    logger.info(f"Changed to: {os.getcwd()}")

    try:
        agent_name = "test_agent_working_directory"
        logger.info(f"Starting clean test agent: {agent_name}")

        # Step 2: Use clean_agent fixture to start agent with working_dir=/tmp
        # This ensures agent starts with NO history and working directory set to /tmp
        agent_process = clean_agent(agent_name, working_dir="/tmp")
        logger.info(f"✓ Clean agent started (PID: {agent_process.pid})")

        # Step 3: Open dwcli session
        session = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            # Start dwcli
            logger.info(f"Starting dwcli for agent: {agent_name}")
            session.start(timeout=15.0)
            logger.info(f"✓ dwcli started (PID: {session.pid if session.pid else 'unknown'})")

            # Wait for UI to render
            time.sleep(2.0)
            logger.info("✓ UI rendered")

            # Step 4: Ask agent about its current working directory
            query = "What is your current working directory?"
            logger.info(f"Sending query: '{query}'")

            session.send_input(query, press_enter=False)
            time.sleep(0.3)
            session.send_key("enter")
            logger.info("✓ Query sent to dwcli")

            # Wait for AI response (up to 60 seconds)
            logger.info("Waiting for AI response...")
            response_received = False

            for i in range(120):  # 120 * 0.5s = 60 seconds max
                time.sleep(0.5)
                screen_current = session.get_screen_text()

                # Check if still thinking
                if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                    if i % 20 == 0:
                        logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                    continue

                # Check for response containing /tmp
                # Agent might respond with:
                # - "/tmp"
                # - "My current working directory is /tmp"
                # - "I'm in /tmp"
                # - etc.
                if "/tmp" in screen_current:
                    response_received = True
                    logger.info(f"✓ AI response received after {i * 0.5:.1f} seconds")
                    break

                if i % 40 == 0 and i > 0:
                    logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

            if not response_received:
                logger.warning("⚠ AI did not respond with clear answer within 60 seconds")

            # Wait for screen to stabilize
            logger.info("✓ Processing complete, waiting for screen to stabilize...")
            time.sleep(2.0)
            screen_final = session.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_final,
                title="FINAL SCREEN (should show agent's working directory is /tmp)",
            ))

            # Step 5: CRITICAL CHECK - Verify no errors displayed in TUI
            logger.info("\n" + "=" * 80)
            logger.info("CRITICAL CHECK: Error message verification")
            logger.info("=" * 80)

            if "Error" in screen_final:
                logger.error("\n" + "=" * 80)
                logger.error("❌ CRITICAL FAILURE: Error message found in TUI output!")
                logger.error("=" * 80)
                logger.error("\n" + create_terminal_snapshot(
                    screen_final,
                    title="FAILED - Error message displayed in TUI",
                ))

                pytest.fail(
                    f"Error message verification FAILED!\\n\\n"
                    f"Expected: No 'Error' messages in TUI output\\n"
                    f"Actual: 'Error' found in terminal output\\n\\n"
                    f"The TUI should not display error messages during normal operation.\\n\\n"
                    f"Screen content length: {len(screen_final)} chars"
                )

            logger.info("✓ No error messages found in TUI output")

            # Step 6: CRITICAL CHECK - Verify tool call is displayed in TUI
            logger.info("\n" + "=" * 80)
            logger.info("CRITICAL CHECK: Tool call display verification")
            logger.info("=" * 80)

            # Check for tool call indicators in the terminal output
            # Possible indicators:
            # - "execute" (tool name)
            # - "pwd" (command being executed)
            # - Tool call markers/icons
            # - Any visible indication that a tool was called
            tool_call_visible = False
            tool_call_indicators = [
                "execute",  # Tool name
                "pwd",      # Command that would be called
                "Execute",  # Capitalized version
                "tool",     # Generic tool indicator
                "Tool",     # Capitalized
                "🔧",       # Tool emoji if used
                "▸",        # Arrow/marker if used
            ]

            for indicator in tool_call_indicators:
                if indicator in screen_final:
                    tool_call_visible = True
                    logger.info(f"✓ Tool call indicator found: '{indicator}'")
                    break

            if not tool_call_visible:
                logger.error("\n" + "=" * 80)
                logger.error("❌ CRITICAL FAILURE: Tool call NOT visible in TUI output!")
                logger.error("=" * 80)
                logger.error("\n" + create_terminal_snapshot(
                    screen_final,
                    title="FAILED - Tool call NOT displayed (silent tool execution)",
                ))
                logger.error("\nTool call indicators searched:")
                for indicator in tool_call_indicators:
                    logger.error(f"  - '{indicator}': NOT FOUND")

                pytest.fail(
                    f"Tool call display verification FAILED!\n\n"
                    f"Expected: Tool call (e.g., execute('pwd')) to be VISIBLE in TUI\n"
                    f"Actual: NO tool call indicators found in terminal output\n\n"
                    f"This indicates that tool invocations are happening silently,\n"
                    f"which makes it impossible for users to see what the agent is doing.\n\n"
                    f"Tool execution should be transparent and visible to users!\n\n"
                    f"Searched for indicators: {tool_call_indicators}\n"
                    f"Screen content length: {len(screen_final)} chars"
                )

            logger.info("✓ Tool call is visible in TUI output")

            # Step 7: Use AI validation to verify response
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION: Checking if agent reports /tmp as working directory")
            logger.info("=" * 80)

            validation_result = verify_screenshot_contains_sync(
                screenshot_text=screen_final,
                original_query=query,
                expected_response_contains="/tmp",
                model="haiku",  # Use cheapest model
            )

            logger.info(f"Validation result: {validation_result}")

            # Step 8: Determine test outcome
            if validation_result.is_valid:
                logger.info("\n" + "=" * 80)
                logger.info("✅ TEST PASSED: Agent correctly reports /tmp as working directory")
                logger.info("✅ Tool call is visible in TUI")
                logger.info(f"   Validation comment: {validation_result.comment}")
                logger.info("=" * 80)
            else:
                logger.error("\n" + "=" * 80)
                logger.error("❌ TEST FAILED: Agent does NOT report /tmp as working directory")
                logger.error(f"   Validation comment: {validation_result.comment}")
                logger.error("=" * 80)
                logger.error("\n" + create_terminal_snapshot(
                    screen_final,
                    title="FAILED VALIDATION - Agent working directory is NOT /tmp",
                ))

                # Check if agent reported different directory
                if "/repos" in screen_final:
                    logger.error("⚠ Agent appears to be in /repos directory instead of /tmp!")
                    logger.error("   This indicates --dir /tmp was not respected.")

                pytest.fail(
                    f"Agent working directory verification failed!\n"
                    f"Expected: Agent to report /tmp as current working directory\n"
                    f"Validation: {validation_result.comment}\n"
                    f"Query: '{query}'\n"
                    f"Agent should have responded that its current directory is /tmp, but validation failed."
                )

        finally:
            session.close()

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {os.getcwd()}")

    # Note: Agent cleanup is handled by clean_agent fixture


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
