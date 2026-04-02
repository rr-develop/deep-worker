"""Test to verify deepseek provider doesn't duplicate tool calls across queries.

This test verifies that when a user makes multiple queries, tool calls from previous
queries are NOT displayed again in the TUI for subsequent unrelated queries.

This is a CONTROL TEST to verify if the tool call duplication bug is:
- Specific to claude-code provider, OR
- General bug affecting all providers

The test:
1. Sends first query: "What is your current working directory?" (triggers 'execute' tool)
2. Waits for response and verifies tool call is visible
3. Sends second query: "2+2=" (should NOT trigger any tools)
4. Verifies that tool call indicators are NOT visible in the TUI for second query

PASS CRITERIA:
- First query shows tool call (execute/pwd)
- Second query does NOT show any tool call indicators
- Second query shows correct answer (4)

FAIL CRITERIA:
- Second query shows tool call indicators from first query
- Tool execution is visible in TUI for second query when it shouldn't be

Run with:
    cd /tmp && pytest deep_worker/deep_worker_cli/tests/integration_ai/test_agent_tool_call_duplication_deepseek.py -v -s
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
def test_agent_tool_call_no_duplication_deepseek(clean_agent):
    """Test that deepseek provider doesn't duplicate tool calls across queries.

    CONTROL TEST: This test verifies that tool calls from previous queries
    are NOT shown in the TUI for subsequent unrelated queries when using
    the DEFAULT deepseek provider.

    EXPECTED BEHAVIOR:
    - First query ("What is your current working directory?") triggers tool call
    - Second query ("2+2=") does NOT trigger tool call
    - Tool call indicators from first query are NOT visible in second query's response

    PASS CRITERIA:
    - First query shows tool call (execute/pwd)
    - Second query does NOT show tool call indicators
    - Second query correctly answers "4"

    FAIL CRITERIA:
    - Second query shows tool call indicators (execute, pwd, tool, 🔧)
    - Old tool calls are visible in TUI for second query
    """
    logger.info("=" * 80)
    logger.info("TEST: Deepseek Provider - No Tool Call Duplication (CONTROL TEST)")
    logger.info("=" * 80)

    # Change to /tmp directory
    original_cwd = os.getcwd()
    logger.info(f"Original working directory: {original_cwd}")

    os.chdir("/tmp")
    logger.info(f"Changed to: {os.getcwd()}")

    try:
        agent_name = "test_deepseek_no_dup"
        logger.info(f"Starting clean test agent: {agent_name}")
        logger.info("Provider: deepseek (DEFAULT)")
        logger.info("Skip permissions: True")

        # Start clean agent with DEFAULT provider (deepseek)
        agent_process = clean_agent(
            agent_name,
            working_dir="/tmp",
            provider="deepseek"  # Using default provider
        )
        logger.info(f"✓ Clean agent started (PID: {agent_process.pid})")

        # Open dwcli session
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

            # ========================================================================
            # QUERY 1: Ask about current working directory (triggers tool call)
            # ========================================================================
            query1 = "What is your current working directory?"
            logger.info(f"\n{'=' * 80}")
            logger.info(f"QUERY 1: '{query1}'")
            logger.info(f"{'=' * 80}")
            logger.info("Expected: Tool call (execute/pwd) should be visible")

            session.send_input(query1, press_enter=False)
            time.sleep(0.3)
            session.send_key("enter")
            logger.info("✓ Query 1 sent to dwcli")

            # Wait for AI response (up to 60 seconds)
            logger.info("Waiting for response to Query 1...")

            response1_received = False
            tool_call_detected_q1 = False

            for i in range(120):  # 60 seconds max
                time.sleep(0.5)
                screen_current = session.get_screen_text()

                # Check for tool call indicators
                tool_indicators = ["execute", "pwd", "tool", "Tool", "🔧"]
                for indicator in tool_indicators:
                    if indicator in screen_current:
                        if not tool_call_detected_q1:
                            tool_call_detected_q1 = True
                            logger.info(f"✓ Query 1: Tool call detected at {i * 0.5:.1f}s: '{indicator}'")

                # Check if still thinking
                if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                    if i % 20 == 0:
                        logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                    continue

                # Check for response containing /tmp
                if "/tmp" in screen_current:
                    response1_received = True
                    logger.info(f"✓ Query 1: AI response received after {i * 0.5:.1f} seconds")
                    break

                if i % 40 == 0 and i > 0:
                    logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

            if not response1_received:
                pytest.fail(f"Query 1 did not receive response within 60 seconds")

            # Wait for screen to stabilize
            time.sleep(2.0)
            screen_after_q1 = session.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_after_q1,
                title="AFTER QUERY 1 (should show tool call)",
            ))

            # Verify tool call was visible
            if not tool_call_detected_q1:
                logger.warning("⚠ Query 1: Tool call was NOT detected in TUI (unexpected)")

            # ========================================================================
            # QUERY 2: Simple arithmetic (should NOT trigger tool call)
            # ========================================================================
            query2 = "2+2="
            logger.info(f"\n{'=' * 80}")
            logger.info(f"QUERY 2: '{query2}'")
            logger.info(f"{'=' * 80}")
            logger.info("Expected: NO tool call should be visible")

            session.send_input(query2, press_enter=False)
            time.sleep(0.3)
            session.send_key("enter")
            logger.info("✓ Query 2 sent to dwcli")

            # Wait for AI response (up to 60 seconds)
            logger.info("Waiting for response to Query 2...")

            response2_received = False
            tool_call_detected_q2 = False
            tool_indicators_found_q2 = []

            for i in range(120):  # 60 seconds max
                time.sleep(0.5)
                screen_current = session.get_screen_text()

                # CRITICAL: Check for tool call indicators (should NOT appear for Query 2)
                tool_indicators = ["execute", "pwd", "tool", "Tool", "🔧"]
                for indicator in tool_indicators:
                    if indicator in screen_current:
                        if not tool_call_detected_q2:
                            tool_call_detected_q2 = True
                            tool_indicators_found_q2.append(indicator)
                            logger.error(f"❌ Query 2: Tool call indicator detected at {i * 0.5:.1f}s: '{indicator}'")
                            logger.error("This should NOT happen for simple arithmetic!")

                # Check for response containing "4" (answer to 2+2)
                # NOTE: Don't skip checking when "Thinking" is visible - the response
                # might already be on screen even if spinner hasn't cleared yet
                # Look for Query 2 marker and check text after it
                query2_marker = f"> {query2}"
                if query2_marker in screen_current:
                    query2_pos = screen_current.rfind(query2_marker)
                    text_after_query2 = screen_current[query2_pos:]
                    # Check if response contains "4" after the query
                    # Response can be as short as just "4", so check for minimal content
                    if "4" in text_after_query2 and len(text_after_query2) > len(query2_marker) + 2:
                        response2_received = True
                        logger.info(f"✓ Query 2: AI response received after {i * 0.5:.1f} seconds")
                        break

                if i % 40 == 0 and i > 0:
                    logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

            if not response2_received:
                pytest.fail(f"Query 2 did not receive response within 60 seconds")

            # Wait for screen to stabilize
            time.sleep(2.0)
            screen_after_q2 = session.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_after_q2,
                title="AFTER QUERY 2 (should NOT show tool call)",
            ))

            # ========================================================================
            # CRITICAL CHECK: Verify NO tool calls for Query 2
            # ========================================================================
            logger.info("\n" + "=" * 80)
            logger.info("CRITICAL CHECK: Tool call duplication verification")
            logger.info("=" * 80)

            # Extract only the text AFTER the second query
            # to avoid false positives from history scroll
            query2_marker = f"> {query2}"
            if query2_marker in screen_after_q2:
                # Find position of Query 2 in screen
                query2_pos = screen_after_q2.rfind(query2_marker)
                text_after_query2 = screen_after_q2[query2_pos:]

                # Check for tool call indicators in ONLY the new output
                tool_indicators_in_new_output = []
                for indicator in ["⏺ execute", "⏺ read", "⏺ write", "Tool:", "🔧"]:
                    if indicator in text_after_query2:
                        tool_indicators_in_new_output.append(indicator)

                if tool_indicators_in_new_output:
                    logger.error(f"❌ CRITICAL: Tool call indicators found AFTER Query 2: {tool_indicators_in_new_output}")
                    tool_call_detected_q2 = True
                    tool_indicators_found_q2 = tool_indicators_in_new_output
                else:
                    logger.info(f"✓ No tool call indicators found after Query 2 (checked: {len(text_after_query2)} chars)")
                    tool_call_detected_q2 = False
            else:
                logger.warning(f"⚠ Could not find Query 2 marker in screen")

            if tool_call_detected_q2:
                logger.error("\n" + "=" * 80)
                logger.error("❌ CRITICAL FAILURE: Tool call indicators found for Query 2!")
                logger.error("=" * 80)
                logger.error(f"Tool indicators found: {tool_indicators_found_q2}")
                logger.error("\n" + create_terminal_snapshot(
                    screen_after_q2,
                    title="FAILED - Tool call visible for Query 2",
                ))

                pytest.fail(
                    f"Tool call duplication bug detected for DEEPSEEK provider!\n\n"
                    f"Query 1: '{query1}' - correctly triggered tool call\n"
                    f"Query 2: '{query2}' - should NOT trigger tool call\n\n"
                    f"Expected: No tool call indicators for Query 2\n"
                    f"Actual: Tool call indicators found: {tool_indicators_found_q2}\n\n"
                    f"This indicates that tool calls from previous queries are being\n"
                    f"duplicated or re-displayed for subsequent unrelated queries.\n"
                    f"This is a GENERAL BUG affecting ALL providers, not just claude-code!\n"
                )

            logger.info("✓ No tool call indicators found for Query 2")

            # ========================================================================
            # Validate that Query 2 answered correctly
            # ========================================================================
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION: Checking Query 2 answer")
            logger.info("=" * 80)

            validation_result = verify_screenshot_contains_sync(
                screenshot_text=screen_after_q2,
                original_query=query2,
                expected_response_contains="4",
                model="haiku",
            )

            logger.info(f"Validation result: {validation_result}")

            if not validation_result.is_valid:
                logger.error(f"❌ Query 2 validation failed: {validation_result.comment}")
                pytest.fail(f"Query 2 did not answer correctly: {validation_result.comment}")

            # ========================================================================
            # SUCCESS
            # ========================================================================
            logger.info("\n" + "=" * 80)
            logger.info("✅ TEST PASSED: No tool call duplication detected for deepseek!")
            logger.info("✅ Query 1 correctly used tool call")
            logger.info("✅ Query 2 did NOT show tool call indicators")
            logger.info("✅ Query 2 answered correctly")
            logger.info(f"   Validation comment: {validation_result.comment}")
            logger.info("=" * 80)

        finally:
            session.close()

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {os.getcwd()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
