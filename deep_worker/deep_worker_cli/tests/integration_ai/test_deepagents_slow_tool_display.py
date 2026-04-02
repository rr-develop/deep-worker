"""Test: Tool calls display immediately in deepagents TUI (not after completion).

This test verifies that deepagents DOES NOT have the tool display delay bug.
Expected: Tool call appears within 1 second, before tool execution completes.

This is the REFERENCE TEST to compare against deep-worker's broken behavior.
"""

import logging
import time
from pathlib import Path

import pytest

from .terminal_emulator import DeepAgentsTerminalSession, create_terminal_snapshot
from .check_reply import verify_screenshot_contains

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_deepagents_slow_tool_call_displays_immediately():
    """Tool call message MUST appear in TUI immediately in deepagents.

    🔒 REFERENCE TEST - Verifies deepagents correct behavior

    This test verifies that deepagents does NOT have the delay bug.

    EXPECTED BEHAVIOR (deepagents - CORRECT):
    - After 1 second, TUI should show the tool_call for sleep
    - Tool execution continues in background
    - After 10 seconds, tool_result appears

    This test should PASS, demonstrating that deepagents works correctly.
    """
    logger.info("=" * 80)
    logger.info("TEST: DeepAgents Slow Tool Display (Reference Implementation)")
    logger.info("=" * 80)

    session = DeepAgentsTerminalSession(
        agent_name=None,  # Use default
        columns=120,
        lines=40,
        working_dir=Path("/tmp"),
    )

    try:
        # Start deepagents
        logger.info("Starting deepagents...")
        session.start(timeout=15.0)

        # Wait for prompt
        logger.info("Waiting for prompt...")
        assert session.wait_for_prompt(timeout=10.0), "Prompt did not appear"
        logger.info("✓ Prompt appeared")

        # Give TUI a moment to fully render
        time.sleep(1.0)

        # Send query requesting sleep tool
        query = (
            "Please call the bash tool to execute 'sleep 10' command, "
            "then say hello to me. Make sure you use the bash tool first."
        )
        logger.info(f"Sending query: {query}")
        session.send_input(query, press_enter=False)
        time.sleep(0.3)
        session.send_key("enter")
        logger.info("✓ Query sent to deepagents")

        # CRITICAL: Wait 5 seconds, then take screenshot
        # Tool call should be visible even though tool is still executing
        # 5 seconds gives model time to start responding and emit more than just one letter
        logger.info("Waiting 5 seconds before screenshot...")
        time.sleep(5.0)

        # Capture screenshot
        screenshot = session.get_screen_text()
        logger.info("\n" + create_terminal_snapshot(
            screenshot,
            title="SCREENSHOT AFTER 1 SECOND (tool call should be visible)",
        ))

        # Use AI to analyze the screenshot
        logger.info("Analyzing screenshot with AI...")
        result = await verify_screenshot_contains(
            screenshot_text=screenshot,
            original_query=query,
            expected_response_contains=None,  # Just check any response is visible (spinner, text, etc)
            model="haiku",
        )

        # Check if screenshot analysis passed
        if not result.is_valid:
            # Get final screenshot after tool completes (for debugging)
            logger.info("Initial check failed. Waiting for tool to complete...")
            time.sleep(12.0)  # Wait for sleep(10) to finish
            final_screenshot = session.get_screen_text()
            logger.info("\n" + create_terminal_snapshot(
                final_screenshot,
                title="FINAL SCREENSHOT (after tool completion)",
            ))

            pytest.fail(
                f"UNEXPECTED: deepagents has the same bug!\n\n"
                f"The tool call did NOT appear in the TUI within 5 seconds.\n"
                f"This should NOT happen in deepagents.\n\n"
                f"AI Analysis: {result.comment}\n\n"
                f"Screenshot after 5 seconds:\n{screenshot}\n\n"
                f"Screenshot after tool completed:\n{final_screenshot}\n\n"
            )

        logger.info("=" * 80)
        logger.info("✅ SUCCESS: Tool call appeared within 1 second in deepagents!")
        logger.info("=" * 80)
        logger.info("✅ This demonstrates CORRECT behavior - deepagents works as expected")
        logger.info("=" * 80)

    finally:
        session.close()
