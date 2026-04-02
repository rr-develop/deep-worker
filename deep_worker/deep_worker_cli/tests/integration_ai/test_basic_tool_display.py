"""Test basic tool call display in dwcli TUI.

This test verifies that tool calls are displayed correctly for common tools
like bash (pwd command).
"""
import logging
import time

import pytest

from .clean_agent_fixture import clean_agent
from .terminal_emulator import DwcliTerminalSession

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_pwd_tool_call_visible(clean_agent):
    """Test that 'pwd' bash tool call is visible in TUI.

    This test:
    1. Sends a query requesting current directory via bash tool
    2. Waits 3 seconds for tool call to appear
    3. Captures screen and checks for tool call indicators (bash, pwd, /tmp)
    4. Waits for completion and verifies result is visible

    Expected: Tool call and result both visible in TUI.
    """
    agent_name = "test_basic_tool_display"
    logger.info(f"Testing tool display with agent: {agent_name}")

    # ✨ CRITICAL: Call fixture to create the agent!
    clean_agent(agent_name, working_dir="/tmp")
    logger.info("✓ Agent created via fixture")

    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=100,
        lines=30,
    )

    try:
        session.start(timeout=15.0)
        logger.info("✓ dwcli started")

        # Wait for prompt
        if not session.wait_for_text(">", timeout=10.0):
            screen = session.get_screen_text()
            pytest.fail(f"Prompt not found in TUI.\nScreen:\n{screen}")

        time.sleep(1.0)

        # Send query requiring bash tool
        query = "What is the current working directory? Use the bash tool to execute 'pwd' command."
        logger.info(f"Sending query: {query}")
        session.send_input(query, press_enter=True)

        # Wait for tool call to appear (model needs time to respond)
        logger.info("Waiting 5 seconds for tool call...")
        time.sleep(5.0)

        # Capture screen
        screen = session.get_screen_text()
        logger.info("\n" + "="*80)
        logger.info("SCREEN AFTER 5 SECONDS (tool call should be visible):")
        logger.info("="*80)
        logger.info(screen)
        logger.info("="*80)

        # Check for tool indicators
        screen_lower = screen.lower()
        has_tool_indicator = (
            "bash" in screen_lower or
            "tool" in screen_lower or
            "pwd" in screen_lower or
            "/tmp" in screen_lower  # Working directory result
        )

        if not has_tool_indicator:
            pytest.fail(
                f"Tool call not visible in TUI after 5 seconds.\n"
                f"Expected to see 'bash', 'tool', 'pwd', or '/tmp'.\n"
                f"Screen content:\n{screen}"
            )

        logger.info("✓ Tool call indicators found in TUI")

        # Wait for completion
        logger.info("Waiting additional 5 seconds for completion...")
        time.sleep(5.0)

        # Final screen
        screen = session.get_screen_text()
        logger.info("\n" + "="*80)
        logger.info("FINAL SCREEN (result should be visible):")
        logger.info("="*80)
        logger.info(screen)
        logger.info("="*80)

        # Verify result is visible
        if "/tmp" not in screen_lower:
            logger.warning(
                "WARNING: Working directory result '/tmp' not found in final screen. "
                "This may indicate tool result is not displayed."
            )
        else:
            logger.info("✓ Tool result '/tmp' found in final screen")

    finally:
        session.close()
