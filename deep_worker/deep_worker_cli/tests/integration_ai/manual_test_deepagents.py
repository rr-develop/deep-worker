#!/usr/bin/env python3
"""Manual test script for deepagents terminal session.

This script demonstrates testing deepagents in a controlled terminal environment.
It sends a query "2+2" and verifies that:
1. The query is displayed on screen (doesn't disappear)
2. The AI responds with the answer
3. Both are visible in the terminal

Run with:
    python tests/integration_ai/manual_test_deepagents.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from terminal_emulator import DeepAgentsTerminalSession, create_terminal_snapshot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run manual test of deepagents terminal session."""
    logger.info("=" * 80)
    logger.info("Manual Test: DeepAgents Terminal Session")
    logger.info("=" * 80)

    # Create session
    session = DeepAgentsTerminalSession(
        agent_name=None,  # Use default agent
        columns=120,
        lines=40,
        working_dir=Path("/tmp"),
    )

    try:
        # Start deepagents
        logger.info("Starting deepagents...")
        session.start(timeout=15.0)
        logger.info("✓ deepagents started successfully")

        # Wait for prompt
        logger.info("Waiting for prompt to appear...")
        if session.wait_for_prompt(timeout=10.0):
            logger.info("✓ Prompt appeared")
        else:
            logger.warning("⚠ Prompt did not appear within timeout")

        # Get initial screen
        time.sleep(1.0)
        screen_before = session.get_screen_text()
        logger.info("\n" + create_terminal_snapshot(
            screen_before,
            title="INITIAL SCREEN (before sending query)",
        ))

        # Send query
        query = "2+2"
        logger.info(f"\nSending query: '{query}'")
        session.send_input(query, press_enter=True)

        # Wait a moment for query to appear
        time.sleep(0.5)
        screen_after_query = session.get_screen_text()
        logger.info("\n" + create_terminal_snapshot(
            screen_after_query,
            title="SCREEN AFTER QUERY (should show '2+2')",
        ))

        # Check if query is visible
        if query in screen_after_query:
            logger.info(f"✓ Query '{query}' IS VISIBLE on screen (NOT disappeared!)")
        else:
            logger.error(f"✗ Query '{query}' NOT visible on screen (disappeared!)")

        # Wait for AI response
        logger.info("\nWaiting for AI response (up to 60 seconds)...")
        time.sleep(2.0)  # Initial wait

        # Check for response periodically
        response_found = False
        for i in range(58):  # 58 more seconds (total 60)
            time.sleep(1.0)
            screen_current = session.get_screen_text()

            # Look for answer "4" or response indicators
            # Check for: actual "4" answer, "Assistant", thinking indicators, etc
            indicators = ["4", "four", "equals", "Assistant", "Thinking", "answer"]
            found_indicators = [ind for ind in indicators if ind.lower() in screen_current.lower()]

            if found_indicators:
                response_found = True
                logger.info(f"✓ AI response indicators detected after {i + 2} seconds: {found_indicators}")
                # Wait a bit more to get full response
                time.sleep(2.0)
                break

            if i % 10 == 0 and i > 0:
                logger.info(f"  Still waiting... ({i + 2}s elapsed)")

        # Final screen
        screen_final = session.get_screen_text()
        logger.info("\n" + create_terminal_snapshot(
            screen_final,
            title="FINAL SCREEN (after AI response)",
        ))

        # Analysis
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS")
        logger.info("=" * 80)

        query_visible = query in screen_final
        answer_visible = "4" in screen_final

        logger.info(f"Query '{query}' visible: {query_visible}")
        logger.info(f"Answer '4' visible: {answer_visible}")

        if query_visible and answer_visible:
            logger.info("✓ SUCCESS: Both query and answer are visible!")
        elif query_visible:
            logger.warning("⚠ PARTIAL: Query visible but answer not found")
        elif answer_visible:
            logger.warning("⚠ PARTIAL: Answer visible but query disappeared")
        else:
            logger.error("✗ FAILURE: Neither query nor answer visible")

        # Keep session open for manual inspection
        logger.info("\n" + "=" * 80)
        logger.info("Test complete. Press Enter to close session...")
        logger.info("=" * 80)
        input()

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("Closing session...")
        session.close()
        logger.info("✓ Session closed")


if __name__ == "__main__":
    main()
