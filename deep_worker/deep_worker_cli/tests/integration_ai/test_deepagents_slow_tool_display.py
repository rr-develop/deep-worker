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

logger = logging.getLogger(__name__)


def test_deepagents_slow_tool_call_displays_immediately():
    """Tool call message MUST appear in deepagents TUI BEFORE the tool finishes.

    🔒 REFERENCE TEST — verifies deepagents' correct streaming behavior.

    The test asks deepagents to run ``sleep 10`` via the bash tool, then
    polls the TUI screenshot every 0.5 s looking for the tool-call marker
    (``bash`` / ``sleep 10`` / ``execute``). The marker MUST appear within
    8 s — i.e. well before the 10-second sleep finishes.

    History note: previously this test used a Claude Haiku LLM judge to
    decide whether "any response is visible". That introduced flakiness
    (judge had to see both the original query AND any response in a
    120×40 snapshot that scrolls), a 30-second runtime, and external API
    dependency. The substring poll below is deterministic, costs no API
    calls, and finishes in <10 s.
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

        # Send query requesting sleep tool. Kept short so it fits in a
        # single 120-column line and isn't scrolled out of view.
        query = "Run 'sleep 10' via the bash tool, then say hello."
        logger.info(f"Sending query: {query}")
        session.send_input(query, press_enter=False)
        time.sleep(0.3)
        session.send_key("enter")
        t_send = time.time()
        logger.info("✓ Query sent to deepagents")

        # Poll for a tool-call marker. It MUST appear before the 10-second
        # sleep finishes — that's the whole point of the test (streaming
        # display, not "wait for tool then render").
        tool_markers = ("bash", "sleep 10", "execute", "tool_call")
        deadline = t_send + 8.0  # 2 s safety margin before sleep(10) finishes
        appeared_at: float | None = None
        screenshot = ""

        while time.time() < deadline:
            screenshot = session.get_screen_text()
            lower = screenshot.lower()
            if any(m in lower for m in tool_markers):
                appeared_at = time.time() - t_send
                logger.info(f"✓ Tool-call marker appeared after {appeared_at:.2f}s")
                break
            time.sleep(0.25)

        logger.info("\n" + create_terminal_snapshot(
            screenshot,
            title=f"SCREENSHOT @ t={time.time() - t_send:.1f}s",
        ))

        if appeared_at is None:
            # Capture final screenshot after the sleep finishes for debugging
            time.sleep(max(0.0, 10.5 - (time.time() - t_send)))
            final_screenshot = session.get_screen_text()
            logger.info("\n" + create_terminal_snapshot(
                final_screenshot,
                title="FINAL SCREENSHOT (after tool completion)",
            ))
            pytest.fail(
                "UNEXPECTED: deepagents tool-call did not stream to the TUI before "
                "sleep(10) finished.\n"
                f"None of {tool_markers!r} appeared in the screenshot within 8 s.\n\n"
                f"Screenshot @ 8s:\n{screenshot}\n\n"
                f"Screenshot after tool completed:\n{final_screenshot}\n"
            )

        logger.info("=" * 80)
        logger.info(
            f"✅ SUCCESS: Tool-call appeared in {appeared_at:.2f}s "
            f"(well before sleep(10) finished)"
        )
        logger.info("=" * 80)

    finally:
        session.close()
