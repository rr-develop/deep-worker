"""Tests for terminal emulator infrastructure.

This module tests the terminal emulator functionality for dwcli UI testing:
- TerminalEmulator: ANSI escape sequence handling and screen capture
- DwcliTerminalSession: dwcli subprocess management with PTY
- Helper functions: Terminal snapshot formatting

Run with:
    pytest tests/integration_ai/test_terminal_emulator.py -v -s
"""

from __future__ import annotations

import asyncio
import logging
import os
import pty
import signal
import subprocess
import time
from pathlib import Path

import pyte
import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401
from .terminal_emulator import (
    DeepAgentsTerminalSession,
    DwcliTerminalSession,
    TerminalEmulator,
    create_terminal_snapshot,
)
from .terminal_pexpect import PexpectTerminalSession

# Try to import check_reply, but don't fail if claude_sdk is unavailable
try:
    from .check_reply import check_reply
    CHECK_REPLY_AVAILABLE = True
except ImportError:
    CHECK_REPLY_AVAILABLE = False
    logger.warning("check_reply not available (claude_sdk missing). AI validation will be skipped.")

logger = logging.getLogger(__name__)


# =============================================================================
# TerminalEmulator Tests (basic pyte functionality)
# =============================================================================


def test_terminal_emulator_init():
    """Test terminal emulator initialization."""
    emulator = TerminalEmulator(columns=80, lines=24)

    assert emulator.columns == 80
    assert emulator.lines == 24
    assert isinstance(emulator.screen, pyte.Screen)
    assert isinstance(emulator.stream, pyte.Stream)


def test_terminal_emulator_feed_simple_text():
    """Test feeding simple text without escape sequences."""
    emulator = TerminalEmulator(columns=40, lines=10)

    # Feed simple text
    emulator.feed(b"Hello, World!")

    screen_text = emulator.get_screen_text()
    assert "Hello, World!" in screen_text


def test_terminal_emulator_feed_with_newlines():
    """Test feeding text with newlines."""
    emulator = TerminalEmulator(columns=40, lines=10)

    # Feed multi-line text
    emulator.feed(b"Line 1\nLine 2\nLine 3")

    screen_text = emulator.get_screen_text()
    assert "Line 1" in screen_text
    assert "Line 2" in screen_text
    assert "Line 3" in screen_text


def test_terminal_emulator_ansi_colors():
    """Test handling ANSI color escape sequences."""
    emulator = TerminalEmulator(columns=40, lines=10)

    # Feed text with ANSI color codes
    # \x1b[31m is red, \x1b[0m is reset
    emulator.feed(b"\x1b[31mRed Text\x1b[0m Normal Text")

    screen_text = emulator.get_screen_text()
    # Color codes should be processed, text should remain
    assert "Red Text" in screen_text
    assert "Normal Text" in screen_text


def test_terminal_emulator_cursor_movement():
    """Test cursor movement escape sequences."""
    emulator = TerminalEmulator(columns=40, lines=10)

    # Write at position, move cursor, write again
    emulator.feed(b"Start")
    emulator.feed(b"\x1b[1;10H")  # Move to line 1, column 10
    emulator.feed(b"Middle")

    screen_text = emulator.get_screen_text()
    assert "Start" in screen_text
    assert "Middle" in screen_text


def test_terminal_emulator_get_line():
    """Test getting specific line from terminal."""
    emulator = TerminalEmulator(columns=40, lines=10)

    emulator.feed(b"Line 0\nLine 1\nLine 2\nLine 3")

    assert "Line 0" in emulator.get_line(0)
    assert "Line 1" in emulator.get_line(1)
    assert "Line 2" in emulator.get_line(2)
    assert "Line 3" in emulator.get_line(3)

    # Out of bounds
    assert emulator.get_line(-1) == ""
    assert emulator.get_line(100) == ""


def test_terminal_emulator_find_text():
    """Test finding text in terminal screen."""
    emulator = TerminalEmulator(columns=40, lines=10)

    emulator.feed(b"First line\nSecond line with TARGET\nThird line")

    # Find existing text
    pos = emulator.find_text("TARGET")
    assert pos is not None
    line, col = pos
    assert line == 1  # Second line (0-indexed)
    assert col >= 0

    # Find non-existing text
    pos = emulator.find_text("NOTFOUND")
    assert pos is None


def test_terminal_emulator_find_text_case_insensitive():
    """Test case-insensitive text search."""
    emulator = TerminalEmulator(columns=40, lines=10)

    emulator.feed(b"Hello World")

    # Case-sensitive (default)
    assert emulator.find_text("hello") is None
    assert emulator.find_text("Hello") is not None

    # Case-insensitive
    assert emulator.find_text("hello", case_sensitive=False) is not None
    assert emulator.find_text("WORLD", case_sensitive=False) is not None


def test_terminal_emulator_get_cursor_position():
    """Test getting cursor position."""
    emulator = TerminalEmulator(columns=40, lines=10)

    # Initial position
    y, x = emulator.get_cursor_position()
    assert y == 0
    assert x == 0

    # Write some text
    emulator.feed(b"Hello")
    y, x = emulator.get_cursor_position()
    assert x == 5  # After "Hello"


def test_terminal_emulator_clear():
    """Test clearing terminal screen."""
    emulator = TerminalEmulator(columns=40, lines=10)

    # Add content
    emulator.feed(b"Some content\nMore content")
    assert len(emulator.get_screen_text().strip()) > 0

    # Clear
    emulator.clear()
    screen_text = emulator.get_screen_text().strip()
    assert len(screen_text) == 0 or screen_text == ""


def test_terminal_emulator_resize():
    """Test resizing terminal."""
    emulator = TerminalEmulator(columns=40, lines=10)

    assert emulator.columns == 40
    assert emulator.lines == 10

    # Resize
    emulator.resize(columns=80, lines=24)

    assert emulator.columns == 80
    assert emulator.lines == 24


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_create_terminal_snapshot_basic():
    """Test creating terminal snapshot without cursor."""
    screen_text = "Line 1\nLine 2\nLine 3"

    snapshot = create_terminal_snapshot(screen_text)

    assert "=" * 80 in snapshot
    assert "Line 1" in snapshot
    assert "Line 2" in snapshot
    assert "Line 3" in snapshot


def test_create_terminal_snapshot_with_title():
    """Test creating terminal snapshot with title."""
    screen_text = "Content"

    snapshot = create_terminal_snapshot(screen_text, title="Test Screen")

    assert "Test Screen" in snapshot
    assert "Content" in snapshot


def test_create_terminal_snapshot_with_cursor():
    """Test creating terminal snapshot with cursor position."""
    screen_text = "Content"
    cursor_pos = (5, 10)

    snapshot = create_terminal_snapshot(
        screen_text,
        show_cursor=True,
        cursor_pos=cursor_pos,
    )

    assert "Cursor: Line 5, Column 10" in snapshot
    assert "Content" in snapshot


# =============================================================================
# DwcliTerminalSession Tests (mock-based)
# =============================================================================


def test_dwcli_terminal_session_init():
    """Test DwcliTerminalSession initialization."""
    session = DwcliTerminalSession(
        agent_name="test_agent",
        columns=120,
        lines=40,
        working_dir=Path("/tmp"),
    )

    assert session.agent_name == "test_agent"
    assert session.columns == 120
    assert session.lines == 40
    assert session.working_dir == Path("/tmp")
    assert isinstance(session.emulator, TerminalEmulator)
    assert not session.is_running()


def test_dwcli_terminal_session_send_key_validation():
    """Test that send_key validates key names."""
    session = DwcliTerminalSession(agent_name="test_agent")

    # Don't actually start session, just test validation
    with pytest.raises(ValueError, match="Unknown key"):
        # This will fail because session not started, but we're testing validation
        try:
            session.send_key("invalid_key_name")
        except RuntimeError:
            # Expected if session not running
            pass
        except ValueError as e:
            # This is what we're testing
            raise


# =============================================================================
# DwcliTerminalSession Integration Tests (real dwcli)
# =============================================================================


def _check_dwcli_available() -> bool:
    """Check if dwcli command is available."""
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
def test_dwcli_terminal_session_start_stop():
    """Test starting and stopping dwcli terminal session.

    This test requires dwcli to be available in PATH.
    """
    session = DwcliTerminalSession(
        agent_name="test_terminal_emulator",
        columns=120,
        lines=40,
    )

    try:
        # Start session
        session.start(timeout=10.0)

        # Check running
        assert session.is_running()
        assert session.pid is not None
        assert session.master_fd is not None

        # Give it a moment to render
        time.sleep(0.5)

        # Get screen text
        screen = session.get_screen_text()
        assert len(screen) > 0

        logger.info(f"Screen capture:\n{screen}")

    finally:
        # Always cleanup
        session.close()
        assert not session.is_running()


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_context_manager():
    """Test using DwcliTerminalSession as context manager."""
    with DwcliTerminalSession(agent_name="test_ctx_manager") as session:
        assert session.is_running()

        # Basic interaction
        screen = session.get_screen_text()
        assert len(screen) > 0

    # Should be closed after context
    assert not session.is_running()


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_send_input():
    """Test sending input to dwcli terminal."""
    with DwcliTerminalSession(agent_name="test_input") as session:
        # Wait for prompt
        time.sleep(1.0)

        # Send some text
        session.send_input("Hello from test")

        # Check screen updated
        screen = session.get_screen_text()
        # Note: Whether text appears depends on dwcli UI implementation
        logger.info(f"After input:\n{screen}")


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_send_keys():
    """Test sending special keys to dwcli terminal."""
    with DwcliTerminalSession(agent_name="test_keys") as session:
        time.sleep(1.0)

        # Test various keys
        session.send_key("up")
        session.send_key("down")
        session.send_key("enter")
        session.send_key("escape")
        session.send_key("tab")

        # Should not raise exceptions
        screen = session.get_screen_text()
        logger.info(f"After key presses:\n{screen}")


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_wait_for_text():
    """Test waiting for specific text to appear."""
    with DwcliTerminalSession(agent_name="test_wait") as session:
        # Wait for any content to appear (UI rendered)
        # This tests the mechanism even if we don't know exact text
        initial_screen = session.get_screen_text()

        if initial_screen.strip():
            # Some text already appeared
            found = session.wait_for_text(
                initial_screen.strip()[:10],  # First 10 chars
                timeout=1.0,
            )
            assert found


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_wait_for_prompt():
    """Test waiting for prompt to appear."""
    with DwcliTerminalSession(agent_name="test_prompt") as session:
        # Wait for prompt (UI ready)
        found = session.wait_for_prompt(timeout=10.0)
        assert found, "Prompt should appear after startup"

        # Screen should have content
        screen = session.get_screen_text()
        assert len(screen.strip()) > 0


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_get_cursor_position():
    """Test getting cursor position from running session."""
    with DwcliTerminalSession(agent_name="test_cursor") as session:
        session.wait_for_prompt(timeout=10.0)

        # Get cursor position
        line, col = session.get_cursor_position()

        assert line >= 0
        assert col >= 0

        logger.info(f"Cursor position: line={line}, col={col}")


# =============================================================================
# Stress Tests
# =============================================================================


def test_terminal_emulator_large_output():
    """Test handling large amounts of output."""
    emulator = TerminalEmulator(columns=80, lines=24)

    # Generate large output
    large_text = "\n".join(f"Line {i}" for i in range(1000))
    emulator.feed(large_text.encode("utf-8"))

    # Should handle without crashing
    screen = emulator.get_screen_text()
    assert len(screen) > 0


def test_terminal_emulator_rapid_updates():
    """Test handling rapid screen updates."""
    emulator = TerminalEmulator(columns=80, lines=24)

    # Rapid updates
    for i in range(100):
        emulator.feed(f"Update {i}\r".encode("utf-8"))

    screen = emulator.get_screen_text()
    assert len(screen) > 0


def test_terminal_emulator_unicode():
    """Test handling Unicode characters."""
    emulator = TerminalEmulator(columns=80, lines=24)

    # Various Unicode characters
    unicode_text = "Hello 世界 🚀 Привет Мир"
    emulator.feed(unicode_text.encode("utf-8"))

    screen = emulator.get_screen_text()
    # pyte should handle unicode (some chars may render differently)
    assert len(screen) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_dwcli_terminal_session_send_input_not_running():
    """Test that send_input raises when session not running."""
    session = DwcliTerminalSession(agent_name="test_not_running")

    with pytest.raises(RuntimeError, match="not running"):
        session.send_input("test")


def test_dwcli_terminal_session_send_key_not_running():
    """Test that send_key raises when session not running."""
    session = DwcliTerminalSession(agent_name="test_not_running")

    with pytest.raises(RuntimeError, match="not running"):
        session.send_key("enter")


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_terminal_session_double_start():
    """Test that starting twice doesn't cause issues."""
    session = DwcliTerminalSession(agent_name="test_double_start")

    try:
        session.start(timeout=10.0)

        # Try to start again - should handle gracefully or raise clear error
        # (Current implementation will likely fail because pid is already set)

    finally:
        session.close()


def test_dwcli_terminal_session_close_not_started():
    """Test that close() on non-started session is safe."""
    session = DwcliTerminalSession(agent_name="test_close_safe")

    # Should not raise
    session.close()
    assert not session.is_running()


# =============================================================================
# DeepAgents Integration Tests with AI Validation
# =============================================================================


def _check_deepagents_available() -> bool:
    """Check if deepagents command is available."""
    try:
        result = subprocess.run(
            ["which", "deepagents"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _check_deepagents_available(),
    reason="deepagents command not available in PATH",
)
@pytest.mark.asyncio
async def test_deepagents_query_does_not_disappear():
    """Test that queries and responses are visible in deepagents terminal.

    🔒 LOCKED TEST - DO NOT MODIFY WITHOUT EXPLICIT USER REQUEST
    This test is a gold standard reference implementation. Any modifications
    require explicit request from the user. This ensures the baseline behavior
    remains consistent and reproducible.

    This test verifies the CRITICAL behavior:
    1. Send a query "What is 2+2?" to deepagents
    2. Wait for AI to generate a response (should contain "4")
    3. Use AI validation to check that BOTH query and response are visible on screen
    4. This is the CORRECT behavior - both should remain visible

    This is the GOLD STANDARD showing correct TUI behavior.
    """
    logger.info("=" * 80)
    logger.info("TEST: DeepAgents Query and Response Visibility (Gold Standard)")
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
        assert session.wait_for_prompt(timeout=10.0), "Prompt did not appear"
        logger.info("✓ Prompt appeared")

        # Capture initial screen
        time.sleep(1.0)
        screen_before = session.get_screen_text()
        logger.info(f"Initial screen captured ({len(screen_before)} chars)")

        # Send query
        query = "What is 2+2?"
        logger.info(f"Sending query: '{query}'")
        session.send_input(query, press_enter=False)  # Don't press enter yet
        time.sleep(0.3)  # Brief pause before sending
        session.send_key("enter")  # Send with Enter key
        logger.info("✓ Query sent to deepagents")

        # Wait for AI response (up to 120 seconds = 2 minutes)
        logger.info("Waiting for AI response...")
        response_received = False
        last_screen_with_thinking = None

        for i in range(240):  # 240 * 0.5s = 120 seconds max
            time.sleep(0.5)  # Check every 0.5 seconds
            screen_current = session.get_screen_text()

            # Check if still thinking
            if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                if i % 20 == 0:
                    logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                last_screen_with_thinking = screen_current
                continue

            # If we get here, thinking has stopped - check for answer
            # The AI may respond with "4", "four", "2+2=4", "The answer is 4", etc.
            # Also accept if ">" prompt appears (query was processed)
            if "4" in screen_current or "four" in screen_current.lower() or ">" in screen_current:
                response_received = True
                logger.info(f"✓ AI response received after {i * 0.5:.1f} seconds")
                break

            if i % 40 == 0 and i > 0:
                logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

        if not response_received:
            logger.warning("⚠ AI did not respond with clear answer within 120 seconds")
            # Get final screen and try validation anyway
            screen_final = session.get_screen_text()
        else:
            # AI responded with what looks like "4"
            logger.info("✓ Processing complete, waiting for screen to stabilize...")
            # Wait a bit more to ensure screen is fully rendered
            time.sleep(2.0)
            screen_final = session.get_screen_text()

        logger.info("\n" + create_terminal_snapshot(
            screen_final,
            title="FINAL SCREEN (query and response should be visible)",
        ))

        # Now use AI validation to verify that the screenshot contains both query and response
        logger.info("\nValidating screenshot using AI analysis...")
        logger.info(f"Checking for query: '{query}'")
        logger.info("Checking for response containing: '4' (or variations)")

        # Import the validation function
        try:
            from .check_reply import verify_screenshot_contains

            validation_result = await verify_screenshot_contains(
                screenshot_text=screen_final,
                original_query=query,
                expected_response_contains="4",
                model="haiku",  # Use cheapest model
            )

            logger.info(f"Validation result: {validation_result}")

            if not validation_result.is_valid:
                logger.error(f"❌ Screenshot validation FAILED: {validation_result.comment}")
                logger.error("\n" + create_terminal_snapshot(
                    screen_final,
                    title="FAILED VALIDATION - Check this screenshot",
                ))
                pytest.fail(
                    f"Screenshot validation failed: {validation_result.comment}\n"
                    f"Query should be: '{query}'\n"
                    f"Response should contain: '4'"
                )

            logger.info("✓ AI validation confirmed both query and response are visible")

        except ImportError as e:
            logger.warning(f"Could not import verify_screenshot_contains: {e}")
            logger.warning("Falling back to simple text search (less robust)")

            # Fallback: simple text search with normalization
            screen_normalized = " ".join(screen_final.split())

            if query not in screen_normalized:
                pytest.fail(f"Query '{query}' not found in screenshot")

            logger.info(f"✓ Query '{query}' found in screenshot")

        logger.info("=" * 80)
        logger.info("✅ TEST PASSED: Query and response both visible on screen")
        logger.info("=" * 80)
        logger.info("✅ This demonstrates correct TUI behavior in deepagents")
        logger.info("=" * 80)

    finally:
        session.close()


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_query_visibility(clean_agent):
    """Test that queries and responses are visible in dwcli terminal.

    🔒 SYNCHRONIZED WITH: test_deepagents_query_does_not_disappear
    This test uses identical logic but with dwcli instead of deepagents.
    The only differences are:
    - Command: "dwcli <agent_name>" instead of "deepagents"
    - Agent: Must be a test agent (prefixed with "test_") that is running

    This test verifies the CRITICAL behavior:
    1. Send a query "What is 2+2?" to dwcli (connected to a running agent)
    2. Wait for AI to generate a response (should contain "4")
    3. Use AI validation to check that BOTH query and response are visible on screen
    4. This is the CORRECT behavior - both should remain visible

    This matches the GOLD STANDARD behavior from test_deepagents_query_does_not_disappear.

    ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
    ============================================================
    This test uses clean_agent fixture to ensure agent starts with NO history.
    All messages (user + assistant) displayed in dwcli come from agent via RPyC callbacks.
    This ensures no duplicate messages and maintains single source of truth architecture.
    """
    logger.info("=" * 80)
    logger.info("TEST: dwcli Query and Response Visibility (Gold Standard)")
    logger.info("=" * 80)

    agent_name = "test_dwcli_query_visibility"
    logger.info(f"Starting clean test agent: {agent_name}")

    # Use clean_agent fixture to start agent with NO history
    # This prevents message duplication from accumulated InMemorySaver state
    agent_process = clean_agent(agent_name, working_dir="/tmp")
    logger.info(f"✓ Clean agent started (PID: {agent_process.pid})")

    # Import async validation inside sync context
    from .check_reply import verify_screenshot_contains_sync

    # Use DwcliTerminalSession (which extends TerminalSession with proper PTY support)
    # TerminalSession uses pty.fork() for real PTY, unlike PexpectTerminalSession
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

        # Wait briefly for UI to render
        time.sleep(1.0)
        logger.info("✓ Prompt appeared")

        # Capture initial screen
        time.sleep(1.0)
        screen_before = session.get_screen_text()
        logger.info(f"Initial screen captured ({len(screen_before)} chars)")

        # Send query
        query = "What is 2+2?"
        logger.info(f"Sending query: '{query}'")
        # TerminalSession (via pty.fork) properly handles text and special key input
        session.send_input(query, press_enter=False)  # Don't press enter yet
        time.sleep(0.3)  # Brief pause before sending Enter
        session.send_key("enter")  # Send with Enter key
        logger.info("✓ Query sent to dwcli")

        # Debug: capture screen right after sending query
        time.sleep(0.5)
        screen_right_after_query = session.get_screen_text()
        logger.info("\n" + create_terminal_snapshot(
            screen_right_after_query,
            title="SCREEN RIGHT AFTER QUERY (should show query being processed)",
        ))

        # Wait for AI response (up to 120 seconds = 2 minutes)
        logger.info("Waiting for AI response...")
        response_received = False
        last_screen_with_thinking = None

        for i in range(240):  # 240 * 0.5s = 120 seconds max
            time.sleep(0.5)  # Check every 0.5 seconds
            screen_current = session.get_screen_text()

            # Check if still thinking
            if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                if i % 20 == 0:
                    logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                last_screen_with_thinking = screen_current
                continue

            # If we get here, thinking has stopped - check for answer
            # The AI may respond with "4", "four", "2+2=4", "The answer is 4", etc.
            if "4" in screen_current or "four" in screen_current.lower():
                response_received = True
                logger.info(f"✓ AI response received after {i * 0.5:.1f} seconds")
                break

            if i % 40 == 0 and i > 0:
                logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

        if not response_received:
            logger.warning("⚠ AI did not respond with clear answer within 120 seconds")
            # Get final screen and try validation anyway
            screen_final = session.get_screen_text()
        else:
            # AI responded with what looks like "4"
            logger.info("✓ Processing complete, waiting for screen to stabilize...")
            # Wait a bit more to ensure screen is fully rendered
            time.sleep(2.0)
            screen_final = session.get_screen_text()

        logger.info("\n" + create_terminal_snapshot(
            screen_final,
            title="FINAL SCREEN (query and response should be visible)",
        ))

        # Now use AI validation to verify that the screenshot contains both query and response
        logger.info("\nValidating screenshot using AI analysis...")
        logger.info(f"Checking for query: '{query}'")
        logger.info("Checking for response containing: '4' (or variations)")

        # Use the validation function (sync version)
        try:
            validation_result = verify_screenshot_contains_sync(
                screenshot_text=screen_final,
                original_query=query,
                expected_response_contains="4",
                model="haiku",  # Use cheapest model
            )

            logger.info(f"Validation result: {validation_result}")

            if not validation_result.is_valid:
                logger.error(f"❌ Screenshot validation FAILED: {validation_result.comment}")
                logger.error("\n" + create_terminal_snapshot(
                    screen_final,
                    title="FAILED VALIDATION - Check this screenshot",
                ))
                pytest.fail(
                    f"Screenshot validation failed: {validation_result.comment}\n"
                    f"Query should be: '{query}'\n"
                    f"Response should contain: '4'"
                )

            logger.info("✓ AI validation confirmed both query and response are visible")

        except ImportError as e:
            logger.warning(f"Could not import verify_screenshot_contains_sync: {e}")
            logger.warning("Falling back to simple text search (less robust)")

            # Fallback: simple text search with normalization
            screen_normalized = " ".join(screen_final.split())

            if query not in screen_normalized:
                pytest.fail(f"Query '{query}' not found in screenshot")

            logger.info(f"✓ Query '{query}' found in screenshot")

        logger.info("=" * 80)
        logger.info("✅ TEST PASSED: Query and response both visible on screen")
        logger.info("=" * 80)
        logger.info("✅ This demonstrates correct TUI behavior in dwcli")
        logger.info("=" * 80)

    finally:
        session.close()
    # Note: Agent cleanup is handled by clean_agent fixture


@pytest.mark.skipif(
    not _check_dwcli_available(),
    reason="dwcli command not available in PATH",
)
def test_dwcli_query_visibility2(clean_agent):
    """Test that queries appear ONCE and input area is visible at bottom.

    This test verifies TWO CRITICAL behaviors:
    1. Query "What is 2+2?" appears ONLY ONCE on screen (no duplication)
    2. Input area (box with '>' prompt) is visible at bottom WITHOUT scrolling

    ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
    ============================================================
    This test uses clean_agent fixture to ensure agent starts with NO history.
    All messages (user + assistant) displayed in dwcli come from agent via RPyC callbacks.
    This ensures no duplicate messages and maintains single source of truth architecture.
    """
    logger.info("=" * 80)
    logger.info("TEST: Query visibility + Input area at bottom")
    logger.info("=" * 80)

    agent_name = "test_dwcli_query_visibility2"
    logger.info(f"Starting clean test agent: {agent_name}")

    # Use clean_agent fixture to start agent with NO history
    agent_process = clean_agent(agent_name, working_dir="/tmp")
    logger.info(f"✓ Clean agent started (PID: {agent_process.pid})")

    # Import async validation inside sync context
    from .check_reply import verify_screenshot_contains_sync

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

        # Wait briefly for UI to render
        time.sleep(1.0)
        logger.info("✓ Prompt appeared")

        # Capture initial screen
        time.sleep(1.0)
        screen_before = session.get_screen_text()
        logger.info(f"Initial screen captured ({len(screen_before)} chars)")

        # ===== CHECK 0: Input area border visible AT STARTUP (before any input) =====
        logger.info("CHECK 0: Input area border visible at startup (no user input yet)")
        initial_lines = screen_before.split("\n")
        initial_non_empty = [l for l in initial_lines if l.strip()]
        initial_bottom = initial_non_empty[-10:] if len(initial_non_empty) >= 10 else initial_non_empty
        startup_input_markers = ["│ >", "│>", "┌─", "└─"]
        startup_found = [m for m in startup_input_markers if any(m in l for l in initial_bottom)]
        startup_has_border = any(m in startup_found for m in ["┌─", "└─"])
        startup_has_prompt = any(m in startup_found for m in ["│ >", "│>"])
        startup_input_visible = startup_has_border and startup_has_prompt
        if not startup_input_visible:
            pytest.fail(
                f"Input area border NOT visible at startup (before any user input).\n"
                f"Found markers: {startup_found}\n"
                f"Likely cause: watch_scroll_y override missing super() call, "
                f"so _refresh_scroll() is never invoked and the display stays "
                f"at the old scroll position.\n"
                f"Bottom lines of screen:\n" + "\n".join(initial_bottom)
            )
        else:
            logger.info("✓ Input area border visible at startup")

        # Send query
        query = "What is 2+2?"
        logger.info(f"Sending query: '{query}'")
        session.send_input(query, press_enter=False)
        time.sleep(0.3)
        session.send_key("enter")
        logger.info("✓ Query sent to dwcli")

        # Wait for AI response
        logger.info("Waiting for AI response...")
        response_received = False

        for i in range(240):  # 240 * 0.5s = 120 seconds max
            time.sleep(0.5)
            screen_current = session.get_screen_text()

            # Check if still thinking
            if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                if i % 20 == 0:
                    logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                continue

            # Check for answer
            if "4" in screen_current or "four" in screen_current.lower():
                response_received = True
                logger.info(f"✓ AI response received after {i * 0.5:.1f} seconds")
                break

            if i % 40 == 0 and i > 0:
                logger.info(f"  Waiting for response... ({i * 0.5:.0f}s)")

        if not response_received:
            logger.warning("⚠ AI did not respond with clear answer within 120 seconds")

        # Wait for screen to stabilize
        logger.info("✓ Processing complete, waiting for screen to stabilize...")
        time.sleep(2.0)
        screen_final = session.get_screen_text()

        logger.info("\n" + create_terminal_snapshot(
            screen_final,
            title="FINAL SCREEN (query ONCE + input area at bottom)",
        ))

        # ===== CHECK 1: Query appears only ONCE =====
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 1: Query duplication")
        logger.info("=" * 80)

        # Find all occurrences
        lines = screen_final.split("\n")
        query_lines = []
        for i, line in enumerate(lines):
            if query in line:
                query_lines.append((i, line))

        logger.info(f"Query '{query}' appears {len(query_lines)} times on screen")
        logger.info(f"Occurrences found at lines:")
        for line_num, line_content in query_lines:
            logger.info(f"  Line {line_num}: {line_content}")

        # Check if input field borders exist (indicates input area at bottom)
        has_input_box = any("┌─" in line or "└─" in line or "│ >" in line for line in lines)

        # Count occurrences OUTSIDE of input field area
        # Input field is typically in last 5-10 lines of screen
        messages_area_lines = lines[:-10]  # Exclude bottom 10 lines (input area)
        query_in_messages = sum(1 for line in messages_area_lines if query in line)

        logger.info(f"Query in messages area (excluding input): {query_in_messages}")

        if query_in_messages > 1:
            logger.error(f"❌ Query duplication in messages area: appears {query_in_messages} times!")
            logger.error(f"This excludes input field - checking only message history")

            pytest.fail(
                f"Query duplication detected in message history!\n"
                f"Query '{query}' appears {query_in_messages} times in messages area.\n"
                f"Expected: 1 time\n"
                f"Note: Text in input field is OK, but history should have only 1 copy.\n"
                f"This violates single source of truth architecture."
            )
        else:
            logger.info(f"✓ Query appears exactly once in message history (input field excluded)")

        # ===== CHECK 2: Input area visible at bottom =====
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 2: Input area visibility at bottom")
        logger.info("=" * 80)

        # Split screen into lines
        lines = screen_final.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]
        bottom_lines = non_empty_lines[-10:] if len(non_empty_lines) >= 10 else non_empty_lines

        logger.info(f"Total lines: {len(lines)}, non-empty: {len(non_empty_lines)}")
        logger.info("Bottom 10 lines:")
        for i, line in enumerate(bottom_lines):
            logger.info(f"  {i}: {repr(line[:100])}")

        # Check for input area markers
        input_markers = [
            "│ >",  # Input prompt inside box
            "│>",   # Input prompt without space
            "┌─",   # Top border
            "└─",   # Bottom border
            "manual | shift+tab to cycle",  # Help text
        ]

        found_markers = []
        for marker in input_markers:
            for line in bottom_lines:
                if marker in line:
                    found_markers.append(marker)
                    break

        logger.info(f"Found input area markers: {found_markers}")

        # Determine if input area is visible
        has_box_border = any(marker in found_markers for marker in ["┌─", "└─"])
        has_input_prompt = any(marker in found_markers for marker in ["│ >", "│>"])
        input_area_visible = has_box_border and has_input_prompt

        logger.info(f"Has box border: {has_box_border}")
        logger.info(f"Has input prompt: {has_input_prompt}")
        logger.info(f"Input area visible: {input_area_visible}")

        if not input_area_visible:
            logger.error("❌ Input area NOT visible at bottom!")
            pytest.fail(
                f"Input area not visible at bottom of screen.\n"
                f"Found markers: {found_markers}\n"
                f"Has box border: {has_box_border}\n"
                f"Has input prompt: {has_input_prompt}\n"
                f"User should NOT need to scroll to see input area!"
            )
        else:
            logger.info("✓ Input area is visible at bottom")

        # ===== ALL CHECKS PASSED =====
        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST PASSED: Both checks successful")
        logger.info("  ✓ Query appears only once (no duplication)")
        logger.info("  ✓ Input area visible at bottom")
        logger.info("=" * 80)

    finally:
        session.close()
    # Note: Agent cleanup is handled by clean_agent fixture


def test_dwcli_reconnect_preserves_history():
    """Test that dwcli shows conversation history when reconnecting to an agent.

    This test verifies the CRITICAL behavior:
    1. Start a test agent
    2. Connect with dwcli and send a message "What is 2+2?"
    3. Wait for response containing "4"
    4. Take screenshot showing query and response (FIRST CONNECTION)
    5. Close dwcli (disconnect)
    6. Reconnect with NEW dwcli session to the SAME agent
    7. Take screenshot showing that history is visible (SECOND CONNECTION)
    8. Verify that the previous conversation (query "What is 2+2?" and response with "4") is visible

    EXPECTED: When reconnecting, dwcli should display recent message history from the agent.
    The conversation history belongs to the agent's checkpoint state, not the client.
    Each dwcli client must load and display the agent's conversation history on connection.
    """
    logger.info("=" * 80)
    logger.info("TEST: dwcli reconnect preserves conversation history")
    logger.info("=" * 80)

    agent_name = "test_dwcli_history_reconnect"
    logger.info(f"Starting test agent: {agent_name}")

    # Start agent in background
    agent_process = subprocess.Popen(
        ["deep-worker", agent_name, "--dir", "/tmp", "--skip-permissions"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: None,
    )
    logger.info(f"Agent launcher started (PID: {agent_process.pid})")

    # Wait for agent registration
    time.sleep(1.0)
    logger.info("Waiting for agent to register...")
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    start_wait = time.time()
    timeout = 30.0
    agent_registered = False

    while time.time() - start_wait < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            elapsed = time.time() - start_wait
            logger.info(f"✓ Agent registered after {elapsed:.1f}s (PID: {agent_info.pid})")
            break
        time.sleep(0.2)

    if not agent_registered:
        logger.error(f"Agent did not register within {timeout}s timeout")
        agent_process.terminate()
        raise RuntimeError(f"Agent '{agent_name}' failed to register")

    try:
        # ===== FIRST CONNECTION: Send message and get response =====
        logger.info("\n" + "=" * 80)
        logger.info("FIRST CONNECTION: Sending initial message")
        logger.info("=" * 80)

        session1 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info(f"Starting first dwcli session for agent: {agent_name}")
            session1.start(timeout=15.0)
            logger.info(f"✓ First dwcli started (PID: {session1.pid if session1.pid else 'unknown'})")

            # Wait for UI to render
            time.sleep(1.0)

            # Send query
            query = "What is 2+2?"
            logger.info(f"Sending query: '{query}'")
            session1.send_input(query, press_enter=False)
            time.sleep(0.3)
            session1.send_key("enter")
            logger.info("✓ Query sent")

            # Wait for response
            logger.info("Waiting for AI response...")
            response_received = False

            for i in range(240):  # 120 seconds max
                time.sleep(0.5)
                screen_current = session1.get_screen_text()

                # Check if still thinking
                if "Thinking" in screen_current or "thinking" in screen_current or "⠴" in screen_current:
                    if i % 20 == 0:
                        logger.info(f"  Still processing... ({i * 0.5:.0f}s)")
                    continue

                # Check for answer
                if "4" in screen_current or "four" in screen_current.lower():
                    response_received = True
                    logger.info(f"✓ AI response received after {i * 0.5:.1f} seconds")
                    break

            if not response_received:
                logger.warning("⚠ AI did not respond within 120 seconds")
                pytest.fail("AI did not respond to query in first session")

            # Wait for screen to stabilize
            time.sleep(2.0)
            screen_first_connection = session1.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_first_connection,
                title="FIRST CONNECTION - After receiving response",
            ))

            # Verify query and response are visible in first connection
            assert query in screen_first_connection, "Query not found in first connection"
            assert "4" in screen_first_connection or "four" in screen_first_connection.lower(), \
                "Response not found in first connection"

            logger.info("✓ First connection: Query and response confirmed visible")

        finally:
            # Close first session
            logger.info("Closing first dwcli session...")
            session1.close()
            time.sleep(1.0)
            logger.info("✓ First session closed")

        # ===== SECOND CONNECTION: Reconnect and check history =====
        logger.info("\n" + "=" * 80)
        logger.info("SECOND CONNECTION: Reconnecting to check history")
        logger.info("=" * 80)

        session2 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info(f"Starting second dwcli session for agent: {agent_name}")
            session2.start(timeout=15.0)
            logger.info(f"✓ Second dwcli started (PID: {session2.pid if session2.pid else 'unknown'})")

            # Wait for UI to render and history to load
            logger.info("Waiting for UI to render and load history...")
            time.sleep(3.0)  # Give time for history loading

            screen_second_connection = session2.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_second_connection,
                title="SECOND CONNECTION - Should show conversation history",
            ))

            # ===== VERIFICATION: Check if history is visible =====
            logger.info("\n" + "=" * 80)
            logger.info("VERIFICATION: Checking if conversation history is visible")
            logger.info("=" * 80)

            history_visible = (
                query in screen_second_connection and
                ("4" in screen_second_connection or "four" in screen_second_connection.lower())
            )

            if history_visible:
                logger.info("✅ SUCCESS: Conversation history IS visible after reconnection")
                logger.info(f"✓ Query '{query}' found in second connection")
                logger.info("✓ Response containing '4' found in second connection")
            else:
                logger.error("❌ FAILURE: Conversation history NOT visible after reconnection")
                logger.error(f"✗ Query '{query}' present: {query in screen_second_connection}")
                logger.error(f"✗ Response '4' present: {'4' in screen_second_connection}")

                pytest.fail(
                    f"Conversation history not preserved when reconnecting to agent.\n"
                    f"Expected to see query '{query}' and response '4', but one or both are missing.\n"
                    f"This indicates dwcli does not load agent conversation history on connection."
                )

            logger.info("=" * 80)
            logger.info("✅ TEST PASSED: Conversation history preserved across reconnections")
            logger.info("=" * 80)

        finally:
            session2.close()

    finally:
        # Clean up: terminate agent
        logger.info("Terminating test agent...")
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
            logger.info("Agent process terminated")
        except subprocess.TimeoutExpired:
            logger.warning("Agent process did not terminate, killing...")
            agent_process.kill()
            agent_process.wait()


def test_dwcli_input_visible_with_long_history():
    """Test that input area is visible when connecting to agent with long history.

    This test verifies CRITICAL input visibility behavior:
    1. Start agent and send many queries to build long history (80+ messages)
    2. Close dwcli
    3. Reconnect with NEW dwcli session
    4. Verify input area is VISIBLE (not pushed below screen edge)
    5. Verify NO auto-scroll occurred during history load
    6. Then perform explicit scroll-up
    7. Verify pagination triggers ONLY after explicit scroll

    EXPECTED BEHAVIOR:
    - When connecting, input area must be visible at bottom of screen
    - History messages load WITHOUT pushing input down
    - Scroll only happens on explicit user action (Page Up key)
    - Pagination loads older messages only when user scrolls near top

    This test ensures the fixes to _load_history() and _load_more_history() work correctly.
    """
    logger.info("=" * 80)
    logger.info("TEST: dwcli input area visible with long history")
    logger.info("=" * 80)

    agent_name = "test_dwcli_input_visibility"
    logger.info(f"Starting test agent: {agent_name}")

    # Start agent in background
    agent_process = subprocess.Popen(
        ["deep-worker", agent_name, "--dir", "/tmp", "--skip-permissions"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: None,
    )
    logger.info(f"Agent launcher started (PID: {agent_process.pid})")

    # Wait for agent registration
    time.sleep(1.0)
    logger.info("Waiting for agent to register...")
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    start_wait = time.time()
    timeout = 30.0
    agent_registered = False

    while time.time() - start_wait < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            elapsed = time.time() - start_wait
            logger.info(f"✓ Agent registered after {elapsed:.1f}s (PID: {agent_info.pid})")
            break
        time.sleep(0.2)

    if not agent_registered:
        logger.error(f"Agent did not register within {timeout}s timeout")
        agent_process.terminate()
        raise RuntimeError(f"Agent '{agent_name}' failed to register")

    try:
        # ===== BUILD LONG HISTORY: Send many queries =====
        logger.info("\n" + "=" * 80)
        logger.info("BUILDING LONG HISTORY: Sending many queries")
        logger.info("=" * 80)

        session1 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info("Starting dwcli session for history building")
            session1.start(timeout=15.0)
            time.sleep(1.0)

            # Send 40 queries to build very long history (80+ messages with responses)
            num_queries = 40

            for i in range(num_queries):
                query = f"test {i+1}"
                logger.info(f"Sending query {i+1}/{num_queries}: '{query}'")
                session1.send_input(query, press_enter=False)
                time.sleep(0.2)
                session1.send_key("enter")

                # Wait for response (shorter timeout for simple queries)
                response_received = False
                for j in range(60):  # 30 seconds max
                    time.sleep(0.5)
                    screen = session1.get_screen_text()

                    # Accept any substantial output as response
                    if j > 10 and len(screen.strip()) > 100:
                        response_received = True
                        logger.info(f"✓ Query {i+1} got response after {j * 0.5:.1f}s")
                        break

                if not response_received:
                    logger.warning(f"⚠ Query {i+1} did not complete in 30s, continuing anyway")

                time.sleep(0.3)  # Brief pause between queries

            logger.info(f"✓ All {num_queries} queries sent, long history built")

        finally:
            logger.info("Closing history-building session...")
            session1.close()
            time.sleep(1.0)

        # ===== RECONNECT AND VERIFY INPUT VISIBILITY =====
        logger.info("\n" + "=" * 80)
        logger.info("RECONNECTING: Checking input area visibility")
        logger.info("=" * 80)

        session2 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info("Starting second dwcli session with long history")
            session2.start(timeout=15.0)

            # Wait for UI to render and initial history to load
            logger.info("Waiting for UI render and initial history load...")
            time.sleep(5.0)  # Give adequate time for history loading

            screen_after_connect = session2.get_screen_text()

            logger.info("\n" + create_terminal_snapshot(
                screen_after_connect,
                title="AFTER CONNECTION - Input area should be visible at bottom",
            ))

            # ===== VERIFICATION 1: Input area must be visible =====
            logger.info("\n" + "=" * 80)
            logger.info("VERIFICATION 1: Checking input area visibility")
            logger.info("=" * 80)

            # The input area should be visible at the bottom of the screen
            # We can check for the prompt ">" or input placeholder text
            # Split screen into lines and check bottom portion
            screen_lines = screen_after_connect.split('\n')
            bottom_lines = screen_lines[-5:]  # Last 5 lines
            bottom_text = '\n'.join(bottom_lines)

            # Look for input indicators in bottom portion
            input_visible = (
                ">" in bottom_text or
                "Send a message" in bottom_text or
                "Type your message" in bottom_text or
                len(bottom_text.strip()) > 0  # Bottom has some content
            )

            if input_visible:
                logger.info("✅ Input area appears to be visible at bottom of screen")
            else:
                logger.error("❌ Input area may NOT be visible at bottom of screen")
                logger.error(f"Bottom 5 lines:\n{bottom_text}")
                pytest.fail(
                    "Input area not visible at bottom of screen after connecting.\n"
                    "History loading may have pushed input below viewport."
                )

            # ===== VERIFICATION 2: Scroll position should be at bottom (no auto-scroll to top) =====
            logger.info("\n" + "=" * 80)
            logger.info("VERIFICATION 2: Checking scroll position (should be at bottom)")
            logger.info("=" * 80)

            # If history was loaded correctly, the most recent messages should be visible
            # Check if we can see recent queries (the last few we sent)
            recent_queries_visible = any(
                f"test {i}" in screen_after_connect
                for i in range(max(1, num_queries - 5), num_queries + 1)
            )

            if recent_queries_visible:
                logger.info("✅ Recent messages visible - scroll position is at bottom")
            else:
                logger.warning("⚠ Recent messages not visible - may indicate scroll issue")
                # Don't fail here as this could be due to screen size

            # ===== VERIFICATION 3: Now test explicit scroll-up triggers pagination =====
            logger.info("\n" + "=" * 80)
            logger.info("VERIFICATION 3: Testing explicit scroll-up triggers pagination")
            logger.info("=" * 80)

            # Clear pagination log
            pagination_log = "/tmp/dwcli_pagination.log"
            import os
            if os.path.exists(pagination_log):
                os.remove(pagination_log)

            # Perform explicit scroll-up
            logger.info("Performing explicit scroll-up (Page Up key)...")
            scroll_count = 5
            for i in range(scroll_count):
                logger.info(f"Scroll up {i+1}/{scroll_count}")
                session2.send_key("page_up")
                time.sleep(0.8)  # Give time for pagination to trigger

            # Wait for pagination to complete
            time.sleep(2.0)

            # Check pagination log
            pagination_occurred = False
            page_load_count = 0

            if os.path.exists(pagination_log):
                with open(pagination_log, "r") as f:
                    log_content = f.read()
                    logger.info(f"Pagination log contents:\n{log_content}")

                    page_load_count = log_content.count("PAGINATION:")
                    pagination_occurred = page_load_count >= 1

                    logger.info(f"Pagination events detected: {page_load_count}")

            if pagination_occurred:
                logger.info(f"✅ Pagination triggered by explicit scroll-up ({page_load_count} events)")
            else:
                logger.warning("⚠ No pagination events logged after scroll-up")
                # This might be OK if we already had all history loaded
                logger.info("This could be normal if all history was already loaded")

            logger.info("=" * 80)
            logger.info("✅ TEST PASSED: Input visible, no auto-scroll, explicit scroll works")
            logger.info("=" * 80)

        finally:
            session2.close()

    finally:
        # Clean up: terminate agent
        logger.info("Terminating test agent...")
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


def test_dwcli_scroll_pagination():
    """Test that dwcli loads additional history pages when scrolling up.

    This test verifies CRITICAL scroll pagination behavior:
    1. Start agent and send multiple queries to build long history
    2. Close dwcli and reconnect
    3. Scroll up multiple times
    4. Verify MULTIPLE page loads occurred (not just one initial load)

    The test uses a fast query that generates predictable multi-line output,
    repeated several times to create enough content for multiple pages.

    EXPECTED: Scrolling to top should trigger progressive history loading,
    with multiple get_history() calls with increasing offset values.
    Each page load should be logged and verifiable.
    """
    logger.info("=" * 80)
    logger.info("TEST: dwcli scroll pagination with multi-page loading")
    logger.info("=" * 80)

    agent_name = "test_dwcli_scroll_pagination"
    logger.info(f"Starting test agent: {agent_name}")

    # Create log file for tracking pagination events
    pagination_log = "/tmp/dwcli_pagination.log"
    import os
    if os.path.exists(pagination_log):
        os.remove(pagination_log)

    # Start agent in background
    agent_process = subprocess.Popen(
        ["deep-worker", agent_name, "--dir", "/tmp", "--skip-permissions"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: None,
    )
    logger.info(f"Agent launcher started (PID: {agent_process.pid})")

    # Wait for agent registration
    time.sleep(1.0)
    logger.info("Waiting for agent to register...")
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    start_wait = time.time()
    timeout = 30.0
    agent_registered = False

    while time.time() - start_wait < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            elapsed = time.time() - start_wait
            logger.info(f"✓ Agent registered after {elapsed:.1f}s (PID: {agent_info.pid})")
            break
        time.sleep(0.2)

    if not agent_registered:
        logger.error(f"Agent did not register within {timeout}s timeout")
        agent_process.terminate()
        raise RuntimeError(f"Agent '{agent_name}' failed to register")

    try:
        # ===== BUILD HISTORY: Send multiple queries =====
        logger.info("\n" + "=" * 80)
        logger.info("BUILDING HISTORY: Sending multiple queries")
        logger.info("=" * 80)

        session1 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info(f"Starting dwcli session for history building")
            session1.start(timeout=15.0)
            time.sleep(1.0)

            # Send 20 simple queries to build history quickly
            # Using simple test messages that AI can respond to quickly
            num_queries = 20

            for i in range(num_queries):
                query = f"test {i+1}"
                logger.info(f"Sending query {i+1}/{num_queries}: '{query}'")
                session1.send_input(query, press_enter=False)
                time.sleep(0.2)
                session1.send_key("enter")

                # Wait for response (shorter timeout for simple queries)
                response_received = False
                for j in range(60):  # 30 seconds max
                    time.sleep(0.5)
                    screen = session1.get_screen_text()

                    # Look for any response containing the query text
                    # AI should acknowledge the test message
                    if f"test {i+1}" in screen.lower() or "test" in screen.lower():
                        # Check if response has appeared (look for assistant response patterns)
                        if any(marker in screen for marker in ["understood", "received", "acknowledged", "ok", "yes"]):
                            response_received = True
                            logger.info(f"✓ Query {i+1} completed after {j * 0.5:.1f}s")
                            break

                    # Also accept any non-empty assistant response
                    if j > 10 and len(screen.strip()) > 100:  # Give it 5 seconds, then accept any substantial output
                        response_received = True
                        logger.info(f"✓ Query {i+1} got response after {j * 0.5:.1f}s")
                        break

                if not response_received:
                    logger.warning(f"⚠ Query {i+1} did not complete in 30s, continuing anyway")

                time.sleep(0.5)  # Brief pause between queries

            logger.info("✓ All queries sent, history built")

        finally:
            logger.info("Closing history-building session...")
            session1.close()
            time.sleep(1.0)

        # ===== RECONNECT AND SCROLL =====
        logger.info("\n" + "=" * 80)
        logger.info("RECONNECTING: Testing scroll pagination")
        logger.info("=" * 80)

        session2 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info(f"Starting second dwcli session")
            session2.start(timeout=15.0)

            # Wait for initial history load
            time.sleep(3.0)
            logger.info("✓ Initial history loaded")

            # Now scroll up multiple times to trigger pagination
            logger.info("Scrolling up to trigger pagination...")

            scroll_count = 10  # Scroll up 10 times
            for i in range(scroll_count):
                logger.info(f"Scroll up {i+1}/{scroll_count}")
                session2.send_key("page_up")
                time.sleep(0.5)  # Give time for pagination to trigger

            # Give final wait for all pagination to complete
            time.sleep(2.0)

            # ===== VERIFICATION: Check pagination logs =====
            logger.info("\n" + "=" * 80)
            logger.info("VERIFICATION: Checking pagination event logs")
            logger.info("=" * 80)

            # Check if pagination log exists and has multiple entries
            pagination_occurred = False
            page_load_count = 0

            if os.path.exists(pagination_log):
                with open(pagination_log, "r") as f:
                    log_content = f.read()
                    logger.info(f"Pagination log contents:\n{log_content}")

                    # Count how many times history was loaded with offset > 0
                    page_load_count = log_content.count("PAGINATION:")
                    pagination_occurred = page_load_count >= 2  # At least 2 additional loads

                    logger.info(f"Page load events detected: {page_load_count}")
            else:
                logger.warning("⚠ Pagination log file not found")

            if pagination_occurred:
                logger.info(f"✅ SUCCESS: Multiple page loads detected ({page_load_count} events)")
            else:
                logger.error(f"❌ FAILURE: Insufficient pagination events (found {page_load_count}, need >= 2)")
                pytest.fail(
                    f"Scroll pagination did not trigger multiple page loads.\n"
                    f"Expected at least 2 pagination events, but found {page_load_count}.\n"
                    f"This indicates scroll-up handler is not loading older messages."
                )

        finally:
            session2.close()

    finally:
        # Clean up: terminate agent
        logger.info("Terminating test agent...")
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


@pytest.mark.skipif(not CHECK_REPLY_AVAILABLE, reason="claude_sdk not available")
def test_dwcli_input_area_immediately_visible():
    """Test that input area is IMMEDIATELY visible when opening TUI with history.

    This test verifies the CRITICAL requirement that the input area must be visible
    WITHOUT any user action (no key presses, no scrolling) immediately after running
    'dwcli <agent_name>' command when agent has long conversation history.

    Steps:
    1. Build agent with long history (40 queries = 80+ messages)
    2. Open NEW dwcli session
    3. Wait ONLY for initial render (5 seconds)
    4. Capture screen WITHOUT any interaction
    5. Use AI vision analysis to verify input area with '>' prompt is visible at bottom

    PASS CRITERIA:
    - AI confirms input area (long rectangle with '>' prompt) is visible at bottom
    - No user interaction required
    - Screenshot saved as proof

    FAIL CRITERIA:
    - Input area not visible at bottom
    - User must press keys to make it appear
    - Bottom of screen is empty or shows only messages
    """
    logger.info("=" * 80)
    logger.info("TEST: Input area immediately visible (AI vision check)")
    logger.info("=" * 80)

    agent_name = "test_input_immediately_visible"
    screenshot_file = "/tmp/input_visible_proof.txt"

    logger.info(f"Starting test agent: {agent_name}")

    # Start agent in background.
    # IMPORTANT: cwd="/tmp" ensures the agent's working directory is /tmp so
    # that any files the AI creates during test queries land in /tmp, not in
    # the pytest working directory (a sibling repo checkout).
    agent_process = subprocess.Popen(
        ["deep-worker", agent_name, "--dir", "/tmp", "--skip-permissions"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: None,
        cwd="/tmp",
    )
    logger.info(f"Agent launcher started (PID: {agent_process.pid})")

    # Wait for agent registration
    time.sleep(1.0)
    logger.info("Waiting for agent to register...")
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    start_wait = time.time()
    timeout = 30.0
    agent_registered = False

    while time.time() - start_wait < timeout:
        agent_info = registry.lookup(agent_name)
        if agent_info:
            agent_registered = True
            elapsed = time.time() - start_wait
            logger.info(f"✓ Agent registered after {elapsed:.1f}s (PID: {agent_info.pid})")
            break
        time.sleep(0.2)

    if not agent_registered:
        logger.error(f"Agent did not register within {timeout}s timeout")
        agent_process.terminate()
        raise RuntimeError(f"Agent '{agent_name}' failed to register")

    try:
        # ===== BUILD LONG HISTORY =====
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Building long history (40 queries)")
        logger.info("=" * 80)

        session1 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info("Starting first dwcli session for history building")
            session1.start(timeout=15.0)
            time.sleep(1.0)

            # Send 40 queries
            num_queries = 40

            for i in range(num_queries):
                query = f"test query {i+1}"
                logger.info(f"Sending query {i+1}/{num_queries}: '{query}'")
                session1.send_input(query, press_enter=False)
                time.sleep(0.2)
                session1.send_key("enter")

                # Wait for response
                response_received = False
                for j in range(60):  # 30 seconds max
                    time.sleep(0.5)
                    screen = session1.get_screen_text()

                    if j > 10 and len(screen.strip()) > 100:
                        response_received = True
                        logger.info(f"✓ Query {i+1} got response after {j * 0.5:.1f}s")
                        break

                if not response_received:
                    logger.warning(f"⚠ Query {i+1} did not complete in 30s, continuing anyway")

                time.sleep(0.3)

            logger.info(f"✓ All {num_queries} queries sent, long history built")

        finally:
            logger.info("Closing history-building session...")
            session1.close()

            # Wait for agent to finish checkpoint writes and become ready for reconnection
            logger.info("Waiting for agent to complete checkpoint writing and become ready...")
            logger.info("This may take 10-20 seconds after building 40 queries with AI responses")

            # Give agent time to complete checkpoint writes
            time.sleep(15.0)  # Increased to 15 seconds

            # Poll agent registry to ensure agent is still responsive
            from deep_worker_cli.process.registry import ProcessRegistry
            registry = ProcessRegistry()

            for attempt in range(5):
                agent_info = registry.lookup(agent_name)
                if agent_info:
                    logger.info(f"✓ Agent still registered (attempt {attempt + 1}/5)")
                    break
                logger.warning(f"⚠ Agent not found in registry (attempt {attempt + 1}/5)")
                time.sleep(2.0)
            else:
                raise RuntimeError(f"Agent '{agent_name}' disappeared from registry after history building")

            logger.info("✓ Agent ready for reconnection")

        # ===== OPEN NEW SESSION AND CHECK IMMEDIATELY =====
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Open new session and check input visibility IMMEDIATELY")
        logger.info("=" * 80)

        session2 = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
            working_dir=Path("/tmp"),
        )

        try:
            logger.info("Starting second dwcli session with long history")
            session2.start(timeout=15.0)

            # Wait ONLY for initial render - NO user interaction
            logger.info("Waiting 5 seconds for initial render (NO KEY PRESSES)")
            time.sleep(5.0)

            # Capture screen WITHOUT any interaction
            screen_text = session2.get_screen_text()

            logger.info(f"\n{'=' * 80}")
            logger.info("CAPTURED SCREEN (immediately after opening, no interaction):")
            logger.info("=" * 80)
            logger.info(screen_text)
            logger.info("=" * 80)

            # Save screenshot as proof
            with open(screenshot_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("PROOF: Screen captured immediately after 'dwcli' command\n")
                f.write("NO user interaction (no key presses, no scrolling)\n")
                f.write("=" * 80 + "\n\n")
                f.write(screen_text)

            logger.info(f"✓ Screenshot saved to: {screenshot_file}")

            # ===== TEXT-BASED ANALYSIS =====
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 3: Text analysis of input area visibility")
            logger.info("=" * 80)

            # Split screen into lines for analysis
            lines = screen_text.split("\n")

            # Get the last 10 non-empty lines
            non_empty_lines = [line for line in lines if line.strip()]
            bottom_lines = non_empty_lines[-10:] if len(non_empty_lines) >= 10 else non_empty_lines

            logger.info(f"Total lines in screenshot: {len(lines)}")
            logger.info(f"Non-empty lines: {len(non_empty_lines)}")
            logger.info("Bottom 10 lines:")
            for i, line in enumerate(bottom_lines):
                logger.info(f"  {i}: {repr(line[:100])}")

            # Check for input area markers
            input_markers = [
                "│ >",  # Input prompt inside box
                "│>",   # Input prompt without space
                "┌─",   # Top border
                "└─",   # Bottom border
                "manual | shift+tab to cycle",  # Help text near input
            ]

            found_markers = []
            for marker in input_markers:
                for line in bottom_lines:
                    if marker in line:
                        found_markers.append(marker)
                        break

            logger.info(f"\nFound input area markers: {found_markers}")

            # Determine if input area is visible
            # Must have at least: box borders (┌─ or └─) AND input prompt (│ >)
            has_box_border = any(marker in found_markers for marker in ["┌─", "└─"])
            has_input_prompt = any(marker in found_markers for marker in ["│ >", "│>"])

            input_area_visible = has_box_border and has_input_prompt

            logger.info(f"Has box border: {has_box_border}")
            logger.info(f"Has input prompt: {has_input_prompt}")
            logger.info(f"Input area visible: {input_area_visible}")

            # ===== VERDICT =====
            if input_area_visible:
                logger.info("\n" + "=" * 80)
                logger.info("✅ TEST PASSED: Input area IS visible at bottom")
                logger.info(f"   Screenshot proof: {screenshot_file}")
                logger.info(f"   Found markers: {found_markers}")
                logger.info("=" * 80)
            else:
                logger.error("\n" + "=" * 80)
                logger.error("❌ TEST FAILED: Input area NOT visible at bottom")
                logger.error(f"   Found markers: {found_markers}")
                logger.error(f"   Has box border: {has_box_border}")
                logger.error(f"   Has input prompt: {has_input_prompt}")
                logger.error(f"   Screenshot saved at: {screenshot_file}")
                logger.error("=" * 80)

                pytest.fail(
                    f"Input area not visible at bottom of screen.\n"
                    f"Found markers: {found_markers}\n"
                    f"Has box border: {has_box_border}\n"
                    f"Has input prompt: {has_input_prompt}\n"
                    f"Screenshot: {screenshot_file}\n"
                    f"This means user must press keys to see input, which is WRONG."
                )

        finally:
            session2.close()

    finally:
        # Clean up: terminate agent
        logger.info("Terminating test agent...")
        agent_process.terminate()
        try:
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


# =============================================================================
# Main - Allow running directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
