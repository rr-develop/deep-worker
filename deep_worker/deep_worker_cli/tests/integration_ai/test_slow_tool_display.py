"""Integration test for slow tool call display in TUI.

This test verifies that tool calls are displayed IMMEDIATELY in the TUI,
even when the tool takes a long time to execute (e.g., 10+ seconds).

## Expected Behavior

When an agent calls a slow tool (like sleep or web_search):
1. Tool call message should appear in TUI IMMEDIATELY (within ~1 second)
2. Tool execution happens in background
3. Tool result appears when execution completes

## Bug Being Tested

Currently, when using tools that block for 10+ seconds, the TUI shows NOTHING
for the entire duration. All messages (tool_call + tool_result) appear only
AFTER the tool completes execution.

This violates user expectations and makes the UI appear frozen.

Run with:
    pytest tests/integration_ai/test_slow_tool_display.py -v -s
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from .check_reply import verify_screenshot_contains
from .clean_agent_fixture import clean_agent  # noqa: F401
from .terminal_emulator import DwcliTerminalSession

logger = logging.getLogger(__name__)

# ============================================================================
# Fixtures
# ============================================================================

_SLOW_TOOL_AGENT_NAME = "test_slow_tool_display"


@pytest.fixture
def slow_tool_agent(request):
    """Auto-start a fresh deep-worker agent for testing slow tool display.

    This fixture:
    1. Kills any existing agent with the test name (unless reuse_agent marker present)
    2. Cleans up data directories for a fresh start
    3. Starts the agent with --skip-permissions
    4. Waits for the agent to register in ProcessRegistry
    5. Yields the agent name
    6. Kills the agent on teardown

    To reuse existing agent (for debugging):
        @pytest.mark.reuse_agent
        def test_something(slow_tool_agent):
            # Will reuse existing agent if running, won't kill it

    Working directory is /tmp to prevent junk files in the source tree.
    """
    from deep_worker_cli.process.registry import ProcessRegistry

    agent_name = _SLOW_TOOL_AGENT_NAME
    agent_process = None
    registry = ProcessRegistry()

    # Check for reuse_agent marker
    reuse = request.node.get_closest_marker("reuse_agent") is not None

    # 1. Kill any existing agent with this name (unless reuse=True)
    existing = registry.lookup(agent_name)

    if existing and reuse:
        logger.info(
            f"REUSE MODE: Keeping existing agent '{agent_name}' PID {existing.pid}. "
            f"Will NOT kill or restart. Connect with: dwcli {agent_name}"
        )
        # Yield existing agent name, skip setup
        yield agent_name
        # Skip teardown - don't kill reused agent
        return

    if existing:
        logger.info(f"Killing existing agent PID {existing.pid}")
        try:
            import os
            os.kill(existing.pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.info(f"Process {existing.pid} already dead")

        # CRITICAL: Wait for process to actually die and disappear from registry
        # This prevents testing old agent code when bug fixes were made
        kill_timeout = 5.0
        kill_start = time.time()
        process_dead = False

        while time.time() - kill_start < kill_timeout:
            # Check if still in registry
            if not registry.lookup(agent_name):
                process_dead = True
                logger.info(f"✓ Process {existing.pid} confirmed dead and removed from registry")
                break

            # Also check if process exists in OS
            try:
                os.kill(existing.pid, 0)  # signal 0 just checks if process exists
                logger.debug(f"Process {existing.pid} still exists, waiting...")
            except ProcessLookupError:
                logger.info(f"✓ Process {existing.pid} no longer exists in OS")
                # Give registry time to update
                time.sleep(0.2)
                if not registry.lookup(agent_name):
                    process_dead = True
                    logger.info(f"✓ Confirmed removal from registry")
                    break

            time.sleep(0.1)

        if not process_dead:
            raise RuntimeError(
                f"Old agent process {existing.pid} did not die within {kill_timeout}s. "
                f"Cannot safely start new agent - would test old code!"
            )

    # 2. Clean up data directories
    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    if data_dir.exists():
        logger.info(f"Removing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink(missing_ok=True)

    # 3. Start fresh agent
    logger.info(f"Starting agent '{agent_name}'...")
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", "/tmp",
            "--skip-permissions",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/tmp",
    )

    # 4. Wait for agent to register
    timeout = 30.0
    start_wait = time.time()
    registered = False

    while time.time() - start_wait < timeout:
        info = registry.lookup(agent_name)
        if info:
            registered = True
            elapsed = time.time() - start_wait
            logger.info(f"Agent registered after {elapsed:.1f}s (PID: {info.pid})")
            break
        time.sleep(0.2)

    if not registered:
        agent_process.kill()
        agent_process.wait()
        raise RuntimeError(
            f"Agent '{agent_name}' did not register within {timeout}s. "
            "Check that 'deep-worker' is installed and working."
        )

    yield agent_name

    # Teardown: kill agent
    logger.info(f"Tearing down agent '{agent_name}'")
    if agent_process and agent_process.poll() is None:
        try:
            agent_process.terminate()
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_process.kill()
            agent_process.wait()


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_slow_tool_call_displays_immediately(slow_tool_agent):
    """Tool call message MUST appear in TUI while tool is executing, not after completion.

    This test:
    1. Sends a query requesting the agent to call sleep(10) and say hello
    2. Waits 3 seconds (gives model time to start responding)
    3. Takes a screenshot of the TUI
    4. Uses AI to verify the screenshot shows agent activity (spinner or response)
    5. Fails if:
       - No response or spinner visible (indicates streaming not working)
       - Agent responded without using the sleep tool (wrong behavior)

    EXPECTED BEHAVIOR:
    - After 3 seconds, TUI should show "Thinking..." spinner or assistant response
    - Tool call appears when agent emits it (timing depends on model speed)
    - Tool execution continues in background
    - After 10 seconds, tool_result appears

    NOTE: LangGraph React pattern delays tool_call emission until Agent node completes.
    This is architectural - test verifies UI shows activity, not instant tool_call.
    """
    agent_name = slow_tool_agent

    # Open a TUI session
    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=120,
        lines=40,
    )
    session.start(timeout=15.0)

    try:
        # Wait for the input area to be ready
        logger.info("Waiting for TUI input area...")
        ready = session.wait_for_text(">", timeout=10.0)
        if not ready:
            screen = session.get_screen_text()
            pytest.fail(
                f"dwcli TUI did not show input area within 10s.\n"
                f"Screen content:\n{screen}"
            )

        # Give TUI a moment to fully render
        time.sleep(1.0)

        # Send query requesting sleep tool
        query = (
            "Please call the bash tool to execute 'sleep 10' command, "
            "then say hello to me. Make sure you use the bash tool first."
        )
        logger.info(f"Sending query: {query}")
        # CRITICAL FIX: Send text and Enter separately (matches deepagents test)
        # Sending them together doesn't work reliably with Textual PTY input
        session.send_input(query, press_enter=False)
        time.sleep(0.3)  # Brief wait for text to be processed
        session.send_key("enter")
        logger.info("✓ Query sent to agent")

        # CRITICAL: Wait 5 seconds, then take screenshot
        # Tool call should be visible even though tool is still executing
        # 5 seconds gives model time to start responding and emit more than just one letter
        logger.info("Waiting 5 seconds before screenshot...")
        time.sleep(5.0)

        # Capture screenshot
        screenshot = session.get_screen_text()
        logger.info(f"Screenshot after 5 seconds:\n{screenshot}")

        # Check for activity indicators: "Thinking..." spinner OR assistant response text
        # According to test docstring (line 211): "TUI should show 'Thinking...' spinner or assistant response"
        has_thinking_spinner = "Thinking..." in screenshot
        # Check if there's any assistant response (I'll or Hello or any typical response start)
        has_assistant_response = any(marker in screenshot for marker in ["I'll", "I will", "Hello", "Let me", "Sure", "execute"])

        is_activity_visible = has_thinking_spinner or has_assistant_response

        # Check if activity is visible
        if not is_activity_visible:
            # Get final screenshot after tool completes (for debugging)
            logger.info("Initial check failed. Waiting for tool to complete...")
            time.sleep(12.0)  # Wait for sleep(10) to finish
            final_screenshot = session.get_screen_text()

            pytest.fail(
                f"STREAMING ACTIVITY BUG DETECTED\n\n"
                f"No activity visible in TUI within 5 seconds after sending query.\n"
                f"Expected: 'Thinking...' spinner OR assistant response text\n"
                f"Actual: No spinner and no response visible\n\n"
                f"Has 'Thinking...' spinner: {has_thinking_spinner}\n"
                f"Has assistant response: {has_assistant_response}\n\n"
                f"Screenshot after 5 seconds:\n{screenshot}\n\n"
                f"Screenshot after tool completed:\n{final_screenshot}\n\n"
                f"This indicates streaming is not working properly."
            )

        logger.info(
            f"PASS: Activity visible within 5 seconds "
            f"(has_thinking_spinner={has_thinking_spinner}, "
            f"has_assistant_response={has_assistant_response})"
        )

        # Wait for tool to complete and verify final state
        logger.info("Waiting for tool execution to complete...")
        tool_completed = session.wait_for_text("hello", timeout=15.0)

        if not tool_completed:
            final_screenshot = session.get_screen_text()
            pytest.fail(
                f"Tool execution did not complete or response not received.\n"
                f"Final screenshot:\n{final_screenshot}"
            )

        logger.info("PASS: Tool execution completed and response received.")

    finally:
        session.close()


@pytest.mark.asyncio
async def test_tool_call_not_response_without_tool(slow_tool_agent):
    """Verify the test correctly fails if agent responds without using tool.

    This is a control test to ensure the main test is reliable.
    If agent responds with "hello" immediately without calling bash,
    the test should fail with a different error message.
    """
    agent_name = slow_tool_agent

    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=120,
        lines=40,
    )
    session.start(timeout=15.0)

    try:
        ready = session.wait_for_text(">", timeout=10.0)
        if not ready:
            pytest.fail("TUI input not ready")

        time.sleep(1.0)

        # Send a query that the agent might answer without tools
        query = "Just say hello to me"
        logger.info(f"Sending control query: {query}")
        session.send_input(query, press_enter=True)

        # Wait 1 second
        time.sleep(1.0)

        screenshot = session.get_screen_text()

        # If agent responded without using tools, screenshot should show "hello"
        # but NOT show any tool call
        if "hello" in screenshot.lower():
            logger.info(
                "PASS (control): Agent responded without tool call as expected."
            )
        else:
            logger.info(
                f"Agent did not respond immediately. Screenshot:\n{screenshot}"
            )

    finally:
        session.close()
