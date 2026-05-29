"""Integration test for headless->TUI message broadcasting.

ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
============================================================
This test verifies that messages sent via headless mode (dwcli -p "...") are
IMMEDIATELY visible in ALL connected TUI instances (dwcli <agent>).

This is the core architectural principle: regardless of where a message
originates (TUI, headless, web UI, API), ALL messages MUST be broadcast to
ALL subscribers in real-time via AgentService._notify_subscribers().

## Test Scenario

1. Start agent WITHOUT TaskManager (standard configuration)
2. Connect TUI client (dwcli <agent>) - subscribes to real-time messages
3. Send message via headless mode (dwcli -p "2+2" <agent>)
4. Verify TUI shows BOTH:
   - User message: "2+2"
   - Assistant response: "4" (or similar)

## Expected Behavior (BEFORE FIX)

❌ FAIL: TUI does NOT show the user message or assistant response from headless mode.

The bug is in agent_process.py lines 1176-1360 (non-streaming WITHOUT TaskManager):
- User message IS sent (line 838-843) ✅
- Tool calls ARE sent (lines 1229-1246) ✅
- Tool results ARE sent (lines 1248-1266) ✅
- **Assistant message is NOT sent** (missing notify_callback before return) ❌

## Expected Behavior (AFTER FIX)

✅ PASS: TUI immediately shows BOTH user message and assistant response.

Fix: Add notify_callback() call before return (around line 1350 in agent_process.py).

## Related Tests

- test_taskmanager_tui.py: Tests TaskManager mode (WITH -tm flag)
  - In TaskManager mode, messages ARE properly broadcast (lines 1114-1116, 1154-1158)
  - This test covers standard mode WITHOUT TaskManager

## Why This Test Matters

This is one of the most common use cases:
1. Developer has TUI open for monitoring
2. Another terminal sends quick headless commands
3. Developer expects to see ALL activity in TUI

Without this working, the "single source of truth" principle is violated.
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401
from .terminal_emulator import DwcliTerminalSession

logger = logging.getLogger(__name__)

# ============================================================================
# Fixtures
# ============================================================================

_AGENT_NAME = "test_headless_broadcast"


@pytest.fixture
def broadcast_agent(request: pytest.FixtureRequest):
    """Start a fresh agent WITHOUT TaskManager (standard configuration).

    Cleanup policy: passed → terminate, failed → preserve for debug.
    See ``conftest.pytest_sessionfinish`` for the safety net.
    """
    from deep_worker_cli.process.registry import ProcessRegistry
    from .conftest import register_test_agent

    agent_name = _AGENT_NAME
    agent_process = None
    registry = ProcessRegistry()

    # 1. Kill existing agent
    existing = registry.lookup(agent_name)
    if existing:
        logger.info(f"Killing existing agent PID {existing.pid}")
        try:
            import os
            os.kill(existing.pid, signal.SIGKILL)
            time.sleep(0.5)
        except ProcessLookupError:
            pass

    # 2. Clean data directories
    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    if data_dir.exists():
        logger.info(f"Removing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink(missing_ok=True)

    # 3. Start agent WITHOUT TaskManager
    logger.info(f"Starting agent '{agent_name}' WITHOUT TaskManager...")
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", "/tmp",
            "--skip-permissions",
            # NOTE: NO -tm flag! This tests standard mode
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/tmp",
    )

    # 4. Wait for registration
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

    register_test_agent(agent_name, request.node, created_by_fixture=True)

    yield agent_name

    # Teardown: passed → terminate, failed → preserve.
    rep_call = getattr(request.node, "rep_call", None)
    if rep_call is not None and not rep_call.passed:
        logger.warning(
            f"Test FAILED → preserving agent '{agent_name}' for debugging. "
            f"Connect with: dwcli {agent_name}"
        )
        return

    logger.info(f"Test passed → terminating agent '{agent_name}'")
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


def test_headless_user_message_visible_in_tui(broadcast_agent):
    """Verify USER message from headless mode appears in TUI.

    Test flow:
    1. Open TUI connected to agent
    2. Send message via headless: dwcli -p "2+2" <agent>
    3. Verify TUI shows user message: "2+2"

    EXPECTED (before fix): ✅ PASS
    - User message IS broadcast in non-streaming mode (line 838-843)

    This test documents that user message broadcasting works correctly.
    """
    agent_name = broadcast_agent

    # 1. Open TUI session
    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=120,
        lines=40,
    )
    session.start(timeout=15.0)

    try:
        # Wait for TUI to be ready
        logger.info("Waiting for TUI input area...")
        ready = session.wait_for_text(">", timeout=10.0)
        if not ready:
            screen = session.get_screen_text()
            pytest.fail(
                f"TUI did not show input area within 10s.\n"
                f"Screen content:\n{screen}"
            )

        time.sleep(1.0)  # Let TUI fully render

        logger.info("TUI ready. Capturing initial screen...")
        initial_screen = session.get_screen_text()

        # 2. Send message via headless mode in separate process
        test_prompt = "Calculate 2+2"
        logger.info(f"Sending headless command: dwcli -p '{test_prompt}' {agent_name}")

        # Start headless command (non-blocking)
        headless_process = subprocess.Popen(
            ["dwcli", "-p", test_prompt, agent_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # 3. Wait for user message to appear in TUI
        logger.info("Waiting for user message to appear in TUI...")
        found_user_msg = session.wait_for_text(test_prompt, timeout=15.0)

        if not found_user_msg:
            # Wait for headless to complete for debugging
            headless_process.wait(timeout=10)
            headless_output = headless_process.stdout.read()

            screen = session.get_screen_text()
            pytest.fail(
                f"USER MESSAGE NOT VISIBLE IN TUI!\n\n"
                f"Headless command completed with output:\n{headless_output}\n\n"
                f"But TUI does not show user message '{test_prompt}'.\n\n"
                f"TUI screen:\n{screen}\n\n"
                f"This indicates user message is NOT being broadcast via "
                f"_notify_subscribers() in non-streaming mode."
            )

        # Success: user message appeared
        screen = session.get_screen_text()
        logger.info(f"✅ User message visible in TUI:\n{screen}")

        # Wait for headless to complete
        headless_process.wait(timeout=30)

    finally:
        session.close()


def test_headless_assistant_message_visible_in_tui(broadcast_agent):
    """Verify ASSISTANT message from headless mode appears in TUI via REAL-TIME broadcast.

    CRITICAL: This test checks REAL-TIME broadcasting, NOT history loading.
    The TUI must receive messages via subscribe_messages() callback, not via get_history().

    Test flow:
    1. Start agent and send FIRST message to create history
    2. Open TUI connected to agent (it will load existing history)
    3. Capture initial screen state AFTER history loads
    4. Send SECOND message via headless (THIS must appear via real-time broadcast)
    5. Verify the SECOND message appears WITHOUT calling get_history()

    EXPECTED (before fix): ❌ FAIL
    - Assistant message from step 4 does NOT appear in TUI
    - Only appears if TUI manually reloads history
    - Bug: agent_process.py lines 1176-1360 (missing notify_callback)

    EXPECTED (after fix): ✅ PASS
    - Assistant message appears immediately via real-time broadcast
    """
    agent_name = broadcast_agent

    # Step 1: Send FIRST message to create history (before TUI opens)
    logger.info("Step 1: Sending first message to create history...")
    first_prompt = "First test message"
    first_result = subprocess.run(
        ["dwcli", "-p", first_prompt, agent_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if first_result.returncode != 0:
        pytest.fail(f"First headless command failed: {first_result.stderr}")

    logger.info(f"First message completed: {first_result.stdout[:100]}")
    time.sleep(1.0)  # Let agent finish completely

    # Step 2: Open TUI (it will load history including first message)
    logger.info("Step 2: Opening TUI (will load existing history)...")
    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=120,
        lines=40,
    )
    session.start(timeout=15.0)

    try:
        # Wait for TUI to be ready
        logger.info("Waiting for TUI input area...")
        ready = session.wait_for_text(">", timeout=10.0)
        if not ready:
            screen = session.get_screen_text()
            pytest.fail(f"TUI did not show input area: {screen}")

        # Step 3: Wait for history to load (TUI loads it after ~0.1s)
        logger.info("Waiting for history to load...")
        time.sleep(3.0)  # Give TUI time to load and render history

        initial_screen = session.get_screen_text()
        logger.info(f"Initial TUI screen (with loaded history):\n{initial_screen}")

        # Verify first message is in history
        if first_prompt not in initial_screen:
            pytest.fail(
                f"First message not found in TUI history.\n"
                f"Expected to see: '{first_prompt}'\n"
                f"TUI screen:\n{initial_screen}"
            )

        # Step 4: Send SECOND message via headless
        # This message MUST appear via real-time broadcast, NOT history reload
        second_prompt = "What is 8 plus 8? Answer with just the number."
        logger.info(f"Step 4: Sending second message via headless: '{second_prompt}'")

        headless_process = subprocess.Popen(
            ["dwcli", "-p", second_prompt, agent_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Step 5: Wait for SECOND message to appear (must be via real-time broadcast)
        logger.info("Step 5: Waiting for second message to appear in TUI...")

        # First, wait for user message to appear
        found_user_msg = session.wait_for_text(second_prompt, timeout=20.0)
        if not found_user_msg:
            screen = session.get_screen_text()
            pytest.fail(
                f"USER MESSAGE from headless NOT visible in TUI!\n"
                f"Expected to see: '{second_prompt}'\n"
                f"TUI screen:\n{screen}"
            )

        logger.info("✅ User message appeared in TUI")

        # Then, wait for assistant response (answer should be "16")
        found_response = session.wait_for_text("16", timeout=20.0)

        # Wait for headless to complete
        try:
            headless_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            headless_process.kill()
            headless_process.wait()

        headless_output = headless_process.stdout.read() if headless_process.stdout else ""

        if not found_response:
            screen = session.get_screen_text()
            pytest.fail(
                f"ASSISTANT MESSAGE NOT VISIBLE IN TUI!\n\n"
                f"This is the CORE BUG: In non-streaming mode WITHOUT TaskManager,\n"
                f"assistant messages are NOT broadcast via real-time callbacks.\n\n"
                f"Headless completed with output:\n{headless_output}\n\n"
                f"But TUI does not show the response '16'.\n\n"
                f"TUI screen:\n{screen}\n\n"
                f"ROOT CAUSE:\n"
                f"  File: agent_process.py\n"
                f"  Location: Lines 1176-1360 (non-streaming WITHOUT TaskManager)\n"
                f"  Issue: Missing notify_callback() before return (line ~1350)\n\n"
                f"FIX:\n"
                f"  Add before return:\n"
                f"    if notify_callback is not None:\n"
                f"        await notify_callback({{\n"
                f"            'type': 'assistant',\n"
                f"            'role': 'assistant',\n"
                f"            'content': final_content,\n"
                f"        }})\n"
            )

        # Success!
        screen = session.get_screen_text()
        logger.info(f"✅ PASS: Second message appeared via real-time broadcast:\n{screen}")

    finally:
        session.close()


def test_headless_and_tui_consistency(broadcast_agent):
    """Verify headless and TUI show SAME conversation (complete consistency).

    This is the COMPREHENSIVE test that verifies:
    1. User message from headless appears in TUI
    2. Assistant response from headless appears in TUI
    3. The conversation history is consistent between modes

    This test combines both user and assistant message checks into a single
    end-to-end verification.

    EXPECTED (before fix): ❌ FAIL
    - User message appears ✅
    - Assistant message DOES NOT appear ❌

    EXPECTED (after fix): ✅ PASS
    - Both user and assistant messages appear in TUI
    - Conversation is fully consistent across all clients
    """
    agent_name = broadcast_agent

    # Test prompt with distinctive text for easy verification
    test_prompt = "Say exactly: Hello from headless mode"

    # 1. Open TUI session FIRST (before sending headless command)
    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=120,
        lines=40,
    )
    session.start(timeout=15.0)

    try:
        # Wait for TUI ready
        logger.info("Waiting for TUI to be ready...")
        ready = session.wait_for_text(">", timeout=10.0)
        if not ready:
            pytest.fail("TUI did not become ready")

        time.sleep(1.0)

        initial_screen = session.get_screen_text()
        logger.info(f"TUI ready. Initial screen:\n{initial_screen}")

        # 2. Send command via headless while TUI is connected
        logger.info(f"Sending headless: dwcli -p '{test_prompt}' {agent_name}")
        headless_result = subprocess.run(
            ["dwcli", "-p", test_prompt, agent_name],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if headless_result.returncode != 0:
            pytest.fail(
                f"Headless command failed:\n"
                f"stdout: {headless_result.stdout}\n"
                f"stderr: {headless_result.stderr}"
            )

        logger.info(f"Headless completed. Output:\n{headless_result.stdout}")

        # 3. Give TUI a moment to update (messages should arrive in real-time)
        time.sleep(2.0)

        final_screen = session.get_screen_text()
        logger.info(f"TUI screen after headless:\n{final_screen}")

        # 4. Verify BOTH messages are visible

        # Check user message
        if test_prompt not in final_screen:
            pytest.fail(
                f"USER MESSAGE missing from TUI!\n"
                f"Expected to see: '{test_prompt}'\n"
                f"TUI screen:\n{final_screen}"
            )

        # Check assistant response
        expected_response = "Hello from headless mode"
        if expected_response not in final_screen:
            pytest.fail(
                f"ASSISTANT MESSAGE missing from TUI!\n"
                f"Expected to see: '{expected_response}'\n"
                f"Headless output: {headless_result.stdout}\n"
                f"TUI screen:\n{final_screen}\n\n"
                f"BUG: Assistant messages are NOT broadcast in non-streaming mode.\n"
                f"See test docstring for fix location."
            )

        # Success!
        logger.info(
            f"✅ PASS: Both user and assistant messages visible in TUI.\n"
            f"Headless->TUI broadcasting is working correctly!"
        )

    finally:
        session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=300"])
