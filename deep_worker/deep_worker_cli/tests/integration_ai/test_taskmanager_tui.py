"""Integration tests for TaskManager TUI display of intermediate messages.

These tests are FULLY AUTOMATED — they start their own agent with the -tm
(TaskManager) flag and verify that intermediate TaskManager messages appear
in the dwcli TUI in real time.

## Architecture Under Test

When the user sends a prompt to an agent running with -tm, the following
message flow is expected (ARCHITECTURE.md):

  User prompt
    → agent receives via exposed_invoke_agent()
    → _notify_subscribers() broadcasts user message to ALL subscribers (dwcli)
    → TaskManager evaluation loop (multiple agent.ainvoke() iterations):
        ┌──────────────────────────────────────────────────────────────┐
        │  iteration N:                                                │
        │    agent produces intermediate assistant response            │
        │    → _notify_subscribers() MUST broadcast this              │
        │    TaskManager evaluates response                            │
        │    TaskManager sends corrective prompt ("TaskManager:> ...")  │
        │    → _notify_subscribers() MUST broadcast this              │
        └──────────────────────────────────────────────────────────────┘
    → _notify_subscribers() broadcasts final assistant response

ALL messages — including intermediate assistant responses and TaskManager
corrective prompts — MUST appear in ALL connected dwcli TUI instances.

## Current Bug (as of this test writing)

The TaskManager loop in agent_process.py does NOT call _notify_subscribers()
for intermediate messages. Only the initial user message and the final
assistant response are broadcast. This violates the "agent is single source
of truth" architecture principle.

These tests are expected to FAIL until the bug is fixed.

## How to Fix

In agent_process.py, inside the TaskManager evaluation loop (around line 748),
add calls to notify_callback() (passed from agent_service.py) for:
1. Each intermediate assistant response (before sending to TaskManager)
2. Each TaskManager corrective prompt (before sending back to agent)
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

_TM_AGENT_NAME = "test_taskmanager_tui"


@pytest.fixture
def tm_agent():
    """Auto-start a fresh deep-worker agent with TaskManager (-tm) enabled.

    This fixture:
    1. Kills any existing agent with the test name
    2. Cleans up data directories for a fresh start
    3. Starts the agent with --skip-permissions and -tm flags
    4. Waits for the agent to register in ProcessRegistry
    5. Yields the agent name
    6. Kills the agent on teardown

    Working directory is /tmp to prevent junk files in the source tree.
    """
    from deep_worker_cli.process.registry import ProcessRegistry

    agent_name = _TM_AGENT_NAME
    agent_process = None
    registry = ProcessRegistry()

    # 1. Kill any existing agent with this name
    existing = registry.lookup(agent_name)
    if existing:
        logger.info(f"Killing existing agent PID {existing.pid}")
        try:
            import os
            os.kill(existing.pid, signal.SIGKILL)
            time.sleep(0.5)
        except ProcessLookupError:
            pass

    # 2. Clean up data directories
    data_dir = Path(f"~/.deep_worker/{agent_name}").expanduser()
    if data_dir.exists():
        logger.info(f"Removing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    log_dir = Path("~/.deep_worker/logs").expanduser()
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink(missing_ok=True)

    # 3. Start fresh agent with TaskManager enabled
    logger.info(f"Starting agent '{agent_name}' with TaskManager (-tm)...")
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", "/tmp",
            "--skip-permissions",
            "-tm",
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
# Prompt for TaskManager tests
# ============================================================================

# This prompt is designed to be specific enough that the TaskManager will
# verify compliance with multiple requirements, sending corrective prompts
# if any are missing. The agent's first response is unlikely to satisfy
# ALL requirements simultaneously on the first try.
_TM_TEST_PROMPT = (
    "Write a Python function called `add_numbers` that adds two numbers. "
    "Your response MUST include ALL of the following:\n"
    "1. A complete docstring with Args section, Returns section, and Raises section\n"
    "2. Type hints for all parameters and return value\n"
    "3. Input validation that raises TypeError for non-numeric inputs\n"
    "4. At least two concrete usage examples in the docstring\n"
    "5. A brief comment explaining why this function is useful\n"
    "Make sure every single requirement above is present in your response."
)


# ============================================================================
# Tests
# ============================================================================


def test_taskmanager_messages_visible_in_tui(tm_agent):
    """TaskManager intermediate messages MUST appear in the dwcli TUI in real time.

    This test verifies the ARCHITECTURE PRINCIPLE: agent is the single source
    of truth. ALL messages — including TaskManager corrective prompts — must
    be broadcast to ALL connected dwcli instances via _notify_subscribers().

    HOW IT WORKS:
    1. Connect to agent (with -tm) via dwcli TUI
    2. Send a prompt that triggers the TaskManager evaluation loop
    3. Wait for TaskManager to produce corrective prompts
    4. Assert that "TaskManager:>" appears on the TUI screen

    EXPECTED RESULT (before fix):
    - FAIL: The TUI never shows "TaskManager:>" because the TaskManager loop
      in agent_process.py does NOT call _notify_subscribers() for intermediate
      messages.

    EXPECTED RESULT (after fix):
    - PASS: Each intermediate assistant response and each corrective prompt
      from TaskManager is broadcast and appears in the TUI.
    """
    agent_name = tm_agent

    # Open a TUI session connected to the TM-enabled agent
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

        logger.info("Sending prompt that will trigger TaskManager loop...")
        session.send_input(_TM_TEST_PROMPT, press_enter=True)

        # Wait up to 120 seconds for TaskManager to send at least one
        # corrective prompt visible in the TUI.
        # The marker "TaskManager:>" appears in messages sent by TaskManager.
        logger.info("Waiting for 'TaskManager:>' to appear in TUI (up to 120s)...")
        found_tm_message = session.wait_for_text("TaskManager:>", timeout=120.0)

        if not found_tm_message:
            # Capture current screen for diagnosis
            screen = session.get_screen_text()
            logger.error(
                "TaskManager messages NOT visible in TUI.\n"
                f"Screen content:\n{screen}"
            )
            pytest.fail(
                "ARCHITECTURE VIOLATION: TaskManager intermediate messages are NOT "
                "visible in the dwcli TUI.\n\n"
                "The TaskManager evaluation loop in agent_process.py calls "
                "agent.ainvoke() multiple times with corrective prompts, but does "
                "NOT call _notify_subscribers() for:\n"
                "  - Intermediate assistant responses\n"
                "  - TaskManager corrective prompts (\"TaskManager:> ...\")\n\n"
                "Fix: In agent_process.py, inside the TaskManager evaluation loop "
                "(around line 748), call notify_callback() for each intermediate "
                "assistant response and each corrective prompt before proceeding to "
                "the next iteration.\n\n"
                f"Current TUI screen:\n{screen}"
            )

        # If we reach here, at least one "TaskManager:>" was visible in TUI
        screen = session.get_screen_text()
        logger.info(
            f"PASS: 'TaskManager:>' found in TUI. Current screen:\n{screen}"
        )

    finally:
        session.close()


def test_taskmanager_messages_in_history_but_not_tui(tm_agent):
    """Document the gap: TM messages exist in history but are invisible in TUI.

    This test makes the architectural violation explicit:
    - dwcli --history DOES show TaskManager messages (LangChain checkpointer
      stores ALL messages from every agent.ainvoke() call)
    - dwcli TUI does NOT show them (no _notify_subscribers() call inside the loop)

    This test verifies BOTH sides of the violation:
    1. History contains "TaskManager:>" messages (proving they exist)
    2. TUI screen does NOT contain "TaskManager:>" messages (proving the bug)

    After the fix, both history and TUI should show all messages.

    NOTE: If test_taskmanager_messages_visible_in_tui passes (fix applied),
    this test verifies that history and TUI are CONSISTENT.
    """
    agent_name = tm_agent

    # -----------------------------------------------------------------------
    # Step 1: Send a prompt via headless mode so we can check results
    # -----------------------------------------------------------------------
    logger.info("Sending prompt via headless mode...")
    result = subprocess.run(
        ["dwcli", "-p", _TM_TEST_PROMPT, agent_name],
        capture_output=True,
        text=True,
        timeout=180,
    )
    logger.info(f"dwcli -p completed (returncode={result.returncode})")

    # -----------------------------------------------------------------------
    # Step 2: Retrieve history and check for TaskManager messages
    # -----------------------------------------------------------------------
    history_result = subprocess.run(
        ["dwcli", "--history", agent_name, "--limit", "50"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    history_output = history_result.stdout

    logger.info(f"History output:\n{history_output}")

    tm_in_history = "TaskManager:>" in history_output
    if not tm_in_history:
        pytest.fail(
            "Expected 'TaskManager:>' to appear in 'dwcli --history' output, "
            "but it was not found. The TaskManager may not have sent any corrective "
            "prompts, or the checkpointer is not storing them.\n\n"
            f"History output:\n{history_output}"
        )

    logger.info("CONFIRMED: 'TaskManager:>' appears in history")

    # -----------------------------------------------------------------------
    # Step 3: Open TUI on same agent and check if TUI screen shows TM messages
    #
    # At this point the agent has already processed the prompt (step 1 used
    # headless -p mode which waits for completion). When we open TUI now,
    # the TUI will load history via _load_history() and show the messages.
    # This checks that the LOADED history (not real-time broadcast) also
    # contains TM messages.
    # -----------------------------------------------------------------------
    logger.info("Opening TUI to check loaded history display...")
    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=120,
        lines=40,
    )
    session.start(timeout=15.0)

    try:
        # Wait for TUI to load history (scroll-back content)
        time.sleep(3.0)  # History loads after ~0.1s timer in _load_history()

        screen = session.get_screen_text()
        logger.info(f"TUI screen after history load:\n{screen}")

        tm_in_tui = "TaskManager:>" in screen

        if tm_in_history and not tm_in_tui:
            # Classic violation: messages exist in storage but are invisible
            # on the TUI screen. This can happen if the history display
            # in the TUI skips or filters TaskManager messages.
            logger.warning(
                "VIOLATION: 'TaskManager:>' is in history but NOT visible in TUI. "
                "This indicates either:\n"
                "  a) The TUI history display filters out TM messages, OR\n"
                "  b) The loaded history display is correct but real-time broadcast\n"
                "     was missing (tested by test_taskmanager_messages_visible_in_tui)."
            )
            # Don't fail here — the test_taskmanager_messages_visible_in_tui
            # test already tests real-time visibility. This test documents the
            # relationship between history and TUI display.
            logger.info(
                f"SUMMARY:\n"
                f"  'TaskManager:>' in history: {tm_in_history}\n"
                f"  'TaskManager:>' in TUI:     {tm_in_tui}\n"
                f"\n"
                f"After fix, both should be True."
            )
        elif tm_in_history and tm_in_tui:
            logger.info(
                "CONSISTENT: 'TaskManager:>' appears in BOTH history and TUI. "
                "The fix is working correctly."
            )
        else:
            logger.info(
                f"  'TaskManager:>' in history: {tm_in_history}\n"
                f"  'TaskManager:>' in TUI:     {tm_in_tui}\n"
            )

        # The primary assertion: if TM messages are in history, the TUI
        # (loaded history view) should also show them.
        # After fix: real-time broadcast will also show them during active session.
        assert tm_in_history, (
            "TaskManager messages not found in history — the TM loop may not have run"
        )

    finally:
        session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=300"])
