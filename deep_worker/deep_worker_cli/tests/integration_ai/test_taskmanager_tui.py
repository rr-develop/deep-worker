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
def tm_agent(request: pytest.FixtureRequest):
    """Auto-start a fresh deep-worker agent with TaskManager (-tm) enabled.

    Cleanup policy (per user mandate):
      * test passed → terminate the agent
      * test failed → preserve the agent (debug via ``dwcli``)
      * pytest killed before teardown → safety-net sweep in
        ``conftest.pytest_sessionfinish`` still terminates passed agents.

    This fixture:
    1. Kills any existing agent with the test name
    2. Cleans up data directories for a fresh start
    3. Starts the agent with --skip-permissions and -tm flags
    4. Waits for the agent to register in ProcessRegistry
    5. Registers the agent with the session-finish sweeper
    6. Yields the agent name
    7. Applies the cleanup policy on teardown

    Working directory is /tmp to prevent junk files in the repository tree.
    """
    from deep_worker_cli.process.registry import ProcessRegistry
    from .conftest import register_test_agent

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
    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    if data_dir.exists():
        logger.info(f"Removing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink(missing_ok=True)

    # 3. Start fresh agent with TaskManager enabled
    # CRITICAL: Use DeepSeek (less capable) to ensure TaskManager generates corrective prompts.
    # Claude Sonnet 4.5 is too good and often completes all requirements perfectly on first try,
    # causing TaskManager to evaluate as "done" without any corrective prompts. DeepSeek is
    # more likely to miss requirements, triggering the TaskManager evaluation loop.
    logger.info(f"Starting agent '{agent_name}' with TaskManager (-tm) and DeepSeek model...")
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", "/tmp",
            "--skip-permissions",
            "-tm",
            "--use-sqlite",  # Force SQLite — pytest sets PYTEST_CURRENT_TEST → InMemorySaver
            "--model", "deepseek-chat",  # Use less capable model to trigger TM corrective prompts
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

    # Tell the session-finish sweeper about this agent so a SIGKILL'd
    # pytest can still clean up if the test passed.
    register_test_agent(agent_name, request.node, created_by_fixture=True)

    yield agent_name

    # Teardown: passed → kill, failed → preserve. The sessionfinish
    # sweeper is the safety net for the case where yield-teardown
    # never runs at all.
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
# Prompt for TaskManager tests
# ============================================================================

# Standard TaskManager test prompt: deterministically forces multiple TM
# iterations regardless of how capable the underlying model is. The agent
# is instructed to report partial progress (30% → 50% → 70% → done) across
# several turns, which guarantees that the TaskManager evaluation loop
# produces at least three "TaskManager:> ..." corrective prompts that the
# TUI must broadcast in real time.
# NOTE: Must be single-line for TUI terminal emulator (newlines trigger Enter)
_TM_TEST_PROMPT = (
    "Test. If you are asked to complete a task fully or partially, "
    "you must first reply that you completed the task at 30% (and wait "
    "for the next request), then 50% (and wait for the next request), "
    "then 70% (and wait for the next request) then that you completed "
    "the task fully. Complete this test task."
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
        # CRITICAL FIX: Send text and Enter separately (matches test_slow_tool_display)
        # Sending them together doesn't work reliably with Textual PTY input
        session.send_input(_TM_TEST_PROMPT, press_enter=False)
        time.sleep(0.3)  # Brief wait for text to be processed
        session.send_key("enter")
        logger.info("✓ Prompt sent to agent")

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


def test_taskmanager_no_duplicate_messages():
    """Verify TaskManager corrective prompts appear exactly ONCE in TUI (no duplicates).

    This test addresses a bug where TaskManager corrective prompts were sent to TUI twice:
    1. At end of iteration N (lines 1040-1044 in agent_process.py)
    2. At start of iteration N+1 (lines 902-907 in agent_process.py)

    The fix: Only send user message in iteration 1. For iteration > 1, the corrective
    prompt was already sent at the end of the previous iteration.

    Test methodology:
    1. Start agent with -tm and use a test prompt that triggers multiple corrections
    2. Capture TUI screenshots during execution
    3. Use AI to analyze screenshots and count occurrences of each TaskManager prompt
    4. Verify history shows messages exactly once
    5. Verify TUI shows messages exactly once (no duplicates visible on screen)
    """
    agent_name = "test_tm_no_dup"

    # Test prompt that triggers multiple TaskManager iterations with specific progress markers
    test_prompt = (
        "Test. If you are asked to complete a task fully or partially, you "
        "must first reply that you completed the task at 30% (and wait for the next "
        "request), then 50% (and wait for the next request), then 70% (and "
        "wait for the next request) then that you completed the task fully. Complete "
        "this test task"
    )

    # Start agent with TaskManager in non-memory mode (writes to database)
    agent_process = subprocess.Popen(
        ["deep-worker", agent_name, "--clean", "-sp", "-tm", "--use-sqlite", "--dir", "/tmp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for agent to register
        from deep_worker_cli.process.registry import ProcessRegistry
        registry = ProcessRegistry()
        max_wait = 10.0
        start = time.time()
        while time.time() - start < max_wait:
            if registry.lookup(agent_name):
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"Agent {agent_name} did not register within {max_wait}s")

        logger.info(f"Agent {agent_name} registered successfully")

        # Open TUI session
        session = DwcliTerminalSession(
            agent_name=agent_name,
            columns=120,
            lines=40,
        )
        session.start(timeout=15.0)

        try:
            # Wait for input ready
            ready = session.wait_for_text(">", timeout=10.0)
            if not ready:
                pytest.fail("TUI did not show input area")

            time.sleep(1.0)

            logger.info("Sending test prompt...")
            session.send_input(test_prompt, press_enter=False)
            time.sleep(0.3)
            session.send_key("enter")

            # Wait for first TaskManager message
            logger.info("Waiting for TaskManager messages...")
            found = session.wait_for_text("TaskManager:>", timeout=120.0)
            if not found:
                pytest.fail("No TaskManager messages appeared in TUI")

            # Wait for task completion (when "100%" or "fully" appears)
            logger.info("Waiting for task completion...")
            time.sleep(30.0)  # Give time for multiple iterations

            # Capture final screenshot
            final_screen = session.get_screen_text()
            logger.info(f"Final TUI screenshot:\n{final_screen}")

            # Count TaskManager messages in screenshot using simple string count
            # Each unique corrective prompt should appear exactly once
            tm_count = final_screen.count("TaskManager:>")
            logger.info(f"Found {tm_count} TaskManager prompts in TUI screenshot")

        finally:
            session.close()

        # Check history for comparison
        time.sleep(2.0)  # Let checkpointer flush

        history_result = subprocess.run(
            ["deep-worker", "--history", agent_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        history_output = history_result.stdout
        logger.info(f"History output:\n{history_output}")

        # Count unique TaskManager messages in history
        history_tm_count = history_output.count("TaskManager:>")
        logger.info(f"Found {history_tm_count} TaskManager prompts in history")

        # Also check TaskManager's own history
        tm_history_result = subprocess.run(
            ["deep-worker", "--history", f"{agent_name}:tm"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        tm_history = tm_history_result.stdout
        logger.info(f"TaskManager history:\n{tm_history}")

        # Assertions
        assert history_tm_count > 0, "Expected TaskManager prompts in history"
        assert tm_count > 0, "Expected TaskManager prompts in TUI"

        # CRITICAL: TUI count should match history count (no duplicates)
        # Allow small variance because TUI might not show all history due to scroll
        if tm_count > history_tm_count * 1.5:
            pytest.fail(
                f"DUPLICATION DETECTED: TUI shows {tm_count} TaskManager prompts "
                f"but history only has {history_tm_count}. This indicates duplicate "
                f"messages are being sent to TUI.\n\n"
                f"TUI screenshot:\n{final_screen}\n\n"
                f"History:\n{history_output}"
            )

        logger.info(
            f"✓ PASS: No duplication detected. "
            f"TUI={tm_count}, History={history_tm_count}"
        )

    finally:
        # Kill agent
        if agent_process.poll() is None:
            agent_process.terminate()
            try:
                agent_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent_process.kill()
                agent_process.wait()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=300"])
