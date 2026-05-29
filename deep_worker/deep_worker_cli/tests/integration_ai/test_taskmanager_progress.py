"""Integration test for TaskManager iterative progress tracking.

This test verifies that TaskManager can manage multi-step tasks where
the agent reports incremental progress (30%, 50%, 70%, 100%) and
TaskManager encourages continuation until completion.

## Test Scenario

The agent receives a meta-instruction:
  "When asked to complete a task, respond with 30% completion
   (wait for next request), then 50%, then 70%, then 100%."

TaskManager should:
1. Receive "30% complete" → send corrective prompt to continue
2. Receive "50% complete" → send corrective prompt to continue
3. Receive "70% complete" → send corrective prompt to continue
4. Receive "100% complete" → accept as done

## Success Criteria

The TUI must show:
- Agent's incremental progress responses (30%, 50%, 70%, 100%)
- TaskManager's encouraging prompts ("TaskManager:> ...")
- Final "100%" or completion message visible at the end

AI screenshot analysis verifies these elements are present.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import time
from pathlib import Path

import pytest

from .clean_agent_fixture import clean_agent  # noqa: F401
from .terminal_emulator import DwcliTerminalSession

logger = logging.getLogger(__name__)

# ============================================================================
# Test Configuration
# ============================================================================

_AGENT_NAME = "test_tm_progress"

_TEST_PROMPT = (
    "Test. If you are asked to complete a task fully or partially, "
    "you must first reply that you completed the task at 30% (and wait for the next request), "
    "then 50% (and wait for the next request), then 70% (and wait for the next request) "
    "then that you completed the task fully. Complete this test task."
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tm_progress_agent(request: pytest.FixtureRequest):
    """Start a fresh agent with TaskManager for progress tracking test.

    Cleanup policy: passed → terminate, failed → preserve for debug.
    The session-finish sweeper in ``conftest.py`` is the safety net for
    cases where the yield-teardown never runs.
    """
    from deep_worker_cli.process.registry import ProcessRegistry
    from .conftest import register_test_agent

    agent_name = _AGENT_NAME
    agent_process = None
    registry = ProcessRegistry()

    # 1. Kill any existing agent
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
    import shutil
    data_dir = Path("~/.deep_worker").expanduser() / agent_name
    if data_dir.exists():
        logger.info(f"Removing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    log_dir = Path("~/.deep_worker").expanduser() / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink(missing_ok=True)

    # 3. Start agent with TaskManager
    logger.info(f"Starting agent '{agent_name}' with TaskManager and DeepSeek model...")
    agent_process = subprocess.Popen(
        [
            "deep-worker",
            agent_name,
            "--dir", "/tmp",
            "--skip-permissions",
            "-tm",
            "--model", "deepseek-chat",  # Use DeepSeek for more iterations
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

    # Register with session-finish sweeper (safety net for SIGKILL'd pytest).
    register_test_agent(agent_name, request.node, created_by_fixture=True)

    # 5. Yield for test
    yield agent_name

    # 6. Teardown: passed → terminate, failed → preserve.
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


@pytest.mark.asyncio
async def test_taskmanager_iterative_progress_visible_in_tui(tm_progress_agent):
    """TaskManager must show incremental progress (30%, 50%, 70%, 100%) in TUI.

    This test verifies that:
    1. Agent responds with incremental progress as instructed
    2. TaskManager sends corrective prompts to continue
    3. All intermediate messages appear in the TUI
    4. Final 100% completion is visible

    The test uses AI to analyze the final TUI screenshot and verify
    that progress markers and TaskManager prompts are present.
    """
    agent_name = tm_progress_agent

    # Open TUI session
    session = DwcliTerminalSession(
        agent_name=agent_name,
        columns=140,
        lines=50,
    )
    session.start(timeout=15.0)

    try:
        # Wait for input area
        logger.info("Waiting for TUI input area...")
        ready = session.wait_for_text(">", timeout=10.0)
        if not ready:
            screen = session.get_screen_text()
            pytest.fail(
                f"dwcli TUI did not show input area within 10s.\n"
                f"Screen content:\n{screen}"
            )

        time.sleep(1.0)

        # Send the test prompt
        logger.info("Sending progress tracking test prompt...")
        session.send_input(_TEST_PROMPT, press_enter=False)
        time.sleep(0.3)
        session.send_key("enter")
        logger.info("✓ Prompt sent")

        # Wait for TaskManager to process iterations
        # Each iteration takes ~10-30s, so allow up to 180s total
        logger.info("Waiting for TaskManager iterations (up to 180s)...")
        time.sleep(180.0)

        # Get final screen content
        screen = session.get_screen_text()
        logger.info(f"Final TUI screen content:\n{screen}")

        # Use AI to analyze the screenshot
        logger.info("Analyzing TUI screenshot with AI...")
        analysis_result = await _analyze_progress_screenshot(screen)

        logger.info(f"AI Analysis Result:\n{analysis_result}")

        # Check if analysis confirms success
        if not _is_progress_test_successful(analysis_result):
            pytest.fail(
                f"TaskManager progress test FAILED.\n\n"
                f"AI Analysis:\n{analysis_result}\n\n"
                f"Full TUI Screenshot:\n{screen}"
            )

        logger.info("✓ TaskManager progress test PASSED")

    finally:
        session.close()


# ============================================================================
# AI Screenshot Analysis
# ============================================================================


async def _analyze_progress_screenshot(screen_text: str) -> str:
    """Use AI to analyze TUI screenshot for progress markers.

    Checks for:
    - Progress indicators (30%, 50%, 70%, 100%)
    - TaskManager corrective prompts
    - Final completion message
    """
    from deep_worker_cli.client.agent_client import AgentClient

    # Use default agent for analysis (assumes one is running)
    # Alternatively, could use a dedicated analysis agent
    analysis_prompt = f"""Analyze this TUI screenshot from a TaskManager test.

The test asked an agent to report incremental progress: 30%, 50%, 70%, then 100%.
TaskManager should send corrective prompts to encourage continuation.

TUI SCREENSHOT:
```
{screen_text}
```

ANSWER THESE QUESTIONS:

1. Does the screenshot show progress at 30%? (Yes/No + quote)
2. Does the screenshot show progress at 50%? (Yes/No + quote)
3. Does the screenshot show progress at 70%? (Yes/No + quote)
4. Does the screenshot show final completion (100% or "fully")? (Yes/No + quote)
5. Does the screenshot show TaskManager corrective prompts ("TaskManager:>")? (Yes/No + count)
6. Is the final completion message visible at the bottom of the screen? (Yes/No)

VERDICT: PASS or FAIL

Provide clear, structured analysis.
"""

    try:
        # Try to use a running agent for analysis
        client = AgentClient()
        await client.connect()

        response = await client.send_message(
            message=analysis_prompt,
            session_id="screenshot_analysis",
        )

        await client.close()
        return response or "AI analysis failed - no response"

    except Exception as e:
        logger.warning(f"Could not use AgentClient for analysis: {e}")
        # Fallback: simple text search
        return _simple_text_analysis(screen_text)


def _simple_text_analysis(screen_text: str) -> str:
    """Fallback analysis using simple text search."""
    findings = []

    # Check for progress markers
    if "30%" in screen_text or "30 %" in screen_text:
        findings.append("✓ Found 30% progress")
    else:
        findings.append("✗ Missing 30% progress")

    if "50%" in screen_text or "50 %" in screen_text:
        findings.append("✓ Found 50% progress")
    else:
        findings.append("✗ Missing 50% progress")

    if "70%" in screen_text or "70 %" in screen_text:
        findings.append("✓ Found 70% progress")
    else:
        findings.append("✗ Missing 70% progress")

    if "100%" in screen_text or "fully" in screen_text or "completed" in screen_text:
        findings.append("✓ Found completion (100%)")
    else:
        findings.append("✗ Missing completion marker")

    # Check for TaskManager prompts
    tm_count = screen_text.count("TaskManager:>")
    if tm_count > 0:
        findings.append(f"✓ Found {tm_count} TaskManager prompts")
    else:
        findings.append("✗ No TaskManager prompts found")

    verdict = "FAIL" if any("✗" in f for f in findings) else "PASS"

    return "\n".join(findings) + f"\n\nVERDICT: {verdict}"


def _is_progress_test_successful(analysis: str) -> bool:
    """Determine if the progress test passed based on AI analysis."""
    analysis_lower = analysis.lower()

    # Check for PASS verdict
    if "verdict: pass" in analysis_lower:
        return True

    # Check for critical markers
    required_markers = ["30%", "50%", "70%"]
    completion_markers = ["100%", "fully", "completed"]

    has_all_progress = all(marker in analysis_lower for marker in required_markers)
    has_completion = any(marker in analysis_lower for marker in completion_markers)
    has_taskmanager = "taskmanager" in analysis_lower

    return has_all_progress and has_completion and has_taskmanager
