"""End-to-end test: restart_self is SYNCHRONOUS, agent picks up notice
and continues without going to idle.

Requirements (set by project owner 2026-06-12):

  1. `restart_self` MUST be a SYNCHRONOUS tool that restarts the agent
     AT CALL TIME (not "after the current turn completes", not deferred).
     Calling it must NOT leave the agent idle.

  2. After restart, the agent receives a SYSTEM message via the next
     system prompt informing it of WHEN it was restarted and by WHAT
     cause ("self" for a self-initiated restart). The notice is purely
     informational — the agent is NOT required to acknowledge it.

  3. The agent does NOT enter idle automatically after restart — it
     CONTINUES working on the pending task on the freshly-loaded code.

  4. Integration test: spin up a real test agent, give it a test prompt
     ("when you receive a system notice about a restart, reply RECEIVED"),
     trigger `restart_self`, verify the agent received the system notice
     AND replied "RECEIVED" after restart. If not — test fails.

  5. The test prompt MUST NOT be embedded in the agent's system prompt;
     it is a separate user message.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.storage_paths import deep_worker_home


pytestmark = pytest.mark.integration


_AGENT_NAME = "test_restart_self_e2e"
_MARKER = "RECEIVED"

# Test prompt is SEPARATE from the system prompt. It is a regular user
# message. It tells the agent what to do when it receives a restart
# notice — but the notice itself is injected by AgentProcess, not by
# this prompt.
_TEST_PROMPT = (
    "Right now and only now, immediately call the `restart_self` tool "
    "exactly once (no arguments). When, after the restart, you receive "
    "a SYSTEM notice that you were restarted — reply with a single "
    f"word: {_MARKER}. No other words."
)


def _claude_code_available() -> bool:
    return shutil.which("claude") is not None


def _wait_for_pid_change(
    agent_name: str,
    old_pid: int,
    timeout: float = 30.0,
) -> int:
    """Wait until registry reports a different PID for the agent.

    Returns the new PID. Raises TimeoutError if no change within timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        registry = ProcessRegistry()
        info = registry.lookup(agent_name)
        if info and info.pid != old_pid:
            return info.pid
        time.sleep(0.2)
    raise TimeoutError(
        f"Agent {agent_name} PID did not change from {old_pid} "
        f"within {timeout}s"
    )


def _wait_for_registry(agent_name: str, timeout: float = 30.0) -> int:
    """Wait for agent to register and return its PID."""
    start = time.time()
    while time.time() - start < timeout:
        registry = ProcessRegistry()
        info = registry.lookup(agent_name)
        if info:
            return info.pid
        time.sleep(0.2)
    raise TimeoutError(
        f"Agent {agent_name} did not appear in registry within {timeout}s"
    )


def _stop_via_cli(agent_name: str) -> None:
    """Best-effort stop using deep-worker CLI."""
    cmd = shutil.which("deep-worker")
    if cmd is None:
        return
    try:
        subprocess.run(
            [cmd, agent_name, "--stop"],
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass


@pytest.fixture
def cli_started_agent():
    """Start an agent via deep-worker CLI (production-style launch).

    Yields the agent_name. Cleans up by sending --stop afterwards.
    """
    if not _claude_code_available():
        pytest.skip("claude CLI not installed; skipping restart_self e2e")

    cmd = shutil.which("deep-worker")
    if cmd is None:
        pytest.skip("deep-worker not on PATH")

    # Ensure clean state — kill any lingering instance.
    _stop_via_cli(_AGENT_NAME)
    # Wipe data dir so each run is fresh.
    data_dir = deep_worker_home() / _AGENT_NAME
    if data_dir.exists():
        import shutil as _sh
        _sh.rmtree(data_dir, ignore_errors=True)

    # Start agent in background (CLI daemonizes it).
    proc = subprocess.run(
        [
            cmd, _AGENT_NAME,
            "--provider", "claude-code",
            "--model", "opus",
            "--skip-permissions",
            "--dir", "/tmp",
        ],
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"deep-worker failed to start agent: "
            f"rc={proc.returncode} stderr={proc.stderr.decode()[:500]}"
        )

    pid = _wait_for_registry(_AGENT_NAME, timeout=30.0)

    try:
        yield (_AGENT_NAME, pid)
    finally:
        _stop_via_cli(_AGENT_NAME)


@pytest.mark.asyncio
async def test_restart_self_synchronous_with_notice_and_continuation(
    cli_started_agent,
):
    """Full e2e: invoke -> restart_self -> notice -> RECEIVED.

    Verifies all four requirements from the project owner.
    """
    agent_name, old_pid = cli_started_agent

    # 1. Send the test prompt — must NOT include the system notice
    #    text, only the user instruction.
    client = AgentClient(agent_name)
    await client.connect()

    # We don't await the full invoke because the agent will restart
    # mid-turn — the RPyC connection will drop. We fire-and-forget,
    # then immediately switch to polling for the PID change.
    invoke_task = asyncio.create_task(
        client.invoke_agent(_TEST_PROMPT, stream=False)
    )

    # 2. Wait for the agent's PID to change — proof of real restart.
    #    Use thread executor for blocking _wait_for_pid_change.
    loop = asyncio.get_running_loop()
    try:
        new_pid = await loop.run_in_executor(
            None,
            lambda: _wait_for_pid_change(agent_name, old_pid, timeout=60.0),
        )
    except TimeoutError as e:
        pytest.fail(f"Restart did not happen: {e}")
    finally:
        # Cancel the original invoke (it died with the old process).
        invoke_task.cancel()
        try:
            await invoke_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await client.disconnect()
        except Exception:
            pass

    assert new_pid != old_pid, (
        f"New PID {new_pid} same as old {old_pid} — restart did not "
        f"create a new process"
    )

    # 3. Connect to the NEW process. Give it a moment to come up fully.
    await asyncio.sleep(2.0)

    client2 = AgentClient(agent_name)
    await client2.connect()

    try:
        # 4. Inspect history — the agent should have processed the
        #    system notice automatically and produced an assistant
        #    message containing the marker.
        deadline = time.time() + 90.0
        found_notice = False
        found_marker = False
        history: list = []

        while time.time() < deadline:
            resp = await client2._send_command(
                {"command": "get_history", "limit": 200, "offset": 0}
            )
            history = resp.get("history", []) or []

            for msg in history:
                msg_type = msg.get("type", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(b.get("text", "")) if isinstance(b, dict) else str(b)
                        for b in content
                    )
                content_str = str(content)

                if msg_type == "system" and "restart" in content_str.lower():
                    found_notice = True
                if msg_type == "ai" and _MARKER in content_str:
                    found_marker = True

            if found_notice and found_marker:
                break
            await asyncio.sleep(2.0)

        assert found_notice, (
            f"System notice about restart not found in history. "
            f"Got {len(history)} messages: "
            f"{[(m.get('type'), str(m.get('content'))[:80]) for m in history]}"
        )
        assert found_marker, (
            f"Agent did not reply with marker '{_MARKER}' after restart. "
            f"Got {len(history)} messages: "
            f"{[(m.get('type'), str(m.get('content'))[:80]) for m in history]}"
        )

    finally:
        await client2.disconnect()
