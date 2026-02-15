"""Integration test for dwcli headless mode with status monitoring.

This test verifies that:
1. dwcli -p sends commands to agent in headless mode
2. Agent status changes from 'ready' to 'working'/'waiting_input' during execution
3. Agent status returns to 'ready' after completion

This mirrors the behavior of CcoSession status tracking.
"""

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient

from . import agent_fixture


@pytest.fixture
def test_agent(request):
    """Provide test agent - uses existing agent instead of creating new one.

    This avoids issues with agent startup timeouts in integration tests.
    """
    # Simple fixture that just returns agent info for existing agent
    class AgentInfo:
        def __init__(self, name):
            self.agent_name = name

    # Use existing agent (user should start one before running test)
    agent_name = "example_status_test_agent"

    # Verify agent exists
    result = subprocess.run(
        ["dwcli", "--list"],
        capture_output=True,
        text=True,
        timeout=5
    )

    if agent_name not in result.stdout:
        pytest.skip(f"Agent '{agent_name}' not running. Start with: deep-worker {agent_name}")

    yield AgentInfo(agent_name)


async def get_agent_status_via_cco(agent_name: str) -> dict:
    """Get agent status via cco --status command.

    Args:
        agent_name: Name of the agent

    Returns:
        Status dict with 'status' field
    """
    try:
        result = subprocess.run(
            ["cco", "--status", agent_name],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Parse output like:
        # Session 'test_agent' status: ✅ ready
        # Message: Agent status: ready

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'status:' in line.lower():
                    # Extract status from "status: ✅ ready"
                    parts = line.split('status:')
                    if len(parts) > 1:
                        status_part = parts[1].strip()
                        # Remove emoji
                        for emoji in ['✅', '🔄', '⏳', '❓']:
                            status_part = status_part.replace(emoji, '').strip()
                        return {'status': status_part, 'success': True}

        return {'status': 'unknown', 'success': False, 'error': result.stderr}
    except Exception as e:
        return {'status': 'unknown', 'success': False, 'error': str(e)}


async def monitor_status_changes(agent_name: str, duration: float = 15.0) -> list[tuple[float, str]]:
    """Monitor agent status changes over time.

    Args:
        agent_name: Name of agent to monitor
        duration: How long to monitor (seconds)

    Returns:
        List of (timestamp, status) tuples
    """
    status_history = []
    start_time = time.time()
    last_status = None

    while time.time() - start_time < duration:
        result = await get_agent_status_via_cco(agent_name)
        current_status = result.get('status', 'unknown')

        # Record status changes
        if current_status != last_status:
            timestamp = time.time() - start_time
            status_history.append((timestamp, current_status))
            last_status = current_status

        await asyncio.sleep(0.2)  # Check every 200ms

    return status_history


@pytest.mark.asyncio
async def test_status_changes_during_headless_execution(test_agent):
    """Test that status changes from ready -> working/waiting_input -> ready.

    This test:
    1. Verifies agent is ready initially
    2. Sends headless command that takes time (sleep 10 seconds)
    3. Monitors status changes in parallel
    4. Verifies status changed to working or waiting_input during execution
    5. Verifies status returned to ready after completion
    """
    agent_name = test_agent.agent_name

    # Step 1: Verify agent is ready initially
    initial_status = await get_agent_status_via_cco(agent_name)
    assert initial_status['success'], f"Failed to get initial status: {initial_status.get('error')}"
    assert initial_status['status'] == 'ready', \
        f"Agent should be ready initially, got: {initial_status['status']}"

    # Step 2: Start status monitoring in background
    monitor_task = asyncio.create_task(
        monitor_status_changes(agent_name, duration=15.0)
    )

    # Give monitor time to start
    await asyncio.sleep(0.5)

    # Step 3: Send headless command via subprocess (non-blocking)
    # Command: "Wait 10 seconds and say 'hello'"
    prompt = "Execute a sleep command for 10 seconds, then say 'hello'. Use bash sleep command."

    cmd_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "-p", prompt,
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Step 4: Wait for command to complete
    stdout, stderr = await cmd_process.communicate()

    # Step 5: Wait for monitor to finish
    status_history = await monitor_task

    # Step 6: Verify command succeeded
    assert cmd_process.returncode == 0, \
        f"dwcli command failed: {stderr.decode()}"

    response = stdout.decode().strip()
    assert len(response) > 0, "Should get non-empty response"
    assert "hello" in response.lower(), \
        f"Response should contain 'hello', got: {response}"

    # Step 7: Verify status changes occurred
    print(f"\n🔍 Status history ({len(status_history)} changes):")
    for timestamp, status in status_history:
        print(f"  {timestamp:6.2f}s: {status}")

    # Extract all statuses seen
    statuses_seen = {status for _, status in status_history}

    # CRITICAL: Status must have changed from ready during execution
    assert len(status_history) >= 2, \
        f"Expected at least 2 status changes (ready -> busy -> ready), got: {status_history}"

    # Verify we saw non-ready status during execution
    busy_statuses = {'working', 'waiting_input'}
    assert any(status in busy_statuses for status in statuses_seen), \
        f"Expected to see 'working' or 'waiting_input' status during execution. " \
        f"Statuses seen: {statuses_seen}"

    # Step 8: Verify agent returned to ready
    final_status = await get_agent_status_via_cco(agent_name)
    assert final_status['success'], f"Failed to get final status: {final_status.get('error')}"
    assert final_status['status'] == 'ready', \
        f"Agent should return to 'ready' after completion, got: {final_status['status']}"

    print(f"✅ Test passed! Agent status changed appropriately during execution")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Race condition: cannot read from same stream reader in parallel")
async def test_status_via_client_sdk(test_agent):
    """Test status monitoring via AgentClient SDK (not cco).

    This verifies that DwSession.get_status() properly reflects status changes.

    NOTE: Currently skipped due to race condition when trying to monitor status
    while sending commands on the same client connection.
    """
    agent_name = test_agent.agent_name

    # Connect via SDK
    client = AgentClient(agent_name)
    await client.connect()

    try:
        # Get initial status
        initial_status = await client.get_status()
        assert initial_status.status.value == 'idle' or initial_status.status.value == 'ready', \
            f"Expected initial status 'idle' or 'ready', got: {initial_status.status.value}"

        # Send command and monitor status
        prompt = "Sleep for 5 seconds using bash sleep, then say 'done'"

        # Start monitoring in background
        async def monitor():
            statuses = []
            for _ in range(30):  # Monitor for up to 6 seconds
                status = await client.get_status()
                statuses.append((time.time(), status.status.value))
                await asyncio.sleep(0.2)
            return statuses

        monitor_task = asyncio.create_task(monitor())

        # Give monitor time to start
        await asyncio.sleep(0.3)

        # Send command
        response = await client.invoke_agent(prompt, stream=False)

        # Wait for monitoring to finish
        status_history = await monitor_task

        # Verify response
        assert response['success'], f"Command failed: {response.get('error')}"

        # Verify status changes
        statuses_seen = {status for _, status in status_history}
        print(f"\n🔍 SDK Status history: {statuses_seen}")

        # Should see at least some status change
        assert len(statuses_seen) > 1, \
            f"Expected status changes, but only saw: {statuses_seen}"

    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Allow running directly with: python test_dwcli_headless_status.py
    pytest.main([__file__, "-v", "-s"])
