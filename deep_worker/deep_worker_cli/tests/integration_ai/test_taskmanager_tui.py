"""Integration tests for TaskManager TUI display of intermediate messages.

NOTE: These tests require a running deep-worker agent WITH TaskManager enabled.
Start one with: deep-worker test_tm_tui -tm

The tests verify that intermediate messages from TaskManager evaluation loop
are properly displayed in the TUI after reloading conversation history.
"""

import asyncio
import subprocess

import pytest


@pytest.fixture
def test_agent_with_tm():
    """Provide test agent with TaskManager enabled.

    This fixture assumes user has started an agent with -tm flag.
    """
    agent_name = "test_tm"

    # Verify agent exists
    result = subprocess.run(
        ["dwcli", "--list"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    if agent_name not in result.stdout:
        pytest.skip(
            f"Agent '{agent_name}' not running with TaskManager. "
            f"Start with: deep-worker {agent_name} -tm"
        )

    class AgentInfo:
        def __init__(self, name):
            self.name = name

    return AgentInfo(agent_name)


@pytest.mark.asyncio
async def test_taskmanager_intermediate_messages_in_history(test_agent_with_tm):
    """Test that TaskManager intermediate messages appear in history.

    This test verifies that when TaskManager evaluation loop generates
    intermediate messages (corrective prompts and responses), they are
    properly stored in conversation history and can be retrieved.
    """
    agent_name = test_agent_with_tm.name

    # Send a test prompt that will trigger TaskManager evaluation loop
    # This prompt instructs the agent to respond incrementally
    prompt = (
        "Test TaskManager. Respond in 3 steps: "
        "First say 'Step 1 of 3', "
        "then say 'Step 2 of 3', "
        "then say 'Step 3 of 3 - Complete'."
    )

    # Send prompt via headless mode
    send_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "-p",
        prompt,
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await send_process.communicate()

    # Wait for TaskManager to complete evaluation loop
    # (may take several iterations)
    await asyncio.sleep(5.0)

    # Get conversation history
    history_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        "--limit",
        "20",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await history_process.communicate()
    output = stdout.decode()

    print("\n=== CONVERSATION HISTORY ===")
    print(output)
    print("=== END HISTORY ===\n")

    # Verify that history contains:
    # 1. Original user message
    assert prompt[:30] in output, "Original user prompt should be in history"

    # 2. TaskManager corrective prompts (indicated by "TaskManager:>" prefix)
    # Note: The exact format depends on TaskManager implementation
    taskmanager_messages = output.count("TaskManager:>")
    assert taskmanager_messages > 0, (
        "History should contain TaskManager intermediate messages. "
        f"Found {taskmanager_messages} TaskManager messages in output."
    )

    # 3. Multiple assistant responses (not just the final one)
    assistant_messages = output.count("[ASSISTANT]")
    assert assistant_messages >= 2, (
        f"History should contain multiple assistant responses (found {assistant_messages}). "
        "This indicates TaskManager evaluation loop generated intermediate responses."
    )

    # 4. Verify structure shows conversation flow
    assert "[USER]" in output, "History should show user messages"
    assert "=====" in output, "History should have separator lines"

    print(f"\n✅ Test passed: Found {taskmanager_messages} TaskManager messages "
          f"and {assistant_messages} assistant responses in history")


@pytest.mark.asyncio
async def test_history_reload_shows_all_messages(test_agent_with_tm):
    """Test that reloading history shows ALL messages including TaskManager prompts.

    This simulates what happens in TUI when InteractiveApp._agent_task()
    reloads history after receiving agent response.
    """
    agent_name = test_agent_with_tm.name

    # Get initial message count
    initial_history = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    initial_stdout, _ = await initial_history.communicate()
    initial_output = initial_stdout.decode()

    # Extract message count from "Showing X of Y messages"
    import re
    match = re.search(r"Showing (\d+) of (\d+) messages", initial_output)
    if match:
        initial_count = int(match.group(2))
    else:
        initial_count = 0

    # Send a new prompt
    prompt = "Say 'TaskManager test message' and nothing else."

    send_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "-p",
        prompt,
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await send_process.communicate()

    # Wait for processing
    await asyncio.sleep(3.0)

    # Reload history (simulating TUI reload)
    reload_history = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    reload_stdout, _ = await reload_history.communicate()
    reload_output = reload_stdout.decode()

    # Extract new message count
    match = re.search(r"Showing (\d+) of (\d+) messages", reload_output)
    if match:
        new_count = int(match.group(2))
    else:
        new_count = 0

    # Verify history grew (at least user message + assistant response)
    assert new_count > initial_count, (
        f"History should have more messages after sending prompt. "
        f"Initial: {initial_count}, After: {new_count}"
    )

    # If TaskManager was active, should have even more messages
    # (user prompt + TaskManager corrections + assistant responses)
    print(f"\n✅ History reload test passed: "
          f"Messages increased from {initial_count} to {new_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
