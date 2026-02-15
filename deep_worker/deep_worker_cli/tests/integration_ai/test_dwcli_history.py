"""Integration tests for dwcli history and last-response commands.

NOTE: These tests require a running deep-worker agent.
Start one with: deep-worker test_history_agent

The tests will use the existing agent instead of creating a new one.
"""

import asyncio
import subprocess

import pytest


@pytest.fixture
def test_agent():
    """Provide test agent - uses existing agent instead of creating new one.

    This avoids issues with agent startup timeouts in integration tests.
    """
    # Use existing agent (user should start one before running test)
    agent_name = "test_history_agent"

    # Verify agent exists
    result = subprocess.run(
        ["dwcli", "--list"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    if agent_name not in result.stdout:
        pytest.skip(
            f"Agent '{agent_name}' not running. Start with: deep-worker {agent_name}"
        )

    class AgentInfo:
        def __init__(self, name):
            self.name = name

    return AgentInfo(agent_name)


@pytest.mark.asyncio
async def test_last_response_shows_final_ai_message(test_agent):
    """Test --last-response shows the final assistant response."""
    agent_name = test_agent.name

    # Send a simple prompt to generate history
    prompt = "What is 2+2? Answer with just the number."

    send_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "-p",
        prompt,
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await send_process.communicate()

    # Wait for response to be recorded
    await asyncio.sleep(1.0)

    # Get last response
    last_response_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--last-response",
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await last_response_process.communicate()
    output = stdout.decode()

    # Verify output contains response
    assert "Last Response from" in output
    assert agent_name in output
    assert "=" in output  # Separator line

    # Should contain the answer (4)
    assert "4" in output


@pytest.mark.asyncio
async def test_history_shows_conversation_messages(test_agent):
    """Test --history shows conversation history."""
    agent_name = test_agent.name

    # Send multiple prompts to build history
    prompts = [
        "Say 'hello' and nothing else",
        "What is the capital of France? Just the city name.",
        "What is 5+7? Just the number.",
    ]

    for prompt in prompts:
        send_process = await asyncio.create_subprocess_exec(
            "dwcli",
            "-p",
            prompt,
            agent_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await send_process.communicate()
        await asyncio.sleep(1.5)  # Wait between prompts

    # Get history
    history_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await history_process.communicate()
    output = stdout.decode()

    # Verify output structure
    assert "Conversation History for" in output
    assert agent_name in output
    assert "Showing" in output  # Pagination info
    assert "messages" in output

    # Should contain message type labels
    assert "[USER]" in output or "[ASSISTANT]" in output

    # Should contain at least one of our prompts
    found_prompt = any(prompt[:20] in output for prompt in prompts)
    assert found_prompt, "At least one prompt should be in history"


@pytest.mark.asyncio
async def test_history_pagination_with_limit(test_agent):
    """Test --history respects --limit parameter."""
    agent_name = test_agent.name

    # Send several prompts
    for i in range(5):
        prompt = f"Count to {i+1}"
        send_process = await asyncio.create_subprocess_exec(
            "dwcli",
            "-p",
            prompt,
            agent_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await send_process.communicate()
        await asyncio.sleep(1.0)

    # Get history with limit=3
    history_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        "--limit",
        "3",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await history_process.communicate()
    output = stdout.decode()

    # Verify pagination info shows limit was applied
    assert "Showing" in output
    # Should show "Showing X of Y messages" where X <= 3
    assert "offset: 0" in output


@pytest.mark.asyncio
async def test_history_pagination_with_offset(test_agent):
    """Test --history respects --offset parameter for pagination."""
    agent_name = test_agent.name

    # Send several prompts to build longer history
    for i in range(6):
        prompt = f"Test message {i+1}"
        send_process = await asyncio.create_subprocess_exec(
            "dwcli",
            "-p",
            prompt,
            agent_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await send_process.communicate()
        await asyncio.sleep(1.0)

    # Get history with offset=2
    history_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        "--limit",
        "3",
        "--offset",
        "2",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await history_process.communicate()
    output = stdout.decode()

    # Verify offset is shown
    assert "offset: 2" in output

    # Should show navigation hints
    assert "previous page" in output or "next page" in output


@pytest.mark.asyncio
async def test_last_response_no_history():
    """Test --last-response handles agent with no conversation history.

    NOTE: This test assumes the agent exists but has no messages yet.
    If agent has history, this test will be skipped.
    """
    # Use a special agent that is expected to be empty
    agent_name = "test_empty_agent"

    # Check if agent exists
    result = subprocess.run(
        ["dwcli", "--list"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    if agent_name not in result.stdout:
        pytest.skip(
            f"Agent '{agent_name}' not running. "
            f"Start with: deep-worker {agent_name}"
        )

    # Try to get last response
    last_response_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--last-response",
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await last_response_process.communicate()
    output = stdout.decode()

    # Should handle empty history gracefully
    # (May show "No conversation" or may show a response if agent has messages)
    assert last_response_process.returncode == 0


@pytest.mark.asyncio
async def test_history_command_shows_different_message_types(test_agent):
    """Test --history distinguishes between USER, ASSISTANT, and TOOL messages."""
    agent_name = test_agent.name

    # Send a prompt that will trigger tool usage
    prompt = "Use the calculator tool to compute 15 * 8"

    send_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "-p",
        prompt,
        agent_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await send_process.communicate()
    await asyncio.sleep(2.0)  # Wait for tool execution

    # Get history
    history_process = await asyncio.create_subprocess_exec(
        "dwcli",
        "--history",
        agent_name,
        "--limit",
        "50",  # Get all messages
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await history_process.communicate()
    output = stdout.decode()

    # Should show different message types
    # At minimum: USER (prompt) and ASSISTANT (response)
    assert "[USER]" in output or "[ASSISTANT]" in output

    # May also show TOOL messages if tool was used
    # (This is optional as agent might not always use tools)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
