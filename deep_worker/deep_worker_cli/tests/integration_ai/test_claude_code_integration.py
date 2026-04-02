"""Integration test for claude-code provider.

This test verifies that deep-worker can successfully use the claude-code provider
for real agent interactions (no mocks).

⚠️ CRITICAL: PORT USAGE IN INTEGRATION TESTS ⚠️
================================================
These tests use agent_fixture which creates REAL agents with RPyC servers.

NEVER pass fixed rpyc_port values to agent_fixture or AgentProcess!
ALWAYS rely on automatic port assignment (rpyc_port=True).

Fixed ports cause flaky tests when:
- Multiple tests run in parallel
- Previous test didn't clean up
- Port already in use

The agent_fixture automatically uses dynamic port assignment.
DO NOT modify this behavior!
================================================

Run with:
    pytest tests/integration_ai/test_claude_code_integration.py -v -s
"""

from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient

from . import agent_fixture


@pytest.fixture
def claude_code_agent(request):
    """Provide test agent using claude-code provider.

    This fixture:
    - Creates a test agent with claude-code provider
    - No API key required (uses local Claude Code installation)
    - Reuses existing agent if found
    - Cleans up on test success
    - Preserves on test failure for debugging
    """
    with agent_fixture(
        agent_name="test_claude_code_integration",
        working_dir=Path("/tmp"),
        model="sonnet",  # Claude Code simple model name
        provider="claude-code",
        skip_permissions=True,
        api_key=None,  # No API key needed for claude-code
    ) as agent:
        yield agent


@pytest.mark.asyncio
async def test_claude_code_simple_arithmetic(claude_code_agent):
    """Test claude-code agent can perform simple arithmetic.

    This test demonstrates:
    - Basic agent interaction via IPC with claude-code provider
    - Simple content validation (no AI meta-query for speed)
    """
    # Connect to agent via IPC
    client = AgentClient(claude_code_agent.agent_name)
    await client.connect()

    try:
        # Send query to agent
        query = "What is 2 + 2? Just give me the number."
        response = await client.invoke_agent(query, stream=False)

        # Extract response content
        content = response.get("response", {}).get("content", "")

        # Simple validation: check that response mentions the answer
        assert content, "Response should not be empty"
        assert "4" in content, f"Response should contain '4'. Got: {content[:200]}"

        print(f"\n✅ Simple arithmetic test passed!")
        print(f"Query: {query}")
        print(f"Response: {content[:200]}")

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_claude_code_factual_question(claude_code_agent):
    """Test claude-code agent can answer factual questions.

    This test demonstrates:
    - Factual knowledge retrieval with claude-code
    - Validation of correct answers
    """
    client = AgentClient(claude_code_agent.agent_name)
    await client.connect()

    try:
        # Ask factual question
        query = "What is the capital of France? Just give me the city name."
        response = await client.invoke_agent(query)
        content = response["response"]["content"]

        # Simple validation: check that response mentions Paris
        assert content, "Response should not be empty"
        assert "Paris" in content or "paris" in content.lower(), \
            f"Response should mention Paris. Got: {content[:200]}"

        print(f"\n✅ Factual question test passed!")
        print(f"Query: {query}")
        print(f"Response: {content[:200]}")

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_claude_code_calculation(claude_code_agent):
    """Test claude-code agent can perform calculations.

    This test demonstrates:
    - Mathematical reasoning with claude-code
    - Validation of correct calculations
    """
    client = AgentClient(claude_code_agent.agent_name)
    await client.connect()

    try:
        # Query that requires calculation
        query = "Calculate the area of a 5x5 square. Just give me the number."
        response = await client.invoke_agent(query)
        content = response["response"]["content"]

        # Simple validation: check that response mentions 25
        assert content, "Response should not be empty"
        assert "25" in content, f"Response should contain '25'. Got: {content[:200]}"

        print(f"\n✅ Calculation test passed!")
        print(f"Query: {query}")
        print(f"Response: {content[:200]}")

    finally:
        await client.disconnect()


@pytest.fixture
def claude_code_haiku_agent(request):
    """Provide test agent using claude-code provider with haiku model.

    ⚠️ IMPORTANT: This creates a separate agent from claude_code_agent
    to test model switching capabilities.
    """
    with agent_fixture(
        agent_name="test_claude_code_haiku",
        working_dir=Path("/tmp"),
        model="haiku",  # Fast model (instead of sonnet)
        provider="claude-code",
        skip_permissions=True,
        api_key=None,
    ) as agent:
        yield agent


@pytest.mark.asyncio
async def test_claude_code_model_haiku(claude_code_haiku_agent):
    """Test claude-code with haiku model specifically.

    This test demonstrates:
    - Using different Claude Code models (haiku instead of sonnet)
    - Model switching capabilities
    """
    client = AgentClient(claude_code_haiku_agent.agent_name)
    await client.connect()

    try:
        # Simple query suitable for haiku
        query = "What is 10 + 5? Just the number."
        response = await client.invoke_agent(query, stream=False)
        content = response["response"]["content"]

        assert content, "Response should not be empty"
        assert "15" in content, f"Response should contain '15'. Got: {content[:200]}"

        print(f"\n✅ Haiku model test passed!")
        print(f"Query: {query}")
        print(f"Response: {content[:200]}")

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_claude_code_streaming(claude_code_agent):
    """Test claude-code agent with streaming enabled.

    This test demonstrates:
    - Streaming response handling with claude-code
    - Incremental content delivery
    """
    client = AgentClient(claude_code_agent.agent_name)
    await client.connect()

    try:
        # Send query with streaming
        query = "Count from 1 to 5."
        response = await client.invoke_agent(query, stream=True)

        # For streaming, response structure may differ
        # Just verify we got something back
        assert response, "Response should not be empty"

        print(f"\n✅ Streaming test passed!")
        print(f"Query: {query}")
        print(f"Response structure: {type(response)}")

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_claude_code_agent_status(claude_code_agent):
    """Test getting agent status with claude-code.

    This test demonstrates:
    - Agent status API works with claude-code provider
    - Session information retrieval
    """
    client = AgentClient(claude_code_agent.agent_name)
    await client.connect()

    try:
        # Get agent status
        status = await client.get_status()

        assert status, "Status should not be None"
        assert hasattr(status, 'status'), "Status should have 'status' attribute"

        print(f"\n✅ Agent status test passed!")
        print(f"Status: {status.status}")
        print(f"Agent name: {claude_code_agent.agent_name}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Allow running directly with: python test_claude_code_integration.py
    pytest.main([__file__, "-v", "-s"])
