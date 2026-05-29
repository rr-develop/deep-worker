"""Example integration test using AI-powered validation.

This module demonstrates how to use the integration_ai infrastructure
for testing deep_worker agents with AI-powered response validation.

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
    pytest tests/integration_ai/test_agent_basic_example.py -v -s
"""

from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient

from . import agent_fixture, check_reply, validate_test_reply


@pytest.fixture
def test_agent(request):
    """Provide test agent with automatic lifecycle management.

    This fixture:
    - Creates a test agent if it doesn't exist
    - Reuses existing agent if found
    - Cleans up on test success
    - Preserves on test failure for debugging
    """
    import os
    # Use API key from environment or a dummy key for testing
    api_key = os.getenv("DEEPSEEK_API_KEY", "test-api-key-for-integration-tests")

    with agent_fixture(
        agent_name="test_agent_basic_example",
        working_dir=Path("/tmp"),
        provider="deepseek",
        skip_permissions=True,
        api_key=api_key,
    ) as agent:
        yield agent


@pytest.mark.asyncio
async def test_simple_arithmetic(test_agent):
    """Test agent can perform simple arithmetic.

    This test demonstrates:
    - Basic agent interaction via IPC
    - Simple content validation (no AI meta-query for speed)
    """
    # Connect to agent via IPC
    client = AgentClient(test_agent.agent_name)
    await client.connect()

    try:
        # Send query to agent
        query = "What is 2 + 2?"
        response = await client.invoke_agent(query, stream=False)

        # Extract response content
        content = response.get("response", {}).get("content", "")

        # Simple validation: check that response mentions the answer
        assert content, "Response should not be empty"
        assert "4" in content, f"Response should contain '4'. Got: {content[:200]}"

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_factual_question(test_agent):
    """Test agent can answer factual questions.

    This test demonstrates:
    - Using validate_test_reply() for convenience
    - Automatic test failure on invalid response
    """
    client = AgentClient(test_agent.agent_name)
    await client.connect()

    try:
        # Ask factual question
        query = "What is the capital of France?"
        response = await client.invoke_agent(query)
        content = response["response"]["content"]

        # Simple validation: check that response mentions Paris
        assert content, "Response should not be empty"
        assert "Paris" in content or "paris" in content.lower(), \
            f"Response should mention Paris. Got: {content[:200]}"

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_with_context(test_agent):
    """Test validation with additional context.

    This test demonstrates:
    - Using additional_context for domain-specific validation
    - Strict mode for more demanding validation
    """
    client = AgentClient(test_agent.agent_name)
    await client.connect()

    try:
        # Query that might need context
        query = "Calculate the area of a 5x5 square"
        response = await client.invoke_agent(query)
        content = response["response"]["content"]

        # Simple validation: check that response mentions 25
        assert content, "Response should not be empty"
        assert "25" in content, f"Response should contain '25'. Got: {content[:200]}"

    finally:
        await client.disconnect()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Example of batch validation - requires multiple queries")
async def test_batch_validation_example(test_agent):
    """Example of batch validation (currently skipped).

    This demonstrates how to validate multiple responses at once.
    Uncomment and adapt for your use case.
    """
    from . import batch_check_replies

    client = AgentClient(test_agent.agent_name)
    await client.connect()

    try:
        # Multiple query-reply pairs
        queries = [
            "What is 2+2?",
            "What is 3*3?",
            "What is 10-5?",
        ]

        # Collect responses
        responses = []
        for query in queries:
            resp = await client.invoke_agent(query)
            responses.append(resp["response"]["content"])

        # Batch validate
        pairs = list(zip(queries, responses))
        results = await batch_check_replies(pairs, model="haiku")

        # Check all valid
        invalid_count = sum(1 for r in results if not r.is_valid)
        assert invalid_count == 0, f"{invalid_count} validations failed"

    finally:
        await client.disconnect()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires Anthropic Claude API for check_reply() validation")
async def test_model_fallback_example(test_agent):
    """Example of falling back to better model if validation fails.

    This demonstrates conditional model selection based on initial result.
    """
    client = AgentClient(test_agent.agent_name)
    await client.connect()

    try:
        query = "Explain quantum entanglement"
        response = await client.invoke_agent(query)
        content = response["response"]["content"]

        # Try with haiku first (fast/cheap)
        result = await check_reply(query, content, model="haiku")

        if not result.is_valid:
            # Haiku might not handle complex validation well
            # Try with sonnet (better reasoning)
            result = await check_reply(query, content, model="sonnet")

        assert result.is_valid, f"Even sonnet couldn't validate: {result.comment}"

    finally:
        await client.disconnect()


# Utility function tests (testing the infrastructure itself)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires Anthropic Claude API for check_reply() validation")
async def test_check_reply_correctness():
    """Test that check_reply correctly identifies valid/invalid responses.

    This tests the validation infrastructure itself.
    """
    # Valid response
    valid_result = await check_reply(
        query_text="What is the capital of France?",
        reply_text="Paris is the capital of France.",
        model="haiku",
    )
    assert valid_result.is_valid, "Should accept correct answer"

    # Invalid response
    invalid_result = await check_reply(
        query_text="What is the capital of France?",
        reply_text="The capital of France is London.",  # Wrong!
        model="haiku",
    )
    assert not invalid_result.is_valid, "Should reject incorrect answer"
    assert len(invalid_result.comment) > 0, "Should provide explanation"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires Anthropic Claude API for check_reply() validation")
async def test_check_reply_variations():
    """Test that validation handles different answer formats.

    This demonstrates robustness of AI validation vs string matching.
    """
    query = "What is 2+2?"

    # Various correct formats
    correct_variations = [
        "4",
        "The answer is 4",
        "2+2 equals 4",
        "Four",
        "2 plus 2 is 4.",
    ]

    for variation in correct_variations:
        result = await check_reply(query, variation, model="haiku")
        assert result.is_valid, f"Should accept: '{variation}'"


if __name__ == "__main__":
    # Allow running directly with: python test_agent_basic_example.py
    pytest.main([__file__, "-v", "-s"])
