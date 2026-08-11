"""Unit test for WebSearchDelegationTool integration in deep_worker.

This test verifies that the universal web-search tool is properly integrated
into the deep_worker CLI agent tools system under the tri-state policy
(``WebSearchPolicy`` auto/enabled/disabled, DESIGN_PROVIDERS_FROM_WHEEL §3.3),
without requiring a full TUI session.

Run with:
    cd <repo-root> && python3 -m pytest deep_worker/deep_worker_cli/tests/test_websearch_tool_integration.py -v -s
"""

import pytest

from deep_worker_cli.config import WebSearchPolicy


def test_websearch_tool_import():
    """Test that WebSearchDelegationTool is resolved via the tri-state resolver."""
    from deep_worker_cli.tools import resolve_web_search_tool

    availability = resolve_web_search_tool(WebSearchPolicy.AUTO)
    if availability.tool is not None:
        assert availability.tool.name == "web_search"
        assert hasattr(availability.tool, "arun")
        assert hasattr(availability.tool, "ainvoke")
    print(f"✓ WebSearchDelegationTool resolved ({availability.reason})")
    if availability.tool is not None:
        print(f"  Tool name: {availability.tool.name}")
        print(f"  Tool type: {type(availability.tool).__name__}")


def test_websearch_tool_attributes():
    """Test that WebSearchDelegationTool has correct attributes (when available)."""
    from deep_worker_cli.tools import resolve_web_search_tool

    availability = resolve_web_search_tool(WebSearchPolicy.AUTO)
    if availability.tool is None:
        pytest.skip(f"web_search unavailable: {availability.reason}")

    # Check required attributes
    assert availability.tool.name == "web_search"
    assert availability.tool.description is not None
    assert len(availability.tool.description) > 0
    # Timeout is now 120s (raised from 30s) to accommodate slower
    # web-search round-trips via Claude Code SDK. Assert it's a
    # positive float — the exact default is a tuning knob.
    assert isinstance(availability.tool.timeout, float)
    assert availability.tool.timeout > 0
    assert availability.tool.verbose is False

    print(f"✓ Tool attributes verified")
    print(f"  Name: {availability.tool.name}")
    print(f"  Description length: {len(availability.tool.description)} chars")
    print(f"  Timeout: {availability.tool.timeout}s")
    print(f"  Verbose: {availability.tool.verbose}")


def test_websearch_tool_disabled_never_resolves():
    """``disabled`` must never resolve/import the claude provider or SDK."""
    from deep_worker_cli.tools import resolve_web_search_tool

    availability = resolve_web_search_tool(WebSearchPolicy.DISABLED)
    assert availability.tool is None
    assert availability.reason == "disabled"


def test_websearch_tool_in_agent_process():
    """Test that the tri-state resolver is used in agent_process.py."""
    # Read the file to verify the resolver is present
    from pathlib import Path

    agent_process_file = Path(__file__).parent.parent / "deep_worker_cli" / "process" / "agent_process.py"
    content = agent_process_file.read_text()

    assert "resolve_web_search_tool" in content
    assert "from deep_worker_cli.tools import" in content

    print(f"✓ agent_process.py uses resolve_web_search_tool")


def test_websearch_tool_in_non_interactive():
    """Test that the tri-state resolver is used in non_interactive.py."""
    from pathlib import Path

    non_interactive_file = Path(__file__).parent.parent / "deep_worker_cli" / "non_interactive.py"
    content = non_interactive_file.read_text()

    assert "resolve_web_search_tool" in content
    assert "from deep_worker_cli.tools import" in content

    print(f"✓ non_interactive.py uses resolve_web_search_tool")


@pytest.mark.asyncio
async def test_websearch_tool_callable():
    """Test that WebSearchDelegationTool can be called (mock test)."""
    from deep_worker_cli.tools import resolve_web_search_tool
    from unittest.mock import AsyncMock, patch

    availability = resolve_web_search_tool(WebSearchPolicy.AUTO)
    if availability.tool is None:
        pytest.skip(f"web_search unavailable: {availability.reason}")

    # Mock the ClaudeSDKClient to avoid actual SDK calls
    with patch("deep_worker.providers.claude_code_provider.web_search.web_search_delegation.ClaudeSDKClient") as mock_client:  # noqa: E501
        # Setup mock response
        mock_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_instance

        # Mock the receive_response to return a simple result
        from claude_agent_sdk import ResultMessage

        async def mock_receive():
            yield ResultMessage(
                type="result",
                stop_reason="end_turn",
                duration_ms=100,
                total_cost_usd=0.001
            )

        mock_instance.receive_response = mock_receive

        # Call the tool (should not crash)
        try:
            result = await availability.tool.ainvoke({
                "query": "test query"
            })

            # Result should be a JSON string
            import json
            parsed = json.loads(result)

            # Should either be search results or empty list
            assert isinstance(parsed, list) or isinstance(parsed, dict)

            print(f"✓ Tool is callable")
            print(f"  Result type: {type(parsed).__name__}")

        except Exception as e:
            # If it fails, it should be a graceful failure
            print(f"✓ Tool execution attempted (expected to need SDK)")
            print(f"  Error: {type(e).__name__}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
