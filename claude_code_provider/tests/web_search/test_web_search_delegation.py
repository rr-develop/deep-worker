# pyright: reportGeneralTypeIssues=false, reportTypedDictNotRequiredAccess=false
"""Unit tests for WebSearch delegation tool.

This module contains comprehensive tests for the WebSearchDelegationTool,
including input validation, hook enforcement, result parsing, and error handling.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from claude_code_provider.web_search.web_search_delegation import (
    WebSearchDelegationTool,
    WebSearchInput,
)
from claude_code_provider.web_search.web_search_hooks import (
    create_enforcement_hook,
    create_logging_hook,
    create_validation_hook,
)


class TestWebSearchInput:
    """Test input schema validation."""

    def test_valid_basic_input(self):
        """Test basic input with only query."""
        input_data = WebSearchInput(query="Python tutorials")
        assert input_data.query == "Python tutorials"
        assert input_data.allowed_domains is None
        assert input_data.blocked_domains is None

    def test_valid_input_with_domains(self):
        """Test input with domain filters."""
        input_data = WebSearchInput(
            query="Python",
            allowed_domains=["python.org", "github.com"],
            blocked_domains=["spam.com"]
        )
        assert input_data.query == "Python"
        assert input_data.allowed_domains == ["python.org", "github.com"]
        assert input_data.blocked_domains == ["spam.com"]

    def test_empty_query_allowed(self):
        """Test that empty query is technically valid (though not useful)."""
        input_data = WebSearchInput(query="")
        assert input_data.query == ""

    def test_schema_fields(self):
        """Test that schema has correct field descriptions."""
        schema = WebSearchInput.model_json_schema()
        assert "query" in schema["properties"]
        assert "allowed_domains" in schema["properties"]
        assert "blocked_domains" in schema["properties"]


class TestWebSearchDelegationTool:
    """Test WebSearchDelegationTool functionality."""

    def test_tool_metadata(self):
        """Test tool name, description, and schema."""
        tool = WebSearchDelegationTool()

        assert tool.name == "web_search"
        assert "Search the web" in tool.description
        assert tool.args_schema == WebSearchInput

    def test_tool_configuration(self):
        """Test tool configuration parameters."""
        tool = WebSearchDelegationTool(timeout=60.0, verbose=True)

        assert tool.timeout == 60.0
        assert tool.verbose is True
        assert tool.model == "claude-sonnet-4-5-20250929"

    def test_sync_run_not_supported(self):
        """Test that synchronous run raises NotImplementedError."""
        tool = WebSearchDelegationTool()

        with pytest.raises(NotImplementedError) as exc_info:
            tool._run(query="test")  # type: ignore[reportPrivateUsage]

        assert "async operations" in str(exc_info.value)
        assert "arun()" in str(exc_info.value)

    def test_build_search_prompt_basic(self):
        """Test basic search prompt building."""
        tool = WebSearchDelegationTool()
        prompt = tool._build_search_prompt("Python tutorials", None, None)  # type: ignore[reportPrivateUsage]

        assert "Search the web for: Python tutorials" in prompt
        assert "Use the WebSearch tool" in prompt

    def test_build_search_prompt_with_allowed_domains(self):
        """Test prompt with allowed domains."""
        tool = WebSearchDelegationTool()
        prompt = tool._build_search_prompt(  # type: ignore[reportPrivateUsage]
            "Python",
            allowed_domains=["python.org", "stackoverflow.com"],
            blocked_domains=None
        )

        assert "python.org" in prompt
        assert "stackoverflow.com" in prompt
        assert "Only include results from these domains" in prompt

    def test_build_search_prompt_with_blocked_domains(self):
        """Test prompt with blocked domains."""
        tool = WebSearchDelegationTool()
        prompt = tool._build_search_prompt(  # type: ignore[reportPrivateUsage]
            "Python",
            allowed_domains=None,
            blocked_domains=["spam.com", "ads.com"]
        )

        assert "spam.com" in prompt
        assert "ads.com" in prompt
        assert "Exclude results from these domains" in prompt

    def test_parse_websearch_result_empty(self):
        """Test parsing empty result."""
        tool = WebSearchDelegationTool()
        results = tool._parse_websearch_result(None)  # type: ignore[reportPrivateUsage]
        assert results == []

    def test_parse_websearch_result_json_string(self):
        """Test parsing JSON string result."""
        tool = WebSearchDelegationTool()
        json_result = json.dumps([
            {
                "title": "Python Tutorial",
                "url": "https://python.org/tutorial",
                "snippet": "Learn Python"
            }
        ])

        results = tool._parse_websearch_result(json_result)  # type: ignore[reportPrivateUsage]

        assert len(results) == 1
        assert results[0]["title"] == "Python Tutorial"
        assert results[0]["url"] == "https://python.org/tutorial"
        assert results[0]["snippet"] == "Learn Python"

    def test_parse_websearch_result_list_with_text_blocks(self):
        """Test parsing list with MCP text blocks."""
        tool = WebSearchDelegationTool()
        mcp_result = [
            {
                "type": "text",
                "text": json.dumps([
                    {"title": "Result 1", "url": "https://example.com", "snippet": "Snippet 1"}
                ])
            }
        ]

        results = tool._parse_websearch_result(mcp_result)  # type: ignore[reportPrivateUsage]

        assert len(results) == 1
        assert results[0]["title"] == "Result 1"

    def test_parse_websearch_result_plain_text(self):
        """Test parsing plain text result."""
        tool = WebSearchDelegationTool()
        text_result = "Some plain text result"

        results = tool._parse_websearch_result(text_result)  # type: ignore[reportPrivateUsage]

        assert len(results) == 1
        assert results[0]["title"] == "Search Result"
        assert results[0]["snippet"] == "Some plain text result"

    def test_parse_websearch_result_list_of_dicts(self):
        """Test parsing list of result dicts."""
        tool = WebSearchDelegationTool()
        list_result = [
            {"title": "Result 1", "url": "https://ex1.com", "snippet": "Snippet 1"},
            {"title": "Result 2", "url": "https://ex2.com", "snippet": "Snippet 2"}
        ]

        results = tool._parse_websearch_result(list_result)  # type: ignore[reportPrivateUsage]

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[1]["title"] == "Result 2"

    @pytest.mark.asyncio
    async def test_arun_timeout_error(self):
        """Test that timeout errors are handled gracefully."""
        tool = WebSearchDelegationTool(timeout=0.001, verbose=False)

        with patch("claude_code_provider.web_search.web_search_delegation.ClaudeSDKClient") as mock_client:
            # Mock client that takes too long
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            # Create async generator that never yields
            async def never_completes():
                await asyncio.sleep(10)
                if False:  # Never reached
                    yield

            # Mock receive_response to return the async generator
            mock_instance.receive_response = never_completes

            # Execute with short timeout - use ainvoke instead of arun
            result = await tool.ainvoke({"query": "test query"})

            # Should return error JSON
            parsed = json.loads(result)
            assert "error" in parsed
            assert "timeout" in parsed["error"].lower()
            assert parsed["query"] == "test query"

    @pytest.mark.asyncio
    async def test_arun_exception_handling(self):
        """Test that exceptions are caught and returned as JSON."""
        tool = WebSearchDelegationTool(verbose=False)

        with patch("claude_code_provider.web_search.web_search_delegation.ClaudeSDKClient") as mock_client:
            # Mock client that raises exception
            mock_client.return_value.__aenter__.side_effect = Exception("Connection failed")

            # Execute - use ainvoke instead of arun
            result = await tool.ainvoke({"query": "test query"})

            # Should return error JSON
            parsed = json.loads(result)
            assert "error" in parsed
            assert "Connection failed" in parsed["error"]
            assert parsed["query"] == "test query"


class TestWebSearchHooks:
    """Test hook functions."""

    @pytest.mark.asyncio
    async def test_enforcement_hook_enforces_params(self):
        """Test that enforcement hook replaces Claude's parameters."""
        original_params = {
            "query": "original query",
            "allowed_domains": ["original.com"]
        }

        hook = create_enforcement_hook(original_params)

        # Simulate Claude modifying the query
        hook_input = {
            "tool_name": "WebSearch",
            "tool_input": {
                "query": "MODIFIED query",
                "allowed_domains": ["modified.com"]
            },
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp"
        }

        # Call hook
        result = await hook(hook_input, "tool_123", {"signal": None})  # type: ignore[arg-type]

        # Verify original params were enforced
        assert result["hookSpecificOutput"]["updatedInput"] == original_params
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "original query" in result["hookSpecificOutput"]["permissionDecisionReason"]

    @pytest.mark.asyncio
    async def test_enforcement_hook_ignores_other_tools(self):
        """Test that enforcement hook doesn't affect other tools."""
        original_params = {"query": "test"}
        hook = create_enforcement_hook(original_params)

        # Different tool
        hook_input = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/file.txt"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp"
        }

        # Call hook
        result = await hook(hook_input, "tool_123", {"signal": None})  # type: ignore[arg-type]

        # Should return empty (no intervention)
        assert result == {}

    @pytest.mark.asyncio
    async def test_logging_hook(self):
        """Test logging hook logs without modification."""
        hook = create_logging_hook(verbose=False)

        hook_input = {
            "tool_name": "WebSearch",
            "tool_input": {"query": "test"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp"
        }

        # Call hook
        result = await hook(hook_input, "tool_123", {"signal": None})  # type: ignore[arg-type]

        # Should allow without modification
        assert result == {}

    @pytest.mark.asyncio
    async def test_validation_hook_denies_long_query(self):
        """Test validation hook denies overly long queries."""
        hook = create_validation_hook(max_query_length=10)

        hook_input = {
            "tool_name": "WebSearch",
            "tool_input": {"query": "This is a very long query that exceeds the limit"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp"
        }

        # Call hook
        result = await hook(hook_input, "tool_123", {"signal": None})  # type: ignore[arg-type]

        # Should deny
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "too long" in result["hookSpecificOutput"]["permissionDecisionReason"].lower()

    @pytest.mark.asyncio
    async def test_validation_hook_allows_valid_query(self):
        """Test validation hook allows valid queries."""
        hook = create_validation_hook(max_query_length=100)

        hook_input = {
            "tool_name": "WebSearch",
            "tool_input": {"query": "Python"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp"
        }

        # Call hook
        result = await hook(hook_input, "tool_123", {"signal": None})  # type: ignore[arg-type]

        # Should allow
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"

    @pytest.mark.asyncio
    async def test_validation_hook_requires_domain_restriction(self):
        """Test validation hook can require domain restrictions."""
        hook = create_validation_hook(
            allowed_domains_only=["trusted.com"]
        )

        # Unrestricted search
        hook_input = {
            "tool_name": "WebSearch",
            "tool_input": {"query": "test"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp"
        }

        result = await hook(hook_input, "tool_123", {"signal": None})  # type: ignore[arg-type]

        # Should deny
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "restricted to" in result["hookSpecificOutput"]["permissionDecisionReason"].lower()


# Integration test marker
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_websearch_delegation():
    """Integration test with real Claude Code SDK.

    This test requires:
    - Claude CLI installed and authenticated
    - Internet connection

    It performs a real web search and verifies the results.
    """
    pytest.skip("Skipping integration test - requires Claude CLI setup")

    tool = WebSearchDelegationTool(verbose=True, timeout=60.0)

    # Perform real search
    result = await tool.arun(
        query="Python 3.12 release date",
        allowed_domains=["python.org"]
    )

    # Parse result
    parsed = json.loads(result)

    # Should not be an error
    assert "error" not in parsed

    # Should be a list of results
    assert isinstance(parsed, list)
    assert len(parsed) > 0

    # Check first result structure
    first = parsed[0]
    assert "title" in first
    assert "url" in first
    assert "snippet" in first

    # Verify domain filtering worked
    assert "python.org" in first["url"]

    print(f"✅ Search returned {len(parsed)} results")
    print(f"✅ First result: {first['title']}")
