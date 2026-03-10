"""Tests for ProxyMCPServer - in-process MCP server for tool delegation."""

import asyncio

import pytest

from claude_code_provider.proxy_mcp_server import ProxyMCPServer
from claude_code_provider.custom_types import MCPResult, ToolMetadata


class TestProxyMCPServerInit:
    """Tests for ProxyMCPServer initialization."""

    def test_init_with_empty_tools(self) -> None:
        """ProxyMCPServer should initialize with empty tools list."""
        proxy = ProxyMCPServer([])
        assert proxy.mcp_server_config is not None
        assert proxy.mcp_server_config["type"] == "sdk"
        assert proxy.mcp_server_config["name"] == "deepworker"

    def test_init_with_single_tool(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """ProxyMCPServer should initialize with a single tool."""
        single_tool = [sample_tool_metadata[0]]
        proxy = ProxyMCPServer(single_tool)
        assert proxy.mcp_server_config is not None
        assert proxy.mcp_server_config["type"] == "sdk"
        assert len(proxy.tool_handlers) == 1

    def test_init_with_multiple_tools(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """ProxyMCPServer should initialize with multiple tools."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        assert proxy.mcp_server_config is not None
        assert len(proxy.tool_handlers) == len(sample_tool_metadata)


class TestProxyMCPServerPendingCalls:
    """Tests for pending calls management."""

    def test_has_pending_calls_initially_false(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """has_pending_calls should return False initially."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        assert proxy.has_pending_calls() is False

    def test_get_pending_calls_initially_empty(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """get_pending_calls should return empty list initially."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        assert proxy.get_pending_calls() == []

    @pytest.mark.asyncio
    async def test_pending_call_after_handler_invocation(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """After handler is called, there should be a pending call."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        # Start handler in background (it will block on Future)
        handler = proxy.tool_handlers[0].handler
        task = asyncio.create_task(handler({"arg": "test_value"}))

        # Give it time to start and block
        await asyncio.sleep(0.1)

        # Should have pending call now
        assert proxy.has_pending_calls() is True
        pending = proxy.get_pending_calls()
        assert len(pending) == 1
        assert pending[0]["name"] == "test_tool"
        assert pending[0]["args"] == {"arg": "test_value"}

        # Cancel task to prevent hanging
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestProxyMCPServerHandlerBlocking:
    """Tests for handler blocking behavior."""

    @pytest.mark.asyncio
    async def test_handler_blocks_on_future(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """Handler should block until Future is resolved."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler

        # Start handler in background
        task = asyncio.create_task(handler({"arg": "test"}))

        # Give it time to start
        await asyncio.sleep(0.1)

        # Task should not be done (blocked on Future)
        assert not task.done()

        # Cancel to cleanup
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_handler_returns_after_resolve(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """Handler should return result after Future is resolved."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler

        # Start handler in background
        task = asyncio.create_task(handler({"arg": "test"}))

        # Give it time to start
        await asyncio.sleep(0.1)

        # Get pending call ID
        pending = proxy.get_pending_calls()
        assert len(pending) == 1
        call_id = pending[0]["id"]

        # Resolve with result
        result: MCPResult = {"content": [{"type": "text", "text": "Success"}]}
        proxy.resolve_call(call_id, result)

        # Handler should complete
        handler_result = await task
        assert handler_result == result


class TestProxyMCPServerResolveCall:
    """Tests for resolve_call functionality."""

    @pytest.mark.asyncio
    async def test_resolve_call_with_valid_id(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """resolve_call should resolve Future with given result."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler

        # Start handler
        task = asyncio.create_task(handler({"arg": "test"}))
        await asyncio.sleep(0.1)

        # Get call ID
        pending = proxy.get_pending_calls()
        call_id = pending[0]["id"]

        # Resolve
        result: MCPResult = {"content": [{"type": "text", "text": "Result"}]}
        proxy.resolve_call(call_id, result)

        # Should get result
        handler_result = await task
        assert handler_result == result

    def test_resolve_call_with_invalid_id(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """resolve_call with invalid ID should not raise error."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        # Should not raise
        proxy.resolve_call("nonexistent_id", {"content": []})

    @pytest.mark.asyncio
    async def test_resolve_call_removes_from_pending(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """resolve_call should remove call from pending list."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler

        # Start handler
        task = asyncio.create_task(handler({"arg": "test"}))
        await asyncio.sleep(0.1)

        # Verify pending
        assert proxy.has_pending_calls() is True
        call_id = proxy.get_pending_calls()[0]["id"]

        # Resolve
        proxy.resolve_call(call_id, {"content": []})
        await task

        # Should be removed from pending
        assert proxy.has_pending_calls() is False
        assert proxy.get_pending_calls() == []


class TestProxyMCPServerTimeout:
    """Tests for handler timeout behavior."""

    @pytest.mark.asyncio
    async def test_handler_timeout(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """Handler should return error after timeout."""
        # Create proxy with short timeout for testing
        proxy = ProxyMCPServer(sample_tool_metadata, timeout=0.2)
        handler = proxy.tool_handlers[0].handler

        # Call handler without resolving
        result = await handler({"arg": "test"})

        # Should return timeout error
        assert "is_error" in result
        assert result["is_error"] is True
        assert "timeout" in result["content"][0]["text"].lower()


class TestProxyMCPServerMultipleCalls:
    """Tests for handling multiple simultaneous tool calls."""

    @pytest.mark.asyncio
    async def test_multiple_pending_calls(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """Should handle multiple pending calls simultaneously."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        # Start multiple handlers
        handler1 = proxy.tool_handlers[0].handler
        handler2 = proxy.tool_handlers[1].handler

        task1 = asyncio.create_task(handler1({"arg": "call1"}))
        task2 = asyncio.create_task(handler2({"a": 1, "b": 2}))

        await asyncio.sleep(0.1)

        # Should have 2 pending calls
        assert len(proxy.get_pending_calls()) == 2

        # Resolve both
        pending = proxy.get_pending_calls()
        for call in pending:
            proxy.resolve_call(call["id"], {"content": [{"type": "text", "text": "Done"}]})

        # Both should complete
        await task1
        await task2

        # No pending calls left
        assert proxy.has_pending_calls() is False

    @pytest.mark.asyncio
    async def test_resolve_calls_independently(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """Should resolve calls independently."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        # Start 2 handlers
        task1 = asyncio.create_task(proxy.tool_handlers[0].handler({"arg": "1"}))
        task2 = asyncio.create_task(proxy.tool_handlers[0].handler({"arg": "2"}))

        await asyncio.sleep(0.1)

        pending = proxy.get_pending_calls()
        assert len(pending) == 2

        # Resolve first
        proxy.resolve_call(pending[0]["id"], {"content": [{"type": "text", "text": "First"}]})
        result1 = await task1

        # Second should still be pending
        assert proxy.has_pending_calls() is True

        # Resolve second
        proxy.resolve_call(pending[1]["id"], {"content": [{"type": "text", "text": "Second"}]})
        result2 = await task2

        assert result1["content"][0]["text"] == "First"
        assert result2["content"][0]["text"] == "Second"


class TestProxyMCPServerErrorHandling:
    """Tests for error handling in ProxyMCPServer."""

    @pytest.mark.asyncio
    async def test_resolve_with_error_flag(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        """Should handle results with is_error flag."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        task = asyncio.create_task(proxy.tool_handlers[0].handler({"arg": "test"}))
        await asyncio.sleep(0.1)

        call_id = proxy.get_pending_calls()[0]["id"]

        # Resolve with error
        error_result: MCPResult = {
            "content": [{"type": "text", "text": "Error occurred"}],
            "is_error": True,
        }
        proxy.resolve_call(call_id, error_result)

        result = await task
        assert result["is_error"] is True
        assert "Error occurred" in result["content"][0]["text"]
