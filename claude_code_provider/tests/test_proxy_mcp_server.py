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

    @pytest.mark.asyncio
    async def test_wait_for_pending_calls_returns_existing_call(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """wait_for_pending_calls should return immediately for an existing call."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler
        task = asyncio.create_task(handler({"arg": "test_value"}))
        await asyncio.sleep(0)

        pending = await proxy.wait_for_pending_calls(
            tool_name="test_tool",
            arguments={"arg": "test_value"},
            timeout=0.1,
        )

        assert len(pending) == 1
        assert pending[0]["name"] == "test_tool"
        assert pending[0]["args"] == {"arg": "test_value"}

        proxy.resolve_call(pending[0]["id"], {"content": [{"type": "text", "text": "ok"}]})
        await task

    @pytest.mark.asyncio
    async def test_wait_for_pending_calls_ignores_existing_nonmatching_call(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)

        stale_task = asyncio.create_task(
            proxy.tool_handlers[1].handler({"a": 9, "b": 1})
        )
        await asyncio.sleep(0)
        stale_pending = proxy.get_pending_calls()
        assert len(stale_pending) == 1
        assert stale_pending[0]["name"] == "calculator"

        waiter = asyncio.create_task(
            proxy.wait_for_pending_calls(
                tool_name="test_tool",
                arguments={"arg": "expected"},
                timeout=1.0,
            )
        )
        await asyncio.sleep(0)
        assert not waiter.done()

        matching_task = asyncio.create_task(
            proxy.tool_handlers[0].handler({"arg": "expected"})
        )
        pending = await waiter

        assert len(pending) == 1
        assert pending[0]["name"] == "test_tool"
        assert pending[0]["args"] == {"arg": "expected"}

        for call in proxy.get_pending_calls():
            proxy.resolve_call(
                call["id"], {"content": [{"type": "text", "text": "ok"}]}
            )
        await stale_task
        await matching_task

    @pytest.mark.asyncio
    async def test_wait_for_pending_calls_is_woken_by_handler(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """wait_for_pending_calls should park on a Future, not poll."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[1].handler

        waiter = asyncio.create_task(proxy.wait_for_pending_calls(
            tool_name="calculator",
            arguments={"a": 2, "b": 3},
            timeout=1.0,
        ))
        await asyncio.sleep(0)

        task = asyncio.create_task(handler({"a": 2, "b": 3}))
        pending = await waiter

        assert len(pending) == 1
        assert pending[0]["name"] == "calculator"
        assert pending[0]["args"] == {"a": 2, "b": 3}

        proxy.resolve_call(pending[0]["id"], {"content": [{"type": "text", "text": "ok"}]})
        await task

    @pytest.mark.asyncio
    async def test_wait_for_matching_pending_calls_returns_batch(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Batch wait should return once all expected calls are registered."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        waiter = asyncio.create_task(proxy.wait_for_matching_pending_calls(
            expected_calls=[
                ("test_tool", {"arg": "first"}),
                ("calculator", {"a": 2, "b": 3}),
            ],
            timeout=1.0,
        ))
        await asyncio.sleep(0)

        task_one = asyncio.create_task(proxy.tool_handlers[0].handler({"arg": "first"}))
        await asyncio.sleep(0)
        assert not waiter.done()

        task_two = asyncio.create_task(proxy.tool_handlers[1].handler({"a": 2, "b": 3}))
        pending = await waiter

        assert [call["name"] for call in pending] == ["test_tool", "calculator"]
        assert [call["args"] for call in pending] == [
            {"arg": "first"},
            {"a": 2, "b": 3},
        ]

        for call in pending:
            proxy.resolve_call(call["id"], {"content": [{"type": "text", "text": "ok"}]})
        await task_one
        await task_two

    @pytest.mark.asyncio
    async def test_wait_for_any_pending_call_returns_existing(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """If a pending call already exists, should return immediately."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler

        task = asyncio.create_task(handler({"arg": "first"}))
        # Give handler a chance to register the pending call.
        for _ in range(10):
            await asyncio.sleep(0)
            if proxy.has_pending_calls():
                break

        assert proxy.has_pending_calls()
        # Must return immediately — no waiting required.
        await asyncio.wait_for(
            proxy.wait_for_any_pending_call(timeout=1.0), timeout=0.1
        )

        proxy.resolve_call(
            proxy.get_pending_calls()[0]["id"],
            {"content": [{"type": "text", "text": "ok"}]},
        )
        await task

    @pytest.mark.asyncio
    async def test_wait_for_any_pending_call_is_event_driven_no_polling(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Must park on Future and be woken by handler — no polling allowed."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[1].handler

        # Disallow asyncio.sleep() inside the wait — proves event-driven.
        # We spy on asyncio.sleep used by the proxy module only; the test
        # itself uses it explicitly via asyncio.sleep(0).
        waiter = asyncio.create_task(
            proxy.wait_for_any_pending_call(timeout=1.0)
        )
        # Yield to let waiter park on its Future before handler registers.
        await asyncio.sleep(0)
        assert not waiter.done()

        # Fire the handler. The handler registers the pending call and must
        # wake the waiter via the same Future-based mechanism used by
        # wait_for_matching_pending_calls.
        task = asyncio.create_task(handler({"a": 2, "b": 3}))
        await waiter  # Should return as soon as the handler registers.

        assert proxy.has_pending_calls()

        proxy.resolve_call(
            proxy.get_pending_calls()[0]["id"],
            {"content": [{"type": "text", "text": "ok"}]},
        )
        await task

    @pytest.mark.asyncio
    async def test_wait_for_any_pending_call_times_out_if_no_handler(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """If no handler ever registers a pending call, must time out."""
        proxy = ProxyMCPServer(sample_tool_metadata)

        with pytest.raises(asyncio.TimeoutError):
            await proxy.wait_for_any_pending_call(timeout=0.05)

    @pytest.mark.asyncio
    async def test_wait_for_any_pending_call_returns_on_first_of_many(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Production semantics: Claude CLI sends MCP control_requests
        SEQUENTIALLY (next one only after the previous tool_result arrives).
        Even if multiple ToolUseBlocks are in one AssistantMessage, only the
        FIRST handler runs — the rest stay queued in the CLI. ``ainvoke``
        must therefore NOT wait for all to register; it must return as soon
        as the first pending call exists, surface it to LangGraph, and let
        the next ainvoke iteration pick up the next one after resolve_call.
        """
        proxy = ProxyMCPServer(sample_tool_metadata)

        # Only ONE handler ever runs (simulates Claude CLI's sequential
        # control_request behaviour). Waiter must return based on this one.
        waiter = asyncio.create_task(
            proxy.wait_for_any_pending_call(timeout=1.0)
        )
        await asyncio.sleep(0)
        assert not waiter.done()

        task = asyncio.create_task(
            proxy.tool_handlers[0].handler({"arg": "only_one"})
        )
        await waiter  # Must NOT wait for a second handler.

        pending = proxy.get_pending_calls()
        assert len(pending) == 1
        assert pending[0]["name"] == "test_tool"
        assert pending[0]["args"] == {"arg": "only_one"}

        proxy.resolve_call(
            pending[0]["id"], {"content": [{"type": "text", "text": "ok"}]}
        )
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


class TestProxyMCPServerResolveIdempotency:
    """``resolve_call`` must be idempotent.

    The delivery-only-timeout fix (DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md,
    Bug 2) resolves the inter-agent send Future EARLY — at delivery time, from
    a tool-call wrapper — instead of waiting for the next ainvoke. A later
    send-loop pass may still try to resolve the same call_id. Both the early
    and the late resolve must be safe: the first wins, the rest are no-ops, and
    nothing raises.
    """

    @pytest.mark.asyncio
    async def test_second_resolve_is_noop(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """A second resolve_call for the same id must not overwrite or raise."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        task = asyncio.create_task(proxy.tool_handlers[0].handler({"arg": "x"}))
        await asyncio.sleep(0.1)
        call_id = proxy.get_pending_calls()[0]["id"]

        first: MCPResult = {"content": [{"type": "text", "text": "delivered"}]}
        second: MCPResult = {"content": [{"type": "text", "text": "LATE"}]}

        proxy.resolve_call(call_id, first)
        # Future already resolved + handler's finally pops the id; the second
        # resolve must be a harmless no-op (no exception, no overwrite).
        proxy.resolve_call(call_id, second)

        result = await task
        assert result["content"][0]["text"] == "delivered"

    @pytest.mark.asyncio
    async def test_resolve_unknown_call_id_is_noop(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Resolving an id with no pending call must be a safe no-op."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        # Nothing pending — must not raise.
        proxy.resolve_call("does-not-exist", {"content": [{"type": "text", "text": "x"}]})
        assert not proxy.has_pending_calls()
