# pyright: reportPrivateUsage=false, reportUnusedImport=false, reportAttributeAccessIssue=false
"""Tests for ProxyMCPServer Future-based proxy architecture.

⚠️ ARCHITECTURAL INVARIANT: Tool calls MUST go through LangGraph ToolNode.
ProxyMCPServer handlers block on asyncio.Future until LangGraph executes
the tool and calls resolve_call(). This ensures:
  1. Tool calls visible in result.messages (for dwcli broadcast)
  2. Checkpointer saves tool call history
  3. interrupt_on (HITL approval) works
  4. Tool execution is transparent to all consumers

DO NOT replace with direct execution in MCP handler!
Direct execution makes tools invisible to LangGraph and dwcli.
"""

import asyncio
import itertools
from typing import Any

import pytest

from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)
from deep_worker.providers.claude_code_provider.proxy_mcp_server import ProxyMCPServer
from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata


# Helper: invoke a handler with a Claude tool_use_id in the SDK ContextVar
# (required by the post-2026-06-14 contract — see incident document).
_id_counter = itertools.count(1)


async def _start_handler(
    proxy: ProxyMCPServer, handler_index: int, args: dict[str, Any]
) -> tuple[asyncio.Task[Any], str]:
    cid = f"toolu_direct_{next(_id_counter):05d}"
    handler = proxy.tool_handlers[handler_index].handler

    async def _runner() -> Any:
        return await handler(args)

    token = set_current_claude_tool_use_id(cid)
    try:
        task = asyncio.create_task(_runner())
        await asyncio.sleep(0)
        return task, cid
    finally:
        reset_current_claude_tool_use_id(token)


@pytest.fixture
def tools_metadata() -> list[ToolMetadata]:
    return [
        {"name": "ls", "description": "List files", "input_schema": {"properties": {"path": {"type": "string"}}}},
        {"name": "read_file", "description": "Read a file", "input_schema": {"properties": {"path": {"type": "string"}}}},
    ]


class TestProxyFutureArchitecture:
    """Tests that ProxyMCPServer uses Future-based proxy for ALL tools."""

    @pytest.mark.asyncio
    async def test_handler_blocks_on_future(self, tools_metadata: list[ToolMetadata]):
        """Handler must block on Future — NOT execute tool directly."""
        proxy = ProxyMCPServer(tools_metadata)

        task, cid = await _start_handler(proxy, 0, {"path": "/tmp"})
        await asyncio.sleep(0.05)

        # Handler must be pending (blocked on Future), not done
        assert not task.done(), "Handler returned immediately — proxy is broken!"
        assert proxy.has_pending_calls(), "No pending calls — Future not created!"

        # Resolve it — by the Claude tool_use_id (the key is the same one).
        proxy.resolve_call(cid, {
            "content": [{"type": "text", "text": "file1.txt"}],
        })

        result = await task
        assert result["content"][0]["text"] == "file1.txt"

    @pytest.mark.asyncio
    async def test_pending_calls_contain_tool_info(self, tools_metadata: list[ToolMetadata]):
        """Pending calls must contain tool name and args for LangGraph."""
        proxy = ProxyMCPServer(tools_metadata)

        task, cid = await _start_handler(proxy, 0, {"path": "/tmp"})
        await asyncio.sleep(0.05)

        pending = proxy.get_pending_calls()
        assert len(pending) == 1
        assert pending[0]["name"] == "ls"
        assert pending[0]["args"] == {"path": "/tmp"}
        assert pending[0]["id"] == cid  # key IS the Claude tool_use_id

        # Cleanup
        proxy.resolve_call(cid, {
            "content": [{"type": "text", "text": "ok"}],
        })
        await task

    @pytest.mark.asyncio
    async def test_resolve_call_unblocks_handler(self, tools_metadata: list[ToolMetadata]):
        """resolve_call must unblock the Future handler and return result to Claude."""
        proxy = ProxyMCPServer(tools_metadata)

        task, cid = await _start_handler(proxy, 0, {"path": "/tmp"})
        await asyncio.sleep(0.05)

        proxy.resolve_call(cid, {
            "content": [{"type": "text", "text": "resolved result"}],
        })

        result = await asyncio.wait_for(task, timeout=2.0)
        assert result["content"][0]["text"] == "resolved result"
        assert not proxy.has_pending_calls()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tool_calls(self, tools_metadata: list[ToolMetadata]):
        """Multiple tools called concurrently — all must be pending."""
        proxy = ProxyMCPServer(tools_metadata)

        task1, _ = await _start_handler(proxy, 0, {"path": "/tmp"})
        task2, _ = await _start_handler(proxy, 1, {"path": "/tmp/file.txt"})
        await asyncio.sleep(0.05)

        pending = proxy.get_pending_calls()
        assert len(pending) == 2

        # Resolve both
        for call in pending:
            proxy.resolve_call(call["id"], {
                "content": [{"type": "text", "text": f"result for {call['name']}"}],
            })

        r1 = await asyncio.wait_for(task1, timeout=2.0)
        r2 = await asyncio.wait_for(task2, timeout=2.0)
        assert "result for" in r1["content"][0]["text"]
        assert "result for" in r2["content"][0]["text"]
        assert not proxy.has_pending_calls()

    @pytest.mark.asyncio
    async def test_no_direct_execution(self, tools_metadata: list[ToolMetadata]):
        """ProxyMCPServer must NOT have tool_executors or direct execution.

        ⚠️ REGRESSION GUARD: If this test fails, someone added direct execution
        back. Tool calls MUST go through LangGraph ToolNode via Future proxy.
        """
        proxy = ProxyMCPServer(tools_metadata)

        # No tool_executors attribute
        assert not hasattr(proxy, '_tool_executors') or not proxy._tool_executors, (
            "ProxyMCPServer has _tool_executors — direct execution is forbidden! "
            "Tools must go through LangGraph ToolNode via Future proxy."
        )
