"""TDD for ``ClaudeCodeChatModel.resolve_pending_tool`` (Bug 2).

DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md: the inter-agent send Future must be
resolvable at DELIVERY time (right after the tool ran), not deferred to the
next ainvoke. The model exposes a small public, idempotent
``resolve_pending_tool(thread_id, tool_call_id, content)`` that the
claude-code-only delivery-resolve middleware calls, so proxy-MCP knowledge
stays inside the provider and the resolve is a supported API rather than a
reach into privates.
"""

from __future__ import annotations

import asyncio

import pytest

from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)
from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
from deep_worker.providers.claude_code_provider.proxy_mcp_server import ProxyMCPServer
from deep_worker.providers.claude_code_provider.custom_types import MCPResult, ToolMetadata


_TOOL_META: list[ToolMetadata] = [
    {
        "name": "send_message_to_agent",
        "description": "send a message to another agent",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["agent_name", "message"],
        },
    }
]


async def _register_pending(proxy: ProxyMCPServer) -> str:
    """Drive the handler so a pending call exists; return its call_id.

    Incident 2026-06-14: the handler MUST be invoked with a Claude
    tool_use_id in the SDK ContextVar — the synthetic-id fallback is
    gone. ``set_current_claude_tool_use_id`` is set BEFORE
    ``create_task`` so the task's context snapshot captures the id.
    """
    cid = "toolu_resolve_pending"
    token = set_current_claude_tool_use_id(cid)
    try:
        asyncio.create_task(
            proxy.tool_handlers[0].handler({"agent_name": "beta", "message": "hi"})
        )
    finally:
        reset_current_claude_tool_use_id(token)
    for _ in range(20):
        await asyncio.sleep(0)
        if proxy.has_pending_calls():
            break
    assert proxy.has_pending_calls()
    return proxy.get_pending_calls()[0]["id"]


@pytest.mark.asyncio
async def test_resolve_pending_tool_resolves_matching_future() -> None:
    """resolve_pending_tool resolves the proxy Future for that thread+id."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "agentA"
    proxy = ProxyMCPServer(_TOOL_META)
    # Inject a live proxy for this thread (no real Claude CLI).
    model.client_manager.clients[tid] = (object(), proxy)

    call_id = await _register_pending(proxy)

    result: MCPResult = {"content": [{"type": "text", "text": "delivered"}]}
    model.resolve_pending_tool(tid, call_id, result)

    # Future resolved → the parked handler returns the delivered result.
    fut = proxy.pending_calls.get(call_id)
    # Either already popped (handler finished) or the future is done.
    if fut is not None:
        assert fut["future"].done()


@pytest.mark.asyncio
async def test_resolve_pending_tool_unknown_thread_is_noop() -> None:
    """Unknown thread_id must be a safe no-op (no raise)."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    # No client registered for this thread.
    model.resolve_pending_tool(
        "ghost", "whatever", {"content": [{"type": "text", "text": "x"}]}
    )


@pytest.mark.asyncio
async def test_resolve_pending_tool_unknown_call_id_is_noop() -> None:
    """Known thread but unknown call_id must be a safe no-op."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "agentB"
    proxy = ProxyMCPServer(_TOOL_META)
    model.client_manager.clients[tid] = (object(), proxy)

    model.resolve_pending_tool(
        tid, "no-such-id", {"content": [{"type": "text", "text": "x"}]}
    )
    assert not proxy.has_pending_calls()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
