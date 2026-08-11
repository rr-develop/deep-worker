"""Orphan reconciliation on the proxy: a tool call orphaned by a restart
must NEVER park the handler or make the per-id waiter wait.

When the agent is restarted mid-tool-call, the resumed CLI session may
replay a ``tool_use`` whose id never gets a matching control-channel
result (or gets one under a different id). Both
``_create_tool_handler``'s ``await future`` and
``wait_for_pending_call_by_id`` are intentionally timeout-free, so an
orphaned id deadlocks forever.

The fix: ``mark_orphaned_tool_calls(ids)`` tells the proxy these ids were
interrupted by a restart. Thereafter:
  * a handler that registers an orphaned id resolves IMMEDIATELY with an
    "interrupted by restart" result (never parks), and
  * ``wait_for_pending_call_by_id`` for an orphaned id returns
    immediately (never waits).

No timeouts, no polling — structural reconciliation.
See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)

from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata
from deep_worker.providers.claude_code_provider.proxy_mcp_server import ProxyMCPServer

pytestmark = pytest.mark.asyncio


_META: list[ToolMetadata] = [
    {
        "name": "history_memory",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }
]


async def test_handler_for_orphan_resolves_immediately_without_parking() -> None:
    proxy = ProxyMCPServer(_META)
    orphan_id = "toolu_ORPHAN"
    proxy.mark_orphaned_tool_calls([orphan_id])

    raw = proxy._create_tool_handler("history_memory")  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id(orphan_id)
    try:
        # Must complete on its own — no resolve_call, no ToolNode, nothing.
        result = await asyncio.wait_for(raw({"q": "x"}), timeout=2.0)
    finally:
        reset_current_claude_tool_use_id(token)

    text = str(result)
    assert "interrupt" in text.lower() or "restart" in text.lower()
    # The orphaned id must NOT linger as a live pending call.
    assert orphan_id not in proxy.pending_calls


async def test_wait_for_pending_call_by_id_returns_immediately_for_orphan() -> None:
    proxy = ProxyMCPServer(_META)
    orphan_id = "toolu_ORPHAN2"
    proxy.mark_orphaned_tool_calls([orphan_id])

    # No handler will ever register this id, yet the waiter must not hang.
    result = await asyncio.wait_for(
        proxy.wait_for_pending_call_by_id(orphan_id), timeout=2.0
    )
    assert result["id"] == orphan_id


async def test_non_orphan_id_still_parks_normally() -> None:
    """Regression: a normal (non-orphan) id must keep its event-driven
    park — the fix must not short-circuit healthy calls."""
    proxy = ProxyMCPServer(_META)
    raw = proxy._create_tool_handler("history_memory")  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id("toolu_LIVE")
    try:
        task = asyncio.ensure_future(raw({"q": "x"}))
        # Give it a tick to register + park.
        await asyncio.sleep(0.05)
        assert not task.done(), "healthy handler must park awaiting resolve_call"
        assert "toolu_LIVE" in proxy.pending_calls
        # Resolve it normally.
        proxy.resolve_call(
            "toolu_LIVE", {"content": [{"type": "text", "text": "ok"}]}
        )
        result = await asyncio.wait_for(task, timeout=2.0)
        assert "ok" in str(result)
    finally:
        reset_current_claude_tool_use_id(token)
