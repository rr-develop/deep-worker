"""Safeguards against waiting forever for a tool call that will never
route through our MCP control channel.

Two layers (defence in depth):

1. We never enter the wait for a tool the model named that is NOT one of
   our registered proxy tools (covered structurally by the prefix-only
   classification in chat_model — see
   ``test_event_based_tool_detection.py::TestBareNameNotMisclassifiedAsProxy``).

2. If a wait was somehow started and Claude reports
   ``No such tool available: <name>`` for it, the proxy must be able to
   CANCEL that waiter so the turn does not park forever. This module
   tests the proxy primitive ``cancel_pending_waiter`` and the helper
   ``parse_no_such_tool`` used by chat_model to detect the message.

See docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)

from claude_code_provider.custom_types import ToolMetadata
from claude_code_provider.proxy_mcp_server import (
    ProxyMCPServer,
    parse_no_such_tool,
)

pytestmark = pytest.mark.asyncio

_META: list[ToolMetadata] = [
    {
        "name": "read_file",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }
]


# ── parse_no_such_tool: detect the CLI's rejection message ──────────────


def test_parse_no_such_tool_extracts_name() -> None:
    assert (
        parse_no_such_tool(
            "<tool_use_error>Error: No such tool available: execute"
            "</tool_use_error>"
        )
        == "execute"
    )


def test_parse_no_such_tool_plain_text() -> None:
    assert parse_no_such_tool("Error: No such tool available: foo_bar") == "foo_bar"


def test_parse_no_such_tool_none_when_absent() -> None:
    assert parse_no_such_tool("everything is fine") is None
    assert parse_no_such_tool("") is None


# ── cancel_pending_waiter: break a stuck id-keyed wait ──────────────────


async def test_cancel_pending_waiter_unblocks_waiter() -> None:
    proxy = ProxyMCPServer(_META)
    stuck_id = "toolu_NEVER_ARRIVES"

    async def waiter() -> object:
        return await proxy.wait_for_pending_call_by_id(stuck_id)

    task = asyncio.ensure_future(waiter())
    await asyncio.sleep(0.05)
    assert not task.done(), "waiter must be parked"

    # The CLI said this tool does not exist → cancel our waiter.
    cancelled = proxy.cancel_pending_waiter(stuck_id)
    assert cancelled is True

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)
    # And the waiter entry is gone.
    assert stuck_id not in proxy._pending_id_waiters  # type: ignore[attr-defined]


async def test_cancel_pending_waiter_noop_for_unknown_id() -> None:
    proxy = ProxyMCPServer(_META)
    assert proxy.cancel_pending_waiter("toolu_does_not_exist") is False


async def test_cancel_does_not_touch_a_live_pending_call() -> None:
    """Cancelling a waiter must not disturb an unrelated live pending call
    (a real proxy tool currently executing)."""
    proxy = ProxyMCPServer(_META)
    raw = proxy._create_tool_handler("read_file")  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id("toolu_LIVE")
    try:
        live = asyncio.ensure_future(raw({"path": "/x"}))
        await asyncio.sleep(0.05)
        assert "toolu_LIVE" in proxy.pending_calls
        # cancel an unrelated waiter id
        assert proxy.cancel_pending_waiter("toolu_OTHER") is False
        assert "toolu_LIVE" in proxy.pending_calls
        # resolve the live one normally
        proxy.resolve_call(
            "toolu_LIVE", {"content": [{"type": "text", "text": "ok"}]}
        )
        res = await asyncio.wait_for(live, timeout=2.0)
        assert "ok" in str(res)
    finally:
        reset_current_claude_tool_use_id(token)
