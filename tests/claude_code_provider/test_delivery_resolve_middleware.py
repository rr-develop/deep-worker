"""TDD for the claude-code-only delivery-resolve middleware (Bug 2).

DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md: this middleware uses the
``awrap_tool_call`` seam — which runs INSIDE the tools node, before control
returns to the model node (and thus before SendMessageCompletionMiddleware
jumps to end). On a successful ``send_message_to_agent`` tool result it
resolves the proxy-MCP Future at DELIVERY time via the model's public
``resolve_pending_tool(thread_id, tool_call_id, content)``, so the Claude CLI
gets its tool_result immediately instead of timing out while the recipient
works. On a failed delivery it does NOT resolve (the model must still see the
error). The timeout on the proxy handler then bounds delivery only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from deep_worker.providers.claude_code_provider.delivery_resolve_middleware import (
    InterAgentDeliveryResolveMiddleware,
)


# --- minimal stand-ins for the langgraph tool-call objects ---------------- #


@dataclass
class _FakeRuntime:
    config: dict[str, Any]


@dataclass
class _FakeRequest:
    tool_call: dict[str, Any]
    tool: Any
    state: Any
    runtime: _FakeRuntime


class _RecordingModel:
    """Captures resolve_pending_tool calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    def resolve_pending_tool(
        self, thread_id: str, tool_call_id: str, content: Any
    ) -> None:
        self.calls.append((thread_id, tool_call_id, content))


def _make_request(
    *, name: str, call_id: str, thread_id: str
) -> _FakeRequest:
    return _FakeRequest(
        tool_call={"name": name, "id": call_id, "args": {}},
        tool=None,
        state={"messages": []},
        runtime=_FakeRuntime(config={"configurable": {"thread_id": thread_id}}),
    )


@pytest.mark.asyncio
async def test_resolves_on_successful_send() -> None:
    """A successful send_message_to_agent tool result resolves the Future
    immediately (before the turn ends)."""
    model = _RecordingModel()
    mw = InterAgentDeliveryResolveMiddleware(model)
    req = _make_request(
        name="send_message_to_agent", call_id="call_1", thread_id="agentA"
    )

    delivered = json.dumps(
        {"success": True, "target_agent": "beta", "delivered_content": "alpha -> beta :> hi"}
    )

    async def handler(r: _FakeRequest) -> ToolMessage:
        return ToolMessage(content=delivered, tool_call_id="call_1",
                           name="send_message_to_agent")

    result = await mw.awrap_tool_call(req, handler)

    # Resolved once, for the right thread + call id, and the result was
    # returned to the graph unchanged.
    assert len(model.calls) == 1
    tid, cid, _content = model.calls[0]
    assert tid == "agentA"
    assert cid == "call_1"
    assert isinstance(result, ToolMessage)
    assert result.content == delivered


@pytest.mark.asyncio
async def test_does_not_resolve_on_failed_send() -> None:
    """A failed delivery must NOT be resolved early — the model has to see
    the error and react."""
    model = _RecordingModel()
    mw = InterAgentDeliveryResolveMiddleware(model)
    req = _make_request(
        name="send_message_to_agent", call_id="call_2", thread_id="agentA"
    )

    failed = json.dumps({"success": False, "error": "agent_not_found"})

    async def handler(r: _FakeRequest) -> ToolMessage:
        return ToolMessage(content=failed, tool_call_id="call_2",
                           name="send_message_to_agent")

    result = await mw.awrap_tool_call(req, handler)

    assert model.calls == []
    assert result.content == failed


@pytest.mark.asyncio
async def test_ignores_other_tools() -> None:
    """Non-send tools are passed through untouched (no resolve)."""
    model = _RecordingModel()
    mw = InterAgentDeliveryResolveMiddleware(model)
    req = _make_request(name="read_file", call_id="call_3", thread_id="agentA")

    async def handler(r: _FakeRequest) -> ToolMessage:
        return ToolMessage(content="file contents", tool_call_id="call_3",
                           name="read_file")

    result = await mw.awrap_tool_call(req, handler)

    assert model.calls == []
    assert result.content == "file contents"


@pytest.mark.asyncio
async def test_unparsable_result_does_not_resolve() -> None:
    """If the tool result can't be parsed as a success dict, don't resolve
    (be conservative; the normal late path still works)."""
    model = _RecordingModel()
    mw = InterAgentDeliveryResolveMiddleware(model)
    req = _make_request(
        name="send_message_to_agent", call_id="call_4", thread_id="agentA"
    )

    async def handler(r: _FakeRequest) -> ToolMessage:
        return ToolMessage(content="not json at all", tool_call_id="call_4",
                           name="send_message_to_agent")

    result = await mw.awrap_tool_call(req, handler)
    assert model.calls == []
    assert result.content == "not json at all"


@pytest.mark.asyncio
async def test_resolve_failure_does_not_break_tool_turn() -> None:
    """If resolve_pending_tool raises, the tool result is still returned."""

    class _BoomModel:
        def resolve_pending_tool(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("proxy gone")

    mw = InterAgentDeliveryResolveMiddleware(_BoomModel())
    req = _make_request(
        name="send_message_to_agent", call_id="call_5", thread_id="agentA"
    )
    delivered = json.dumps({"success": True, "delivered_content": "x"})

    async def handler(r: _FakeRequest) -> ToolMessage:
        return ToolMessage(content=delivered, tool_call_id="call_5",
                           name="send_message_to_agent")

    result = await mw.awrap_tool_call(req, handler)
    assert result.content == delivered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
