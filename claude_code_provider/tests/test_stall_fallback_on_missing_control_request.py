"""Regression: a proxy ToolUseBlock whose control_request never arrives must
NOT put the agent into error — it must degrade to a normal tool_call.

Root cause (observed 2026-05-27):
  In a session whose history mixes bare (``grep``) and prefixed
  (``mcp__deepworker__grep``) tool names — e.g. a session originally
  ``--import-claude``'d from a raw ``claude`` transcript and later synthesized —
  the CLI re-emits BOTH forms for one logical call in a single AssistantMessage.
  The CLI then fails to dispatch the MCP ``control_request`` for the legitimate
  (prefixed) block, so the proxy handler never registers a pending call. The old
  code waited 30s, killed the client and raised RuntimeError → agent ``error``.

Principle (per user): an undispatched / duplicate / unrecognized tool block must
NOT hang the agent. We already captured the visible ToolUseBlock (name + args);
on a registration timeout we surface it as the tool_call so LangGraph runs the
tool normally. The agent proceeds instead of erroring.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as langchain_tool

_claude_sdk_path = Path(__file__).parent.parent.parent.parent / "claude_agent_sdk"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage as ClaudeAssistantMessage,
    TextBlock,
    ToolUseBlock,
)

from claude_code_provider import ClaudeCodeChatModel  # noqa: E402


@langchain_tool
def grep(pattern: str) -> str:
    """Search text."""
    return pattern


def _make_model_with_proxy(receive_response: Any, mock_proxy: Any) -> Any:
    """Build a ClaudeCodeChatModel whose client/proxy are mocked."""
    mock_client = MagicMock()
    mock_client.receive_response = receive_response
    mock_client.query = AsyncMock()

    cm = patch("claude_code_provider.chat_model.get_global_client_manager")
    mock_get_manager = cm.start()
    mock_manager = MagicMock()
    mock_manager.get_or_create_client = AsyncMock(
        return_value=(mock_client, mock_proxy)
    )
    mock_manager.get_sent_count.return_value = 0
    mock_manager.update_sent_count = MagicMock()
    mock_manager.cleanup_client = AsyncMock()
    mock_manager.record_session_uuid = MagicMock()
    mock_get_manager.return_value = mock_manager

    model = ClaudeCodeChatModel(model="opus").bind_tools([grep])
    return model, mock_manager, cm


@pytest.mark.asyncio
async def test_double_block_missing_control_request_degrades_to_tool_call() -> None:
    """One AssistantMessage with BOTH bare ``grep`` and ``mcp__deepworker__grep``;
    the proxy never registers a pending call (control_request never dispatched).

    Expected (post-fix): ainvoke returns an AIMessage carrying the visible
    ``grep`` tool_call instead of raising — the agent proceeds.
    Pre-fix this raises RuntimeError after a 30s wait.
    """

    async def mock_receive_response() -> Any:
        # The exact shape observed live: text, then a SINGLE AssistantMessage
        # whose content has the duplicate (bare + prefixed) proxy blocks.
        yield ClaudeAssistantMessage(
            content=[TextBlock(text="Let me search.")],
            model="claude-opus-4",
        )
        yield ClaudeAssistantMessage(
            content=[
                ToolUseBlock(
                    id="call_bare",
                    name="grep",
                    input={"pattern": "unsubscribe"},
                ),
                ToolUseBlock(
                    id="call_pref",
                    name="mcp__deepworker__grep",
                    input={"pattern": "unsubscribe"},
                ),
            ],
            model="claude-opus-4",
        )

    mock_proxy = MagicMock()
    # The handler NEVER registers — simulate the undispatched control_request.
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []

    async def _never(*_a: Any, **_k: Any) -> None:
        raise asyncio.TimeoutError()

    mock_proxy.wait_for_any_pending_call = AsyncMock(side_effect=_never)

    model, mock_manager, cm = _make_model_with_proxy(mock_receive_response, mock_proxy)
    try:
        result = await model.ainvoke(
            [HumanMessage(content="search please")],
            config={"configurable": {"thread_id": "t-stall"}},
        )
    finally:
        cm.stop()

    # The agent must NOT error: it returns the visible tool_call so LangGraph
    # can execute it normally.
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "grep"
    assert result.tool_calls[0]["args"] == {"pattern": "unsubscribe"}
    # And it must NOT have killed the client (no error-recovery path).
    mock_manager.cleanup_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_pending_call_still_takes_priority() -> None:
    """When the proxy DOES register a pending call, that authoritative call is
    used (not the visible-block fallback). Guards against regressing the happy
    path / arg-divergence handling."""

    async def mock_receive_response() -> Any:
        yield ClaudeAssistantMessage(
            content=[
                ToolUseBlock(
                    id="call_vis",
                    name="grep",
                    input={"pattern": "VISIBLE_arg"},
                )
            ],
            model="claude-opus-4",
        )

    mock_proxy = MagicMock()
    mock_proxy.has_pending_calls.return_value = True
    # Authoritative control-channel args differ from the visible block.
    mock_proxy.get_pending_calls.return_value = [
        {"id": "grep_1", "name": "grep", "args": {"pattern": "CONTROL_arg"}}
    ]
    mock_proxy.wait_for_any_pending_call = AsyncMock(return_value=None)

    model, _mgr, cm = _make_model_with_proxy(mock_receive_response, mock_proxy)
    try:
        result = await model.ainvoke(
            [HumanMessage(content="search please")],
            config={"configurable": {"thread_id": "t-ok"}},
        )
    finally:
        cm.stop()

    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1
    # The authoritative (control-channel) args win, not the visible fallback.
    assert result.tool_calls[0]["args"] == {"pattern": "CONTROL_arg"}
