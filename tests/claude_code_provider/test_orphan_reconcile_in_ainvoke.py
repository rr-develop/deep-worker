"""TDD: ainvoke reconciles tool calls orphaned by a restart.

On the first turn after a restart (persisted state + prior-turn history
+ no live client), ainvoke must detect tool calls the model requested
but for which no ToolMessage result exists, and tell the proxy to
reconcile them via ``mark_orphaned_tool_calls`` — so neither the handler
nor ``wait_for_pending_call_by_id`` ever parks on an id whose result will
never come.

See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel
from deep_worker.providers.claude_code_provider.session_state import save_session_state


def _mock_client() -> tuple[Mock, Mock]:
    c = Mock()
    c.query = AsyncMock()

    async def _recv():
        return
        yield
    c.receive_response = _recv
    proxy = Mock()
    proxy.has_pending_calls = Mock(return_value=False)
    proxy.resolve_call = Mock()
    proxy.mark_orphaned_tool_calls = Mock()
    return c, proxy


# History where the LAST turn's tool call was orphaned by a restart:
# AIMessage requested tool "c_orphan" but there is NO ToolMessage for it.
_ORPHAN_HISTORY = [
    HumanMessage(content="do the task", id="h1"),
    AIMessage(content="t1 done", id="a1",
              tool_calls=[{"id": "c_done", "name": "read_file", "args": {}}]),
    ToolMessage(content="result", tool_call_id="c_done", id="t1"),
    HumanMessage(content="next", id="h2"),
    AIMessage(content="calling", id="a2",
              tool_calls=[{"id": "c_orphan", "name": "history_memory", "args": {}}]),
    # process restarted here — no ToolMessage for c_orphan
]

_CLEAN_HISTORY = [
    HumanMessage(content="do the task", id="h1"),
    AIMessage(content="t1 done", id="a1",
              tool_calls=[{"id": "c_done", "name": "read_file", "args": {}}]),
    ToolMessage(content="result", tool_call_id="c_done", id="t1"),
    HumanMessage(content="continue", id="h2"),
]


@pytest.mark.asyncio
async def test_orphaned_tool_call_is_marked_on_restart() -> None:
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "orphan_restart_agent"
    save_session_state(tid, sent_count=5, session_uuid="prior-uuid")

    mock_client, mock_proxy = _mock_client()

    async def fake_get(**kwargs: Any) -> tuple[Mock, Mock]:
        return (mock_client, mock_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        with patch(
            "claude_code_provider.session_restore.resolve_restore_session",
            return_value="resumed-uuid",
        ):
            await model.ainvoke(
                _ORPHAN_HISTORY, config={"configurable": {"thread_id": tid}}
            )

    mock_proxy.mark_orphaned_tool_calls.assert_called_once()
    marked = list(mock_proxy.mark_orphaned_tool_calls.call_args.args[0])
    assert marked == ["c_orphan"], (
        f"only the unresolved tool call must be marked; got {marked}"
    )


@pytest.mark.asyncio
async def test_no_marking_when_history_fully_resolved() -> None:
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "clean_restart_agent"
    save_session_state(tid, sent_count=3, session_uuid="prior-uuid")

    mock_client, mock_proxy = _mock_client()

    async def fake_get(**kwargs: Any) -> tuple[Mock, Mock]:
        return (mock_client, mock_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        with patch(
            "claude_code_provider.session_restore.resolve_restore_session",
            return_value="resumed-uuid",
        ):
            await model.ainvoke(
                _CLEAN_HISTORY, config={"configurable": {"thread_id": tid}}
            )

    mock_proxy.mark_orphaned_tool_calls.assert_not_called()


@pytest.mark.asyncio
async def test_no_marking_without_persisted_state() -> None:
    """A brand-new agent (no persisted state) is not a restart — even with
    an unresolved tool call in history, nothing is marked (restore itself
    does not run)."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "fresh_no_state_agent"  # conftest isolates HOME → no persisted state

    mock_client, mock_proxy = _mock_client()

    async def fake_get(**kwargs: Any) -> tuple[Mock, Mock]:
        return (mock_client, mock_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        await model.ainvoke(
            _ORPHAN_HISTORY, config={"configurable": {"thread_id": tid}}
        )

    mock_proxy.mark_orphaned_tool_calls.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
