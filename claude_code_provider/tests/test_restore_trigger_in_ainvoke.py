"""TDD: ainvoke triggers history-restore on a genuine RESTART, and ONLY then.

DESIGN_HISTORY_RESTORE_ON_RESTART.md §3.3. The restore path
(``resolve_restore_session`` → ``get_or_create_client(resume_session_id=...)``)
must fire when:
  * the live CLI client does not exist yet (first turn after (re)start), AND
  * the history carries prior turns (an AIMessage/ToolMessage), AND
  * persisted session state exists (proof the agent ran before).

It must NOT fire for a brand-new agent's first turn, an in-process live tool
round-trip, or anything without persisted state — those run the normal send
loop. The autouse conftest fixture isolates HOME, so "persisted state" here is
created explicitly via save_session_state.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from claude_code_provider.chat_model import ClaudeCodeChatModel
from claude_code_provider.session_state import load_session_state, save_session_state


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
    return c, proxy


_RESTART_HISTORY = [
    HumanMessage(content="do the task", id="h1"),
    AIMessage(content="working", id="a1",
              tool_calls=[{"id": "c1", "name": "x", "args": {}}]),
    ToolMessage(content="result", tool_call_id="c1", id="t1"),
    HumanMessage(content="continue", id="h2"),
]


@pytest.mark.asyncio
async def test_restore_fires_on_restart_with_persisted_state() -> None:
    """Persisted state + prior-turn history + no live client → resume."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "restart_agent"
    # Simulate a prior run: state was persisted, then the process restarted
    # (the global ClientManager has no live client for tid — conftest cleared
    # it).
    save_session_state(tid, sent_count=3, session_uuid="prior-uuid")

    mock_client, mock_proxy = _mock_client()
    captured: dict[str, Any] = {}

    async def fake_get(**kwargs: Any) -> tuple[Mock, Mock]:
        captured.update(kwargs)
        return (mock_client, mock_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        with patch(
            "claude_code_provider.session_restore.resolve_restore_session",
            return_value="resumed-uuid",
        ) as mock_resolve:
            await model.ainvoke(
                _RESTART_HISTORY, config={"configurable": {"thread_id": tid}}
            )

    mock_resolve.assert_called_once()
    assert captured.get("resume_session_id") == "resumed-uuid", (
        "on a genuine restart, ainvoke must pass the resolved resume session "
        "id to get_or_create_client"
    )
    assert mock_resolve.call_args is not None
    assert mock_resolve.call_args.args[1] == _RESTART_HISTORY[:-1]
    mock_client.query.assert_awaited_once_with("continue", session_id=tid)
    state = load_session_state(tid)
    assert state is not None
    assert state["sent_count"] == len(_RESTART_HISTORY)


@pytest.mark.asyncio
async def test_restore_skipped_without_persisted_state() -> None:
    """No persisted state (brand-new agent or test) → no restore, even with a
    prior-turn-shaped history."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "fresh_agent"  # nothing persisted (conftest isolates HOME)

    mock_client, mock_proxy = _mock_client()
    captured: dict[str, Any] = {}

    async def fake_get(**kwargs: Any) -> tuple[Mock, Mock]:
        captured.update(kwargs)
        return (mock_client, mock_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        with patch(
            "claude_code_provider.session_restore.resolve_restore_session",
            return_value="should-not-be-used",
        ) as mock_resolve:
            await model.ainvoke(
                _RESTART_HISTORY, config={"configurable": {"thread_id": tid}}
            )

    mock_resolve.assert_not_called()
    assert captured.get("resume_session_id") is None


@pytest.mark.asyncio
async def test_restore_skipped_for_first_user_message_only() -> None:
    """A brand-new conversation (single HumanMessage) never restores, even if
    state somehow existed — there are no prior turns to carry over."""
    model = ClaudeCodeChatModel(base_options=None).bind_tools([])
    tid = "firstturn_agent"
    save_session_state(tid, sent_count=0, session_uuid=None)

    mock_client, mock_proxy = _mock_client()
    captured: dict[str, Any] = {}

    async def fake_get(**kwargs: Any) -> tuple[Mock, Mock]:
        captured.update(kwargs)
        return (mock_client, mock_proxy)

    with patch.object(model.client_manager, "get_or_create_client", side_effect=fake_get):
        with patch(
            "claude_code_provider.session_restore.resolve_restore_session",
            return_value="x",
        ) as mock_resolve:
            await model.ainvoke(
                [HumanMessage(content="hi", id="h1")],
                config={"configurable": {"thread_id": tid}},
            )

    mock_resolve.assert_not_called()
    assert captured.get("resume_session_id") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
