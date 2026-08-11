"""Regression: ainvoke must NOT drop the final TextBlock when a turn both
sends a new HumanMessage AND resolves a prior tool call.

Incident (observed live, 2026-05-21): the TaskManager manager-agent ran a
verify react-loop. On the turn that produced its STATUS/REASONING/MESSAGE
verdict, ``ainvoke`` was driven with ``new_messages`` containing BOTH a
resolved ``ToolMessage`` (from the previous react step) AND a fresh
``HumanMessage``. Claude streamed a single ``AssistantMessage([TextBlock(
verdict)])`` followed by ``ResultMessage`` — but ``ainvoke`` returned
``AIMessage(content='')``. The empty verdict made
``_evaluate_with_manager`` log ``[TaskManager] Evaluation failed`` and
silently break the loop, losing the subordinate's valid answer.

The log proved the TextBlock arrived (the ``Received TextBlock`` debug
line fired) yet content came back empty. These tests reproduce that exact
message shape against the real ``ainvoke`` and assert the verdict text is
preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_claude_sdk_path = Path(__file__).parent.parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (
    AssistantMessage as ClaudeAssistantMessage,
    ResultMessage,
    TextBlock,
)

from deep_worker.providers.claude_code_provider import ClaudeCodeChatModel


def _result_message() -> ResultMessage:
    return ResultMessage(
        subtype="result",
        duration_ms=10,
        duration_api_ms=5,
        is_error=False,
        num_turns=1,
        session_id="test-thread",
        total_cost_usd=0.0,
        usage={"input_tokens": 1, "output_tokens": 1},
    )


@pytest.mark.asyncio
async def test_text_preserved_when_human_message_and_tool_resolved_same_turn() -> None:
    """The observed repro: a turn that resolves a prior tool AND sends a new
    HumanMessage, then receives a single verdict TextBlock, must return that
    text — not an empty AIMessage."""
    mock_client = MagicMock()
    mock_proxy = MagicMock()
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []
    mock_proxy.wait_for_any_pending_call = AsyncMock(return_value=None)

    verdict = (
        "STATUS: done\n"
        "REASONING: I have verified the key technical claims.\n"
        "MESSAGE: NONE"
    )

    async def mock_receive_response():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=verdict)],
            model="claude-opus-4",
        )
        yield _result_message()

    mock_client.receive_response = mock_receive_response
    mock_client.query = AsyncMock()

    with patch(
        "claude_code_provider.chat_model.get_global_client_manager"
    ) as mock_get_manager:
        mock_manager = MagicMock()
        mock_manager.get_or_create_client = AsyncMock(
            return_value=(mock_client, mock_proxy)
        )
        # Two messages already sent in prior turns; the two new ones below
        # are the just-resolved ToolMessage and the fresh HumanMessage.
        mock_manager.get_sent_count.return_value = 2
        mock_manager.get_effective_session_id.return_value = "test-thread"
        mock_manager.update_sent_count = MagicMock()
        mock_get_manager.return_value = mock_manager

        model = ClaudeCodeChatModel(model="opus")
        mock_client.query = AsyncMock()

        # The exact production shape: prior context (2 msgs) + a resolved
        # ToolMessage + a new HumanMessage. tool_resolved=True AND
        # user_message_sent=True in the same ainvoke.
        messages = [
            HumanMessage(content="earlier turn 1"),
            AIMessage(content="earlier turn 2"),
            ToolMessage(content="grep result", tool_call_id="grep_13"),
            HumanMessage(content="Now evaluate and return the verdict."),
        ]

        result = await model.ainvoke(
            messages,
            config={"configurable": {"thread_id": "test-thread"}},
        )

    assert isinstance(result, AIMessage)
    assert result.content == verdict, (
        "ainvoke dropped the verdict TextBlock on a mixed query+resolve turn "
        "(observed live, 2026-05-21): the manager's STATUS/REASONING/MESSAGE was "
        "lost, making the evaluation silently fail."
    )


@pytest.mark.asyncio
async def test_text_preserved_when_only_tool_resolved() -> None:
    """A turn that resolves a tool (no new HumanMessage) and receives a
    final TextBlock must also preserve it. This is the second observed
    failure shape (user_message_sent=False, tool_resolved=True)."""
    mock_client = MagicMock()
    mock_proxy = MagicMock()
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []
    mock_proxy.wait_for_any_pending_call = AsyncMock(return_value=None)

    verdict = (
        "STATUS: in_progress\n"
        "REASONING: The test counts add up but coverage is partial.\n"
        "MESSAGE: Please run the full backend suite and report."
    )

    async def mock_receive_response():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=verdict)],
            model="claude-opus-4",
        )
        yield _result_message()

    mock_client.receive_response = mock_receive_response
    mock_client.query = AsyncMock()

    with patch(
        "claude_code_provider.chat_model.get_global_client_manager"
    ) as mock_get_manager:
        mock_manager = MagicMock()
        mock_manager.get_or_create_client = AsyncMock(
            return_value=(mock_client, mock_proxy)
        )
        mock_manager.get_sent_count.return_value = 1
        mock_manager.get_effective_session_id.return_value = "test-thread"
        mock_manager.update_sent_count = MagicMock()
        mock_get_manager.return_value = mock_manager

        model = ClaudeCodeChatModel(model="opus")

        messages = [
            HumanMessage(content="earlier turn"),
            ToolMessage(content="grep result", tool_call_id="grep_19"),
        ]

        result = await model.ainvoke(
            messages,
            config={"configurable": {"thread_id": "test-thread"}},
        )

    assert isinstance(result, AIMessage)
    assert result.content == verdict, (
        "ainvoke dropped the verdict TextBlock on a tool-resolve-only turn "
        "(observed live, 2026-05-21 second failure)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
