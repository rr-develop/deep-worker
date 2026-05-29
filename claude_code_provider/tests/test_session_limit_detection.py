"""Regression: a Claude CLI session/usage-limit response must raise a typed
error, not be returned as normal assistant content.

Incident (observed 2026-05-21): when the Claude account hit its usage cap,
the CLI returned a single degenerate TextBlock —
``You've hit your session limit · resets 5:40pm (Europe/Amsterdam)`` — as
the entire turn. ``ainvoke`` passed it through as ordinary
``AIMessage.content``. Downstream that looked like a real (if useless)
answer: the TaskManager kept correcting it, the agent silently settled to
idle, and the web UI showed no error. 197 such markers accumulated in the
agent log.

Correct behavior: detect the limit response and raise
``ClaudeUsageLimitError`` so the agent process can transition to
status=ERROR with a human-readable message instead of failing silently.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_claude_sdk_path = Path(__file__).parent.parent.parent.parent / "claude_agent_sdk"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import (
    AssistantMessage as ClaudeAssistantMessage,
    ResultMessage,
    TextBlock,
)

from claude_code_provider import ClaudeCodeChatModel
from claude_code_provider.chat_model import ClaudeUsageLimitError


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


def _make_manager(mock_client: MagicMock, mock_proxy: MagicMock) -> MagicMock:
    mock_manager = MagicMock()
    mock_manager.get_or_create_client = AsyncMock(
        return_value=(mock_client, mock_proxy)
    )
    mock_manager.get_sent_count.return_value = 0
    mock_manager.get_effective_session_id.return_value = "test-thread"
    mock_manager.update_sent_count = MagicMock()
    return mock_manager


@pytest.mark.asyncio
async def test_session_limit_text_raises_typed_error() -> None:
    """The exact production limit response must raise ClaudeUsageLimitError."""
    limit_text = "You've hit your session limit · resets 5:40pm (Europe/Amsterdam)"

    mock_client = MagicMock()
    mock_proxy = MagicMock()
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []

    async def mock_receive_response():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=limit_text)],
            model="claude-opus-4",
        )
        yield _result_message()

    mock_client.receive_response = mock_receive_response
    mock_client.query = AsyncMock()

    with patch(
        "claude_code_provider.chat_model.get_global_client_manager"
    ) as mock_get_manager:
        mock_get_manager.return_value = _make_manager(mock_client, mock_proxy)
        model = ClaudeCodeChatModel(model="opus")

        with pytest.raises(ClaudeUsageLimitError) as exc_info:
            await model.ainvoke(
                [HumanMessage(content="do work")],
                config={"configurable": {"thread_id": "test-thread"}},
            )

    # The error must carry the human-readable cause (incl. reset time) so the
    # agent process can put it in error_message for the web UI.
    assert "session limit" in str(exc_info.value).lower()
    assert "5:40pm" in str(exc_info.value)


@pytest.mark.asyncio
async def test_session_limit_resets_11pm_variant_raises() -> None:
    """The other observed reset-time variant must also be detected."""
    limit_text = "You've hit your session limit · resets 11:10pm (Europe/Amsterdam)"

    mock_client = MagicMock()
    mock_proxy = MagicMock()
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []

    async def mock_receive_response():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=limit_text)],
            model="claude-opus-4",
        )
        yield _result_message()

    mock_client.receive_response = mock_receive_response
    mock_client.query = AsyncMock()

    with patch(
        "claude_code_provider.chat_model.get_global_client_manager"
    ) as mock_get_manager:
        mock_get_manager.return_value = _make_manager(mock_client, mock_proxy)
        model = ClaudeCodeChatModel(model="opus")

        with pytest.raises(ClaudeUsageLimitError):
            await model.ainvoke(
                [HumanMessage(content="do work")],
                config={"configurable": {"thread_id": "test-thread"}},
            )


@pytest.mark.asyncio
async def test_normal_text_mentioning_the_phrase_in_context_is_not_a_false_positive() -> None:
    """A legitimate answer that merely discusses session limits must NOT be
    treated as a limit hit. The detector keys on the CLI's exact standalone
    marker, not on the words appearing anywhere in a longer reply."""
    normal_text = (
        "To handle rate limiting, catch the error when the user has hit "
        "their session limit and back off. Here's the implementation:\n"
        "```python\nif resp.status == 429:\n    backoff()\n```"
    )

    mock_client = MagicMock()
    mock_proxy = MagicMock()
    mock_proxy.has_pending_calls.return_value = False
    mock_proxy.get_pending_calls.return_value = []

    async def mock_receive_response():
        yield ClaudeAssistantMessage(
            content=[TextBlock(text=normal_text)],
            model="claude-opus-4",
        )
        yield _result_message()

    mock_client.receive_response = mock_receive_response
    mock_client.query = AsyncMock()

    with patch(
        "claude_code_provider.chat_model.get_global_client_manager"
    ) as mock_get_manager:
        mock_get_manager.return_value = _make_manager(mock_client, mock_proxy)
        model = ClaudeCodeChatModel(model="opus")

        result = await model.ainvoke(
            [HumanMessage(content="how do I handle rate limits?")],
            config={"configurable": {"thread_id": "test-thread"}},
        )

    assert isinstance(result, AIMessage)
    assert result.content == normal_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
