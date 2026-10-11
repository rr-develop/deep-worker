"""``keep=("tokens", 0)`` must yield an EMPTY tail — summary only.

The upstream langchain base class cannot express this: in
``_find_token_based_cutoff`` a non-positive budget is silently coerced
(``if target_token_count <= 0: target_token_count = 1``), and
``_find_safe_cutoff_point`` then walks the boundary back to avoid splitting
an AIMessage(tool_calls) / ToolMessage pair. Both conspire to leave at
least one message in the tail. The override lives in our subclass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.summarization import SummarizationMiddleware

from .test_summarization_middleware import MockBackend, make_mock_model


def _mw(keep) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=make_mock_model(),
        backend=MockBackend(),
        trigger=("messages", 100),
        keep=keep,
    )


def _plain_conversation() -> list:
    return [
        HumanMessage(content="Q1", id="h1"),
        AIMessage(content="A1", id="a1"),
        HumanMessage(content="Q2", id="h2"),
        AIMessage(content="A2", id="a2"),
    ]


def _tool_conversation() -> list:
    """Ends with an AI(tool_calls) -> ToolMessage pair."""
    return [
        HumanMessage(content="Q1", id="h1"),
        AIMessage(content="A1", id="a1"),
        AIMessage(
            content="",
            id="a2",
            tool_calls=[{"id": "call_1", "name": "execute", "args": {}}],
        ),
        ToolMessage(content="result", id="t1", tool_call_id="call_1"),
    ]


class TestKeepZeroEmptiesTheTail:
    def test_plain_conversation_keeps_nothing(self) -> None:
        messages = _plain_conversation()
        cutoff = _mw(("tokens", 0))._determine_cutoff_index(messages)
        assert cutoff == len(messages)

    def test_tool_pair_is_not_rescued_into_the_tail(self) -> None:
        """keep=0 must beat the AI/Tool pair-preservation walk-back."""
        messages = _tool_conversation()
        cutoff = _mw(("tokens", 0))._determine_cutoff_index(messages)
        assert cutoff == len(messages)

    def test_empty_message_list_is_not_summarizable(self) -> None:
        assert _mw(("tokens", 0))._determine_cutoff_index([]) == 0

    def test_partition_yields_empty_preserved(self) -> None:
        messages = _tool_conversation()
        mw = _mw(("tokens", 0))
        cutoff = mw._determine_cutoff_index(messages)
        to_summarize, preserved = mw._partition_messages(messages, cutoff)
        assert preserved == []
        assert len(to_summarize) == len(messages)


class TestKeepZeroEndToEnd:
    """A real summarization run must leave state == [summary] exactly."""

    @pytest.mark.asyncio
    async def test_requested_summary_leaves_summary_only(self) -> None:
        backend = MockBackend()
        model = make_mock_model()
        model.ainvoke = AsyncMock(return_value=MagicMock(text="ZERO-TAIL SUMMARY"))
        mw = SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("messages", 100),
            keep=("tokens", 0),
        )
        messages = _tool_conversation()

        result = await mw.apply_requested_summary("t-zero", messages)

        assert result.summarized is True
        assert result.replacement_messages is not None
        # Exactly one message: the summary. No raw tail whatsoever.
        assert len(result.replacement_messages) == 1
        only = result.replacement_messages[0]
        assert only.additional_kwargs["lc_source"] == "summarization"
        assert "ZERO-TAIL SUMMARY" in only.content
        # The entire history — including the trailing tool pair — was offloaded.
        assert backend.write_calls, "history must still be offloaded"

    @pytest.mark.asyncio
    async def test_summary_only_state_is_a_valid_next_turn_input(self) -> None:
        """``[summary]`` must be a legal prompt for the following turn.

        The summary is a HumanMessage; a state ending on it is exactly what a
        fresh conversation looks like, so both ACP (new session) and stateless
        API providers accept it. The regression this guards against is a tail
        that ends on an orphaned ToolMessage — every provider rejects a tool
        result with no preceding tool call.
        """
        backend = MockBackend()
        model = make_mock_model()
        model.ainvoke = AsyncMock(return_value=MagicMock(text="S"))
        mw = SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("messages", 100),
            keep=("tokens", 0),
        )

        result = await mw.apply_requested_summary("t-zero", _tool_conversation())

        assert result.replacement_messages is not None
        assert isinstance(result.replacement_messages[-1], HumanMessage)
        assert not any(
            isinstance(m, ToolMessage) for m in result.replacement_messages
        ), "an orphaned ToolMessage would be rejected by every provider"


class TestNonZeroKeepIsUnchanged:
    """The override must not perturb any other retention policy."""

    def test_token_budget_still_keeps_a_tail(self) -> None:
        messages = _plain_conversation()
        cutoff = _mw(("tokens", 1_000_000))._determine_cutoff_index(messages)
        # Everything fits in the budget -> nothing to summarize.
        assert cutoff == 0

    def test_message_policy_still_preserves_pairs(self) -> None:
        messages = _tool_conversation()
        cutoff = _mw(("messages", 1))._determine_cutoff_index(messages)
        # Cutting at len-1 would orphan the ToolMessage; the base class walks
        # back to include its AIMessage.
        assert cutoff == 2
        _, preserved = _mw(("messages", 1))._partition_messages(messages, cutoff)
        assert isinstance(preserved[0], AIMessage)
        assert preserved[0].tool_calls
        assert isinstance(preserved[-1], ToolMessage)
