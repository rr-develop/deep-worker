"""A turn's reply must carry THIS turn's text, not the branch's history.

Defect (agent_a, TaskManager branch ``agent_a:tm``, measured on the live
checkpoint DB 2026-10-11):

``_consume_turn`` parks its frame-local accumulators on ``state.queued_text``
right before ``response_fut.result()`` so a raising ``.result()`` cannot
destroy the turn's work (``_preserve_turn_work``, added by ``dffa000``). On the
raising path the next ainvoke is supposed to pick that text up and prepend it.

But the SUCCESSFUL path returns through ``_finalize_response`` without ever
clearing what it just parked. ``state`` is per-``agent_thread_id`` and outlives
the turn, so the next turn in the same branch starts with
``_BoundedTextAcc(state.queued_text)`` and inherits the previous turn's text —
then parks the sum, and so on. The reply grows by one whole turn per turn.

Measured on ``agent_a:tm`` (``threads.db``, read-only): one AIMessage per
evaluation carrying 23, 24, 25, 26, then 27 ``STATUS:``/``REASONING:``/
``MESSAGE:`` blocks at lengths 36086 → 40871 chars, each seam glued mid-line
(``...MESSAGE: NONESTATUS: done``) because the parts are concatenated with no
separator. The TaskManager then re-read its own earlier verdicts as part of the
new one, which is what made it loop.

Why the TM branch and not the main dialog: a turn that ends in a ``tool_call``
returns at the mid-turn pause BEFORE the preserve step, so nothing is parked.
A text-only turn — every TaskManager verdict — goes through the preserve step.
Verified by execution both ways; the main dialog's checkpoints show 0 such
messages against 27 in the TM branch (same scan, positive control).

These tests use the REAL ``ThreadState`` and call the REAL ``_consume_turn``;
only the collaborators that would need a live CLI subprocess are stubbed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.custom_types import (
    NOTIF_QUEUE_MAX_ITEMS,
    ThreadState,
)
from deep_worker.providers.acp.profile import AcpProfile

SESSION = "d6ddeed4-900e-408a-8e37-a54743d5ee3e"
THREAD = "agent_1:tm"


def _model() -> AcpChatModel:
    """``AcpChatModel`` without constructor side effects."""
    model = object.__new__(AcpChatModel)
    model._acp_client = MagicMock()
    model._acp_client.session_cancel = AsyncMock()
    model._profile = AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
    )
    model._tools_metadata = []
    model._resolved_model_name = "test-model"
    model._usage_event_callback = None
    model._proxy_mcp = MagicMock()
    model._proxy_mcp.server_name = "deep_worker_proxy"
    model._start_pump = lambda _state, _tid: None  # type: ignore[method-assign]
    model._correlate_proxy_tool_calls = AsyncMock()  # type: ignore[method-assign]
    model._start_live_context_watch = AsyncMock()  # type: ignore[method-assign]
    model._stop_live_context_watch = AsyncMock()  # type: ignore[method-assign]
    model._maybe_report_model = AsyncMock()  # type: ignore[method-assign]
    model._emit_usage_event = AsyncMock()  # type: ignore[method-assign]
    return model


def _branch_state() -> ThreadState:
    """One state per ``agent_thread_id`` — it outlives every turn."""
    state = ThreadState()
    state.session_id = SESSION
    state.session_live_in_process = True
    return state


def _text_notif(text: str) -> dict:
    return {
        "method": "session/update",
        "params": {
            "sessionId": SESSION,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


def _tool_notif(tcid: str) -> dict:
    return {
        "method": "session/update",
        "params": {
            "sessionId": SESSION,
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": tcid,
                "title": "execute",
                "rawInput": {"command": "echo hi"},
            },
        },
    }


async def _text_only_turn(
    model: AcpChatModel, state: ThreadState, text: str
) -> object:
    """One complete, SUCCESSFUL text-only turn — the TaskManager's shape.

    A fresh queue and future per turn is what ``_agenerate`` does for every
    ``session/prompt``; the STATE is deliberately reused, because that is the
    thing which is per-branch and persistent.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    state.current_notif_queue = queue
    state.pending_session_prompt_future = fut
    queue.put_nowait(_text_notif(text))
    fut.set_result({"id": 1, "result": {"stopReason": "end_turn"}})
    return await model._consume_turn(state, THREAD)


class TestTurnTextIsNotCarriedOver:
    """The core contract: turn N's content is turn N's text."""

    @pytest.mark.asyncio
    async def test_second_turn_does_not_contain_the_first(self) -> None:
        model = _model()
        state = _branch_state()

        first = await _text_only_turn(model, state, "FIRST-TURN-TEXT")
        second = await _text_only_turn(model, state, "SECOND-TURN-TEXT")

        assert "FIRST-TURN-TEXT" in first.content
        assert "SECOND-TURN-TEXT" in second.content
        assert "FIRST-TURN-TEXT" not in second.content, (
            "the second turn's reply carried the first turn's text — the "
            "branch state kept what _preserve_turn_work parked and the next "
            "turn prepended it (agent_1:tm grew one verdict per evaluation)"
        )

    @pytest.mark.asyncio
    async def test_reply_length_does_not_grow_across_turns(self) -> None:
        """The measured signature: same-size turns, monotonically growing reply."""
        model = _model()
        state = _branch_state()

        lengths = [
            len((await _text_only_turn(model, state, f"verdict-{i} ")).content)
            for i in range(5)
        ]

        assert len(set(lengths)) == 1, (
            f"five equally sized turns produced growing replies {lengths} — "
            f"each turn is re-emitting the ones before it"
        )

    @pytest.mark.asyncio
    async def test_branch_state_is_clean_after_a_successful_turn(self) -> None:
        """Nothing may be left parked once the turn has been delivered.

        This is the mechanism itself rather than its symptom: residue on the
        state is what the NEXT turn picks up.
        """
        model = _model()
        state = _branch_state()

        await _text_only_turn(model, state, "delivered answer")

        assert list(state.queued_text) == [], (
            "the successful path parked the turn's text and returned without "
            "clearing it, so the next turn in this branch inherits it"
        )
        assert list(state.queued_tool_calls) == []

    @pytest.mark.asyncio
    async def test_many_turns_stay_independent(self) -> None:
        """27 turns is what the live TM branch had accumulated."""
        model = _model()
        state = _branch_state()

        contents = [
            (await _text_only_turn(model, state, f"turn-{i} ")).content
            for i in range(10)
        ]

        for i, content in enumerate(contents):
            others = [f"turn-{j} " for j in range(10) if j != i]
            leaked = [o for o in others if o in content]
            assert not leaked, (
                f"turn {i} leaked text from turns {leaked}"
            )


class TestPreservationStillWorks:
    """The fix must not re-open the defect ``_preserve_turn_work`` closed.

    Without these, "clear the residue" could be implemented by simply not
    preserving anything, which would silently restore the failure mode where a
    harness error destroys a completed turn's output.
    """

    @pytest.mark.asyncio
    async def test_work_still_survives_a_failing_result(self) -> None:
        from deep_worker.providers.acp.acp_client import AcpSinkOverflowError

        model = _model()
        state = _branch_state()
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state.current_notif_queue = queue
        state.pending_session_prompt_future = fut
        for part in ("alpha ", "beta ", "gamma"):
            queue.put_nowait(_text_notif(part))
        fut.set_exception(AcpSinkOverflowError("overflow (capacity 1000)"))

        with pytest.raises(AcpSinkOverflowError):
            await model._consume_turn(state, THREAD)

        assert list(state.queued_text) == ["alpha ", "beta ", "gamma"], (
            "a turn killed by a harness error must still leave its work "
            "recoverable on the state — that is why the parking exists"
        )

    @pytest.mark.asyncio
    async def test_midturn_pause_still_hands_text_to_the_next_ainvoke(
        self,
    ) -> None:
        """A mid-turn tool pause is NOT a turn boundary.

        The text streamed before the pause belongs to the SAME turn, so it must
        still reach the caller — here via the returned AIMessage, which is what
        LangGraph appends before running the tool.
        """
        model = _model()
        state = _branch_state()
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state.current_notif_queue = queue
        state.pending_session_prompt_future = fut
        queue.put_nowait(_text_notif("thinking out loud "))
        queue.put_nowait(_tool_notif("call_1"))

        message = await model._consume_turn(state, THREAD)

        assert message.tool_calls, "the tool_call must surface"
        assert "thinking out loud " in message.content, (
            "text streamed before a mid-turn tool pause is part of this turn"
        )

    @pytest.mark.asyncio
    async def test_pump_text_still_reaches_the_resumed_turn(self) -> None:
        """Text the pump accumulated during a tool run must not be dropped.

        This is the legitimate consumer of ``state.queued_text``: it is still
        the SAME turn, resumed. A fix that cleared the buffer indiscriminately
        at entry would lose it.
        """
        model = _model()
        state = _branch_state()
        state.queued_text = ["streamed while the tool ran "]

        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state.current_notif_queue = queue
        state.pending_session_prompt_future = fut
        queue.put_nowait(_text_notif("and the conclusion"))
        fut.set_result({"id": 1, "result": {"stopReason": "end_turn"}})

        message = await model._consume_turn(state, THREAD)

        assert message.content == (
            "streamed while the tool ran and the conclusion"
        ), (
            "the resumed turn must still prepend what the pump accumulated "
            "for THIS turn"
        )
