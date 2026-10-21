"""A long turn must keep a drainer on the notification sink, and a harness
error must not destroy the work the turn already produced.

Defect (agent_a, session ``d6ddeed4-900e-408a-8e37-a54743d5ee3e``,
2026-10-02 ~10:56Z and ~11:28Z):

``AcpSinkOverflowError: ACP notification sink overflowed ... (capacity 1000)``

The notification queue is created ONCE PER TURN (``chat_model.py`` step 12),
not per ``ainvoke``, so capacity 1000 is a per-TURN budget. While LangGraph's
ToolNode runs a tool, the background pump is what keeps reading it.

``_consume_turn`` has two branches that hand ``tool_calls`` back mid-turn:
  * the in-loop mid-stream pause — always started the pump;
  * the early return that replays tool_calls the PREVIOUS pump had queued —
    did NOT.

On the second one ``_adopt_pump`` had already cancelled the old pump on
re-entry and is a silent no-op when none exists, so a long tool dispatched
through it (``wait_for_job``, 10-20 min) ran with NOBODY reading the sink.
The client then failed the whole turn, and because the accumulators are frame
locals of ``_consume_turn``, the raise discarded every update already drained.

These tests use the REAL ``ThreadState`` and call the REAL ``_consume_turn``;
only the collaborators that would need a live CLI subprocess are stubbed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker.providers.acp.acp_client import AcpSinkOverflowError
from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.custom_types import (
    NOTIF_QUEUE_MAX_ITEMS,
    ThreadState,
)
from deep_worker.providers.acp.profile import AcpProfile


def _model(pump_starts: list[str]) -> AcpChatModel:
    """``AcpChatModel`` without constructor side effects.

    ``_start_pump`` is recorded rather than run: the assertion is about
    whether a drainer is ARMED on this path, and a real pump task would race
    the test's own queue reads. Everything else on the path under test —
    the branch logic, the drain loop, the preservation step — is production
    code.
    """
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
    model._start_pump = lambda _state, tid: pump_starts.append(tid)  # type: ignore[method-assign]
    model._correlate_proxy_tool_calls = AsyncMock()  # type: ignore[method-assign]
    model._start_live_context_watch = AsyncMock()  # type: ignore[method-assign]
    model._stop_live_context_watch = AsyncMock()  # type: ignore[method-assign]
    return model


def _state(queue: asyncio.Queue, fut: asyncio.Future) -> ThreadState:
    state = ThreadState()
    state.session_id = "d6ddeed4-900e-408a-8e37-a54743d5ee3e"
    state.session_live_in_process = True
    state.current_notif_queue = queue
    state.pending_session_prompt_future = fut
    return state


def _text_notif(text: str) -> dict:
    return {
        "method": "session/update",
        "params": {
            "sessionId": "d6ddeed4-900e-408a-8e37-a54743d5ee3e",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


class TestMidTurnResumeArmsADrainer:
    """The replay-queued-tool_calls path must leave a reader on the sink."""

    @pytest.mark.asyncio
    async def test_pump_restarted_when_replaying_queued_tool_calls(self) -> None:
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        # What a previous pump accumulated while ToolNode was busy.
        state.queued_tool_calls = [
            {
                "id": "call_1",
                "name": "wait_for_job",
                "args": {"job_id": "job_1"},
                "type": "tool_call",
            }
        ]

        message = await model._consume_turn(state, "agent_1")

        assert [tc["name"] for tc in message.tool_calls] == ["wait_for_job"]
        # The response future is deliberately kept in flight, which means the
        # sink is STILL registered in the client and the reader keeps routing
        # into this very queue — so a drainer is mandatory, not optional.
        assert state.pending_session_prompt_future is fut
        assert not fut.done()
        assert pump_starts == ["agent_1"], (
            "the mid-turn replay path handed tool_calls to LangGraph while the "
            "sink stayed registered, but armed no drainer — a long tool then "
            "fills the 1000-slot queue and the turn is failed"
        )

    @pytest.mark.asyncio
    async def test_in_loop_pause_still_arms_a_drainer(self) -> None:
        """The branch that was already correct must stay correct."""
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        queue.put_nowait(
            {
                "method": "session/update",
                "params": {
                    "sessionId": state.session_id,
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "call_2",
                        "title": "execute",
                        "rawInput": {"command": "grep -n foo bar.py"},
                    },
                },
            }
        )

        message = await model._consume_turn(state, "agent_1")

        assert message.tool_calls, "expected the tool_call to surface"
        assert pump_starts == ["agent_1"]

    @pytest.mark.asyncio
    async def test_text_only_turn_needs_no_drainer(self) -> None:
        """A text-only turn cancels the session, so nothing more is routed.

        Guards against "arm a pump unconditionally": here the prompt future is
        cleared, the sink is gone, and a pump would spin on a dead queue.
        """
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        state.text_only_turn_active = True
        state.queued_tool_calls = [
            {"id": "c", "name": "execute", "args": {}, "type": "tool_call"}
        ]

        await model._consume_turn(state, "agent_1")

        assert pump_starts == []
        assert state.pending_session_prompt_future is None


class TestOverflowDoesNotDestroyTheWork:
    """A harness error must not take the turn's completed output with it."""

    @pytest.mark.asyncio
    async def test_drained_text_survives_sink_overflow(self) -> None:
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        for part in ("alpha ", "beta ", "gamma"):
            queue.put_nowait(_text_notif(part))
        # Exactly what acp_client does on sink.full().
        fut.set_exception(
            AcpSinkOverflowError(
                f"ACP notification sink overflowed for session "
                f"{state.session_id!r} (capacity {NOTIF_QUEUE_MAX_ITEMS})"
            )
        )

        with pytest.raises(AcpSinkOverflowError):
            await model._consume_turn(state, "agent_1")

        assert state.queued_text == ["alpha ", "beta ", "gamma"], (
            "the updates drained before the raise were frame locals and were "
            "discarded with the frame — the turn's work was destroyed by a "
            "harness error"
        )

    @pytest.mark.asyncio
    async def test_drained_tool_calls_survive_sink_overflow(self) -> None:
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        queue.put_nowait(
            {
                "method": "session/update",
                "params": {
                    "sessionId": state.session_id,
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "call_9",
                        "title": "execute",
                        "rawInput": {"command": "echo hi"},
                    },
                },
            }
        )
        fut.set_exception(AcpSinkOverflowError("overflow (capacity 1000)"))

        with pytest.raises(AcpSinkOverflowError):
            await model._consume_turn(state, "agent_1")

        assert state.queued_tool_calls, (
            "a tool_call drained before the raise must survive on the state"
        )

    @pytest.mark.asyncio
    async def test_overflow_is_still_loud(self) -> None:
        """Preserving work must NOT soften the error into a silent success.

        A silent error is the one failure mode the owner calls a mortal sin:
        the turn must still end with a visible terminal exception.
        """
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        queue.put_nowait(_text_notif("partial answer"))
        fut.set_exception(AcpSinkOverflowError("overflow (capacity 1000)"))

        with pytest.raises(AcpSinkOverflowError, match="overflow"):
            await model._consume_turn(state, "agent_1")

    @pytest.mark.asyncio
    async def test_preserved_text_is_bounded(self) -> None:
        """Preservation reuses the bounded-append caps — no unbounded growth."""
        from deep_worker.providers.acp.custom_types import QUEUED_TEXT_MAX_BYTES

        pump_starts: list[str] = []
        model = _model(pump_starts)
        state = ThreadState()
        state.queued_text = []
        state.queued_tool_calls = []
        # Drive the BYTE cap rather than the 10_000-chunk cap: both run through
        # the same guard, and the byte path needs a handful of chunks instead of
        # ten thousand (``_append_bounded_text`` re-sums the buffer on every
        # append, so the chunk path costs ~2s of pure CPU).
        big = "y" * (QUEUED_TEXT_MAX_BYTES // 4)
        over_cap = [big] * 6

        model._preserve_turn_work(state, over_cap, [])

        assert state.queued_text, "under the cap, work must still be preserved"
        assert sum(len(c) for c in state.queued_text) <= QUEUED_TEXT_MAX_BYTES


class TestDrainerLessExitIsNamed:
    """A turn that exits with a live future and no pump must SAY so.

    The two fixes above close the two paths we know about. This invariant is
    aimed at the THIRD one: any future branch that hands control back while
    the sink is still registered will announce itself here with a session id,
    instead of being reconstructed from an ``AcpSinkOverflowError`` raised
    20 minutes later at an unrelated ``response_fut.result()``.

    It logs rather than raises on purpose — an ``assert`` would vanish under
    ``python -O``, i.e. exactly in a production run.
    """

    @pytest.mark.asyncio
    async def test_live_future_without_pump_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pump_starts: list[str] = []
        model = _model(pump_starts)
        # Simulate the defect's shape directly: a branch that returns
        # tool_calls mid-turn WITHOUT arming a drainer.
        model._start_pump = lambda _state, _tid: None  # type: ignore[method-assign]
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        state.queued_tool_calls = [
            {
                "id": "call_1",
                "name": "wait_for_job",
                "args": {"job_id": "job_1"},
                "type": "tool_call",
            }
        ]

        with caplog.at_level("ERROR"):
            await model._consume_turn(state, "agent_1")

        assert state.pump_task is None
        assert not fut.done()
        drainerless = [
            r for r in caplog.records if "NO pump" in r.getMessage()
        ]
        assert drainerless, (
            "left _consume_turn with the sink registered and no reader, and "
            "said nothing — this is the exact silence that made the original "
            "defect expensive to localise"
        )
        assert state.session_id in drainerless[0].getMessage(), (
            "the log must name the session, otherwise it cannot be tied to an "
            "incident"
        )

    @pytest.mark.asyncio
    async def test_no_false_alarm_when_a_pump_is_armed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive control for the test above.

        Without this, a check that never fires would pass the first test's
        sibling by accident: 'no log' from a correct run and 'no log' from a
        broken invariant are indistinguishable.
        """
        pump_starts: list[str] = []
        model = _model(pump_starts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        state = _state(queue, fut)
        # The real path arms a pump; _model records it, so stand in for the
        # task the production code would have assigned.
        state.pump_task = MagicMock()
        state.queued_tool_calls = [
            {
                "id": "call_1",
                "name": "wait_for_job",
                "args": {"job_id": "job_1"},
                "type": "tool_call",
            }
        ]

        with caplog.at_level("ERROR"):
            await model._consume_turn(state, "agent_1")

        assert pump_starts == ["agent_1"]
        assert not [r for r in caplog.records if "NO pump" in r.getMessage()], (
            "a correctly drained mid-turn pause must not raise a false alarm"
        )
