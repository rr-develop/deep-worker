"""The number the CLI reports must reach the UI as it arrives, not at turn end.

Owner directive, restated 2026-08-26 after seeing the web UI stuck:

    "There must be one source of truth, and it must be the CORRECT one, the
    one that actually holds the most up-to-date data, and that is the one
    that must be displayed in the web UI"

What was actually happening, measured on a live agent (claude-code/opus,
``-l 250000``): the ACP layer received the CLI's ``usage_update`` during the
turn and merely stashed it in ``state.usage_snapshot``, publishing a
``token_stats`` row only at the turn boundary. So:

  * ``acp_session_state.json`` advanced 98163 -> 159414 -> 167831 as the
    agent worked, while
  * ``token_stats.jsonl`` — the ONLY file the UI status bar reads — had not
    been written since 14:02:30, before the agent was even restarted,

and the UI therefore showed "Provider: claude-code / Model: opus /
Context: 286.2k / Limit: 250k" — the provider and model of the live agent
next to a context number (286194) produced by the PREVIOUS process on a
DIFFERENT provider (grok). Two sources, one of them stale, presented as one
fact.

The rule these tests pin: there is exactly one place a new context reading
enters the model, and publishing it to the recorder is part of accepting it.
A reading that only ``client_state`` can see is not published.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.custom_types import ThreadState


def _model_with_recorder():
    """A bare ACP model whose snapshots are captured, not written."""
    model = object.__new__(AcpChatModel)
    seen: list[tuple[int, str | None]] = []

    async def _emit(used: int, *, session_id: str) -> None:
        seen.append((used, session_id))

    object.__setattr__(model, "_emit_context_snapshot", _emit)
    return model, seen


def _usage_update(used: int, size: int | None = None) -> dict:
    """One ``session/update`` notification as the CLI sends it."""
    update: dict = {"sessionUpdate": "usage_update", "used": used}
    if size is not None:
        update["size"] = size
    return {"params": {"update": update}}


def _apply(model, state, notif) -> None:
    model._apply_update(state, "agent-1", notif, [], [])


class TestAReadingIsPublishedWhenItArrives:
    """``usage_update`` is a model call finishing — publish it then."""

    @pytest.mark.asyncio
    async def test_the_reading_reaches_the_recorder_mid_turn(self) -> None:
        """RED: today the row is written only at the turn boundary."""
        model, seen = _model_with_recorder()
        state = ThreadState()
        state.session_id = "sess-1"

        _apply(model, state, _usage_update(159_414, 1_000_000))
        await model._publish_pending_context(state)

        assert seen == [(159_414, "sess-1")], (
            f"the CLI's reading never reached the recorder: {seen}. The UI "
            "reads token_stats.jsonl and nothing else, so a number only "
            "client_state can see is a number the operator cannot see"
        )

    @pytest.mark.asyncio
    async def test_one_row_per_model_call(self) -> None:
        """Three model calls are three readings, in order.

        Holding them until turn end collapses a whole turn's growth into a
        single jump — the frozen-then-jumping gauge the operator reported
        (40.9k for nine and a half minutes, then 71.7k).
        """
        model, seen = _model_with_recorder()
        state = ThreadState()
        state.session_id = "sess-1"

        for used in (98_163, 159_414, 167_831):
            _apply(model, state, _usage_update(used))
            await model._publish_pending_context(state)

        assert [u for u, _ in seen] == [98_163, 159_414, 167_831], (
            f"readings collapsed or reordered: {seen}"
        )

    @pytest.mark.asyncio
    async def test_the_same_number_is_not_republished(self) -> None:
        """A repeated reading is not news; the row must not be duplicated."""
        model, seen = _model_with_recorder()
        state = ThreadState()
        state.session_id = "sess-1"

        for _ in range(3):
            _apply(model, state, _usage_update(167_831))
            await model._publish_pending_context(state)

        assert [u for u, _ in seen] == [167_831], f"duplicate rows: {seen}"

    @pytest.mark.asyncio
    async def test_the_gauge_and_the_published_row_are_one_value(self) -> None:
        """One source of truth: what the recorder gets is what the guard and
        the summarizer read, at the same instant. They diverged in
        production — the gauge at 167831 while the last published row said
        286194 from another provider."""
        model, seen = _model_with_recorder()
        state = ThreadState()
        state.session_id = "sess-1"

        _apply(model, state, _usage_update(167_831))
        await model._publish_pending_context(state)

        assert state.last_context_used == 167_831, (
            "client_state was not updated with the reading that was published"
        )
        assert seen[-1][0] == state.last_context_used

    @pytest.mark.asyncio
    async def test_a_size_only_update_publishes_nothing(self) -> None:
        """No ``used`` means no context reading. POLICIES §13: never invent
        one — a zero would render as "context collapsed"."""
        model, seen = _model_with_recorder()
        state = ThreadState()
        state.session_id = "sess-1"

        _apply(model, state, {"params": {"update": {
            "sessionUpdate": "usage_update", "size": 1_000_000}}})
        await model._publish_pending_context(state)

        assert seen == [], f"a size-only update produced a reading: {seen}"

    @pytest.mark.asyncio
    async def test_publishing_never_breaks_the_turn(self) -> None:
        """Telemetry is best-effort. A broken recorder must not kill a turn,
        and must not leave the reading stuck as pending forever."""
        model = object.__new__(AcpChatModel)

        async def _boom(used: int, *, session_id: str) -> None:
            raise RuntimeError("recorder down")

        object.__setattr__(model, "_emit_context_snapshot", _boom)
        state = ThreadState()
        state.session_id = "sess-1"

        _apply(model, state, _usage_update(167_831))
        await model._publish_pending_context(state)  # must not raise

        assert state.last_context_used == 167_831


class TestPublishingDoesNotDamageTheStamp:
    """Publishing per model call must not cost the turn's richer stamp."""

    @pytest.mark.asyncio
    async def test_the_context_window_survives_the_final_flush(self) -> None:
        """Regression, introduced while making readings live.

        ``_finalize_response`` stamps from ``usage_snapshot``, which carries
        BOTH the reading and the context WINDOW, and then consumes that
        snapshot. ``_flush_live_context`` then ran a second stamp from the
        live gauge — which knows the reading but not the window — and
        overwrote ``size`` with None. Before readings were published for
        every ACP provider the live gauge was only ever set for grok, so
        nothing noticed.
        """
        from langchain_core.messages import AIMessage

        from deep_worker_cli.context_limit.observed_context import (
            annotate_context_observation,
            context_observation,
        )

        model = object.__new__(AcpChatModel)
        state = ThreadState()
        state.session_id = "sess-1"
        state.live_context_used = 468_146
        state.usage_snapshot = None  # already consumed by the turn's emit

        message = AIMessage(content="done")
        annotate_context_observation(
            message, used=468_146, size=1_000_000, first_id="h-first"
        )

        await model._flush_live_context(state, message)

        assert context_observation(message) == {
            "used": 468_146,
            "size": 1_000_000,
            "first_id": "h-first",
        }, "the final flush overwrote a richer stamp with a size-less one"


class TestWiring:
    """A publisher nobody calls is the same bug wearing a new name."""

    @pytest.mark.parametrize("method", ["_consume_turn", "_drain_pump"])
    def test_every_consumer_publishes_after_applying_an_update(
        self, method: str
    ) -> None:
        """Both paths see updates — the foreground consumer and the
        background pump — and the model is entered through either depending
        on who saw the notification first. A publisher wired into only one
        of them loses every reading of the other."""
        fn = getattr(AcpChatModel, method)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        assert "_apply_update" in calls, (
            f"{method} no longer applies session updates — re-seat this pin"
        )
        published = calls.count("_publish_pending_context")
        # Both consumers read updates in TWO places: the streaming loop, and
        # the drain that runs when the prompt future resolves. Each needs its
        # own publish — a single call satisfies "a publisher exists" while
        # half the readings still never reach the recorder, which is how the
        # first version of this pin passed a mutation that deleted one.
        assert published >= 2, (
            f"{method} publishes from {published} of its update-reading "
            "sites: the streaming loop and the resolve-time drain each carry "
            "readings, and whichever is unwired loses them silently. The UI "
            "then keeps showing whatever was written last — in production, "
            "another provider's number from another process"
        )
