"""TDD for the grok token-stats usage-event callback (DESIGN §11.4/§5.2).

Hermetic: reuses the same fake ACP client / proxy as ``test_chat_model``.
Covers:

* ``set_usage_event_callback`` installs the callback; ainvoke emits
  ``invocation_start`` (no record yet).
* PER-TURN TOKEN USAGE (live-verified 0.2.73, DESIGN §2.2 live-probe): the
  ``session/prompt`` RESULT ``_meta`` (``inputTokens``/``outputTokens``/
  ``cachedReadTokens``/``totalTokens``) is emitted as a plain token record
  (cache hit = ``cachedReadTokens``).
* ``usage_update`` (ACP session/update kind) accumulates the LATEST
  ``used``/``size`` gauge into ``state.usage_snapshot`` — top-level fields
  and the ``content``-dict fallback.
* At the turn boundary ``_finalize_response`` emits the terminal event and
  resets the accumulator; no ``uuid`` / ``turn_id`` travels (grok has no
  stable provider id, so its records are never deduplicated).
* No usage data in a turn → no terminal record (no data, honest).
* The callback is best-effort: exceptions are logged, never break the turn;
  ``asyncio.CancelledError`` propagates.
* ``_finalize_response`` without ``state`` (existing direct-call tests)
  never emits.

TWO shared-layer semantics changed with 3fe834d1 and are pinned here under
their new shape rather than the fork's:

* MID-TURN PUBLISHING. A ``usage_update`` IS a model call finishing, so the
  reading is published the moment it arrives (``_publish_pending_context`` →
  one record carrying ``context_used`` and no ``context_size``) instead of
  being held to the turn boundary. Holding them is what made the web UI sit
  frozen and then jump, and let the UI show a stale number from a previous
  process next to the live provider (2026-08-26). A turn with one gauge
  therefore emits TWO terminal records: the mid-turn reading and the
  turn-boundary one.
* HYBRID EMIT. ``_meta`` token usage and the ``usage_update`` gauge are no
  longer mutually exclusive: the terminal record carries BOTH. They answer
  different questions — ``usage`` is per-turn tokens (UI "Last turn"), the
  gauge is the live context comparable with ``-l`` (UI "Context"). When
  ``_meta`` suppressed the gauge, ``latest_context_by_kind`` fell back to
  ``total_tokens`` and the UI painted a per-turn number as the current
  context ("2265.4k vs -l 400k", agent_a 2026-08-22).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.grok_provider import acp_types  # noqa: E402
from deep_worker.providers.grok_provider.chat_model import GrokChatModel  # noqa: E402
# The shared fakes live in the sibling test module. This directory has no
# __init__.py (pytest rootdir-relative collection), and the old
# ``deep_worker.providers.grok_provider.tests`` package no longer exists —
# the grok tests moved to the repo-root ``tests/`` tree. Import the sibling
# by adding THIS directory to sys.path.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from test_chat_model import (  # noqa: E402
    _FakeAcpClient,
    _FakeProxy,
    install_client_state,
)


class _Recorder:
    """Captures every usage-event callback invocation."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.raise_on_terminal: BaseException | None = None

    async def __call__(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))
        if kwargs.get("event") == "terminal" and self.raise_on_terminal is not None:
            raise self.raise_on_terminal

    @property
    def kinds(self) -> list[str]:
        return [e["event"] for e in self.events]

    @property
    def terminals(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "terminal"]

    @property
    def boundary(self) -> dict:
        """The TURN-BOUNDARY terminal record.

        Mid-turn readings (``_publish_pending_context``) and the boundary
        record share the ``terminal`` event name, so "the last one" is the
        boundary one — it is emitted from ``_finalize_response`` after the
        queue is fully drained. Asserted as the last element rather than the
        only one on purpose: a turn with a gauge legitimately emits both.
        """
        terms = self.terminals
        assert terms, f"no terminal record emitted; got {self.kinds!r}"
        return terms[-1]


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def model(fake_home, monkeypatch):
    cs = install_client_state(monkeypatch)
    m = GrokChatModel(model="grok-4")
    m._client_state = cs
    m._acp_client = _FakeAcpClient()
    m._proxy_mcp = _FakeProxy()
    m._proxy_started = True
    m._acp_started = True
    m._working_dir = "/tmp"
    return m


def _cfg(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


async def _run_text_turn(
    model, recorder, updates: list[dict], meta: dict | None = None
) -> AIMessage:
    """Drive one text-only ainvoke with the given session/update payloads."""
    acp = model._acp_client
    task = asyncio.create_task(
        model.ainvoke([HumanMessage(content="hi")], config=_cfg("agent_x"))
    )
    await asyncio.sleep(0.01)
    sid = f"sess-{acp.session_counter}"
    for upd in updates:
        acp.push_update(sid, upd)
    acp.finish("end_turn", meta=meta)
    return await asyncio.wait_for(task, timeout=1)


class TestAcpTypes:
    def test_usage_update_is_handled_kind(self) -> None:
        assert "usage_update" in acp_types.HANDLED_SESSION_UPDATE_KINDS


class TestUsageEventCallback:
    async def test_invocation_start_and_terminal_snapshot(self, model) -> None:
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        msg = await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update", "used": 1200, "size": 200000},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "hello"}},
        ])
        assert msg.content == "hello"
        # ONE invocation_start, then terminal record(s) — and it is always
        # FIRST: the UI must know a turn began before any number lands.
        assert rec.kinds[0] == "invocation_start"
        assert rec.kinds.count("invocation_start") == 1
        assert set(rec.kinds[1:]) == {"terminal"}
        # The gauge was published mid-turn the moment the usage_update
        # arrived (not held to the boundary) — a record carrying the reading
        # and no window yet.
        mid = rec.terminals[0]
        assert mid["context_used"] == 1200
        assert mid.get("context_size") is None
        # ...and the turn-boundary record carries the full gauge.
        term = rec.boundary
        assert term["status"] == "ok"
        assert term["context_used"] == 1200
        assert term["context_size"] == 200000
        assert term.get("usage") is None
        # grok: no stable provider id → never deduplicated. The shared layer
        # emits no uuid/turn_id key at all, which satisfies the same oracle.
        assert term.get("uuid") is None
        assert term.get("turn_id") is None
        assert term["session_id"] is not None

    async def test_terminal_emits_meta_token_usage(self, model) -> None:
        """LIVE-RED-DISCOVERY (0.2.73): session/prompt RESULT._meta carries
        per-turn token usage — recorded as a plain token record, cache hit =
        cachedReadTokens (DESIGN §2.2 live-probe branch)."""
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        msg = await _run_text_turn(model, rec, [
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "hello"}},
        ], meta={
            "modelId": "grok-4.5",
            "inputTokens": 19331,
            "outputTokens": 38,
            "cachedReadTokens": 128,
            "totalTokens": 19370,
        })
        assert msg.content == "hello"
        # No usage_update in this turn → nothing to publish mid-turn, so the
        # boundary record is the only terminal one.
        assert rec.kinds == ["invocation_start", "terminal"]
        term = rec.boundary
        assert term["status"] == "ok"
        assert term["usage"] == {
            "input_tokens": 19331,
            "output_tokens": 38,
            "cached_input_tokens": 128,
            "total_tokens": 19370,
        }
        # With no gauge observed there is no context to report — honest null,
        # NOT total_tokens masquerading as the live context.
        assert term.get("context_used") is None
        assert term.get("context_size") is None

    async def test_meta_usage_and_snapshot_emit_together(self, model) -> None:
        """BOTH a usage_update gauge and ``_meta`` usage present → ONE hybrid
        record carrying both.

        Was ``test_meta_usage_prefers_token_record_over_snapshot``: the fork
        made the two mutually exclusive and dropped the gauge when ``_meta``
        had totals. That is the defect, not the contract —
        ``latest_context_by_kind`` then fell back to ``total_tokens`` and the
        UI painted a per-turn number (2265.4k) in the Context field of an
        agent launched with ``-l 400k`` (agent_a 2026-08-22). They answer
        different questions and must both survive: ``usage`` is per-turn
        tokens, the gauge is the live context comparable with ``-l``.
        """
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update", "used": 500, "size": 1000},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "ok"}},
        ], meta={
            "modelId": "grok-4.5",
            "inputTokens": 10,
            "outputTokens": 2,
            "cachedReadTokens": 3,
            "totalTokens": 12,
        })
        term = rec.boundary
        assert term["usage"] == {
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_input_tokens": 3,
            "total_tokens": 12,
        }
        # The gauge is NOT suppressed by the token record.
        assert term["context_used"] == 500
        assert term["context_size"] == 1000
        # And it is the GAUGE, not total_tokens, that reports the context.
        assert term["context_used"] != term["usage"]["total_tokens"]

    async def test_meta_without_token_fields_falls_back_to_snapshot(
        self, model
    ) -> None:
        """_meta without inputTokens/outputTokens must NOT be treated as a
        token record — the usage_update gauge fallback still applies."""
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update", "used": 42, "size": 50000},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "ok"}},
        ], meta={"modelId": "grok-4.5"})
        term = rec.boundary
        assert term["context_used"] == 42
        assert term["context_size"] == 50000
        assert term.get("usage") is None

    async def test_latest_usage_update_wins(self, model) -> None:
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update", "used": 100, "size": 100000},
            {"sessionUpdate": "usage_update", "used": 9000, "size": 200000},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "done"}},
        ])
        term = rec.boundary
        assert term["context_used"] == 9000
        assert term["context_size"] == 200000

    async def test_usage_update_content_dict_fallback(self, model) -> None:
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update",
             "content": {"used": 42, "size": 50000}},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "ok"}},
        ])
        term = rec.boundary
        assert term["context_used"] == 42
        assert term["context_size"] == 50000

    async def test_no_usage_update_no_terminal_record(self, model) -> None:
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        await _run_text_turn(model, rec, [
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "hi there"}},
        ])
        assert [e["event"] for e in rec.events] == ["invocation_start"]
        st = model._client_state.get("agent_x")
        assert st is not None
        assert st.usage_snapshot is None

    async def test_snapshot_reset_after_turn(self, model) -> None:
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update", "used": 5, "size": 10},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "one"}},
        ])
        st = model._client_state.get("agent_x")
        assert st is not None
        assert st.usage_snapshot is None, "accumulator must reset at turn end"

    async def test_callback_error_does_not_break_turn(self, model) -> None:
        rec = _Recorder()
        rec.raise_on_terminal = RuntimeError("stats boom")
        model.set_usage_event_callback(rec)
        msg = await _run_text_turn(model, rec, [
            {"sessionUpdate": "usage_update", "used": 1, "size": 2},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "still works"}},
        ])
        assert msg.content == "still works"

    async def test_cancelled_error_propagates(self, model) -> None:
        rec = _Recorder()
        rec.raise_on_terminal = asyncio.CancelledError()
        model.set_usage_event_callback(rec)
        with pytest.raises(asyncio.CancelledError):
            await _run_text_turn(model, rec, [
                {"sessionUpdate": "usage_update", "used": 1, "size": 2},
                {"sessionUpdate": "agent_message_chunk",
                 "content": {"text": "boom"}},
            ])

    async def test_no_callback_is_noop(self, model) -> None:
        msg = await _run_text_turn(model, _Recorder(), [
            {"sessionUpdate": "usage_update", "used": 1, "size": 2},
            {"sessionUpdate": "agent_message_chunk",
             "content": {"text": "plain"}},
        ])
        assert msg.content == "plain"

    async def test_finalize_without_state_does_not_emit(self, model) -> None:
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        msg = await model._finalize_response(
            {"result": {"stopReason": "end_turn"}}, ["ok"]
        )
        assert msg.content == "ok"
        assert rec.events == []

    async def test_mid_turn_pump_accumulates_snapshot(self, model) -> None:
        """usage_update arriving while the mid-turn pump drains must land in
        the same per-turn accumulator (queued_response path emits it)."""
        rec = _Recorder()
        model.set_usage_event_callback(rec)
        st = model._client_state.get_or_create("agent_x")
        st.session_id = "sess-pump"
        st.sent_count = 0
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait({
            "method": "session/update",
            "params": {"sessionId": "sess-pump",
                       "update": {"sessionUpdate": "usage_update",
                                  "used": 777, "size": 99999}},
        })
        q.put_nowait({
            "method": "session/update",
            "params": {"sessionId": "sess-pump",
                       "update": {"sessionUpdate": "agent_message_chunk",
                                  "content": {"text": "pumped"}}},
        })
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result({"result": {"stopReason": "end_turn"}})
        st.current_notif_queue = q
        st.pending_session_prompt_future = fut

        msg = await asyncio.wait_for(
            model._consume_turn(st, "agent_x"), timeout=1
        )
        assert msg.content == "pumped"
        # _consume_turn is entered directly (no ainvoke), so NO
        # invocation_start — only terminal records.
        assert set(rec.kinds) == {"terminal"}
        # The reading is published as it is drained (mid-turn), then again at
        # the boundary with the window. Both must carry THIS turn's gauge.
        assert rec.terminals[0]["context_used"] == 777
        term = rec.boundary
        assert term["context_used"] == 777
        assert term["context_size"] == 99999
        # The accumulator is reset at the boundary, as on any other turn.
        assert st.usage_snapshot is None
