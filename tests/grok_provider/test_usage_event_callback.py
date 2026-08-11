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
  and the ``content``-dict fallback. Used ONLY when no ``_meta`` usage was
  seen (fallback).
* At the turn boundary ``_finalize_response`` emits ONE terminal event and
  resets the accumulator; ``event_key`` stays null (grok, MINOR v3).
* No usage data in a turn → no terminal record (no data, honest).
* The callback is best-effort: exceptions are logged, never break the turn;
  ``asyncio.CancelledError`` propagates.
* ``_finalize_response`` without ``state`` (existing direct-call tests)
  never emits.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.grok_provider import acp_types  # noqa: E402
from deep_worker.providers.grok_provider.chat_model import GrokChatModel  # noqa: E402
from deep_worker.providers.grok_provider.tests.test_chat_model import (  # noqa: E402
    _FakeAcpClient,
    _FakeProxy,
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


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def model(fake_home, monkeypatch):
    from deep_worker.providers.grok_provider import chat_model as cm
    from deep_worker.providers.grok_provider.client_state import GrokClientState

    cs = GrokClientState()
    monkeypatch.setattr(cm, "get_global_client_state", lambda: cs)
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
        kinds = [e["event"] for e in rec.events]
        assert kinds == ["invocation_start", "terminal"]
        term = rec.events[-1]
        assert term["status"] == "ok"
        assert term["context_used"] == 1200
        assert term["context_size"] == 200000
        assert term.get("usage") is None
        # grok: no stable provider id → never deduplicated.
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
        kinds = [e["event"] for e in rec.events]
        assert kinds == ["invocation_start", "terminal"]
        term = rec.events[-1]
        assert term["status"] == "ok"
        assert term["usage"] == {
            "input_tokens": 19331,
            "output_tokens": 38,
            "cached_input_tokens": 128,
            "total_tokens": 19370,
        }
        # Token record wins over any accumulated usage_update gauge.
        assert term.get("context_used") is None
        assert term.get("context_size") is None

    async def test_meta_usage_prefers_token_record_over_snapshot(self, model) -> None:
        """If BOTH a usage_update gauge and _meta usage are present, the
        token record wins (it is the honest per-turn data)."""
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
        term = rec.events[-1]
        assert term["usage"] == {
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_input_tokens": 3,
            "total_tokens": 12,
        }
        assert term.get("context_used") is None

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
        term = rec.events[-1]
        assert term["event"] == "terminal"
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
        term = rec.events[-1]
        assert term["event"] == "terminal"
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
        term = rec.events[-1]
        assert term["event"] == "terminal"
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
        st.grok_session_id = "sess-pump"
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
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result({"result": {"stopReason": "end_turn"}})
        st.current_notif_queue = q
        st.pending_session_prompt_future = fut

        msg = await asyncio.wait_for(
            model._consume_turn(st, "agent_x"), timeout=1
        )
        assert msg.content == "pumped"
        # _consume_turn is entered directly (no ainvoke) — only the terminal
        # event is emitted here.
        assert [e["event"] for e in rec.events] == ["terminal"]
        assert rec.events[-1]["context_used"] == 777
        assert rec.events[-1]["context_size"] == 99999
