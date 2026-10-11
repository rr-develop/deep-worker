"""Post-summary reset must drop queued mid-turn residue (incident 2026-08-19)."""

from __future__ import annotations

import asyncio

from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import ThreadState


def test_record_summarization_clears_queued_tool_calls(
    tmp_path, monkeypatch
) -> None:
    from deep_worker.providers.acp import session_state as ss

    monkeypatch.setattr(ss, "deep_worker_home", lambda: tmp_path)
    state = AcpClientState()
    state.set_profile("claude-code", None)
    st = state.get_or_create("rg1_worker")
    st.session_id = "f44ee866"
    st.sent_count = 722
    st.queued_text.append("partial")
    st.queued_tool_calls.append({"id": "bash-2", "name": "Bash", "args": {}})
    st.queued_response = {"stopReason": "end_turn"}
    loop = asyncio.new_event_loop()
    fut: asyncio.Future = loop.create_future()
    st.pending_session_prompt_future = fut

    state.record_summarization("rg1_worker")

    after = state.get("rg1_worker")
    assert after is not None
    assert after.session_id is None
    assert after.sent_count == 0
    assert after.queued_text == []
    assert after.queued_tool_calls == []
    assert after.queued_response is None
    assert after.pending_session_prompt_future is None
    assert fut.cancelled()
    loop.close()


def test_record_summarization_clears_context_gauge(tmp_path, monkeypatch) -> None:
    """The gauge describes the session we just dropped.

    ``extra_floor`` reads ``last_context_used``; leaving the pre-summary
    468k behind makes the next turn (or a ``--restart`` before it)
    re-trigger summarization on a context that was already collapsed.
    Must be cleared in memory AND on disk (reviewer B1, 2026-08-21).
    """
    from deep_worker.providers.acp import session_state as ss

    monkeypatch.setattr(ss, "deep_worker_home", lambda: tmp_path)
    state = AcpClientState()
    state.set_profile("claude-code", None)
    st = state.get_or_create("rg1_worker")
    st.session_id = "f44ee866"
    st.sent_count = 722
    st.last_context_used = 468_000
    st.last_response_at = "2026-08-19T15:14:02+00:00"
    state.persist("rg1_worker")

    def _load() -> dict:
        loaded = ss.load_session_state(
            "rg1_worker", provider="claude-code", fingerprint=None
        )
        assert loaded is not None
        return loaded

    assert _load()["last_context_used"] == 468_000

    state.record_summarization("rg1_worker")

    after = state.get("rg1_worker")
    assert after is not None
    assert after.last_context_used is None
    assert after.last_response_at is None

    on_disk = _load()
    assert on_disk["last_context_used"] is None
    assert on_disk["last_response_at"] is None
