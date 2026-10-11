"""The context gauge must reach disk as soon as it is stamped.

Reviewer blocker (2026-08-21): ``_stamp_context_observation``
wrote ``state.last_context_used`` / ``state.last_response_at`` into memory,
but ``persist()`` only ran at the START of the next turn
(``chat_model.py`` session/new|load paths). A ``--restart`` after a
completed turn therefore lost the gauge, and ``extra_floor``
(``agent.py``, DESIGN §6) came back blind on the first ``before_model`` —
the exact failure mode of a related internal incident (analysis not included here).

These tests pin that the write is durable at stamp time, not one turn later.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.profile import AcpProfile
from deep_worker.providers.acp import session_state as session_state_mod


THREAD = "gauge_agent"


def _model(client_state: AcpClientState) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._acp_client = MagicMock()
    model._acp_client.session_cancel = AsyncMock()
    model._profile = AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
    )
    model._client_state = client_state
    model._proxy_mcp = None
    model._tools_metadata = []
    model._usage_event_callback = None
    return model


def _isolate(tmp_path, monkeypatch) -> None:
    """Point the session-state file at a temp dir (never touch prod state)."""
    monkeypatch.setattr(
        session_state_mod,
        "session_state_path",
        lambda thread_id: tmp_path / "acp_session_state.json",
    )


def _persisted(tmp_path) -> dict:
    path = tmp_path / "acp_session_state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def test_stamp_persists_gauge_immediately(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    cs = AcpClientState()
    cs.set_profile("claude-code", {"model": "fable"})
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    st.usage_snapshot = {"used": 12345, "size": 200000}

    model = _model(cs)
    model._stamp_context_observation(st, AIMessage(content="hi"))

    entry = _persisted(tmp_path).get(THREAD, {})
    assert entry.get("last_context_used") == 12345, (
        "gauge was not on disk right after the stamp — a --restart here "
        "loses it and extra_floor comes back blind"
    )
    assert entry.get("last_response_at"), "last_response_at not persisted"


def test_no_snapshot_does_not_write_a_bogus_gauge(tmp_path, monkeypatch) -> None:
    """No usage snapshot → no stamp → nothing claimed on disk."""
    _isolate(tmp_path, monkeypatch)
    cs = AcpClientState()
    cs.set_profile("claude-code", {"model": "fable"})
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    st.usage_snapshot = None

    model = _model(cs)
    model._stamp_context_observation(st, AIMessage(content="hi"))

    entry = _persisted(tmp_path).get(THREAD, {})
    assert not entry.get("last_context_used")


def test_persist_state_resolves_the_owning_key(tmp_path, monkeypatch) -> None:
    """``persist_state`` must find the slot by identity, not by guessing.

    Two threads share one container; persisting one must not write the
    other's numbers under the wrong key.
    """
    _isolate(tmp_path, monkeypatch)
    cs = AcpClientState()
    cs.set_profile("claude-code", {"model": "fable"})
    a = cs.get_or_create("agent_a")
    b = cs.get_or_create("agent_b")
    a.session_id, b.session_id = "sess-a", "sess-b"
    a.last_context_used, b.last_context_used = 111, 222

    cs.persist_state(b)

    data = _persisted(tmp_path)
    assert data.get("agent_b", {}).get("last_context_used") == 222
    assert "agent_a" not in data or not data["agent_a"].get("last_context_used")


def test_persist_state_is_silent_for_an_unknown_state(tmp_path, monkeypatch) -> None:
    """A detached ThreadState must not raise or write anything."""
    from deep_worker.providers.acp.custom_types import ThreadState

    _isolate(tmp_path, monkeypatch)
    cs = AcpClientState()
    cs.set_profile("claude-code", {"model": "fable"})
    cs.persist_state(ThreadState())
    assert _persisted(tmp_path) == {}
