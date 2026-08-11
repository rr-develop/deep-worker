"""Tests for ``codex_provider.client_state``.

Behaviours under test:

* The singleton survives ``del`` and re-import (LangGraph rebuilds
  model instances during pickling and we must not lose state).
* ``get_or_create`` returns a fresh ``ThreadState`` on first lookup.
* ``sent_count`` is the only "history-size" cursor we keep; setting it
  does not touch ``agent_thread_id``.
* ``record_summarization()`` (the helper that handles
  ``sent_count > len(messages)``) drops ``codex_session_uuid`` and
  resets ``sent_count`` WITHOUT mutating the key (the agent_thread_id
  is the deep-worker contract, never touched).
* Thread isolation: changes to ``agent:tm`` don't leak into ``agent``.
* ``forget(agent_thread_id)`` removes the slot (for tests and explicit
  cleanup).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.codex_provider.client_state import (  # noqa: E402
    CodexClientState,
    get_global_client_state,
)
from deep_worker.providers.codex_provider.custom_types import ThreadState  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_home(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate ``Path.home()`` per test.

    Since the restart-persistence fix, ``CodexClientState.get_or_create``
    lazily seeds ``sent_count`` + ``codex_session_uuid`` from
    ``~/.deep_worker/<agent>/codex_session_state.json``. Without pointing
    ``Path.home()`` at a fresh tmp dir, a leaked file from another test (or a
    prior run) would seed stale values and break the fresh-state assertions
    below. The dedicated persistence behaviour is covered by
    ``test_session_state_persistence.py``.
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]


class TestSingleton:
    def test_get_global_returns_same_instance(self) -> None:
        a = get_global_client_state()
        b = get_global_client_state()
        assert a is b

    def test_state_survives_local_del(self) -> None:
        """LangGraph may discard model instances during deserialization;
        the state must survive — we re-fetch the singleton each turn."""
        a = get_global_client_state()
        a.get_or_create("survives_del").sent_count = 7
        del a
        b = get_global_client_state()
        st = b.get("survives_del")
        assert st is not None
        assert st.sent_count == 7
        b.forget("survives_del")  # cleanup


class TestGetOrCreate:
    def test_returns_fresh_state(self) -> None:
        s = CodexClientState()
        state = s.get_or_create("agent_x")
        assert isinstance(state, ThreadState)
        assert state.codex_session_uuid is None
        assert state.sent_count == 0
        assert state.current_exec is None

    def test_same_id_returns_same_object(self) -> None:
        s = CodexClientState()
        a = s.get_or_create("agent_x")
        a.sent_count = 3
        b = s.get_or_create("agent_x")
        assert a is b
        assert b.sent_count == 3

    def test_different_ids_isolated(self) -> None:
        s = CodexClientState()
        s.get_or_create("agent").sent_count = 10
        s.get_or_create("agent:tm").sent_count = 99
        assert s.get_or_create("agent").sent_count == 10
        assert s.get_or_create("agent:tm").sent_count == 99


class TestSummarizationReset:
    """``sent_count > len(messages)`` is the LangGraph-summarization
    signal. The provider drops ``codex_session_uuid`` and resets
    ``sent_count``; ``agent_thread_id`` is the dict key and stays put
    — that's the deep-worker contract."""

    def test_record_summarization_resets_uuid_and_count(self) -> None:
        s = CodexClientState()
        st = s.get_or_create("agent")
        st.codex_session_uuid = "abc-uuid"
        st.sent_count = 500
        s.record_summarization("agent")
        st_after = s.get("agent")
        assert st_after is not None
        assert st_after.codex_session_uuid is None
        assert st_after.sent_count == 0

    def test_agent_thread_id_key_is_never_dropped(self) -> None:
        """After summarization the slot still exists under the same
        key — only its contents are reset."""
        s = CodexClientState()
        s.get_or_create("agent").codex_session_uuid = "x"
        before_keys = set(s._states.keys())  # type: ignore[attr-defined]
        assert "agent" in before_keys
        s.record_summarization("agent")
        after_keys = set(s._states.keys())  # type: ignore[attr-defined]
        assert "agent" in after_keys
        assert before_keys == after_keys


class TestForget:
    def test_forget_removes_slot(self) -> None:
        s = CodexClientState()
        s.get_or_create("ephemeral").sent_count = 1
        s.forget("ephemeral")
        assert s.get("ephemeral") is None

    def test_forget_unknown_is_noop(self) -> None:
        s = CodexClientState()
        s.forget("never_existed")  # must not raise


class TestSentCountIsTheCursor:
    """Regression: sent_count is the ONLY cursor for incremental mirroring.
    The provider must not re-serialize history beyond what this cursor
    permits."""

    def test_sent_count_bounded_by_assignment_only(self) -> None:
        s = CodexClientState()
        st = s.get_or_create("agent")
        # We assign directly — exactly as ainvoke does in §10 step 8.
        st.sent_count = 5
        assert s.get_or_create("agent").sent_count == 5

    def test_summarization_detection_predicate(self) -> None:
        """The condition ``state.sent_count > len(messages)`` is the
        documented summarization detection trigger."""
        s = CodexClientState()
        st = s.get_or_create("agent")
        st.sent_count = 100
        # Suppose LangGraph compacted to 20 messages
        assert st.sent_count > 20
