"""TDD for the grok provider process-global client state.

``GrokClientState`` keys per-``agent_thread_id`` ``_ThreadState`` slots and
maintains a reverse ``grok_session_id -> agent_thread_id`` index so the
single shared ACP reader_task can route ``session/update`` notifications
back to the owning thread (DESIGN §11A.4, §12).

L3: on summarization we reset the slot CONTENTS (drop session id, zero
sent_count) and evict the stale session from the reverse index — but the
key (the agent_thread_id) is NEVER mutated.

Hermetic: ``Path.home()`` monkeypatched so the lazy disk-seed in
``get_or_create`` has a clean tmp dir.
"""
from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from grok_provider.client_state import GrokClientState  # noqa: E402
from grok_provider.custom_types import ThreadState  # noqa: E402
from grok_provider.session_state import session_state_path  # noqa: E402


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestGetOrCreate:
    def test_creates_fresh_slot(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("a")
        assert isinstance(st, ThreadState)
        assert st.sent_count == 0
        assert st.grok_session_id is None

    def test_returns_same_slot(self, fake_home) -> None:
        cs = GrokClientState()
        st1 = cs.get_or_create("a")
        st1.sent_count = 5
        st2 = cs.get_or_create("a")
        assert st2 is st1
        assert st2.sent_count == 5  # hot slot not re-read from disk

    def test_lazy_seed_from_disk(self, fake_home) -> None:
        from grok_provider.session_state import save_session_state

        save_session_state("seeded", sent_count=9, session_id="sess-9")
        cs = GrokClientState()
        st = cs.get_or_create("seeded")
        assert st.sent_count == 9
        assert st.grok_session_id == "sess-9"

    def test_thread_state_has_passthrough_futures(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("a")
        assert isinstance(st.builtin_passthrough_futures, dict)
        assert st.builtin_passthrough_futures == {}


class TestSessionIndex:
    def test_register_then_resolve(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("agentA")
        st.grok_session_id = "sess-A"
        cs.register_session("sess-A", "agentA")
        assert cs.thread_for_session("sess-A") is st

    def test_resolve_unknown_returns_none(self, fake_home) -> None:
        cs = GrokClientState()
        assert cs.thread_for_session("nope") is None

    def test_two_agents_distinct_sessions(self, fake_home) -> None:
        cs = GrokClientState()
        sa = cs.get_or_create("agentA")
        sb = cs.get_or_create("agentB")
        cs.register_session("sA", "agentA")
        cs.register_session("sB", "agentB")
        assert cs.thread_for_session("sA") is sa
        assert cs.thread_for_session("sB") is sb
        assert sa is not sb


class TestSummarization:
    def test_reset_drops_session_and_zeroes_count(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("agentA")
        st.sent_count = 12
        st.grok_session_id = "sess-old"
        cs.register_session("sess-old", "agentA")

        cs.record_summarization("agentA")

        assert st.grok_session_id is None
        assert st.sent_count == 0

    def test_reset_evicts_stale_session_from_index(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("agentA")
        st.grok_session_id = "sess-old"
        cs.register_session("sess-old", "agentA")

        cs.record_summarization("agentA")

        # A late update for the dead session must NOT route anywhere.
        assert cs.thread_for_session("sess-old") is None

    def test_key_is_never_mutated(self, fake_home) -> None:
        cs = GrokClientState()
        cs.get_or_create("agentA")
        cs.record_summarization("agentA")
        # The agent_thread_id slot still exists under the same key.
        assert cs.get("agentA") is not None

    def test_reset_persists(self, fake_home) -> None:
        from grok_provider.session_state import load_session_state

        cs = GrokClientState()
        st = cs.get_or_create("agentA")
        st.sent_count = 3
        st.grok_session_id = "sess-x"
        cs.persist("agentA")
        cs.record_summarization("agentA")
        loaded = load_session_state("agentA")
        assert loaded == {"sent_count": 0, "session_id": None}


class TestPersist:
    def test_persist_writes_disk_with_perms(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("agentA")
        st.sent_count = 4
        st.grok_session_id = "sess-y"
        cs.persist("agentA")
        path = session_state_path("agentA")
        assert path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_persist_noop_when_slot_missing(self, fake_home) -> None:
        cs = GrokClientState()
        cs.persist("never_created")  # must not raise
        assert not session_state_path("never_created").exists()


class TestThreadStateQueue:
    def test_notif_queue_is_asyncio_queue_or_none(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("a")
        # Starts None; ainvoke assigns a live queue per turn.
        assert st.current_notif_queue is None
        st.current_notif_queue = asyncio.Queue()
        assert isinstance(st.current_notif_queue, asyncio.Queue)
