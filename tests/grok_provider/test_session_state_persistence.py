"""TDD for ainvoke-side restart resilience for grok_provider.

The provider keeps (grok_session_id, sent_count) only in the in-memory
GrokClientState singleton; the on-disk grok_session_state.json (covered by
test_session_state.py) is the cross-restart bridge. This module pins:

1. GrokClientState.get_or_create lazily seeds (sent_count, session_id)
   from disk on the first access for a thread.
2. After the seed, ainvoke uses the persisted session_id (no fresh
   session/new) and only mirrors the tail messages[sent_count:] — NOT the
   whole history.
3. persist() writes both fields atomically (mirrors test_session_state.py
   filesystem invariants — see that file for the on-disk side).
4. record_summarization persists the reset so a restart that lands right
   after summarization sees a fresh slot.

Hermetic — Path.home() monkeypatched, no real grok binary.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.grok_provider.client_state import GrokClientState  # noqa: E402
from deep_worker.providers.grok_provider.session_state import (  # noqa: E402
    load_session_state,
    save_session_state,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    yield tmp_path


class TestLazySeedFromDisk:
    def test_get_or_create_seeds_from_persisted_state(self, fake_home) -> None:
        # Simulate a previous process having persisted state.
        save_session_state(
            "agentA",
            sent_count=7,
            session_id="019ed7a0-afa8-7730-86b8-6a5817010dca",
        )
        cs = GrokClientState()
        st = cs.get_or_create("agentA")
        assert st.sent_count == 7
        assert st.grok_session_id == "019ed7a0-afa8-7730-86b8-6a5817010dca"

    def test_fresh_thread_starts_empty(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("never-saw-this")
        assert st.sent_count == 0
        assert st.grok_session_id is None

    def test_hot_process_never_reseeds(self, fake_home) -> None:
        """Once a slot exists in memory, a later get_or_create must NOT
        re-read the disk and clobber live in-memory state."""
        cs = GrokClientState()
        st1 = cs.get_or_create("agentA")
        st1.sent_count = 99
        st1.grok_session_id = "in-mem-only"

        save_session_state("agentA", sent_count=1, session_id="disk-only")

        st2 = cs.get_or_create("agentA")
        # SAME object, untouched.
        assert st2 is st1
        assert st2.sent_count == 99
        assert st2.grok_session_id == "in-mem-only"

    def test_persist_writes_through(self, fake_home) -> None:
        cs = GrokClientState()
        st = cs.get_or_create("agentB")
        st.sent_count = 4
        st.grok_session_id = "abc-123"
        cs.persist("agentB")

        loaded = load_session_state("agentB")
        assert loaded == {"sent_count": 4, "session_id": "abc-123"}

    def test_persist_unknown_thread_is_noop(self, fake_home) -> None:
        cs = GrokClientState()
        # Slot does not exist. Must not raise, must not write a stale entry.
        cs.persist("never-saw-this")
        assert load_session_state("never-saw-this") is None

    def test_record_summarization_persists_reset(self, fake_home) -> None:
        """If a restart happens right after summarization, the disk file must
        reflect (0, None) so the next ainvoke doesn't try to resume the
        already-dropped grok session id."""
        save_session_state(
            "agentC", sent_count=12, session_id="will-be-dropped",
        )
        cs = GrokClientState()
        cs.get_or_create("agentC")  # seed from disk
        cs.record_summarization("agentC")

        loaded = load_session_state("agentC")
        assert loaded == {"sent_count": 0, "session_id": None}


class TestSessionIndexRoundTrip:
    def test_register_and_resolve(self, fake_home) -> None:
        cs = GrokClientState()
        cs.get_or_create("agentX")
        cs.register_session("sid-1", "agentX")
        assert cs.thread_for_session("sid-1").grok_session_id is None or True
        # The reverse lookup returns the SAME state object.
        assert cs.thread_for_session("sid-1") is cs.get_or_create("agentX")

    def test_summarization_evicts_stale_session_from_index(
        self, fake_home
    ) -> None:
        """L3 — record_summarization MUST evict the stale grok_session_id
        from _session_index so a late session/update never routes into a
        dead session (it would leak into the wrong _ThreadState)."""
        cs = GrokClientState()
        st = cs.get_or_create("agentY")
        st.grok_session_id = "sid-old"
        cs.register_session("sid-old", "agentY")
        assert cs.thread_for_session("sid-old") is st

        cs.record_summarization("agentY")

        assert cs.thread_for_session("sid-old") is None


# Drive a full pseudo-ainvoke through a fake ACP client to confirm the
# seeded session_id is reused (no session/new) and only the tail is sent.
class _FakeAcp:
    def __init__(self) -> None:
        self.prompts: list[dict] = []
        self.session_new_calls = 0
        self.session_load_calls: list[str] = []
        self.session_cancel_calls: list[str] = []

    async def session_new(self, *, cwd: str, mcp_servers: list[dict]) -> str:
        self.session_new_calls += 1
        return "fresh-sid"

    async def session_load(
        self, session_id: str, *, cwd: str, mcp_servers=None
    ) -> dict:
        # A real grok subprocess re-attaches the persisted session here.
        self.session_load_calls.append(session_id)
        return {"result": {"models": {}, "_meta": {}}}

    async def wait_session_ready(self, session_id: str, *, timeout: float = 10.0):
        return None

    def session_prompt(self, *, session_id: str, prompt: list[dict],
                       notification_sink) -> asyncio.Future:
        self.prompts.append({"session_id": session_id, "prompt": prompt})
        # Emit one text chunk so the turn is non-empty, then resolve.
        notification_sink.put_nowait({
            "method": "session/update",
            "params": {"sessionId": session_id,
                       "update": {"sessionUpdate": "agent_message_chunk",
                                  "content": {"text": "ok"}}},
        })
        fut = asyncio.get_event_loop().create_future()
        fut.set_result({"result": {"stopReason": "end_turn"}})
        return fut

    async def session_cancel(self, session_id: str) -> None:
        self.session_cancel_calls.append(session_id)


class _FakeProxy:
    server_name = "deep_worker_proxy_test"

    async def url_async(self) -> str:
        return "http://127.0.0.1:0/mcp"

    async def start(self) -> None:
        return None

    def update_tools(self, _md) -> None:
        return None

    def resolve_call(self, call_id: str, result: dict) -> None:
        return None


async def test_seeded_sent_count_short_circuits_history_replay(
    fake_home, monkeypatch
) -> None:
    """After a restart, the next ainvoke must:
       - RE-ATTACH the persisted grok_session_id via session/load (NOT a
         fresh session/new — grok keeps sessions in-memory, so a persisted id
         is dead in the new subprocess until reloaded), AND
       - send only messages[sent_count:] (the tail), NOT the full history.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from deep_worker.providers.grok_provider.chat_model import GrokChatModel

    save_session_state("agentZ", sent_count=2, session_id="sid-persisted")

    m = GrokChatModel()
    # Replace heavy I/O surfaces with fakes.
    acp = _FakeAcp()
    proxy = _FakeProxy()
    m._acp_client = acp
    m._proxy_mcp = proxy
    m._acp_started = True
    m._proxy_started = True
    m._working_dir = "/tmp"

    # Simulate a previous conversation: 2 messages already mirrored, plus 1
    # new HumanMessage that we want delivered as the tail.
    history = [
        HumanMessage(content="msg-0"),
        AIMessage(content="resp-0"),
        HumanMessage(content="msg-1-NEW"),
    ]
    cfg = {"configurable": {"thread_id": "agentZ"}}
    await m.ainvoke(history, config=cfg)

    # The persisted session was RE-ATTACHED via session/load (restart resume),
    # NOT recreated with session/new.
    assert acp.session_new_calls == 0
    assert acp.session_load_calls == ["sid-persisted"]
    assert acp.session_cancel_calls == ["sid-persisted"]
    # The prompt carried ONLY the tail, not the full history.
    assert len(acp.prompts) == 1
    sent_text = acp.prompts[0]["prompt"][0]["text"]
    assert "msg-1-NEW" in sent_text
    assert "msg-0" not in sent_text and "resp-0" not in sent_text
    assert acp.prompts[0]["session_id"] == "sid-persisted"


async def test_failed_session_load_falls_back_to_fresh_session(
    fake_home, monkeypatch
) -> None:
    """If session/load fails (grok pruned the on-disk session), the provider
    must fall back to a FRESH session and re-send the full prompt rather than
    wedge the agent with an empty turn."""
    from langchain_core.messages import AIMessage, HumanMessage

    from deep_worker.providers.grok_provider.chat_model import GrokChatModel

    save_session_state("agentQ", sent_count=2, session_id="sid-gone")

    class _LoadFailsAcp(_FakeAcp):
        async def session_load(self, session_id, *, cwd, mcp_servers=None):
            raise RuntimeError("session not found")

    m = GrokChatModel()
    acp = _LoadFailsAcp()
    m._acp_client = acp
    m._proxy_mcp = _FakeProxy()
    m._acp_started = True
    m._proxy_started = True
    m._working_dir = "/tmp"

    history = [
        HumanMessage(content="msg-0"),
        AIMessage(content="resp-0"),
        HumanMessage(content="msg-1-NEW"),
    ]
    await m.ainvoke(history, config={"configurable": {"thread_id": "agentQ"}})

    # Load was attempted, failed, and we recreated the session.
    assert acp.session_new_calls == 1
    assert len(acp.prompts) == 1
    # The fresh session knows nothing → the FULL history's user text is resent.
    sent_text = acp.prompts[0]["prompt"][0]["text"]
    assert "msg-0" in sent_text and "msg-1-NEW" in sent_text
    # And it used the brand-new session id, not the dead one.
    assert acp.prompts[0]["session_id"] == "fresh-sid"
