"""TDD for ainvoke-side restart resilience for grok_provider.

Since 3fe834d1 the grok provider inherits the shared ACP layer, so the state
container is ``acp.client_state.AcpClientState`` and the on-disk bridge is
``acp_session_state.json`` (covered by test_session_state.py). The provider
keeps ``(session_id, sent_count)`` only in the in-memory ``AcpClientState``
singleton; the file is the cross-restart bridge. This module pins:

1. AcpClientState.get_or_create lazily seeds (sent_count, session_id)
   from disk on the first access for a thread.
2. After the seed, ainvoke uses the persisted session_id (no fresh
   session/new) and only mirrors the tail messages[sent_count:] — NOT the
   whole history.
3. persist() writes both fields atomically (mirrors test_session_state.py
   filesystem invariants — see that file for the on-disk side).
4. record_summarization persists the reset so a restart that lands right
   after summarization sees a fresh slot.

Persisted entries are scoped by ``provider`` + session ``fingerprint``, which
the model supplies to the state container via ``set_profile`` at init; these
tests do the same so the seeding path under test is the production one.

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

from deep_worker.providers.acp.client_state import AcpClientState  # noqa: E402
from deep_worker.providers.acp.session_state import (  # noqa: E402
    load_session_state,
    save_session_state,
)
from deep_worker.providers.grok_provider.profile import (  # noqa: E402
    build_profile,
)

# The provider id + session fingerprint the grok provider really scopes its
# persisted state with (see chat_model: set_profile(profile.provider,
# profile.fingerprint())). Read from the live profile, not hardcoded.
PROVIDER = build_profile().provider
FINGERPRINT = build_profile().fingerprint()


def _new_client_state() -> AcpClientState:
    """A fresh state container profiled exactly as the model profiles it."""
    cs = AcpClientState()
    cs.set_profile(PROVIDER, FINGERPRINT)
    return cs


def _save(thread_id: str, *, sent_count: int, session_id: str | None) -> None:
    save_session_state(
        thread_id,
        provider=PROVIDER,
        session_id=session_id,
        sent_count=sent_count,
        fingerprint=FINGERPRINT,
    )


def _load(thread_id: str) -> dict | None:
    return load_session_state(
        thread_id, provider=PROVIDER, fingerprint=FINGERPRINT
    )


def _core(loaded: dict | None) -> dict | None:
    """The (sent_count, session_id) pair these tests are about.

    ``load_session_state`` also reports the context gauge
    (``last_response_at`` / ``last_context_used`` / ``model_id``), which this
    module does not exercise.
    """
    if loaded is None:
        return None
    return {"sent_count": loaded["sent_count"], "session_id": loaded["session_id"]}


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    yield tmp_path


@pytest.fixture
def isolated_global_state(monkeypatch):
    """Give each test its OWN process-global client state.

    ``AcpChatModel.__init__`` grabs the module-level singleton, so without
    this a slot seeded by one test would survive into the next one and the
    disk-seeding path under test would be skipped.
    """
    import deep_worker.providers.acp.client_state as cs_mod

    monkeypatch.setattr(cs_mod, "_GLOBAL_CLIENT_STATE", None)
    return cs_mod


class TestLazySeedFromDisk:
    def test_get_or_create_seeds_from_persisted_state(self, fake_home) -> None:
        # Simulate a previous process having persisted state.
        _save(
            "agentA",
            sent_count=7,
            session_id="019ed7a0-afa8-7730-86b8-6a5817010dca",
        )
        cs = _new_client_state()
        st = cs.get_or_create("agentA")
        assert st.sent_count == 7
        assert st.session_id == "019ed7a0-afa8-7730-86b8-6a5817010dca"

    def test_fresh_thread_starts_empty(self, fake_home) -> None:
        cs = _new_client_state()
        st = cs.get_or_create("never-saw-this")
        assert st.sent_count == 0
        assert st.session_id is None

    def test_hot_process_never_reseeds(self, fake_home) -> None:
        """Once a slot exists in memory, a later get_or_create must NOT
        re-read the disk and clobber live in-memory state."""
        cs = _new_client_state()
        st1 = cs.get_or_create("agentA")
        st1.sent_count = 99
        st1.session_id = "in-mem-only"

        _save("agentA", sent_count=1, session_id="disk-only")

        st2 = cs.get_or_create("agentA")
        # SAME object, untouched.
        assert st2 is st1
        assert st2.sent_count == 99
        assert st2.session_id == "in-mem-only"

    def test_persist_writes_through(self, fake_home) -> None:
        cs = _new_client_state()
        st = cs.get_or_create("agentB")
        st.sent_count = 4
        st.session_id = "abc-123"
        cs.persist("agentB")

        assert _core(_load("agentB")) == {"sent_count": 4, "session_id": "abc-123"}

    def test_persist_unknown_thread_is_noop(self, fake_home) -> None:
        cs = _new_client_state()
        # Slot does not exist. Must not raise, must not write a stale entry.
        cs.persist("never-saw-this")
        assert _load("never-saw-this") is None

    def test_record_summarization_persists_reset(self, fake_home) -> None:
        """If a restart happens right after summarization, the disk file must
        reflect (0, None) so the next ainvoke doesn't try to resume the
        already-dropped CLI session id."""
        _save("agentC", sent_count=12, session_id="will-be-dropped")
        cs = _new_client_state()
        cs.get_or_create("agentC")  # seed from disk
        cs.record_summarization("agentC")

        assert _core(_load("agentC")) == {"sent_count": 0, "session_id": None}


class TestSessionIndexRoundTrip:
    def test_register_and_resolve(self, fake_home) -> None:
        cs = _new_client_state()
        cs.get_or_create("agentX")
        cs.register_session("sid-1", "agentX")
        # The reverse lookup returns the SAME state object.
        assert cs.thread_for_session("sid-1") is cs.get_or_create("agentX")

    def test_summarization_evicts_stale_session_from_index(
        self, fake_home
    ) -> None:
        """L3 — record_summarization MUST evict the stale session_id
        from _session_index so a late session/update never routes into a
        dead session (it would leak into the wrong ThreadState)."""
        cs = _new_client_state()
        st = cs.get_or_create("agentY")
        st.session_id = "sid-old"
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
        self.session_close_calls: list[str] = []

    async def ensure_started(self) -> None:
        # The shared layer drives CLI startup through the client; the fake is
        # "already started" so there is nothing to spawn.
        return None

    async def session_new(
        self, *, cwd: str, mcp_servers: list[dict], meta: dict | None = None
    ) -> str:
        self.session_new_calls += 1
        return "fresh-sid"

    async def session_load(
        self, session_id: str, *, cwd: str, mcp_servers=None, meta=None
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

    async def session_close(self, session_id: str) -> bool:
        self.session_close_calls.append(session_id)
        return True


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


def _wire_fakes(model, acp) -> None:
    """Replace the model's heavy I/O surfaces with the fakes."""
    model._acp_client = acp
    model._proxy_mcp = _FakeProxy()
    model._acp_started = True
    model._proxy_started = True
    model._working_dir = "/tmp"


async def test_seeded_sent_count_short_circuits_history_replay(
    fake_home, isolated_global_state, monkeypatch
) -> None:
    """After a restart, the next ainvoke must:
       - RE-ATTACH the persisted session_id via session/load (NOT a
         fresh session/new — grok keeps sessions in-memory, so a persisted id
         is dead in the new subprocess until reloaded), AND
       - send only messages[sent_count:] (the tail), NOT the full history.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from deep_worker.providers.grok_provider.chat_model import GrokChatModel

    _save("agentZ", sent_count=2, session_id="sid-persisted")

    m = GrokChatModel()
    acp = _FakeAcp()
    _wire_fakes(m, acp)

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
    fake_home, isolated_global_state, monkeypatch
) -> None:
    """If session/load fails (grok pruned the on-disk session), the provider
    must fall back to a FRESH session and re-send the full prompt rather than
    wedge the agent with an empty turn."""
    from langchain_core.messages import AIMessage, HumanMessage

    from deep_worker.providers.grok_provider.chat_model import GrokChatModel

    _save("agentQ", sent_count=2, session_id="sid-gone")

    class _LoadFailsAcp(_FakeAcp):
        async def session_load(
            self, session_id, *, cwd, mcp_servers=None, meta=None
        ):
            raise RuntimeError("session not found")

    m = GrokChatModel()
    acp = _LoadFailsAcp()
    _wire_fakes(m, acp)

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
