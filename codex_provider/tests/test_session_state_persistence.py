"""TDD for persisting per-``agent_thread_id`` Codex session state
(``sent_count`` + ``codex_session_uuid``) across agent restarts.

DESIGN_HISTORY_RESTORE_ON_RESTART.md (codex_provider). Today the Codex
provider keeps ``sent_count`` and ``codex_session_uuid`` only in the
in-memory ``CodexClientState`` singleton, so a restart resets them to 0 /
None and the first ainvoke replays the WHOLE LangGraph history into a brand
new (empty) Codex session — dropping every AIMessage and tool result, so
Codex "forgets" everything but the raw user prompts.

This module exercises:

1. ``codex_provider.session_state`` — the on-disk persistence layer
   (mirror of ``claude_code_provider.session_state``).
2. ``CodexClientState`` lazy-seed on ``get_or_create`` + ``persist()`` +
   persist-on-summarization-reset.
3. ``CodexChatModel.ainvoke`` restart path — first turn after a restart uses
   the persisted ``sent_count`` (small tail, not whole history) and resumes
   the persisted Codex session uuid.
4. The resume fallback — if ``codex exec resume <uuid>`` fails (stale/missing
   session), retry ONCE with a fresh session; a fresh-session spawn failure
   (no resume attempted) still propagates.

All tests are hermetic: no real ``codex`` binary, no network. ``Path.home()``
is monkeypatched to a tmp dir for the on-disk part, and the existing fake
exec-runner pattern from ``test_chat_model.py`` is reused for ainvoke tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider import CodexChatModel  # noqa: E402
from codex_provider.client_state import (  # noqa: E402
    CodexClientState,
    get_global_client_state,
)
from codex_provider.exec_runner import (  # noqa: E402
    AgentMessageItem,
    ItemCompletedEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)
from codex_provider.session_state import (  # noqa: E402
    load_session_state,
    save_session_state,
    session_state_path,
)


# ===========================================================================
# 1. session_state.py — the on-disk persistence layer
# ===========================================================================
class TestSessionStateFile:
    def test_path_strips_tm_and_epoch_suffixes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        # main agent, :tm, and a (defensive) epoch-rotated id all map to the
        # SAME per-agent data dir.
        assert session_state_path("agent_main").parent.name == "agent_main"
        assert session_state_path("agent_main:tm").parent.name == "agent_main"
        assert session_state_path("agent_main__epoch_3").parent.name == "agent_main"
        assert session_state_path("agent_main:tm__epoch_2").parent.name == "agent_main"
        # MUST be the codex-specific filename (no collision with claude's).
        assert session_state_path("agent_main").name == "codex_session_state.json"

    def test_save_then_load_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agentX", sent_count=12, session_uuid="abc-123")
        state = load_session_state("agentX")
        assert state is not None
        assert state["sent_count"] == 12
        assert state["session_uuid"] == "abc-123"

    def test_load_missing_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        assert load_session_state("never_saved") is None

    def test_corrupt_json_returns_none_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        path = session_state_path("agentCorrupt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not valid json ::::")
        # Must degrade to None, never raise.
        assert load_session_state("agentCorrupt") is None

    def test_per_thread_entries_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The main agent and its :tm thread keep SEPARATE sent_count/uuid
        even though they share the same data dir / state file."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agentY", sent_count=5, session_uuid="main-uuid")
        save_session_state("agentY:tm", sent_count=99, session_uuid="tm-uuid")

        main = load_session_state("agentY")
        tm = load_session_state("agentY:tm")
        assert main is not None and tm is not None
        assert main["sent_count"] == 5 and main["session_uuid"] == "main-uuid"
        assert tm["sent_count"] == 99 and tm["session_uuid"] == "tm-uuid"
        # Same underlying file.
        assert session_state_path("agentY") == session_state_path("agentY:tm")

    def test_save_preserves_other_thread_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read-modify-write save must not clobber a sibling thread's
        entry in the same file."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agentZ", sent_count=5, session_uuid="main-uuid")
        save_session_state("agentZ:tm", sent_count=7, session_uuid="tm-uuid")
        # Re-save the main thread — the :tm entry must survive.
        save_session_state("agentZ", sent_count=6, session_uuid="main-uuid-2")

        tm = load_session_state("agentZ:tm")
        assert tm is not None
        assert tm["sent_count"] == 7
        assert tm["session_uuid"] == "tm-uuid"
        main = load_session_state("agentZ")
        assert main is not None
        assert main["sent_count"] == 6 and main["session_uuid"] == "main-uuid-2"

    def test_session_uuid_optional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sent_count can be persisted before we know the session uuid."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agentW", sent_count=3, session_uuid=None)
        state = load_session_state("agentW")
        assert state is not None
        assert state["sent_count"] == 3
        assert state["session_uuid"] is None


# ===========================================================================
# 2. CodexClientState — lazy seed + persist + persist-on-reset
# ===========================================================================
class TestClientStateSeedAndPersist:
    def test_get_or_create_seeds_from_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A FRESH CodexClientState (simulating a restarted process) seeds
        sent_count + codex_session_uuid from disk on first get_or_create."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agent_r", sent_count=8, session_uuid="resume-uuid")

        fresh = CodexClientState()  # brand-new, empty _states
        st = fresh.get_or_create("agent_r")
        assert st.sent_count == 8
        assert st.codex_session_uuid == "resume-uuid"

    def test_get_or_create_no_file_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        fresh = CodexClientState()
        st = fresh.get_or_create("agent_never_persisted")
        assert st.sent_count == 0
        assert st.codex_session_uuid is None
        assert st.current_exec is None

    def test_get_or_create_does_not_reseed_existing_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once a slot exists in memory (hot process), get_or_create must
        NOT re-read the disk and clobber live in-memory values."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agent_hot", sent_count=8, session_uuid="disk-uuid")
        s = CodexClientState()
        first = s.get_or_create("agent_hot")  # seeded from disk
        first.sent_count = 42  # live mutation
        first.codex_session_uuid = "live-uuid"
        again = s.get_or_create("agent_hot")
        assert again is first
        assert again.sent_count == 42
        assert again.codex_session_uuid == "live-uuid"

    def test_persist_round_trips_across_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """persist() writes the slot's current values; a NEW CodexClientState
        (restart) seeds them back via get_or_create."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        before = CodexClientState()
        st = before.get_or_create("agent_p")
        st.sent_count = 17
        st.codex_session_uuid = "persisted-uuid"
        before.persist("agent_p")

        after = CodexClientState()  # restart
        st2 = after.get_or_create("agent_p")
        assert st2.sent_count == 17
        assert st2.codex_session_uuid == "persisted-uuid"

    def test_record_summarization_persists_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a summarization reset, a restart must NOT reload the stale
        pre-summary uuid: the reset is persisted."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        s = CodexClientState()
        st = s.get_or_create("agent_s")
        st.sent_count = 500
        st.codex_session_uuid = "stale-uuid"
        s.persist("agent_s")  # disk now has the pre-summary values

        s.record_summarization("agent_s")  # must also persist the reset

        restarted = CodexClientState()
        st2 = restarted.get_or_create("agent_s")
        assert st2.sent_count == 0
        assert st2.codex_session_uuid is None

    def test_persist_is_best_effort_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """persist() on a slot that does not exist must be a harmless no-op,
        never raise."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        s = CodexClientState()
        s.persist("agent_missing")  # no slot — must not raise


# ===========================================================================
# Fake exec-runner (same contract as test_chat_model.py)
# ===========================================================================
class _FakeExecHandle:
    """Stand-in for ExecHandle that runs the REAL event_pump against a
    scripted async generator (mirrors test_chat_model.py)."""

    def __init__(self, events: list[Any]) -> None:
        import asyncio as _asyncio

        from codex_provider.event_pump import event_pump

        async def _gen():
            for ev in events:
                yield ev

        self._scripted_gen = _gen()
        self.events = self._scripted_gen
        self.ainvoke_queue: _asyncio.Queue[Any] = _asyncio.Queue(maxsize=64)
        self.native_futures: dict[str, _asyncio.Future[str]] = {}
        self.closed = False
        self.pump_error: BaseException | None = None
        self.pump_task: _asyncio.Task[None] = _asyncio.create_task(
            event_pump(self, proxy_mcp_server_name="deepworker_test"),
            name="fake-codex-event-pump",
        )

    async def aclose(self) -> None:
        import asyncio as _asyncio

        self.closed = True
        if not self.pump_task.done():
            self.pump_task.cancel()
            try:
                await self.pump_task
            except (_asyncio.CancelledError, BaseException):
                pass


class _FakeExecRunner:
    """Captures spawn args and returns scripted ExecHandles, one per
    queued event-list (FIFO)."""

    def __init__(self) -> None:
        self.spawn_calls: list[Any] = []
        self.next_events: list[list[Any]] = []

    def queue_events(self, events: list[Any]) -> None:
        self.next_events.append(events)

    async def spawn(
        self, args: Any, *, proxy_mcp_server_name: str | None = None
    ) -> Any:
        _ = proxy_mcp_server_name
        self.spawn_calls.append(args)
        events = self.next_events.pop(0) if self.next_events else []
        return _FakeExecHandle(events)


def _stub_proxy() -> MagicMock:
    proxy = MagicMock()
    proxy.start = AsyncMock()
    proxy.has_pending_calls = MagicMock(return_value=False)
    proxy.tool_names = []
    proxy.server_name = "deepworker_test"
    return proxy


# ===========================================================================
# 3. ainvoke restart path — uses persisted sent_count + resumes uuid
# ===========================================================================
@pytest.mark.asyncio
class TestAinvokeRestartPath:
    async def test_restart_seeds_sent_count_and_resumes_uuid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The crux: after a restart (fresh singleton, but persisted state on
        disk), the first ainvoke must (a) mirror only messages[N:] (not the
        whole history) and (b) spawn with ExecArgs.session_uuid == persisted
        uuid (resume), not a fresh session."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        # Simulate a prior run that persisted state for "agent_x".
        save_session_state("agent_x", sent_count=2, session_uuid="uuid-RESTART")

        # Simulate restart: the process-global singleton is empty.
        get_global_client_state()._states.clear()  # type: ignore[attr-defined]

        runner = _FakeExecRunner()
        runner.queue_events([
            ItemCompletedEvent(item=AgentMessageItem(id="i1", text="resumed ok")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = _stub_proxy()
        m._proxy_started = True

        # History = 3 messages; persisted sent_count = 2 → delta is just msg3.
        await m.ainvoke(
            [
                HumanMessage(content="msg1"),
                AIMessage(content="reply1"),
                HumanMessage(content="msg3"),  # the only unsent tail
            ],
            config={"configurable": {"thread_id": "agent_x"}},
        )

        assert len(runner.spawn_calls) == 1
        args = runner.spawn_calls[0]
        # (b) resume the persisted uuid — NOT a fresh session.
        assert args.session_uuid == "uuid-RESTART"
        # (a) only the unsent tail was mirrored into the prompt.
        assert "msg3" in args.prompt
        assert "msg1" not in args.prompt

    async def test_turn_persists_sent_count_and_uuid_to_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A normal turn writes sent_count + the freshly-learned uuid to disk
        so the NEXT process restart can resume."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        get_global_client_state()._states.clear()  # type: ignore[attr-defined]

        runner = _FakeExecRunner()
        runner.queue_events([
            ThreadStartedEvent(thread_id="codex-uuid-NEW"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="hi")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = _stub_proxy()
        m._proxy_started = True

        await m.ainvoke(
            [HumanMessage(content="hello")],
            config={"configurable": {"thread_id": "agent_persist"}},
        )

        on_disk = load_session_state("agent_persist")
        assert on_disk is not None
        assert on_disk["sent_count"] == 1
        assert on_disk["session_uuid"] == "codex-uuid-NEW"


# ===========================================================================
# 4. Resume fallback — stale/missing session → one fresh-session retry
# ===========================================================================
@pytest.mark.asyncio
class TestResumeFallback:
    async def test_failed_resume_falls_back_to_fresh_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First spawn with a resume uuid yields a turn-failure (stale/missing
        session). The provider must retry ONCE with session_uuid=None, clear +
        persist the reset, and return the fresh session's answer — no error."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        save_session_state("agent_f", sent_count=2, session_uuid="uuid-DEAD")
        get_global_client_state()._states.clear()  # type: ignore[attr-defined]

        runner = _FakeExecRunner()
        # First spawn (resume uuid-DEAD): the session no longer exists.
        runner.queue_events([
            TurnFailedEvent(error_message="session not found: uuid-DEAD"),
        ])
        # Second spawn (fresh, session_uuid=None): succeeds.
        runner.queue_events([
            ThreadStartedEvent(thread_id="uuid-FRESH"),
            ItemCompletedEvent(item=AgentMessageItem(id="i0", text="fresh answer")),
            TurnCompletedEvent(),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = _stub_proxy()
        m._proxy_started = True

        result = await m.ainvoke(
            [
                HumanMessage(content="msg1"),
                AIMessage(content="reply1"),
                HumanMessage(content="msg3"),
            ],
            config={"configurable": {"thread_id": "agent_f"}},
        )

        # Two spawns happened: the failed resume + the fresh retry.
        assert len(runner.spawn_calls) == 2
        # First carried the (dead) resume uuid; second was a fresh session.
        assert runner.spawn_calls[0].session_uuid == "uuid-DEAD"
        assert runner.spawn_calls[1].session_uuid is None
        # The fallback replayed the FULL history into the fresh session,
        # not just the post-restart tail.
        assert "msg1" in runner.spawn_calls[1].prompt
        assert "msg3" in runner.spawn_calls[1].prompt
        # No exception — the fresh session's text is returned.
        assert isinstance(result, AIMessage)
        assert "fresh answer" in result.content
        # The dead uuid was cleared and the reset persisted to disk.
        st = get_global_client_state().get("agent_f")
        assert st is not None
        assert st.codex_session_uuid == "uuid-FRESH"
        on_disk = load_session_state("agent_f")
        assert on_disk is not None
        assert on_disk["session_uuid"] == "uuid-FRESH"

    async def test_fresh_session_failure_still_raises_no_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh-session (no resume) turn failure is a genuine error — it
        must propagate, with NO spurious second spawn."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]
        get_global_client_state()._states.clear()  # type: ignore[attr-defined]

        runner = _FakeExecRunner()
        runner.queue_events([
            TurnFailedEvent(error_message="real API error 500"),
        ])
        m = CodexChatModel()
        m._exec_runner = runner
        m._proxy_mcp = _stub_proxy()
        m._proxy_started = True

        with pytest.raises(RuntimeError, match="real API error 500"):
            await m.ainvoke(
                [HumanMessage(content="hello")],  # no persisted uuid → fresh
                config={"configurable": {"thread_id": "agent_nofresh"}},
            )
        # No retry — only the single fresh spawn happened.
        assert len(runner.spawn_calls) == 1
        assert runner.spawn_calls[0].session_uuid is None
