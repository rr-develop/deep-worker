"""grok: a context reading per model call, not one at the end of the turn.

Owner directive 2026-08-25: "All providers must track live data, and not
only at the very end, but after every call to the model."

Why grok needed its own source. It sends no ``usage_update`` (live capture:
a whole turn yields only user_message_chunk / agent_thought_chunk /
agent_message_chunk / available_commands_update), and its ``_meta`` arrives
only with the ``session/prompt`` REPLY — i.e. once, at the very end of a
turn that may span a dozen model calls. Measured on a live agent: eight
inferences inside one turn while the recorded context did not move at all,
climbing 260740 → 274277 unseen — past the operator's ``-l 250000``.

The number is available live in grok's own telemetry: ``~/.grok/logs/
unified.jsonl`` appends ``shell.turn.inference_done`` per model call with
``ctx.prompt_tokens`` (the full prompt, cached prefix included) and the
session id in ``sid`` — which is the same uuid our ACP client gets from
``session/new`` (verified live: 01a039b6-82f7-…). One shared file carries
every grok session on the host, so filtering by ``sid`` is mandatory.

Same shape as the codex fix: driven by filesystem change events, never a
timer, and only the appended bytes are parsed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from deep_worker.providers.acp.context_watch import (
    grok_live_log_path,
    parse_grok_inferences,
)
from deep_worker_cli.context_limit.live_gauge import LiveContextWatcher

SID = "01a039b6-82f7-7283-842d-79ac6874c2c5"
OTHER = "01a03000-0000-0000-0000-000000000000"


def _inference(sid: str, prompt: int, cached: int = 0) -> str:
    return json.dumps(
        {
            "ts": "2026-08-25T17:47:44.000Z",
            "src": "shell",
            "sid": sid,
            "msg": "shell.turn.inference_done",
            "ctx": {
                "loop_index": 1,
                "prompt_tokens": prompt,
                "cached_prompt_tokens": cached,
                "completion_tokens": 12,
            },
        }
    )


def _noise(sid: str = SID) -> str:
    return json.dumps({"ts": "2026-08-25T17:47:00.000Z", "sid": sid, "msg": "shell.tool.exec_done", "ctx": {"tool_name": "grep"}})


class TestParsing:
    def test_reads_prompt_tokens_of_our_session(self) -> None:
        chunk = "\n".join([_noise(), _inference(SID, 260740)])
        assert parse_grok_inferences(chunk, SID) == [260740]

    def test_other_sessions_are_not_ours(self) -> None:
        """The log is host-wide: every grok agent and every sub-session
        writes into it. Reading someone else's number would report another
        agent's context as this agent's."""
        chunk = "\n".join([_inference(OTHER, 999_999), _inference(SID, 100)])
        assert parse_grok_inferences(chunk, SID) == [100]

    def test_order_is_preserved(self) -> None:
        chunk = "\n".join(
            [_inference(SID, 260740), _inference(SID, 262944), _inference(SID, 265768)]
        )
        assert parse_grok_inferences(chunk, SID) == [260740, 262944, 265768]

    def test_partial_tail_and_garbage_survive(self) -> None:
        chunk = _inference(SID, 5) + "\n" + '{"ts": "2026-08'
        assert parse_grok_inferences(chunk, SID) == [5]
        assert parse_grok_inferences("nonsense", SID) == []
        assert parse_grok_inferences("", SID) == []

    def test_cached_prefix_is_not_the_context(self) -> None:
        """``prompt_tokens`` already includes the cached prefix; reporting
        ``cached_prompt_tokens`` instead would under-report a warm session."""
        chunk = _inference(SID, 274277, cached=273536)
        assert parse_grok_inferences(chunk, SID) == [274277]

    def test_missing_session_id_reports_nothing(self) -> None:
        assert parse_grok_inferences(_inference(SID, 10), "") == []


class TestLogPath:
    def test_defaults_to_the_grok_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROK_HOME", raising=False)
        p = grok_live_log_path()
        assert p is not None
        assert p.name == "unified.jsonl"
        assert p.parent.name == "logs"

    def test_honours_grok_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """The agent may run with GROK_HOME pointing elsewhere (that is how
        credentials survive the test sandbox), so the gauge must follow it."""
        monkeypatch.setenv("GROK_HOME", str(tmp_path))
        assert grok_live_log_path() == tmp_path / "logs" / "unified.jsonl"


class _FakeChanges:
    def __init__(self) -> None:
        self._q: asyncio.Queue[object] = asyncio.Queue()

    def signal(self) -> None:
        self._q.put_nowait(object())

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._q.get()


@pytest.mark.asyncio
class TestLiveWatcher:
    async def test_one_reading_per_model_call(self, tmp_path) -> None:
        path = tmp_path / "unified.jsonl"
        path.write_text("", encoding="utf-8")
        seen: list[int] = []

        async def on_context(used: int, total: int | None) -> None:
            seen.append(used)

        changes = _FakeChanges()
        watcher = LiveContextWatcher(
            path=path,
            parse=lambda chunk: parse_grok_inferences(chunk, SID),
            on_context=on_context,
            changes_factory=lambda _p: changes,
        )
        await watcher.start()
        try:
            for value in (260740, 262944, 265768):
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(_inference(SID, value) + "\n")
                changes.signal()
                await asyncio.sleep(0.05)
        finally:
            await watcher.stop()
        assert seen == [260740, 262944, 265768]

    async def test_the_last_call_of_a_short_turn_is_not_lost(
        self, tmp_path
    ) -> None:
        """Change events are debounced and a turn can end milliseconds after
        its last model call — measured live on grok, where the reading was
        appended, the prompt reply arrived, the turn ended and the watcher
        was cancelled before waking. ``stop`` must flush."""
        path = tmp_path / "unified.jsonl"
        path.write_text("", encoding="utf-8")
        seen: list[int] = []

        async def on_context(used: int, total: int | None) -> None:
            seen.append(used)

        changes = _FakeChanges()  # never signalled: the turn ends first
        watcher = LiveContextWatcher(
            path=path,
            parse=lambda chunk: parse_grok_inferences(chunk, SID),
            on_context=on_context,
            changes_factory=lambda _p: changes,
        )
        await watcher.start()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(_inference(SID, 274277) + "\n")
        await watcher.stop()
        assert seen == [274277]

    async def test_history_on_disk_is_not_replayed(self, tmp_path) -> None:
        """The file holds every session ever run on this host."""
        path = tmp_path / "unified.jsonl"
        path.write_text(_inference(SID, 111) + "\n", encoding="utf-8")
        seen: list[int] = []

        async def on_context(used: int, total: int | None) -> None:
            seen.append(used)

        changes = _FakeChanges()
        watcher = LiveContextWatcher(
            path=path,
            parse=lambda chunk: parse_grok_inferences(chunk, SID),
            on_context=on_context,
            changes_factory=lambda _p: changes,
        )
        await watcher.start()
        try:
            changes.signal()
            await asyncio.sleep(0.05)
            assert seen == []
        finally:
            await watcher.stop()


class TestAcpWiring:
    def test_the_gauge_runs_for_the_whole_turn(self) -> None:
        """Started when the turn begins, stopped in its ``finally`` — a
        cancelled turn must leak neither the task nor its inotify handle."""
        import inspect

        from deep_worker.providers.acp.chat_model import AcpChatModel

        consume = inspect.getsource(AcpChatModel._consume_turn)
        assert "await self._start_live_context_watch(state)" in consume
        finally_tail = consume.split("finally:")[-1]
        assert "_stop_live_context_watch" in finally_tail

    async def test_a_live_reading_updates_the_gauge_the_summarizer_reads(
        self,
    ) -> None:
        """The floor must move per model call, not once per turn.

        ``provider_context_token_counter(floor=...)`` and the context guard
        both read ``client_state.last_context_used``. The summarizer runs
        ``abefore_model``, i.e. BEFORE every model call inside a turn — so a
        gauge that only updates when the turn ends leaves it deciding on a
        number that is one whole turn stale. Measured on grok: 98 model
        calls in a single turn, 18525 -> 97368 tokens and beyond.
        """
        from deep_worker.providers.acp import chat_model as acp_cm
        from deep_worker.providers.acp.custom_types import ThreadState

        model = object.__new__(acp_cm.AcpChatModel)
        state = ThreadState()
        state.session_id = SID
        seen: list[int] = []

        async def _emit(used, *, session_id):  # noqa: ANN001
            seen.append(used)

        object.__setattr__(model, "_emit_context_snapshot", _emit)
        object.__setattr__(
            model,
            "_profile",
            type("P", (), {"live_context_source": "grok_unified_log"})(),
        )
        callback = acp_cm.AcpChatModel._live_context_callback(model, state, SID)
        await callback(120_000, None)

        assert state.live_context_used == 120_000
        assert state.last_context_used == 120_000, (
            "the live reading never reached the gauge the summarizer and the "
            f"guard read (last_context_used={state.last_context_used!r}): "
            "both decide on a turn-old number"
        )
        assert seen == [120_000]

    def test_grok_profile_declares_the_source(self) -> None:
        from deep_worker.providers.grok_provider.profile import build_profile

        assert build_profile().live_context_source == "grok_unified_log"

    def test_claude_code_needs_no_log_watcher(self) -> None:
        """claude-agent-acp sends usage_update per call already; a second
        source would be duplicate machinery, not extra safety."""
        from deep_worker.providers.acp.profile import AcpProfile

        assert AcpProfile(provider="x", cmd=("x",)).live_context_source is None

    def test_live_reading_is_used_for_the_stamp(self) -> None:
        """The stamp must carry the FRESHEST provider number, not the one
        that happened to arrive with the prompt reply."""
        from langchain_core.messages import AIMessage

        from deep_worker.providers.acp.chat_model import AcpChatModel
        from deep_worker.providers.acp.custom_types import ThreadState
        from deep_worker_cli.context_limit.observed_context import (
            context_observation,
        )

        state = ThreadState()
        state.live_context_used = 274_277
        msg = AcpChatModel._stamp_context_observation(
            object.__new__(AcpChatModel),
            state,
            AIMessage(content="ok"),
            meta={"inputTokens": 260_740},
        )
        obs = context_observation(msg)
        assert obs is not None
        assert obs["used"] == 274_277
