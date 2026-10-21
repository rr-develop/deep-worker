"""codex: the context gauge must move on every model call, not once a turn.

Owner report 2026-08-25: agent_a showed ``Context: 40.9k`` frozen while
the agent worked, then jumped to ``71.7k``. Both numbers are real records —
``prompt_tokens`` 40884 at 12:16:19 and 71716 at 12:25:51, 9.5 minutes
apart. The provider emits usage in exactly one place, ``TurnCompletedEvent``,
and a codex *turn* is not a model call: it is one prompt → final answer,
covering the whole internal loop. Measured on agent_a's own rollouts: 48
``token_count`` events across 4 ``task_complete`` — ~12 model calls per turn
whose context growth (18k → 21k → 26k → 27k → 28k …) the UI never saw.

The transport cannot help: on codex-cli 0.149.1 both ``--json`` and
``--experimental-json`` carry usage ONLY in ``turn.completed`` (verified
live — a 3-step turn produced 3 item.started / 5 item.completed and zero
token-bearing events until the end). codex's app-server protocol *does*
have ``thread/tokenUsage/updated``, but that is a different transport.

So the live number comes from the rollout codex writes as it goes: one
``token_count`` per model call, carrying ``info.last_token_usage`` (this
call's input = the context it read) and ``info.total_token_usage``.
Verified live that the file is appended DURING a turn (agent_a: +1664
bytes and one new event in 45 s).

Owner constraint: event-driven, no polling. The watcher below is driven by
filesystem change events (``watchfiles``, already a declared dependency and
already the LSP change-detector's mechanism); the tests inject the change
stream so they never touch inotify or a real codex.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from deep_worker.providers.codex_provider.context_watch import (
    RolloutContextWatcher,
    latest_context_used,
    parse_token_counts,
)


def _token_count_line(last_input: int, total_input: int, ts: str = "2026-08-25T12:00:00.000Z") -> str:
    return json.dumps(
        {
            "timestamp": ts,
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": last_input,
                        "cached_input_tokens": 0,
                        "output_tokens": 12,
                    },
                    "total_token_usage": {"input_tokens": total_input},
                },
            },
        }
    )


def _other_line(kind: str = "reasoning") -> str:
    return json.dumps({"timestamp": "2026-08-25T12:00:00.000Z", "payload": {"type": kind}})


class TestParsing:
    def test_reads_the_per_call_input_tokens(self) -> None:
        chunk = "\n".join([_other_line(), _token_count_line(18028, 18028)])
        counts = parse_token_counts(chunk)
        assert [c.context_used for c in counts] == [18028]
        assert counts[0].total_input == 18028

    def test_several_calls_in_one_chunk_keep_their_order(self) -> None:
        chunk = "\n".join(
            [
                _token_count_line(18028, 18028),
                _other_line("agent_message"),
                _token_count_line(21089, 39117),
                _token_count_line(26161, 65278),
            ]
        )
        assert [c.context_used for c in parse_token_counts(chunk)] == [
            18028,
            21089,
            26161,
        ]

    def test_latest_wins(self) -> None:
        chunk = "\n".join([_token_count_line(1, 1), _token_count_line(7, 8)])
        assert latest_context_used(chunk) == 7

    def test_garbage_and_partial_lines_are_survivable(self) -> None:
        """The file is being appended while we read: the tail can be half a
        line. A partial line must never raise and never be mistaken for a
        measurement."""
        chunk = _token_count_line(5, 5) + "\n" + '{"timestamp": "2026-08'
        assert latest_context_used(chunk) == 5
        assert latest_context_used("not json at all") is None
        assert latest_context_used("") is None

    def test_a_turn_total_is_not_mistaken_for_the_context(self) -> None:
        """``total_token_usage`` is the cumulative spend of the exec process
        (turn.completed reports the same sum: 5 calls of ~21.5k came back as
        108058). Reporting it as context is the 2026-08-22 "507.3k vs -l
        250k" defect."""
        chunk = _token_count_line(21770, 108058)
        assert latest_context_used(chunk) == 21770


class _FakeChanges:
    """Stands in for ``watchfiles.awatch``: yields once per queued signal."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self.stopped = False

    def signal(self) -> None:
        self._queue.put_nowait(object())

    def __aiter__(self) -> "_FakeChanges":
        return self

    async def __anext__(self) -> object:
        item = await self._queue.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item


@pytest.mark.asyncio
class TestWatcher:
    async def _watcher(self, tmp_path, changes, seen):
        path = tmp_path / "rollout-01a038d1.jsonl"
        path.write_text("", encoding="utf-8")

        async def on_context(used: int, total: int | None) -> None:
            seen.append((used, total))

        watcher = RolloutContextWatcher(
            path=path,
            on_context=on_context,
            changes_factory=lambda _p: changes,
        )
        return path, watcher

    async def test_emits_once_per_model_call(self, tmp_path) -> None:
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()
        path, watcher = await self._watcher(tmp_path, changes, seen)
        await watcher.start()
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(_token_count_line(18028, 18028) + "\n")
            changes.signal()
            await asyncio.sleep(0.05)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(_token_count_line(21089, 39117) + "\n")
            changes.signal()
            await asyncio.sleep(0.05)
        finally:
            await watcher.stop()
        assert [u for u, _ in seen] == [18028, 21089]

    async def test_the_last_call_of_a_short_turn_is_not_lost(
        self, tmp_path
    ) -> None:
        """``stop`` must flush before releasing the handle.

        Live on codex (2026-08-26, one-word prompt through the provider):
        the turn produced ZERO readings and an unstamped message, because a
        change event is debounced and the turn ended before the watcher woke
        — the same defect already fixed in the ACP gauge. Since the codex
        stamp is built from ``last_provider_context`` AFTER the ``finally``
        that stops the watch, a missing flush costs the stamp too: guard and
        summarizer see UNKNOWN for the whole turn.
        """
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()  # never signalled: the turn ends first
        path, watcher = await self._watcher(tmp_path, changes, seen)
        await watcher.start()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(_token_count_line(21089, 39117) + "\n")
        await watcher.stop()
        assert [u for u, _ in seen] == [21089], (
            f"the last model call of the turn was lost: {seen}"
        )

    async def test_a_burst_in_one_event_reports_every_call(
        self, tmp_path
    ) -> None:
        """Two model calls between two wake-ups are two readings, not one.

        Tool round-trips are fast and the change stream is debounced at
        200 ms, so several ``token_count`` lines routinely land in a single
        event. Reporting only the newest silently drops model calls — the
        very thing the cadence rule forbids — and it is invisible in any
        fixture that signals after every single append.
        """
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()
        path, watcher = await self._watcher(tmp_path, changes, seen)
        await watcher.start()
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(_token_count_line(18028, 18028) + "\n")
                fh.write(_token_count_line(21089, 39117) + "\n")
                fh.write(_token_count_line(24500, 63617) + "\n")
            changes.signal()
            await asyncio.sleep(0.05)
        finally:
            await watcher.stop()
        assert [u for u, _ in seen] == [18028, 21089, 24500], (
            f"a burst collapsed into {seen}: model calls were dropped"
        )

    async def test_history_already_on_disk_is_never_replayed(

        self, tmp_path
    ) -> None:
        """``start`` takes end-of-file as the baseline.

        A session's rollout holds EVERY earlier turn (agent_a: 906 KB and
        growing). Replaying it would publish a stale reading as if it were
        happening now, and re-reading megabytes per event is the polling cost
        we are avoiding, merely hidden. A change event that adds no new
        ``token_count`` must therefore emit NOTHING."""
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()
        path, watcher = await self._watcher(tmp_path, changes, seen)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(_token_count_line(100, 100) + "\n")
        await watcher.start()
        try:
            # Something touched the file but appended no measurement.
            changes.signal()
            await asyncio.sleep(0.05)
            assert seen == [], "pre-existing history was replayed"

            with path.open("a", encoding="utf-8") as fh:
                fh.write(_token_count_line(200, 300) + "\n")
            changes.signal()
            await asyncio.sleep(0.05)
        finally:
            await watcher.stop()
        assert [u for u, _ in seen] == [200]

    async def test_unchanged_value_is_not_re_emitted(self, tmp_path) -> None:
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()
        path, watcher = await self._watcher(tmp_path, changes, seen)
        await watcher.start()
        try:
            for _ in range(3):
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(_token_count_line(4242, 4242) + "\n")
                changes.signal()
                await asyncio.sleep(0.04)
        finally:
            await watcher.stop()
        assert [u for u, _ in seen] == [4242]

    async def test_a_failing_callback_never_kills_the_watcher(
        self, tmp_path
    ) -> None:
        """Telemetry must never take the turn down with it."""
        calls: list[int] = []

        path = tmp_path / "rollout-x.jsonl"
        path.write_text("", encoding="utf-8")

        async def on_context(used: int, total: int | None) -> None:
            calls.append(used)
            raise RuntimeError("stats backend down")

        changes = _FakeChanges()
        watcher = RolloutContextWatcher(
            path=path, on_context=on_context, changes_factory=lambda _p: changes
        )
        await watcher.start()
        try:
            for value in (11, 22):
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(_token_count_line(value, value) + "\n")
                changes.signal()
                await asyncio.sleep(0.04)
        finally:
            await watcher.stop()
        assert calls == [11, 22]

    async def test_stop_is_idempotent_and_leaves_no_task(self, tmp_path) -> None:
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()
        _path, watcher = await self._watcher(tmp_path, changes, seen)
        await watcher.start()
        await watcher.stop()
        await watcher.stop()
        assert watcher.task is None or watcher.task.done()

    async def test_missing_file_is_not_fatal(self, tmp_path) -> None:
        """``--ephemeral`` sessions and pruned rollouts have no file; the
        turn must run exactly as before, just without a live gauge."""
        seen: list[tuple[int, int | None]] = []
        changes = _FakeChanges()

        async def on_context(used: int, total: int | None) -> None:
            seen.append((used, total))

        watcher = RolloutContextWatcher(
            path=tmp_path / "nope.jsonl",
            on_context=on_context,
            changes_factory=lambda _p: changes,
        )
        await watcher.start()
        changes.signal()
        await asyncio.sleep(0.04)
        await watcher.stop()
        assert seen == []


class TestChatModelWiring:
    def test_the_watch_is_retried_after_the_rollout_appears(self) -> None:
        """RED, live 2026-08-26: the gauge never started at all.

        ``rollout_path_for`` resolves by globbing for the session uuid, and
        codex creates that file a beat AFTER announcing ``thread.started``.
        Both existing call sites run at or before that announcement, so on
        this host every probe resolved ``None``, returned silently, and the
        turn produced zero readings and an unstamped message — the guard and
        the summarizer saw UNKNOWN for the whole turn. It is a race, which
        is why longer turns sometimes looked fine.

        The loop must therefore keep retrying while the watch is unset. The
        pin requires a call reached for ANY event, not one nested inside the
        ``ThreadStartedEvent`` branch — that branch is exactly the site that
        already exists and already fires too early.
        """
        import ast
        import inspect
        import textwrap

        from deep_worker.providers.codex_provider.chat_model import CodexChatModel

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(CodexChatModel._consume_turn))
        )
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
        assert loops, "the event loop of _consume_turn was restructured"

        def _is_thread_started_branch(node) -> bool:
            if not isinstance(node, ast.If):
                return False
            test = node.test
            return (
                isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == "isinstance"
                and len(test.args) == 2
                and getattr(test.args[1], "id", "") == "ThreadStartedEvent"
            )

        unconditional = 0
        for loop in loops:
            for stmt in loop.body:
                if _is_thread_started_branch(stmt):
                    continue  # the site that already fires too early
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "_start_context_watch"
                    ):
                        unconditional += 1
        assert unconditional, (
            "_consume_turn only starts the context watch from the "
            "ThreadStartedEvent branch (and before the loop). Codex has not "
            "written the rollout yet at that moment, so the watch resolves "
            "None and the whole turn goes unmeasured"
        )

    def test_snapshot_emission_carries_no_turn_id(self) -> None:
        """codex records are de-duplicated on ``session_id|turn_id``
        (token_stats.UsageEventCallback._event_key). A mid-turn snapshot that
        reused the turn id would be swallowed as a duplicate — every update
        after the first would vanish, which is precisely the frozen gauge
        being fixed. Asserted on the CALL, not on the source text: the
        rationale lives in the docstring, so a source grep passes even when
        the argument is wrong."""
        from deep_worker.providers.codex_provider.chat_model import (
            CodexChatModel,
        )

        captured: list[dict] = []

        async def _cb(**kwargs) -> None:
            captured.append(kwargs)

        model = CodexChatModel.__new__(CodexChatModel)
        object.__setattr__(model, "_usage_event_callback", _cb)
        object.__setattr__(model, "_resolved_model_name", "gpt-5.6-sol")

        asyncio.run(
            model._emit_context_snapshot(4242, 99999, session_id="sess-1")
        )

        assert len(captured) == 1
        call = captured[0]
        assert call["turn_id"] is None
        assert call["context_used"] == 4242
        assert call["session_id"] == "sess-1"
        # usage must stay None: a snapshot is not a token record, and a
        # non-None usage would trigger the rollout override path.
        assert call["usage"] is None
        # The cumulative total must NOT leak into the context field.
        assert call.get("context_size") in (None, 0)

    def test_emit_usage_event_accepts_context_fields(self) -> None:
        import inspect

        from deep_worker.providers.codex_provider.chat_model import (
            CodexChatModel,
        )

        params = inspect.signature(CodexChatModel._emit_usage_event).parameters
        assert "context_used" in params
        assert "context_size" in params

    def test_watcher_is_started_and_stopped_around_the_turn(self) -> None:
        import inspect

        from deep_worker.providers.codex_provider.chat_model import (
            CodexChatModel,
        )

        src = inspect.getsource(CodexChatModel)
        assert "_start_context_watch" in src
        assert "_stop_context_watch" in src
        # The stop must be in a finally — a cancelled turn must not leak the
        # watcher task or its inotify handle.
        consume = inspect.getsource(CodexChatModel._consume_turn)
        finally_tail = consume.split("finally:")[-1]
        assert "_stop_context_watch" in finally_tail
