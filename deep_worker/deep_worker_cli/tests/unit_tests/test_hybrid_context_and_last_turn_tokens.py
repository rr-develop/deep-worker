"""Regression: an ACP-style terminal event carrying BOTH ``usage`` AND
``context_used/size`` must record BOTH — the snapshot goes into
``context_used/size`` (semantics=context_snapshot) so the UI Context
field shows a value comparable to ``-l``, and the per-turn total lands
in ``total_tokens`` on the SAME record so the UI can render it as
"Last turn" separately.

The 2026-08-22 defect: claude-code
on ``-l 400k`` showed ``Context: 2265.4k tokens`` while the operator's
cap was 400k. Root cause traced by chat_model._finalize_response:

  * when ``_meta.totalTokens`` was present, the provider emitted a
    tokens-semantics record with total_tokens=2265k AND cleared
    ``state.usage_snapshot``;
  * the actual observed context from ``usage_update`` (used=390k,
    size=1_000_000) was dropped on the floor;
  * ``latest_context_by_kind`` then fell back to ``total_tokens`` for
    the ``context_used`` field, so the UI painted "2265.4k tokens" as
    the Context — a per-turn number, nonsensical vs ``-l 400k``.

Fix contract:

  1. The event callback MUST record BOTH a context snapshot AND per-turn
     token totals in ONE record when the provider hands over both.
  2. ``latest_context_by_kind`` picks up the honest snapshot (not the
     token fallback) whenever the record carries ``context_used`` /
     ``context_size``, AND the record's ``total_tokens`` is preserved
     for the UI to render as "Last turn".
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from deep_worker_cli.stats.token_stats import (
    KIND_AGENT,
    STATUS_OK,
    TokenUsageRecorder,
    USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
    latest_context_by_kind,
    read_stats_file,
)


class TestHybridSnapshotAndPerTurnTokens:
    def test_snapshot_plus_usage_records_both_on_one_record(
        self, tmp_path: Path
    ) -> None:
        """Simulate the ACP finalize path: the terminal event fires with
        both ``usage`` and ``context_used/size`` in the same call.  The
        recorder must persist ONE row with both semantics: a
        context_snapshot for the UI Context field AND per-turn
        total_tokens for the UI "Last turn" row.
        """
        recorder = TokenUsageRecorder(tmp_path)
        cb = recorder.make_usage_event_callback(
            provider="claude-code",
            kind=KIND_AGENT,
            model_name="fable",
        )

        async def _fire() -> None:
            await cb(event="invocation_start")
            # ACP finalize path (chat_model._finalize_response) fires both
            # numbers in ONE terminal call now that the fix routes them
            # together. Per-turn totals come from _meta; snapshot from
            # usage_update.
            await cb(
                event="terminal",
                status=STATUS_OK,
                usage={
                    "input_tokens": 2_200_000,
                    "output_tokens": 65_400,
                    "cached_input_tokens": 1_800_000,
                    "total_tokens": 2_265_400,
                },
                session_id="s1",
                uuid="turn-1",
                context_used=390_000,
                context_size=1_000_000,
            )

        asyncio.run(_fire())
        records = read_stats_file(recorder.path).records
        # Filter to the terminal (status=ok) rows we care about.
        ok = [r for r in records if r.status == STATUS_OK]
        assert len(ok) == 1, (
            f"exactly one terminal record expected; got {len(ok)}: {ok!r}"
        )
        rec = ok[0]
        # Contract 1: snapshot semantics, honest context_used/size.
        assert rec.usage_semantics == USAGE_SEMANTICS_CONTEXT_SNAPSHOT, (
            f"record must be a context_snapshot when the provider gave a "
            f"snapshot; got semantics={rec.usage_semantics!r}"
        )
        assert rec.context_used == 390_000, (
            f"context_used must be the observed snapshot, not the per-turn "
            f"total; got {rec.context_used!r}"
        )
        assert rec.context_size == 1_000_000, (
            f"context_size must be the CLI window size; got {rec.context_size!r}"
        )
        # Contract 2: per-turn totals preserved on the SAME row.
        assert rec.total_tokens == 2_265_400, (
            f"per-turn total_tokens must be preserved so the UI can render "
            f"'Last turn'; got {rec.total_tokens!r}"
        )
        assert rec.prompt_tokens == 2_200_000
        assert rec.completion_tokens == 65_400

    def test_latest_context_by_kind_picks_snapshot_over_token_fallback(
        self, tmp_path: Path
    ) -> None:
        """The aggregator must NEVER surface a per-turn total as
        ``context_used`` when a real snapshot is present in the same
        record.  This is the assertion the UI status bar depends on."""
        recorder = TokenUsageRecorder(tmp_path)
        cb = recorder.make_usage_event_callback(
            provider="claude-code",
            kind=KIND_AGENT,
            model_name="fable",
        )

        async def _fire() -> None:
            await cb(event="invocation_start")
            await cb(
                event="terminal",
                status=STATUS_OK,
                usage={
                    "input_tokens": 2_200_000,
                    "output_tokens": 65_400,
                    "total_tokens": 2_265_400,
                },
                session_id="s1",
                uuid="turn-1",
                context_used=390_000,
                context_size=1_000_000,
            )

        asyncio.run(_fire())
        records = read_stats_file(recorder.path).records
        snap = latest_context_by_kind(records)
        assert KIND_AGENT in snap, "no agent snapshot surfaced by aggregator"
        agent = snap[KIND_AGENT]
        # THIS is the assertion that would have caught the 2265.4k bug.
        assert agent.context_used == 390_000, (
            f"UI's Context field must show the observed snapshot (390k), "
            f"NOT the per-turn total (2265k); got context_used="
            f"{agent.context_used!r}"
        )
        assert agent.context_size == 1_000_000, (
            f"UI's Context field must include the CLI window size so it "
            f"can render as '390k/1000k'; got {agent.context_size!r}"
        )
        assert agent.usage_semantics == USAGE_SEMANTICS_CONTEXT_SNAPSHOT
        # Aggregator MUST plumb ``total_tokens`` through to the snapshot
        # dataclass so ``_fill_context_fields`` — which reads it via
        # ``getattr(snap, 'total_tokens', None)`` — can surface it as
        # ``agent_last_turn_tokens``. Pinned specifically because
        # ``UsageRecord.total_tokens`` (test 1) is a WEAKER assertion —
        # dropping the ``total_tokens=rec.total_tokens`` kwarg from both
        # aggregator ContextSnapshot constructors keeps this test file
        # green while ``Last turn`` silently disappears from the UI in
        # production (mutation check, 2026-08-22).
        assert agent.total_tokens == 2_265_400, (
            f"aggregator ContextSnapshot must carry the per-turn total so "
            f"the web _fill_context_fields helper can surface it as "
            f"agent_last_turn_tokens; got total_tokens={agent.total_tokens!r}"
        )
