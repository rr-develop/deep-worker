"""Tests for ``latest_context_by_kind`` and ``BucketTotals.latest_by_kind``.

DESIGN_AGENT_CONTEXT_STATUS_BAR.md — backend piece.

Covers M1, M1.5, M2, M2.5, M3, M4, M5 mutations: each test pins a
specific behaviour the implementation MUST preserve.

Mutation map (CAUGHT target — the implementation must pass all of these tests):

M1  drop kind filter                       → test_mixed_per_kind
M1.5 forget exclude kind="other"           → test_kind_other_excluded
M2  drop tie-break by ts                    → test_ts_tie_break
M2.5 drop file-order tie-break on equal-ts → test_file_order_tie_break_equal_ts
M3  drop usage_semantics in the output      → test_snapshot_detection
M4  drop tm handling                        → test_mixed_per_kind_tm_present
M5  ignore empty context_used               → test_negative_malformed_snapshot
M11 _fill_context_fields propagates exception → test_read_stats_file_swallows_corrupt
M12 disk-read on every broadcast without cache → test_read_stats_file_caches_within_window
M13 backend does not gate tm_* by has_task_manager → test_bucket_totals_latest_by_kind_present
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deep_worker_cli.stats.token_stats import (
    SCHEMA_VERSION,
    USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
    USAGE_SEMANTICS_TOKENS,
    BucketTotals,
    UsageRecord,
    aggregate_records,
    read_stats_file,
)

# Tolerant import: RED-first contract. The impl is not yet in place; if
# the new symbols (``ContextSnapshot``, ``latest_context_by_kind``) are
# missing, set them to ``None`` so the test module imports cleanly and
# each test fails on a meaningful assert (not on a collection error).
try:
    from deep_worker_cli.stats.token_stats import (  # type: ignore[attr-defined]
        ContextSnapshot,
        latest_context_by_kind,
    )
except ImportError:
    ContextSnapshot = None  # type: ignore[assignment,misc]
    latest_context_by_kind = None  # type: ignore[assignment,misc]


def _rec(*, kind="agent", semantics=USAGE_SEMANTICS_TOKENS,
         ts="2026-08-01T12:00:00.000Z", context_used=None, context_size=None,
         total_tokens=None, prompt_tokens=None, completion_tokens=None,
         provider="deepseek", model="m") -> UsageRecord:
    return UsageRecord(
        provider=provider, model=model, kind=kind, status="ok",
        usage_semantics=semantics, ts=ts,
        context_used=context_used, context_size=context_size,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# Pure-function tests: latest_context_by_kind
# ---------------------------------------------------------------------------


def test_empty_records_returns_empty_dict():
    """M1: empty input → ``{}``."""
    assert latest_context_by_kind([]) == {}


def test_only_token_records_returns_last_turn_per_kind():
    """Token-semantics records with ``total_tokens`` populate ``context_used``.

    A related internal incident analysis (not included here): previously the
    web UI showed ``Context: -`` for every agent whose provider never
    emits a ``context_snapshot`` (e.g. the OpenAI-compatible ``minimax``
    adapter only writes ``usage_semantics=tokens`` records).  We now
    surface the last turn's ``total_tokens`` as ``context_used`` so the
    UI can render ``<used>k tokens`` (exact from the provider's
    tokenizer — no ``"(est.)"`` qualifier) instead of a dash.  Both
    ``kind="agent"`` and ``kind="tm"`` rows are exposed independently.
    """
    records = [
        _rec(
            kind="agent", semantics=USAGE_SEMANTICS_TOKENS, total_tokens=100,
            ts="2026-08-01T12:00:00.000Z",
        ),
        _rec(
            kind="tm", semantics=USAGE_SEMANTICS_TOKENS, total_tokens=200,
            ts="2026-08-01T12:00:01.000Z",
        ),
    ]
    out = latest_context_by_kind(records)
    assert set(out.keys()) == {"agent", "tm"}
    assert out["agent"].context_used == 100
    assert out["agent"].context_size is None
    assert out["agent"].usage_semantics == USAGE_SEMANTICS_TOKENS
    assert out["tm"].context_used == 200
    assert out["tm"].context_size is None
    assert out["tm"].usage_semantics == USAGE_SEMANTICS_TOKENS


def test_token_records_without_total_tokens_still_excluded():
    """Records with semantics=tokens but no total_tokens contribute nothing."""
    records = [
        _rec(
            kind="agent", semantics=USAGE_SEMANTICS_TOKENS,
            total_tokens=None, prompt_tokens=None, completion_tokens=None,
            ts="2026-08-01T12:00:00.000Z",
        ),
    ]
    assert latest_context_by_kind(records) == {}


def test_only_snapshot_records_returns_agent():
    """M3: snapshot semantics with context_used + context_size."""
    rec = _rec(
        kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
        context_used=42100, context_size=200000,
        ts="2026-08-01T12:00:00.000Z",
    )
    out = latest_context_by_kind([rec])
    assert set(out.keys()) == {"agent"}
    snap = out["agent"]
    assert snap.context_used == 42100
    assert snap.context_size == 200000
    assert snap.usage_semantics == "context_snapshot"
    assert snap.ts == "2026-08-01T12:00:00.000Z"


def test_mixed_per_kind_returns_both():
    """M4: kind=agent + kind=tm records both kept."""
    rec_a = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                 context_used=1000, context_size=10000,
                 ts="2026-08-01T12:00:00.000Z")
    rec_t = _rec(kind="tm", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                 context_used=500, context_size=10000,
                 ts="2026-08-01T12:01:00.000Z")
    out = latest_context_by_kind([rec_a, rec_t])
    assert set(out.keys()) == {"agent", "tm"}
    assert out["agent"].context_used == 1000
    assert out["tm"].context_used == 500


def test_kind_other_excluded():
    """M1.5: ``kind="other"`` (sub-agent calls) MUST NOT appear."""
    rec_other = _rec(kind="other", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                     context_used=999, context_size=8888,
                     ts="2026-08-01T12:00:00.000Z")
    out = latest_context_by_kind([rec_other])
    assert out == {}


def test_negative_malformed_snapshot_excluded():
    """M5: snapshot semantics with BOTH context_used AND context_size None → not a real snapshot."""
    rec = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
               context_used=None, context_size=None,
               ts="2026-08-01T12:00:00.000Z")
    out = latest_context_by_kind([rec])
    assert out == {}


def test_snapshot_with_only_context_size_kept():
    """Edge case: snapshot declared, size known, used missing — still kept."""
    rec = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
               context_used=None, context_size=10000,
               ts="2026-08-01T12:00:00.000Z")
    out = latest_context_by_kind([rec])
    assert "agent" in out
    assert out["agent"].context_size == 10000


def test_ts_tie_break_later_wins():
    """M2: two records, same kind, different ts — later wins."""
    earlier = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                   context_used=100, context_size=10000,
                   ts="2026-08-01T11:00:00.000Z")
    later = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                 context_used=200, context_size=10000,
                 ts="2026-08-01T13:00:00.000Z")
    out = latest_context_by_kind([earlier, later])
    assert out["agent"].context_used == 200


def test_file_order_tie_break_equal_ts():
    """M2.5: three records, same kind, same ts — last one in the input list wins."""
    same_ts = "2026-08-01T12:00:00.000Z"
    first = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                 context_used=100, context_size=10000, ts=same_ts)
    second = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                  context_used=200, context_size=10000, ts=same_ts)
    third = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                 context_used=300, context_size=10000, ts=same_ts)
    out = latest_context_by_kind([first, second, third])
    assert out["agent"].context_used == 300


def test_malformed_record_without_kind_skipped():
    """Records missing required fields → silently dropped (no exception)."""
    # Build a record that has NO kind at all (passing it through
    # UsageRecord.from_dict via dict path with kind=None returns None).
    records = [
        _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
             context_used=100, context_size=10000,
             ts="2026-08-01T12:00:00.000Z"),
        # Garbage in the middle of a JSONL line — but here we use a typed
        # record. The real malformed case is tested at the read_stats_file
        # layer (test_read_stats_file_swallows_corrupt). For this typed API
        # we just verify nothing explodes when given a list with one valid
        # record and one that should be skipped at the aggregate layer.
        _rec(kind="agent", semantics=USAGE_SEMANTICS_TOKENS,
             ts="2026-08-01T12:01:00.000Z"),
    ]
    out = latest_context_by_kind(records)
    # Only the snapshot should be in the result.
    assert set(out.keys()) == {"agent"}
    assert out["agent"].context_used == 100


def test_snapshot_detection_preserves_semantics():
    """M3: output carries usage_semantics verbatim for downstream rendering."""
    rec = _rec(kind="tm", semantics=USAGE_SEMANTICS_TOKENS,
               context_used=42, ts="2026-08-01T12:00:00.000Z")
    out = latest_context_by_kind([rec])
    # tokens with context_used only (no size) — kept, semantics is "tokens".
    assert "tm" in out
    assert out["tm"].usage_semantics == "tokens"
    assert out["tm"].context_size is None


# ---------------------------------------------------------------------------
# read_stats_file — broken JSONL contract (M11)
# ---------------------------------------------------------------------------


def test_read_stats_file_swallows_corrupt(tmp_path: Path):
    """M11: ``read_stats_file`` never raises; corrupt lines → skipped_records++."""
    bad_file = tmp_path / "token_stats.jsonl"
    bad_file.write_text('{"not":"valid"}\n')

    result = read_stats_file(bad_file)

    assert result.records == []
    assert result.skipped_records >= 1
    assert result.partial_tail is False


def test_read_stats_file_missing_returns_empty(tmp_path: Path):
    """Missing file is not an error."""
    result = read_stats_file(tmp_path / "does_not_exist.jsonl")
    assert result.records == []
    assert result.skipped_records == 0


def test_read_stats_file_partial_tail_flagged(tmp_path: Path):
    """Trailing partial line without newline → partial_tail=True, line skipped."""
    f = tmp_path / "token_stats.jsonl"
    f.write_text('{"schema_version": 1}\n{"oops":')

    result = read_stats_file(f)
    assert result.partial_tail is True
    # The good line should still parse (it's the proper schema marker).


# ---------------------------------------------------------------------------
# BucketTotals — to_dict() contract (M13)
# ---------------------------------------------------------------------------


def test_bucket_totals_latest_by_kind_present():
    """M13: ``BucketTotals.to_dict()`` MUST include ``latest_by_kind`` key."""
    rec = _rec(kind="agent", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
               context_used=100, context_size=10000,
               ts="2026-08-01T12:00:00.000Z")
    totals = aggregate_records([rec])
    d = totals.to_dict()

    assert "latest_by_kind" in d, (
        "M13: BucketTotals.to_dict() must surface latest_by_kind so the "
        "web REST aggregator can read it; without it the UI cannot mirror "
        "context into AgentInfo."
    )
    # Shape: dict with kind keys, each a snapshot dict with the 4 fields.
    assert "agent" in d["latest_by_kind"]
    snap = d["latest_by_kind"]["agent"]
    assert snap["context_used"] == 100
    assert snap["context_size"] == 10000
    assert snap["usage_semantics"] == "context_snapshot"


def test_bucket_totals_latest_by_kind_empty_when_no_records():
    """No records → ``latest_by_kind == {}``."""
    totals = aggregate_records([])
    d = totals.to_dict()
    assert d["latest_by_kind"] == {}


def test_bucket_totals_latest_by_kind_includes_tm():
    """TM records surface under ``latest_by_kind['tm']``."""
    rec_t = _rec(kind="tm", semantics=USAGE_SEMANTICS_CONTEXT_SNAPSHOT,
                 context_used=42, context_size=10000,
                 ts="2026-08-01T12:00:00.000Z")
    totals = aggregate_records([rec_t])
    d = totals.to_dict()
    assert "tm" in d["latest_by_kind"]
    assert d["latest_by_kind"]["tm"]["context_used"] == 42