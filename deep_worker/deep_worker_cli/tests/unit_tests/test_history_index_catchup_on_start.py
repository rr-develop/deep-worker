"""Unit tests for history-index catch-up on agent start.

Bug: when the index exists but the JSONL has grown since the last
indexing, the agent's startup currently SKIPS re-indexing — the guard
`if jsonl_files and not history_index_exists(summaries_path):` short-
circuits as soon as any index dir is present.

These tests pin down the expected behaviour: **every agent start
recomputes a cheap "pending lines" delta. If it's > 0, indexing runs.**

See ``docs/DESIGN_HISTORY_INDEX_CATCHUP_ON_START.md``.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: _count_pending_history_lines
# ---------------------------------------------------------------------------


def test_pending_count_helper_with_missing_hash_file(tmp_path: Path) -> None:
    """No line_hashes.json → helper returns total lines in all JSONL files."""
    from deep_worker_cli.agent import _count_pending_history_lines

    summaries = tmp_path
    j1 = summaries / "a.jsonl"
    j1.write_text("line1\nline2\nline3\n", encoding="utf-8")
    j2 = summaries / "b.jsonl"
    j2.write_text("only-one\n", encoding="utf-8")

    pending = _count_pending_history_lines(summaries, [j1, j2])
    assert pending == 4  # 3 + 1


def test_pending_count_helper_ignores_stale_entries(tmp_path: Path) -> None:
    """line_hashes.json references a missing file → ignored, no raise."""
    from deep_worker_cli.agent import _count_pending_history_lines

    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()

    j1 = summaries / "live.jsonl"
    j1.write_text("a\nb\nc\nd\n", encoding="utf-8")
    hashes = {
        str(j1): 2,
        str(summaries / "ghost.jsonl"): 9999,
    }
    (index_dir / "line_hashes.json").write_text(
        json.dumps(hashes), encoding="utf-8",
    )

    pending = _count_pending_history_lines(summaries, [j1])
    assert pending == 2


def test_pending_count_is_zero_when_index_is_current(tmp_path: Path) -> None:
    """If recorded line counts equal current ones, helper returns 0."""
    from deep_worker_cli.agent import _count_pending_history_lines

    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()

    j1 = summaries / "x.jsonl"
    j1.write_text("a\nb\nc\n", encoding="utf-8")
    (index_dir / "line_hashes.json").write_text(
        json.dumps({str(j1): 3}), encoding="utf-8",
    )

    pending = _count_pending_history_lines(summaries, [j1])
    assert pending == 0


def test_pending_count_never_negative_when_file_shrank(tmp_path: Path) -> None:
    """File shorter than recorded → still report current count as pending."""
    from deep_worker_cli.agent import _count_pending_history_lines

    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()

    j1 = summaries / "x.jsonl"
    j1.write_text("only-one\n", encoding="utf-8")
    (index_dir / "line_hashes.json").write_text(
        json.dumps({str(j1): 50}), encoding="utf-8",
    )

    pending = _count_pending_history_lines(summaries, [j1])
    assert pending == 1


# ---------------------------------------------------------------------------
# Decision: _should_catchup_history_index(summaries_dir) -> (bool, int)
# ---------------------------------------------------------------------------


def test_catchup_decision_triggers_when_jsonl_grew(tmp_path: Path) -> None:
    """Decision returns True + delta when pending > 0."""
    from deep_worker_cli.agent import _should_catchup_history_index

    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()
    (index_dir / "lancedb").mkdir()

    j1 = summaries / "h.jsonl"
    j1.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    (index_dir / "line_hashes.json").write_text(
        json.dumps({str(j1): 2}), encoding="utf-8",
    )

    needs_catchup, pending = _should_catchup_history_index(summaries)
    assert needs_catchup is True
    assert pending == 3


def test_no_catchup_when_index_is_current(tmp_path: Path) -> None:
    """Decision returns False when nothing changed."""
    from deep_worker_cli.agent import _should_catchup_history_index

    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()
    (index_dir / "lancedb").mkdir()

    j1 = summaries / "h.jsonl"
    j1.write_text("a\nb\nc\n", encoding="utf-8")
    (index_dir / "line_hashes.json").write_text(
        json.dumps({str(j1): 3}), encoding="utf-8",
    )

    needs_catchup, pending = _should_catchup_history_index(summaries)
    assert needs_catchup is False
    assert pending == 0


def test_first_run_reports_all_lines_pending(tmp_path: Path) -> None:
    """No index dir → all lines are pending (existing first-run path)."""
    from deep_worker_cli.agent import _should_catchup_history_index

    summaries = tmp_path
    j1 = summaries / "h.jsonl"
    j1.write_text("a\nb\nc\nd\ne\nf\n", encoding="utf-8")

    needs_catchup, pending = _should_catchup_history_index(summaries)
    assert needs_catchup is True
    assert pending == 6


def test_empty_summaries_dir_no_catchup(tmp_path: Path) -> None:
    """No JSONL files → no catchup needed."""
    from deep_worker_cli.agent import _should_catchup_history_index

    needs_catchup, pending = _should_catchup_history_index(tmp_path)
    assert needs_catchup is False
    assert pending == 0


def test_missing_summaries_dir_no_catchup(tmp_path: Path) -> None:
    """Non-existent summaries dir → no work."""
    from deep_worker_cli.agent import _should_catchup_history_index

    missing = tmp_path / "does_not_exist"
    needs_catchup, pending = _should_catchup_history_index(missing)
    assert needs_catchup is False
    assert pending == 0


# ---------------------------------------------------------------------------
# Integration with the startup block (regression for the actual call-site)
# ---------------------------------------------------------------------------


def test_startup_block_invokes_indexer_when_stale(tmp_path: Path, monkeypatch) -> None:
    """
    Regression for the specific bug:

    A stale index (index dir exists but JSONL grew) MUST trigger
    ``index_all_history_files`` at startup.

    We don't run the full ``create_cli_agent`` (too many deps). Instead
    we exercise the exact decision branch by calling
    ``_should_catchup_history_index`` against a stale-index layout and
    asserting that the production code path would call the indexer.
    """
    # Layout: index exists (so the OLD guard would have SKIPPED), but
    # JSONL has grown beyond what's recorded.
    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()
    (index_dir / "lancedb").mkdir()  # makes history_index_exists() True

    j1 = summaries / "h.jsonl"
    j1.write_text("a\nb\nc\nd\n", encoding="utf-8")
    (index_dir / "line_hashes.json").write_text(
        json.dumps({str(j1): 1}), encoding="utf-8",
    )

    from deep_worker_cli.agent import _should_catchup_history_index

    needs_catchup, pending = _should_catchup_history_index(summaries)

    # OLD behaviour would have been: history_index_exists -> True ->
    # skip indexing entirely. THIS test asserts the new behaviour:
    # we DO need catchup even though the index dir exists.
    assert needs_catchup is True, (
        "Catch-up must trigger when index exists but JSONL has new lines. "
        "If this fails, the agent will silently serve stale search results "
        "after every restart."
    )
    assert pending == 3


# ---------------------------------------------------------------------------
# Regression: U+2028 desync (single source of truth for line counting)
# ---------------------------------------------------------------------------

# U+2028 LINE SEPARATOR inside a JSON string value. Valid JSON on one physical
# line, but str.splitlines() treats it as a line break. The cheap-check
# (physical \n lines) and the indexer's recorded count MUST still agree.
# Use chr(0x2028) — a literal glyph in source survives editors/transport as a
# plain space and would silently disable this regression guard.
_U2028 = chr(0x2028)


def test_no_catchup_when_records_contain_u2028(tmp_path: Path) -> None:
    """A fully-indexed file whose records embed U+2028 must report 0 pending.

    Regression for the infinite-reindex bug and its false-green trap:

    * ``ensure_ascii=False`` writes the RAW U+2028 (as real summaries do). With
      the default escaping (``\\u2028``), ``str.splitlines()`` would not break
      on it and the whole test would pass even against the buggy code.
    * The recorded count is derived the SAME way the real indexer records it —
      ``len(_read_jsonl_lines(...))`` — NOT via a hand-picked constant. So if
      ``_read_jsonl_lines`` ever regresses to ``str.splitlines()`` it records a
      shattered count (1) while the cheap-check counts physical \\n lines (3),
      the delta is non-zero, and this test goes RED (as it must).
    """
    from deep_worker_cli.agent import _count_pending_history_lines
    from deep_worker.providers.semantic_search.history_indexer import _read_jsonl_lines

    summaries = tmp_path
    index_dir = summaries / ".history_index"
    index_dir.mkdir()

    j1 = summaries / "h.jsonl"
    j1.write_text(
        json.dumps({"i": 0, "t": f"a{_U2028}b{_U2028}c"}, ensure_ascii=False) + "\n"
        + json.dumps({"i": 1, "t": "plain"}, ensure_ascii=False) + "\n"
        + json.dumps({"i": 2, "t": f"x{_U2028}y"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Record exactly what the indexer would persist to line_hashes.json.
    recorded = len(_read_jsonl_lines(j1))
    (index_dir / "line_hashes.json").write_text(
        json.dumps({str(j1): recorded}), encoding="utf-8",
    )

    pending = _count_pending_history_lines(summaries, [j1])
    assert pending == 0, (
        "U+2028 inside record content must NOT create phantom pending lines: "
        "the indexer's recorded count and the startup cheap-check must use one "
        "shared counter"
    )
