"""Incremental no-op at equality.

``_pending_chunks_for_file`` decides how much of a JSONL file to (re)index.
The bug: it used ``total_entries > last_line_count`` to pick the incremental
slice, so at EQUALITY (nothing changed) it fell through to the ``else`` branch
and re-chunked the ENTIRE file. Combined with the count desync, this made every
agent start re-index the whole history.

Contract now:
  * recorded == current  -> zero pending work (no chunks, lines_read <= 0);
  * recorded  < current  -> only the tail (new records) is chunked;
  * file shrank          -> re-index from scratch (unchanged behaviour).
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_search.history_indexer import (
    IndexConfig,
    _line_hashes_path,
    _pending_chunks_for_file,
    _save_line_counts,
    count_jsonl_lines,
    get_history_index_dir,
)


def _fake_tokenize(text: str) -> list[int]:
    return [hash(w) & 0xFFFF for w in text.split()]


def _write_records(path: Path, n: int) -> None:
    objs = [
        {
            "summarized_at": "ts1",
            "message_type": "human",
            "message": {"data": {"content": f"message number {i} with words"}},
        }
        for i in range(n)
    ]
    path.write_text(
        "\n".join(json.dumps(o) for o in objs) + "\n",
        encoding="utf-8",
    )


def _record_count(index_dir: Path, jsonl: Path, count: int) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _save_line_counts(_line_hashes_path(index_dir), {str(jsonl): count})


class TestPendingChunksNoOp:
    def test_no_pending_when_count_unchanged(self, tmp_path: Path) -> None:
        summaries = tmp_path
        jsonl = summaries / "h.jsonl"
        _write_records(jsonl, 5)
        index_dir = get_history_index_dir(summaries)
        _record_count(index_dir, jsonl, count_jsonl_lines(jsonl))  # == 5

        chunks, _total_entries, lines_read = _pending_chunks_for_file(
            jsonl, index_dir, IndexConfig(), _fake_tokenize, 100000, 0,
        )

        # Equality is a strict no-op: zero new lines, zero chunks. Crucially
        # NOT a full re-index of all 5 records (the pre-fix bug).
        assert lines_read == 0
        assert chunks == []

    def test_only_tail_when_grew(self, tmp_path: Path) -> None:
        summaries = tmp_path
        jsonl = summaries / "h.jsonl"
        _write_records(jsonl, 5)
        index_dir = get_history_index_dir(summaries)
        _record_count(index_dir, jsonl, 3)  # 2 new records pending

        chunks, total_entries, lines_read = _pending_chunks_for_file(
            jsonl, index_dir, IndexConfig(), _fake_tokenize, 100000, 0,
        )

        assert total_entries == 5
        assert lines_read == 2
        assert len(chunks) >= 1  # only the 2 tail records got chunked

    def test_shrank_file_is_noop_at_this_layer(self, tmp_path: Path) -> None:
        """A file shorter than recorded yields negative lines_read → no work.

        (Pre-existing behaviour; kept intentionally. The important property is
        that the count desync no longer forces a spurious full re-index.)
        """
        summaries = tmp_path
        jsonl = summaries / "h.jsonl"
        _write_records(jsonl, 2)
        index_dir = get_history_index_dir(summaries)
        _record_count(index_dir, jsonl, 50)  # recorded > current

        chunks, _total_entries, lines_read = _pending_chunks_for_file(
            jsonl, index_dir, IndexConfig(), _fake_tokenize, 100000, 0,
        )

        assert lines_read < 0
        assert chunks == []
