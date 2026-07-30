"""Phase 1 "Analyzing history…" progress during the chunking pass.

Before this, `_chunk_messages` iterated the message groups silently, so the
long tokenization/chunking pass (~87s on ~8k messages) left the progress bar
frozen at zero with no feedback. This adds an OPTIONAL group-level progress
callback so the "Analyzing history…" phase ticks by group.

Contract:
  * callback fires (0, total_groups) at the start and (total_groups,
    total_groups) at the end;
  * it is monotonic non-decreasing and never exceeds total_groups;
  * `analyze_progress_step` controls tick frequency (step 1 → a tick per
    group; step > total → only the start and final ticks);
  * with NO callback, `_chunk_messages` returns exactly the same chunks as
    before (backward compatible).
"""

from __future__ import annotations

from semantic_search.config import IndexConfig
from semantic_search.history_indexer import (
    _chunk_messages,
    _group_by_summarized_event,
    _pending_chunks_for_file,
    _line_hashes_path,
    _save_line_counts,
    get_history_index_dir,
)


def _tok(text: str) -> list[int]:
    return [hash(w) & 0xFFFF for w in text.split()]


def _entries(n: int) -> list[dict]:
    # Each entry gets a distinct summarized_at so it forms its own group,
    # giving exactly n groups — makes group-tick assertions precise.
    return [
        {
            "summarized_at": f"ts{i}",
            "message_type": "human",
            "message": {"data": {"content": f"message number {i} with several words"}},
        }
        for i in range(n)
    ]


class TestChunkMessagesGroupProgress:
    def test_callback_brackets_start_and_end(self) -> None:
        groups = _group_by_summarized_event(_entries(10))
        assert len(groups) == 10
        seen: list[tuple[int, int]] = []
        _chunk_messages(
            groups,
            effective_max=100000,
            tokenize_fn=_tok,
            on_group_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen[0] == (0, 10)
        assert seen[-1] == (10, 10)

    def test_monotonic_non_decreasing_and_bounded(self) -> None:
        groups = _group_by_summarized_event(_entries(25))
        seen: list[tuple[int, int]] = []
        _chunk_messages(
            groups,
            effective_max=100000,
            tokenize_fn=_tok,
            on_group_progress=lambda d, t: seen.append((d, t)),
        )
        dones = [d for d, _ in seen]
        assert dones == sorted(dones)
        assert all(0 <= d <= 25 for d in dones)
        assert all(t == 25 for _, t in seen)

    def test_step_one_ticks_every_group(self) -> None:
        groups = _group_by_summarized_event(_entries(6))
        seen: list[int] = []
        _chunk_messages(
            groups,
            effective_max=100000,
            tokenize_fn=_tok,
            step=1,
            on_group_progress=lambda d, t: seen.append(d),
        )
        # start (0) + one per group (1..6)
        assert seen == [0, 1, 2, 3, 4, 5, 6]

    def test_large_step_only_start_and_final(self) -> None:
        groups = _group_by_summarized_event(_entries(6))
        seen: list[int] = []
        _chunk_messages(
            groups,
            effective_max=100000,
            tokenize_fn=_tok,
            step=1000,
            on_group_progress=lambda d, t: seen.append(d),
        )
        assert seen == [0, 6]

    def test_no_callback_is_backward_compatible(self) -> None:
        groups = _group_by_summarized_event(_entries(8))
        with_cb = _chunk_messages(
            groups, effective_max=100000, tokenize_fn=_tok,
            on_group_progress=lambda d, t: None,
        )
        without_cb = _chunk_messages(
            groups, effective_max=100000, tokenize_fn=_tok,
        )
        assert [t for t, _ in with_cb] == [t for t, _ in without_cb]

    def test_empty_groups_still_brackets(self) -> None:
        seen: list[tuple[int, int]] = []
        _chunk_messages(
            [], effective_max=100000, tokenize_fn=_tok,
            on_group_progress=lambda d, t: seen.append((d, t)),
        )
        # Degenerate but well-defined: dedup collapses the start and final
        # ticks into a SINGLE (0, 0) emission, so seen == [(0, 0)] and
        # seen[0] is seen[-1].
        assert seen[0] == (0, 0)
        assert seen[-1] == (0, 0)
        assert seen == [(0, 0)]


class TestPendingChunksForwardsAnalyzeProgress:
    def test_forwards_group_progress(self, tmp_path) -> None:
        summaries = tmp_path
        jsonl = summaries / "h.jsonl"
        import json

        objs = _entries(12)
        jsonl.write_text(
            "\n".join(json.dumps(o) for o in objs) + "\n", encoding="utf-8"
        )
        index_dir = get_history_index_dir(summaries)
        index_dir.mkdir(parents=True, exist_ok=True)
        _save_line_counts(_line_hashes_path(index_dir), {str(jsonl): 0})

        seen: list[tuple[int, int]] = []
        _pending_chunks_for_file(
            jsonl, index_dir, IndexConfig(), _tok, 100000, 0,
            on_group_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen, "analyze progress callback must be forwarded to _chunk_messages"
        assert seen[0][0] == 0
        assert seen[-1][0] == seen[-1][1]  # reached the total
