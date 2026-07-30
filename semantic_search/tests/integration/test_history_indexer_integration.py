"""Integration test for history_indexer — indexes real JSONL data.

This test creates a temporary summaries directory with synthetic JSONL
files, indexes them using the real ``intfloat/multilingual-e5-small`` embedder
(loaded strictly offline), and verifies that the index can be searched.
``MockEmbedding`` can no longer be used because the chunker requires the
embedder's own XLM-R tokenizer and RC-1 forbids a tiktoken fallback. See
``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from semantic_search.history_indexer import (  # noqa: E402
    index_all_history_files,
    index_history_file,
    history_index_exists,
    clear_history_index,
    get_history_index_dir,
)
from semantic_search.history_retriever import search_history  # noqa: E402


def _make_jsonl(summaries_dir: Path, thread_id: str, lines: list[dict]) -> Path:
    """Create a JSONL history file in summaries_dir and return its path."""
    dest = summaries_dir / f"{thread_id}.jsonl"
    content = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    if content:
        content += "\n"
    dest.write_text(content, encoding="utf-8")
    return dest


@pytest.fixture
def summaries_dir(tmp_path: Path) -> Path:
    """Create a temporary summaries directory."""
    d = tmp_path / "summaries"
    d.mkdir(parents=True)
    return d


@pytest.fixture(scope="module")
def embed_model() -> Any:
    """Real strictly-offline HuggingFaceEmbedding(intfloat/multilingual-e5-small).

    Module-scoped so the model is loaded once. ``MockEmbedding`` cannot be used
    any more: the chunker requires a real XLM-R tokenizer (RC-1 forbids the
    silent tiktoken fallback). See docs/DESIGN_FIX_TOKENIZER_MISMATCH.md.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


class TestHistoryIndexerIntegration:
    """Integration tests that index real JSONL data and verify searchability."""

    def test_index_and_search_single_file(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Index one JSONL file, then search it."""
        _make_jsonl(summaries_dir, "thread_1", [
            {
                "schema_version": 4,
                "kind": "deepagents.summarization_history.message",
                "summarized_at": "2026-01-01T00:00:00",
                "message_index": 0,
                "message_type": "human",
                "message": {"data": {"content": "What is the capital of France?"}},
            },
            {
                "schema_version": 4,
                "kind": "deepagents.summarization_history.message",
                "summarized_at": "2026-01-01T00:00:00",
                "message_index": 1,
                "message_type": "ai",
                "message": {"data": {"content": "The capital of France is Paris."}},
            },
        ])

        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=embed_model,
        )
        assert stats.files_processed == 1
        assert stats.chunks_indexed >= 1

        # Verify index exists
        assert history_index_exists(summaries_dir)

        # Search
        results = search_history(
            query="capital of France",
            summaries_dir=summaries_dir,
            top_k=5,
            embed_model=embed_model,
        )
        assert len(results) >= 1
        assert "Paris" in results[0].content

    def test_index_and_search_multiple_files(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Index multiple JSONL files and search across them."""
        _make_jsonl(summaries_dir, "thread_a", [
            {"summarized_at": "ts1", "message_index": 0, "message_type": "human",
             "message": {"data": {"content": "How do I implement binary search?"}}},
            {"summarized_at": "ts1", "message_index": 1, "message_type": "ai",
             "message": {"data": {"content": "Binary search requires a sorted array. You check the middle element and recurse on the left or right half."}}},
        ])
        _make_jsonl(summaries_dir, "thread_b", [
            {"summarized_at": "ts2", "message_index": 0, "message_type": "human",
             "message": {"data": {"content": "What is a linked list?"}}},
            {"summarized_at": "ts2", "message_index": 1, "message_type": "ai",
             "message": {"data": {"content": "A linked list is a linear data structure where elements are stored in nodes, each pointing to the next."}}},
        ])

        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=embed_model,
        )
        assert stats.files_processed >= 2

        # Search for binary search content
        results = search_history(
            query="binary search sorted array",
            summaries_dir=summaries_dir,
            top_k=5,
            embed_model=embed_model,
        )
        assert len(results) >= 1
        assert any("Binary search" in r.content for r in results)

        # Search for linked list content
        results = search_history(
            query="linked list data structure",
            summaries_dir=summaries_dir,
            top_k=5,
            embed_model=embed_model,
        )
        assert len(results) >= 1
        assert any("linked list" in r.content.lower() for r in results)

    def test_clear_and_reindex(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Clear index and verify it's gone."""
        _make_jsonl(summaries_dir, "thread_x", [
            {"summarized_at": "ts1", "message_index": 0, "message_type": "ai",
             "message": {"data": {"content": "Hello world test"}}},
        ])
        index_all_history_files(summaries_dir=summaries_dir, embed_model=embed_model)
        assert history_index_exists(summaries_dir)

        clear_history_index(summaries_dir)
        assert not history_index_exists(summaries_dir)

    def test_second_index_is_noop_with_u2028_records(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Regression: a file whose records embed U+2028 must NOT be re-indexed
        on the second pass.

        Pre-fix, ``_read_jsonl_lines`` split on ``str.splitlines()`` (which
        breaks on U+2028), so the recorded record-count desynced from the
        physical ``\\n`` count and every subsequent ``index_all_history_files``
        re-chunked and re-embedded the WHOLE file. This asserts the second pass
        processes zero files/chunks.
        """
        u2028 = " "
        _make_jsonl(summaries_dir, "thread_u", [
            {"summarized_at": "ts1", "message_index": 0, "message_type": "human",
             "message": {"data": {"content": f"line one{u2028}still same record"}}},
            {"summarized_at": "ts1", "message_index": 1, "message_type": "ai",
             "message": {"data": {"content": f"a{u2028}b{u2028}c multi separator"}}},
            {"summarized_at": "ts1", "message_index": 2, "message_type": "human",
             "message": {"data": {"content": "a plain trailing record"}}},
        ])

        first = index_all_history_files(
            summaries_dir=summaries_dir, embed_model=embed_model,
        )
        assert first.files_processed == 1
        assert first.chunks_indexed >= 1

        # Second pass over the UNCHANGED file must be a strict no-op.
        second = index_all_history_files(
            summaries_dir=summaries_dir, embed_model=embed_model,
        )
        assert second.files_processed == 0, (
            "U+2028 record content must not trigger a spurious full re-index"
        )
        assert second.chunks_indexed == 0
        assert second.lines_read <= 0

    def test_search_returns_metadata(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Verify that search results include file_path, score, and timestamp."""
        _make_jsonl(summaries_dir, "thread_m", [
            {"summarized_at": "2026-03-15T10:00:00", "message_index": 0, "message_type": "human",
             "message": {"data": {"content": "What is the weather like today?"}}},
            {"summarized_at": "2026-03-15T10:00:00", "message_index": 1, "message_type": "ai",
             "message": {"data": {"content": "The weather is sunny and warm at 22 degrees."}}},
        ])
        index_all_history_files(summaries_dir=summaries_dir, embed_model=embed_model)

        results = search_history(
            query="weather sunny",
            summaries_dir=summaries_dir,
            top_k=5,
            embed_model=embed_model,
        )
        assert len(results) >= 1
        r = results[0]
        assert r.file_path  # non-empty
        assert 0.0 <= r.score <= 1.0
        assert r.content


class TestHistoryIndexerOnProgress:
    """Integration tests for the on_progress callback."""

    def test_on_progress_called_for_all_files(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Verify on_progress fires once per file during indexing."""
        files = ["thread_a", "thread_b", "thread_c"]
        for name in files:
            _make_jsonl(summaries_dir, name, [
                {"summarized_at": "ts1", "message_index": 0, "message_type": "human",
                 "message": {"data": {"content": f"Test message in {name}"}}},
                {"summarized_at": "ts1", "message_index": 1, "message_type": "ai",
                 "message": {"data": {"content": f"Response in {name}"}}},
            ])

        called: list[str] = []

        def on_progress(filename: str) -> None:
            called.append(filename)

        index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=embed_model,
            on_progress=on_progress,
        )

        assert len(called) == 3
        # Verify search works after indexing with progress
        results = search_history(
            query="test message",
            summaries_dir=summaries_dir,
            top_k=5,
            embed_model=embed_model,
        )
        assert len(results) >= 1

    def test_on_progress_empty_dir_no_errors(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Verify on_progress works with empty summaries dir."""
        called: list[str] = []

        def on_progress(filename: str) -> None:
            called.append(filename)

        index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=embed_model,
            on_progress=on_progress,
        )
        assert len(called) == 0

    def test_analyze_phase_precedes_embed_phase(
        self, summaries_dir: Path, embed_model: Any
    ) -> None:
        """Phase 1 "Analyzing history…" must reach (total, total) BEFORE the
        first Phase 2 (embed/message) tick, and be monotonic.

        This is the real end-to-end ordering the CLI relies on: the analyze
        bar fills during the (slow) chunking pass, then the indexing bar takes
        over — no silent gap while chunking runs.
        """
        _make_jsonl(summaries_dir, "thread_phases", [
            {"summarized_at": f"ts{i}", "message_index": i,
             "message_type": "human" if i % 2 == 0 else "ai",
             "message": {"data": {"content": f"phase ordering message {i} words"}}}
            for i in range(30)
        ])

        events: list[tuple[str, int, int]] = []

        def on_analyze(done: int, total: int) -> None:
            events.append(("analyze", done, total))

        def on_message(done: int, total: int) -> None:
            events.append(("message", done, total))

        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=embed_model,
            on_message_progress=on_message,
            on_analyze_progress=on_analyze,
        )
        assert stats.files_processed == 1

        analyze = [(d, t) for kind, d, t in events if kind == "analyze"]
        message = [(d, t) for kind, d, t in events if kind == "message"]
        assert analyze, "analyze phase must emit progress"
        assert message, "message phase must emit progress"

        # analyze reaches its total and is monotonic
        assert analyze[-1][0] == analyze[-1][1]
        assert [d for d, _ in analyze] == sorted(d for d, _ in analyze)

        # ordering: the analyze phase completes before the first REAL embed
        # tick. index_all_history_files emits an initial message (0, total) to
        # initialize the bar before any file is touched — that init tick is not
        # "embedding started", so we key on the first message tick with done>0.
        first_real_message_idx = next(
            (i for i, (kind, d, _t) in enumerate(events)
             if kind == "message" and d > 0),
            None,
        )
        last_analyze_idx = max(
            i for i, (kind, *_1) in enumerate(events) if kind == "analyze"
        )
        assert first_real_message_idx is not None, (
            "expected at least one non-zero embed tick"
        )
        assert last_analyze_idx < first_real_message_idx, (
            "Phase 1 (analyze) must finish before Phase 2 (embed) really starts"
        )
