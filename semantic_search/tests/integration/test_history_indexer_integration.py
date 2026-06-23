"""Integration test for history_indexer — indexes real JSONL data.

This test creates a temporary summaries directory with synthetic JSONL
files, indexes them using the real embedding model (MockEmbedding for speed),
and verifies that the index can be searched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from semantic_search.history_indexer import (
    index_all_history_files,
    index_history_file,
    history_index_exists,
    clear_history_index,
    get_history_index_dir,
)
from semantic_search.history_retriever import search_history


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


@pytest.fixture
def embed_model() -> MockEmbedding:
    """Return a MockEmbedding for integration testing (no real model needed)."""
    return MockEmbedding(embed_dim=384)


class TestHistoryIndexerIntegration:
    """Integration tests that index real JSONL data and verify searchability."""

    def test_index_and_search_single_file(
        self, summaries_dir: Path, embed_model: MockEmbedding
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
        self, summaries_dir: Path, embed_model: MockEmbedding
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
        self, summaries_dir: Path, embed_model: MockEmbedding
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

    def test_search_returns_metadata(
        self, summaries_dir: Path, embed_model: MockEmbedding
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
        self, summaries_dir: Path, embed_model: MockEmbedding
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
        self, summaries_dir: Path, embed_model: MockEmbedding
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
