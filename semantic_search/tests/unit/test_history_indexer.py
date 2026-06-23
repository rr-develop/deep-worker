"""Unit tests for semantic_search.history_indexer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_search.history_indexer import (
    HistoryIndexStats,
    get_history_index_dir,
    _read_jsonl_lines,
    _group_by_summarized_event,
    _chunk_messages,
)


class TestGetHistoryIndexDir:
    def test_returns_dot_history_index(self) -> None:
        path = get_history_index_dir(Path("/tmp/summaries"))
        assert str(path).endswith(".history_index")
        assert str(path) == "/tmp/summaries/.history_index"


class TestReadJsonlLines:
    def test_reads_valid_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"message_type": "human", "message_index": 0}),
            json.dumps({"message_type": "ai", "message_index": 1}),
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = _read_jsonl_lines(f)
        assert len(result) == 2
        assert result[0]["message_index"] == 0
        assert result[1]["message_index"] == 1

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert _read_jsonl_lines(tmp_path / "empty.jsonl") == []

    def test_skips_bad_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "mixed.jsonl"
        f.write_text(
            '{"message_type": "human"}\n'
            "not valid json\n"
            '{"message_type": "ai"}\n',
            encoding="utf-8",
        )
        result = _read_jsonl_lines(f)
        assert len(result) == 2

    def test_nonexistent_file_returns_empty(self) -> None:
        assert _read_jsonl_lines(Path("/nonexistent/file.jsonl")) == []


class TestGroupBySummarizedEvent:
    def test_groups_consecutive_same_timestamp(self) -> None:
        entries = [
            {"summarized_at": "2026-01-01T00:00:00", "message_type": "human"},
            {"summarized_at": "2026-01-01T00:00:00", "message_type": "ai"},
            {"summarized_at": "2026-01-02T00:00:00", "message_type": "human"},
        ]
        groups = _group_by_summarized_event(entries)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_empty_returns_empty(self) -> None:
        assert _group_by_summarized_event([]) == []

    def test_single_entry(self) -> None:
        groups = _group_by_summarized_event([{"summarized_at": "ts1"}])
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_missing_summarized_at_uses_none(self) -> None:
        entries = [
            {"message_type": "human"},
            {"message_type": "ai"},
        ]
        groups = _group_by_summarized_event(entries)
        assert len(groups) == 1
        assert len(groups[0]) == 2


class TestChunkMessages:
    def test_single_group_within_limit(self) -> None:
        groups = [
            [
                {
                    "summarized_at": "2026-01-01T00:00:00",
                    "message_type": "human",
                    "message": {"data": {"content": "short message"}},
                },
            ],
        ]
        chunks = _chunk_messages(groups, max_tokens=100000)
        assert len(chunks) == 1
        text, meta = chunks[0]
        assert "short message" in text
        assert meta["timestamp"] == "2026-01-01T00:00:00"

    def test_empty_groups_returns_empty(self) -> None:
        assert _chunk_messages([], max_tokens=1000) == []

    def test_skip_empty_content(self) -> None:
        groups = [
            [
                {
                    "summarized_at": "ts1",
                    "message_type": "human",
                    "message": {"data": {"content": ""}},
                },
            ],
        ]
        chunks = _chunk_messages(groups, max_tokens=100000)
        assert len(chunks) == 0

    def test_message_types_in_metadata(self) -> None:
        groups = [
            [
                {"summarized_at": "ts1", "message_type": "human", "message": {"data": {"content": "hi"}}},
                {"summarized_at": "ts1", "message_type": "ai", "message": {"data": {"content": "hello"}}},
            ],
        ]
        chunks = _chunk_messages(groups, max_tokens=100000)
        assert len(chunks) == 1
        text, meta = chunks[0]
        assert "User: hi" in text
        assert "Assistant: hello" in text
        assert "human" in meta["message_types"]
        assert "ai" in meta["message_types"]

    def test_line_range_in_metadata(self) -> None:
        groups = [
            [
                {"summarized_at": "ts1", "message_type": "human", "message": {"data": {"content": "msg1"}}},
                {"summarized_at": "ts1", "message_type": "ai", "message": {"data": {"content": "msg2"}}},
            ],
        ]
        chunks = _chunk_messages(groups, max_tokens=100000)
        assert len(chunks) == 1
        _, meta = chunks[0]
        assert "line_range" in meta

    def test_chunk_multiple_groups(self) -> None:
        groups = [
            [
                {"summarized_at": "ts1", "message_type": "human", "message": {"data": {"content": "first group"}}},
            ],
            [
                {"summarized_at": "ts2", "message_type": "ai", "message": {"data": {"content": "second group"}}},
            ],
        ]
        chunks = _chunk_messages(groups, max_tokens=100000)
        assert len(chunks) >= 1


class TestLoadOrCreateStorage:
    """Tests for _load_or_create_storage — directory permissions."""

    def test_index_dir_created_with_correct_permissions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the .history_index directory is created with mode 0o700."""
        import stat as stat_module

        # Mock LanceDBVectorStore and SimpleDocumentStore to avoid heavy deps
        monkeypatch.setattr(
            "semantic_search.history_indexer.LanceDBVectorStore",
            lambda **kw: type("FakeVectorStore", (), {"uri": kw.get("uri", "")})(),
        )
        monkeypatch.setattr(
            "semantic_search.history_indexer.SimpleDocumentStore",
            type("FakeDocStore", (), {"from_persist_path": staticmethod(lambda p: type("FakeDoc", (), {})())}),
        )

        from semantic_search.history_indexer import _load_or_create_storage

        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir(parents=True)

        index_dir = summaries_dir / ".history_index"
        assert not index_dir.exists()

        _load_or_create_storage(index_dir)

        assert index_dir.exists()
        assert index_dir.is_dir()

        # Verify permissions — mode should match 0o700
        mode = index_dir.stat().st_mode
        assert stat_module.S_IMODE(mode) == 0o700, (
            f"Expected 0o700, got {oct(stat_module.S_IMODE(mode))}"
        )


class TestHistoryIndexStats:
    def test_default_values(self) -> None:
        stats = HistoryIndexStats()
        assert stats.lines_read == 0
        assert stats.chunks_indexed == 0
        assert stats.files_processed == 0
        assert stats.errors == 0

    def test_custom_values(self) -> None:
        stats = HistoryIndexStats(lines_read=10, chunks_indexed=5, files_processed=2, errors=0)
        assert stats.lines_read == 10
        assert stats.chunks_indexed == 5
        assert stats.files_processed == 2

    def test_repr(self) -> None:
        stats = HistoryIndexStats(lines_read=5, chunks_indexed=3)
        r = repr(stats)
        assert "lines_read=5" in r
        assert "chunks_indexed=3" in r


class TestIndexAllHistoryFilesOnProgress:
    """Tests for the on_progress callback in index_all_history_files."""

    def test_on_progress_called_per_file(self, tmp_path: Path) -> None:
        """Verify callback fires once per .jsonl file."""
        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir(parents=True)

        # Create 3 small jsonl files
        for i in range(3):
            fp = summaries_dir / f"thread_{i}.jsonl"
            fp.write_text(
                json.dumps({"summarized_at": "ts", "message_index": 0,
                            "message_type": "human",
                            "message": {"data": {"content": f"hello {i}"}}})
                + "\n",
                encoding="utf-8",
            )

        calls: list[str] = []

        def on_progress(filename: str) -> None:
            calls.append(filename)

        from semantic_search.history_indexer import index_all_history_files
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=MockEmbedding(embed_dim=384),
            on_progress=on_progress,
        )

        assert len(calls) == 3
        assert stats.files_processed == 3

    def test_on_progress_filenames(self, tmp_path: Path) -> None:
        """Verify callback receives correct stem/filename."""
        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir(parents=True)

        fp = summaries_dir / "thread_alpha.jsonl"
        fp.write_text(
            json.dumps({"summarized_at": "ts", "message_index": 0,
                        "message_type": "human",
                        "message": {"data": {"content": "test"}}})
            + "\n",
            encoding="utf-8",
        )

        calls: list[str] = []

        def on_progress(filename: str) -> None:
            calls.append(filename)

        from semantic_search.history_indexer import index_all_history_files
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=MockEmbedding(embed_dim=384),
            on_progress=on_progress,
        )

        assert len(calls) == 1
        assert "thread_alpha" in calls[0]

    def test_on_progress_none_still_works(self, tmp_path: Path) -> None:
        """Verify backward compat: on_progress=None works normally."""
        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir(parents=True)

        fp = summaries_dir / "thread_z.jsonl"
        fp.write_text(
            json.dumps({"summarized_at": "ts", "message_index": 0,
                        "message_type": "human",
                        "message": {"data": {"content": "test"}}})
            + "\n",
            encoding="utf-8",
        )

        from semantic_search.history_indexer import index_all_history_files
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        # Should not raise
        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=MockEmbedding(embed_dim=384),
            on_progress=None,
        )
        assert stats.files_processed >= 1
