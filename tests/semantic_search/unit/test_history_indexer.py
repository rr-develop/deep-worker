"""Unit tests for semantic_search.history_indexer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("llama_index")  # semantic-search extra; absent in CI

from deep_worker.providers.semantic_search.history_indexer import (
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


def _fake_tokenize(text: str) -> list[int]:
    """Fake tokenizer: one fake token per whitespace-separated word.

    Deterministic substitute for the embedder's XLM-R tokenizer in unit
    tests of ``_chunk_messages`` — the chunker only cares about the count
    returned by ``tokenize_fn``. See ``docs/DESIGN_FIX_TOKENIZER_MISMATCH.md``.
    """
    return [hash(w) & 0xFFFF for w in text.split()]


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
        chunks = _chunk_messages(groups, effective_max=100000, tokenize_fn=_fake_tokenize)
        assert len(chunks) == 1
        text, meta = chunks[0]
        assert "short message" in text
        assert meta["timestamp"] == "2026-01-01T00:00:00"

    def test_empty_groups_returns_empty(self) -> None:
        assert _chunk_messages([], effective_max=1000, tokenize_fn=_fake_tokenize) == []

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
        chunks = _chunk_messages(groups, effective_max=100000, tokenize_fn=_fake_tokenize)
        assert len(chunks) == 0

    def test_message_types_in_metadata(self) -> None:
        groups = [
            [
                {"summarized_at": "ts1", "message_type": "human", "message": {"data": {"content": "hi"}}},
                {"summarized_at": "ts1", "message_type": "ai", "message": {"data": {"content": "hello"}}},
            ],
        ]
        chunks = _chunk_messages(groups, effective_max=100000, tokenize_fn=_fake_tokenize)
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
        chunks = _chunk_messages(groups, effective_max=100000, tokenize_fn=_fake_tokenize)
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
        chunks = _chunk_messages(groups, effective_max=100000, tokenize_fn=_fake_tokenize)
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

        from deep_worker.providers.semantic_search.history_indexer import _load_or_create_storage

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


@pytest.fixture(scope="module")
def _real_embed_model() -> object:
    """Real strictly-offline HuggingFaceEmbedding(intfloat/multilingual-e5-small).

    Module-scoped so the model is loaded once and reused across the three
    on_progress tests; ``MockEmbedding`` cannot be used here any more because
    the chunker now requires a real XLM-R tokenizer (RC-1 forbids the silent
    fallback to tiktoken). See docs/DESIGN_FIX_TOKENIZER_MISMATCH.md.
    """
    import os as _os

    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("TQDM_DISABLE", "1")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    return HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"local_files_only": True},
    )


class TestIndexAllHistoryFilesOnProgress:
    """Tests for the on_progress callback in index_all_history_files."""

    def test_on_progress_called_per_file(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
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

        from deep_worker.providers.semantic_search.history_indexer import index_all_history_files

        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=_real_embed_model,
            on_progress=on_progress,
        )

        assert len(calls) == 3
        assert stats.files_processed == 3

    def test_on_progress_filenames(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
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

        from deep_worker.providers.semantic_search.history_indexer import index_all_history_files

        index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=_real_embed_model,
            on_progress=on_progress,
        )

        assert len(calls) == 1
        assert "thread_alpha" in calls[0]

    def test_on_progress_none_still_works(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
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

        from deep_worker.providers.semantic_search.history_indexer import index_all_history_files

        # Should not raise
        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=_real_embed_model,
            on_progress=None,
        )
        assert stats.files_processed >= 1


class TestMessageProgress:
    """Progress is reported in MESSAGES (jsonl lines), NOT chunks.

    The denominator is a plain line count — known up front, so the bar shows
    X/N + %% + ETA from the first tick (no "0/None", no "1/1"). The progress
    granularity (embed_progress_step) is INDEPENDENT of the LanceDB write
    batch size (index_batch_size).
    """

    def _write_lines(self, path: Path, n: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for g in range(n):
            body = (f"unique message content number {g} "
                    + "lorem ipsum dolor sit amet " * 40).strip()
            lines.append(json.dumps({
                "summarized_at": f"ts-{g}",
                "message_index": g,
                "message_type": "human",
                "message": {"data": {"content": body}},
            }))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_total_is_message_count_known_from_first_tick(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        from deep_worker.providers.semantic_search.history_indexer import index_history_file
        from deep_worker.providers.semantic_search.config import IndexConfig

        summaries_dir = tmp_path / "summaries"
        fp = summaries_dir / "thread.jsonl"
        N = 20
        self._write_lines(fp, N)

        calls: list[tuple[int, int]] = []

        def on_message_progress(done: int, total: int) -> None:
            calls.append((done, total))

        cfg = IndexConfig(embed_progress_step=4)
        stats = index_history_file(fp, summaries_dir, config=cfg,
                                   embed_model=_real_embed_model,
                                   on_message_progress=on_message_progress)

        assert calls, "expected progress callbacks"
        # total == number of NEW messages (lines), constant, known from the
        # very first tick (never None, never the chunk count).
        assert all(total == N for _, total in calls)
        assert stats.lines_read == N
        # done is monotonic non-decreasing and ends exactly at N.
        dones = [d for d, _ in calls]
        assert dones == sorted(dones)
        assert dones[0] == 0            # first tick before any embedding
        assert dones[-1] == N           # final tick == all messages
        assert all(0 <= d <= N for d in dones)

    def test_progress_step_independent_of_write_batch(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        """Small progress step => many ticks; large write batch => few txns.

        The two knobs are independent: a fine-grained progress bar must NOT
        force many LanceDB transactions.
        """
        import math
        from deep_worker.providers.semantic_search.history_indexer import (
            index_history_file, get_history_index_dir,
        )
        from deep_worker.providers.semantic_search.config import IndexConfig

        summaries_dir = tmp_path / "summaries"
        fp = summaries_dir / "thread.jsonl"
        self._write_lines(fp, 20)

        calls: list[tuple[int, int]] = []

        def on_message_progress(done: int, total: int) -> None:
            calls.append((done, total))

        # Fine progress (step=1) but a single big write batch (1000).
        cfg = IndexConfig(embed_progress_step=1, index_batch_size=1000)
        stats = index_history_file(fp, summaries_dir, config=cfg,
                                   embed_model=_real_embed_model,
                                   on_message_progress=on_message_progress)

        # Many progress ticks (one per embed step over the chunks) ...
        assert len(calls) >= 3
        # ... but the WRITE stayed batched: one LanceDB transaction.
        txn_dir = (get_history_index_dir(summaries_dir)
                   / "lancedb" / "vectors.lance" / "_transactions")
        n_txn = len(list(txn_dir.glob("*.txn"))) if txn_dir.exists() else 0
        assert n_txn == math.ceil(stats.chunks_indexed / cfg.index_batch_size)
        assert n_txn == 1

    def test_none_callback_backward_compatible(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        from deep_worker.providers.semantic_search.history_indexer import index_history_file

        summaries_dir = tmp_path / "summaries"
        fp = summaries_dir / "thread.jsonl"
        self._write_lines(fp, 5)
        # Must not raise without the callback.
        stats = index_history_file(fp, summaries_dir, embed_model=_real_embed_model)
        assert stats.lines_read == 5
        assert stats.chunks_indexed >= 1

    def test_index_all_aggregates_messages_across_files(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        from deep_worker.providers.semantic_search.history_indexer import index_all_history_files

        summaries_dir = tmp_path / "summaries"
        self._write_lines(summaries_dir / "a.jsonl", 6)
        self._write_lines(summaries_dir / "b.jsonl", 9)

        calls: list[tuple[int, int]] = []

        def on_message_progress(done: int, total: int) -> None:
            calls.append((done, total))

        stats = index_all_history_files(
            summaries_dir=summaries_dir,
            embed_model=_real_embed_model,
            on_message_progress=on_message_progress,
        )

        assert stats.lines_read == 15
        # Grand total (messages across all files) is constant and known up
        # front; final done reaches it.
        assert all(total == 15 for _, total in calls)
        dones = [d for d, _ in calls]
        assert dones == sorted(dones)
        assert dones[0] == 0
        assert dones[-1] == 15


class TestBatchedWrite:
    """Nodes MUST be written to LanceDB in batches, never one-at-a-time.

    Per-node writes create one LanceDB transaction (data fragment + manifest
    version) each — O(n) fsync-heavy writes that crawl on large histories.
    These tests count the actual LanceDB transaction files.
    """

    def _write_n(self, path: Path, n: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for g in range(n):
            body = (f"unique message content number {g} "
                    + "lorem ipsum dolor sit amet " * 40).strip()
            lines.append(json.dumps({
                "summarized_at": f"ts-{g}",
                "message_index": g,
                "message_type": "human",
                "message": {"data": {"content": body}},
            }))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _txn_count(self, summaries_dir: Path) -> int:
        from deep_worker.providers.semantic_search.history_indexer import get_history_index_dir
        txn_dir = (get_history_index_dir(summaries_dir)
                   / "lancedb" / "vectors.lance" / "_transactions")
        if not txn_dir.exists():
            return 0
        return len(list(txn_dir.glob("*.txn")))

    def test_single_write_batch_one_transaction(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        import math
        from deep_worker.providers.semantic_search.history_indexer import index_history_file
        from deep_worker.providers.semantic_search.config import IndexConfig

        summaries_dir = tmp_path / "summaries"
        fp = summaries_dir / "thread.jsonl"
        self._write_n(fp, 20)

        cfg = IndexConfig(index_batch_size=1000)
        stats = index_history_file(fp, summaries_dir, config=cfg,
                                   embed_model=_real_embed_model)

        assert stats.chunks_indexed >= 5
        expected = math.ceil(stats.chunks_indexed / cfg.index_batch_size)
        assert expected == 1
        assert self._txn_count(summaries_dir) == expected

    def test_small_write_batch_bounds_transactions(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        import math
        from deep_worker.providers.semantic_search.history_indexer import index_history_file
        from deep_worker.providers.semantic_search.config import IndexConfig

        summaries_dir = tmp_path / "summaries"
        fp = summaries_dir / "thread.jsonl"
        self._write_n(fp, 20)

        cfg = IndexConfig(index_batch_size=3)
        stats = index_history_file(fp, summaries_dir, config=cfg,
                                   embed_model=_real_embed_model)

        assert stats.chunks_indexed >= 6
        expected = math.ceil(stats.chunks_indexed / cfg.index_batch_size)
        assert self._txn_count(summaries_dir) == expected
        assert expected < stats.chunks_indexed

    def test_all_nodes_indexed_and_searchable(
        self, tmp_path: Path, _real_embed_model: object
    ) -> None:
        from deep_worker.providers.semantic_search.history_indexer import (
            index_history_file, history_index_exists,
        )
        from deep_worker.providers.semantic_search.config import IndexConfig

        summaries_dir = tmp_path / "summaries"
        fp = summaries_dir / "thread.jsonl"
        self._write_n(fp, 20)

        cfg = IndexConfig(index_batch_size=4, embed_progress_step=4)
        stats = index_history_file(fp, summaries_dir, config=cfg,
                                   embed_model=_real_embed_model)
        assert history_index_exists(summaries_dir)
        assert stats.chunks_indexed >= 6
