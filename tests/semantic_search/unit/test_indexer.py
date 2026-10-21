"""Unit tests for semantic_search.indexer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("llama_index")  # semantic-search extra; absent in CI

from deep_worker.providers.semantic_search.indexer import IndexStats, index_repository


@pytest.fixture(scope="module")
def _real_embed_model() -> object:
    """Real strictly-offline HuggingFaceEmbedding(intfloat/multilingual-e5-small).

    Module-scoped so the model is loaded once and reused. ``MockEmbedding``
    cannot be used here any more because the indexer now requires a real
    XLM-R tokenizer (RC-1 forbids the silent fallback to tiktoken).
    See docs/DESIGN_FIX_TOKENIZER_MISMATCH.md.
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


class TestIndexStats:
    def test_default_values_are_zero(self) -> None:
        stats = IndexStats()
        assert stats.added == 0
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.skipped == 0
        assert stats.total_nodes == 0

    def test_custom_values(self) -> None:
        stats = IndexStats(added=3, updated=1, deleted=2, skipped=10, total_nodes=50)
        assert stats.added == 3
        assert stats.updated == 1
        assert stats.deleted == 2
        assert stats.skipped == 10
        assert stats.total_nodes == 50

    def test_total_files_property(self) -> None:
        stats = IndexStats(added=3, updated=1, deleted=2, skipped=10)
        assert stats.total_files == 3 + 1 + 2 + 10

    def test_repr_contains_key_info(self) -> None:
        stats = IndexStats(added=5, skipped=20)
        r = repr(stats)
        assert "added" in r.lower() or "5" in r


class TestIndexRepositoryErrors:
    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            index_repository(tmp_path / "nonexistent")

    def test_raises_for_file_not_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1")
        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            index_repository(f)


class TestIndexRepositoryWithMocks:
    """Unit tests mocking the vector store while using a real XLM-R tokenizer."""

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_creates_index_dir(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        # Arrange: repo with one Python file
        (tmp_path / "main.py").write_text("def hello(): pass")

        # Act
        index_repository(tmp_path, embedding_model=_real_embed_model)

        # Assert: .vector_index directory was created
        assert (tmp_path / ".vector_index").is_dir()

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_stats_added_for_new_files(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        # Arrange
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")

        # Act
        stats = index_repository(tmp_path, embedding_model=_real_embed_model)

        # Assert: 2 new files
        assert stats.added == 2
        assert stats.updated == 0
        assert stats.deleted == 0

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_empty_repo_returns_zero_stats(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        stats = index_repository(tmp_path, embedding_model=_real_embed_model)

        assert stats.added == 0
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.skipped == 0

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_reset_wipes_existing_index(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        # Arrange: pre-existing index dir with a sentinel file
        index_dir = tmp_path / ".vector_index"
        index_dir.mkdir()
        sentinel = index_dir / "old_data.json"
        sentinel.write_text("{}")

        # Act
        index_repository(
            tmp_path,
            embedding_model=_real_embed_model,
            reset=True,
        )

        # Assert: sentinel was wiped
        assert not sentinel.exists()

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_skips_unchanged_files_on_second_run(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        embed = _real_embed_model
        (tmp_path / "main.py").write_text("def hello(): pass")

        # First run
        stats1 = index_repository(tmp_path, embedding_model=embed)
        assert stats1.added == 1

        # Second run — file unchanged
        stats2 = index_repository(tmp_path, embedding_model=embed)
        assert stats2.skipped == 1
        assert stats2.added == 0
        assert stats2.updated == 0

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_detects_modified_file(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        embed = _real_embed_model
        f = tmp_path / "main.py"
        f.write_text("def hello(): pass")

        index_repository(tmp_path, embedding_model=embed)

        # Modify the file
        f.write_text("def hello(): return 42")

        stats = index_repository(tmp_path, embedding_model=embed)
        assert stats.updated == 1
        assert stats.skipped == 0

    @patch("semantic_search.indexer.LanceDBVectorStore")
    @patch("semantic_search.indexer.VectorStoreIndex")
    @patch("semantic_search.indexer.SimpleDocumentStore")
    def test_detects_deleted_file(
        self,
        mock_docstore_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_vs_cls: MagicMock,
        tmp_path: Path,
        _real_embed_model: object,
    ) -> None:
        embed = _real_embed_model
        f = tmp_path / "main.py"
        f.write_text("def hello(): pass")

        index_repository(tmp_path, embedding_model=embed)

        # Delete the file
        f.unlink()

        stats = index_repository(tmp_path, embedding_model=embed)
        assert stats.deleted == 1
        assert stats.added == 0
