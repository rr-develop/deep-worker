"""Unit tests for semantic_search.retriever."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from semantic_search.exceptions import IndexNotFoundError
from semantic_search.retriever import SearchResult, search


class TestSearchResult:
    def test_required_fields(self) -> None:
        result = SearchResult(
            file_path="/repo/src/main.py",
            score=0.87,
            content="def hello(): pass",
            language="python",
            node_type="leaf",
        )
        assert result.file_path == "/repo/src/main.py"
        assert result.score == 0.87
        assert result.content == "def hello(): pass"
        assert result.language == "python"
        assert result.node_type == "leaf"

    def test_language_can_be_none(self) -> None:
        result = SearchResult(
            file_path="/repo/README.md",
            score=0.5,
            content="# Title",
            language=None,
            node_type="leaf",
        )
        assert result.language is None

    def test_node_type_leaf_or_parent(self) -> None:
        for nt in ("leaf", "parent"):
            result = SearchResult(
                file_path="/repo/a.py",
                score=0.9,
                content="x",
                language="python",
                node_type=nt,
            )
            assert result.node_type == nt


class TestSearchErrors:
    def test_raises_index_not_found_for_unindexed_repo(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(IndexNotFoundError):
            search("find something", repo_path=tmp_path)

    def test_raises_index_not_found_with_helpful_message(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(IndexNotFoundError, match=str(tmp_path)):
            search("find something", repo_path=tmp_path)


class TestSearchWithMocks:
    def _mock_node(self, text: str, file_path: str, language: str = "python") -> MagicMock:
        node = MagicMock()
        node.get_content.return_value = text
        node.metadata = {
            "file_path": file_path,
            "language": language,
        }
        node.score = 0.9
        return node

    def _mock_node_with_score(
        self, text: str, file_path: str, score: float = 0.9
    ) -> MagicMock:
        node_with_score = MagicMock()
        node_with_score.node = self._mock_node(text, file_path)
        node_with_score.score = score
        return node_with_score

    @patch("semantic_search.retriever.AutoMergingRetriever")
    @patch("semantic_search.retriever.StorageContext")
    @patch("semantic_search.retriever.VectorStoreIndex")
    @patch("semantic_search.retriever.LanceDBVectorStore")
    @patch("semantic_search.retriever.SimpleDocumentStore")
    def test_returns_list_of_search_results(
        self,
        mock_ds_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_sc_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Arrange: create .vector_index to pass exists() check
        (tmp_path / ".vector_index").mkdir()

        node_with_score = self._mock_node_with_score(
            "def hello(): pass", str(tmp_path / "main.py")
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [node_with_score]
        mock_retriever_cls.return_value = mock_retriever

        embed = MockEmbedding(embed_dim=16)

        # Act
        results = search("hello function", repo_path=tmp_path, embedding_model=embed)

        # Assert
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    @patch("semantic_search.retriever.AutoMergingRetriever")
    @patch("semantic_search.retriever.StorageContext")
    @patch("semantic_search.retriever.VectorStoreIndex")
    @patch("semantic_search.retriever.LanceDBVectorStore")
    @patch("semantic_search.retriever.SimpleDocumentStore")
    def test_result_has_correct_fields(
        self,
        mock_ds_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_sc_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / ".vector_index").mkdir()

        file_path = str(tmp_path / "main.py")
        node_with_score = self._mock_node_with_score("def hello(): pass", file_path, score=0.85)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [node_with_score]
        mock_retriever_cls.return_value = mock_retriever

        embed = MockEmbedding(embed_dim=16)

        results = search("hello", repo_path=tmp_path, embedding_model=embed)

        r = results[0]
        assert r.file_path == file_path
        assert r.content == "def hello(): pass"
        assert r.score == pytest.approx(0.85)
        assert r.language == "python"

    @patch("semantic_search.retriever.AutoMergingRetriever")
    @patch("semantic_search.retriever.StorageContext")
    @patch("semantic_search.retriever.VectorStoreIndex")
    @patch("semantic_search.retriever.LanceDBVectorStore")
    @patch("semantic_search.retriever.SimpleDocumentStore")
    def test_respects_top_k(
        self,
        mock_ds_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_sc_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / ".vector_index").mkdir()

        nodes = [
            self._mock_node_with_score(f"chunk {i}", str(tmp_path / f"f{i}.py"), score=0.9 - i * 0.1)
            for i in range(5)
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes[:3]
        mock_retriever_cls.return_value = mock_retriever

        embed = MockEmbedding(embed_dim=16)

        results = search("query", repo_path=tmp_path, top_k=3, embedding_model=embed)

        assert len(results) == 3

    @patch("semantic_search.retriever.AutoMergingRetriever")
    @patch("semantic_search.retriever.StorageContext")
    @patch("semantic_search.retriever.VectorStoreIndex")
    @patch("semantic_search.retriever.LanceDBVectorStore")
    @patch("semantic_search.retriever.SimpleDocumentStore")
    def test_empty_results_returns_empty_list(
        self,
        mock_ds_cls: MagicMock,
        mock_vs_cls: MagicMock,
        mock_index_cls: MagicMock,
        mock_sc_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        (tmp_path / ".vector_index").mkdir()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        embed = MockEmbedding(embed_dim=16)

        results = search("nonexistent thing", repo_path=tmp_path, embedding_model=embed)

        assert results == []
