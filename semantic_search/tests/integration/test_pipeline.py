"""Integration tests for the full index + search pipeline.

Uses MockEmbedding — no real model download required.
Marked with pytest.mark.integration for optional selective running.
"""

from __future__ import annotations

import pytest
import subprocess
from pathlib import Path

from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from semantic_search.exceptions import IndexNotFoundError
from semantic_search.indexer import IndexStats, index_repository
from semantic_search.retriever import SearchResult, search


EMBED_DIM = 32  # small dimension for tests


def _init_git_repo(repo_path: Path) -> None:
    """Initialise *repo_path* as a minimal git repository.

    ``scanner.scan_files`` requires the repository to be a git repo so it
    can honour ``.gitignore``. Tests build a fake repo on-disk and must
    therefore run ``git init`` (+ a quick ``git add -A``) before calling
    the indexer. We deliberately leave the files *unstaged* — the scanner
    accepts untracked-but-not-ignored files just fine, and avoiding a
    real ``git commit`` keeps the fixture fast and isolated from
    ``user.name``/``user.email`` global config.
    """
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def mock_embed() -> MockEmbedding:
    return MockEmbedding(embed_dim=EMBED_DIM)


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    """A small fake Python repository (real git repo on disk)."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n\n"
        "def subtract(a: int, b: int) -> int:\n"
        "    \"\"\"Subtract b from a.\"\"\"\n"
        "    return a - b\n"
    )
    (src / "utils.py").write_text(
        "def format_result(value: float) -> str:\n"
        "    \"\"\"Format a numeric result for display.\"\"\"\n"
        "    return f'{value:.2f}'\n"
    )
    (tmp_path / "README.md").write_text(
        "# Calculator\n\nA simple calculator library.\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'calculator'\nversion = '0.1.0'\n"
    )
    _init_git_repo(tmp_path)
    return tmp_path


@pytest.fixture
def mixed_repo(tmp_path: Path) -> Path:
    """A repository with multiple languages (real git repo on disk)."""
    (tmp_path / "app.py").write_text("def main(): pass\n")
    (tmp_path / "index.ts").write_text("export function hello(): string { return 'hi'; }\n")
    (tmp_path / "Main.java").write_text(
        "public class Main {\n    public static void main(String[] args) {}\n}\n"
    )
    (tmp_path / "README.md").write_text("# Mixed repo\n")
    _init_git_repo(tmp_path)
    return tmp_path


class TestIndexRepository:
    def test_creates_vector_index_directory(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)
        assert (python_repo / ".vector_index").is_dir()

    def test_returns_index_stats(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        stats = index_repository(python_repo, embedding_model=mock_embed)
        assert isinstance(stats, IndexStats)

    def test_indexes_all_files(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        stats = index_repository(python_repo, embedding_model=mock_embed)
        # calculator.py, utils.py, README.md, pyproject.toml = 4 files
        assert stats.added == 4
        assert stats.updated == 0
        assert stats.deleted == 0

    def test_creates_nodes(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        stats = index_repository(python_repo, embedding_model=mock_embed)
        assert stats.total_nodes > 0

    def test_incremental_skips_unchanged_files(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        stats2 = index_repository(python_repo, embedding_model=mock_embed)

        assert stats2.skipped == 4  # all 4 files unchanged
        assert stats2.added == 0
        assert stats2.updated == 0

    def test_incremental_detects_new_file(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        # Add a new file
        (python_repo / "src" / "new_module.py").write_text("def new_func(): pass\n")

        stats2 = index_repository(python_repo, embedding_model=mock_embed)

        assert stats2.added == 1
        assert stats2.skipped == 4

    def test_incremental_detects_modified_file(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        # Modify a file
        calc = python_repo / "src" / "calculator.py"
        calc.write_text(calc.read_text() + "\ndef multiply(a, b): return a * b\n")

        stats2 = index_repository(python_repo, embedding_model=mock_embed)

        assert stats2.updated == 1
        assert stats2.skipped == 3

    def test_incremental_detects_deleted_file(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        # Delete a file
        (python_repo / "src" / "utils.py").unlink()

        stats2 = index_repository(python_repo, embedding_model=mock_embed)

        assert stats2.deleted == 1
        assert stats2.skipped == 3

    def test_reset_reindexes_everything(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        stats2 = index_repository(python_repo, embedding_model=mock_embed, reset=True)

        # After reset, all files are "new" again
        assert stats2.added == 4
        assert stats2.skipped == 0

    def test_mixed_language_repo(
        self, mixed_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        stats = index_repository(mixed_repo, embedding_model=mock_embed)
        # app.py, index.ts, Main.java, README.md = 4 files
        assert stats.added == 4


class TestSearch:
    def test_raises_for_unindexed_repo(
        self, tmp_path: Path, mock_embed: MockEmbedding
    ) -> None:
        with pytest.raises(IndexNotFoundError):
            search("hello", repo_path=tmp_path, embedding_model=mock_embed)

    def test_returns_list_of_search_results(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        results = search("add two numbers", repo_path=python_repo, embedding_model=mock_embed)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_have_required_fields(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        results = search("calculator", repo_path=python_repo, top_k=1, embedding_model=mock_embed)

        assert len(results) >= 1
        r = results[0]
        assert isinstance(r.file_path, str)
        assert isinstance(r.score, float)
        assert isinstance(r.content, str)
        assert r.node_type in ("leaf", "parent")

    def test_top_k_limits_results(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        results_3 = search("function", repo_path=python_repo, top_k=3, embedding_model=mock_embed)
        results_1 = search("function", repo_path=python_repo, top_k=1, embedding_model=mock_embed)

        assert len(results_3) <= 3
        assert len(results_1) <= 1

    def test_results_sorted_by_score_descending(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        results = search("add subtract", repo_path=python_repo, top_k=5, embedding_model=mock_embed)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_file_paths_are_absolute(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        results = search("function", repo_path=python_repo, top_k=3, embedding_model=mock_embed)

        for r in results:
            assert Path(r.file_path).is_absolute(), f"Not absolute: {r.file_path}"

    def test_language_metadata_preserved(
        self, python_repo: Path, mock_embed: MockEmbedding
    ) -> None:
        index_repository(python_repo, embedding_model=mock_embed)

        results = search("def add", repo_path=python_repo, top_k=5, embedding_model=mock_embed)

        # At least one Python result should have language=python
        python_results = [r for r in results if r.language == "python"]
        assert len(python_results) >= 1
