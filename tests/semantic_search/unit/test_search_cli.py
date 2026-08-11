"""Unit tests for semantic_search.search_cli."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_worker.providers.semantic_search.exceptions import IndexNotFoundError
from deep_worker.providers.semantic_search.search_cli import (
    find_vector_index,
    load_query_from_file,
    main,
    parse_args,
)


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_parse_query_string(self) -> None:
        """Test parsing -q with query string."""
        args = parse_args(["-q", "find authentication code"])
        assert args.query == "find authentication code"
        assert args.query_file is None
        assert args.directory is None

    def test_parse_query_file(self) -> None:
        """Test parsing -qf with query file path."""
        args = parse_args(["-qf", "/path/to/query.txt"])
        assert args.query is None
        assert args.query_file == Path("/path/to/query.txt")
        assert args.directory is None

    def test_parse_with_directory(self) -> None:
        """Test parsing with optional -d directory."""
        args = parse_args(["-q", "test", "-d", "/repo/path"])
        assert args.query == "test"
        assert args.directory == Path("/repo/path")

    def test_parse_with_top_k(self) -> None:
        """Test parsing with optional -k top_k."""
        args = parse_args(["-q", "test", "-k", "10"])
        assert args.query == "test"
        assert args.top_k == 10

    def test_default_top_k_is_5(self) -> None:
        """Test default top_k value."""
        args = parse_args(["-q", "test"])
        assert args.top_k == 5

    def test_requires_query_or_query_file(self) -> None:
        """Test that either -q or -qf is required."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_cannot_use_both_q_and_qf(self) -> None:
        """Test that -q and -qf are mutually exclusive."""
        with pytest.raises(SystemExit):
            parse_args(["-q", "query", "-qf", "file.txt"])


class TestLoadQueryFromFile:
    """Test loading query text from a file."""

    def test_load_simple_query(self, tmp_path: Path) -> None:
        """Test loading a simple query from a file."""
        query_file = tmp_path / "query.txt"
        query_file.write_text("find authentication logic", encoding="utf-8")

        result = load_query_from_file(query_file)
        assert result == "find authentication logic"

    def test_load_multiline_query(self, tmp_path: Path) -> None:
        """Test loading a multiline query (should be joined)."""
        query_file = tmp_path / "query.txt"
        query_file.write_text("find code that handles\nuser authentication\nand session management", encoding="utf-8")

        result = load_query_from_file(query_file)
        assert result == "find code that handles user authentication and session management"

    def test_strip_whitespace(self, tmp_path: Path) -> None:
        """Test that leading/trailing whitespace is stripped."""
        query_file = tmp_path / "query.txt"
        query_file.write_text("  \n  find something  \n  ", encoding="utf-8")

        result = load_query_from_file(query_file)
        assert result == "find something"

    def test_raises_on_nonexistent_file(self, tmp_path: Path) -> None:
        """Test error when query file doesn't exist."""
        query_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="Query file not found"):
            load_query_from_file(query_file)

    def test_raises_on_empty_file(self, tmp_path: Path) -> None:
        """Test error when query file is empty."""
        query_file = tmp_path / "empty.txt"
        query_file.write_text("   \n  ", encoding="utf-8")

        with pytest.raises(ValueError, match="Query file is empty"):
            load_query_from_file(query_file)


class TestFindVectorIndex:
    """Test vector index discovery."""

    def test_finds_index_in_current_directory(self, tmp_path: Path) -> None:
        """Test finding index in the specified directory."""
        index_dir = tmp_path / ".vector_index"
        index_dir.mkdir()

        result = find_vector_index(tmp_path)
        assert result == tmp_path

    def test_finds_index_in_parent_directory(self, tmp_path: Path) -> None:
        """Test finding index by walking up the directory tree."""
        # Create structure: tmp_path/.vector_index and tmp_path/subdir/
        index_dir = tmp_path / ".vector_index"
        index_dir.mkdir()
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = find_vector_index(subdir)
        assert result == tmp_path

    def test_finds_index_in_grandparent(self, tmp_path: Path) -> None:
        """Test finding index multiple levels up."""
        index_dir = tmp_path / ".vector_index"
        index_dir.mkdir()
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)

        result = find_vector_index(deep_dir)
        assert result == tmp_path

    def test_raises_when_no_index_found(self, tmp_path: Path) -> None:
        """Test error when no index is found anywhere."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with pytest.raises(IndexNotFoundError, match="No vector index found"):
            find_vector_index(subdir)

    def test_stops_at_filesystem_root(self, tmp_path: Path) -> None:
        """Test that search stops at filesystem root."""
        # This test ensures we don't search forever
        with pytest.raises(IndexNotFoundError):
            find_vector_index(tmp_path)


class TestMain:
    """Test main CLI entry point."""

    @patch("semantic_search.search_cli.search")
    def test_main_with_query_string(
        self,
        mock_search: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test main() with -q argument."""
        # Setup: create index directory
        (tmp_path / ".vector_index").mkdir()
        mock_search.return_value = [
            MagicMock(
                file_path=str(tmp_path / "main.py"),
                score=0.92,
                content="def authenticate(user, password):\n    pass",
                language="python",
                node_type="leaf",
            )
        ]

        # Act
        sys.argv = ["search_cli", "-q", "authentication", "-d", str(tmp_path)]
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_search.assert_called_once()
        captured = capsys.readouterr()
        assert "main.py" in captured.out
        assert "92.0%" in captured.out

    @patch("semantic_search.search_cli.search")
    def test_main_with_query_file(
        self,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test main() with -qf argument."""
        # Setup
        (tmp_path / ".vector_index").mkdir()
        query_file = tmp_path / "query.txt"
        query_file.write_text("find authentication", encoding="utf-8")
        mock_search.return_value = []

        # Act
        sys.argv = ["search_cli", "-qf", str(query_file), "-d", str(tmp_path)]
        exit_code = main()

        # Assert
        assert exit_code == 0
        mock_search.assert_called_once()
        # Check that the query was loaded from file
        call_args = mock_search.call_args
        assert call_args is not None
        # call_args.kwargs contains keyword arguments
        assert call_args.kwargs["query"] == "find authentication"

    @patch("semantic_search.search_cli.search")
    def test_main_finds_index_automatically(
        self,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that main() finds index by walking up the tree."""
        # Setup: index in parent, run from subdirectory
        (tmp_path / ".vector_index").mkdir()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        mock_search.return_value = []

        # Act: run from subdir without -d
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            sys.argv = ["search_cli", "-q", "test"]
            exit_code = main()
        finally:
            os.chdir(original_cwd)

        # Assert: should find parent's index
        assert exit_code == 0
        call_args = mock_search.call_args
        assert call_args is not None
        assert call_args[1]["repo_path"] == tmp_path

    def test_main_exits_1_when_no_index_found(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that main() exits with code 1 when no index found."""
        sys.argv = ["search_cli", "-q", "test", "-d", str(tmp_path)]
        exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No vector index found" in captured.err

    def test_main_exits_1_on_file_not_found(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that main() exits with code 1 for missing query file."""
        sys.argv = ["search_cli", "-qf", str(tmp_path / "missing.txt")]
        exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Query file not found" in captured.err

    @patch("semantic_search.search_cli.search")
    def test_main_shows_no_results_message(
        self,
        mock_search: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test message when no results are found."""
        (tmp_path / ".vector_index").mkdir()
        mock_search.return_value = []

        sys.argv = ["search_cli", "-q", "nonexistent", "-d", str(tmp_path)]
        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No results found" in captured.out
