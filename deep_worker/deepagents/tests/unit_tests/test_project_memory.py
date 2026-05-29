"""Unit tests for project memory middleware and tools.

Tests the project_memory tool, has_semantic_index utility, and
ProjectMemoryMiddleware functionality.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepagents.middleware.project_memory import COGNITIVE_HINT, ProjectMemoryMiddleware
from deepagents.tools.project_memory import project_memory
from deepagents.utils.project_index import has_semantic_index


# ============================================================================
# Tests for has_semantic_index utility
# ============================================================================


def test_has_semantic_index_with_valid_index() -> None:
    """Test has_semantic_index with a valid complete index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / ".vector_index"
        index_dir.mkdir()

        # Create all required components
        lancedb_dir = index_dir / "lancedb"
        lancedb_dir.mkdir()

        docstore_dir = index_dir / "docstore"
        docstore_dir.mkdir()
        (docstore_dir / "docstore.json").write_text("{}")

        (index_dir / "file_hashes.json").write_text("{}")

        # Should return True with all components present
        assert has_semantic_index(Path(tmpdir)) is True


def test_has_semantic_index_without_index() -> None:
    """Test has_semantic_index with no index directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No .vector_index directory created
        assert has_semantic_index(Path(tmpdir)) is False


def test_has_semantic_index_with_partial_index() -> None:
    """Test has_semantic_index with incomplete index (missing components)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / ".vector_index"
        index_dir.mkdir()

        # Only create lancedb, missing docstore and file_hashes
        lancedb_dir = index_dir / "lancedb"
        lancedb_dir.mkdir()

        # Should return False due to missing components
        assert has_semantic_index(Path(tmpdir)) is False


# ============================================================================
# Tests for project_memory tool
# ============================================================================


def test_project_memory_tool_success() -> None:
    """Test project_memory tool with successful search results."""
    # Mock runtime with state containing working_dir
    runtime = SimpleNamespace(
        state={"working_dir": "/path/to/test_project"},
        tool_call_id="test_call_1",
    )

    # Mock SearchResult
    mock_result = SimpleNamespace(
        file_path="/path/to/test_project/src/auth.py",
        score=0.892,
        language="python",
        node_type="parent",
        content="def authenticate(user, password):\n    # Auth implementation\n    ...",
    )

    with patch(
        "deepagents.tools.project_memory._ensure_semantic_search",
        return_value=(mock_search := MagicMock(), type("_", (Exception,), {})),
    ):
        mock_search.return_value = [mock_result]

        # Call the tool
        result = project_memory.func("authentication implementation", runtime=runtime, top_k=5)

        # Verify search was called correctly
        mock_search.assert_called_once_with(
            "authentication implementation",
            repo_path=Path("/path/to/test_project"),
            top_k=5,
        )

        # Verify output format
        assert "Project Memory: 'authentication implementation'" in result
        assert "Result 1 (score=0.892) [python]" in result
        assert "File: /path/to/test_project/src/auth.py" in result
        assert "Type: parent" in result
        assert "def authenticate(user, password):" in result


def test_project_memory_tool_index_not_found() -> None:
    """Test project_memory tool when index doesn't exist."""
    from semantic_search.exceptions import IndexNotFoundError

    runtime = SimpleNamespace(
        state={"working_dir": "/path/to/no_index"},
        tool_call_id="test_call_2",
    )

    with patch(
        "deepagents.tools.project_memory._ensure_semantic_search",
        return_value=(mock_search := MagicMock(), IndexNotFoundError),
    ):
        mock_search.side_effect = IndexNotFoundError("Index not found")

        result = project_memory.func("some query", runtime=runtime, top_k=5)

        # Should return error message with "Error:" prefix
        assert "Error: Project memory not available." in result
        assert "/path/to/no_index" in result
        assert "does not have a semantic search index" in result


def test_project_memory_tool_no_results() -> None:
    """Test project_memory tool when search returns no results."""
    runtime = SimpleNamespace(
        state={"working_dir": "/path/to/test_project"},
        tool_call_id="test_call_3",
    )

    with patch(
        "deepagents.tools.project_memory._ensure_semantic_search",
        return_value=(mock_search := MagicMock(), type("_", (Exception,), {})),
    ):
        mock_search.return_value = []  # No results

        result = project_memory.func("nonexistent feature", runtime=runtime, top_k=5)

        # Should return informative message
        assert "Project Memory: No results found for 'nonexistent feature'" in result
        assert "semantic index may not contain information" in result
        assert "grep or read_file" in result  # Suggests alternatives


def test_project_memory_tool_invalid_top_k() -> None:
    """Test project_memory tool with invalid top_k parameter."""
    runtime = SimpleNamespace(
        state={"working_dir": "/path/to/test_project"},
        tool_call_id="test_call_4",
    )

    # Test top_k too small
    result = project_memory.func("query", runtime=runtime, top_k=0)
    assert "Error: top_k must be between 1 and 20" in result

    # Test top_k too large
    result = project_memory.func("query", runtime=runtime, top_k=21)
    assert "Error: top_k must be between 1 and 20" in result


def test_project_memory_tool_runtime_fallback() -> None:
    """Test project_memory tool falls back to cwd when working_dir not in state."""
    # Runtime without working_dir in state
    runtime = SimpleNamespace(
        state={},  # No working_dir
        tool_call_id="test_call_5",
    )

    mock_result = SimpleNamespace(
        file_path="/current/dir/file.py",
        score=0.75,
        language="python",
        node_type="leaf",
        content="code here",
    )

    with patch(
        "deepagents.tools.project_memory._ensure_semantic_search",
        return_value=(mock_search := MagicMock(), type("_", (Exception,), {})),
    ):
        mock_search.return_value = [mock_result]

        with patch("deepagents.tools.project_memory.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/fallback/dir")

            result = project_memory.func("test query", runtime=runtime, top_k=3)

            # Should use Path.cwd() as fallback
            mock_search.assert_called_once_with(
                "test query",
                repo_path=Path("/fallback/dir"),
                top_k=3,
            )


# ============================================================================
# Tests for ProjectMemoryMiddleware
# ============================================================================


def test_middleware_registers_tool_with_index() -> None:
    """Test middleware registers project_memory tool when index exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / ".vector_index"
        index_dir.mkdir()

        # Create all required components
        (index_dir / "lancedb").mkdir()
        (index_dir / "docstore").mkdir()
        (index_dir / "docstore" / "docstore.json").write_text("{}")
        (index_dir / "file_hashes.json").write_text("{}")

        middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

        # Should have registered the tool
        assert middleware._has_index is True
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "project_memory"


def test_middleware_does_not_register_without_index() -> None:
    """Test middleware doesn't register tool when index is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No index directory created
        middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

        # Should NOT have registered any tools
        assert middleware._has_index is False
        assert len(middleware.tools) == 0


def test_middleware_injects_hint_with_index() -> None:
    """Test middleware injects cognitive hint when index exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / ".vector_index"
        index_dir.mkdir()
        (index_dir / "lancedb").mkdir()
        (index_dir / "docstore").mkdir()
        (index_dir / "docstore" / "docstore.json").write_text("{}")
        (index_dir / "file_hashes.json").write_text("{}")

        middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

        # Create mock request and handler
        original_system_message = SystemMessage(content="Original prompt")
        request = ModelRequest(
            model="test-model",
            messages=[],
            system_message=original_system_message,
        )

        # Mock handler that returns a response
        def mock_handler(req: ModelRequest) -> ModelResponse:
            # Check that system message was modified
            assert req.system_message is not None
            # The hint is wrapped in content_blocks with \n\n prefix
            content_str = str(req.system_message.content)
            assert "Project Knowledge Memory" in content_str
            assert "project_memory" in content_str  # Tool name mentioned in hint
            return ModelResponse(result=[])

        response = middleware.wrap_model_call(request, mock_handler)

        # Handler should have been called
        assert response is not None


def test_middleware_does_not_inject_hint_without_index() -> None:
    """Test middleware doesn't inject hint when index is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No index
        middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

        original_system_message = SystemMessage(content="Original prompt")
        request = ModelRequest(
            model="test-model",
            messages=[],
            system_message=original_system_message,
        )

        # Mock handler
        handler_called = False

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal handler_called
            handler_called = True
            # System message should be unchanged
            assert req.system_message == original_system_message
            assert "Project Knowledge Memory" not in str(req.system_message.content)
            return ModelResponse(result=[])

        response = middleware.wrap_model_call(request, mock_handler)

        assert handler_called
        assert response is not None


def test_middleware_sets_working_dir_in_state() -> None:
    """Test middleware sets working_dir in state via before_agent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / ".vector_index"
        index_dir.mkdir()
        (index_dir / "lancedb").mkdir()
        (index_dir / "docstore").mkdir()
        (index_dir / "docstore" / "docstore.json").write_text("{}")
        (index_dir / "file_hashes.json").write_text("{}")

        middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

        # Mock state and runtime
        state = {}
        runtime = SimpleNamespace()

        state_update = middleware.before_agent(state, runtime)

        # Should return state update with working_dir
        assert state_update is not None
        assert "working_dir" in state_update
        assert state_update["working_dir"] == str(Path(tmpdir))


def test_middleware_sets_working_dir_even_without_index() -> None:
    """Test middleware ALWAYS sets working_dir, even when the index is missing.

    This guards the intentional behaviour documented in
    ``ProjectMemoryMiddleware.before_agent`` ("CRITICAL FIX"): working_dir
    must be set regardless of index existence so tools/agents never fall
    back to ``Path.cwd()`` (often ``/`` for daemon processes). The earlier
    expectation that ``before_agent`` returns ``None`` without an index
    was reverted by that fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # No index
        middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

        state = {}
        runtime = SimpleNamespace()

        state_update = middleware.before_agent(state, runtime)

        # working_dir is set unconditionally, independent of index presence
        assert state_update is not None
        assert state_update["working_dir"] == str(Path(tmpdir))
