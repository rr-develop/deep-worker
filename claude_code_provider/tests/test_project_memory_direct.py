"""Tests for project_memory tool execution without LangGraph runtime.

TDD: Verifies that project_memory works when called with runtime=None
(direct MCP execution path). The tool should use a fallback working_dir
instead of crashing.
"""

from typing import Any  # type: ignore[reportUnusedImport]
from unittest.mock import MagicMock

import pytest  # type: ignore[reportUnusedImport]

from deepagents.tools.project_memory import project_memory


class TestProjectMemoryWithoutRuntime:
    """Test project_memory when runtime is None (MCP direct execution)."""

    def test_runtime_none_does_not_crash(self):
        """Tool should not return 'requires runtime context' when runtime=None."""
        # project_memory with runtime=None should attempt to search,
        # not immediately return an error
        result = project_memory.invoke({"query": "test", "runtime": None})
        # Should NOT contain the "requires runtime context" error
        assert "requires runtime context" not in result

    def test_runtime_none_with_working_dir_kwarg(self):
        """Tool should accept working_dir as direct kwarg when runtime=None."""
        # When called from MCP handler, working_dir can be passed directly
        result = project_memory.invoke({
            "query": "test",
            "runtime": None,
            "working_dir": "/tmp/example_project",
        })
        assert "requires runtime context" not in result

    def test_mock_runtime_with_state(self):
        """Tool should work with a simple mock runtime that has state."""
        mock_runtime = MagicMock()
        mock_runtime.state = {"working_dir": "/tmp/example_project"}

        result = project_memory.invoke({
            "query": "test",
            "runtime": mock_runtime,
        })
        assert "requires runtime context" not in result
