"""Integration tests for project memory middleware.

Tests the full integration of ProjectMemoryMiddleware with agents,
including tool registration and usage with real semantic search.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from deepagents.backends.state import StateBackend
from deepagents.graph import create_agent
from deepagents.middleware.project_memory import COGNITIVE_HINT, ProjectMemoryMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


class TestProjectMemoryIntegration:
    """Integration tests for ProjectMemoryMiddleware with agents."""

    def test_agent_with_project_memory(self):
        """Test agent with project_memory tool when index exists.

        Verifies that:
        - project_memory tool is registered and available
        - middleware properly configures the agent with the tool
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid semantic index structure
            index_dir = Path(tmpdir) / ".vector_index"
            index_dir.mkdir()
            (index_dir / "lancedb").mkdir()
            (index_dir / "docstore").mkdir()
            (index_dir / "docstore" / "docstore.json").write_text("{}")
            (index_dir / "file_hashes.json").write_text("{}")

            # Create agent with ProjectMemoryMiddleware
            middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

            agent = create_agent(
                model=GenericFakeChatModel(
                    messages=iter([HumanMessage(content="Done")])
                ),
                middleware=[middleware],
            )

            # Verify tool is registered
            assert "project_memory" in agent.nodes["tools"].bound._tools_by_name.keys()

            # Verify middleware detected the index
            assert middleware._has_index is True
            assert len(middleware.tools) == 1
            assert middleware.tools[0].name == "project_memory"

    def test_agent_without_project_memory(self):
        """Test agent without project_memory tool when index is missing.

        Verifies that:
        - project_memory tool is NOT registered
        - middleware does not configure the tool when index is absent
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # No index directory created - no semantic index available

            # Create agent with ProjectMemoryMiddleware (but no index)
            middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))

            agent = create_agent(
                model=GenericFakeChatModel(
                    messages=iter([HumanMessage(content="Done")])
                ),
                middleware=[middleware],
            )

            # Verify middleware did not detect index
            assert middleware._has_index is False
            assert len(middleware.tools) == 0

            # If tools node exists, verify project_memory is not there
            if "tools" in agent.nodes:
                assert "project_memory" not in agent.nodes["tools"].bound._tools_by_name.keys()

