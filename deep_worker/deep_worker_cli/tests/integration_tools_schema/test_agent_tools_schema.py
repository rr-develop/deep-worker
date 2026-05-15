"""Integration test: verify all expected tools appear in agent schema.

Creates a real agent (with all middleware and CLI tools) in /tmp and
checks that every expected tool is available in the compiled graph's
ToolNode.  This catches regressions where tools are silently dropped
during agent construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from deep_worker_cli.tools import fetch_url, http_request

# ---------------------------------------------------------------------------
# Expected tools
# ---------------------------------------------------------------------------

# Tools provided by FilesystemMiddleware (always present)
FILESYSTEM_TOOLS = {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}

# Tools provided by deep_worker_cli/tools.py (always present in this test)
CLI_TOOLS = {"http_request", "fetch_url"}

# Tool provided by SubAgentMiddleware
SUBAGENT_TOOLS = {"task"}

# Tool provided by TodoListMiddleware
TODO_TOOLS = {"write_todos"}

# All tools expected when project_memory is NOT available (no index in /tmp)
# NOTE: send_message_to_agent is NOT included here — it is created by
# AgentProcess.make_send_message_to_agent_tool() and only added when
# running inside the process manager, not in create_cli_agent().
EXPECTED_TOOLS_NO_PROJECT_MEMORY = (
    FILESYSTEM_TOOLS | CLI_TOOLS | SUBAGENT_TOOLS | TODO_TOOLS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_working_dir() -> Iterator[Path]:
    """Provide a clean /tmp subdirectory with no semantic index."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dw_agent_tools_test_") as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentToolsSchema:
    """Agent creation and tool schema verification."""

    def _tool_names_from_graph(self, agent_graph) -> set[str]:
        """Extract tool names from the compiled agent graph's ToolNode."""
        tools_node = agent_graph.nodes["tools"]
        return set(tools_node.bound.tools_by_name.keys())

    def test_all_cli_tools_present(self, tmp_working_dir: Path) -> None:
        """All CLI tools should be registered when creating a local agent."""
        from langchain.chat_models import init_chat_model
        from deep_worker_cli.agent import create_cli_agent

        model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        tools = [http_request, fetch_url]

        agent, _backend = create_cli_agent(
            model=model,
            assistant_id="test_integration_tools",
            tools=tools,
            project_memory_dir=tmp_working_dir,
            enable_shell=True,
            enable_memory=False,
            enable_skills=False,
            auto_approve=True,
        )

        tool_names = self._tool_names_from_graph(agent)

        missing = EXPECTED_TOOLS_NO_PROJECT_MEMORY - tool_names
        assert not missing, (
            f"Agent is missing expected tools: {missing}\n"
            f"  expected: {sorted(EXPECTED_TOOLS_NO_PROJECT_MEMORY)}\n"
            f"  actual:   {sorted(tool_names)}"
        )

        # Also check no unexpected tools leaked in
        unexpected = tool_names - EXPECTED_TOOLS_NO_PROJECT_MEMORY
        if unexpected:
            print(f"NOTE: unexpected tools found (may be harmless): {sorted(unexpected)}")

    def test_project_memory_tool_present_when_index_exists(
        self, tmp_working_dir: Path,
    ) -> None:
        """project_memory tool should appear when a .vector_index exists."""
        from langchain.chat_models import init_chat_model
        from deep_worker_cli.agent import create_cli_agent

        # Create fake semantic index
        index_dir = tmp_working_dir / ".vector_index"
        index_dir.mkdir()
        (index_dir / "lancedb").mkdir()
        (index_dir / "docstore").mkdir()
        (index_dir / "docstore" / "docstore.json").write_text("{}")
        (index_dir / "file_hashes.json").write_text("{}")

        model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        tools = [http_request, fetch_url]

        agent, _backend = create_cli_agent(
            model=model,
            assistant_id="test_integration_tools_pm",
            tools=tools,
            project_memory_dir=tmp_working_dir,  # has .vector_index
            enable_shell=True,
            enable_memory=False,
            enable_skills=False,
            auto_approve=True,
        )

        tool_names = self._tool_names_from_graph(agent)

        expected_with_pm = EXPECTED_TOOLS_NO_PROJECT_MEMORY | {"project_memory"}
        missing = expected_with_pm - tool_names
        assert not missing, (
            f"Agent is missing expected tools (including project_memory): {missing}\n"
            f"  expected: {sorted(expected_with_pm)}\n"
            f"  actual:   {sorted(tool_names)}"
        )

    def test_project_memory_absent_without_index(
        self, tmp_working_dir: Path,
    ) -> None:
        """project_memory tool should NOT appear when no .vector_index."""
        from langchain.chat_models import init_chat_model
        from deep_worker_cli.agent import create_cli_agent

        model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        tools = [http_request, fetch_url]

        agent, _backend = create_cli_agent(
            model=model,
            assistant_id="test_integration_tools_no_pm",
            tools=tools,
            project_memory_dir=tmp_working_dir,  # no .vector_index
            enable_shell=True,
            enable_memory=False,
            enable_skills=False,
            auto_approve=True,
        )

        tool_names = self._tool_names_from_graph(agent)

        assert "project_memory" not in tool_names, (
            "project_memory tool registered despite no semantic index being present"
        )
