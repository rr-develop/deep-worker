"""Integration tests for project memory middleware.

Tests the full integration of ProjectMemoryMiddleware with agents,
including tool registration and the index-dependent cognitive hint.

The contract these tests pin changed with the shared-service migration
(design §5.0, blocker B1). Registration of ``project_memory`` used to be
conditional on a local ``.vector_index``; it no longer is. The tool now
takes an explicit repository, so an agent sitting in an unindexed
directory must still be able to query an indexed one — gating
registration on the local index made that impossible, and it also meant
the tool's availability depended on directory contents at construction
time.

What is index-dependent is the *hint*, not the *tool*: presence is
resolved against the service in ``abefore_agent`` and decides which
guidance goes into the system prompt.
"""

import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from deepagents.graph import create_agent
from deepagents.middleware.project_memory import (
    COGNITIVE_HINT,
    MISSING_INDEX_HINT,
    ProjectMemoryMiddleware,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


class _FakeStatus:
    def __init__(self, exists: bool, complete: bool) -> None:
        self.exists = exists
        self.complete = complete


class _FakeClient:
    """Stands in for the semantic-memory service.

    The middleware asks the service whether an index exists; these tests
    are about the middleware's reaction to that answer, so the answer is
    supplied directly rather than by starting a real service.
    """

    def __init__(self, *, exists: bool, complete: bool = True) -> None:
        self._status = _FakeStatus(exists, complete)

    async def index_status(self, kind, tenant):  # noqa: ANN001, ARG002
        return self._status

    async def list_tenants(self, kind):  # noqa: ANN001, ARG002
        return []


def _agent(middleware: ProjectMemoryMiddleware):
    return create_agent(
        model=GenericFakeChatModel(messages=iter([HumanMessage(content="Done")])),
        middleware=[middleware],
    )


class TestProjectMemoryIntegration:
    """Integration tests for ProjectMemoryMiddleware with agents."""

    def test_tool_is_registered_when_the_index_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / ".vector_index"
            index_dir.mkdir()
            (index_dir / "lancedb").mkdir()
            (index_dir / "docstore").mkdir()
            (index_dir / "docstore" / "docstore.json").write_text("{}")
            (index_dir / "file_hashes.json").write_text("{}")

            middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))
            agent = _agent(middleware)

            assert "project_memory" in agent.nodes["tools"].bound._tools_by_name
            assert len(middleware.tools) == 1
            assert middleware.tools[0].name == "project_memory"

    def test_tool_is_registered_even_without_a_local_index(self):
        """B1: an unindexed working directory must not remove the tool.

        This is the inversion of the old contract. The tool accepts an
        explicit ``repo_path``, so withholding it here would deny the
        agent access to every *other* indexed repository as well.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            middleware = ProjectMemoryMiddleware(working_dir=Path(tmpdir))
            agent = _agent(middleware)

            assert "project_memory" in agent.nodes["tools"].bound._tools_by_name
            assert len(middleware.tools) == 1

    @pytest.mark.asyncio
    async def test_index_presence_is_resolved_against_the_service(self):
        """``_has_index`` is an answer from the service, not a stat() call.

        It is therefore unset until ``abefore_agent`` runs — asserting on
        it straight after construction would pin a value that no longer
        means anything.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            middleware = ProjectMemoryMiddleware(
                working_dir=Path(tmpdir), client=_FakeClient(exists=True)
            )
            assert middleware._has_index is False

            await middleware.abefore_agent({"messages": []}, None)

            assert middleware._has_index is True
            assert middleware._cognitive_hint == COGNITIVE_HINT

    @pytest.mark.asyncio
    async def test_missing_index_swaps_the_hint_not_the_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            middleware = ProjectMemoryMiddleware(
                working_dir=Path(tmpdir), client=_FakeClient(exists=False)
            )

            await middleware.abefore_agent({"messages": []}, None)

            assert middleware._has_index is False
            assert middleware._cognitive_hint == MISSING_INDEX_HINT.format(
                tenants="(none)"
            )
            assert len(middleware.tools) == 1
