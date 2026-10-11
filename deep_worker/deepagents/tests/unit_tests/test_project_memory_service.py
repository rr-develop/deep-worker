from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deepagents.middleware.project_memory import ProjectMemoryMiddleware
from deepagents.tools.project_memory import project_memory


@pytest.mark.asyncio
async def test_tool_validates_before_any_rpc() -> None:
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient"
    ) as client_cls:
        result = await project_memory.coroutine("q", top_k=0)
    assert result == "Error: top_k must be between 1 and 20"
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_tool_uses_status_then_search(tmp_path: Path) -> None:
    client = SimpleNamespace(
        index_status=AsyncMock(return_value=SimpleNamespace(exists=True, complete=True)),
        search_project=AsyncMock(return_value="service result"),
    )
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ):
        result = await project_memory.coroutine(
            "q", runtime=SimpleNamespace(state={"working_dir": str(tmp_path)})
        )
    assert result == "service result"
    client.search_project.assert_awaited_once_with("q", tmp_path.resolve(), 5)


@pytest.mark.asyncio
async def test_tool_renders_missing_index_with_tenants(tmp_path: Path) -> None:
    client = SimpleNamespace(
        index_status=AsyncMock(return_value=SimpleNamespace(exists=False, complete=False)),
        list_tenants=AsyncMock(
            return_value=SimpleNamespace(tenants=["/path/to/repo_a"], total=1)
        ),
        search_project=AsyncMock(),
    )
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ):
        result = await project_memory.coroutine(
            "q", repo_path=str(tmp_path)
        )
    assert "no semantic index" in result
    assert "/path/to/repo_a" in result
    client.search_project.assert_not_awaited()


def _missing_index_client(tenants: list[str] | None = None) -> SimpleNamespace:
    listed = ["/path/to/repo_a"] if tenants is None else tenants
    return SimpleNamespace(
        index_status=AsyncMock(
            return_value=SimpleNamespace(exists=False, complete=False)
        ),
        list_tenants=AsyncMock(
            return_value=SimpleNamespace(tenants=listed, total=len(listed))
        ),
        search_project=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_missing_index_message_for_an_explicit_repo_is_not_about_cwd(
    tmp_path: Path,
) -> None:
    """An explicitly named repository is not "the agent's working directory".

    Saying so is false, and the advice that follows it — "pass repo_path
    explicitly" — is what the agent has just done. The message has to
    send it somewhere it can actually go: the list of indexed targets.
    """
    client = _missing_index_client()
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ):
        result = await project_memory.coroutine("q", repo_path=str(tmp_path))

    assert "agent's working directory" not in result
    assert "repo_path=" not in result
    assert "That repository has not been indexed" in result
    assert "/path/to/repo_a" in result


@pytest.mark.asyncio
async def test_explicit_repo_message_promises_no_list_when_there_is_none(
    tmp_path: Path,
) -> None:
    """"Pick one of the repositories below" needs a below to point at.

    A host with no index at all is an ordinary state, not an edge case,
    and there the invitation to choose is the same species of lie as the
    working-directory claim: it describes something the reader cannot see.
    """
    client = _missing_index_client(tenants=[])
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ):
        result = await project_memory.coroutine("q", repo_path=str(tmp_path))

    assert "below" not in result
    assert "There are no indexed repositories on this host right now." in result
    assert "That repository has not been indexed" in result
    assert "agent's working directory" not in result


@pytest.mark.asyncio
async def test_missing_index_message_for_the_default_target_still_names_cwd(
    tmp_path: Path,
) -> None:
    """The defaulted case keeps its original wording and its remedy."""
    client = _missing_index_client()
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ):
        result = await project_memory.coroutine(
            "q", runtime=SimpleNamespace(state={"working_dir": str(tmp_path)})
        )

    assert "This is the agent's working directory" in result
    assert 'repo_path="<path-to-repo>"' in result
    assert "/path/to/repo_a" in result


def test_middleware_registers_without_local_index(tmp_path: Path) -> None:
    middleware = ProjectMemoryMiddleware(working_dir=tmp_path)
    assert [tool.name for tool in middleware.tools] == ["project_memory"]


@pytest.mark.asyncio
async def test_middleware_status_selects_missing_index_hint(tmp_path: Path) -> None:
    client = SimpleNamespace(
        index_status=AsyncMock(return_value=SimpleNamespace(exists=False, complete=False)),
        list_tenants=AsyncMock(
            return_value=SimpleNamespace(tenants=["/path/to/repo_a"], total=1)
        ),
    )
    middleware = ProjectMemoryMiddleware(working_dir=tmp_path, client=client)
    update = await middleware.abefore_agent({}, None)
    assert update == {"working_dir": str(tmp_path)}
    assert "only with an explicit" in middleware._cognitive_hint
    assert "/path/to/repo_a" in middleware._cognitive_hint


@pytest.mark.asyncio
async def test_middleware_service_failure_never_escapes(tmp_path: Path) -> None:
    client = SimpleNamespace(
        index_status=AsyncMock(side_effect=RuntimeError("service down"))
    )
    middleware = ProjectMemoryMiddleware(working_dir=tmp_path, client=client)
    update = await middleware.abefore_agent({}, None)
    assert update == {"working_dir": str(tmp_path)}
    assert middleware._has_index is False
    assert "service unavailable" in middleware._cognitive_hint
