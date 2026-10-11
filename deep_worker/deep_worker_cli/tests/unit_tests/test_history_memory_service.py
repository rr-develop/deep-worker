from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deep_worker_cli.history_memory_tool import history_memory
from deep_worker_cli.middleware.history_memory import HistoryMemoryMiddleware


@pytest.mark.asyncio
async def test_history_tool_validates_before_rpc() -> None:
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient"
    ) as client_cls:
        result = await history_memory.coroutine("q", top_k=21)
    assert result == "Error: top_k must be between 1 and 20"
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_history_tool_targets_another_agent() -> None:
    client = SimpleNamespace(
        index_status=AsyncMock(return_value=SimpleNamespace(exists=True, complete=True)),
        search_history=AsyncMock(return_value="history result"),
    )
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ), patch(
        "deep_worker_cli.semantic_memory.client.deep_worker_home"
    ) as home:
        home.return_value = __import__("pathlib").Path("/memory")
        result = await history_memory.coroutine("q", agent_name="alice")
    assert result == "history result"
    client.search_history.assert_awaited_once_with(
        "q", __import__("pathlib").Path("/memory/agents/alice/summaries"), 5
    )


def _missing_index_client(tenants: list[str] | None = None) -> SimpleNamespace:
    listed = ["bob"] if tenants is None else tenants
    return SimpleNamespace(
        index_status=AsyncMock(
            return_value=SimpleNamespace(exists=False, complete=False)
        ),
        list_tenants=AsyncMock(
            return_value=SimpleNamespace(tenants=listed, total=len(listed))
        ),
        search_history=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_missing_index_message_for_a_named_agent_is_not_about_us() -> None:
    """A named peer's history is not "this agent's own history".

    The twin of the project-side defect: the message described the
    default target while the agent had explicitly asked about someone
    else, and then advised passing ``agent_name`` — which it just did.
    """
    client = _missing_index_client()
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ), patch("deep_worker_cli.semantic_memory.client.deep_worker_home") as home:
        home.return_value = __import__("pathlib").Path("/memory")
        result = await history_memory.coroutine("q", agent_name="alice")

    assert "This agent's own history" not in result
    assert "agent_name=" not in result
    assert "That agent's history has not been indexed" in result
    assert "alice" in result
    assert "bob" in result
    client.search_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_agent_message_promises_no_list_when_there_is_none() -> None:
    """The history twin: no invitation to choose from an empty list."""
    client = _missing_index_client(tenants=[])
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ), patch("deep_worker_cli.semantic_memory.client.deep_worker_home") as home:
        home.return_value = __import__("pathlib").Path("/memory")
        result = await history_memory.coroutine("q", agent_name="alice")

    assert "below" not in result
    assert "No agent histories are indexed on this host right now." in result
    assert "That agent's history has not been indexed" in result
    assert "This agent's own history" not in result


@pytest.mark.asyncio
async def test_missing_index_message_for_our_own_history_keeps_its_wording() -> None:
    client = _missing_index_client()
    with patch(
        "deep_worker_cli.semantic_memory.client.SemanticMemoryClient",
        return_value=client,
    ), patch("deep_worker_cli.semantic_memory.client.deep_worker_home") as home:
        home.return_value = __import__("pathlib").Path("/memory")
        result = await history_memory.coroutine("q")

    assert "This agent's own history has not been indexed" in result
    assert 'agent_name="<name>"' in result
    assert "bob" in result


def test_enabled_middleware_registers_without_index() -> None:
    middleware = HistoryMemoryMiddleware("alice", enabled=True)
    assert [tool.name for tool in middleware.tools] == ["history_memory"]


@pytest.mark.asyncio
async def test_history_middleware_service_failure_never_escapes() -> None:
    client = SimpleNamespace(
        index_status=AsyncMock(side_effect=RuntimeError("service down"))
    )
    middleware = HistoryMemoryMiddleware("alice", enabled=True, client=client)
    update = await middleware.abefore_agent({}, None)
    assert update is not None
    assert middleware._has_index is False
    assert "service unavailable" in middleware._cognitive_hint
