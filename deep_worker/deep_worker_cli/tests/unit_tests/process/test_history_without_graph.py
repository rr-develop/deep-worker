"""History is read from the checkpoint and does NOT depend on a built graph.

Owner requirement (2026-08-29): "The history is there and it must not require
MCP or anything else."

Before: ``_handle_get_history`` and ``_handle_get_last_response`` started with
``if not self.agent`` and answered "Agent not initialized". For an agent that
failed to initialize (for example because of an unreachable external MCP) this
meant an HTTP 500 on Full Dialog and a silently empty history elsewhere --
while the conversation sat on disk untouched.

The checkpointer is created BEFORE the parts of initialization that can fail,
so it can always be read. The only real failure is when there is neither a
graph nor a checkpointer: then a visible error is required, NOT an empty
history, otherwise "could not read" is indistinguishable from "nothing to
read".
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager


class _StubCheckpointTuple:
    def __init__(self, messages: list[Any]) -> None:
        self.checkpoint = {"channel_values": {"messages": messages}}


class _StubCheckpointer:
    """Checkpointer double: returns the given messages via aget_tuple."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages
        self.calls = 0

    async def aget_tuple(self, config: Any) -> Any:
        self.calls += 1
        return _StubCheckpointTuple(self._messages)


def _broken_agent(messages: list[Any] | None) -> AgentProcess:
    """An agent whose graph was NOT built (``self.agent is None``)
    but whose checkpointer is alive -- exactly the state of an agent that
    failed to initialize."""
    from deep_worker_cli.process.message_dict_cache import MessageDictCache

    ap = AgentProcess.__new__(AgentProcess)
    ap.name = "broken_probe"
    ap.agent = None
    ap.session_manager = SessionManager(data_dir=Path(tempfile.mkdtemp()))
    ap.task_manager = None
    ap._message_dict_cache = MessageDictCache()
    ap._checkpointer = _StubCheckpointer(messages) if messages is not None else None
    return ap


@pytest.mark.asyncio
async def test_history_is_readable_without_a_graph() -> None:
    """Main contract: no graph -- the history is still served."""
    agent = _broken_agent(
        [HumanMessage(content="hello"), AIMessage(content="reply")]
    )

    result = await agent._handle_get_history({})

    assert result["success"] is True, result
    assert result["total"] == 2
    assert len(result["history"]) == 2


@pytest.mark.asyncio
async def test_history_does_not_claim_agent_not_initialized() -> None:
    """The old refusal text must not come back while the checkpointer is alive."""
    agent = _broken_agent([HumanMessage(content="hello")])

    result = await agent._handle_get_history({})

    assert result.get("error") != "Agent not initialized"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_last_response_is_readable_without_a_graph() -> None:
    """Same guarantee for last-response, otherwise the UI shows "No response yet"."""
    agent = _broken_agent(
        [HumanMessage(content="question"), AIMessage(content="this is the reply")]
    )

    result = await agent._handle_get_last_response()

    assert result["success"] is True, result
    assert result["last_response"]["content"] == "this is the reply"


@pytest.mark.asyncio
async def test_unreadable_state_is_an_error_not_an_empty_history() -> None:
    """Neither graph nor checkpointer -> a visible error, not an empty list.

    An empty history here would be a silent lie: the operator could not tell
    "there is no conversation" from "we could not read it".
    """
    agent = _broken_agent(None)

    result = await agent._handle_get_history({})

    assert result["success"] is False
    assert "history" not in result
    assert "checkpointer" in result["error"].lower()


@pytest.mark.asyncio
async def test_graph_is_preferred_when_present() -> None:
    """If a graph is present, read through it; do not touch the checkpointer directly."""

    class _State:
        values = {"messages": [AIMessage(content="from the graph")]}

    class _Graph:
        async def aget_state(self, config: Any) -> Any:
            return _State()

    agent = _broken_agent([HumanMessage(content="from the checkpointer")])
    agent.agent = _Graph()

    result = await agent._handle_get_history({})

    assert result["success"] is True
    assert result["history"][0]["content"] == "from the graph"
    assert agent._checkpointer.calls == 0, "graph is present -- direct read is not needed"


@pytest.mark.asyncio
async def test_pagination_still_applies_without_a_graph() -> None:
    """The response format is unchanged: limit/offset work on the same path."""
    agent = _broken_agent([AIMessage(content=f"m{i}") for i in range(5)])

    result = await agent._handle_get_history({"limit": 2})

    assert result["success"] is True
    assert result["total"] == 5
    assert [m["content"] for m in result["history"]] == ["m3", "m4"]
