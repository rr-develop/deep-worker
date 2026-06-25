"""Unit tests for the ``thread_id_override`` parameter of ``_handle_invoke`` (RC-4).

The A2A executor isolates each A2A ``context_id`` into its own LangGraph
checkpointer history by passing ``thread_id_override``. Default ``None`` MUST
preserve the existing behaviour (``thread_id == self.name``) so no current
caller changes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager


def _bare_agent() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_a2a_threadid_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "agentT"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    return agent


def _fake_graph_capturing_config() -> tuple[MagicMock, dict]:
    captured: dict = {}
    ai = MagicMock()
    ai.type = "ai"
    ai.content = "ok"
    ai.tool_calls = []

    async def _ainvoke(payload, config):
        captured["config"] = config
        return {"messages": [ai]}

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=_ainvoke)
    return graph, captured


@pytest.mark.asyncio
async def test_default_thread_id_is_agent_name():
    agent = _bare_agent()
    graph, captured = _fake_graph_capturing_config()
    agent.agent = graph

    await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
    )

    assert captured["config"]["configurable"]["thread_id"] == "agentT"


@pytest.mark.asyncio
async def test_override_sets_thread_id():
    agent = _bare_agent()
    graph, captured = _fake_graph_capturing_config()
    agent.agent = graph

    await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        thread_id_override="agentT:a2a:ctx-123",
    )

    assert captured["config"]["configurable"]["thread_id"] == "agentT:a2a:ctx-123"


@pytest.mark.asyncio
async def test_empty_override_falls_back_to_agent_name():
    agent = _bare_agent()
    graph, captured = _fake_graph_capturing_config()
    agent.agent = graph

    await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        thread_id_override="",  # falsy -> default
    )

    assert captured["config"]["configurable"]["thread_id"] == "agentT"
