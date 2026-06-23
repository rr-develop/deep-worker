"""Reproduce issue #51: Agent did not produce a response with last_message_type=ai.

Tests two scenarios that can trigger this and verifies the fix.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


def _make_agent_with_real_session_manager() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_issue51_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "test_issue51"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"
    return agent


@pytest.mark.asyncio
async def test_tool_calls_without_content_gets_accurate_error():
    """Scenario 1: AI msg with tool_calls, no content, no interrupt.
    Should report "produced N tool call(s) but no text response"
    instead of the generic "Agent did not produce a response"."""
    agent = _make_agent_with_real_session_manager()

    ai_with_tools = MagicMock()
    ai_with_tools.type = "ai"
    ai_with_tools.content = ""
    ai_with_tools.tool_calls = [MagicMock()]  # non-empty

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [ai_with_tools]})
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hello", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert "tool call" in result["error"].lower(), (
        f"Error should mention unhandled tool calls, got: {result['error']}"
    )
    assert agent.session_manager.status == AgentStatus.ERROR


@pytest.mark.asyncio
async def test_content_with_tool_calls_is_preserved():
    """Scenario 2: AI msg with both content AND tool_calls.
    Content must NOT be lost (fix #51)."""
    agent = _make_agent_with_real_session_manager()

    ai_with_both = MagicMock()
    ai_with_both.type = "ai"
    ai_with_both.content = "Hello, I will call a tool"  # has text
    ai_with_both.tool_calls = [MagicMock()]  # also has tool calls

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [ai_with_both]})
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hello", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    # Content IS preserved — agent should succeed
    assert result["success"] is True, (
        f"Content was lost despite being present in AI message: {result}"
    )
    response_content = result.get("response", {}).get("content", "")
    assert "Hello, I will call a tool" in response_content, (
        f"Expected content in response, got: {result}"
    )


@pytest.mark.asyncio
async def test_truly_empty_response_still_produces_error():
    """Regression: a truly empty AI message (no content, no tool_calls)
    still produces the original error."""
    agent = _make_agent_with_real_session_manager()

    empty_ai = MagicMock()
    empty_ai.type = "ai"
    empty_ai.content = ""
    empty_ai.tool_calls = []

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [empty_ai]})
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hello", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert "did not produce" in result["error"].lower()
    assert agent.session_manager.status == AgentStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
