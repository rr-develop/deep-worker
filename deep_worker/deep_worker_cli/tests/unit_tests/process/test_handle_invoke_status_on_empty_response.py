"""Regression: ``_handle_invoke`` must not leave the agent stuck in
RUNNING when the turn produces no usable response.

Incident observed 2026-05-20: a codex turn failed (the provider's
event pump died with an API 400), LangGraph returned an empty
messages array, and ``_handle_invoke`` hit its
``"Agent did not produce a response"`` early-return — which returned
``success=False`` but NEVER reset the status from RUNNING. The agent
then advertised ``status="running"`` forever in ``dwcli --list`` and
the web UI, even though nothing was actually running. Three sibling
early-returns share the bug:

  * ``"No messages returned from agent"`` (TaskManager branch)
  * ``"Agent exceeded maximum iterations ..."``
  * ``"Agent did not produce a response"``

Contract under test: every terminal path of ``_handle_invoke`` must
leave the agent in a SETTLED status — ERROR for a genuine failure
(so the UI surfaces it and the operator knows the turn died), never a
dangling RUNNING. The success path (already covered elsewhere) lands
in IDLE.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


def _make_agent_with_real_session_manager() -> AgentProcess:
    """Build a bare ``AgentProcess`` (no ``__init__``) wired to a REAL
    ``SessionManager`` on a temp dir, so status transitions are
    observable through ``agent.session_manager.status``.

    Everything not needed for the ``_handle_invoke`` non-TaskManager
    standard path is stubbed.
    """
    agent = AgentProcess.__new__(AgentProcess)

    tmp = tempfile.mkdtemp(prefix="dwtest_handle_invoke_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))

    # Minimal state ``_handle_invoke`` touches.
    agent.name = "test_handle_invoke_status"
    agent.task_manager = None
    agent._message_subscribers = set()
    agent._last_broadcasted_count = {}
    agent.is_running = False
    agent.current_worker_task = None
    agent.current_thread_id = "default"

    return agent


@pytest.mark.asyncio
async def test_empty_response_does_not_leave_status_running() -> None:
    """The 'Agent did not produce a response' path must settle status.

    We stub ``agent.agent.ainvoke`` to return a result whose last
    message is an empty AIMessage with no tool calls — exactly what
    LangGraph hands back when a codex turn dies in its event pump.
    """
    agent = _make_agent_with_real_session_manager()

    # Empty assistant message — no content, no tool_calls. Mirrors the
    # observed failure shape.
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
    # The crucial assertion: status MUST NOT be left dangling at RUNNING.
    assert agent.session_manager.status != AgentStatus.RUNNING, (
        "after a no-response turn the agent is still marked RUNNING — "
        "this is the 'forever working' bug"
    )
    # A genuine failure should land in ERROR so the UI surfaces it.
    assert agent.session_manager.status == AgentStatus.ERROR


@pytest.mark.asyncio
async def test_error_log_includes_utc_timestamp_and_last_message_id(capsys: pytest.CaptureFixture[str]) -> None:
    agent = _make_agent_with_real_session_manager()

    empty_ai = MagicMock()
    empty_ai.id = "ai-empty-1"
    empty_ai.type = "ai"
    empty_ai.content = ""
    empty_ai.tool_calls = []

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [empty_ai]})
    agent.agent = fake_graph

    await agent._handle_invoke(
        {"command": "invoke", "message": "hello", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    stderr = capsys.readouterr().err
    assert re.search(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00\] \[INVOKE_ERROR\]", stderr)
    assert "input_message_id=user-" in stderr
    assert "last_message_id=ai-empty-1" in stderr
    assert "last_message_type=ai" in stderr


@pytest.mark.asyncio
async def test_exception_log_includes_utc_timestamp_and_input_message_id(capsys: pytest.CaptureFixture[str]) -> None:
    agent = _make_agent_with_real_session_manager()

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=RuntimeError("codex 400"))
    agent.agent = fake_graph

    await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    stderr = capsys.readouterr().err
    assert re.search(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00\] \[INVOKE_ERROR\]", stderr)
    assert "input_message_id=user-" in stderr
    assert "error=Agent invocation failed: codex 400" in stderr


@pytest.mark.asyncio
async def test_no_messages_returned_does_not_leave_status_running() -> None:
    """The TaskManager-branch 'No messages returned from agent' path.

    We don't need a TaskManager to reach an equivalent terminal path:
    an ainvoke returning ``{"messages": []}`` drives the standard
    branch's ``if not messages: break`` → no final content → settle.
    """
    agent = _make_agent_with_real_session_manager()

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": []})
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status != AgentStatus.RUNNING
    assert agent.session_manager.status == AgentStatus.ERROR


@pytest.mark.asyncio
async def test_exception_in_ainvoke_sets_error_status() -> None:
    """A raised exception inside ainvoke must settle to ERROR (this path
    already worked via the broad ``except`` — we lock it in)."""
    agent = _make_agent_with_real_session_manager()

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=RuntimeError("codex 400"))
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.ERROR
    assert "codex 400" in result["error"]


@pytest.mark.asyncio
async def test_successful_response_lands_idle() -> None:
    """Sanity: a normal turn still ends IDLE (no regression)."""
    agent = _make_agent_with_real_session_manager()

    ai = MagicMock()
    ai.type = "ai"
    ai.content = "the answer is 42"
    ai.tool_calls = []

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [ai]})
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "q", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is True
    assert agent.session_manager.status == AgentStatus.IDLE
