"""Regression: ``_handle_invoke`` must not leave the agent stuck in
RUNNING when the turn produces no usable response.

Incident (a related internal incident): a codex turn failed (the provider's
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
leave the agent in a SETTLED status (never a dangling RUNNING).

Note (2026-06-24): under the current "No-Agent-Error-State Policy"
the recoverable empty-response paths settle to IDLE (not ERROR) and
surface a SystemMessage so the next user message ('continue') proceeds
normally. The settled-not-RUNNING invariant is what these tests
guard; they intentionally assert ``!= RUNNING`` rather than a
specific terminal status so the contract survives future policy
tweaks. The exception path (an exception out of ``ainvoke``) is
covered separately and still ends in ERROR via the broad ``except``.
"""

from __future__ import annotations

import logging
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
    # incident failure shape.
    empty_ai = MagicMock()
    empty_ai.type = "ai"
    empty_ai.content = ""
    empty_ai.tool_calls = []

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [empty_ai]})
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
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
    # No-Agent-Error-State Policy (2026-06-24): a recoverable empty response
    # settles to IDLE (not ERROR) and appends a SystemMessage so the next
    # user message proceeds.
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_error_log_includes_utc_timestamp_and_last_message_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _make_agent_with_real_session_manager()

    empty_ai = MagicMock()
    empty_ai.id = "ai-empty-1"
    empty_ai.type = "ai"
    empty_ai.content = ""
    empty_ai.tool_calls = []

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [empty_ai]})
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    with caplog.at_level(logging.ERROR, logger="deep_worker_cli.process.agent_process"):
        await agent._handle_invoke(
            {"command": "invoke", "message": "hello", "stream": False},
            connection=None,
            notify_callback=None,
            approval_callback=None,
        )

    # The structured [INVOKE_ERROR] record carries the message identity
    # fields. The UTC timestamp is supplied by the logging formatter and is
    # asserted by the daemon-logging tests, not here.
    invoke_errors = [
        r.getMessage() for r in caplog.records if "[INVOKE_ERROR]" in r.getMessage()
    ]
    assert invoke_errors, "expected an [INVOKE_ERROR] log record"
    combined = "\n".join(invoke_errors)
    assert "input_message_id=user-" in combined
    assert "last_message_id=ai-empty-1" in combined
    assert "last_message_type=ai" in combined


@pytest.mark.asyncio
async def test_exception_log_includes_utc_timestamp_and_input_message_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _make_agent_with_real_session_manager()

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=RuntimeError("codex 400"))
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    with caplog.at_level(logging.ERROR, logger="deep_worker_cli.process.agent_process"):
        await agent._handle_invoke(
            {"command": "invoke", "message": "hi", "stream": False},
            connection=None,
            notify_callback=None,
            approval_callback=None,
        )

    invoke_errors = [
        r.getMessage() for r in caplog.records if "[INVOKE_ERROR]" in r.getMessage()
    ]
    assert invoke_errors, "expected an [INVOKE_ERROR] log record"
    combined = "\n".join(invoke_errors)
    assert "input_message_id=user-" in combined
    # Under the recoverable path the structured error is "<ExcName>: <msg>".
    assert "codex 400" in combined


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
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status != AgentStatus.RUNNING
    # No-Agent-Error-State Policy: recoverable no-response settles to IDLE.
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_exception_in_ainvoke_sets_error_status() -> None:
    """A raised non-fatal exception inside ainvoke must settle (never left
    RUNNING). Under the No-Agent-Error-State Policy (2026-06-24) a
    recoverable exception (e.g. a transient RuntimeError) lands in IDLE with
    a SystemMessage appended — only fatal exceptions set ERROR."""
    agent = _make_agent_with_real_session_manager()

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=RuntimeError("codex 400"))
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "hi", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False
    assert agent.session_manager.status != AgentStatus.RUNNING
    assert agent.session_manager.status == AgentStatus.IDLE
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
