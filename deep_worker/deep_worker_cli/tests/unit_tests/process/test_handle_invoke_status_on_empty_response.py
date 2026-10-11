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
from langchain_core.messages import AIMessage, ToolMessage

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


# ---------------------------------------------------------------------------
# Turn-ending tools: successful ToolMessage tail is a legitimate turn end.
#
# Regression reported internally (2026-08-22, repo_graph extractor).
#
# ``ToolCompletionMiddleware`` (deep_worker_cli/middleware/tool_completion.py)
# ends the LangGraph turn immediately after a registered tool succeeds by
# returning ``{"jump_to": "end"}`` with NO subsequent ``AIMessage``. The tail
# of ``state["messages"]`` is therefore a SUCCESSFUL ``ToolMessage`` — that
# IS the completed turn.
#
# The old ``_handle_invoke`` only recognised a non-empty AI text as a final
# response, so a successful turn-ending tool_call landed in the "no final
# content" recovery path with ``success=False`` and the operator saw
# "Agent did not produce a response" — while the turn had, in fact,
# succeeded. Symptom: repo_graph's ``repo_graph__submit_fragments`` /
# ``repo_graph__submit_relations`` closing a DocumentSession left the
# client with ``success:false``.
#
# The fix reads ``ToolMessage.status`` (LangChain's own signal for
# whether the tool call itself succeeded or errored). It MUST be
# fail-closed: a message with a missing / unrecognised ``status`` is
# NOT a success. LangChain's default of ``status='success'`` is not
# safe to inherit here — a tool that forgets to set status on error
# would silently return success.
# ---------------------------------------------------------------------------


def _tool_call_ai(name: str, tool_call_id: str) -> AIMessage:
    """AIMessage whose only output is the turn-ending tool call — no text.

    This is the exact shape the model produces when it delegates the
    entire turn to the tool (the submit_* pattern).
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": tool_call_id}],
    )


@pytest.mark.asyncio
async def test_terminal_successful_toolmessage_ends_turn_as_success() -> None:
    """A trailing ToolMessage(status='success') from a turn-ending tool
    is a legitimate turn end, NOT ``"Agent did not produce a response"``.

    Repro: `ToolCompletionMiddleware` jumps to ``end``
    after the tool succeeds, so state["messages"] tail is
    ``[AIMessage(tool_calls=[...]), ToolMessage(status='success')]``
    with NO subsequent AIMessage.
    """
    agent = _make_agent_with_real_session_manager()

    ai = _tool_call_ai("repo_graph__submit_fragments", "call_1")
    # A registered turn-ending tool succeeded; ToolCompletionMiddleware
    # emitted no echo (echo_field=None), so this is the tail.
    tool = ToolMessage(
        content='{"success": true, "document_closed": true}',
        tool_call_id="call_1",
        name="repo_graph__submit_fragments",
        status="success",
    )

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [ai, tool]})
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "extract", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is True, (
        "a successful turn-ending ToolMessage tail must be recognised as a "
        "completed turn — this is the 2026-08-22 defect: without "
        "this, the client sees 'Agent did not produce a response' every time "
        "a turn-ending tool closes the turn"
    )
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_terminal_error_toolmessage_still_settles_as_no_response() -> None:
    """Fail-closed: ToolMessage(status='error') is NOT a successful turn end.

    The whole point of the fix is to distinguish "the tool succeeded and
    the middleware ended the turn" from "the tool errored". Middleware
    does NOT end the turn on error (that surfaces to the model), but if
    ``_handle_invoke`` ever sees an error-tail (e.g. non-registered tool
    erroring at the end of a turn), it must NOT declare success.
    """
    agent = _make_agent_with_real_session_manager()

    ai = _tool_call_ai("repo_graph__submit_fragments", "call_2")
    tool = ToolMessage(
        content='{"success": false, "error": "closed session already"}',
        tool_call_id="call_2",
        name="repo_graph__submit_fragments",
        status="error",
    )

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [ai, tool]})
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "extract", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False, (
        "an ERROR-status trailing ToolMessage must NOT be treated as a "
        "completed turn — that would hide real tool failures behind a "
        "silent success and reintroduce the fail-open behaviour rejected "
        "in the design of this fix"
    )
    # The settled-not-RUNNING invariant still holds.
    assert agent.session_manager.status != AgentStatus.RUNNING
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_terminal_toolmessage_with_error_status_is_not_success() -> None:
    """Fail-closed: ``status='error'`` must NOT slip through.

    Distinct from the earlier error-tail test: that one uses a real
    ``ToolMessage(status='error')`` object; this one uses the dict-form
    to pin the same fail-closed rule on the OTHER shape ``_handle_invoke``
    reads. Both shapes must reject ``'error'`` and both must require an
    EXPLICIT ``'success'`` (rejecting missing, ``None``, and anything
    other than the LangChain literal).
    """
    agent = _make_agent_with_real_session_manager()

    ai = _tool_call_ai("some_tool", "call_3")
    tool_dict_error = {
        "role": "tool",
        "content": '{"success": false}',
        "tool_call_id": "call_3",
        "name": "some_tool",
        "status": "error",
    }

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"messages": [ai, tool_dict_error]},
    )
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "extract", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False, (
        "a ToolMessage with an explicit error status must NOT be "
        "treated as a completed turn on any shape"
    )
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_terminal_toolmessage_with_missing_status_key_is_not_success() -> None:
    """Fail-closed: a dict-form tool tail WITHOUT a ``status`` key must
    NOT be treated as success — even though LangChain would default it
    to ``'success'`` on the model side.

    This is what rejects the prototype form
    ``last_msg.get("status", "success")``: with a missing key, that call
    returns ``'success'`` and lets an unmarked tool silently claim the
    turn. The consumer-side rule is explicit: only the LangChain literal
    ``'success'`` counts, and that literal must be present.
    """
    agent = _make_agent_with_real_session_manager()

    ai = _tool_call_ai("some_tool", "call_4")
    # Deliberately omit the ``status`` key. A dict tail with no status
    # signal MUST NOT end the turn on a fail-open default.
    tool_dict_no_status = {
        "role": "tool",
        "content": '{"success": true}',
        "tool_call_id": "call_4",
        "name": "some_tool",
    }

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"messages": [ai, tool_dict_no_status]},
    )
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "extract", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is False, (
        "a tool tail with NO status key must NOT be treated as success "
        "— the fail-open default (``.get('status', 'success')``) would "
        "silently accept it; the fix must be fail-closed"
    )
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_terminal_toolmessage_success_not_last_still_recovers() -> None:
    """A successful ToolMessage that is NOT the last message does not
    end the turn: the real terminal is whatever comes after.

    This nails the 'terminal' part of the fix down: only the tail
    counts. Parallel tool batches (multiple trailing ToolMessages) and
    intermediate successful tool calls must not be misread as the end
    of the turn.
    """
    agent = _make_agent_with_real_session_manager()

    ai1 = _tool_call_ai("some_tool", "call_a")
    tool_ok = ToolMessage(
        content='{"success": true}',
        tool_call_id="call_a", name="some_tool", status="success",
    )
    # An empty AI trails after — the intermediate success should NOT
    # short-circuit into a "turn ended" declaration.
    trailing_empty_ai = AIMessage(content="", tool_calls=[])

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"messages": [ai1, tool_ok, trailing_empty_ai]},
    )
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "x", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    # Trailing empty AI ⇒ the "no final content" path still fires.
    assert result["success"] is False
    assert agent.session_manager.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_terminal_toolmessage_success_dict_form_is_recognised() -> None:
    """Dict-form messages (some serialisation paths) must also be
    recognised: a trailing ``{"role": "tool", "status": "success"}``
    ends the turn as success.

    The production tool node emits real ``ToolMessage`` objects, but
    ``_handle_invoke`` also handles dict-form messages elsewhere in
    the loop; the check must be consistent.
    """
    agent = _make_agent_with_real_session_manager()

    ai = _tool_call_ai("some_tool", "call_d")
    tool_dict = {
        "role": "tool",
        "content": '{"success": true}',
        "tool_call_id": "call_d",
        "name": "some_tool",
        "status": "success",
    }

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"messages": [ai, tool_dict]})
    fake_graph.aupdate_state = AsyncMock()
    fake_graph.aget_state = AsyncMock()
    agent.agent = fake_graph

    result = await agent._handle_invoke(
        {"command": "invoke", "message": "x", "stream": False},
        connection=None,
        notify_callback=None,
        approval_callback=None,
    )

    assert result["success"] is True
    assert agent.session_manager.status == AgentStatus.IDLE
