"""exposed_invoke_agent must not drop the plain request when a caller sends
it in ``message`` alongside ``sum_query``.

Bug 2026-07-04: the web client
(the web frontend's AgentService.send_message) puts the user's request in
``message`` and, at the time, could not send ``query`` at all. So a web call
``send_message(agent, message="task id=XYZ", sum_query="prefix")`` reached
``exposed_invoke_agent`` as ``(message="task id=XYZ", sum_query="prefix",
query=None)``. The sum_query/query branch called
``_run_startup_or_send_query(sum_query, query)`` and NEVER read ``message`` —
so ``message`` was silently dropped and only the summary-shape reached the
model. Downstream: 0/11 drafts, because the actionable text with the
conversation id never arrived.

The CLI send-through (agent_command._send_startup_query_to_running) sends
``message=""`` and puts the request in ``query``; it must stay unaffected.

Contract under test: in the sum_query/query branch, when ``query`` is not
given explicitly but ``message`` is non-empty, ``message`` becomes the plain
query (the second sequential turn). CLI's ``message=""`` stays a no-op.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


class _RecordingAgentProcess:
    """Records the (sum_query, query) actually handed to the shared helper."""

    def __init__(self) -> None:
        self.name = "sumqnorm"
        self.calls: list[tuple[str | None, str | None]] = []
        self._detached_invoke_tasks: set[Any] = set()

    async def _run_startup_or_send_query(
        self,
        sum_query: str | None,
        query: str | None,
        *,
        connection: Any = None,
        notify_callback: Any = None,
        approval_callback: Any = None,
    ) -> None:
        self.calls.append((sum_query, query))


def _make_service(agent: _RecordingAgentProcess) -> AgentService:
    svc = AgentService.__new__(AgentService)
    svc._agent_process = agent  # type: ignore[attr-defined]
    svc._event_loop = None  # type: ignore[attr-defined]

    async def _noop_notify(_msg: dict[str, Any]) -> None:
        return None

    svc._notify_subscribers = _noop_notify  # type: ignore[method-assign]
    return svc


@pytest.mark.asyncio
async def test_web_message_becomes_query_when_sum_query_present() -> None:
    """Web convention: request in ``message``, ``query`` absent. The message
    must be forwarded as the plain query so it runs as the second turn."""
    agent = _RecordingAgentProcess()
    svc = _make_service(agent)

    await svc.exposed_invoke_agent(
        "Draft a response to the recruiter id=XYZ",
        False,
        sum_query="STABLE_PREFIX",
        query=None,
    )

    assert agent.calls == [("STABLE_PREFIX", "Draft a response to the recruiter id=XYZ")]


@pytest.mark.asyncio
async def test_cli_empty_message_is_not_treated_as_query() -> None:
    """CLI convention: message="" and the request is in ``query``. The empty
    message must NOT override the explicit query."""
    agent = _RecordingAgentProcess()
    svc = _make_service(agent)

    await svc.exposed_invoke_agent(
        "",
        False,
        sum_query="SUMMARY",
        query="the real cli query",
    )

    assert agent.calls == [("SUMMARY", "the real cli query")]


@pytest.mark.asyncio
async def test_explicit_query_wins_over_message() -> None:
    """If both a non-empty message and an explicit query are given, the
    explicit query is authoritative (message is not smuggled in)."""
    agent = _RecordingAgentProcess()
    svc = _make_service(agent)

    await svc.exposed_invoke_agent(
        "message text",
        False,
        sum_query="SUMMARY",
        query="explicit query",
    )

    assert agent.calls == [("SUMMARY", "explicit query")]


@pytest.mark.asyncio
async def test_sum_query_only_with_empty_message_stays_sum_only() -> None:
    """sum_query alone with an empty message must remain sum-query-only
    (no phantom second turn)."""
    agent = _RecordingAgentProcess()
    svc = _make_service(agent)

    await svc.exposed_invoke_agent("", False, sum_query="SUMMARY", query=None)

    assert agent.calls == [("SUMMARY", None)]
