"""Regression: an inbound-channel quarantine must surface as a WARNING that
reaches live subscribers WITHOUT changing the agent's status.

Incident (observed 2026-05-22): a flooding client tripped rpyc_async's
inbound quarantine, which cancelled the agent's invoke and the agent
"spontaneously" went INTERRUPTED. After decoupling the invoke from the
channel, the agent keeps working — and the quarantine is now reported as a
non-blocking warning. The user's requirement: log it AND notify
the web app, but do NOT alter agent state.

Contract under test:
  * ``_broadcast_channel_quarantine_warning`` fans a ``channel_quarantined``
    warning event out to every live subscriber;
  * it does NOT call session_manager / change status;
  * the ``_on_inbound_quarantine`` hook is what gets wired into the RPyC
    protocol_config (so quarantine actually triggers the broadcast).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


class _FakeSubscriber:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def on_message(self, payload: str) -> None:
        self.messages.append(payload)


def _make_agent() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    tmp = tempfile.mkdtemp(prefix="dwtest_quar_warn_")
    agent.session_manager = SessionManager(data_dir=Path(tmp))
    agent.name = "quartest"
    agent._message_subscribers = set()
    return agent


_QUAR_INFO = {
    "connid": "conn7",
    "peer": "('127.0.0.1', 50816)",
    "inbound_inflight": 10000,
    "threshold": 10000,
    "request_callbacks": 9999,
}


@pytest.mark.asyncio
async def test_warning_reaches_subscribers() -> None:
    agent = _make_agent()
    sub = _FakeSubscriber()
    agent._message_subscribers.add(sub)

    await agent._broadcast_channel_quarantine_warning(_QUAR_INFO)
    # fire_and_forget_async dispatches on the loop — let it run.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if sub.messages:
            break

    assert sub.messages, "the quarantine warning must reach live subscribers"
    payload = json.loads(sub.messages[-1])
    assert payload["type"] == "channel_quarantined"
    assert payload["level"] == "warning"
    assert payload["connid"] == "conn7"


@pytest.mark.asyncio
async def test_warning_does_not_change_status() -> None:
    agent = _make_agent()
    # Agent is actively working.
    await agent.session_manager.set_status(AgentStatus.RUNNING)
    sub = _FakeSubscriber()
    agent._message_subscribers.add(sub)

    await agent._broadcast_channel_quarantine_warning(_QUAR_INFO)
    for _ in range(20):
        await asyncio.sleep(0.01)
        if sub.messages:
            break

    # CRITICAL: status MUST be unchanged — quarantine is a warning, the agent
    # keeps working (observed 2026-05-22).
    assert agent.session_manager.status == AgentStatus.RUNNING, (
        "channel quarantine must NOT change agent status"
    )


@pytest.mark.asyncio
async def test_broadcast_with_no_subscribers_is_noop() -> None:
    agent = _make_agent()
    # No subscribers — must not raise.
    await agent._broadcast_channel_quarantine_warning(_QUAR_INFO)


def test_hook_schedules_broadcast_and_keeps_status() -> None:
    """The sync hook (_on_inbound_quarantine) must NOT touch status and must
    schedule the async broadcast. Source-level + behavioral check."""
    src = inspect.getsource(AgentProcess._on_inbound_quarantine)
    # The hook must not set status/error directly.
    assert "set_status" not in src
    assert "set_error" not in src
    # It must schedule the broadcast.
    assert "_broadcast_channel_quarantine_warning" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
