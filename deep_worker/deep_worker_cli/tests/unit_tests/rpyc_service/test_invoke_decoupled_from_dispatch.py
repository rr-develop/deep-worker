"""Regression: the agent's invoke must NOT be cancelled when the RPyC
dispatch task that delivered it is cancelled.

Incident (observed 2026-05-22): a DIFFERENT client (the web app) flooded one
RPyC connection until rpyc_async's inbound-quarantine fired. Quarantine
calls ``_drain_inbound_dispatch`` which cancels every parked dispatch task
on that connection. Because ``exposed_invoke_agent`` awaited
``_handle_invoke`` INLINE in the dispatch task, the running agent turn got
``CancelledError`` and was misclassified as a user interrupt → the agent
"spontaneously" went INTERRUPTED and abandoned the task.

The fix decouples the invoke: ``exposed_invoke_agent`` runs
``_handle_invoke`` in a dedicated task and awaits it via ``asyncio.shield``.
Cancelling the awaiter (the dispatch task) must leave the invoke running to
completion. A REAL user interrupt goes through ``exposed_interrupt_agent``
which cancels ``current_worker_task`` (the dedicated task) directly — that
path is unaffected.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


class _FakeAgentProcess:
    """Minimal AgentProcess stand-in. _handle_invoke runs long enough that
    we can cancel the awaiter mid-flight and observe whether the invoke
    survives."""

    def __init__(self, run_seconds: float = 0.4) -> None:
        self.name = "decoupletest"
        self._run_seconds = run_seconds
        self.invoke_started = False
        self.invoke_completed = False
        self.invoke_cancelled = False
        self.current_worker_task: Any = None
        self.is_running = False
        self._detached_invoke_tasks: set[Any] = set()

    async def _handle_invoke(
        self, request: dict[str, Any], connection: Any, notify_callback: Any, approval_callback: Any
    ) -> dict[str, Any]:
        # Mirror production: the invoke records ITS OWN task so a real
        # interrupt can cancel it.
        self.current_worker_task = asyncio.current_task()
        self.is_running = True
        self.invoke_started = True
        try:
            await asyncio.sleep(self._run_seconds)
            self.invoke_completed = True
            return {"success": True, "response": {"role": "assistant", "content": "done"}}
        except asyncio.CancelledError:
            self.invoke_cancelled = True
            raise
        finally:
            self.is_running = False


def _make_service(agent: _FakeAgentProcess) -> AgentService:
    svc = AgentService.__new__(AgentService)
    svc._agent_process = agent  # type: ignore[attr-defined]
    svc._event_loop = None  # type: ignore[attr-defined]

    async def _noop_notify(_msg: dict[str, Any]) -> None:
        return None

    svc._notify_subscribers = _noop_notify  # type: ignore[method-assign]
    return svc


@pytest.mark.asyncio
async def test_cancelling_dispatch_task_does_not_cancel_invoke() -> None:
    """Cancelling the task that awaits exposed_invoke_agent (i.e. the RPyC
    dispatch task) must NOT cancel the underlying invoke — the agent keeps
    working and completes the turn."""
    agent = _FakeAgentProcess(run_seconds=0.4)
    svc = _make_service(agent)

    # The "dispatch task" — exactly what rpyc_async runs and what
    # _drain_inbound_dispatch would cancel on quarantine.
    dispatch_task = asyncio.ensure_future(
        svc.exposed_invoke_agent("do the work", False)
    )

    # Let the invoke get going, then cancel the dispatch task (simulating
    # inbound-quarantine draining this connection's dispatch tasks).
    await asyncio.sleep(0.1)
    assert agent.invoke_started
    dispatch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_task

    # CRITICAL: the invoke must still finish. Give it time past its runtime.
    await asyncio.sleep(0.6)
    assert agent.invoke_completed, (
        "the agent turn was cancelled when its RPyC dispatch task was "
        "cancelled — invoke must be decoupled from the channel (observed "
        "2026-05-22)"
    )
    assert not agent.invoke_cancelled


@pytest.mark.asyncio
async def test_real_interrupt_still_cancels_invoke() -> None:
    """A real user interrupt cancels current_worker_task (the dedicated
    invoke task). That must still stop the turn — decoupling must not break
    the legitimate interrupt path."""
    agent = _FakeAgentProcess(run_seconds=2.0)
    svc = _make_service(agent)

    dispatch_task = asyncio.ensure_future(
        svc.exposed_invoke_agent("long work", False)
    )
    await asyncio.sleep(0.1)
    assert agent.invoke_started
    assert agent.current_worker_task is not None

    # Real interrupt: cancel the dedicated worker task (what
    # exposed_interrupt_agent does).
    agent.current_worker_task.cancel()

    result = json.loads(await dispatch_task)
    assert result.get("interrupted") is True
    assert agent.invoke_cancelled
    assert not agent.invoke_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
