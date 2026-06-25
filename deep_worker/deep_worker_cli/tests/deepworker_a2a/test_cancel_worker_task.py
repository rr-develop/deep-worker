"""Unit tests for ``AgentProcess.cancel_current_worker_task`` (RC-1).

A thin, in-process (no RPyC/netref) wrapper around the existing interrupt
mechanism (``is_running`` + ``current_worker_task.cancel()``) so the A2A
executor's ``cancel()`` can interrupt the running turn directly.
"""

import asyncio

import pytest

from deep_worker_cli.process.agent_process import AgentProcess


def _bare_agent() -> AgentProcess:
    # Focused tests construct AgentProcess via __new__ to avoid full init.
    ap = AgentProcess.__new__(AgentProcess)
    ap.is_running = False
    ap.current_worker_task = None
    return ap


def test_returns_false_when_not_running():
    ap = _bare_agent()
    assert ap.cancel_current_worker_task() is False


def test_returns_false_when_no_task():
    ap = _bare_agent()
    ap.is_running = True
    ap.current_worker_task = None
    assert ap.cancel_current_worker_task() is False


@pytest.mark.asyncio
async def test_cancels_running_task_and_returns_true():
    ap = _bare_agent()

    started = asyncio.Event()

    async def _long():
        started.set()
        await asyncio.sleep(60)

    task = asyncio.ensure_future(_long())
    await started.wait()
    ap.is_running = True
    ap.current_worker_task = task

    result = ap.cancel_current_worker_task()
    assert result is True

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.asyncio
async def test_idempotent_on_already_finished_task():
    ap = _bare_agent()

    async def _noop():
        return 1

    task = asyncio.ensure_future(_noop())
    await task  # finished
    ap.is_running = True
    ap.current_worker_task = task
    # Cancelling an already-done task must not raise and reports False
    # (nothing to interrupt).
    assert ap.cancel_current_worker_task() is False
