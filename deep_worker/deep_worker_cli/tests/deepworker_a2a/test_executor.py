"""Unit tests for ``deepworker_a2a.executor.DeepWorkerAgentExecutor`` (message/send).

Stage 1 covers the non-streaming request/response path: the executor reuses the
agent's ``_handle_invoke`` turn engine and drives the A2A task lifecycle
(submit -> working -> completed/failed) via ``TaskUpdater``. cancel() interrupts
the running turn through ``cancel_current_worker_task`` (RC-1).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deep_worker.providers.deepworker_a2a.executor import DeepWorkerAgentExecutor


class _FakeAgentProcess:
    def __init__(self, result: dict[str, Any]):
        self.name = "agentA"
        self._result = result
        self.captured: dict[str, Any] = {}
        self.cancelled = False

    async def _handle_invoke(self, command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        self.captured["command"] = command
        self.captured["notify_callback"] = notify_callback
        self.captured["approval_callback"] = approval_callback
        self.captured["thread_id_override"] = thread_id_override
        # simulate the turn calling back a streamed chunk
        if notify_callback is not None:
            await notify_callback({"type": "assistant", "role": "assistant",
                                   "content": "partial"})
        return self._result

    def cancel_current_worker_task(self) -> bool:
        self.cancelled = True
        return True


class _FakeEventQueue:
    """Captures events enqueued by TaskUpdater."""

    def __init__(self):
        self.events: list[Any] = []

    async def enqueue_event(self, event):
        self.events.append(event)


class _FakeContext:
    def __init__(self, text="do X", task_id="task-1", context_id="ctx-1"):
        self._text = text
        self.task_id = task_id
        self.context_id = context_id
        self.message = None

    def get_user_input(self, delimiter="\n"):
        return self._text


def _states(events) -> list:
    out = []
    for e in events:
        status = getattr(e, "status", None)
        if status is not None:
            out.append(status.state)
    return out


@pytest.mark.asyncio
async def test_execute_success_drives_submit_working_completed():
    from a2a.types import TaskState

    ap = _FakeAgentProcess(
        {"success": True, "response": {"role": "assistant", "content": "done"}}
    )
    ex = DeepWorkerAgentExecutor(ap)
    eq = _FakeEventQueue()
    ctx = _FakeContext()

    await ex.execute(ctx, eq)

    states = _states(eq.events)
    assert TaskState.TASK_STATE_SUBMITTED in states
    assert TaskState.TASK_STATE_WORKING in states
    assert TaskState.TASK_STATE_COMPLETED in states
    # terminal completed must be last
    assert states[-1] == TaskState.TASK_STATE_COMPLETED


@pytest.mark.asyncio
async def test_execute_passes_thread_id_override_from_context():
    ap = _FakeAgentProcess(
        {"success": True, "response": {"role": "assistant", "content": "done"}}
    )
    ex = DeepWorkerAgentExecutor(ap)
    await ex.execute(_FakeContext(context_id="ctx-XYZ", task_id="t1"),
                     _FakeEventQueue())
    assert ap.captured["thread_id_override"] == "agentA:a2a:ctx-XYZ"
    # approval_callback must be None (HITL handled via A2A model, RC-5)
    assert ap.captured["approval_callback"] is None


@pytest.mark.asyncio
async def test_execute_failure_drives_failed():
    from a2a.types import TaskState

    ap = _FakeAgentProcess({"success": False, "error": "boom"})
    ex = DeepWorkerAgentExecutor(ap)
    eq = _FakeEventQueue()
    await ex.execute(_FakeContext(), eq)
    assert _states(eq.events)[-1] == TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
async def test_execute_interrupted_drives_canceled():
    from a2a.types import TaskState

    ap = _FakeAgentProcess({"success": False, "interrupted": True})
    ex = DeepWorkerAgentExecutor(ap)
    eq = _FakeEventQueue()
    await ex.execute(_FakeContext(), eq)
    assert _states(eq.events)[-1] == TaskState.TASK_STATE_CANCELED


@pytest.mark.asyncio
async def test_cancel_interrupts_turn_and_emits_canceled():
    from a2a.types import TaskState

    ap = _FakeAgentProcess({"success": True, "response": {"content": "x"}})
    ex = DeepWorkerAgentExecutor(ap)
    eq = _FakeEventQueue()
    await ex.cancel(_FakeContext(), eq)
    assert ap.cancelled is True
    assert _states(eq.events)[-1] == TaskState.TASK_STATE_CANCELED


@pytest.mark.asyncio
async def test_execute_input_required_drives_input_required(monkeypatch):
    """HITL (RC-5): a turn needing more input publishes INPUT_REQUIRED, not a
    terminal state, so the A2A client can continue the same task/context."""
    from a2a.types import TaskState

    ap = _FakeAgentProcess(
        {"success": False, "input_required": True,
         "prompt": "Which file should I edit?"}
    )
    ex = DeepWorkerAgentExecutor(ap)
    eq = _FakeEventQueue()
    await ex.execute(_FakeContext(), eq)
    assert _states(eq.events)[-1] == TaskState.TASK_STATE_INPUT_REQUIRED


@pytest.mark.asyncio
async def test_multi_turn_same_context_id_maps_to_same_thread_id():
    """Multi-turn (RC-4): two A2A requests sharing a context_id must resolve to
    the SAME LangGraph thread_id so history carries across turns; a different
    context_id (or a bare task) must NOT collide with it."""
    ap = _FakeAgentProcess(
        {"success": True, "response": {"role": "assistant", "content": "ok"}}
    )
    ex = DeepWorkerAgentExecutor(ap)

    await ex.execute(_FakeContext(context_id="conv-7", task_id="t-a"),
                     _FakeEventQueue())
    first = ap.captured["thread_id_override"]
    await ex.execute(_FakeContext(context_id="conv-7", task_id="t-b"),
                     _FakeEventQueue())
    second = ap.captured["thread_id_override"]
    assert first == second == "agentA:a2a:conv-7"

    # Different conversation -> different thread.
    await ex.execute(_FakeContext(context_id="conv-OTHER", task_id="t-c"),
                     _FakeEventQueue())
    assert ap.captured["thread_id_override"] != first

    # No context_id -> falls back to task_id, still isolated from conv-7.
    await ex.execute(_FakeContext(context_id="", task_id="lonely-task"),
                     _FakeEventQueue())
    assert ap.captured["thread_id_override"] == "agentA:a2a:lonely-task"
