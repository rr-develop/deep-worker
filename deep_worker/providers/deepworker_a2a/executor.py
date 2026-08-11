"""A2A ``AgentExecutor`` for deep-worker agents.

``DeepWorkerAgentExecutor`` bridges the A2A protocol (managed by the SDK's
request handler) to the agent's existing turn engine (``_handle_invoke``). It
does NOT re-implement agent logic: it runs the same turn used by dwcli/web and
maps the result onto the A2A task lifecycle via ``TaskUpdater``.

Key design points (see docs/DESIGN_A2A_PROTOCOL_SUPPORT.md §4):
- RC-2: turn events arrive ONLY via ``notify_callback=bridge.on_event``.
- RC-4: ``context_id`` is mapped to a dedicated LangGraph ``thread_id`` so each
  A2A conversation has its own history, isolated from the agent's main history.
- RC-5: HITL uses the A2A model — ``approval_callback`` is None; an
  ``input_required`` result publishes ``requires_input`` and returns (full HITL
  resume is stage 4).
- RC-1: ``cancel`` interrupts the running turn via
  ``AgentProcess.cancel_current_worker_task()`` (no RPyC/netref).

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers.proto_helpers import new_task
from a2a.types import Part, TaskState

from .bridge import TurnEventBridge

logger = logging.getLogger(__name__)


def _text_part(text: str) -> Part:
    return Part(text=text or "")


def _final_text(result: dict[str, Any]) -> str:
    """Extract the final assistant text from a ``_handle_invoke`` result."""
    resp = result.get("response")
    if isinstance(resp, dict):
        content = resp.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)
    return ""


class DeepWorkerAgentExecutor(AgentExecutor):
    """A2A executor that delegates to the agent's ``_handle_invoke`` turn engine."""

    def __init__(self, agent_process: Any, *, stream: bool = False):
        """Args:
        agent_process: The live ``AgentProcess`` instance.
        stream: Whether intermediate output is streamed as artifacts. For
            ``message/send`` this is False (single final Message); for
            ``message/stream`` the server constructs the executor with
            ``stream=True``. Stage 2 wires the streaming path end to end.
        """
        self.agent_process = agent_process
        self._stream = stream

    def _thread_id_for(self, context_id: str | None, task_id: str | None) -> str:
        cid = context_id or task_id or "default"
        return f"{self.agent_process.name}:a2a:{cid}"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # The SDK requires a Task to be enqueued before any TaskStatusUpdateEvent
        # when this is a fresh request (no pre-existing task in the store).
        if getattr(context, "current_task", None) is None:
            await event_queue.enqueue_event(
                new_task(
                    context.task_id,
                    context.context_id,
                    TaskState.TASK_STATE_SUBMITTED,
                )
            )
        await updater.start_work()

        bridge = TurnEventBridge(updater, stream=self._stream)

        try:
            result = await self.agent_process._handle_invoke(
                {"command": "invoke", "message": text, "stream": True},
                connection=None,
                notify_callback=bridge.on_event,  # RC-2: ONLY path for turn events
                approval_callback=None,            # RC-5: HITL via A2A model
                thread_id_override=self._thread_id_for(
                    context.context_id, context.task_id
                ),
            )
        except asyncio.CancelledError:
            # The turn was cancelled (e.g. via cancel()); cancel() itself emits
            # the CANCELED status. Re-raise so the framework unwinds cleanly.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("A2A turn failed")
            await updater.failed(
                updater.new_agent_message([_text_part(f"error: {exc}")])
            )
            return

        if not isinstance(result, dict):
            await updater.failed(
                updater.new_agent_message([_text_part("error: no result")])
            )
            return

        if result.get("interrupted"):
            await updater.cancel()
            return
        if result.get("input_required"):
            prompt = str(result.get("prompt", "Additional input required."))
            await updater.requires_input(
                updater.new_agent_message([_text_part(prompt)])
            )
            return
        if result.get("success"):
            await updater.complete(
                updater.new_agent_message([_text_part(_final_text(result))])
            )
            return

        await updater.failed(
            updater.new_agent_message([_text_part(str(result.get("error", "failed")))])
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # RC-1: interrupt the running turn in-process (no RPyC/netref).
        try:
            self.agent_process.cancel_current_worker_task()
        except Exception:  # noqa: BLE001 — cancellation is best-effort
            logger.exception("cancel_current_worker_task failed")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()
