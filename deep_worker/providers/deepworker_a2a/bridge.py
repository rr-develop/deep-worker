"""Bridge from deep-worker turn events to A2A streaming events.

A :class:`TurnEventBridge` is passed to the agent core as
``notify_callback=bridge.on_event`` (RC-2: this is the ONLY path — the bridge is
NOT registered in ``AgentProcess._message_subscribers``, whose contract is the
RPyC ``on_message(str)`` netref call). ``on_event`` is ``async`` because the
core awaits ``notify_callback(...)``.

During a turn the bridge translates intermediate output (assistant text chunks,
tool calls / tool results) into A2A artifact updates via
``TaskUpdater.add_artifact(append=True)`` — the canonical A2A streaming
mechanism. Terminal status (completed / failed / canceled / input-required) is
NOT set here; the executor sets it from the turn's final result.

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§4 TurnEventBridge).
"""

from __future__ import annotations

import logging
from typing import Any

from a2a.types import Part

logger = logging.getLogger(__name__)

# Event ``type`` values emitted by the deep-worker core's notify_callback that
# carry streamable text we forward as artifact chunks.
_TEXT_EVENT_TYPES = frozenset({"assistant", "message", "ai"})


class TurnEventBridge:
    """Translate ``notify_callback`` dict events into A2A artifact chunks.

    Args:
        updater: The A2A ``TaskUpdater`` for the current task.
        stream: When False (message/send), intermediate chunks are NOT emitted
            as artifacts (the executor returns a single final Message instead).
            When True (message/stream), chunks are emitted live as artifact
            updates. Default True.
    """

    def __init__(self, updater: Any, *, stream: bool = True):
        self._updater = updater
        self._stream = stream
        self._artifact_id = f"{getattr(updater, 'task_id', 'task')}-output"
        self.saw_event = False
        # The SDK requires the FIRST chunk for an artifact to be append=False
        # (creates the artifact); subsequent chunks append=True.
        self._artifact_started = False

    async def on_event(self, msg: dict[str, Any]) -> None:
        """Receive one turn event (async — the core awaits this)."""
        self.saw_event = True
        if not self._stream:
            return
        try:
            text = self._render(msg)
            if not text:
                return
            # First chunk creates the artifact (append=False); rest append.
            append = self._artifact_started
            await self._updater.add_artifact(
                parts=[Part(text=text)],
                artifact_id=self._artifact_id,
                name="response",
                append=append,
                last_chunk=False,
            )
            self._artifact_started = True
        except Exception:  # noqa: BLE001 — never let a stream hiccup kill the turn
            logger.exception("TurnEventBridge failed to forward event")

    @staticmethod
    def _render(msg: dict[str, Any]) -> str:
        """Render a turn event dict into a text artifact chunk, or "".

        Tool events are folded into a ``[tool:<name>]`` prefixed text part so a
        consumer sees them inline; pure assistant text passes through.
        """
        etype = msg.get("type", "")
        if etype in _TEXT_EVENT_TYPES:
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
        if etype == "tool_call":
            name = msg.get("name", "")
            return f"[tool:{name}]"
        if etype == "tool_result":
            name = msg.get("name", "")
            content = msg.get("content", "")
            return f"[tool_result:{name}] {content}".rstrip()
        # user echoes, agent_interrupted, status, etc. are not artifact output
        return ""
