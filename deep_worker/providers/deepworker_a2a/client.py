"""Outbound A2A client for deep-worker agents.

``A2AOutboundClient`` wraps the a2a-sdk client (``create_client`` /
``A2ACardResolver``) so an agent can initiate A2A interactions with other
A2A-compliant agents — by external URL, or by the *name* of another deep-worker
agent (its A2A URL is looked up in the process registry via ``a2a_url``).

All methods return plain JSON-friendly dicts and never raise to the caller —
mirroring the deep-worker tool convention. This module is the transport behind
the ``a2a_*`` LLM tools (see ``deepworker_a2a.tools``).

Design: ``deepworker_a2a/docs/DESIGN_A2A_PROTOCOL_SUPPORT.md`` (§3.2, §5).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


def _resolve_target_url(target: str, registry: Any | None = None) -> str | None:
    """Resolve ``target`` to an A2A base URL.

    ``target`` is either an http(s) URL (used as-is) or the name of a running
    deep-worker agent, whose ``a2a_url`` is read from the process registry.
    Returns None if a named agent has no A2A endpoint / is not found.
    """
    if target.startswith("http://") or target.startswith("https://"):
        return target
    if registry is None:
        from deep_worker_cli.process.registry import ProcessRegistry

        registry = ProcessRegistry()
    info = registry.lookup(target)
    if info is None:
        return None
    return getattr(info, "a2a_url", None)


class A2AOutboundClient:
    """Thin async wrapper over the a2a-sdk client for outbound calls."""

    def __init__(self, registry: Any | None = None):
        self._registry = registry

    # -- internal helpers ------------------------------------------------

    async def _make_client(self, httpx_client: Any, url: str, *, streaming: bool):
        from a2a.client import ClientConfig, create_client

        return await create_client(
            url, ClientConfig(httpx_client=httpx_client, streaming=streaming)
        )

    @staticmethod
    def _build_request(text: str, context_id: str | None, task_id: str | None):
        from uuid import uuid4

        from a2a.types import Message, Part, Role, SendMessageRequest

        msg = Message(
            message_id=uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
        )
        if context_id:
            msg.context_id = context_id
        if task_id:
            msg.task_id = task_id
        return SendMessageRequest(message=msg)

    @staticmethod
    def _event_to_text(event: Any) -> str:
        """Extract any text carried by a StreamResponse.

        Covers both the streaming case (artifact / message events, handled by
        ``get_stream_response_text``) and the non-streaming case, where the
        single response is a final ``task`` (or terminal ``status_update``)
        whose ``status.message`` holds the agent's answer.
        """
        from a2a.helpers.proto_helpers import (
            get_message_text,
            get_stream_response_text,
        )

        try:
            text = get_stream_response_text(event)
            if text:
                return text
        except Exception:  # noqa: BLE001
            pass

        # Non-streaming: the answer lives in the final task/status message.
        try:
            for holder in (getattr(event, "task", None),
                           getattr(event, "status_update", None)):
                status = getattr(holder, "status", None)
                message = getattr(status, "message", None)
                if message is not None and getattr(message, "parts", None):
                    msg_text = get_message_text(message)
                    if msg_text:
                        return msg_text
        except Exception:  # noqa: BLE001
            pass
        return ""

    @staticmethod
    def _event_ids_state(event: Any) -> dict[str, Any]:
        """Extract task_id / context_id / state from a StreamResponse, if present."""
        out: dict[str, Any] = {}
        task = getattr(event, "task", None)
        if task is not None and getattr(task, "id", ""):
            out["task_id"] = task.id
            out["context_id"] = task.context_id
            if task.HasField("status") if hasattr(task, "HasField") else getattr(task, "status", None):
                out["state"] = _state_name(task.status.state)
        su = getattr(event, "status_update", None)
        if su is not None and getattr(su, "task_id", ""):
            out["task_id"] = su.task_id
            out["context_id"] = su.context_id
            out["state"] = _state_name(su.status.state)
        return out

    # -- public API ------------------------------------------------------

    async def discover(self, target: str) -> dict[str, Any]:
        """Resolve and return a structured AgentCard summary dict."""
        import httpx

        from .card import agent_card_to_summary

        url = _resolve_target_url(target, self._registry)
        if not url:
            return {"success": False, "error": "target_not_found",
                    "message": f"No A2A endpoint for target {target!r}"}
        try:
            from a2a.client import A2ACardResolver

            async with httpx.AsyncClient(timeout=15) as hc:
                resolver = A2ACardResolver(hc, url)
                card = await resolver.get_agent_card()
            summary = agent_card_to_summary(card)
            summary["success"] = True
            return summary
        except Exception as exc:  # noqa: BLE001
            logger.exception("a2a discover failed")
            return {"success": False, "error": "discover_failed", "message": str(exc)}

    async def send(self, target: str, text: str, *, context_id: str | None = None,
                   task_id: str | None = None) -> dict[str, Any]:
        """Send a message and wait for the final result (non-streaming)."""
        import httpx

        url = _resolve_target_url(target, self._registry)
        if not url:
            return {"success": False, "error": "target_not_found",
                    "message": f"No A2A endpoint for target {target!r}"}
        try:
            async with httpx.AsyncClient(timeout=120) as hc:
                client = await self._make_client(hc, url, streaming=False)
                req = self._build_request(text, context_id, task_id)
                texts: list[str] = []
                ids: dict[str, Any] = {}
                async for event in client.send_message(req):
                    t = self._event_to_text(event)
                    if t:
                        texts.append(t)
                    ids.update(self._event_ids_state(event))
            return {"success": True, "answer": "\n".join(texts).strip(), **ids}
        except Exception as exc:  # noqa: BLE001
            logger.exception("a2a send failed")
            return {"success": False, "error": "send_failed", "message": str(exc)}

    async def send_streaming(self, target: str, text: str, *,
                             context_id: str | None = None,
                             task_id: str | None = None) -> dict[str, Any]:
        """Send a message in streaming mode and collect chunks + final answer.

        Returns the concatenated answer plus the list of streamed chunks (so a
        tool can surface progress). Same structured-dict contract as ``send``.
        """
        import httpx

        url = _resolve_target_url(target, self._registry)
        if not url:
            return {"success": False, "error": "target_not_found",
                    "message": f"No A2A endpoint for target {target!r}"}
        try:
            chunks: list[str] = []
            ids: dict[str, Any] = {}
            async with httpx.AsyncClient(timeout=120) as hc:
                client = await self._make_client(hc, url, streaming=True)
                req = self._build_request(text, context_id, task_id)
                async for event in client.send_message(req):
                    t = self._event_to_text(event)
                    if t:
                        chunks.append(t)
                    ids.update(self._event_ids_state(event))
            return {"success": True, "answer": "".join(chunks).strip(),
                    "chunks": chunks, **ids}
        except Exception as exc:  # noqa: BLE001
            logger.exception("a2a send_streaming failed")
            return {"success": False, "error": "stream_failed", "message": str(exc)}

    async def get_task(self, target: str, task_id: str) -> dict[str, Any]:
        import httpx

        from a2a.types import GetTaskRequest

        url = _resolve_target_url(target, self._registry)
        if not url:
            return {"success": False, "error": "target_not_found",
                    "message": f"No A2A endpoint for target {target!r}"}
        try:
            async with httpx.AsyncClient(timeout=30) as hc:
                client = await self._make_client(hc, url, streaming=False)
                task = await client.get_task(GetTaskRequest(id=task_id))
            return {"success": True, "task_id": task.id,
                    "context_id": task.context_id,
                    "state": _state_name(task.status.state)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("a2a get_task failed")
            return {"success": False, "error": "get_task_failed", "message": str(exc)}

    async def cancel_task(self, target: str, task_id: str) -> dict[str, Any]:
        import httpx

        from a2a.types import CancelTaskRequest

        url = _resolve_target_url(target, self._registry)
        if not url:
            return {"success": False, "error": "target_not_found",
                    "message": f"No A2A endpoint for target {target!r}"}
        try:
            async with httpx.AsyncClient(timeout=30) as hc:
                client = await self._make_client(hc, url, streaming=False)
                task = await client.cancel_task(CancelTaskRequest(id=task_id))
            return {"success": True, "task_id": task.id,
                    "state": _state_name(task.status.state)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("a2a cancel_task failed")
            return {"success": False, "error": "cancel_task_failed", "message": str(exc)}


def _state_name(state: Any) -> str:
    try:
        from a2a.types import TaskState

        return TaskState.Name(state)
    except Exception:  # noqa: BLE001
        return str(state)
