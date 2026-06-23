"""Tool-dispatch safety net for the No-Agent-Error-State Policy.

If a tool raises (an unhandled exception inside the tool body, or anywhere
inside the LangGraph tool-dispatch path that this middleware sits in front
of), the agent runtime catches it here, converts it into a structured
``ToolMessage(status="error")`` visible to the model, logs it at ``ERROR``,
and lets the conversation continue.

This is the policy boundary required by §3 of
``docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md`` / ``docs/POLICIES.md`` —
without it a sloppy tool that raises propagates up to the agent invoke
catch-all and used to flip the whole agent into ``AgentStatus.ERROR``.

NEVER use ``print()`` in this module — see §4.6 of the design.  ``stderr``
can block under load and ``BlockingIOError`` is exactly the failure mode the
policy outlaws.  Use :mod:`logging` with our non-blocking handlers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

_logger = logging.getLogger(__name__)

_HandlerSync = Callable[[ToolCallRequest], "ToolMessage | Command[Any]"]
_HandlerAsync = Callable[[ToolCallRequest], Awaitable["ToolMessage | Command[Any]"]]


def _tool_call_id(request: ToolCallRequest) -> str:
    call = getattr(request, "tool_call", None) or {}
    return call.get("id") or "unknown"


def _tool_name(request: ToolCallRequest) -> str:
    call = getattr(request, "tool_call", None) or {}
    name = call.get("name")
    if name:
        return name
    tool = getattr(request, "tool", None)
    return getattr(tool, "name", "<unknown>")


def _format_error_message(request: ToolCallRequest, exc: BaseException) -> str:
    return (
        f"Tool '{_tool_name(request)}' raised an unhandled exception "
        f"({type(exc).__name__}): {exc}.  The agent runtime converted it "
        f"into this error message so you can continue.  Treat it as a tool "
        f"failure — review the error, decide whether to retry with adjusted "
        f"arguments, fall back to a different approach, or report it to the "
        f"user."
    )


class SafeToolCallMiddleware(AgentMiddleware):
    """Catch any exception raised by a tool dispatch and convert it into a
    ``ToolMessage(status="error")``.

    Per the No-Agent-Error-State Policy this middleware MUST be the
    innermost (last in middleware list) tool-call wrapper, so that any
    exception escaping a tool body — including ones thrown by other
    middlewares wrapping the same call — is captured before it bubbles up.
    """

    name = "SafeToolCallMiddleware"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: _HandlerAsync,
    ) -> "ToolMessage | Command[Any]":
        try:
            return await handler(request)
        except asyncio.CancelledError:
            # Cancellation is cooperative; do NOT swallow it.  This is the
            # one exception the policy explicitly allows to propagate.
            raise
        except BaseException as exc:  # noqa: BLE001 — this IS the safety net
            _logger.error(
                "Tool '%s' (call id %s) raised %s: %s",
                _tool_name(request),
                _tool_call_id(request),
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return ToolMessage(
                tool_call_id=_tool_call_id(request),
                name=_tool_name(request),
                status="error",
                content=_format_error_message(request, exc),
            )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: _HandlerSync,
    ) -> "ToolMessage | Command[Any]":
        try:
            return handler(request)
        except BaseException as exc:  # noqa: BLE001 — this IS the safety net
            _logger.error(
                "Tool '%s' (call id %s) raised %s: %s",
                _tool_name(request),
                _tool_call_id(request),
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return ToolMessage(
                tool_call_id=_tool_call_id(request),
                name=_tool_name(request),
                status="error",
                content=_format_error_message(request, exc),
            )
