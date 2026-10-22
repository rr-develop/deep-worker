"""Shared middleware: resolve the inter-agent send Future at delivery.

DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md (Bug 2).

Problem: ``send_message_to_agent`` is fire-and-forget — its delivery result is
known the instant the tool runs. But in the proxy-MCP architecture the tool's
MCP Future is resolved only in the send loop of the *next* ``ChatModel.ainvoke``
(when the resulting ``ToolMessage`` is replayed). For a turn-ending tool that
next ``ainvoke`` never happens in the same graph run —
``SendMessageCompletionMiddleware`` jumps to the graph end first — so the Future
hangs until the next external impulse (120-270s later in production) and the
CLI reports the tool call as timed out, even though delivery succeeded.

Fix: resolve the Future at DELIVERY time using the ``awrap_tool_call`` seam,
which runs INSIDE the tools node (before control returns to the model node, so
before the jump-to-end). On a successful ``send_message_to_agent`` result we
call the model's public ``resolve_pending_tool`` immediately; the proxy handler
then returns to the model's CLI right away. The 120s handler timeout therefore
bounds DELIVERY only (a dead/unreachable channel), never the recipient's work.

This middleware is registered by CAPABILITY — any model exposing
``resolve_pending_tool`` (the duck-typed seam both the claude-code-sdk and ACP
chat models implement) — not by concrete class, so it lives in the shared
provider layer rather than in any single provider package.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)

# The single tool whose successful result we resolve early. Kept in sync with
# the CLI's SendMessageCompletionMiddleware.TOOL_NAME by value.
_TOOL_NAME = "send_message_to_agent"


def _parse_tool_result(raw: Any) -> dict[str, Any] | None:
    """Best-effort parse of a ToolMessage.content payload (JSON or str(dict))."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if parsed is None:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _mcp_content(raw: Any) -> dict[str, Any]:
    """Wrap a ToolMessage.content into the MCPResult shape the proxy expects."""
    if isinstance(raw, list):
        return {"content": raw}
    return {"content": [{"type": "text", "text": str(raw)}]}


class InterAgentDeliveryResolveMiddleware(AgentMiddleware):
    """Resolve the inter-agent send Future at delivery time (duck-typed, any proxy-MCP model)."""

    tools: "Sequence[BaseTool]" = []

    def __init__(self, model: Any) -> None:
        super().__init__()
        # Duck-typed: needs ``resolve_pending_tool(thread_id, id, content)``.
        self._model = model

    async def awrap_tool_call(
        self,
        request: Any,
        handler: "Callable[[Any], Awaitable[Any]]",
    ) -> Any:
        result = await handler(request)

        try:
            tool_call = getattr(request, "tool_call", None) or {}
            if tool_call.get("name") != _TOOL_NAME:
                return result
            if not isinstance(result, ToolMessage):
                return result

            data = _parse_tool_result(result.content)
            if data is None or not data.get("success"):
                # Failed/unparsable delivery: let the normal path surface it.
                return result

            thread_id = self._thread_id(request)
            if thread_id is None:
                return result

            self._model.resolve_pending_tool(
                thread_id, result.tool_call_id, _mcp_content(result.content)
            )
            # After resolve_call, the model finishes turn #1 in the
            # background (UserMessage(ToolResultBlock) +
            # AssistantMessage(TextBlock) + ResultMessage). We must
            # drain that tail here, otherwise the next ainvoke's
            # receive_response would read it BEFORE the model reacts to
            # the peer reply we send via client.query — see
            # a related internal incident analysis (not included here).
            if hasattr(self._model, "drain_after_inter_agent_send"):
                await self._model.drain_after_inter_agent_send(thread_id)
        except Exception:  # noqa: BLE001 — must never break a tool turn
            logger.debug(
                "InterAgentDeliveryResolveMiddleware: early resolve skipped",
                exc_info=True,
            )

        return result

    @staticmethod
    def _thread_id(request: Any) -> str | None:
        runtime = getattr(request, "runtime", None)
        config = getattr(runtime, "config", None)
        if not isinstance(config, dict):
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return None
        tid = configurable.get("thread_id")
        return tid if isinstance(tid, str) else None
