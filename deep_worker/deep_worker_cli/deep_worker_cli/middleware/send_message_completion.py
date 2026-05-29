"""Turn-ending middleware for the ``send_message_to_agent`` tool.

See design document:
    deep_worker_cli/docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md

Summary
-------

When an agent calls ``send_message_to_agent("beta", "hello")`` the LLM
produces a tool-only ``AIMessage`` (empty content, just a tool call),
LangGraph executes the tool, and a ``ToolMessage`` with the delivery
result is appended to state. Without this middleware, the model would
then be invoked again to produce a text response — usually an empty
filler like "Message sent, waiting for reply" that adds no information,
and the actual outgoing message (``agent:<sender>:> <text>``) is buried
inside a ``ToolMessage`` that every history-based client tends to skip.

This middleware hooks ``before_model`` (i.e. **after** the tool has
executed, **before** the model's next call) and, if the last state
message is a successful ``send_message_to_agent`` ``ToolMessage``,
injects a text ``AIMessage`` containing the same prefixed string the
recipient saw and jumps to the graph's end node, skipping that
redundant model call.

Result:

    AIMessage(tool_calls=[send_message_to_agent(...)], content="")
    ToolMessage(name="send_message_to_agent", content='{"success":true,...}')
    AIMessage(content="agent:alpha:> hello beta")            # ← injected
    [turn ends]

Every client that renders the conversation by reading ``AIMessage``
contents (Full Dialog in deep_worker_web, ``dwcli --history``,
subscribers via ``_notify_subscribers`` — anywhere) now sees the
outgoing message as-is, symmetric to how the receiving agent records
the incoming ``HumanMessage``.

On delivery failure (``success: False``) the middleware stays out of
the way so the model can see and react to the error.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage

from deepagents.middleware.message_timestamps import ensure_message_created_at_after

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Name of the tool whose successful ToolMessage triggers the turn-end.
# Kept as a module-level constant so tests and importers can refer to
# the single source of truth rather than hard-coding the string.
TOOL_NAME = "send_message_to_agent"


def _parse_tool_result(raw: Any) -> dict[str, Any] | None:
    """Best-effort parse of a ``ToolMessage.content`` payload produced
    by our tool.

    LangChain serializes dict tool results in at least two ways
    depending on the integration path:

    * JSON (most common): ``'{"success": true, "delivered_content": "..."}'``
    * Python-repr (``str(dict)``): ``"{'success': True, 'delivered_content': '...'}"``

    Returns the parsed dict on success, ``None`` on anything else.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    # Try JSON first (preferred, produced by json.dumps()).
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    # Fall back to Python literal eval for str(dict) style.
    if parsed is None:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


class SendMessageCompletionMiddleware(AgentMiddleware):
    """Treat a successful ``send_message_to_agent`` call as the agent's
    final response for the turn.

    Rationale: an outgoing inter-agent message *is* the agent's answer
    for this turn — just addressed to another agent instead of the
    human user. Making it an explicit text ``AIMessage`` in the
    conversation state means every history-based client renders it
    as-is, without special cases.

    Implementation: ``before_model`` hook with
    ``@hook_config(can_jump_to=["end"])``. When the last state message
    is a successful ``send_message_to_agent`` ``ToolMessage``, returns
    ``{"messages": [AIMessage(content=<delivered>)], "jump_to": "end"}``
    to inject the echo and skip the pending model call. In any other
    situation returns ``None`` so the normal agent loop proceeds.
    """

    # Explicit empty list — this middleware registers no tools of its own.
    # Declared at class level so ``middleware.tools`` is always iterable
    # even before ``__init__`` runs (some langgraph code introspects this).
    tools: list = []

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AgentState,
        runtime: "Runtime[Any] | None" = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        # Defensive: state might be an unusual shape in tests; use .get().
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return None

        last = messages[-1]

        # Gate 1: must be a ToolMessage.
        if not isinstance(last, ToolMessage):
            return None

        # Gate 2: ToolMessage.name must match our tool. (Anthropic
        # sometimes omits .name; getattr with default handles that.)
        if getattr(last, "name", None) != TOOL_NAME:
            return None

        # Gate 3: parse the tool's structured result.
        data = _parse_tool_result(last.content)
        if data is None:
            logger.debug(
                "SendMessageCompletionMiddleware: could not parse "
                "ToolMessage.content; letting the normal flow continue"
            )
            return None

        # Gate 4: success only — failures should surface to the model.
        if not data.get("success"):
            return None

        # Gate 5: delivered_content must be a non-empty string.
        delivered = data.get("delivered_content", "")
        if not isinstance(delivered, str) or not delivered.strip():
            return None

        # All gates passed — end the turn with the echo.
        logger.info(
            "SendMessageCompletionMiddleware: ending turn with echo "
            "(delivered_content preview=%r)",
            delivered[:120],
        )
        echo = AIMessage(content=delivered)
        ensure_message_created_at_after(messages, echo)
        return {"messages": [echo], "jump_to": "end"}
