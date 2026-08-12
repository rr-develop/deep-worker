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

Implementation note
-------------------

The turn-ending mechanism itself is generic and lives in
``deep_worker_cli.middleware.tool_completion``; see
``docs/DESIGN_TOOL_COMPLETION_MIDDLEWARE.md``. This module keeps the
``send_message_to_agent`` declaration and remains the class that is
wired into the agent, so existing importers, ``isinstance`` checks and
the middleware stack position are unaffected.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from deep_worker_cli.middleware.tool_completion import (
    ToolCompletionMiddleware,
    TurnEndingTool,
    _parse_tool_result,  # noqa: F401  — re-exported for backwards compatibility
)

logger = logging.getLogger(__name__)

# Name of the tool whose successful ToolMessage triggers the turn-end.
# Kept as a module-level constant so tests and importers can refer to
# the single source of truth rather than hard-coding the string.
TOOL_NAME = "send_message_to_agent"


class SendMessageCompletionMiddleware(ToolCompletionMiddleware):
    """Treat a successful ``send_message_to_agent`` call as the agent's
    final response for the turn, plus any additional turn-ending tools
    declared by the launcher.

    Rationale: an outgoing inter-agent message *is* the agent's answer
    for this turn — just addressed to another agent instead of the
    human user. Making it an explicit text ``AIMessage`` in the
    conversation state means every history-based client renders it
    as-is, without special cases.

    The built-in ``send_message_to_agent`` declaration is always
    registered first. Additional declarations come from the launcher
    (``create_cli_agent(..., turn_ending_tools=[...])``); a declaration
    that collides with the built-in raises
    ``DuplicateTurnEndingToolError`` rather than being silently ignored.
    """

    def __init__(self, additional: Iterable[TurnEndingTool] = ()) -> None:
        built_in = TurnEndingTool(tool_name=TOOL_NAME)
        super().__init__([built_in, *additional])
