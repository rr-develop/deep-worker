"""Middleware that injects cognitive hints for incoming inter-agent messages.

When an agent receives a ``HumanMessage`` from another agent (detected by the
pattern ``<sender> -> <receiver> :> <body>``), this middleware injects a
system-prompt hint based on the ``reply_back`` flag in the message's
``additional_kwargs``:

* ``reply_back=True`` (default) → instruct the agent to reply via
  ``send_message_to_agent``.
* ``reply_back=False`` → instruct the agent to respond directly to the user.

The hint is injected only once per incoming message (tracked by message id,
or by (sender, content_prefix) as fallback for messages without an id).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)



# Matches "<sender> -> <receiver> :> <body>"
INTER_AGENT_PATTERN = re.compile(r"^(.+?) -> (.+?) :> .+")


def _is_inter_agent_message(msg: HumanMessage, receiver_name: str) -> bool:
    """Check if a HumanMessage is an inter-agent message addressed to us."""
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return False
    # Grok may prepend the cognitive hint above the payload line — scan
    # every line, not only the string start.
    for line in content.splitlines():
        match = INTER_AGENT_PATTERN.match(line.strip())
        if match and match.group(2) == receiver_name:
            return True
    match = INTER_AGENT_PATTERN.match(content)
    if not match:
        return False
    return match.group(2) == receiver_name


def _is_delivered_inter_agent_message(msg: HumanMessage, receiver_name: str) -> bool:
    """True for enqueue_message deliveries, not user text quoting the format."""
    kwargs = getattr(msg, "additional_kwargs", {}) or {}
    if not kwargs.get("inter_agent_delivery"):
        return False
    return _is_inter_agent_message(msg, receiver_name)


def _find_system_message(messages: list) -> SystemMessage | None:
    """Find the first SystemMessage. Returns None if not found."""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            return msg
    return None


def _append_hint_to_system(system_msg: SystemMessage, hint: str) -> SystemMessage:
    """Append a cognitive hint to a SystemMessage's content, preserving its id."""
    current = getattr(system_msg, "content", "") or ""
    separator = "\n\n" if current else ""
    new_content = current + separator + hint
    return SystemMessage(content=new_content, id=system_msg.id)


class InterAgentReplyMiddleware(AgentMiddleware):
    """Inject cognitive hints into the system prompt when an inter-agent
    message is received, guiding the model to either reply back to the
    sending agent (``reply_back=True``) or respond directly to the user
    (``reply_back=False``).

    The hint is injected once per incoming message via ``before_model``.
    On subsequent model calls (after tool execution) the middleware tracks
    already-handled messages and skips them.
    """

    # No tools registered by this middleware.
    tools: list = []

    def __init__(
        self,
        name: str,
        *,
        reply_tool_form: str = "send_message_to_agent",
        grok_mcp_note: bool = False,
        grok_receive_completion: bool = False,
    ) -> None:
        """Args:
            name: The receiving agent's own name (``assistant_id``),
                  used to recognize incoming messages addressed to us.
            reply_tool_form: The PROVIDER-CORRECT call form for the
                  ``send_message_to_agent`` tool, used verbatim in the
                  injected hint. Defaults to the bare name (backward
                  compatible). For claude-code/codex this is the
                  ``mcp__deepworker[_<pid>]__send_message_to_agent`` form
                  (see docs/DESIGN_MCP_PROXY_TOOL_NAME_PREFIX_AWARENESS.md).
            grok_mcp_note: When True, append a short "this is an MCP tool,
                  do not search its schema on disk" clarification to the
                  reply_back=True hint. Used for the grok provider, whose
                  call form stays the bare name but where the model
                  otherwise hunts for the tool schema on disk.
            grok_receive_completion: When True (grok only), a
                  ``reply_back=False`` delivery ends the turn immediately
                  with an ``AIMessage`` summarizing the sender's report —
                  mirroring ``SendMessageCompletionMiddleware`` on the
                  outbound side and avoiding a hung text-only grok turn.
        """
        self._name = name
        self._reply_tool_form = reply_tool_form
        self._grok_mcp_note = grok_mcp_note
        self._grok_receive_completion = grok_receive_completion
        self._handled_message_ids: set[str] = set()
        # Fallback tracking for messages without an id:
        # key = (sender_name, first_80_chars_of_content)
        self._handled_message_fallbacks: set[tuple[str, str]] = set()
    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AgentState,
        runtime: Any = None,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return None

        # Find the latest HumanMessage that looks like an inter-agent message
        # addressed to us and has not been handled yet.
        for msg in reversed(messages):
            if not isinstance(msg, HumanMessage):
                continue

            # Skip already handled messages
            msg_id = getattr(msg, "id", None)
            if msg_id and msg_id in self._handled_message_ids:
                continue

            if not _is_delivered_inter_agent_message(msg, self._name):
                continue

            # For messages without an id, use fallback tracking
            if not msg_id:
                sender = _extract_sender(msg.content)
                if sender is None:
                    continue
                content_prefix = (getattr(msg, "content", "") or "")[:80]
                fallback_key = (sender, content_prefix)
                if fallback_key in self._handled_message_fallbacks:
                    continue

            # Found an unhandled inter-agent message.
            # Extract reply_back from additional_kwargs (default True).
            kwargs = getattr(msg, "additional_kwargs", {}) or {}
            reply_back = kwargs.get("reply_back", True)
            if not isinstance(reply_back, bool):
                reply_back = True

            # Extract sender name from content: "<sender> -> <receiver> :> ..."
            sender = _extract_sender(msg.content) or "unknown"

            # Mark as handled
            if msg_id:
                self._handled_message_ids.add(msg_id)
            else:
                content_prefix = (getattr(msg, "content", "") or "")[:80]
                self._handled_message_fallbacks.add((sender, content_prefix))

            # Build the hint.
            # Policy (2026-06-12 strengthened): the hint uses a
            # CONDITIONAL imperative — the imperative applies to the
            # CHANNEL of the reply ("IF you reply, you MUST use the
            # tool"), NOT to the act of replying ("you MUST reply").
            # The original 2026-06-03 neutral-only phrasing was too
            # weak: the model sometimes ignored the tool and answered
            # in plain text. The earlier fully-imperative phrasing
            # ("You MUST reply…") was the OPPOSITE failure — it
            # triggered Claude's prompt-injection defenses and produced
            # plain-text "I'm waiting" fillers (see a related internal
            # incident analysis; not included here).
            # The conditional form below keeps the channel mandatory
            # while leaving the decision to reply at all to the model,
            # which is what avoids the prompt-injection trip-wire.
            if reply_back:
                grok_clarification = (
                    " (this is a deep-worker MCP tool, NOT a native built-in; "
                    "call it by name — the CLI routes it to our MCP server — "
                    "and do NOT search its schema on disk)"
                    if self._grok_mcp_note
                    else ""
                )
                hint = (
                    f"You just received a message from agent '{sender}'. "
                    f"IF you need to send anything back to '{sender}', "
                    f"you MUST use "
                    f"{self._reply_tool_form}(\"{sender}\", \"<your reply>\")"
                    f"{grok_clarification} — "
                    f"this is a CRITICAL SYSTEM REQUIREMENT and the only "
                    f"way peer agents communicate with each other. "
                    f"If no reply is needed, simply continue normally."
                )
            else:
                if self._grok_receive_completion:
                    body = _extract_inter_agent_body(msg.content)
                    logger.info(
                        "InterAgentReplyMiddleware: grok receive completion "
                        "for reply_back=False from %r",
                        sender,
                    )
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    f"Agent '{sender}' confirmed receipt of "
                                    f"the task: {body}"
                                )
                            )
                        ],
                        "jump_to": "end",
                    }
                grok_no_tool_note = (
                    " Do NOT call send_message_to_agent, execute, grep, or "
                    "any other tool for this — respond with a plain "
                    "assistant text message to the user."
                    if self._grok_mcp_note
                    else ""
                )
                hint = (
                    f"You just received agent '{sender}'s FINAL reply "
                    f"(reply_back=False) — delivery is COMPLETE. The "
                    f"inter-agent line at the end of this message IS the "
                    f"answer you were waiting for; do NOT say the reply "
                    f"will arrive later or that you are still waiting. "
                    f"Summarize for the user what '{sender}' reported, "
                    f"including any markers verbatim."
                    f"{grok_no_tool_note}"
                )

            # Find system message and append hint
            system_msg = _find_system_message(messages)
            if system_msg is None:
                # grok's default system prompt is injected at model-call time
                # (not stored in state).  A freestanding SystemMessage merged
                # via add_messages lands at the *tail*, after the inter-agent
                # HumanMessage, so GrokChatModel's user_text delta may omit the
                # hint.  Annotate the HumanMessage itself so the hint always
                # rides in the user turn text.
                if self._grok_mcp_note:
                    logger.info(
                        "InterAgentReplyMiddleware: annotating inter-agent "
                        "HumanMessage with cognitive hint (grok)"
                    )
                    annotated = HumanMessage(
                        content=f"{hint}\n\n{msg.content}",
                        id=msg.id,
                        additional_kwargs=dict(
                            getattr(msg, "additional_kwargs", {}) or {}
                        ),
                    )
                    return {
                        "messages": [
                            annotated if m is msg else m for m in messages
                        ]
                    }
                # No existing SystemMessage — create a new one with the hint.
                logger.info(
                    "InterAgentReplyMiddleware: no existing SystemMessage; "
                    "creating a new one with the cognitive hint"
                )
                hint_msg = SystemMessage(content=hint)
                return {"messages": [hint_msg] + list(messages)}
            else:
                updated_system = _append_hint_to_system(system_msg, hint)
                return {
                    "messages": [
                        updated_system if m is system_msg else m for m in messages
                    ]
                }

        return None

    def _pending_grok_text_reply(self, messages: list) -> bool:
        """True while grok still owes a text reply for reply_back=False delivery."""
        if not self._grok_mcp_note:
            return False
        reply_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, HumanMessage):
                continue
            if not _is_delivered_inter_agent_message(msg, self._name):
                continue
            kwargs = getattr(msg, "additional_kwargs", {}) or {}
            if kwargs.get("reply_back", True) is False:
                reply_idx = i
                break
        if reply_idx is None:
            return False
        for msg in messages[reply_idx + 1 :]:
            if isinstance(msg, AIMessage):
                content = getattr(msg, "content", "") or ""
                if isinstance(content, str) and content.strip():
                    return False
        return True

    def _maybe_filter_grok_tools(self, request: ModelRequest) -> ModelRequest:
        """Drop tools while grok must answer the user (reply_back=False)."""
        if not self._pending_grok_text_reply(request.messages):
            return request
        grok_turn_hint = (
            "CRITICAL: An inter-agent reply with reply_back=False just "
            "arrived in the latest HumanMessage. Respond to the USER now "
            "in plain text — summarize what the sending agent reported. "
            "Do NOT call any tools. Do NOT say you are still waiting."
        )
        from deepagents.middleware._utils import append_to_system_message

        new_system = append_to_system_message(
            request.system_message,
            grok_turn_hint,
        )
        if request.tools:
            logger.info(
                "InterAgentReplyMiddleware: stripping %d tool(s) for grok "
                "reply_back=False turn (text-only)",
                len(request.tools),
            )
            return request.override(system_message=new_system, tools=[])
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._maybe_filter_grok_tools(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._maybe_filter_grok_tools(request))


def _extract_sender(content: str) -> str | None:
    """Extract sender name from an inter-agent message content."""
    for line in content.splitlines():
        match = INTER_AGENT_PATTERN.match(line.strip())
        if match:
            return match.group(1)
    match = INTER_AGENT_PATTERN.match(content)
    if match:
        return match.group(1)
    return None


def _extract_inter_agent_body(content: str) -> str:
    """Return the payload after ``'<sender> -> <receiver> :> '``."""
    for line in content.splitlines():
        stripped = line.strip()
        if " :> " in stripped:
            return stripped.split(" :> ", 1)[1].strip()
    if " :> " in content:
        return content.split(" :> ", 1)[1].strip()
    return content.strip()
