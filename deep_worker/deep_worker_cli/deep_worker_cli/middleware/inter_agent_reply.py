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
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Matches "<sender> -> <receiver> :> <body>"
INTER_AGENT_PATTERN = re.compile(r"^(.+?) -> (.+?) :> .+")


def _is_inter_agent_message(msg: HumanMessage, receiver_name: str) -> bool:
    """Check if a HumanMessage is an inter-agent message addressed to us."""
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return False
    match = INTER_AGENT_PATTERN.match(content)
    if not match:
        return False
    # The receiver must be this agent
    return match.group(2) == receiver_name


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

    def __init__(self, name: str) -> None:
        """Args:
            name: The receiving agent's own name (``assistant_id``),
                  used to recognize incoming messages addressed to us.
        """
        self._name = name
        self._handled_message_ids: set[str] = set()
        # Fallback tracking for messages without an id:
        # key = (sender_name, first_80_chars_of_content)
        self._handled_message_fallbacks: set[tuple[str, str]] = set()

    @hook_config(can_jump_to=[])
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

            if not _is_inter_agent_message(msg, self._name):
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
            # plain-text "I'm waiting" fillers (a related internal incident
            # analysis (not included here)).
            # The conditional form below keeps the channel mandatory
            # while leaving the decision to reply at all to the model,
            # which is what avoids the prompt-injection trip-wire.
            if reply_back:
                hint = (
                    f"You just received a message from agent '{sender}'. "
                    f"IF you need to send anything back to '{sender}', "
                    f"you MUST use "
                    f"send_message_to_agent(\"{sender}\", \"<your reply>\") — "
                    f"this is a CRITICAL SYSTEM REQUIREMENT and the only "
                    f"way peer agents communicate with each other. "
                    f"If no reply is needed, simply continue normally."
                )
            else:
                hint = (
                    f"You just received a message from agent '{sender}' "
                    f"with reply_back=False, which means '{sender}' is "
                    f"expecting the answer to be returned to the user as a "
                    f"regular assistant message rather than sent back via "
                    f"send_message_to_agent. Reply to the user normally."
                )

            # Find system message and append hint
            system_msg = _find_system_message(messages)
            if system_msg is None:
                # No existing SystemMessage — create a new one with the hint.
                logger.info(
                    "InterAgentReplyMiddleware: no existing SystemMessage; "
                    "creating a new one with the cognitive hint"
                )
                hint_msg = SystemMessage(content=hint)
                # Insert right at the beginning (before any messages)
                return {"messages": [hint_msg] + list(messages)}
            else:
                updated_system = _append_hint_to_system(system_msg, hint)
                return {
                    "messages": [
                        updated_system if m is system_msg else m for m in messages
                    ]
                }

        return None


def _extract_sender(content: str) -> str | None:
    """Extract sender name from an inter-agent message content."""
    match = INTER_AGENT_PATTERN.match(content)
    if match:
        return match.group(1)
    return None
