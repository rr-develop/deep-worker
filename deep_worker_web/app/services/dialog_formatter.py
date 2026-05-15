"""Pure functions that build a plain-text Full Dialog rendering.

Design: ``docs/DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md``.

The renderer consumes the message dict shape produced by
``AgentProcess._handle_get_history`` (RPyC) and returns a plain string.
Three verbosity levels are supported:

- ``final`` (default): one assistant block per turn, containing the LAST
  text-bearing AIMessage.
- ``text``: every text-bearing AIMessage in the turn, in order.
- ``full``: same as ``text`` plus interleaved compact tool-call summaries.

Four sources of user-side messages — direct user input, ``TaskManager:>``
prompts, ``agent:<name>:>`` inter-agent messages, and context-compaction
summarization injections — all produce first-class turns at every
verbosity. Summarization turns are labelled with a dedicated
``--- Summarization ---`` header so the reader can distinguish synthetic
context-management events from real user input; they are NEVER dropped,
because hiding them makes post-compaction agent behaviour unreadable
(see ``docs/DESIGN_SUMMARIZATION_VISIBLE.md``).

This module is intentionally framework-agnostic — it works on plain dicts
and is fully synchronous, so the test suite never needs RPyC or asyncio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Verbosity = Literal["final", "text", "full"]
HumanKind = Literal["user", "taskmanager", "agent_echo", "summarization"]

NO_HISTORY_TEXT = "No conversation history yet."
NO_RESPONSE_TEXT = "(no response yet)"

_TOOL_CALL_ARGS_MAX = 200
_AGENT_PREFIX_SCAN_WINDOW = 64

# Marker substrings that identify summarization-time synthetic
# HumanMessages (context compaction). Detection is unchanged; these
# turns are now labelled with "--- Summarization ---" instead of being
# filtered (see DESIGN_SUMMARIZATION_VISIBLE.md).
_SUMMARIZATION_PREFIX = (
    "You are in the middle of a conversation that has been summarized"
)
_SUMMARIZATION_INLINE_MARKER = "Context Extraction Assistant"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def classify_human(content: str) -> HumanKind:
    """Classify a HumanMessage by its leading marker.

    Order of checks matters; first match wins. All four kinds produce a
    visible rendered turn — ``summarization`` is additionally labelled
    with its own header so the reader can distinguish synthetic
    context-compaction events from real user input.
    """
    if content.startswith("TaskManager:>"):
        return "taskmanager"
    if content.startswith("agent:") and ":> " in content[:_AGENT_PREFIX_SCAN_WINDOW]:
        return "agent_echo"
    head = content[:200]
    if content.startswith(_SUMMARIZATION_PREFIX) or _SUMMARIZATION_INLINE_MARKER in head:
        return "summarization"
    return "user"


def extract_text(msg: dict[str, Any]) -> str:
    """Pull the user-visible text out of an AIMessage dict.

    Handles both ``content: str`` and ``content: list[ContentBlock]``.
    Tool-only / non-text blocks contribute nothing.
    """
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def format_tool_call(tc: dict[str, Any]) -> str:
    """Render a compact ``→ name(args)`` line, truncating long args."""
    name = tc.get("name", "?")
    args = tc.get("args", {})
    try:
        args_str = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        args_str = str(args)
    if len(args_str) > _TOOL_CALL_ARGS_MAX:
        args_str = args_str[: _TOOL_CALL_ARGS_MAX - 3] + "..."
    return f"→ {name}({args_str})"


def _strip_prefix(content: str, kind: HumanKind) -> str:
    """Remove the source-marker prefix from a user-side message body."""
    if kind == "taskmanager":
        # "TaskManager:>" optionally followed by space/newline
        rest = content[len("TaskManager:>"):]
        return rest.lstrip(" \t")  # keep newlines intact for readability
    if kind == "agent_echo":
        idx = content.find(":> ")
        if idx == -1:
            return content
        return content[idx + 3:]
    return content


def _agent_echo_sender(content: str) -> str:
    """Extract the sender name from ``agent:<name>:> ...``."""
    if not content.startswith("agent:"):
        return "?"
    rest = content[len("agent:"):]
    end = rest.find(":> ")
    if end == -1:
        return "?"
    return rest[:end]


def _user_header(kind: HumanKind, content: str) -> str:
    if kind == "user":
        return "--- User ---"
    if kind == "taskmanager":
        return "--- TaskManager ---"
    if kind == "agent_echo":
        return f"--- From agent:{_agent_echo_sender(content)} ---"
    if kind == "summarization":
        return "--- Summarization ---"
    # defensive; classify_human exhaustively covers HumanKind
    return "--- (unknown) ---"


# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One slice of conversation: a HumanMessage + everything until the next.

    ``stream`` is the in-order list of post-Human items as ``(kind, payload)``
    tuples used by ``full`` mode to interleave text and tool-calls. The
    parsed ``ai_text_messages`` list holds just the text-bearing AIMessages
    used by ``final`` and ``text`` modes.
    """

    human: dict[str, Any]
    ai_text_messages: list[str] = field(default_factory=list)
    stream: list[tuple[str, Any]] = field(default_factory=list)
    # entries: ("text", str) | ("tool_call", dict)


def split_into_turns(history: list[dict[str, Any]]) -> list[Turn]:
    """Group a flat history list into turns keyed by HumanMessage."""
    turns: list[Turn] = []
    current: Turn | None = None
    for msg in history:
        msg_type = msg.get("type")
        if msg_type == "HumanMessage":
            if current is not None:
                turns.append(current)
            current = Turn(human=msg)
            continue
        if current is None:
            # AIMessage / ToolMessage before any HumanMessage — skip
            continue
        if msg_type == "AIMessage":
            text = extract_text(msg)
            if text and text.strip():
                current.ai_text_messages.append(text)
                current.stream.append(("text", text))
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    current.stream.append(("tool_call", tc))
        # ToolMessage payloads are not rendered (noisy, redundant with the
        # corresponding tool_call line in full mode).
    if current is not None:
        turns.append(current)
    return turns


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_assistant_block_final(turn: Turn) -> str:
    if turn.ai_text_messages:
        return f"--- Assistant ---\n{turn.ai_text_messages[-1]}"
    return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"


def _render_assistant_block_text(turn: Turn) -> str:
    if not turn.ai_text_messages:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"
    body = "\n\n".join(turn.ai_text_messages)
    return f"--- Assistant ---\n{body}"


def _render_assistant_block_full(turn: Turn) -> str:
    if not turn.stream:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"
    parts: list[str] = []
    for kind, payload in turn.stream:
        if kind == "text":
            parts.append(f"--- Assistant ---\n{payload}")
        elif kind == "tool_call":
            tc_name = payload.get("name", "?") if isinstance(payload, dict) else "?"
            line = format_tool_call(payload) if isinstance(payload, dict) else str(payload)
            parts.append(f"--- Assistant (tool: {tc_name}) ---\n{line}")
    return "\n\n".join(parts)


def _render_turn(turn: Turn, kind: HumanKind, verbosity: Verbosity) -> str:
    raw_content = turn.human.get("content", "")
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    body = _strip_prefix(raw_content, kind)
    header = _user_header(kind, raw_content)

    if verbosity == "final":
        ai_block = _render_assistant_block_final(turn)
    elif verbosity == "text":
        ai_block = _render_assistant_block_text(turn)
    elif verbosity == "full":
        ai_block = _render_assistant_block_full(turn)
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"Unknown verbosity: {verbosity!r}")

    return f"{header}\n{body}\n\n{ai_block}"


def render_dialog(turns: list[Turn], verbosity: str = "final") -> str:
    """Render a list of Turns to a plain-text dialog at the requested verbosity."""
    if verbosity not in ("final", "text", "full"):
        raise ValueError(f"Unknown verbosity: {verbosity!r}")

    blocks: list[str] = []
    for turn in turns:
        raw = turn.human.get("content", "")
        if not isinstance(raw, str):
            raw = str(raw)
        kind = classify_human(raw)
        # Summarization turns are labelled, not filtered (see
        # DESIGN_SUMMARIZATION_VISIBLE.md).
        blocks.append(_render_turn(turn, kind, verbosity))

    if not blocks:
        return NO_HISTORY_TEXT
    return "\n\n".join(blocks)
