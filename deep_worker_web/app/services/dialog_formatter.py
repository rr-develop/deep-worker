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


# Sentinel ``human`` payload used for a leading SystemMessage block that
# precedes any real HumanMessage (e.g. the agent's base system prompt).
# Such a block has no user line — only the system content(s).
_SYSTEM_PREAMBLE_TYPE = "SystemPreamble"


@dataclass
class Turn:
    """One slice of conversation: a HumanMessage + everything until the next.

    ``stream`` is the in-order list of post-Human items as ``(kind, payload)``
    tuples used by ``full`` mode to interleave text, tool-calls and system
    messages. The parsed ``ai_text_messages`` list holds just the
    text-bearing AIMessages used by ``final`` and ``text`` modes.

    ``system_messages`` holds the content of SystemMessages that appear
    inside this turn (the base prompt, or runtime middleware hints such as
    the inter-agent reply hint). They are rendered at ``text`` and ``full``
    verbosity. ⚠️ POLICY: history rendered at ``text`` or higher MUST surface
    system messages — a "full" rendering without them is misleading and is
    forbidden. See DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md
    §"System messages are first-class in history rendering".
    """

    human: dict[str, Any]
    ai_text_messages: list[str] = field(default_factory=list)
    stream: list[tuple[str, Any]] = field(default_factory=list)
    # entries: ("text", str) | ("tool_call", dict) | ("system", str)
    system_messages: list[str] = field(default_factory=list)


def _extract_system_content(msg: dict[str, Any]) -> str:
    """Pull displayable text out of a SystemMessage dict."""
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # Same text-block handling as AIMessages, with a plain-string
        # fallback so nothing is silently dropped.
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "".join(parts)
    return str(c)


def split_into_turns(history: list[dict[str, Any]]) -> list[Turn]:
    """Group a flat history list into turns keyed by HumanMessage.

    SystemMessages are captured (never dropped): a SystemMessage before the
    first HumanMessage becomes a leading ``SystemPreamble`` turn; a
    SystemMessage inside a turn is recorded both in ``system_messages`` and
    in ``stream`` (preserving chronological order for ``full`` mode).
    """
    turns: list[Turn] = []
    current: Turn | None = None
    preamble: Turn | None = None
    for msg in history:
        msg_type = msg.get("type")
        if msg_type == "HumanMessage":
            if current is not None:
                turns.append(current)
            current = Turn(human=msg)
            continue
        if msg_type == "SystemMessage":
            content = _extract_system_content(msg)
            if current is None:
                # Leading system block (base prompt) — accumulate into a
                # synthetic preamble turn so it renders before the first
                # user turn.
                if preamble is None:
                    preamble = Turn(human={"type": _SYSTEM_PREAMBLE_TYPE, "content": ""})
                    turns.append(preamble)
                preamble.system_messages.append(content)
                preamble.stream.append(("system", content))
            else:
                current.system_messages.append(content)
                current.stream.append(("system", content))
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
    # ``text`` mode shows every text-bearing AIMessage AND every
    # SystemMessage in this turn, in chronological order. System messages
    # (base prompt / middleware hints) are first-class — omitting them
    # would make the rendering misleading (forbidden by policy).
    parts: list[str] = []
    for kind, payload in turn.stream:
        if kind == "text":
            parts.append(f"--- Assistant ---\n{payload}")
        elif kind == "system":
            parts.append(f"--- System ---\n{payload}")
    if not turn.ai_text_messages and not turn.system_messages:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"
    if not turn.ai_text_messages:
        # System-only turn (e.g. a leading system prompt): show systems,
        # no assistant placeholder.
        return "\n\n".join(parts)
    return "\n\n".join(parts)


def _render_assistant_block_full(turn: Turn) -> str:
    if not turn.stream:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"
    parts: list[str] = []
    for kind, payload in turn.stream:
        if kind == "text":
            parts.append(f"--- Assistant ---\n{payload}")
        elif kind == "system":
            parts.append(f"--- System ---\n{payload}")
        elif kind == "tool_call":
            tc_name = payload.get("name", "?") if isinstance(payload, dict) else "?"
            line = format_tool_call(payload) if isinstance(payload, dict) else str(payload)
            parts.append(f"--- Assistant (tool: {tc_name}) ---\n{line}")
    return "\n\n".join(parts)


def _render_system_preamble(turn: Turn, verbosity: Verbosity) -> str | None:
    """Render a leading SystemMessage block (base prompt) that precedes any
    HumanMessage. Visible at ``text``/``full``; omitted at ``final``.
    Returns None when there is nothing to show at this verbosity.
    """
    if verbosity == "final":
        return None
    if not turn.system_messages:
        return None
    blocks = [f"--- System ---\n{content}" for content in turn.system_messages]
    return "\n\n".join(blocks)


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


def render_system_prompt(
    system_prompt: str,
    provider: str | None,
    model: str | None,
) -> str:
    """Render the invariant (startup) system prompt as a plain-text block.

    This is mode #4 of the Full Dialog verbosity combo (§9 of
    DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md). The system prompt is the
    provider-AGNOSTIC role/instruction text the agent was built with — it
    does NOT live in the checkpoint and is NOT a conversation turn, so it is
    rendered standalone (not via ``split_into_turns``/``render_dialog``).
    """
    body = system_prompt if (system_prompt and system_prompt.strip()) else "(no system prompt)"
    header = (
        f"--- System Prompt (invariant, provider={provider}, model={model}) ---"
    )
    return f"{header}\n{body}"


def render_tools(tools: list[dict[str, Any]], provider: str | None) -> str:
    """Render the agent's available-tool list as a plain-text block.

    Mode #5 of the verbosity combo (§9). The tool list MAY depend on the
    active provider, so the provider is shown in the header. Each tool is a
    ``- <name>: <description first line>`` line.
    """
    header = f"--- Available Tools (provider={provider}, count={len(tools)}) ---"
    if not tools:
        return f"{header}\n(no tools)"
    lines: list[str] = []
    for t in tools:
        name = t.get("name", "?") if isinstance(t, dict) else "?"
        desc = t.get("description", "") if isinstance(t, dict) else ""
        if not isinstance(desc, str):
            desc = str(desc)
        first_line = desc.splitlines()[0] if desc.strip() else ""
        if first_line:
            lines.append(f"- {name}: {first_line}")
        else:
            lines.append(f"- {name}")
    return header + "\n" + "\n".join(lines)


def render_dialog(turns: list[Turn], verbosity: str = "final") -> str:
    """Render a list of Turns to a plain-text dialog at the requested verbosity."""
    if verbosity not in ("final", "text", "full"):
        raise ValueError(f"Unknown verbosity: {verbosity!r}")

    blocks: list[str] = []
    for turn in turns:
        # A leading SystemMessage block (base prompt) has no user line.
        # Render it as a standalone System block at text/full; skip at final.
        if turn.human.get("type") == _SYSTEM_PREAMBLE_TYPE:
            rendered = _render_system_preamble(turn, verbosity)
            if rendered is not None:
                blocks.append(rendered)
            continue
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
