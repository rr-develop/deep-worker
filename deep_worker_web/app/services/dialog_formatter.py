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
from datetime import datetime, timezone
from typing import Any, Literal

Verbosity = Literal["final", "text", "full"]
HumanKind = Literal["user", "taskmanager", "agent_echo", "summarization"]
TimestampMode = Literal["local", "utc"]

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


def format_timestamp(raw: Any, mode: TimestampMode = "local") -> str:
    """Render a message's ``created_at`` for a block header.

    Messages carry ``created_at`` as ISO-8601 UTC (stamped once and never
    rewritten, see ``deepagents.middleware.message_timestamps``). Readers work
    in wall-clock time, so ``local`` converts to the host timezone; ``utc``
    keeps it as-is.

    The zone is ALWAYS spelled out — ``+02:00`` for local, ``UTC`` for UTC —
    because a bare wall-clock time next to a log line in another zone is how
    incident timelines get misread. Seconds are kept, microseconds dropped.

    Returns ``""`` when there is no usable timestamp: old checkpoints predate
    the stamping middleware, and a missing time must degrade to "no time
    shown" rather than to a fabricated or bogus one.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        # Stamps are documented UTC; a naive value means an older writer.
        parsed = parsed.replace(tzinfo=timezone.utc)
    if mode == "utc":
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    local = parsed.astimezone()
    offset = local.strftime("%z")
    pretty_offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} {pretty_offset}"


def _stamped(label: str, raw: Any, mode: TimestampMode) -> str:
    """``--- Assistant 2026-08-29 14:29:22 +02:00 ---``, or plain when unknown."""
    ts = format_timestamp(raw, mode)
    return f"--- {label} {ts} ---" if ts else f"--- {label} ---"


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


def _user_header(
    kind: HumanKind,
    content: str,
    created_at: Any = None,
    mode: TimestampMode = "local",
) -> str:
    if kind == "user":
        return _stamped("User", created_at, mode)
    if kind == "taskmanager":
        return _stamped("TaskManager", created_at, mode)
    if kind == "agent_echo":
        return _stamped(f"From agent:{_agent_echo_sender(content)}", created_at, mode)
    if kind == "summarization":
        return _stamped("Summarization", created_at, mode)
    # defensive; classify_human exhaustively covers HumanKind
    return _stamped("(unknown)", created_at, mode)


# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------


# Sentinel ``human`` payload used for a leading SystemMessage block that
# precedes any real HumanMessage (e.g. the agent's base system prompt).
# Such a block has no user line — only the system content(s).
_SYSTEM_PREAMBLE_TYPE = "SystemPreamble"

#: Set by the agent runtime on the SystemMessage it appends for a
#: recoverable failure, and forwarded through ``serialize_message``.
#: Matching on this rather than on the ``[runtime]`` text prefix: the flag
#: is structured data put there for the purpose, the prefix is prose that
#: can be reworded without anyone noticing the rendering broke.
RECOVERABLE_ERROR_KEY = "recoverable_error"


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
    # ``created_at`` of each entry, positionally parallel to the lists above.
    # Kept alongside rather than inside them so the existing payload shapes
    # (and every consumer of ``stream``) stay exactly as they were.
    ai_text_timestamps: list[Any] = field(default_factory=list)
    stream_timestamps: list[Any] = field(default_factory=list)
    system_timestamps: list[Any] = field(default_factory=list)
    # Positionally parallel to ``system_messages``: True where the runtime
    # stamped the message as a recoverable failure. Those are the only
    # system messages shown at ``final`` — the base prompt and routine
    # middleware hints stay hidden there, which is the point of that
    # verbosity. See docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §5.
    system_error_flags: list[bool] = field(default_factory=list)


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
            created = msg.get("created_at")
            is_error = msg.get(RECOVERABLE_ERROR_KEY) is True
            if current is None:
                # Leading system block (base prompt) — accumulate into a
                # synthetic preamble turn so it renders before the first
                # user turn.
                if preamble is None:
                    preamble = Turn(human={"type": _SYSTEM_PREAMBLE_TYPE, "content": ""})
                    turns.append(preamble)
                preamble.system_messages.append(content)
                preamble.system_timestamps.append(created)
                preamble.system_error_flags.append(is_error)
                preamble.stream.append(("system", content))
                preamble.stream_timestamps.append(created)
            else:
                current.system_messages.append(content)
                current.system_timestamps.append(created)
                current.system_error_flags.append(is_error)
                current.stream.append(("system", content))
                current.stream_timestamps.append(created)
            continue
        if current is None:
            # AIMessage / ToolMessage before any HumanMessage — skip
            continue
        if msg_type == "AIMessage":
            created = msg.get("created_at")
            text = extract_text(msg)
            if text and text.strip():
                current.ai_text_messages.append(text)
                current.ai_text_timestamps.append(created)
                current.stream.append(("text", text))
                current.stream_timestamps.append(created)
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    current.stream.append(("tool_call", tc))
                    # A tool call carries its own stamp; fall back to the
                    # message's when the provider did not supply one.
                    current.stream_timestamps.append(
                        tc.get("created_at") or created
                    )
        # ToolMessage payloads are not rendered (noisy, redundant with the
        # corresponding tool_call line in full mode).
    if current is not None:
        turns.append(current)
    return turns


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _stream_stamp(turn: Turn, index: int) -> Any:
    """``created_at`` for ``turn.stream[index]``, tolerant of a short list."""
    if index < len(turn.stream_timestamps):
        return turn.stream_timestamps[index]
    return None


def _render_assistant_block_final(turn: Turn, mode: TimestampMode = "local") -> str:
    # A recoverable failure is shown even here. ``final`` hides system
    # messages by design — but hiding THIS one made a failed turn read as
    # a finished one: the model's last musing was rendered as the answer
    # while the runtime error was dropped entirely. Reported by the owner
    # 2026-09-04 ("it just looked like the model's final answer").
    # See docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §5.
    failures = [
        (content, turn.system_timestamps[index] if index < len(turn.system_timestamps) else None)
        for index, content in enumerate(turn.system_messages)
        if index < len(turn.system_error_flags) and turn.system_error_flags[index]
    ]

    parts: list[str] = []
    if turn.ai_text_messages:
        stamp = (
            turn.ai_text_timestamps[-1]
            if turn.ai_text_timestamps
            else None
        )
        parts.append(
            f"{_stamped('Assistant', stamp, mode)}\n{turn.ai_text_messages[-1]}"
        )
    elif not failures:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"

    parts.extend(
        f"{_stamped('System', stamp, mode)}\n{content}" for content, stamp in failures
    )
    return "\n\n".join(parts)


def _render_assistant_block_text(turn: Turn, mode: TimestampMode = "local") -> str:
    # ``text`` mode shows every text-bearing AIMessage AND every
    # SystemMessage in this turn, in chronological order. System messages
    # (base prompt / middleware hints) are first-class — omitting them
    # would make the rendering misleading (forbidden by policy).
    parts: list[str] = []
    for index, (kind, payload) in enumerate(turn.stream):
        stamp = _stream_stamp(turn, index)
        if kind == "text":
            parts.append(f"{_stamped('Assistant', stamp, mode)}\n{payload}")
        elif kind == "system":
            parts.append(f"{_stamped('System', stamp, mode)}\n{payload}")
    if not turn.ai_text_messages and not turn.system_messages:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"
    if not turn.ai_text_messages:
        # System-only turn (e.g. a leading system prompt): show systems,
        # no assistant placeholder.
        return "\n\n".join(parts)
    return "\n\n".join(parts)


def _render_assistant_block_full(turn: Turn, mode: TimestampMode = "local") -> str:
    if not turn.stream:
        return f"--- Assistant ---\n{NO_RESPONSE_TEXT}"
    parts: list[str] = []
    for index, (kind, payload) in enumerate(turn.stream):
        stamp = _stream_stamp(turn, index)
        if kind == "text":
            parts.append(f"{_stamped('Assistant', stamp, mode)}\n{payload}")
        elif kind == "system":
            parts.append(f"{_stamped('System', stamp, mode)}\n{payload}")
        elif kind == "tool_call":
            tc_name = payload.get("name", "?") if isinstance(payload, dict) else "?"
            line = format_tool_call(payload) if isinstance(payload, dict) else str(payload)
            parts.append(
                f"{_stamped(f'Assistant (tool: {tc_name})', stamp, mode)}\n{line}"
            )
    return "\n\n".join(parts)


def _render_system_preamble(
    turn: Turn, verbosity: Verbosity, mode: TimestampMode = "local"
) -> str | None:
    """Render a leading SystemMessage block (base prompt) that precedes any
    HumanMessage. Visible at ``text``/``full``; omitted at ``final``.
    Returns None when there is nothing to show at this verbosity.
    """
    if verbosity == "final":
        return None
    if not turn.system_messages:
        return None
    blocks: list[str] = []
    for index, content in enumerate(turn.system_messages):
        stamp = (
            turn.system_timestamps[index]
            if index < len(turn.system_timestamps)
            else None
        )
        blocks.append(f"{_stamped('System', stamp, mode)}\n{content}")
    return "\n\n".join(blocks)


def _render_turn(
    turn: Turn,
    kind: HumanKind,
    verbosity: Verbosity,
    mode: TimestampMode = "local",
) -> str:
    raw_content = turn.human.get("content", "")
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    body = _strip_prefix(raw_content, kind)
    header = _user_header(kind, raw_content, turn.human.get("created_at"), mode)

    if verbosity == "final":
        ai_block = _render_assistant_block_final(turn, mode)
    elif verbosity == "text":
        ai_block = _render_assistant_block_text(turn, mode)
    elif verbosity == "full":
        ai_block = _render_assistant_block_full(turn, mode)
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


def render_dialog(
    turns: list[Turn],
    verbosity: str = "final",
    timestamps: str = "local",
) -> str:
    """Render a list of Turns to a plain-text dialog at the requested verbosity.

    ``timestamps`` selects the zone of the per-message stamp in each block
    header: ``local`` (host time, default) or ``utc``. Either way the zone is
    printed, so a rendering can never be misread as the wrong clock.
    """
    if verbosity not in ("final", "text", "full"):
        raise ValueError(f"Unknown verbosity: {verbosity!r}")
    if timestamps not in ("local", "utc"):
        raise ValueError(f"Unknown timestamps mode: {timestamps!r}")
    mode: TimestampMode = "utc" if timestamps == "utc" else "local"

    blocks: list[str] = []
    for turn in turns:
        # A leading SystemMessage block (base prompt) has no user line.
        # Render it as a standalone System block at text/full; skip at final.
        if turn.human.get("type") == _SYSTEM_PREAMBLE_TYPE:
            rendered = _render_system_preamble(turn, verbosity, mode)
            if rendered is not None:
                blocks.append(rendered)
            continue
        raw = turn.human.get("content", "")
        if not isinstance(raw, str):
            raw = str(raw)
        kind = classify_human(raw)
        # Summarization turns are labelled, not filtered (see
        # DESIGN_SUMMARIZATION_VISIBLE.md).
        blocks.append(_render_turn(turn, kind, verbosity, mode))

    if not blocks:
        return NO_HISTORY_TEXT
    return "\n\n".join(blocks)
