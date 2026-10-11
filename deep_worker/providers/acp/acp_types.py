"""ACP (Agent Client Protocol) event taxonomy for the shared ACP layer.

Any ACP-speaking CLI (Claude Code, Codex, Grok, Gemini CLI, ...) launched in
stdio mode emits JSON-RPC 2.0 over stdin/stdout. This module names the
notification/update kinds the shared layer acts on and the tiny pure
classifiers used to route them. No I/O, no asyncio — just data + functions,
so it is trivially unit-testable and importable from anywhere.

Vendor-specific noise is NOT hardcoded here: each provider profile supplies
its own vendor prefix (``_x.ai/`` for grok, ``_claude/`` for Claude Code,
...). See ``profile.py``.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# session/update — the "kind" lives in update["sessionUpdate"].
# ---------------------------------------------------------------------------
# Kinds we explicitly act on in ainvoke / the reader. Anything not in here
# is logged-and-ignored (forward-compatible with new ACP update kinds).
HANDLED_SESSION_UPDATE_KINDS: frozenset[str] = frozenset(
    {
        "user_message_chunk",  # echo of our own prompt — ignored
        "agent_thought_chunk",  # reasoning — optionally stored
        "agent_message_chunk",  # assistant text — accumulated into final_text
        "tool_call",  # a tool invocation begins
        "tool_call_update",  # streaming/terminal status for a tool_call
        "plan",  # agent plan — surfaced as update_plan tool_call
        "usage_update",  # context gauge (used/size/cost) — token-stats snapshot
    }
)


def session_update_kind(update: dict[str, Any]) -> str | None:
    """Return ``update["sessionUpdate"]`` or ``None`` if absent."""
    if not isinstance(update, dict):
        return None
    kind = update.get("sessionUpdate")
    return kind if isinstance(kind, str) else None


def is_ignored_vendor_notification(
    method: str | None, vendor_prefix: str
) -> bool:
    """True for a provider's proprietary ``<prefix>*`` notifications.

    Providers emit extra notifications (mcp_initialized, git_head_changed,
    settings/update, ...) that duplicate state we already track or are
    irrelevant to LangGraph history; the reader drops them. Standard ACP
    methods (``session/update``, ``session/request_permission``, ``fs/*``,
    ``terminal/*``) are NOT vendor notifications and return False.
    """
    return bool(method) and vendor_prefix and method.startswith(vendor_prefix)


# ---------------------------------------------------------------------------
# stopReason — appears in the session/prompt RESULT, ends the turn.
# A tool call does NOT carry a stopReason (the prompt stays in-flight).
# ---------------------------------------------------------------------------
TERMINAL_STOP_REASONS: frozenset[str] = frozenset(
    {
        "end_turn",
        "max_tokens",
        "refusal",
        "cancelled",
        "max_turn_requests",
    }
)


def is_terminal_stop_reason(reason: str | None) -> bool:
    """True iff ``reason`` is a documented ACP stopReason.

    Defensive: an unknown value (or ``None``) returns False so the turn is
    never silently ended on a value we don't understand.
    """
    return reason in TERMINAL_STOP_REASONS


# ---------------------------------------------------------------------------
# Tool name extraction — where the REAL tool name lives in a tool_call update.
# ---------------------------------------------------------------------------
def tool_name_from_update(
    update: dict[str, Any], style: str = "from_title"
) -> str | None:
    """Return the CLI's real tool name for a ``tool_call`` update.

    Providers differ:

    * ``"from_title"`` (grok, default): ``title`` IS the tool name
      (``Shell``, ``Read``, ``CallMcpTool``, ...).
    * ``"from_claude_code_meta"`` (claude-agent-acp): the REAL name is in
      ``_meta.claudeCode.toolName``; ``title`` is human-readable prose
      (e.g. ``"Read /path (1-30)"``).

    Returns ``None`` when the update carries no usable name.
    """
    if not isinstance(update, dict):
        return None
    if style == "from_claude_code_meta":
        meta = update.get("_meta")
        if isinstance(meta, dict):
            claude = meta.get("claudeCode")
            if isinstance(claude, dict):
                name = claude.get("toolName")
                if isinstance(name, str) and name:
                    return name
        return None
    title = update.get("title")
    return title if isinstance(title, str) and title else None


def mcp_prefixed_tool_name(name: str, proxy_server_name: str) -> str | None:
    """If ``name`` is ``mcp__<server>__<tool>`` for OUR server, return the
    inner tool name (``<tool>``); else ``None``.

    claude-agent-acp surfaces MCP tools with the Claude Code naming scheme
    ``mcp__<server>__<tool>`` (live-verified 0.68.0) and the server is
    encoded in the name, not in ``rawInput``.
    """
    prefix = f"mcp__{proxy_server_name}__"
    if name.startswith(prefix):
        inner = name[len(prefix):]
        return inner or None
    return None


def is_proxy_mcp_dispatch(
    update: dict[str, Any],
    proxy_server_name: str,
    mcp_dispatch_style: str,
) -> bool:
    """True iff this ``tool_call`` update is the CLI dispatching one of OUR
    proxy MCP tools (channel A), not a built-in.

    Two dispatch mechanisms exist among ACP CLIs:

    * ``"dispatch_builtin"`` (grok): the CLI wraps an MCP tool call in a
      dispatch built-in. Two wrapper forms are live
      (see a related internal incident analysis, not included here):
      grok ≤1.0.4 ``CallMcpTool{server, toolName, arguments}`` (the server
      is a separate field) and grok 1.0.5+
      ``use_tool{tool_name="<server>__<tool>", tool_input={…}}`` (the
      server is MANGLED into the tool name). Recognition is deliberately
      title-agnostic — the contract is carried by the ``rawInput`` shape,
      so the next rename of the wrapper built-in alone does not break it.
    * ``"mcp_name_prefix"`` (claude-agent-acp): MCP tools surface as
      ordinary ``tool_call`` updates whose name is
      ``mcp__<server>__<tool>``; the server is encoded in the name.

    The style lives in ``AcpProfile.mcp_dispatch_style``.
    """
    if not isinstance(update, dict):
        return False
    if mcp_dispatch_style == "mcp_name_prefix":
        name = tool_name_from_update(update, "from_claude_code_meta")
        if name is None:
            name = update.get("title")
        if isinstance(name, str):
            return mcp_prefixed_tool_name(name, proxy_server_name) is not None
        return False
    # dispatch_builtin (default)
    return proxy_dispatch_inner(update, proxy_server_name) is not None


def proxy_dispatch_inner(
    update: dict[str, Any],
    proxy_server_name: str,
) -> tuple[str, dict[str, Any]] | None:
    """Extract ``(inner_tool_name, args)`` from a ``dispatch_builtin``
    wrapper update targeting OUR proxy server, else ``None``.

    The single seam ``_proxy_tool_from_update`` relies on; recognises both
    live wrapper forms:

    * ≤1.0.4 — ``rawInput = {server, toolName, arguments}``: match on
      ``server == proxy_server_name``, inner name in ``toolName``, args in
      ``arguments``;
    * 1.0.5+ — ``rawInput = {tool_name, tool_input}`` with
      ``tool_name == "<server>__<tool>"``: match on the full
      ``"<proxy_server_name>__"`` prefix (a server that merely starts with
      ours must NOT match), inner name is the suffix, args in
      ``tool_input``.

    Missing/odd-typed args default to ``{}``; an empty inner name returns
    ``None`` (nothing dispatchable).
    """
    if not isinstance(update, dict):
        return None
    raw_input = update.get("rawInput")
    if not isinstance(raw_input, dict):
        return None
    # ≤1.0.4 form: explicit server field.
    if raw_input.get("server") == proxy_server_name:
        inner = raw_input.get("toolName")
        if not isinstance(inner, str) or not inner:
            return None
        args = raw_input.get("arguments")
        return inner, args if isinstance(args, dict) else {}
    # 1.0.5+ form: server mangled into tool_name.
    mangled = raw_input.get("tool_name")
    prefix = f"{proxy_server_name}__"
    if isinstance(mangled, str) and mangled.startswith(prefix):
        inner = mangled[len(prefix):]
        if not inner:
            return None
        args = raw_input.get("tool_input")
        return inner, args if isinstance(args, dict) else {}
    return None
