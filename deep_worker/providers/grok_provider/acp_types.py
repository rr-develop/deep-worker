"""ACP (Agent Client Protocol) event taxonomy for the grok provider.

The xAI Grok CLI, when launched as ``grok agent stdio``, speaks the
Agent Client Protocol v1 (https://agentclientprotocol.com/) — a JSON-RPC
2.0 dialogue over stdio. This module is the single place that names the
notification/update kinds we care about and the tiny pure classifiers
used to route them. No I/O, no asyncio — just data + functions, so it is
trivially unit-testable and importable from anywhere in the provider.

See docs/DESIGN_GROK_PROVIDER.md §9.4 (notifications) and §11A
(built-in tool passthrough).
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# session/update — the "kind" lives in update["sessionUpdate"].
# ---------------------------------------------------------------------------
# Kinds we explicitly act on in ainvoke / the reader. Anything not in here
# is logged-and-ignored (forward-compatible with new grok update kinds).
HANDLED_SESSION_UPDATE_KINDS: frozenset[str] = frozenset(
    {
        "user_message_chunk",  # echo of our own prompt — ignored
        "agent_thought_chunk",  # reasoning — optionally stored
        "agent_message_chunk",  # assistant text — accumulated into final_text
        "tool_call",  # a tool invocation begins
        "tool_call_update",  # streaming/terminal status for a tool_call
        "plan",  # grok's plan — surfaced as update_plan tool_call
        "usage_update",  # context gauge (used/size/cost) — token-stats snapshot
    }
)


def session_update_kind(update: dict[str, Any]) -> str | None:
    """Return ``update["sessionUpdate"]`` or ``None`` if absent."""
    if not isinstance(update, dict):
        return None
    kind = update.get("sessionUpdate")
    return kind if isinstance(kind, str) else None


# ---------------------------------------------------------------------------
# Vendor (_x.ai/*) notifications — pure noise for our purposes.
# ---------------------------------------------------------------------------
_VENDOR_PREFIX = "_x.ai/"


def is_ignored_vendor_notification(method: str | None) -> bool:
    """True for grok's proprietary ``_x.ai/*`` notifications.

    These (mcp_initialized, git_head_changed, queue/changed,
    settings/update, sessions/changed, models/update,
    session/prompt_complete, announcements/update, ...) duplicate state
    we already track or are irrelevant to LangGraph history, so the
    reader drops them. Standard ACP methods (``session/update``,
    ``session/request_permission``, ``fs/*``, ``terminal/*``) are NOT
    vendor notifications and return False.
    """
    return bool(method) and method.startswith(_VENDOR_PREFIX)


# ---------------------------------------------------------------------------
# Built-in tool taxonomy. Grok runs these LOCALLY inside its own process
# (Q1 live-check 2026-06-17: cannot be disabled, no permission prompt).
# We record them into LangGraph history via builtin_tool_passthrough.
# CallMcpTool is DELIBERATELY excluded — it is grok's dispatcher for our
# proxy MCP tools and routes to channel A, never to passthrough.
# ---------------------------------------------------------------------------
# Live-verified on grok 0.2.54: GenerateImage. Other names are pre-registered
# candidates from the Imagine skill / design doc so ToolNode can route them
# without a name→GrokBuiltin mismatch when Grok CLI adds or aliases them.
# Anything still unknown routes via GrokBuiltin (PR-1). None are published to
# proxy MCP — no channel-A conflict.
IMAGINE_GROK_BUILTINS: frozenset[str] = frozenset(
    {
        "GenerateImage",
        "image_gen",
        "image_edit",
        "EditImage",
        "image_to_video",
        "reference_to_video",
    }
)

KNOWN_GROK_BUILTINS: frozenset[str] = frozenset(
    {
        "Shell",
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "Task",
        "Plan",
    }
) | IMAGINE_GROK_BUILTINS


def is_proxy_mcp_dispatch(update: dict[str, Any], proxy_server_name: str) -> bool:
    """True iff this ``tool_call`` update is grok dispatching one of OUR
    proxy MCP tools (channel A), not a built-in.

    Grok wraps an MCP tool call in a ``CallMcpTool`` built-in whose
    ``rawInput.server`` names the MCP server. When that server is our
    in-agent proxy, the inner tool is a deep-worker tool already parked on
    a Future by the proxy MCP handler — so the passthrough layer must NOT
    treat it as a local built-in.
    """
    if not isinstance(update, dict):
        return False
    if update.get("title") != "CallMcpTool":
        return False
    raw_input = update.get("rawInput")
    if not isinstance(raw_input, dict):
        return False
    return raw_input.get("server") == proxy_server_name


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
