"""Shared TypedDicts and dataclasses for the Codex provider.

Mirrors ``claude_code_provider/custom_types.py`` in spirit but with
Codex-specific names. Nothing in here ever escapes the provider
package — it's all internal wiring.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, TypedDict

from typing_extensions import NotRequired


class ToolMetadata(TypedDict):
    """LangChain-tool metadata extracted by ``bind_tools`` and forwarded
    to the in-process proxy MCP server.

    - ``name``         — tool name used by Codex when calling it.
    - ``description``  — text shown to the model.
    - ``input_schema`` — JSON Schema for the tool arguments.
    - ``long_running`` — TRI-STATE wait policy (HARD POLICY, 2026-06-16,
      identical across providers). Unset/``None`` (DEFAULT) and ``True``
      → the job-handle mechanism: after ``long_running_wait_seconds`` the
      handler parks the still-running Future as a job and returns a
      NON-error ``in_progress`` notice (``job_id`` + ``wait_for_job`` /
      ``kill_job``); work is NOT cancelled, the tool NEVER hangs forever.
      ``False`` → an EXPLICIT opt-out, allowed ONLY together with an
      explicit ``timeout``; opting out without a timeout is FORBIDDEN
      (raises at construction). On its timeout the handler returns a
      NON-error "timed out, still running" notice — never ``is_error``.
    - ``timeout``      — REQUIRED iff ``long_running`` is explicitly
      ``False``. Seconds to wait before the timeout notice. See
      ``docs/POLICIES.md`` §7.1.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    long_running: NotRequired[bool]
    timeout: NotRequired[float]


class MCPResult(TypedDict, total=False):
    """Result returned from an MCP ``tools/call`` handler.

    The handler returns this dict; the MCP server marshals it to wire
    format. ``content`` is a list of MCP content blocks (typically
    ``{"type": "text", "text": "..."}``); ``is_error`` flips error
    semantics for the model.
    """

    content: list[dict[str, Any]]
    is_error: bool


class JobHandle(TypedDict):
    """An in-flight tool call parked as a background job after its wait
    elapsed. The Future is NOT cancelled — the real work continues on the
    LangGraph ToolNode side and is later resolved. The agent was handed an
    intermediate handle and can ``wait_for_job`` / ``kill_job`` it.
    """

    future: "asyncio.Future[MCPResult]"
    tool_name: str
    args: dict[str, Any]


class PendingCall(TypedDict):
    """Public-shape representation of a pending tool call — what we
    return to the agent layer as LangChain ``tool_calls``."""

    id: str
    name: str
    args: dict[str, Any]


class PendingCallInternal(TypedDict):
    """Server-side state for one tool call awaiting LangGraph resolution.

    The ``future`` is created by the MCP handler and resolved by
    ``CodexChatModel.ainvoke`` when the matching ToolMessage arrives.
    """

    future: asyncio.Future[MCPResult]
    name: str
    args: dict[str, Any]


# ---------------------------------------------------------------------------
# Per-agent_thread_id state. ``codex_session_uuid`` is what Codex CLI calls
# its session id (the "thread_id" in its wire protocol). We rename here to
# avoid clashing with deep-worker's own ``thread_id`` (which is the agent
# name and is NEVER mutated by the provider). See DESIGN_CODEX_PROVIDER.md §3.
# ---------------------------------------------------------------------------
@dataclass
class ThreadState:
    """Per-``agent_thread_id`` state kept in the process-global singleton."""

    # UUID Codex returns in its first ``thread.started`` event for this
    # agent_thread_id. ``None`` means "next ainvoke will start a fresh
    # Codex session". Reset to None on LangGraph summarization.
    codex_session_uuid: str | None = None
    # Number of LangGraph messages we've already mirrored into the Codex
    # session. ``messages[sent_count:]`` is the delta to send next turn.
    sent_count: int = 0
    # Live in-flight ExecHandle when we are mid-Codex-turn (i.e. between
    # an ainvoke that returned tool_calls and the next ainvoke that
    # resolves them). ``None`` when no turn is in progress.
    current_exec: Any | None = field(default=None, repr=False)
