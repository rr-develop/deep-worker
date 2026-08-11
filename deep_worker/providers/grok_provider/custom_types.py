"""Shared TypedDicts and dataclasses for the grok provider.

Mirrors ``codex_provider/custom_types.py`` in spirit, with grok-specific
names. Nothing here ever escapes the provider package — it's internal
wiring. See docs/DESIGN_GROK_PROVIDER.md §6, §11A, §12.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, TypedDict

from typing_extensions import NotRequired


class ToolMetadata(TypedDict):
    """LangChain-tool metadata extracted by ``bind_tools`` and forwarded to
    the in-process proxy MCP server.

    - ``name``         — tool name grok uses when calling it via CallMcpTool.
    - ``description``  — text shown to the model.
    - ``input_schema`` — JSON Schema for the tool arguments.
    - ``long_running`` — TRI-STATE wait policy (HARD POLICY, identical
      across providers). Unset/``None`` (DEFAULT) and ``True`` → the
      job-handle mechanism: after ``long_running_wait_seconds`` the handler
      parks the still-running Future as a job and returns a NON-error
      ``in_progress`` notice (``job_id`` + ``wait_for_job`` / ``kill_job``);
      work is NOT cancelled, the tool NEVER hangs forever. ``False`` → an
      EXPLICIT opt-out, allowed ONLY together with an explicit ``timeout``;
      opting out without a timeout is FORBIDDEN (raises at construction). On
      its timeout the handler returns a NON-error "timed out, still running"
      notice — never ``is_error``.
    - ``timeout``      — REQUIRED iff ``long_running`` is explicitly
      ``False``. Seconds to wait before the timeout notice.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    long_running: NotRequired[bool]
    timeout: NotRequired[float]


class MCPResult(TypedDict, total=False):
    """Result returned from an MCP ``tools/call`` handler. ``content`` is a
    list of MCP content blocks (typically ``{"type":"text","text":"..."}``);
    ``is_error`` flips error semantics for the model."""

    content: list[dict[str, Any]]
    is_error: bool


class JobHandle(TypedDict):
    """An in-flight tool call parked as a background job after its wait
    elapsed. The Future is NOT cancelled — the real work continues on the
    LangGraph ToolNode side and is resolved later. The agent reaches it via
    the canonical ``job_N`` id from ``ParkedJobRegistry`` (NOT the
    originating ``call_id``). See docs/DESIGN_FIX_SANDBOX_EXECUTE_JOB_ID.md
    (a related internal incident analysis).

    ``tool_call_id`` retains the originating ``call_id`` so ``resolve_call``
    (entered by the CLI with that id) can complete the parked Future via
    dual-key lookup in the shared registry.
    """

    future: "asyncio.Future[MCPResult]"
    tool_name: str
    args: dict[str, Any]
    tool_call_id: NotRequired[str]


class PendingCall(TypedDict):
    """Public-shape representation of a pending tool call — what we surface to
    the agent layer as LangChain ``tool_calls``."""

    id: str
    name: str
    args: dict[str, Any]


class PendingCallInternal(TypedDict):
    """Server-side state for one proxy-MCP tool call awaiting LangGraph
    resolution. The ``future`` is created by the MCP handler and resolved by
    ``GrokChatModel.ainvoke`` when the matching ToolMessage arrives."""

    future: asyncio.Future[MCPResult]
    name: str
    args: dict[str, Any]


# ---------------------------------------------------------------------------
# Per-agent_thread_id state. ``grok_session_id`` is the ACP session UUID
# returned by ``session/new`` (grok's notion of a conversation thread). We
# rename to avoid clashing with deep-worker's own ``thread_id`` (the agent
# name, NEVER mutated by the provider). See DESIGN §6, §12.
# ---------------------------------------------------------------------------
@dataclass
class ThreadState:
    """Per-``agent_thread_id`` state kept in the process-global singleton."""

    # ACP session UUID from ``session/new``. ``None`` means "next ainvoke
    # starts a fresh session". Reset to None on LangGraph summarization (L3).
    grok_session_id: str | None = None

    # Has ``grok_session_id`` been established in THE CURRENT grok subprocess?
    # A session id loaded from disk (a prior process) is NOT live here: grok
    # keeps sessions in-memory, so prompting a persisted id without first
    # ``session/load``-ing it is silently dropped → empty turn (the bug that
    # broke every restarted agent's first message, 2026-06-18). ``session/new``
    # sets this True directly; a persisted id starts False and triggers a
    # ``session/load`` on first use. Process-local (never persisted).
    session_live_in_process: bool = False

    # Number of LangGraph messages already mirrored into the grok session.
    # ``messages[sent_count:]`` is the delta to send next turn.
    sent_count: int = 0

    # Chars of the FIRST SystemMessage already delivered to grok CLI (via the
    # initial ``<system>`` block and/or prior ``System:>`` lines). Middleware
    # appends to that message in-place; only the suffix is forwarded later.
    forwarded_first_system_chars: int = 0

    # ids (or synthetic keys) of additional runtime SystemMessages already
    # forwarded as ``System:>`` lines (middleware may inject extra messages).
    forwarded_extra_system_keys: set[str] = field(default_factory=set, repr=False)

    # The in-flight ``session/prompt`` response Future when we are mid-turn
    # (between an ainvoke that surfaced tool_calls and the next ainvoke that
    # resolves them). ``None`` when no turn is in progress. Used by the
    # mid-turn-vs-new-turn branch in ainvoke (DESIGN §10 step 9).
    pending_session_prompt_future: "asyncio.Future[Any] | None" = field(
        default=None, repr=False
    )

    # The notification queue for the live turn. The reader pushes
    # ``session/update`` events here; ``_consume_turn`` drains it. Re-used
    # across a mid-turn re-entry WITHOUT creating a new subscription.
    current_notif_queue: "asyncio.Queue[Any] | None" = field(
        default=None, repr=False
    )

    # Set when a text-only turn called session/load with mcp_servers=[] so
    # grok cannot invoke stale proxy tools; cleared on the next tool turn.
    mcp_detached_for_text_only: bool = False

    # True for the duration of one ``ainvoke`` tagged ``_grok_text_only_turn``.
    # Used to drop spurious grok built-in / MCP tool_calls that would hang
    # LangGraph when middleware stripped all tools (inter-agent reply_back=False).
    text_only_turn_active: bool = False

    # ACP toolCallId -> proxy MCP handler call_id. Grok surfaces CallMcpTool
    # (or a bound tool by plain name) with its own toolCallId, but the proxy
    # parks Futures under a synthetic id — resolve_call must use this map.
    proxy_call_id_by_acp_id: dict[str, str] = field(default_factory=dict, repr=False)

    # toolCallId -> Future for grok built-in tool passthrough (channel B).
    # Lives on the per-thread state (NOT the shared client) so two agents
    # sharing one grok CLI process never collide on a toolCallId. The reader
    # resolves these on ``tool_call_update`` status in {completed, failed,
    # cancelled}. See DESIGN §11A.4/§11A.5.
    builtin_passthrough_futures: dict[str, "asyncio.Future[dict]"] = field(
        default_factory=dict, repr=False
    )

    # ── Mid-turn pump (DESIGN §10.2; fix #2 2026-06-18) ──────────────────
    # When ``_consume_turn`` returns ``AIMessage(tool_calls=[...])`` for a
    # channel-B built-in, LangGraph's ToolNode invokes the passthrough body
    # which parks on ``builtin_passthrough_futures[tcid]``. NOTHING else
    # would drain ``current_notif_queue`` until the next ainvoke — so the
    # reader would push ``tool_call_update status=completed`` into the queue
    # but the Future would never resolve and ``_arun`` would hang forever
    # (test_grok 2026-06-18 incident).
    #
    # The pump task is a single asyncio.Task spawned at the mid-turn pause
    # that keeps draining the same ``current_notif_queue``: it resolves
    # built-in Futures, accumulates ``agent_message_chunk`` text, and queues
    # any FRESH ``tool_call`` events grok emits during the pump window for
    # the NEXT ainvoke to consume. It exits when (a) the next ainvoke takes
    # over by calling ``cancel()`` and re-entering ``_consume_turn`` on the
    # same future/queue, OR (b) the in-flight ``pending_session_prompt_future``
    # resolves (grok finished the turn while ToolNode was running, e.g. an
    # immediate ``end_turn`` after the built-in completed).
    pump_task: "asyncio.Task[None] | None" = field(default=None, repr=False)
    # Text accumulated by the pump between the mid-turn pause and the next
    # ainvoke. The next ainvoke prepends it to its own ``final_text_parts``.
    queued_text: list[str] = field(default_factory=list, repr=False)
    # Fresh tool_calls grok emitted while the pump was running. The next
    # ainvoke surfaces them as part of its returned AIMessage.tool_calls.
    queued_tool_calls: list[dict] = field(default_factory=list, repr=False)
    # Set by the pump when it observed the response future resolving with a
    # terminal stopReason. The next ainvoke returns the final AIMessage
    # without re-entering _consume_turn.
    queued_response: "dict | None" = field(default=None, repr=False)

    # ── Token-stats context gauge (DESIGN_AGENT_TOKEN_STATS.md §5.2) ────────
    # Latest ``usage_update`` snapshot for the CURRENT turn:
    # {"used": int, "size": int} — context gauge of the grok session.
    # Updated by ``_apply_update`` on every usage_update (from either
    # ``_consume_turn`` or the mid-turn pump); emitted once at the turn
    # boundary (``_finalize_response``) as usage_semantics="context_snapshot"
    # and reset to None. Never persisted.
    usage_snapshot: "dict[str, int] | None" = field(default=None, repr=False)
