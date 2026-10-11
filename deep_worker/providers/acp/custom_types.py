"""Shared TypedDicts and dataclasses for the ACP provider layer.

Mirrors ``codex_provider/custom_types.py`` / ``grok_provider/custom_types.py``
in spirit, with provider-agnostic names. Nothing here ever escapes the
provider package — it's internal wiring.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, TypedDict

from typing_extensions import NotRequired


class ToolMetadata(TypedDict):
    """LangChain-tool metadata extracted by ``bind_tools`` and forwarded to
    the in-process proxy MCP server.

    - ``name``         — tool name the CLI uses when calling it via the MCP
      dispatch built-in.
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
    originating ``call_id``).

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
    ``AcpChatModel.ainvoke`` when the matching ToolMessage arrives."""

    future: asyncio.Future[MCPResult]
    name: str
    args: dict[str, Any]


# Bounded-state caps (DESIGN §4.5). Every collection below is capacity
# bounded; overflow behaves per the documented policy (reject-new for the
# call-id map, fail-loud backpressure for sinks, bounded partial buffers).
MAX_SINKS_PER_SESSION = 1  # one active turn per session (one-in-flight)
CALL_ID_MAP_MAX_ENTRIES = 512  # proxy_call_id_by_acp_id / future maps
QUEUED_TEXT_MAX_CHUNKS = 10_000  # partial-buffer chunk cap
QUEUED_TEXT_MAX_BYTES = 16 * 1024 * 1024  # partial-buffer byte cap (~_READ_LIMIT x2)

# Notification sink / current_notif_queue capacity. The queue is created with
# this maxsize (chat_model) and the reader checks ``sink.full()`` against the
# SAME bound (acp_client). Overflow → fail-loud: the active turn's response
# future is failed with AcpSinkOverflowError instead of hanging or dropping.
NOTIF_QUEUE_MAX_ITEMS = 1000

# Fresh tool_calls accumulated by the mid-turn pump (queued_tool_calls) and
# the per-turn accumulator. Overflow → fail-loud (same mechanism).
QUEUED_TOOL_CALLS_MAX = 256

# Held-back refine-pending toolCallIds (claude-agent-acp streams args in a
# follow-up tool_call_update). Overflow → reject-new (fail-closed: the call is
# never surfaced to LangGraph, so the tool simply does not run).
REFINE_PENDING_MAX = 256

# Dedup-key cap for forwarded runtime SystemMessage keys. Overflow →
# reject-new: the SystemMessage text still rides the prompt, but the key stops
# being deduplicated (worst case: a duplicate System:> line on the next turn).
FORWARDED_EXTRA_KEYS_MAX = 512

# IDLE bound on an in-flight ``session/prompt``.
#
# ``_consume_turn`` used to wait on {notification, response} with NO timeout,
# so only a CRASH (EOF) was handled. "Alive but silent" — a wedged CLI, or a
# reader that died before the guard existed — hung the agent forever with
# nothing logged.
#
# This is deliberately an IDLE timer, not a total turn budget: it resets on
# EVERY notification, so a legitimately long turn (big edit, long tool run,
# extended thinking) never trips it as long as the CLI keeps streaming. Only
# TOTAL silence for this long is treated as a wedged channel — the turn is
# cancelled and failed loudly rather than left hanging.
#
# Generous by design: a false positive kills a healthy turn, which is worse
# than waiting a few more minutes. Operators can override per-deployment.
PROMPT_IDLE_TIMEOUT_ENV = "DEEP_WORKER_ACP_PROMPT_IDLE_TIMEOUT"
PROMPT_IDLE_TIMEOUT_DEFAULT = 900.0  # 15 minutes of complete silence

# Client JSON-RPC request map (_pending_requests). Overflow → reject-new
# (raise before writing). Normally tiny (one in-flight reply at a time);
# this cap only guards against a wedged CLI that stops replying.
MAX_PENDING_REQUESTS = 4096

# toolCallId -> toolName map used to resolve a permission ask to the CLI's
# own tool name. Entries are normally evicted the moment the call reaches a
# terminal status, and the whole map is cleared on teardown. This cap is the
# backstop for the one path neither covers: a CLI that asks for permission
# and then never reports the call as completed/failed (wedged or killed
# mid-turn). Oldest-first eviction — the newest ids are the ones an
# in-flight permission ask can still reference.
MAX_TOOL_NAME_ENTRIES = 4096


@dataclass
class ThreadState:
    """Per-``agent_thread_id`` state kept in the process-global singleton.

    ``session_id`` is the ACP session UUID returned by ``session/new`` (the
    CLI's notion of a conversation thread). We rename to avoid clashing with
    deep-worker's own ``thread_id`` (the agent name, NEVER mutated by the
    provider).
    """

    # ACP session UUID from ``session/new``. ``None`` means "next ainvoke
    # starts a fresh session". Reset to None on LangGraph summarization (L3).
    session_id: str | None = None

    # Has ``session_id`` been established in THE CURRENT CLI subprocess?
    # A session id loaded from disk (a prior process) is NOT live here: most
    # ACP CLIs keep sessions in-memory, so prompting a persisted id without
    # first ``session/load``-ing it is silently dropped → empty turn.
    # ``session/new`` sets this True directly; a persisted id starts False
    # and triggers a ``session/load`` on first use (only when the profile
    # reports session-load support). Process-local (never persisted).
    session_live_in_process: bool = False

    # Number of LangGraph messages already mirrored into the CLI session.
    # ``messages[sent_count:]`` is the delta to send next turn. MONOTONIC:
    # only ever incremented; reset to 0 together with session_id (fresh
    # session or summarization), never on its own.
    sent_count: int = 0

    # Chars of the FIRST SystemMessage already delivered to the CLI (via the
    # initial ``<system>`` block and/or prior ``System:>`` lines). Middleware
    # appends to that message in-place; only the suffix is forwarded later.
    forwarded_first_system_chars: int = 0

    # ids (or synthetic keys) of additional runtime SystemMessages already
    # forwarded as ``System:>`` lines (middleware may inject extra messages).
    forwarded_extra_system_keys: set[str] = field(default_factory=set, repr=False)

    # Freshest context reading from the provider's own telemetry, updated
    # after EVERY model call by the live gauge.
    # Beats the prompt reply's ``_meta`` when stamping: that
    # one is a turn-boundary value and is already stale by then.
    live_context_used: int | None = None
    # The live gauge for the turn in flight. Process-local, never persisted.
    live_context_watcher: Any = None

    # Has the per-turn deep-worker tool contract been delivered in THE
    # CURRENT PROCESS? The proxy MCP server name embeds the PID, so a
    # contract delivered by a previous process names DEAD tools; a session
    # resumed via ``session/load`` must receive a fresh contract on its
    # first prompt here (see a related internal incident analysis, not included here).
    # Process-local (never persisted).
    tool_contract_sent_in_process: bool = False

    # The in-flight ``session/prompt`` response Future when we are mid-turn
    # (between an ainvoke that surfaced tool_calls and the next ainvoke that
    # resolves them). ``None`` when no turn is in progress.
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
    # the CLI cannot invoke stale proxy tools; cleared on the next tool turn.
    mcp_detached_for_text_only: bool = False

    # True for the duration of one ``ainvoke`` tagged ``_text_only_turn``.
    # Used to drop spurious built-in / MCP tool_calls that would hang
    # LangGraph when middleware stripped all tools (inter-agent reply_back=False).
    text_only_turn_active: bool = False

    # ACP toolCallId -> proxy MCP handler call_id. The CLI surfaces the
    # dispatch built-in (or a bound tool by plain name) with its own
    # toolCallId, but the proxy parks Futures under a synthetic id —
    # resolve_call must use this map. BOUNDED (CALL_ID_MAP_MAX_ENTRIES):
    # overflow rejects NEW entries (reject-new, DESIGN §4.5).
    proxy_call_id_by_acp_id: dict[str, str] = field(default_factory=dict, repr=False)

    # toolCallId -> Future for CLI built-in tool passthrough (channel B).
    # Lives on the per-thread state (NOT the shared client) so two agents
    # sharing one CLI process never collide on a toolCallId. The reader
    # resolves these on ``tool_call_update`` status in {completed, failed,
    # cancelled}. BOUNDED like the call-id map.
    builtin_passthrough_futures: dict[str, "asyncio.Future[dict]"] = field(
        default_factory=dict, repr=False
    )

    # toolCallId -> ALREADY-KNOWN result payload, for a synthetic tool_call
    # whose outcome is a completed fact at emission time and for which the CLI
    # sends no ``tool_call_update`` (``update_plan``: a plan IS the result).
    # A plain dict rather than a resolved Future because ``_apply_update`` is
    # sync by contract and legitimately runs with NO event loop, and because a
    # Future bound to one loop cannot be awaited on another. The passthrough
    # body pops from here BEFORE parking, so such a call never blocks.
    # BOUNDED like the futures map.
    builtin_passthrough_results: dict[str, dict] = field(
        default_factory=dict, repr=False
    )

    # ── Mid-turn pump ───────────────────────────────────────────────────
    # When ``_consume_turn`` returns ``AIMessage(tool_calls=[...])`` for a
    # channel-B built-in, LangGraph's ToolNode invokes the passthrough body
    # which parks on ``builtin_passthrough_futures[tcid]``. NOTHING else
    # would drain ``current_notif_queue`` until the next ainvoke — so the
    # reader would push ``tool_call_update status=completed`` into the queue
    # but the Future would never resolve and ``_arun`` would hang forever.
    #
    # The pump task is a single asyncio.Task spawned at the mid-turn pause
    # that keeps draining the same ``current_notif_queue``: it resolves
    # built-in Futures, accumulates ``agent_message_chunk`` text, and queues
    # any FRESH ``tool_call`` events the CLI emits during the pump window for
    # the NEXT ainvoke to consume. It exits when (a) the next ainvoke takes
    # over by calling ``cancel()`` and re-entering ``_consume_turn`` on the
    # same future/queue, OR (b) the in-flight ``pending_session_prompt_future``
    # resolves (the CLI finished the turn while ToolNode was running).
    pump_task: "asyncio.Task[None] | None" = field(default=None, repr=False)
    # Text accumulated by the pump between the mid-turn pause and the next
    # ainvoke. The next ainvoke prepends it to its own ``final_text_parts``.
    # BOUNDED (chunk + byte caps); overflow → fail-loud (DESIGN §4.5).
    queued_text: list[str] = field(default_factory=list, repr=False)
    # Fresh tool_calls the CLI emitted while the pump was running. The next
    # ainvoke surfaces them as part of its returned AIMessage.tool_calls.
    queued_tool_calls: list[dict] = field(default_factory=list, repr=False)
    # Set by the pump when it observed the response future resolving with a
    # terminal stopReason. The next ainvoke returns the final AIMessage
    # without re-entering _consume_turn.
    queued_response: "dict | None" = field(default=None, repr=False)

    # ── Refine-pending tool calls (claude-agent-acp streams args) ────────
    # claude-agent-acp surfaces ``tool_call`` with ``rawInput={}`` (pending)
    # and streams the real arguments in a follow-up ``tool_call_update``
    # WITHOUT a status (the "refine" event). While a tcid sits in this set we
    # hold back its tool_call from LangGraph; the refine event surfaces it.
    # Transient — cleared when the turn ends. Per-thread like the futures map.
    refine_pending_tool_calls: set[str] = field(default_factory=set, repr=False)

    # ── Token-stats context gauge ─────────────────────────────────────────
    # Latest ``usage_update`` snapshot for the CURRENT turn:
    # {"used": int, "size": int} — context gauge of the CLI session.
    # Updated by ``_apply_update`` on every usage_update; emitted once at the
    # turn boundary (``_finalize_response``) and reset to None. Never persisted.
    usage_snapshot: "dict[str, int] | None" = field(default=None, repr=False)

    # ── Observed-context stamp (DESIGN_CONTEXT_LIMIT_GUARANTEE)
    # ─────────────────────────────────────────────────────────
    # Id of the first state (non-System) message of the CURRENT request,
    # captured by ``_agenerate``. Every AIMessage returned for this request
    # is stamped with ``usage_snapshot.used`` + this id (validity key) so the
    # summarizer/guard can use the CLI's real context size instead of the
    # chars/4 heuristic — and can tell a stale stamp (post-summarization
    # tail) from a live one. Never persisted.
    context_first_id: str | None = field(default=None, repr=False)

    # Survives restart via acp_session_state.json (DESIGN_SUMMARIZATION_HOT_COLD §3).
    last_response_at: str | None = None
    last_context_used: int | None = None
    last_model_id: str | None = None
