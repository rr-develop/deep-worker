"""ProxyMCPServer - In-process MCP server that delegates tool execution to LangGraph.

⚠️ ARCHITECTURAL INVARIANT: ALL tool calls MUST go through LangGraph ToolNode.
═══════════════════════════════════════════════════════════════════════════════
ProxyMCPServer registers deep-worker tools as MCP tools for Claude Code CLI.
When Claude calls a tool, the handler blocks on asyncio.Future. This causes:

  1. ainvoke() detects pending_calls → returns AIMessage(tool_calls=[...])
  2. LangGraph routes to ToolNode → ToolNode executes the tool
  3. ToolMessage result → next ainvoke() → resolve_call() → Future unblocks
  4. MCP handler returns result to Claude CLI → Claude continues

This ensures tool calls are VISIBLE in LangGraph state (result.messages),
which is REQUIRED for:
  - dwcli TUI broadcasting tool calls to subscribers
  - Checkpointer saving tool call history
  - interrupt_on (HITL approval) before tool execution
  - TaskManager seeing tool activity

DO NOT replace with direct execution in MCP handler!
Direct execution (calling tool.ainvoke inside handler) makes tools INVISIBLE
to LangGraph — dwcli shows nothing, checkpointer saves nothing, HITL breaks.
This was the bug in commits a8cbdcd/01872d7 that broke dwcli.
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# ============================================================================
# CRITICAL: CLAUDE_SDK MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP/VENV!
# ============================================================================
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))

from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: E402
from claude_agent_sdk._mcp_context import (  # noqa: E402
    get_current_claude_tool_use_id,
)

from claude_code_provider.custom_types import (  # noqa: E402
    JobHandle,
    MCPResult,
    PendingCall,
    PendingCallInternal,
    ToolMetadata,
)


# ════════════════════════════════════════════════════════════════════════════
# POLICY — TIMEOUTS AND IDENTITY (INCIDENT 2026-06-14 + POLICIES.md §7)
# ════════════════════════════════════════════════════════════════════════════
# 1. ONE identity end-to-end. The MCP ``_meta.claudecode/toolUseId`` is the
#    single key for a logical tool call on EVERY layer: ``pending_calls``
#    here, ``ToolUseBlock.id`` in the message stream, ``tool_call_id`` in
#    ``ToolMessage``, ``resolve_call(id, ...)`` from chat_model. NEVER
#    synthesise a second id ("call_counter", "id(args)", …). Past attempts
#    to do so produced six fix-attempts in five days on this same site
#    (4887e0d/da860ff/eba8b39/5b0d819/555e8d0/f0eec37/8d0ae52) and the
#    "two near-identical copies of the same block" symptom in production.
#
#    ⚠️ "ONE identity" = ONE id PER CALL (the ``toolu_*`` tool_use_id). It
#    is NOT "one id per message block". A single assistant message block
#    (``message.id`` = ``msg_*``) carries MANY tool calls, each with its
#    own ``toolu_*`` (measured: up to 52; ``msg_*`` is NEVER equal to any
#    ``toolu_*``). ``ToolUseBlock.id`` above means the per-call
#    ``content[].id`` (= the ``toolu_*``), NOT ``msg_*``. Never key on a
#    message/block id, never assume "one block = one call". See
#    docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md.
#
# 2. NO TIMEOUT EVER BECOMES AN ERROR. A timeout MUST NOT report
#    ``is_error=true`` to the LLM. The previous 120-second cap returned
#    ``is_error=true, "Tool execution timeout after 120 seconds"`` — a lie
#    that made Claude retry and DUPLICATE the real side effect. On any
#    timeout the handler returns an INFORMATIONAL result (still-running +
#    job_id + actions) and the agent simply continues / retries.
#
# 3. EVERY TOOL IS BOUNDED — the job-handle mechanism is the DEFAULT for
#    ALL tools (HARD POLICY 2026-06-16). A handler NEVER does a bare,
#    unbounded ``await future``: after ``long_running_wait_seconds`` (120s
#    default, env-overridable) it parks the live Future as a job (work is
#    NOT cancelled — ``wait_for_job(job_id)`` picks it up) and returns a
#    non-error in_progress handle. A tool may opt OUT only EXPLICITLY
#    (``long_running=False``) and ONLY with its own ``timeout`` — an
#    opt-out without a timeout is rejected at construction, because a bare
#    unbounded wait is exactly what hung ``edit_file`` forever (mj1_init
#    2026-06-16). Provider-agnostic; no exceptions. A previous agent
#    claimed this was "implemented" while ``long_running`` was set on NO
#    tool, so the mechanism was dead and every tool hung — that claim was
#    false; this banner is the corrected, enforced contract.
#
# 4. ANY violation of (1)–(3) is rejected by reviewers per POLICIES.md §7.
# ════════════════════════════════════════════════════════════════════════════


# Default cap before the handler issues a job-handle. This applies to
# EVERY tool by default (the job-handle mechanism is the default — see the
# POLICY banner). It is NOT a deadline for the work — the Future keeps
# running, parked as a job. It is the operator-visible cap on how long the
# model blocks for a fast happy-path answer before being handed a job
# handle to continue. 120s ("~2 minutes") per project policy. Override via
# the ``DW_LONG_RUNNING_WAIT_SECONDS`` env var (wired in __init__).
class DispatchTimeout(Exception):
    """Stage A (control-channel ``tools/call``) did not register within the
    single bounded source of truth. Raised by ``wait_for_pending_call_by_id``
    so ``ainvoke`` surfaces a NON-error ``in_progress`` handle instead of
    hanging forever (the mj1_init / mj2_vac hole). See
    docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md (variant A)."""


def _default_wait_seconds() -> float:
    """THE single source of truth for the tool-wait bound — delegates to
    ``deepagents.middleware.bounded_tool_call.resolve_tool_wait_seconds`` so
    there is NO independent timer constant in this provider (variant A).
    The former private ``_DEFAULT_LONG_RUNNING_WAIT_SECONDS`` (120s) is gone."""
    from deepagents.middleware.bounded_tool_call import resolve_tool_wait_seconds

    return resolve_tool_wait_seconds()


# Claude Code emits this when the model calls a tool name the CLI cannot
# route (e.g. a bare/short name, or a tool that does not exist). Shape:
#   "<tool_use_error>Error: No such tool available: execute</tool_use_error>"
#   "Error: No such tool available: foo"
_NO_SUCH_TOOL_RE = re.compile(r"No such tool available:\s*([^\s<]+)")


def parse_no_such_tool(text: object) -> str | None:
    """If ``text`` is a Claude "No such tool available: <name>" rejection,
    return ``<name>``; otherwise ``None``.

    Used by chat_model to detect that a tool the model named does not
    exist on the CLI side, so any waiter we hold for it can be cancelled
    (``cancel_pending_waiter``) instead of parking forever. See
    docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md.
    """
    if not isinstance(text, str):
        return None
    m = _NO_SUCH_TOOL_RE.search(text)
    return m.group(1) if m else None


class ProxyMCPServer:
    """In-process MCP server that proxies tool calls to LangGraph via Futures.

    Flow:
    1. Claude calls tool → handler creates Future and BLOCKS
    2. ainvoke() detects pending calls → returns AIMessage(tool_calls)
    3. LangGraph executes tool via ToolNode
    4. Next ainvoke() receives ToolMessage → resolve_call() → Future unblocks
    5. Handler returns result to Claude CLI

    Attributes:
        tool_handlers: List of decorated MCP tool handlers
        mcp_server_config: MCP server configuration for Claude SDK
        pending_calls: Dict mapping Claude ``tool_use_id`` to pending call
            info with Future. The KEY is always the id Claude itself
            picked (see policy banner above).
    """

    def __init__(
        self,
        tools_metadata: list[ToolMetadata],
        working_dir: str | None = None,
        agent_name: str | None = None,
        *,
        long_running_wait_seconds: float | None = None,
    ) -> None:
        """Initialize ProxyMCPServer with tool metadata.

        Args:
            tools_metadata: Tool metadata dicts (name, description,
                input_schema, optional ``long_running`` / ``timeout`` —
                see ToolMetadata for the tri-state wait policy).
            working_dir: Agent working directory (preserved for future use).
            agent_name: Agent thread_id (used for subagent thread isolation).
            long_running_wait_seconds: Cap before the DEFAULT job-handle
                path issues a handle, for EVERY tool (not just
                ``long_running`` ones). None → resolve the default
                (120 s, env-overridable). NOT a deadline for the work —
                the Future is parked, not cancelled. See policy banner.

        ⚠ INVARIANT (POLICIES.md §7): NO ``timeout`` parameter here. The
        former ``timeout: float = 120.0`` was the phantom that lied to the
        LLM in incident 2026-06-14. DO NOT reintroduce it under any name
        (including "execution_timeout", "tool_timeout", "wait_for_result").
        """
        self.pending_calls: dict[str, PendingCallInternal] = {}
        self.tool_handlers: list[Any] = []
        self._tool_names = [m["name"] for m in tools_metadata]
        # Default (incl. env override) when the caller did not pin a value.
        if long_running_wait_seconds is None:
            long_running_wait_seconds = _default_wait_seconds()
        self._long_running_wait_seconds = long_running_wait_seconds
        # ── Per-tool wait policy (HARD POLICY, 2026-06-16) ──────────────
        # EVERY tool gets the job-handle mechanism BY DEFAULT. A tool may
        # opt OUT only EXPLICITLY (``long_running=False``), and an explicit
        # opt-out MUST carry its own ``timeout`` — opting out without a
        # timeout is FORBIDDEN (that is the eternal-hang bug). No tool ever
        # does a bare unbounded ``await future``. A timeout NEVER becomes an
        # agent error: the handler returns an informational result and the
        # agent continues. See docs/POLICIES.md §7 and
        # docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md.
        #
        # ``_wait_policy[name]`` = (mode, seconds) where mode is:
        #   "job_handle"     — default: after ``seconds`` park the live
        #                      Future as a job and return an in_progress
        #                      handle (wait_for_job / kill_job).
        #   "timeout_notice" — explicit opt-out: after ``seconds`` return a
        #                      non-error "timed out, still running" notice
        #                      (the Future is parked too, so work is not
        #                      lost and the agent can retry).
        self._wait_policy: dict[str, tuple[str, float]] = {}
        for m in tools_metadata:
            name = m["name"]
            lr = m.get("long_running")  # tri-state: None | True | False
            if lr is False:
                timeout = m.get("timeout")
                if timeout is None:
                    raise ValueError(
                        f"Tool {name!r} sets long_running=False (opting out "
                        f"of the default job-handle mechanism) but provides "
                        f"no 'timeout'. An explicit opt-out MUST carry its "
                        f"own timeout — opting out without one would let the "
                        f"tool hang forever, which is forbidden. Either drop "
                        f"long_running=False (get the job-handle default) or "
                        f"add a 'timeout' (seconds). See POLICIES.md §7."
                    )
                self._wait_policy[name] = ("timeout_notice", float(timeout))
            else:
                # None (unset) or True → default job-handle mechanism.
                self._wait_policy[name] = (
                    "job_handle", self._long_running_wait_seconds
                )
        # Active job-handles for ``long_running`` tools whose Future has
        # NOT yet resolved by the time the operator-visible wait expired.
        # See ``wait_for_job`` / ``kill_job`` (general tools registered at
        # the AgentProcess layer).
        self._jobs: dict[str, JobHandle] = {}
        self._pending_waiters: list[
            tuple[asyncio.Future[None], str | None, dict[str, Any] | None]
        ] = []
        # Per-id one-shot waiters used by ``wait_for_pending_call_by_id``.
        # Keyed by Claude ``tool_use_id``; value is the Future that
        # ``_create_tool_handler`` set_result()s the moment a matching
        # pending registers. Event-driven, no polling.
        self._pending_id_waiters: dict[str, asyncio.Future[PendingCall]] = {}
        self._pending_change_waiters: set[asyncio.Future[None]] = set()
        self._working_dir = working_dir
        self._agent_name = agent_name
        # Tool-call ids orphaned by a restart (model requested the tool,
        # process died before a result was produced). On resume the CLI
        # may replay these tool_use blocks; they must be reconciled
        # IMMEDIATELY (informed, not waited on) so neither the handler
        # (line ~413) nor wait_for_pending_call_by_id (line ~623) parks
        # forever on an id whose result will never come. Populated by
        # ``mark_orphaned_tool_calls`` on the first turn after a restart.
        # See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
        self._orphaned_call_ids: set[str] = set()

        # Create Future-based handler for each tool
        for tool_meta in tools_metadata:
            handler = self._create_tool_handler(tool_meta["name"])

            decorated = tool(
                tool_meta["name"], tool_meta["description"], tool_meta["input_schema"]
            )(handler)

            self.tool_handlers.append(decorated)

        # Create MCP server
        self.mcp_server_config: Any = create_sdk_mcp_server(
            name="deepworker", version="1.0.0", tools=self.tool_handlers
        )

        # [MCP_DIAG] Wrap the ListToolsRequest handler so we log EVERY
        # time the CLI subprocess asks "what tools does this server
        # expose?". The CLI must do this once per session. If after a
        # rotation we never see this log line, the new CLI subprocess
        # never queried our server — that's the missing-MCP smoking gun.
        # Best-effort: any failure in wiring the wrapper must NOT break
        # tool execution.
        try:
            from mcp.types import (  # noqa: PLC0415
                ListToolsRequest as _ListToolsRequest,
            )
            _server_instance = getattr(
                self.mcp_server_config, "instance", None
            )
            _handlers = getattr(_server_instance, "request_handlers", None)
            if _handlers is not None and _ListToolsRequest in _handlers:
                _orig_list_tools = _handlers[_ListToolsRequest]
                _agent_name = self._agent_name

                async def _wrapped_list_tools(req: Any) -> Any:
                    import sys as _sys2
                    print(
                        f"[MCP_DIAG] list_tools_called agent={_agent_name} "
                        f"tool_count={len(self._tool_names)}",
                        file=_sys2.stderr,
                        flush=True,
                    )
                    return await _orig_list_tools(req)

                _handlers[_ListToolsRequest] = _wrapped_list_tools
        except Exception as _exc:  # noqa: BLE001
            import sys as _sys3
            print(
                f"[MCP_DIAG] list_tools_wrap_failed "
                f"agent={self._agent_name} err={_exc!r}",
                file=_sys3.stderr,
                flush=True,
            )

        # [MCP_DIAG] Wrap the ListToolsRequest handler so we log EVERY
        # time the CLI subprocess asks "what tools does this server
        # expose?". The CLI must do this once per session. If after a
        # rotation we never see this log line, the new CLI subprocess
        # never queried our server — that's the missing-MCP smoking gun.
        # Best-effort: any failure in wiring the wrapper must NOT break
        # tool execution.
        try:
            from mcp.types import (  # noqa: PLC0415
                ListToolsRequest as _ListToolsRequest,
            )
            _server_instance = getattr(
                self.mcp_server_config, "instance", None
            )
            _handlers = getattr(_server_instance, "request_handlers", None)
            if _handlers is not None and _ListToolsRequest in _handlers:
                _orig_list_tools = _handlers[_ListToolsRequest]
                _agent_name = self._agent_name

                async def _wrapped_list_tools(req: Any) -> Any:
                    import sys as _sys2
                    print(
                        f"[MCP_DIAG] list_tools_called agent={_agent_name} "
                        f"tool_count={len(self._tool_names)}",
                        file=_sys2.stderr,
                        flush=True,
                    )
                    return await _orig_list_tools(req)

                _handlers[_ListToolsRequest] = _wrapped_list_tools
        except Exception as _exc:  # noqa: BLE001
            import sys as _sys3
            print(
                f"[MCP_DIAG] list_tools_wrap_failed "
                f"agent={self._agent_name} err={_exc!r}",
                file=_sys3.stderr,
                flush=True,
            )

        # [MCP_DIAG] Wrap the ListToolsRequest handler so we log EVERY
        # time the CLI subprocess asks "what tools does this server
        # expose?". The CLI must do this once per session. If after a
        # rotation we never see this log line, the new CLI subprocess
        # never queried our server — that's the missing-MCP smoking gun.
        # Best-effort: any failure in wiring the wrapper must NOT break
        # tool execution.
        try:
            from mcp.types import (  # noqa: PLC0415
                ListToolsRequest as _ListToolsRequest,
            )
            _server_instance = getattr(
                self.mcp_server_config, "instance", None
            )
            _handlers = getattr(_server_instance, "request_handlers", None)
            if _handlers is not None and _ListToolsRequest in _handlers:
                _orig_list_tools = _handlers[_ListToolsRequest]
                _agent_name = self._agent_name

                async def _wrapped_list_tools(req: Any) -> Any:
                    import sys as _sys2
                    print(
                        f"[MCP_DIAG] list_tools_called agent={_agent_name} "
                        f"tool_count={len(self._tool_names)}",
                        file=_sys2.stderr,
                        flush=True,
                    )
                    return await _orig_list_tools(req)

                _handlers[_ListToolsRequest] = _wrapped_list_tools
        except Exception as _exc:  # noqa: BLE001
            import sys as _sys3
            print(
                f"[MCP_DIAG] list_tools_wrap_failed "
                f"agent={self._agent_name} err={_exc!r}",
                file=_sys3.stderr,
                flush=True,
            )

    def _create_tool_handler(self, tool_name: str) -> Any:
        """Create async handler that blocks on Future until LangGraph resolves it.

        ⚠️ ARCHITECTURAL RULE: handler MUST block on Future!
        This ensures tool calls go through LangGraph ToolNode, making them
        visible in result.messages for dwcli broadcast, checkpointing, and HITL.

        DO NOT add direct execution here! See module docstring for details.

        ⚠️ POLICY (POLICIES.md §7, incident 2026-06-14): every step in this
        function MUST keep the SINGLE Claude ``tool_use_id`` as identity.
        No synthetic counter. No id(args). No retry. No silent fallback.
        Refer to the module-level policy banner. Reviewers reject any diff
        that reintroduces a second identity source or a phantom timeout.
        """

        async def handler(args: dict[str, Any]) -> MCPResult:
            # ── A: unified Claude tool_use_id as the only identity ──
            # ``_meta.claudecode/toolUseId`` is propagated to here by the
            # vendored SDK via ``_mcp_context`` ContextVar (set in
            # ``Query._handle_sdk_mcp_request``). If it is missing, the
            # SDK contract is broken — fail loudly. Silent fallback to a
            # synthetic id is exactly what caused the cluster of bugs
            # leading to incident 2026-06-14. See POLICIES.md §7.
            claude_tool_use_id = get_current_claude_tool_use_id()
            if claude_tool_use_id is None:
                raise RuntimeError(
                    "ProxyMCPServer handler invoked without a Claude "
                    "tool_use_id in the SDK ContextVar. The vendored "
                    "claude_agent_sdk MUST propagate "
                    "``_meta.claudecode/toolUseId`` via "
                    "``_mcp_context.set_current_claude_tool_use_id`` "
                    "around the CallToolRequest handler. If you are "
                    "tempted to add a synthetic fallback here, READ "
                    "POLICIES.md §7 first — that is the exact pattern "
                    "that caused incident 2026-06-14."
                )
            call_id = claude_tool_use_id

            # [MCP_DIAG] First entry into ANY tool handler proves the CLI
            # subprocess sees this MCP server alive. If after a rotation
            # the CLI never enters here for ``mcp__deepworker__*`` tools,
            # we have concrete proof the proxy is gone from its view.
            # Best-effort, never raises.
            try:
                import sys as _sys
                print(
                    f"[MCP_DIAG] tool_handler_entry agent={self._agent_name} "
                    f"tool={tool_name} call_id={call_id}",
                    file=_sys.stderr,
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                pass

            # ── Orphan reconciliation (restart) ──
            # If this id was orphaned by a restart, the result will never
            # come from LangGraph (the call belongs to a dead process). Do
            # NOT park on a Future — return the interrupted-notice result
            # immediately so the model is informed and nothing waits.
            # Structural fix, no timeout. See
            # docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
            if call_id in self._orphaned_call_ids:
                self._orphaned_call_ids.discard(call_id)
                return self._interrupted_by_restart_result(tool_name)

            future: asyncio.Future[MCPResult] = asyncio.Future()
            self.pending_calls[call_id] = {
                "future": future, "name": tool_name, "args": args,
            }
            self._wake_matching_pending_waiters(tool_name, args)
            # Wake any waiter parked on this exact id (chat_model's
            # ``wait_for_pending_call_by_id``).
            id_waiter = self._pending_id_waiters.pop(call_id, None)
            if id_waiter is not None and not id_waiter.done():
                id_waiter.set_result({
                    "id": call_id, "name": tool_name, "args": args,
                })

            # ── Bounded wait — HARD POLICY, NO bare await future ──
            # EVERY tool is waited on with a finite bound so it can NEVER
            # hang the turn forever. Two modes (set in __init__):
            #
            #  "job_handle"     (DEFAULT for every tool): wait
            #    ``wait_seconds``; if the Future has not resolved, PARK the
            #    live Future as a job (work continues on the ToolNode side)
            #    and return an in_progress handle so the agent can
            #    ``wait_for_job`` / ``kill_job``.
            #  "timeout_notice" (explicit ``long_running=False`` + its own
            #    ``timeout``): wait ``timeout``; if not resolved, return a
            #    NON-error "timed out, still running" notice (the Future is
            #    parked too, so the work is not lost and the agent may
            #    retry).
            #
            # ⚠️ A timeout is NEVER ``is_error=true`` — it does not put the
            # agent into ERROR; it informs and the agent continues.
            # ⚠️ DO NOT reintroduce a bare ``await future`` (no bound) for
            # ANY tool, and DO NOT default a tool to opting out — both
            # caused the eternal ``edit_file`` hang (mj1_init 2026-06-16).
            # See docs/POLICIES.md §7.
            mode, wait_seconds = self._wait_policy.get(
                tool_name, ("job_handle", self._long_running_wait_seconds)
            )
            try:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=wait_seconds,
                    )
                    return result
                except asyncio.TimeoutError:
                    # Park the live Future (work is NOT cancelled) so it can
                    # be picked up later and is never lost.
                    #
                    # Namespace contract (observed incident, 2026-07-31 —
                    # fourth locus of the same class of defect). The
                    # advertised ``job_id`` MUST be a canonical ``job_N``
                    # from ``ParkedJobRegistry`` so ``wait_for_job``/
                    # ``kill_job`` resolve it. We keep the local
                    # ``self._jobs`` dict under the SAME ``job_N`` key —
                    # not the ``call_id`` — and reach it from
                    # ``resolve_call`` via the registry's ``tool_use_id``
                    # alias (dual-key lookup, §10 reviewer decision).
                    #
                    # If ``BoundedToolCallMiddleware`` (or another proxy
                    # site) already parked this same ``call_id``, reuse
                    # that ``job_N`` instead of minting a second one —
                    # avoids the §3.1 race that used to produce two
                    # different job ids for the same live Future.
                    from deepagents.middleware.parked_job_registry import (  # noqa: PLC0415
                        get_parked_job_registry,
                    )
                    from deepagents.middleware.bounded_tool_call import (  # noqa: PLC0415
                        build_in_progress_notice,
                    )

                    registry = get_parked_job_registry()
                    existing = (
                        registry.resolve_by_alias(call_id) if call_id else None
                    )
                    job_id = (
                        existing.job_id
                        if existing is not None
                        else registry.allocate()
                    )
                    registry.register(
                        job_id,
                        kind="tool",
                        handle=future,
                        alias=call_id or None,
                        tool_name=tool_name,
                    )
                    future.add_done_callback(lambda _f: registry.drop(job_id))

                    self._jobs[job_id] = {
                        "future": future,
                        "tool_name": tool_name,
                        "args": args,
                        "agent_name": self._agent_name,
                        "tool_call_id": call_id,
                    }
                    text = build_in_progress_notice(
                        tool_name, job_id, wait_seconds
                    )
                    if mode == "timeout_notice":
                        # Explicit opt-out tool: same shared notice + a
                        # trailing sentence keeping the older tone. NOT an
                        # error.
                        text = (
                            text + " You may also just continue or retry."
                        )
                    return {"content": [{"type": "text", "text": text}]}
            finally:
                # Remove from pending_calls if still there (it is removed
                # both on normal resolve and on job-park). For long-running
                # tools whose Future has NOT resolved we keep the job
                # entry alive in ``self._jobs`` — that is the handle.
                self.pending_calls.pop(call_id, None)

        return handler

    @property
    def tool_names(self) -> list[str]:
        """Tool names exposed through this proxy MCP server."""
        return list(self._tool_names)

    def has_pending_calls(self) -> bool:
        """Check if there are any pending tool calls."""
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> list[PendingCall]:
        """Get list of pending tool calls for LangGraph to execute."""
        return [
            {"id": call_id, "name": info["name"], "args": info["args"]}
            for call_id, info in self.pending_calls.items()
        ]

    # ── Removed 2026-06-17 (variant A, single source of truth) ──
    # ``wait_for_pending_calls`` / ``wait_for_any_pending_call`` (the last bare
    # unbounded ``await fut``) / ``wait_for_matching_pending_calls`` were
    # PRODUCTION-DEAD (only tests called them) and each carried its own ad-hoc
    # timer or none at all. The only live stage-A wait is
    # ``wait_for_pending_call_by_id`` (bounded by the single source). See
    # docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md.

    def mark_orphaned_tool_calls(self, call_ids: Iterable[str]) -> None:
        """Mark tool-call ids as orphaned by a restart.

        Called on the first turn after a restart with the ids of tool
        calls the model requested but for which no result was ever
        produced (detected from LangGraph history — see
        ``orphaned_tool_calls.find_orphaned_tool_calls``). Thereafter the
        handler and ``wait_for_pending_call_by_id`` short-circuit these
        ids with the interrupted-notice result instead of parking on a
        Future that will never resolve.

        Idempotent; ids are consumed on first use by the handler.
        """
        for cid in call_ids:
            if cid:
                self._orphaned_call_ids.add(cid)

    def _interrupted_by_restart_result(self, tool_name: str) -> MCPResult:
        """The result fed back for a tool call orphaned by a restart.

        Informs the model the call did not run to completion (so it can
        decide whether to retry) — it does NOT fabricate a success."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"⚠ The tool call '{tool_name}' was interrupted by "
                        f"an agent restart and produced no result. It was "
                        f"NOT executed to completion. Do not wait for it; "
                        f"decide whether to retry based on the current task."
                    ),
                }
            ],
            "is_error": True,
        }

    def resolve_call(self, call_id: str, result: MCPResult) -> None:
        """Resolve a pending tool call with result from LangGraph ToolNode.

        Unblocks the Future handler, returning result to Claude CLI.

        ⚠ POLICY (POLICIES.md §7 / INCIDENT 2026-06-15 — double dispatch
        in ``test_claude_code_mcp_prefix``): ``pending_calls[call_id]``
        is dropped SYNCHRONOUSLY, BEFORE we set the Future's result.

        Why this matters
        ────────────────
        After ``set_result``, the spawned handler task is rescheduled but
        does NOT run until the next event-loop tick. Between that
        rescheduling and the handler's ``finally: pending_calls.pop(...)``
        cleanup, other coroutines on the same loop continue executing —
        and one of them is the very same ``ainvoke`` that just called
        us. Specifically, ``ainvoke``'s "tool resolved, check
        ``has_pending_calls`` to decide receive_response vs early-return"
        block (chat_model.py §8) is the next thing to run, and if the
        just-resolved call_id is still in the dict it observes it as a
        live pending call. Result: a second ``AIMessage(tool_calls=[
        same_id])`` is returned to LangGraph, ToolNode re-runs the tool,
        and the agent enters a 180-second timeout with two identical
        log lines:

            [ainvoke] Returning AIMessage with 1 tool_calls: ['ls']
            [ainvoke] Returning AIMessage with 1 tool_calls: ['ls']

        Observed in production (mj1_init 2026-06-14) and reproducibly in
        ``test_claude_code_mcp_prefix`` (both tests) and the cluster
        ``test_restart_self_e2e``. The handler's ``finally`` pop is kept
        as a belt-and-braces idempotent no-op for cancellation /
        parked-job recovery.

        Also accepts a resolve for a parked job (long_running tool whose
        Future was parked in ``self._jobs``) — drops it from the job
        registry after setting the result.
        """
        if call_id in self.pending_calls:
            # SOURCE-OF-TRUTH UPDATE FIRST. The dict reflects "resolved"
            # before any other coroutine gets a chance to observe it.
            future = self.pending_calls.pop(call_id)["future"]
            if not future.done():
                future.set_result(result)
            return
        # Dual-key parked-job lookup (observed fix, 2026-07-31): the local
        # dict is now keyed by canonical ``job_N``, but the SDK still hands
        # us the originating ``tool_use_id`` (``toolu_…``). Resolve either
        # namespace via the shared ``ParkedJobRegistry`` alias.
        job_key: str | None = None
        if call_id in self._jobs:
            job_key = call_id
        else:
            from deepagents.middleware.parked_job_registry import (  # noqa: PLC0415
                get_parked_job_registry,
            )
            parked = get_parked_job_registry().resolve(call_id)
            if parked is not None and parked.job_id in self._jobs:
                job_key = parked.job_id
        if job_key is not None:
            future = self._jobs.pop(job_key)["future"]
            if not future.done():
                future.set_result(result)

    async def wait_for_pending_call_by_id(
        self,
        claude_tool_use_id: str,
    ) -> PendingCall:
        """Event-driven wait for the pending call with this exact id.

        Returns immediately if the pending call already registered;
        otherwise parks on a one-shot Future that ``_create_tool_handler``
        wakes the moment a matching id appears. NO polling, NO timeout —
        the caller is responsible for bounding the wait if needed
        (chat_model.py uses this on the message-stream side).

        ⚠️ FALSE ASSURANCE REMOVED (2026-06-16): an earlier version of
        this docstring claimed "the sequential SDK control-channel
        GUARANTEES the matching control request will arrive". That is
        NOT true and deadlocked mj1_init (2026-06-15): if the caller
        passes the ``tool_use_id`` of a tool call that does NOT go
        through our MCP control channel (e.g. Claude's BUILT-IN
        ``execute`` misclassified as our proxy ``execute`` by base-name),
        no matching control request EVER arrives and this parks forever.
        The matching request arrives only for OUR proxy tools (name
        starts with ``mcp__deepworker__``). See
        docs/FACTS_TOOL_USE_ID_VS_BLOCK_ID.md.

        Args:
            claude_tool_use_id: The Claude ``tool_use_id`` (the per-call
                ``toolu_*``, i.e. ``ToolUseBlock.id`` = ``content[].id``
                in the message stream — NOT the message-block id
                ``msg_*``). MUST be the exact same id the SDK propagates
                via the ContextVar for one of OUR proxy tools.

        Returns:
            A ``PendingCall`` dict with id/name/args once registered.
        """
        existing = self.pending_calls.get(claude_tool_use_id)
        if existing is not None:
            return {
                "id": claude_tool_use_id,
                "name": existing["name"],
                "args": existing["args"],
            }
        # Orphaned by restart: do NOT wait — the matching control request
        # will never arrive under this id. Return a synthetic PendingCall
        # so the caller proceeds (the handler short-circuit above feeds
        # the interrupted-notice result into LangGraph's ToolNode). No
        # timeout. See docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md.
        if claude_tool_use_id in self._orphaned_call_ids:
            return {
                "id": claude_tool_use_id,
                "name": "",
                "args": {},
            }
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[PendingCall] = loop.create_future()
        # Replace any prior waiter for the same id (shouldn't happen in
        # practice — chat_model waits for one id per turn — but be
        # defensive).
        old = self._pending_id_waiters.get(claude_tool_use_id)
        if old is not None and not old.done():
            old.cancel()
        self._pending_id_waiters[claude_tool_use_id] = fut
        try:
            # BOUNDED by the SINGLE source of truth (variant A) — never a bare
            # ``await``. If the control request never arrives (lost dispatch:
            # mj1_init / mj2_vac), raise DispatchTimeout so ainvoke surfaces a
            # NON-error in_progress handle instead of hanging forever.
            return await asyncio.wait_for(
                asyncio.shield(fut), timeout=_default_wait_seconds()
            )
        except asyncio.TimeoutError as exc:
            raise DispatchTimeout(claude_tool_use_id) from exc
        finally:
            # In case we were cancelled before the handler fired.
            if self._pending_id_waiters.get(claude_tool_use_id) is fut:
                self._pending_id_waiters.pop(claude_tool_use_id, None)

    def cancel_pending_waiter(self, claude_tool_use_id: str) -> bool:
        """Cancel an id-keyed waiter parked in ``wait_for_pending_call_by_id``.

        Safety net (defence in depth): the wait has NO timeout because the
        control request for one of OUR proxy tools is expected to arrive.
        But if the call is NOT one of ours — e.g. Claude reported
        ``No such tool available: <name>`` for a tool it named but the CLI
        could not route — that control request will NEVER arrive and the
        waiter would park forever. Detecting that message (see
        ``parse_no_such_tool``) and calling this cancels the waiter so the
        turn can proceed.

        Note: prefix-only classification in chat_model already prevents us
        from ENTERING the wait for a non-proxy/bare name, so this is a
        belt-and-braces second line, not the primary fix.

        Returns True if a live waiter was cancelled, False otherwise. Does
        NOT touch ``pending_calls`` (a real in-flight tool's Future) — only
        the id-keyed waiter.
        """
        fut = self._pending_id_waiters.get(claude_tool_use_id)
        if fut is None or fut.done():
            return False
        fut.cancel()
        # The waiter's own ``finally`` pops it; pop defensively too.
        self._pending_id_waiters.pop(claude_tool_use_id, None)
        return True

    def get_job_registry(self) -> dict[str, JobHandle]:
        """Expose the in-flight long_running job handles.

        Keys are canonical ``job_N`` ids from the shared
        ``ParkedJobRegistry`` — the same namespace ``wait_for_job``/
        ``kill_job`` resolve. Each entry additionally carries the
        originating ``tool_call_id`` in ``entry["tool_call_id"]``, since
        ``resolve_call`` still enters by that field (LangChain protocol).
        """
        return self._jobs

    # ``_matching_pending_calls`` / ``_matching_pending_call_batch`` removed
    # 2026-06-17: their only callers were the production-dead wait_for_*
    # methods deleted above (variant A). ``_wake_matching_pending_waiters``
    # below is the live path that wakes id-agnostic waiters.

    def _wake_matching_pending_waiters(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        for fut, wanted_tool, wanted_args in list(self._pending_waiters):
            if fut.done():
                continue
            if wanted_tool is not None and wanted_tool != tool_name:
                continue
            if wanted_args is not None and wanted_args != arguments:
                continue
            fut.set_result(None)
        for fut in list(self._pending_change_waiters):
            if not fut.done():
                fut.set_result(None)
