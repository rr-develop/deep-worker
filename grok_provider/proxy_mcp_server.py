"""ProxyMCPServer — in-agent-process HTTP MCP server.

⚠️ ARCHITECTURAL INVARIANT (same as claude_code_provider's proxy):
═══════════════════════════════════════════════════════════════════════════════
The MCP handler NEVER executes the tool. It only parks an asyncio.Future and
awaits resolution. The Future is set by ``GrokChatModel.ainvoke`` when the
matching ToolMessage arrives back from LangGraph's ToolNode.

This guarantees every tool call AND every tool result flows through LangGraph
and ends up in ``result["messages"]``, which is the single source of truth for
dwcli broadcast, checkpointer, and TaskManager. See §7.5 of
DESIGN_CODEX_PROVIDER.md for the timing diagram.

DO NOT add direct execution here.
═══════════════════════════════════════════════════════════════════════════════

Transport: streamable HTTP, bound to ``127.0.0.1:<random-free-port>``. No
external process — the listener runs as an asyncio task in the agent's own
event loop. See §7.2–§7.3 of the design doc.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any

from grok_provider.custom_types import (
    JobHandle,
    MCPResult,
    PendingCall,
    PendingCallInternal,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


# Default cap before the handler issues a job-handle. Applies to EVERY tool
# by default (the job-handle mechanism is the default — see POLICIES.md
# §7.1). It is NOT a deadline for the work — the Future is parked, not
# cancelled. 120s ("~2 minutes") per project policy, identical to
# claude_code_provider. Override via ``DW_LONG_RUNNING_WAIT_SECONDS``.
_DEFAULT_LONG_RUNNING_WAIT_SECONDS = 120.0


def _default_wait_seconds() -> float:
    """Resolve the default job-handle wait, honoring the
    ``DW_LONG_RUNNING_WAIT_SECONDS`` env override (positive number)."""
    raw = os.environ.get("DW_LONG_RUNNING_WAIT_SECONDS")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _DEFAULT_LONG_RUNNING_WAIT_SECONDS


class ProxyMCPServer:
    """Hosts an MCP server inside the agent process.

    ``start()`` binds the HTTP listener (idempotent). ``stop()`` tears it
    down (idempotent). Between, grok CLI connects to it per-session: each
    ``session/new`` includes an ``mcpServers`` entry of the form
    ``{"type":"http","name":<server_name>,"url":<url_async()>}`` — grok
    speaks the streamable-HTTP MCP transport. NOTHING is written to any
    global grok config file (learned from a related internal incident), so concurrent agents never
    collide.

    Pending tool calls live in ``pending_calls`` — keyed by a stable
    per-call id we generate. Each call parks an asyncio.Future and waits
    on it with a FINITE bound (HARD POLICY, 2026-06-16, common to all
    providers): either it resolves and the result is returned, or the wait
    elapses and the live Future is PARKED as a job — the handler returns a
    NON-error ``in_progress`` job-handle (``wait_for_job``/``kill_job``).
    A timeout is NEVER ``is_error`` and the work is NEVER cancelled; a tool
    can never hang the turn forever. See ``_handle_tool_call`` and
    ``docs/POLICIES.md`` §7.1.
    """

    def __init__(
        self,
        tools_metadata: list[ToolMetadata],
        long_running_wait_seconds: float | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self._tools_metadata: list[ToolMetadata] = list(tools_metadata)
        # Default job-handle wait (120s, env-overridable). Same policy and
        # contract as claude_code_provider — provider-agnostic, no
        # exceptions. See docs/POLICIES.md §7.1.
        if long_running_wait_seconds is None:
            long_running_wait_seconds = _default_wait_seconds()
        self._long_running_wait_seconds = long_running_wait_seconds
        # ── Per-tool wait policy (HARD POLICY, 2026-06-16) ──────────────
        # EVERY tool gets the job-handle mechanism BY DEFAULT. A tool may
        # opt OUT only EXPLICITLY (``long_running=False``), and that opt-out
        # MUST carry its own ``timeout`` — a timeout-less opt-out is
        # forbidden (it would let the tool hang forever). NO bare unbounded
        # wait. A timeout NEVER becomes an error.
        #   "job_handle"     — default: after ``seconds`` park the live
        #                      Future as a job, return in_progress handle.
        #   "timeout_notice" — explicit opt-out: after ``seconds`` return a
        #                      non-error "timed out, still running" notice.
        self._wait_policy: dict[str, tuple[str, float]] = {}
        for m in tools_metadata:
            name = m["name"]
            lr = m.get("long_running")  # tri-state: None | True | False
            if lr is False:
                t = m.get("timeout")
                if t is None:
                    raise ValueError(
                        f"Tool {name!r} sets long_running=False (opting out "
                        f"of the default job-handle mechanism) but provides "
                        f"no 'timeout'. An explicit opt-out MUST carry its "
                        f"own timeout — opting out without one would let the "
                        f"tool hang forever, which is forbidden. See "
                        f"POLICIES.md §7.1."
                    )
                self._wait_policy[name] = ("timeout_notice", float(t))
            else:
                self._wait_policy[name] = (
                    "job_handle", self._long_running_wait_seconds
                )
        self._host = host
        self._call_counter = 0
        self.pending_calls: dict[str, PendingCallInternal] = {}
        # Live job-handles for tools whose Future has not yet resolved by
        # the time the wait elapsed (work continues; never cancelled).
        self._jobs: dict[str, JobHandle] = {}
        self._pending_waiters: list[
            tuple[asyncio.Future[None], str | None, dict[str, Any] | None]
        ] = []
        # Per-PID server name. Grok passes this name per-session in
        # ``session/new mcpServers``; the PID keeps it unique if several
        # deep-worker agents somehow share a process namespace.
        self.server_name = f"deep_worker_proxy_{os.getpid()}"

        self._port: int | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._uvicorn_server: Any | None = None  # uvicorn.Server, set in start()
        self._session_manager_cm: Any | None = None  # async context manager handle

    def update_tools(self, tools_metadata: list[ToolMetadata]) -> None:
        """Hot-swap the tool set without restarting the listener.

        Called by ``bind_tools()`` when the model's tool set changes.
        The MCP server's ``list_tools`` handler reads ``self._tools_metadata``
        on every request, so the next ``tools/list`` from grok sees the
        new set automatically.
        """
        self._tools_metadata = list(tools_metadata)

    # ------------------------------------------------------------------
    # Public, read-only properties
    # ------------------------------------------------------------------
    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("ProxyMCPServer.port: server not started")
        return self._port

    @property
    def url(self) -> str:
        """Full streamable-HTTP MCP endpoint URL for ``session/new``.

        The listener mounts the session manager on ``/mcp`` (see
        :meth:`start`), so the URL grok must connect to is
        ``http://<host>:<port>/mcp``. Raises if not started."""
        return f"http://{self._host}:{self.port}/mcp"

    async def url_async(self) -> str:
        """Ensure the listener is started, then return :attr:`url`.

        ``GrokChatModel.ainvoke`` calls this lazily right before
        ``session/new`` so the proxy is guaranteed bound before grok tries
        to connect. ``start()`` is idempotent."""
        await self.start()
        return self.url

    @property
    def tool_names(self) -> list[str]:
        return [m["name"] for m in self._tools_metadata]

    # ------------------------------------------------------------------
    # Pending-call accessors (used by GrokChatModel.ainvoke)
    # ------------------------------------------------------------------
    def has_pending_calls(self) -> bool:
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> list[PendingCall]:
        return [
            {"id": call_id, "name": info["name"], "args": info["args"]}
            for call_id, info in self.pending_calls.items()
        ]

    def get_job_registry(self) -> dict[str, JobHandle]:
        """Live job-handles for tool calls parked after their wait elapsed.

        Used by ``wait_for_job`` / ``kill_job`` and by tests. The Futures
        are NOT cancelled — the real work continues on the ToolNode side.
        """
        return self._jobs

    async def wait_for_pending_calls(
        self,
        *,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> list[PendingCall]:
        """Wait until a matching pending call exists, without polling.

        ``GrokChatModel`` may call this to correlate a grok ``tool_call``
        (CallMcpTool dispatch for one of our proxy tools, seen on the ACP
        notification stream) with the MCP ``tools/call`` the handler parks —
        the two can arrive in either order. Return immediately for an
        existing match, otherwise park on a Future that ``_handle_tool_call``
        resolves when it creates a matching pending call. Event-driven, no
        polling.
        """
        existing = self._matching_pending_calls(tool_name, arguments)
        if existing:
            return existing

        fut = asyncio.get_running_loop().create_future()
        waiter = (fut, tool_name, arguments)
        self._pending_waiters.append(waiter)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return self._matching_pending_calls(tool_name, arguments)
        finally:
            with contextlib.suppress(ValueError):
                self._pending_waiters.remove(waiter)

    def resolve_call(
        self,
        call_id: str,
        result: MCPResult,
        *,
        tool_name: str | None = None,
    ) -> None:
        """Set the Future for ``call_id`` to ``result``. The awaiting MCP
        handler unblocks and returns the value to grok CLI. Idempotent /
        no-op for unknown ids. Also resolves a call that was PARKED as a job
        (its Future lives in ``self._jobs``) — the work completing late is
        delivered the same way, and the job entry is dropped.

        When ``call_id`` is an ACP ``toolCallId`` (not the handler's synthetic
        id), pass ``tool_name`` so we can fall back to the oldest matching
        pending/job entry — grok serializes proxy tool calls one at a time."""
        if self._resolve_call_id(call_id, result):
            return
        if tool_name is None:
            return
        for pid, info in list(self.pending_calls.items()):
            if info["name"] == tool_name:
                if self._resolve_call_id(pid, result):
                    return
        for jid, job in list(self._jobs.items()):
            if job["tool_name"] == tool_name:
                if self._resolve_call_id(jid, result):
                    return

    def _resolve_call_id(self, call_id: str, result: MCPResult) -> bool:
        info = self.pending_calls.get(call_id)
        if info is not None:
            fut = info["future"]
            if not fut.done():
                fut.set_result(result)
            return True
        job = self._jobs.pop(call_id, None)
        if job is not None:
            jfut = job["future"]
            if not jfut.done():
                jfut.set_result(result)
            return True
        return False

    def _matching_pending_calls(
        self,
        tool_name: str | None,
        arguments: dict[str, Any] | None,
    ) -> list[PendingCall]:
        out: list[PendingCall] = []
        for call_id, info in self.pending_calls.items():
            if tool_name is not None and info["name"] != tool_name:
                continue
            if arguments is not None and info["args"] != arguments:
                continue
            out.append({"id": call_id, "name": info["name"], "args": info["args"]})
        return out

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

    def _wake_all_pending_waiters(self) -> None:
        for fut, _wanted_tool, _wanted_args in list(self._pending_waiters):
            if not fut.done():
                fut.set_result(None)

    # ------------------------------------------------------------------
    # The actual tool handler — used both by the live MCP server and
    # by tests (via ``_call_tool_for_test``).
    # ------------------------------------------------------------------
    async def _handle_tool_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Park a Future, wait (BOUNDED) for ``resolve_call``, return MCP
        content blocks.

        ── Bounded wait — HARD POLICY, NO bare await future ──
        EVERY tool is waited on with a finite bound so it can NEVER hang the
        turn forever (identical contract to claude_code_provider). Two modes
        set in __init__:

          "job_handle"     (DEFAULT for every tool): wait ``wait_seconds``;
            if the Future has not resolved, PARK the live Future as a job
            (work continues; ``wait_for_job``/``kill_job`` pick it up) and
            return a non-error in_progress handle.
          "timeout_notice" (explicit ``long_running=False`` + own
            ``timeout``): on timeout return a NON-error "timed out, still
            running" notice (the Future is parked too — work not lost).

        ⚠️ A timeout is NEVER ``is_error=true`` — it informs and the agent
        continues / retries. ⚠️ DO NOT reintroduce a bare unbounded
        ``await future`` (it hung edit_file forever, a related internal incident),
        and DO NOT cancel the work on timeout (use ``shield``). See
        docs/POLICIES.md §7.1.
        """
        self._call_counter += 1
        call_id = f"{tool_name}_{self._call_counter}_{id(args)}"

        fut: asyncio.Future[MCPResult] = asyncio.get_event_loop().create_future()
        self.pending_calls[call_id] = {"future": fut, "name": tool_name, "args": args}
        self._wake_matching_pending_waiters(tool_name, args)

        mode, wait_seconds = self._wait_policy.get(
            tool_name, ("job_handle", self._long_running_wait_seconds)
        )
        parked = False
        try:
            # ``shield`` so the timeout fires WITHOUT cancelling the real
            # work — the Future keeps running for ``wait_for_job``.
            result = await asyncio.wait_for(
                asyncio.shield(fut), timeout=wait_seconds
            )
            return result.get("content", [])
        except asyncio.TimeoutError:
            # Park the live Future as a job (NOT cancelled, NOT lost).
            self._jobs[call_id] = {
                "future": fut, "tool_name": tool_name, "args": args,
            }
            parked = True
            if mode == "timeout_notice":
                text = (
                    f"Tool '{tool_name}' timed out after {wait_seconds:g}s "
                    f"but was NOT killed and is still running in the "
                    f"background. status: in_progress. job_id: {call_id}. "
                    f"actions: wait_for_job(job_id=\"{call_id}\", "
                    f"timeout=<seconds>) or kill_job(job_id=\"{call_id}\"). "
                    f"You may also just continue or retry."
                )
            else:
                text = (
                    f"Tool '{tool_name}' is still running after "
                    f"{wait_seconds:g}s. status: in_progress. "
                    f"job_id: {call_id}. actions: wait_for_job(job_id="
                    f"\"{call_id}\", timeout=<seconds>) or kill_job("
                    f"job_id=\"{call_id}\")."
                )
            # NON-error result — never is_error.
            return [{"type": "text", "text": text}]
        finally:
            # Drop the pending entry on every exit. When parked, the live
            # Future stays alive in ``self._jobs`` — that is the handle.
            self.pending_calls.pop(call_id, None)
            del parked

    # Public test-only alias so tests can drive the handler without going
    # through the HTTP transport. Production code path goes through
    # ``_handle_tool_call`` registered as MCP ``call_tool``.
    # (Public name avoids pyright's ``reportPrivateUsage`` in the test
    # module; this is for tests only.)
    async def call_tool_for_test(
        self, tool_name: str, args: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return await self._handle_tool_call(tool_name, args)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Bind the HTTP listener on a free localhost port. Idempotent."""
        if self._server_task is not None:
            return  # already started

        # Build the low-level MCP server with our list_tools / call_tool handlers.
        from mcp.server.lowlevel import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.types import TextContent, Tool

        srv: Server = Server(name=self.server_name, version="0.1.0")

        async def _list_tools_handler() -> list[Tool]:
            return [
                Tool(
                    name=m["name"],
                    description=m["description"],
                    inputSchema=m["input_schema"],
                )
                for m in self._tools_metadata
            ]

        async def _call_tool_handler(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:
            blocks = await self._handle_tool_call(name, arguments)
            # Convert generic block dicts into MCP TextContent (the most
            # common shape; richer MCP content types are not used here).
            out: list[TextContent] = []
            for b in blocks:
                if b.get("type") == "text":
                    out.append(TextContent(type="text", text=str(b.get("text", ""))))
                else:
                    out.append(TextContent(type="text", text=str(b)))
            return out

        srv.list_tools()(_list_tools_handler)
        srv.call_tool()(_call_tool_handler)

        # Wrap in streamable-HTTP session manager → ASGI app
        session_manager = StreamableHTTPSessionManager(app=srv, stateless=True)

        # Build a minimal Starlette app exposing the manager on /mcp
        from starlette.applications import Starlette
        from starlette.routing import Mount

        # ASGI handler signature: ``(scope, receive, send)`` where each
        # is a callable / mapping per the ASGI spec. We re-export them
        # as ``Any`` because the typing surface for ASGI is awkward
        # (per-protocol scope dicts, async-callable typing differences)
        # and we don't add value pinning it strictly here.
        async def asgi_handler(scope: Any, receive: Any, send: Any) -> None:
            try:
                await session_manager.handle_request(scope, receive, send)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Client disconnect / teardown must not wedge the agent loop.
                if exc.__class__.__name__ in (
                    "ClosedResourceError",
                    "BrokenResourceError",
                ):
                    logger.debug(
                        "ProxyMCPServer: client disconnected during request"
                    )
                    return
                raise

        starlette_app = Starlette(
            routes=[Mount("/mcp", app=asgi_handler)],
        )

        # Spawn uvicorn as an asyncio task — in-process, no external
        # subprocess. Use port=0 so the OS atomically allocates a free port;
        # after startup we read the actual bound port from uvicorn's server.
        import uvicorn

        started_event = asyncio.Event()

        class _StartedServer(uvicorn.Server):
            async def startup(self, sockets: Any = None) -> None:
                await super().startup(sockets=sockets)
                if self.started:
                    started_event.set()

        config = uvicorn.Config(
            app=starlette_app,
            host=self._host,
            port=0,
            log_level="warning",
            access_log=False,
            lifespan="off",  # we manage session_manager lifespan ourselves
        )
        self._uvicorn_server = _StartedServer(config)

        # The session manager itself is an async context manager; we
        # enter it for the lifetime of the server.
        self._session_manager_cm = session_manager.run()
        await self._session_manager_cm.__aenter__()

        self._server_task = asyncio.create_task(
            self._uvicorn_server.serve(), name=f"proxy-mcp-{self.server_name}"
        )

        # Event-driven readiness wait: no sleep-loop. Either uvicorn startup
        # sets the event, the server task exits early, or we time out.
        started_waiter = asyncio.create_task(started_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {started_waiter, self._server_task},
                timeout=5.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if started_waiter not in done:
                if self._server_task in done:
                    await self._server_task
                    raise RuntimeError("ProxyMCPServer uvicorn task exited before startup")
                raise TimeoutError("ProxyMCPServer uvicorn startup timed out")
        finally:
            started_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await started_waiter

        servers = getattr(self._uvicorn_server, "servers", None) or []
        sockets = getattr(servers[0], "sockets", None) if servers else None
        if not sockets:
            raise RuntimeError("ProxyMCPServer could not determine uvicorn port")
        self._port = int(sockets[0].getsockname()[1])

    async def stop(self) -> None:
        """Stop the listener; resolve any pending Futures with a
        "server stopping" error block so awaiting handlers don't leak.
        (This is a shutdown signal, NOT a tool timeout — tool timeouts
        never produce ``is_error``; see ``_handle_tool_call``.) Idempotent."""
        if self._server_task is None:
            return
        # Resolve all pending Futures with a "server stopping" message so
        # awaiting handlers don't leak.
        for _call_id, info in list(self.pending_calls.items()):
            fut = info["future"]
            if not fut.done():
                fut.set_result({
                    "content": [{"type": "text", "text": "ProxyMCPServer stopping"}],
                    "is_error": True,
                })
        self.pending_calls.clear()
        self._wake_all_pending_waiters()

        # Ask uvicorn to exit
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._server_task
        self._server_task = None
        self._uvicorn_server = None

        # Drop the session-manager context without calling __aexit__ here.
        # StreamableHTTPSessionManager.run() enters an anyio cancel scope in
        # the uvicorn task; exiting it from this coroutine raises
        # "Attempted to exit cancel scope in a different task" and can wedge
        # in-flight MCP handlers (inter-agent grok turns, 2026-06-28).
        self._session_manager_cm = None

        self._port = None


# ---------------------------------------------------------------------------
# Process-global singleton accessor
# ---------------------------------------------------------------------------
# Lowercase module-private name — pyright treats ALL_CAPS as ``Final``
# and refuses reassignment, but we need lazy construction.
_global_proxy: ProxyMCPServer | None = None


def get_global_proxy_mcp_server(
    tools_metadata: list[ToolMetadata] | None = None,
) -> ProxyMCPServer:
    """Return the process-global :class:`ProxyMCPServer`.

    Lazily constructed on first call. Same instance is returned forever
    after — survives LangGraph's model rebuilds because it lives at
    module scope. If ``tools_metadata`` is provided, the proxy's tool
    set is hot-swapped to that list (no restart).
    """
    global _global_proxy
    if _global_proxy is None:
        _global_proxy = ProxyMCPServer(tools_metadata or [])
    elif tools_metadata is not None:
        _global_proxy.update_tools(tools_metadata)
    return _global_proxy
