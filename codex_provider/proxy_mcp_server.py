"""ProxyMCPServer — in-agent-process HTTP MCP server.

⚠️ ARCHITECTURAL INVARIANT (same as claude_code_provider's proxy):
═══════════════════════════════════════════════════════════════════════════════
The MCP handler NEVER executes the tool. It only parks an asyncio.Future and
awaits resolution. The Future is set by ``CodexChatModel.ainvoke`` when the
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
import socket
from typing import Any

from codex_provider.custom_types import (
    MCPResult,
    PendingCall,
    PendingCallInternal,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class ProxyMCPServer:
    """Hosts an MCP server inside the agent process.

    ``start()`` binds the HTTP listener (idempotent). ``stop()`` tears it
    down (idempotent). Between, Codex CLI connects via
    ``codex mcp add deepworker_<pid> --url http://127.0.0.1:<port>``.

    Pending tool calls live in ``pending_calls`` — keyed by a stable
    per-call id we generate. Each call parks an asyncio.Future, awaits
    up to ``timeout`` seconds, then either returns the resolved result
    or an ``is_error`` block on timeout.
    """

    def __init__(
        self,
        tools_metadata: list[ToolMetadata],
        timeout: float = 120.0,
        host: str = "127.0.0.1",
    ) -> None:
        self._tools_metadata: list[ToolMetadata] = list(tools_metadata)
        self._timeout = timeout
        self._host = host
        self._call_counter = 0
        self.pending_calls: dict[str, PendingCallInternal] = {}
        # Per-PID server name so concurrent deep-worker agents don't
        # clobber each other in ``~/.codex/config.toml``.
        self.server_name = f"deepworker_{os.getpid()}"

        self._port: int | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._uvicorn_server: Any | None = None  # uvicorn.Server, set in start()
        self._session_manager_cm: Any | None = None  # async context manager handle

    def update_tools(self, tools_metadata: list[ToolMetadata]) -> None:
        """Hot-swap the tool set without restarting the listener.

        Called by ``bind_tools()`` when the model's tool set changes.
        The MCP server's ``list_tools`` handler reads ``self._tools_metadata``
        on every request, so the next ``tools/list`` from Codex sees the
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
    def tool_names(self) -> list[str]:
        return [m["name"] for m in self._tools_metadata]

    # ------------------------------------------------------------------
    # Pending-call accessors (used by CodexChatModel.ainvoke)
    # ------------------------------------------------------------------
    def has_pending_calls(self) -> bool:
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> list[PendingCall]:
        return [
            {"id": call_id, "name": info["name"], "args": info["args"]}
            for call_id, info in self.pending_calls.items()
        ]

    def resolve_call(self, call_id: str, result: MCPResult) -> None:
        """Set the Future for ``call_id`` to ``result``. The awaiting MCP
        handler unblocks and returns the value to Codex CLI. Idempotent /
        no-op for unknown ids."""
        info = self.pending_calls.get(call_id)
        if info is None:
            return
        fut = info["future"]
        if not fut.done():
            fut.set_result(result)

    # ------------------------------------------------------------------
    # The actual tool handler — used both by the live MCP server and
    # by tests (via ``_call_tool_for_test``).
    # ------------------------------------------------------------------
    async def _handle_tool_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Park a Future, wait for ``resolve_call``, return MCP content blocks.

        Returns a list of content blocks (the wire format MCP expects). On
        timeout, returns a single ``text`` block with ``is_error`` semantics;
        we don't raise — Codex would otherwise abort the whole turn.
        """
        self._call_counter += 1
        call_id = f"{tool_name}_{self._call_counter}_{id(args)}"

        fut: asyncio.Future[MCPResult] = asyncio.get_event_loop().create_future()
        self.pending_calls[call_id] = {"future": fut, "name": tool_name, "args": args}

        try:
            result = await asyncio.wait_for(fut, timeout=self._timeout)
            return result.get("content", [])
        except asyncio.TimeoutError:
            return [
                {
                    "type": "text",
                    "text": f"Tool '{tool_name}' execution timeout after "
                            f"{self._timeout}s (no ToolMessage arrived from LangGraph)",
                }
            ]
        finally:
            self.pending_calls.pop(call_id, None)

    # Test-only alias so tests can drive the handler without going through
    # the HTTP transport. Production code path goes through ``_handle_call_tool``
    # registered as MCP ``call_tool``.
    async def _call_tool_for_test(
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

        async def asgi_handler(scope, receive, send):  # noqa: ANN001
            await session_manager.handle_request(scope, receive, send)

        starlette_app = Starlette(
            routes=[Mount("/mcp", app=asgi_handler)],
        )

        # Pick a free port deterministically (bind 0, read assigned, release).
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self._host, 0))
        self._port = sock.getsockname()[1]
        sock.close()

        # Spawn uvicorn as an asyncio task — in-process, no external
        # subprocess.
        import uvicorn

        config = uvicorn.Config(
            app=starlette_app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
            lifespan="off",  # we manage session_manager lifespan ourselves
        )
        self._uvicorn_server = uvicorn.Server(config)

        # The session manager itself is an async context manager; we
        # enter it for the lifetime of the server.
        self._session_manager_cm = session_manager.run()
        await self._session_manager_cm.__aenter__()

        self._server_task = asyncio.create_task(
            self._uvicorn_server.serve(), name=f"proxy-mcp-{self.server_name}"
        )

        # Wait briefly until uvicorn signals it is ready. Uvicorn sets
        # ``started`` on its Server object once the listener is bound.
        for _ in range(50):  # up to 5 seconds, in 100ms slices
            if getattr(self._uvicorn_server, "started", False):
                break
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the listener; cancel any pending Futures with a timeout
        error block. Idempotent."""
        if self._server_task is None:
            return
        # Resolve all pending Futures with a "server stopping" message so
        # awaiting handlers don't leak.
        for call_id, info in list(self.pending_calls.items()):
            fut = info["future"]
            if not fut.done():
                fut.set_result({
                    "content": [{"type": "text", "text": "ProxyMCPServer stopping"}],
                    "is_error": True,
                })
        self.pending_calls.clear()

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

        # Exit the session manager context
        if self._session_manager_cm is not None:
            with contextlib.suppress(Exception):
                await self._session_manager_cm.__aexit__(None, None, None)
            self._session_manager_cm = None

        self._port = None


# ---------------------------------------------------------------------------
# Process-global singleton accessor
# ---------------------------------------------------------------------------
_GLOBAL_PROXY: ProxyMCPServer | None = None


def get_global_proxy_mcp_server(
    tools_metadata: list[ToolMetadata] | None = None,
) -> ProxyMCPServer:
    """Return the process-global :class:`ProxyMCPServer`.

    Lazily constructed on first call. Same instance is returned forever
    after — survives LangGraph's model rebuilds because it lives at
    module scope. If ``tools_metadata`` is provided, the proxy's tool
    set is hot-swapped to that list (no restart).
    """
    global _GLOBAL_PROXY
    if _GLOBAL_PROXY is None:
        _GLOBAL_PROXY = ProxyMCPServer(tools_metadata or [])
    elif tools_metadata is not None:
        _GLOBAL_PROXY.update_tools(tools_metadata)
    return _GLOBAL_PROXY
