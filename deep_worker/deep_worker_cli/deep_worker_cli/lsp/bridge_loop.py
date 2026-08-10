"""stdio MCP loop for the bridge, backed by the shared LSP service.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §6 and
``docs/POLICY_NO_INPROCESS_HEAVY_MCP.md``.

The bridge is the cheap half of the split: it owns no project model and
starts nothing heavy. Every tool call is forwarded over gRPC to the one
shared service for the project.

Hard invariants (pinned by tests):

* **Fail loud.** An unreachable service, a missing backend, or a failing
  tool all raise :class:`ToolCallFailed`. They must never come back as
  an empty result — the model would read "no matches" and conclude the
  symbol does not exist, which is a silent wrong answer.
* **No stdio fallback.** There is no code path that starts a backend
  in-process when the service is down. That is precisely the design the
  policy forbids.
* **This module never spawns the backend.** Only
  ``deep_worker_cli.lsp.service*`` may.
* **Event-driven.** The attach stream is consumed by a task that blocks
  on the gRPC stream; nothing polls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Final

import grpc
from grpc import aio as grpc_aio

from deep_worker_cli.lsp.service.proto import (
    lsp_service_pb2 as pb,
    lsp_service_pb2_grpc as pbg,
)

#: MCP tool name -> (gRPC method name, request type, has ``result`` field).
#: Mirrors the 11 tool RPCs in the proto exactly.
_TOOLS: Final[dict[str, tuple[str, str]]] = {
    "find_symbol": ("FindSymbol", "FindSymbolRequest"),
    "find_referencing_symbols": ("FindReferencingSymbols", "RefRequest"),
    "find_declaration": ("FindDeclaration", "DeclRequest"),
    "find_implementations": ("FindImplementations", "ImplRequest"),
    "get_symbols_overview": ("GetSymbolsOverview", "OverviewRequest"),
    "get_diagnostics_for_file": ("GetDiagnosticsForFile", "DiagRequest"),
    "rename_symbol": ("RenameSymbol", "RenameRequest"),
    "replace_symbol_body": ("ReplaceSymbolBody", "ReplaceRequest"),
    "insert_before_symbol": ("InsertBeforeSymbol", "InsertRequest"),
    "insert_after_symbol": ("InsertAfterSymbol", "InsertRequest"),
    "safe_delete_symbol": ("SafeDeleteSymbol", "DeleteRequest"),
}

_CONNECT_TIMEOUT_S: Final = 10.0
_CALL_TIMEOUT_S: Final = 120.0

#: Maximum time the bridge is willing to wait for the first event of
#: the AttachSession stream. The service deliberately delays ATTACHED
#: until the backend (Serena) finishes initializing, so this ceiling
#: covers both session registration and backend startup (~4–8 s with
#: Serena; padded generously).
#:
#: The wait is strictly event-driven: the bridge blocks on an
#: ``asyncio.Event`` that the stream reader sets. No health polling —
#: the project policy forbids polling (``test_no_polling_policy*.py``).
_ATTACH_READY_S: Final = 75.0


def tool_names() -> list[str]:
    """MCP tool names this bridge exposes."""
    return sorted(_TOOLS)


class BridgeUnreachable(RuntimeError):
    """The shared service could not be reached. Terminal by design."""


class ToolCallFailed(RuntimeError):
    """A tool call did not produce a usable result.

    Raised for transport failures, missing backends and backend
    exceptions alike. Callers must surface this as an MCP error rather
    than substituting an empty result.
    """


class BridgeSession:
    """One agent's connection to the shared service."""

    def __init__(self, socket_path: Path | str) -> None:
        self._socket_path = Path(socket_path)
        self._channel: grpc_aio.Channel | None = None
        self._stub: pbg.LspServiceStub | None = None
        self._attach_task: asyncio.Task[None] | None = None
        self._attached = asyncio.Event()

    # ── lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the channel and hold an AttachSession stream.

        The stream is what tells the service an agent is live, so its
        idle-shutdown timer cannot fire underneath us.
        """
        if not self._socket_path.exists():
            raise BridgeUnreachable(
                f"shared LSP service socket not found: {self._socket_path}. "
                f"Start it with: deep-worker lsp serve"
            )
        channel = grpc_aio.insecure_channel(f"unix:{self._socket_path}")
        try:
            await asyncio.wait_for(
                channel.channel_ready(), timeout=_CONNECT_TIMEOUT_S
            )
        except (TimeoutError, grpc.RpcError) as e:
            await channel.close()
            raise BridgeUnreachable(
                f"shared LSP service at {self._socket_path} did not accept a "
                f"connection: {e}"
            ) from e
        self._channel = channel
        self._stub = pbg.LspServiceStub(channel)
        self._attach_task = asyncio.create_task(self._hold_attach_stream())
        # Block on the first stream event. The service holds ATTACHED
        # back until the backend is actually ready, so this one wait
        # covers both "session registered" (keep-alive is in effect)
        # and "backend online" (tool calls will not race Serena's
        # ~5s startup and come back UNIMPLEMENTED, which the model
        # would misread as "symbol does not exist").
        #
        # Purely event-driven — no Health poll loop. If the backend
        # never comes up, the service still emits ATTACHED with a
        # ``backend-not-ready`` marker, so we unblock here and tool
        # calls fail loudly with the real reason.
        try:
            await asyncio.wait_for(self._attached.wait(), timeout=_ATTACH_READY_S)
        except TimeoutError as e:
            await self.close()
            raise BridgeUnreachable(
                f"shared LSP service at {self._socket_path} never confirmed "
                f"the attach stream"
            ) from e

    async def _hold_attach_stream(self) -> None:
        """Consume the attach stream for the session's lifetime.

        Blocks on the stream (event-driven); the first event flips
        ``_attached``. Keepalives arrive on the same stream.
        """
        stub = self._stub
        if stub is None:  # pragma: no cover — connect() sets it first
            return
        try:
            stream = stub.AttachSession(pb.AttachRequest(client_id="bridge"))
            async for _event in stream:
                self._attached.set()
        except asyncio.CancelledError:
            raise
        except grpc.RpcError:
            # Service went away; unblock connect() so it can fail loudly
            # rather than hang.
            self._attached.set()

    async def close(self) -> None:
        if self._attach_task is not None:
            self._attach_task.cancel()
            try:
                await self._attach_task
            except (asyncio.CancelledError, Exception):
                pass
            self._attach_task = None
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
        self._stub = None

    # ── tool proxying ────────────────────────────────────────────────

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Forward one MCP tool call to the service over gRPC.

        Returns the decoded ``content_json``. Raises
        :class:`ToolCallFailed` on any failure — never an empty result.
        """
        if self._stub is None:
            raise ToolCallFailed(
                f"{name}: bridge is not connected to the shared LSP service"
            )
        try:
            rpc_name, request_type = _TOOLS[name]
        except KeyError:
            raise ToolCallFailed(f"unknown tool {name!r}") from None

        args_json = json.dumps(arguments or {})
        request = getattr(pb, request_type)(args_json=args_json)
        method = getattr(self._stub, rpc_name)

        try:
            reply = await method(request, timeout=_CALL_TIMEOUT_S)
        except grpc.RpcError as e:
            code = getattr(e, "code", lambda: "UNKNOWN")()
            detail = getattr(e, "details", lambda: str(e))()
            raise ToolCallFailed(f"{name}: {code} {detail}") from e

        result = reply.result
        if result.is_error:
            raise ToolCallFailed(
                f"{name}: {result.error_message or 'tool reported an error'}"
            )
        if not result.content_json:
            return None
        try:
            return json.loads(result.content_json)
        except json.JSONDecodeError as e:
            raise ToolCallFailed(
                f"{name}: service returned malformed JSON ({e})"
            ) from e
