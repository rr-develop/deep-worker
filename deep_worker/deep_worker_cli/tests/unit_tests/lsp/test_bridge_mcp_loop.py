"""Tests for the bridge's stdio MCP loop (DESIGN §6, commit 5b).

Commit 4 shipped the bridge as auto-start + a reachability check.
Commit 5b turns it into a real MCP surface that forwards every tool
call to the shared service over gRPC.

The invariant that matters most here is **fail-loud**. If the service
is unreachable, or a tool RPC fails, the bridge must raise. Returning
an empty result instead would be read by the model as "this symbol
does not exist" — a silent wrong answer, which is far worse than a
visible error.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deep_worker_cli.lsp.bridge_loop import (
    BridgeSession,
    ToolCallFailed,
    tool_names,
)
from deep_worker_cli.lsp.service.grpc_server import (
    LSPServiceServer,
    ServerConfig,
)
from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb


class _Service:
    """A real gRPC service on a real UDS — no mocks."""

    def __init__(self, server: LSPServiceServer, socket_path: Path) -> None:
        self._server = server
        self.socket_path = socket_path

    def attached_clients(self) -> int:
        return self._server._attached_clients

    def set_tool_handler(self, handler) -> None:
        self._server.set_tool_handler(handler)


@pytest.fixture()
async def running_service(tmp_path: Path):
    sock = tmp_path / "svc" / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            slug="test",
        )
    )
    await server.start()
    server.mark_backend_ready()
    try:
        yield _Service(server, sock)
    finally:
        await server.stop(grace=0.1)


# ── tool surface ─────────────────────────────────────────────────────


def test_tool_names_match_the_service_surface():
    """The 11 tool RPCs from the proto must be exposed 1:1."""
    assert set(tool_names()) == {
        "find_symbol",
        "find_referencing_symbols",
        "find_declaration",
        "find_implementations",
        "get_symbols_overview",
        "get_diagnostics_for_file",
        "rename_symbol",
        "replace_symbol_body",
        "insert_before_symbol",
        "insert_after_symbol",
        "safe_delete_symbol",
    }


# ── forwarding ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_call_is_forwarded_over_grpc(running_service):
    """A tool call must reach the service and return its ToolResult."""
    seen: list[tuple[str, str]] = []

    async def handler(rpc: str, args_json: str) -> pb.ToolResult:
        seen.append((rpc, args_json))
        return pb.ToolResult(content_json=json.dumps({"symbols": ["Foo"]}))

    running_service.set_tool_handler(handler)

    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        result = await session.call_tool("find_symbol", {"name_path": "Foo"})
    finally:
        await session.close()

    assert seen == [("FindSymbol", json.dumps({"name_path": "Foo"}))]
    assert result == {"symbols": ["Foo"]}


@pytest.mark.asyncio
async def test_tool_arguments_are_forwarded_verbatim(running_service):
    """args_json must carry the caller's arguments unchanged.

    Dropping them would make every call look like a no-argument call —
    which the backend would answer with plausible-but-wrong results.
    """
    seen: list[str] = []

    async def handler(rpc: str, args_json: str) -> pb.ToolResult:
        seen.append(args_json)
        return pb.ToolResult(content_json="{}")

    running_service.set_tool_handler(handler)

    args = {"name_path": "Foo/bar", "depth": 2, "include_body": True}
    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        await session.call_tool("find_symbol", args)
    finally:
        await session.close()

    assert json.loads(seen[0]) == args


# ── fail-loud ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_failure_raises_instead_of_empty_result(running_service):
    """A failing tool must raise — never return an empty result."""

    async def boom(rpc: str, args_json: str) -> pb.ToolResult:
        raise RuntimeError("backend exploded")

    running_service.set_tool_handler(boom)

    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        with pytest.raises(ToolCallFailed) as ei:
            await session.call_tool("find_symbol", {"name_path": "Foo"})
    finally:
        await session.close()
    assert "exploded" in str(ei.value) or "failed" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_tool_result_error_flag_raises_not_returns_empty(running_service):
    """A ToolResult carrying ``is_error`` must raise.

    This is the subtle one: the RPC itself succeeds, so there is no gRPC
    error to notice. If the bridge only checked the transport it would
    hand the model an empty result for a tool that actually failed —
    and the model would conclude the symbol does not exist.
    """

    async def failing(rpc: str, args_json: str) -> pb.ToolResult:
        return pb.ToolResult(
            is_error=True, error_message="language server not ready"
        )

    running_service.set_tool_handler(failing)

    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        with pytest.raises(ToolCallFailed) as ei:
            await session.call_tool("find_symbol", {"name_path": "Foo"})
    finally:
        await session.close()
    assert "language server not ready" in str(ei.value)


@pytest.mark.asyncio
async def test_malformed_content_json_raises_not_returns_none(running_service):
    """Undecodable payload is a failure, not an absence of results."""

    async def garbage(rpc: str, args_json: str) -> pb.ToolResult:
        return pb.ToolResult(content_json="{not valid json")

    running_service.set_tool_handler(garbage)

    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        with pytest.raises(ToolCallFailed) as ei:
            await session.call_tool("find_symbol", {"name_path": "Foo"})
    finally:
        await session.close()
    assert "malformed" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_missing_backend_raises_unimplemented(running_service):
    """No tool handler attached => UNIMPLEMENTED, not an empty result."""
    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        with pytest.raises(ToolCallFailed) as ei:
            await session.call_tool("find_symbol", {"name_path": "Foo"})
    finally:
        await session.close()
    assert "UNIMPLEMENTED" in str(ei.value).upper() or "backend" in str(ei.value)


@pytest.mark.asyncio
async def test_unreachable_service_fails_loudly_without_fallback(tmp_path: Path):
    """No stdio fallback: a dead socket must raise, not degrade."""
    session = BridgeSession(tmp_path / "nonexistent.sock")
    with pytest.raises(Exception) as ei:
        await session.connect()
    assert "fallback" not in str(ei.value).lower()


@pytest.mark.asyncio
async def test_connect_blocks_until_the_backend_is_ready(tmp_path: Path):
    """``connect()`` must not return while the backend is still starting.

    The bridge is event-driven: it blocks on the first AttachSession
    event, which the service withholds until the backend is up. If
    connect() returned early, the caller's first tool call would race
    Serena's ~5s startup and come back UNIMPLEMENTED — which the model
    reads as "the symbol does not exist", a silent wrong answer.
    """
    sock = tmp_path / "svc" / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            slug="test",
        )
    )
    await server.start()
    server.set_backend_ready_wait_s(30.0)
    session = BridgeSession(sock)
    try:
        connecting = asyncio.ensure_future(session.connect())
        done, _ = await asyncio.wait({connecting}, timeout=1.0)
        assert not done, "connect() returned before the backend was ready"

        server.mark_backend_ready()
        await asyncio.wait_for(connecting, timeout=10.0)
    finally:
        await session.close()
        await server.stop(grace=0.1)


# ── session lifetime ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_holds_attach_stream_for_its_lifetime(running_service):
    """While connected, the service must count us as an attached client
    so its idle-shutdown timer cannot fire underneath a live agent."""
    session = BridgeSession(running_service.socket_path)
    await session.connect()
    try:
        for _ in range(50):
            if running_service.attached_clients() >= 1:
                break
            await asyncio.sleep(0.02)
        assert running_service.attached_clients() >= 1
    finally:
        await session.close()

    for _ in range(50):
        if running_service.attached_clients() == 0:
            break
        await asyncio.sleep(0.02)
    assert running_service.attached_clients() == 0
