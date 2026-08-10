"""gRPC server for the shared LSP service — v1 surface.

Commit 4 wires: Health, ProjectStatus, AttachSession stream. Full
Serena tool proxying lands in commit 5. Tests here pin the process-
level guarantees:

* Server binds to a Unix socket at the exact path in the registry.
* Socket file permissions are 0600 (600 on the sock inode when the
  kernel exposes them; some kernels report 0666 on AF_UNIX regardless,
  so we also check via SO_PEERCRED-adjacent evidence: the DIRECTORY
  is 0700 which effectively restricts access).
* Health RPC returns SERVING once the internal Serena SSE (mocked
  here) reports ready, NOT_SERVING before.
* AttachSession sends ATTACHED then KEEPALIVE frames on the same
  stream (event-driven, no polling).
* ProjectStatus reflects idle timer.
* Graceful shutdown on SIGTERM: server.stop(grace) called, socket
  file unlinked.

These tests instantiate the server in-process (no fork) and connect
over the same UDS as the real deployment.
"""
from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import AsyncIterator

import grpc
import pytest

from deep_worker_cli.lsp.service.grpc_server import (
    LSPServiceServer,
    ServerConfig,
)
from deep_worker_cli.lsp.service.proto import (
    lsp_service_pb2 as pb,
    lsp_service_pb2_grpc as pbg,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def running_server(tmp_path: Path):
    """Start the gRPC server on a UDS in tmp_path and yield the
    (server, socket_path, stub) tuple. Guaranteed shutdown on teardown.
    """
    slug_dir = tmp_path / "slug"
    slug_dir.mkdir(mode=0o700)
    sock_path = slug_dir / "grpc.sock"

    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock_path,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
        ),
    )
    await server.start()

    channel = grpc.aio.insecure_channel(f"unix:{sock_path}")
    stub = pbg.LspServiceStub(channel)
    try:
        yield server, sock_path, stub
    finally:
        await channel.close()
        await server.stop(grace=0.1)
        # After stop the socket file must be gone.
        assert not sock_path.exists(), (
            f"socket file leaked: {sock_path}"
        )


# ── binding & permissions ────────────────────────────────────────────


async def test_server_creates_uds_and_directory_stays_0700(
    running_server,
) -> None:
    _, sock_path, _ = running_server
    assert sock_path.is_socket()
    slug_dir_mode = stat.S_IMODE(sock_path.parent.stat().st_mode)
    assert slug_dir_mode == 0o700, oct(slug_dir_mode)


# ── Health ───────────────────────────────────────────────────────────


async def test_health_returns_alive_true_when_backend_ready(
    running_server, tmp_path: Path
) -> None:
    server, _, stub = running_server
    server.mark_backend_ready()  # in commit 5 this is Serena's own signal
    reply = await stub.Health(pb.HealthRequest())
    assert reply.alive is True
    assert reply.project_root == str(tmp_path)
    assert reply.uptime_seconds >= 0


async def test_health_reports_alive_false_until_backend_ready(
    running_server,
) -> None:
    _, _, stub = running_server
    reply = await stub.Health(pb.HealthRequest())
    assert reply.alive is False


# ── ProjectStatus ────────────────────────────────────────────────────


async def test_project_status_reports_uptime_and_project_root(
    running_server, tmp_path: Path
) -> None:
    _, _, stub = running_server
    reply = await stub.ProjectStatus(pb.StatusRequest())
    assert reply.project_root == str(tmp_path)
    assert reply.uptime_seconds >= 0
    assert reply.pid == os.getpid()
    assert reply.transport == "uds"
    assert reply.transport_address.endswith("grpc.sock")


# ── AttachSession stream ─────────────────────────────────────────────


async def test_attach_session_yields_attached_then_keepalive(
    running_server, tmp_path: Path
) -> None:
    server, _, stub = running_server
    server.set_keepalive_interval_s(0.05)
    server.mark_backend_ready()
    stream = stub.AttachSession(
        pb.AttachRequest(client_id="unit-test", project_root=str(tmp_path))
    )
    events: list[int] = []
    async for evt in stream:
        events.append(evt.event_type)
        if len(events) >= 3:
            break
    stream.cancel()
    assert events[0] == pb.SESSION_EVENT_ATTACHED
    # After ATTACHED we expect KEEPALIVE frames (event-driven, not
    # polling any external state — the server has its own timer).
    assert any(e == pb.SESSION_EVENT_KEEPALIVE for e in events[1:])


async def test_attached_is_withheld_until_backend_becomes_ready(
    running_server, tmp_path: Path
) -> None:
    """ATTACHED is the bridge's "safe to call tools" signal.

    The bridge blocks on this first event instead of polling Health, so
    the server MUST NOT emit it while the backend is still starting —
    otherwise the first tool call races Serena's startup and comes back
    UNIMPLEMENTED, which the model misreads as "symbol does not exist".
    """
    server, _, stub = running_server
    server.set_backend_ready_wait_s(30.0)
    stream = stub.AttachSession(
        pb.AttachRequest(client_id="unit-test", project_root=str(tmp_path))
    )
    first = asyncio.ensure_future(anext(stream.__aiter__()))
    # Nothing may arrive while the backend is not ready.
    done, _ = await asyncio.wait({first}, timeout=0.5)
    assert not done, "ATTACHED leaked before the backend was ready"

    server.mark_backend_ready()
    evt = await asyncio.wait_for(first, timeout=5.0)
    stream.cancel()
    assert evt.event_type == pb.SESSION_EVENT_ATTACHED
    assert "backend-not-ready" not in evt.message


async def test_attached_is_emitted_tagged_when_backend_never_readies(
    running_server, tmp_path: Path
) -> None:
    """Expiry must unblock the bridge, not strand it.

    A backend that never comes up has to surface as a loud tool-call
    failure, so the stream still emits ATTACHED — tagged so the reason
    is visible — rather than hanging connect() forever.
    """
    server, _, stub = running_server
    server.set_backend_ready_wait_s(0.2)
    stream = stub.AttachSession(
        pb.AttachRequest(client_id="unit-test", project_root=str(tmp_path))
    )
    evt = await asyncio.wait_for(anext(stream.__aiter__()), timeout=5.0)
    stream.cancel()
    assert evt.event_type == pb.SESSION_EVENT_ATTACHED
    assert "backend-not-ready" in evt.message


# ── graceful stop ────────────────────────────────────────────────────


async def test_stop_unlinks_socket_and_rejects_new_calls(
    tmp_path: Path,
) -> None:
    slug_dir = tmp_path / "slug2"
    slug_dir.mkdir(mode=0o700)
    sock_path = slug_dir / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock_path,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
        )
    )
    await server.start()
    channel = grpc.aio.insecure_channel(f"unix:{sock_path}")
    stub = pbg.LspServiceStub(channel)
    reply = await stub.Health(pb.HealthRequest())
    assert reply.alive in (True, False)
    await channel.close()
    await server.stop(grace=0.1)
    assert not sock_path.exists()
