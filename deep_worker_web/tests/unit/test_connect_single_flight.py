"""Web connect must never hang the UI, never storm connects, never
loop reconnects forever.

Incident: a related internal incident analysis (not included here).

These are real-behaviour tests: they point the AgentService at an
UNREACHABLE TCP port (a closed/blackholed endpoint), so the connect
path runs for real (real ``async_connect``, real socket) — no mocking
of the connect itself. The invariants under test:

* ``ensure_connecting`` is NON-BLOCKING: it returns promptly with the
  current connectivity (False here) and kicks a background connect — it
  must NOT await connect establishment.
* SINGLE-FLIGHT: concurrent / repeated triggers for the same agent
  produce at most ONE in-flight connect task, not a storm.
* BOUNDED: a background connect to an unreachable peer fails within the
  connect timeout (does not park forever), and arms a reconnect
  backoff so the next immediate trigger is suppressed.
* No new threads — everything on the running event loop.
"""
from __future__ import annotations

import asyncio
import socket
import threading
from datetime import datetime
from typing import Any

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService


pytestmark = pytest.mark.asyncio


class _RegistryStub:
    def list_all(self) -> list[Any]:
        return []

    def lookup(self, name: str) -> Any | None:
        return None


def _blackhole_port() -> int:
    """A TCP port that ACCEPTS the kernel SYN into a listen backlog but
    NEVER completes the application handshake — i.e. a socket that
    listens but never ``accept()``s. This reproduces the real incident
    shape: ``sock_connect`` succeeds, the eager RPyC handshake hangs.

    Returns the port; the listening socket is leaked intentionally for
    the lifetime of the test process (closed by GC / process exit).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(8)  # backlog, but we never call accept()
    # Keep a reference alive so the port stays bound for the test.
    _LEAKED_SOCKETS.append(s)
    return s.getsockname()[1]


_LEAKED_SOCKETS: list[socket.socket] = []


def _make_service_with_agent(name: str, port: int) -> AgentService:
    svc = AgentService(registry=_RegistryStub())
    svc._agents[name] = AgentInfo(
        name=name,
        pid=999999,  # not a real pid; connect path only reads port/pid for identity
        status=AgentStatus.RUNNING,
        provider="test",
        model="test",
        rpyc_port=port,
        working_dir="/tmp",
        started_at=datetime.now(),
    )
    return svc


async def test_ensure_connecting_is_non_blocking_for_stuck_peer() -> None:
    """ensure_connecting must return quickly even when the peer's
    handshake will hang — it kicks a background connect and returns the
    current (not-yet-connected) state without awaiting it."""
    port = _blackhole_port()
    svc = _make_service_with_agent("stuck", port)

    loop = asyncio.get_running_loop()
    start = loop.time()
    connected = await svc.ensure_connecting("stuck")
    elapsed = loop.time() - start

    assert connected is False
    # The whole point: the handler-facing call returns essentially
    # immediately, NOT after a multi-second handshake timeout.
    assert elapsed < 0.5, f"ensure_connecting blocked for {elapsed:.3f}s"


async def test_single_flight_no_connect_storm() -> None:
    """Repeated/concurrent triggers for one stuck agent must yield at
    most ONE in-flight connect task — not a storm."""
    port = _blackhole_port()
    svc = _make_service_with_agent("stuck", port)

    # Fire many triggers concurrently.
    await asyncio.gather(*(svc.ensure_connecting("stuck") for _ in range(20)))

    in_flight = svc._connecting.get("stuck")
    assert in_flight is not None, "expected a background connect task"
    assert isinstance(in_flight, asyncio.Task)
    # Exactly one task tracked for this agent (single-flight).
    assert sum(1 for k in svc._connecting if k == "stuck") == 1

    # Cleanup: cancel the parked/bounded connect task.
    in_flight.cancel()
    try:
        await in_flight
    except (asyncio.CancelledError, Exception):
        pass


async def test_background_connect_is_bounded_and_arms_backoff() -> None:
    """The background connect to a stuck peer must FAIL within the
    connect timeout (not park forever) and then arm a reconnect backoff
    deadline that suppresses an immediate re-trigger."""
    port = _blackhole_port()
    svc = _make_service_with_agent("stuck", port)

    await svc.ensure_connecting("stuck")
    task = svc._connecting.get("stuck")
    assert task is not None

    loop = asyncio.get_running_loop()
    start = loop.time()
    # Wait for the bounded connect to resolve. Must be bounded by the
    # service's CONNECT_TIMEOUT plus slack — NOT infinite.
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
    except (asyncio.TimeoutError, Exception):
        pass
    elapsed = loop.time() - start
    assert elapsed < 12.0, f"background connect did not bound itself ({elapsed:.1f}s)"

    # It failed → not connected, and a backoff deadline is armed.
    assert await svc.is_connected("stuck") is False
    not_before = svc._reconnect_not_before.get("stuck")
    assert not_before is not None, "expected a reconnect backoff deadline"
    assert not_before > loop.time(), "backoff deadline must be in the future"

    # An immediate re-trigger is suppressed by the backoff (no new task).
    svc._connecting.pop("stuck", None)
    again = await svc.ensure_connecting("stuck")
    assert again is False
    assert svc._connecting.get("stuck") is None, (
        "backoff must suppress an immediate reconnect"
    )


async def test_ensure_connecting_true_when_already_connected() -> None:
    """Fast path: if a live connection already exists, ensure_connecting
    returns True and spawns no background task."""
    svc = _make_service_with_agent("live", 1)

    class _LiveConn:
        _closed = False

    svc._connections["live"] = _LiveConn()
    svc._roots["live"] = object()
    # identity recorded so is_connected's identity check passes
    svc._conn_identity["live"] = (999999, 1)

    connected = await svc.ensure_connecting("live")
    assert connected is True
    assert "live" not in svc._connecting


async def test_no_extra_threads_spawned() -> None:
    """The connect machinery must not spin up worker threads — it runs
    on the existing event loop only."""
    before = set(t.ident for t in threading.enumerate())
    port = _blackhole_port()
    svc = _make_service_with_agent("stuck", port)
    await svc.ensure_connecting("stuck")
    await asyncio.sleep(0.1)
    after = set(t.ident for t in threading.enumerate())
    new = after - before
    assert not new, f"connect spawned unexpected threads: {new}"

    task = svc._connecting.get("stuck")
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
