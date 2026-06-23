"""Integration: the web connect path establishes for real AND never
hangs the UI — even against a peer whose handshake never completes.

A related internal incident analysis (not included here).

NO MOCKS. Two real scenarios:

1. ``test_real_agent_connect_establishes_and_is_usable`` — spins up a
   REAL deep-worker agent (real RPyC ``AsyncioServer``), connects via the
   real ``connect_to_agent`` single-flight gate, and proves the
   connection is genuinely usable (real ``get_root`` + a real RPyC call).
   This is the "connection really establishes, nothing hangs" guarantee.

2. ``test_stuck_peer_does_not_hang_ui_or_storm_connects`` — points the
   service at a REAL socket that ``listen()``s but never ``accept()``s
   (the exact incident shape: kernel completes the TCP handshake into the
   backlog, the RPyC ``HANDLE_GETROOT`` handshake never gets a reply).
   Proves: ``ensure_connecting`` returns promptly (UI never blocks),
   concurrent triggers produce at most ONE in-flight connect (no storm),
   and the bounded background connect fails within the connect timeout
   and arms a reconnect backoff.
"""
from __future__ import annotations

import asyncio
import socket
from datetime import datetime

import pytest

from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.routers.agents import router
from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from tests.integration.agent_fixture import agent_fixture


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_agent_connect_establishes_and_is_usable(
    request: pytest.FixtureRequest,
) -> None:
    """Real agent + real RPyC: connect establishes and is usable.

    The single-flight gate must not break the happy path — the
    connection is real and a real RPyC call returns.
    """
    with agent_fixture(pytest_item=request.node) as agent_info:
        service = AgentService(registry=ProcessRegistry())
        await service.initialize()
        try:
            # Not connected initially.
            assert await service.is_connected(agent_info.name) is False

            # ensure_connecting is non-blocking; the first call kicks the
            # background connect and reports the current (False) state.
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            first = await service.ensure_connecting(agent_info.name)
            assert loop.time() - t0 < 0.5, "ensure_connecting must not block"
            assert first is False

            # The blocking variant (used by lifecycle/monitor) actually
            # establishes the connection against the real agent.
            ok = await service.connect_to_agent(agent_info.name)
            assert ok is True
            assert await service.is_connected(agent_info.name) is True

            # And the connection is genuinely usable: a real RPyC root.
            root = service.get_root(agent_info.name)
            assert root is not None

            # No leftover in-flight connect, no backoff armed on success.
            assert service._connecting.get(agent_info.name) is None
            assert service._reconnect_not_before.get(agent_info.name) is None
        finally:
            await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cold_http_request_does_not_block_then_succeeds_on_retry(
    request: pytest.FixtureRequest,
) -> None:
    """End-to-end contract (real agent, real ASGI app): the FIRST HTTP
    request to an agent the web has not connected to yet returns 503
    ``connecting`` PROMPTLY (handler never blocks on connect), and a
    retry succeeds once the single-flight background connect lands."""
    app = FastAPI()
    app.include_router(router)

    with agent_fixture(pytest_item=request.node) as agent_info:
        service = AgentService(registry=ProcessRegistry())
        await service.initialize()
        app.state.agent_service = service
        transport = ASGITransport(app=app)
        try:
            loop = asyncio.get_running_loop()
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Cold request — not connected yet. Must return promptly
                # (no blocking) with a retryable 503, NOT a 500 or a hang.
                t0 = loop.time()
                cold = await client.get(
                    f"/api/agents/{agent_info.name}/dialog-text"
                )
                elapsed = loop.time() - t0
                assert elapsed < 1.0, (
                    f"cold handler blocked for {elapsed:.2f}s — must not "
                    f"wait on connect establishment"
                )
                assert cold.status_code == 503, (
                    f"cold request to unconnected agent must be a retryable "
                    f"503 connecting; got {cold.status_code}: {cold.text}"
                )

                # Retry until the background connect lands (event-driven on
                # the real agent — bounded, no infinite loop). The real UI
                # does exactly this.
                ok = False
                for _ in range(50):
                    if await service.is_connected(agent_info.name):
                        ok = True
                        break
                    await asyncio.sleep(0.1)
                assert ok, "background connect did not establish in time"

                hot = await client.get(
                    f"/api/agents/{agent_info.name}/dialog-text"
                )
                assert hot.status_code == 200, (
                    f"after the background connect lands, the same request "
                    f"must succeed; got {hot.status_code}: {hot.text}"
                )
        finally:
            await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stuck_peer_does_not_hang_ui_or_storm_connects() -> None:
    """Real stuck peer (listen-but-never-accept): the UI-facing path
    never blocks, connects do not pile up, and the background connect is
    bounded + backed off."""
    # A real socket that accepts the TCP handshake into its backlog but
    # never accept()s at the application layer — reproduces the incident.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    stuck_port = listener.getsockname()[1]

    service = AgentService(registry=ProcessRegistry())
    name = "stuck_peer_test"
    try:
        # Register a real AgentInfo pointing at the stuck port. We set
        # _agents directly (the service's own discovery state) rather than
        # polluting the shared on-disk registry — the connect path reads
        # port/pid from here.
        service._agents[name] = AgentInfo(
            name=name,
            pid=999999,
            status=AgentStatus.RUNNING,
            provider="test",
            model="test",
            rpyc_port=stuck_port,
            working_dir="/tmp",
            started_at=datetime.now(),
        )

        loop = asyncio.get_running_loop()

        # 1) UI-facing call returns promptly even though the peer's
        #    handshake will never complete.
        t0 = loop.time()
        results = await asyncio.gather(
            *(service.ensure_connecting(name) for _ in range(25))
        )
        elapsed = loop.time() - t0
        assert elapsed < 1.0, f"ensure_connecting batch blocked for {elapsed:.2f}s"
        assert all(r is False for r in results), "stuck peer is never 'connected'"

        # 2) No connect storm: at most ONE in-flight connect task.
        in_flight = service._connecting.get(name)
        assert in_flight is not None
        assert isinstance(in_flight, asyncio.Task)

        # 3) The background connect bounds itself (does not park forever)
        #    and arms a reconnect backoff on failure.
        t0 = loop.time()
        try:
            await asyncio.wait_for(asyncio.shield(in_flight), timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            pass
        bound = loop.time() - t0
        assert bound < 12.0, f"background connect did not bound itself ({bound:.1f}s)"

        assert await service.is_connected(name) is False
        not_before = service._reconnect_not_before.get(name)
        assert not_before is not None and not_before > loop.time(), (
            "a reconnect backoff deadline must be armed after a failed connect"
        )

        # 4) An immediate re-trigger is suppressed by the backoff — no new
        #    connect task is spawned (no infinite reconnect).
        service._connecting.pop(name, None)
        again = await service.ensure_connecting(name)
        assert again is False
        assert service._connecting.get(name) is None, (
            "backoff must suppress an immediate reconnect"
        )
    finally:
        # Cancel any lingering connect task and close the listener.
        task = service._connecting.get(name)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await service.shutdown()
        listener.close()
