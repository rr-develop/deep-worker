"""End-to-end integration tests for AsyncioServer migration.

This module tests that AgentProcess with AsyncioServer can:
- Start successfully
- Accept client connections
- Handle basic RPC calls
- Shutdown gracefully

Lifecycle note
--------------
``AgentProcess.start()`` intentionally enters its own event loop and
blocks on ``shutdown_event.wait()`` — that is the agent's main loop
and is how the production CLI is designed. A test that does
``await agent.start()`` directly would therefore never return. Use
``_start_agent_in_background(agent)`` below: it schedules ``start()``
as a task, waits until ``rpyc_server`` is ready, and returns the task
so the caller can later trigger shutdown and await the task to drain
the main-loop coroutine cleanly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from deep_worker_cli.process.agent_process import AgentProcess

async def _start_agent_in_background(
    agent: "AgentProcess",
    *,
    ready_timeout: float = 15.0,
    poll_interval: float = 0.05,
) -> asyncio.Task[None]:
    """Launch ``agent.start()`` as a background task and wait for readiness.

    ``AgentProcess.start()`` blocks until ``shutdown_event`` is set, so a
    test MUST NOT ``await`` it directly. Instead we schedule it and poll
    for the two observable readiness signals: ``running is True`` AND
    ``rpyc_server is not None`` (and ``rpyc_port`` assigned when ``0``
    was requested). Polling here is a test-only concession — the
    production code is event-driven.
    """
    start_task: asyncio.Task[None] = asyncio.create_task(agent.start())
    deadline = asyncio.get_event_loop().time() + ready_timeout
    while True:
        if start_task.done():
            # start() returned / raised BEFORE we observed readiness —
            # surface the error to the caller, don't swallow it.
            start_task.result()
            raise RuntimeError("agent.start() returned before readiness signal")
        if agent.running and agent.rpyc_server is not None and (
            agent.rpyc_port is None or agent.rpyc_port > 0
        ):
            return start_task
        if asyncio.get_event_loop().time() > deadline:
            start_task.cancel()
            with suppress_cancelled():
                await start_task
            raise asyncio.TimeoutError(
                f"agent {agent.name!r} did not signal readiness within "
                f"{ready_timeout}s",
            )
        await asyncio.sleep(poll_interval)

async def _stop_agent(
    agent: "AgentProcess", start_task: asyncio.Task[None],
) -> None:
    """Trigger shutdown and drain the main-loop task."""
    await agent.shutdown()
    # agent.start()'s main-loop await will have returned now; drain it.
    try:
        await asyncio.wait_for(start_task, timeout=10.0)
    except asyncio.TimeoutError:
        start_task.cancel()
        with suppress_cancelled():
            await start_task

from contextlib import contextmanager
from typing import Iterator

@contextmanager
def suppress_cancelled() -> Iterator[None]:
    try:
        yield
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 — best-effort drain
        pass

@pytest.mark.asyncio
async def test_agent_process_starts_with_asyncio_server():
    """Test that AgentProcess can start with AsyncioServer."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from pathlib import Path
    import tempfile

    # Create temp directory for test agent
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-asyncio-agent",
            config=config,
            rpyc_port=0,  # Auto-assign port
        )

        start_task = await _start_agent_in_background(agent)
        try:
            # Verify server is running
            assert agent.rpyc_server is not None, "RPyC server should be initialized"
            assert agent.rpyc_port > 0, "RPyC port should be assigned"

            # Verify it's AsyncioServer
            from rpyc.utils.async_server import AsyncioServer

            assert isinstance(
                agent.rpyc_server, AsyncioServer
            ), "Server should be AsyncioServer instance"

        finally:
            await _stop_agent(agent, start_task)

_REDUNDANT_SKIP_REASON = (
    "Same-process AgentProcess + async_connect topology is forbidden "
    "(see docs/RPYC_NETREF_DEADLOCK_ANALYSIS.md). The protocol surface "
    "this test was checking is now covered by the cross-process "
    "test_agent_service_asyncio_e2e.py in this same directory (server "
    "in a child Python process). End-to-end coverage with a real "
    "deep-worker subprocess lives in the slow tier "
    "tests/integration_ai/test_agent_process_lifecycle.py."
)

@pytest.mark.skip(reason=_REDUNDANT_SKIP_REASON)
@pytest.mark.asyncio
async def test_client_can_connect_to_asyncio_server():
    """Replaced by cross-process test_agent_service_asyncio_e2e.py."""

@pytest.mark.skip(reason=_REDUNDANT_SKIP_REASON)
@pytest.mark.asyncio
async def test_multiple_sequential_calls():
    """Replaced by cross-process test_agent_service_asyncio_e2e.py."""

@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Test that AsyncioServer shuts down gracefully."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-shutdown-agent",
            config=config,
            rpyc_port=0,
        )

        # Start and immediately shutdown
        start_task = await _start_agent_in_background(agent)
        assert agent.rpyc_server is not None

        # Shutdown should complete without errors
        await _stop_agent(agent, start_task)

        # Server should be cleaned up
        # (We don't check rpyc_server is None because shutdown might not clear it)
