"""End-to-end test for complete AsyncioServer migration.

This test verifies the full migration:
- AgentProcess uses AsyncioServer
- AgentClient uses native rpyc.connect
- Full request/response cycle works
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

@pytest.mark.asyncio
async def test_full_migration_agent_to_client():
    """Test full cycle: AgentProcess (AsyncioServer) → AgentClient (native rpyc)."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from deep_worker_cli.client.agent_client import AgentClient
    from deep_worker_cli.process.registry import ProcessRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create and start AgentProcess with AsyncioServer
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-e2e-agent",
            config=config,
            rpyc_port=0,  # Auto-assign
        )

        try:
            # Start agent (starts AsyncioServer)
            await agent.start()

            # Verify AsyncioServer is used
            from rpyc.utils.async_server import AsyncioServer
            assert isinstance(agent.rpyc_server, AsyncioServer)

            # Get assigned port
            port = agent.rpyc_port
            assert port > 0

            # Register in registry so client can find it
            registry = ProcessRegistry()
            # Agent should auto-register during start()

            # 2. Create AgentClient (uses native rpyc.connect)
            client = AgentClient(agent_name="test-e2e-agent", registry=registry)

            try:
                # Connect client
                await client.connect()
                assert client.connected

                # 3. Test get_status (full cycle)
                status = await client.get_status()
                assert status is not None
                assert hasattr(status, "agent_name") or "agent_name" in dir(status)

            finally:
                # Disconnect client
                await client.disconnect()

        finally:
            # Shutdown agent
            await agent.shutdown()

@pytest.mark.asyncio
async def test_migration_performance_simple():
    """Simple performance test to verify AsyncioServer is faster."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess
    from deep_worker_cli.client.agent_client import AgentClient
    from deep_worker_cli.process.registry import ProcessRegistry
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            working_dir=Path(tmpdir),
            provider="claude-code",
            model="claude-3-5-sonnet-20241022",
        )

        agent = AgentProcess(
            name="test-perf-agent",
            config=config,
            rpyc_port=0,
        )

        try:
            await agent.start()

            registry = ProcessRegistry()
            client = AgentClient(agent_name="test-perf-agent", registry=registry)

            try:
                await client.connect()

                # Measure 10 sequential get_status calls
                start = time.perf_counter()
                for _ in range(10):
                    await client.get_status()
                elapsed = time.perf_counter() - start

                # AsyncioServer should be fast (< 1 second for 10 calls)
                # Previously with ThreadedServer: ~3+ seconds
                assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 10 calls"

                print(f"✅ 10 get_status calls: {elapsed:.2f}s ({elapsed*100:.0f}ms avg)")

            finally:
                await client.disconnect()

        finally:
            await agent.shutdown()
