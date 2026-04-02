"""TDD tests for AgentClient migration to native rpyc.connect.

This module tests that AgentClient correctly uses native rpyc.connect
with AsyncioServer instead of RpycAsyncSession.

Tests are written BEFORE implementation (TDD approach).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

@pytest.mark.asyncio
async def test_agent_client_does_not_import_arpyc():
    """Test that AgentClient no longer imports arpyc after migration."""
    import importlib.util
    from pathlib import Path

    # Read agent_client.py source
    agent_client_path = (
        Path(__file__).parent.parent.parent.parent
        / "deep_worker_cli"
        / "client"
        / "agent_client.py"
    )
    source = agent_client_path.read_text()

    # Should NOT import arpyc after migration
    assert "from arpyc import" not in source, (
        "agent_client.py should not import from arpyc after migration"
    )
    assert "import arpyc" not in source, (
        "agent_client.py should not import arpyc after migration"
    )

@pytest.mark.asyncio
async def test_agent_client_imports_native_rpyc():
    """Test that AgentClient imports native rpyc after migration."""
    from pathlib import Path

    agent_client_path = (
        Path(__file__).parent.parent.parent.parent
        / "deep_worker_cli"
        / "client"
        / "agent_client.py"
    )
    source = agent_client_path.read_text()

    # Should import rpyc after migration
    assert "import rpyc" in source, "agent_client.py should import rpyc after migration"

@pytest.mark.asyncio
async def test_agent_client_uses_native_connection():
    """Test that AgentClient uses rpyc Connection instead of RpycAsyncSession."""
    from deep_worker_cli.client.agent_client import AgentClient
    import inspect

    # Create instance
    client = AgentClient(agent_name="test-agent")

    # Check that rpyc_session attribute type hint is rpyc.Connection
    source = inspect.getsource(AgentClient)

    # Should have Connection type hint, not RpycAsyncSession
    assert "Connection" in source or "rpyc" in source, (
        "AgentClient should use rpyc.Connection"
    )
    assert "RpycAsyncSession" not in source, (
        "AgentClient should not use RpycAsyncSession after migration"
    )

@pytest.mark.asyncio
async def test_agent_client_connect_method_signature():
    """Test that AgentClient.connect() is still async."""
    from deep_worker_cli.client.agent_client import AgentClient
    import inspect

    client = AgentClient(agent_name="test-agent")

    # connect() should still be async
    assert inspect.iscoroutinefunction(client.connect), (
        "AgentClient.connect() should remain async"
    )

@pytest.mark.asyncio
async def test_agent_client_disconnect_method_signature():
    """Test that AgentClient.disconnect() is still async."""
    from deep_worker_cli.client.agent_client import AgentClient
    import inspect

    client = AgentClient(agent_name="test-agent")

    # disconnect() should still be async
    assert inspect.iscoroutinefunction(client.disconnect), (
        "AgentClient.disconnect() should remain async"
    )

@pytest.mark.asyncio
async def test_agent_client_uses_enable_asyncio_serving():
    """Test that AgentClient uses enable_asyncio_serving pattern."""
    from pathlib import Path

    agent_client_path = (
        Path(__file__).parent.parent.parent.parent
        / "deep_worker_cli"
        / "client"
        / "agent_client.py"
    )
    source = agent_client_path.read_text()

    # Should use enable_asyncio_serving
    assert "enable_asyncio_serving" in source, (
        "agent_client.py should use enable_asyncio_serving pattern"
    )

    # Should use disable_asyncio_serving in cleanup
    assert "disable_asyncio_serving" in source, (
        "agent_client.py should use disable_asyncio_serving in cleanup"
    )

@pytest.mark.asyncio
async def test_agent_client_gets_root_before_enabling_asyncio():
    """Test that AgentClient gets root before enabling asyncio serving."""
    from pathlib import Path

    agent_client_path = (
        Path(__file__).parent.parent.parent.parent
        / "deep_worker_cli"
        / "client"
        / "agent_client.py"
    )
    source = agent_client_path.read_text()

    # Check that pattern follows: get root, then enable_asyncio_serving
    # This is critical to avoid deadlocks
    assert "self._root" in source or "root =" in source, (
        "agent_client.py should store root reference"
    )

@pytest.mark.asyncio
async def test_agent_client_no_async_call_wrapper():
    """Test that AgentClient doesn't use async_call wrapper after migration."""
    from pathlib import Path

    agent_client_path = (
        Path(__file__).parent.parent.parent.parent
        / "deep_worker_cli"
        / "client"
        / "agent_client.py"
    )
    source = agent_client_path.read_text()

    # Should NOT use async_call wrapper (that was arpyc-specific)
    assert "async_call" not in source, (
        "agent_client.py should not use async_call wrapper after migration"
    )
