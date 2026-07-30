"""End-to-end integration tests for AgentService with AsyncioServer.

This module tests that AgentService works correctly with AsyncioServer:
- Server starts (in a child process) and accepts connections
- Async RPC calls work correctly over the wire
- Multiple sequential calls work
- Server shuts down gracefully

Topology
========
**Server runs in a child Python process, client runs in the test
process.** rpyc_async MUST NOT drive both the server and the client
on the same event loop — that deadlocks at first-time
``HANDLE_INSPECT`` (see ``docs/RPYC_NETREF_DEADLOCK_ANALYSIS.md``).
Earlier in-process versions of these tests were the cause of the
7-test HANG cluster; the rewrite below uses
``_subprocess_server.mock_agent_service_subprocess`` to spawn the
server out-of-process and ``async_connect`` to talk to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The local rpyc_async checkout must be on sys.path BEFORE importing rpyc.
_rpyc_async_path = Path("/path/to/rpyc_async")
if str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

from rpyc_async.core import consts  # noqa: E402
from rpyc_async.core.async_connect import async_connect  # noqa: E402

from tests.integration._subprocess_server import (  # noqa: E402
    mock_agent_service_subprocess,
)


async def _async_call(conn, root, method: str, *args):
    """Invoke ``root.<method>(*args)`` without ever issuing a sync_request.

    ``await conn.root.method()`` triggers ``__getattr__`` on the netref
    which goes through the synchronous ``sync_request(HANDLE_GETATTR)``
    — that is the exact same reader-in-callback deadlock we already
    closed for ``HANDLE_INSPECT`` and which rpyc_async's guard now
    rejects loudly. The production ``AgentClient._send_command`` uses
    the same low-level path; we mirror it here.
    """
    method_with_prefix = method if method.startswith("exposed_") else f"exposed_{method}"
    async_result = conn.async_request(
        consts.HANDLE_ASYNC_CALLATTR,
        root,
        method_with_prefix,
        args,
        (),
    )
    return await async_result


@pytest.mark.asyncio
async def test_agent_service_with_asyncio_server():
    """AgentService methods work end-to-end through AsyncioServer.

    Server lives in a child process — see module docstring.
    """
    async with mock_agent_service_subprocess(name="test-agent") as (_, port):
        conn = await async_connect(
            "127.0.0.1", port, config={"allow_all_attrs": True}
        )
        try:
            # Access ``conn.root`` once via the async helper — the
            # production client does the same lazy-root dance.
            root = conn.root

            status_str = await _async_call(conn, root, "get_status")
            status = json.loads(status_str)
            assert isinstance(status, dict)
            assert "success" in status

            metadata_str = await _async_call(conn, root, "get_metadata")
            metadata = json.loads(metadata_str)
            assert isinstance(metadata, dict)
            assert metadata.get("success") is not False, metadata

            for _ in range(3):
                s = json.loads(await _async_call(conn, root, "get_status"))
                assert isinstance(s, dict)
        finally:
            await conn.aclose()


@pytest.mark.asyncio
async def test_agent_service_overridden_base_methods():
    """Overridden ``rpyc.Service`` base methods are reachable async."""
    async with mock_agent_service_subprocess(name="test-agent-2") as (_, port):
        conn = await async_connect(
            "127.0.0.1", port, config={"allow_all_attrs": True}
        )
        try:
            root = conn.root
            service_name = await _async_call(conn, root, "get_service_name")
            assert service_name == "AGENT_SERVICE"

            aliases_str = await _async_call(conn, root, "get_service_aliases")
            aliases = json.loads(aliases_str)
            assert isinstance(aliases, list)
        finally:
            await conn.aclose()


@pytest.mark.asyncio
async def test_server_handles_multiple_clients_sequentially():
    """A single server process accepts two sequential clients."""
    async with mock_agent_service_subprocess(name="test-multi-client") as (_, port):
        conn1 = await async_connect(
            "127.0.0.1", port, config={"allow_all_attrs": True}
        )
        try:
            root1 = conn1.root
            metadata1 = json.loads(await _async_call(conn1, root1, "get_metadata"))
            assert isinstance(metadata1, dict)
        finally:
            await conn1.aclose()

        conn2 = await async_connect(
            "127.0.0.1", port, config={"allow_all_attrs": True}
        )
        try:
            root2 = conn2.root
            metadata2 = json.loads(await _async_call(conn2, root2, "get_metadata"))
            assert isinstance(metadata2, dict)
        finally:
            await conn2.aclose()


@pytest.mark.asyncio
async def test_asyncio_server_graceful_shutdown():
    """The server subprocess shuts down cleanly on SIGTERM."""
    async with mock_agent_service_subprocess(name="test-shutdown") as (proc, port):
        assert port > 0
    assert proc.returncode is not None
