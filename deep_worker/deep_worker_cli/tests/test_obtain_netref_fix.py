"""Test that AgentClient receives proper local objects over RPyC, not netrefs.

This is a regression test for Fix #3 (high CPU caused by netref accumulation).

In the current implementation (client/agent_client.py ~line 151 onward),
`_send_command` calls
`self._conn.async_request(HANDLE_ASYNC_CALLATTR, self._root, "exposed_<name>",
(...), ())` and `await`s the AsyncResult. The server returns a JSON STRING
(see RPYC_SERIALIZATION_DESIGN.md), so the client does `json.loads(...)`
to get a LOCAL dict — no netref dereferencing happens at all during
field access.

These tests verify the happy-path contract against a mocked connection.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_mock_conn(server_reply: dict) -> MagicMock:
    """Build a mock rpyc.Connection that replies to any async_request
    call with ``server_reply`` (JSON-encoded)."""

    async def fake_async_request(*args, **kwargs):
        return json.dumps(server_reply)

    conn = MagicMock()
    conn.async_request = MagicMock(side_effect=fake_async_request)
    conn.root = MagicMock()
    return conn


@pytest.mark.asyncio
async def test_get_status_returns_local_dict_not_netref():
    """get_status() returns a proper LOCAL Python object after JSON decode."""
    from deep_worker_cli.client.agent_client import AgentClient

    client = AgentClient("test_agent")
    client._conn = _make_mock_conn({
        "success": True,
        "status": "ready",
        "error": None,
    })
    client._root = client._conn.root
    client.connected = True

    session_info = await client.get_status()

    # Returned structure is built locally from a plain dict — no
    # further RPCs are needed. If it were a netref, constructing a
    # SessionInfo would require extra sync_request calls (blocked
    # by the guard).
    assert session_info.name == "test_agent"
    assert session_info.status in {"ready", "error", "working"}


@pytest.mark.asyncio
async def test_invoke_agent_returns_local_dict_not_netref():
    """invoke_agent() returns a plain dict — no netref overhead on fields."""
    from deep_worker_cli.client.agent_client import AgentClient

    client = AgentClient("test_agent")
    client._conn = _make_mock_conn({
        "success": True,
        "response": {"role": "assistant", "content": "Hello"},
    })
    client._root = client._conn.root
    client.connected = True

    response = await client.invoke_agent("test message")

    assert isinstance(response, dict)
    assert response["success"] is True
    # Nested dict — exercising `response["response"]["content"]` must
    # stay local (no netref).
    assert response["response"]["content"] == "Hello"
    assert isinstance(response["response"], dict)


@pytest.mark.asyncio
async def test_get_history_returns_local_dict_not_netref():
    """get_history() returns a plain dict with a plain list of messages."""
    from deep_worker_cli.client.agent_client import AgentClient

    client = AgentClient("test_agent")
    client._conn = _make_mock_conn({
        "success": True,
        "history": [{"role": "user", "content": "test"}],
        "total": 1,
        "has_more": False,
    })
    client._root = client._conn.root
    client.connected = True

    response = await client.get_history(limit=50, offset=0)

    assert isinstance(response, dict)
    assert response["success"] is True
    assert isinstance(response["history"], list)
    assert response["history"][0]["content"] == "test"


@pytest.mark.asyncio
async def test_obtain_helper_still_exported():
    """`rpyc.utils.classic.obtain` remains importable — agent_client
    used to call it explicitly; the JSON-string return path makes
    the explicit call unnecessary, but the helper must still be
    available for any ad-hoc netref-to-local conversion."""
    import sys
    from pathlib import Path

    _rpyc_async_path = Path("/path/to/rpyc_async")
    if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
        sys.path.insert(0, str(_rpyc_async_path))

    from rpyc_async.utils.classic import obtain

    test_dict = {"key": "value", "number": 42}
    result = obtain(test_dict)
    assert result == test_dict
    assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
