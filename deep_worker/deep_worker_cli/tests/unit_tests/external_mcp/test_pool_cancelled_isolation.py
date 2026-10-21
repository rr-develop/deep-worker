"""An unreachable server MUST NOT be able to kill the pool — even via ``CancelledError``.

Incident 2026-08-29: the only HTTP server in a per-agent config pointed at a
dead port. ``session.initialize()`` failed with ``httpcore.ConnectError``, but
that is not what came OUT of the mcp SDK: the anyio TaskGroup inside
``streamablehttp_client`` cancels its tasks, and the caller receives
``CancelledError``.

``CancelledError`` inherits from ``BaseException``, not ``Exception`` — so it
went straight through the per-server ``except Exception`` in ``_start_all`` and
killed the whole agent process before RPyC started and before registration in
the registry. The agent never appeared in the web UI, and its history became
impossible to read.

The contract pinned here: per-server isolation (`pool.py` §"fail-loud +
degrade") must contain ANY failure of a single server, including cancellation
from inside an anyio scope. External cancellation (we are being shut down)
must still propagate.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from deep_worker_cli.external_mcp.config import ServerSpec
from deep_worker_cli.external_mcp.pool import ExternalMCPClientPool


class _Streams:
    """Stand-in for (read, write) — contents do not matter, ClientSession is mocked."""


class _StreamsCM:
    async def __aenter__(self) -> Any:
        return (_Streams(), _Streams())

    async def __aexit__(self, *_: Any) -> bool:
        return False


class _DeadServerSession:
    """``initialize()`` fails exactly like the live mcp SDK on a dead port.

    The anyio scope inside the transport is cancelled on ConnectError, and
    ``CancelledError`` comes out — NOT ``ConnectError``.
    """

    async def initialize(self) -> None:
        raise asyncio.CancelledError("Cancelled by cancel scope deadbeef")

    async def list_tools(self) -> Any:  # pragma: no cover — never reached
        raise AssertionError("list_tools must not be reached")


class _LiveServerSession:
    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="alive_tool",
                    description="stub",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )

    async def call_tool(self, name: str, args: dict) -> Any:  # pragma: no cover
        return SimpleNamespace(content=[], isError=False)


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch, sessions_by_order: list[Any]
) -> None:
    """Replaces the mcp SDK entry points that ``_start_all`` imports lazily.

    ``ClientSession`` hands out preset sessions in order of access — this is
    how the test controls which server is "dead" without touching the network.
    """
    import sys

    remaining = list(sessions_by_order)

    class _ClientSession:
        def __init__(self, read: Any, write: Any) -> None:
            self._session = remaining.pop(0)

        async def __aenter__(self) -> Any:
            return self._session

        async def __aexit__(self, *_: Any) -> bool:
            return False

    mcp_mod = SimpleNamespace(
        ClientSession=_ClientSession,
        StdioServerParameters=lambda **kw: SimpleNamespace(**kw),
    )
    http_mod = SimpleNamespace(
        streamablehttp_client=lambda url, headers=None: _StreamsCM()
    )
    stdio_mod = SimpleNamespace(stdio_client=lambda params: _StreamsCM())

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", http_mod)


@pytest.mark.asyncio
async def test_unreachable_http_server_does_not_kill_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single dead HTTP server → the pool comes up empty, without blowing up.

    This is exactly the incident configuration: one server, and it is dead.
    Without the fix, ``CancelledError`` escapes from ``create()``.
    """
    _install_fake_sdk(monkeypatch, [_DeadServerSession()])

    pool = ExternalMCPClientPool()
    await pool._start_all(
        {
            "code_graph_cross_file_update": ServerSpec(
                name="code_graph_cross_file_update",
                type="http",
                url="http://127.0.0.1:40587/cross-file-update-mcp",
            )
        }
    )

    assert pool.tools == [], "a dead server must not yield any tools"


@pytest.mark.asyncio
async def test_dead_server_does_not_take_down_healthy_neighbour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-server isolation: a neighbour in the config must come up.

    This is the property ``CancelledError`` broke — it took down the whole
    ``for name, spec in merged.items()`` loop, not just its own server.
    """
    _install_fake_sdk(monkeypatch, [_DeadServerSession(), _LiveServerSession()])

    pool = ExternalMCPClientPool()
    await pool._start_all(
        {
            "dead": ServerSpec(
                name="dead", type="http", url="http://127.0.0.1:40587/mcp"
            ),
            "alive": ServerSpec(
                name="alive", type="http", url="http://127.0.0.1:40588/mcp"
            ),
        }
    )

    canonical = {t.canonical_name for t in pool.tools}
    assert canonical == {"alive.alive_tool"}, (
        f"the live neighbour must come up, got {canonical}"
    )


@pytest.mark.asyncio
async def test_external_cancellation_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation initiated from OUTSIDE (we are being shut down) must pass through.

    A safeguard against overzealousness: catching ``BaseException`` must not
    turn a normal shutdown into "swallowed and carried on". The sign of
    external cancellation is ``Task.cancelling() > 0`` after ``task.cancel()``.
    """
    started = asyncio.Event()

    class _HangingSession:
        async def initialize(self) -> None:
            started.set()
            await asyncio.Event().wait()  # wait until we are cancelled from outside

    _install_fake_sdk(monkeypatch, [_HangingSession()])

    pool = ExternalMCPClientPool()

    async def _run() -> None:
        await pool._start_all(
            {"hang": ServerSpec(name="hang", type="http", url="http://x/mcp")}
        )

    task = asyncio.create_task(_run())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
