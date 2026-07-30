"""Regression: AgentClient.connect() must BOUND the whole RPyC connect —
including the eager ``HANDLE_GETROOT`` handshake — so a busy/deadlocked peer
can never hang the sender's turn.

Past incident (recurring — same family as a related internal incident
analysis, not included here): ``send_message_to_agent`` is TRUE fire-and-forget on the
RECEIVER side (``exposed_enqueue_message`` returns immediately). But on the
SENDER side ``AgentClient.connect()`` calls
``async_connect(..., timeout=10.0)`` whose ``timeout`` wraps ONLY
``sock_connect``. The eager root handshake inside async_connect —
``await conn.async_request(HANDLE_GETROOT)`` — is UNBOUNDED. When the target
agent is mid-turn (busy), TCP connects fine but the handshake never gets
answered, so ``connect()`` blocks. Bounded only by the outer 120s tool
timeout, ``send_message_to_agent`` then parks as a background job and returns
an ``in_progress`` notice — NOT ``{"success": true}``. That means
``SendMessageCompletionMiddleware`` (which ends the turn ONLY on a successful
send) does NOT fire, so the sending agent WRONGLY keeps making tool calls
after the send instead of ending its turn and going idle.

Fix: wrap the ENTIRE ``async_connect(...)`` in an outer ``asyncio.wait_for``
(bounding TCP + eager handshake), exactly like the web ``AgentService`` does.
On timeout raise ``IPCConnectionError`` so the tool returns a clean
``{"success": false}`` the model can retry — never a 120s park.

These tests mock ONLY the clock (via a hanging async_connect + advancing the
loop clock) so the production bound fires without real waiting.
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import MagicMock, patch

from deep_worker_cli.client.agent_client import AgentClient, IPCConnectionError
from deep_worker_cli.process.registry import ProcessRegistry


def _mock_registry(port: int = 45999):
    reg = MagicMock(spec=ProcessRegistry)
    # connect() only reads ``agent_info.rpyc_port``.
    info = MagicMock()
    info.rpyc_port = port
    reg.lookup.return_value = info
    return reg


@pytest.mark.asyncio
async def test_connect_is_bounded_when_handshake_hangs() -> None:
    """A busy peer whose eager HANDLE_GETROOT handshake never completes must
    NOT hang connect() — it must raise IPCConnectionError within the bound.
    Proven by mocking ONLY the clock."""

    async def _hanging_async_connect(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        # Simulates async_connect where sock_connect succeeded but the eager
        # HANDLE_GETROOT handshake never gets answered (target mid-turn).
        await asyncio.sleep(10_000)
        return MagicMock()

    with patch(
        "deep_worker_cli.client.agent_client.ProcessRegistry"
    ) as reg_cls:
        reg_cls.return_value = _mock_registry()
        with patch(
            "deep_worker_cli.client.agent_client.async_connect",
            side_effect=_hanging_async_connect,
        ):
            client = AgentClient("target")

            loop = asyncio.get_running_loop()
            real_time = loop.time
            skew = {"v": 0.0}
            loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
            try:
                task = asyncio.ensure_future(client.connect())
                await asyncio.sleep(0.02)
                assert not task.done(), "connect must be waiting under the bound"
                # Advance the clock past the connect bound so wait_for fires.
                skew["v"] = AgentClient._CONNECT_TIMEOUT + 5.0
                for _ in range(5):
                    await asyncio.sleep(0)
                    if task.done():
                        break
                with pytest.raises(IPCConnectionError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            finally:
                loop.time = real_time  # type: ignore[method-assign]

            # Never left in a half-connected state.
            assert client.connected is False


@pytest.mark.asyncio
async def test_connect_succeeds_fast_when_peer_responsive() -> None:
    """Happy path unchanged: a responsive peer connects promptly and the
    connection is marked connected."""
    from unittest.mock import AsyncMock

    with patch(
        "deep_worker_cli.client.agent_client.ProcessRegistry"
    ) as reg_cls:
        reg_cls.return_value = _mock_registry()
        with patch(
            "deep_worker_cli.client.agent_client.async_connect",
            new_callable=AsyncMock,
        ) as mock_ac:
            mock_ac.return_value = MagicMock()
            client = AgentClient("target")
            await client.connect()
            assert client.connected is True
            mock_ac.assert_awaited_once()


def test_connect_wraps_async_connect_in_outer_wait_for() -> None:
    """Static guard: connect() MUST bound the whole async_connect in an outer
    asyncio.wait_for. The async_connect ``timeout=`` kwarg alone covers ONLY
    sock_connect, NOT the eager HANDLE_GETROOT handshake — relying on it is the
    exact regression from that past incident. If this guard fails, the
    unbounded-handshake hole is back."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(AgentClient.connect))
    tree = ast.parse(src)

    # Find the `await asyncio.wait_for(async_connect(...))` shape: an
    # `asyncio.wait_for` call whose first positional arg is an `async_connect`
    # call (directly, since async_connect returns a coroutine).
    found = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)):
            continue
        func = node.func
        is_wait_for = (
            isinstance(func, ast.Attribute) and func.attr == "wait_for"
        )
        if not is_wait_for or not node.args:
            continue
        inner = node.args[0]
        if isinstance(inner, ast.Call):
            iname = inner.func
            if (isinstance(iname, ast.Name) and iname.id == "async_connect") or (
                isinstance(iname, ast.Attribute) and iname.attr == "async_connect"
            ):
                found = True
                break

    assert found, (
        "AgentClient.connect() must wrap async_connect(...) in an outer "
        "asyncio.wait_for to bound the eager HANDLE_GETROOT handshake "
        "(async_connect's own timeout covers ONLY sock_connect). Without it a "
        "busy peer hangs the sender's send_message_to_agent → parks as a 120s "
        "job → the turn-ender never fires. See "
        "test_agent_client_connect_bounded.py."
    )
