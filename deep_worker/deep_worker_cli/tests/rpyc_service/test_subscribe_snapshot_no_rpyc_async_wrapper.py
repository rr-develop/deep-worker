"""The initial ``status_change`` snapshot push MUST NOT wrap the
subscriber's ``on_message`` with ``rpyc.async_``.

Background
==========
``rpyc.async_(proxy)`` sends an ``asyncreq(proxy, HANDLE_CALL, ...)`` —
i.e. the **sync** request handler on the peer. When the peer's
``exposed_on_message`` is declared ``async def`` (which is the
deep_worker_web contract), the sync handler just calls it and boxes
the returned coroutine without awaiting it. Result on the peer:

    RuntimeWarning: Boxing coroutine object ...exposed_on_message...
    RuntimeWarning: coroutine 'exposed_on_message' was never awaited

…and the web dispatcher never runs, so no ``status_change`` reaches
the UI. With native rpyc_async, the netref's own ``__call__`` picks
``HANDLE_ASYNC_CALL`` automatically when ``____is_async__`` is True,
and the peer's ``_dispatch_request_async`` awaits the coroutine
correctly. So the right call is just ``await on_message(payload)`` —
no wrapper.

Reentrance concern handled separately
-------------------------------------
Calling ``await on_message(...)`` from inside the same RPC dispatch
would trip rpyc's ``sync_request() from asyncio loop`` guard. But
``exposed_subscribe_messages`` already defers the push via
``loop.call_later(0, ...)``, so by the time ``_deliver_snapshot``
runs, the subscribe RPC has already returned and we are on a fresh
task — no wrapper needed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


def _build_service_with_agent() -> tuple[AgentService, MagicMock]:
    service = AgentService()
    agent = MagicMock()
    agent._message_subscribers = set()
    snapshot: dict[str, Any] = {
        "type": "status_change",
        "status": "running",
        "error_message": None,
    }
    agent._current_status_snapshot = MagicMock(return_value=snapshot)
    service._agent_process = agent
    return service, agent


@pytest.mark.asyncio
async def test_deliver_snapshot_does_not_wrap_with_rpyc_async(netref_subscriber) -> None:
    """The snapshot push must call ``await callback.on_message(payload)``
    directly — NOT through ``rpyc.async_(callback.on_message)(...)``.

    With rpyc_async + AsyncioServer, an async-flagged netref routes
    through ``HANDLE_ASYNC_CALL`` on its own; wrapping with
    ``rpyc.async_`` forces ``HANDLE_CALL`` (sync), which strands the
    peer's async handler as an un-awaited coroutine.
    """
    service, _ = _build_service_with_agent()
    cb = netref_subscriber(AsyncMock())

    # Patch ``rpyc.async_`` in the module under test. If the
    # implementation still wraps, this patch is invoked and we fail.
    import deep_worker_cli.rpyc_service.agent_service as svc_mod

    with patch.object(svc_mod.rpyc, "async_") as async_wrapper:
        await service.exposed_subscribe_messages(cb)
        # Let the deferred ``_deliver_snapshot`` run.
        await asyncio.sleep(0.02)

    assert async_wrapper.call_count == 0, (
        "snapshot push must not wrap on_message with rpyc.async_; "
        "rpyc_async netrefs already pick HANDLE_ASYNC_CALL when "
        "____is_async__ is True"
    )
    # Sanity: the snapshot was actually delivered.
    assert cb.on_message.await_count == 1
    (arg,) = cb.on_message.await_args.args
    payload = json.loads(arg)
    assert payload["type"] == "status_change"
    assert payload["status"] == "running"


@pytest.mark.asyncio
async def test_deliver_snapshot_awaits_async_netref_coroutine(netref_subscriber) -> None:
    """Simulate a real rpyc_async netref: ``callback.on_message(x)`` is
    itself an async operation that must be ``await``-ed by the code
    under test. If the fix forgets the ``await`` (e.g. by using a
    fire-and-forget wrapper), the coroutine is never scheduled and
    the payload never arrives.
    """
    service, _ = _build_service_with_agent()

    delivered: list[str] = []

    async def on_message(payload: str) -> None:
        # Yield to simulate an RPC round-trip.
        await asyncio.sleep(0)
        delivered.append(payload)

    # The netref's ``____conn__.async_request`` returns the coroutine
    # produced by ``on_message`` so the implementation can await it.
    cb = netref_subscriber(on_message)

    await service.exposed_subscribe_messages(cb)
    await asyncio.sleep(0.02)

    assert len(delivered) == 1
    payload = json.loads(delivered[0])
    assert payload["type"] == "status_change"
    assert payload["status"] == "running"
