"""``AgentProcess._dispatch_status_event`` must not wrap subscriber
callbacks with ``rpyc.async_``.

Same reasoning as the snapshot-push test: ``rpyc.async_(netref)`` sends
``HANDLE_CALL`` (sync handler on the peer). If the peer's
``exposed_on_message`` is declared ``async def`` — which is the
deep_worker_web contract — the sync handler does not await the
returned coroutine, the peer emits the telltale

    RuntimeWarning: Boxing coroutine object ...exposed_on_message...
    RuntimeWarning: coroutine 'exposed_on_message' was never awaited

and no status_change reaches the UI.

A plain ``on_message(payload_str)`` on an async-flagged netref
routes via ``HANDLE_ASYNC_CALL`` on its own, which the peer's
``_dispatch_request_async`` awaits correctly. The reentrance concern
is already solved by handing the awaitable to ``fire_and_forget_async``
(2026-05-20), which schedules the dispatch on a fresh Task and so
removes us from the current RPC dispatch frame. The companion file
``test_dispatch_status_event_fire_and_forget.py`` covers the
hung-subscriber / leak / timeout invariants of that wiring.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.process.agent_process import AgentProcess


def _make_bare_agent_process() -> AgentProcess:
    """Build a minimal ``AgentProcess`` without running ``__init__``.

    We only exercise ``_dispatch_status_event`` / ``_current_status_snapshot``
    here — all the disk/config wiring is irrelevant.
    """
    agent = AgentProcess.__new__(AgentProcess)
    agent._message_subscribers = set()
    # ``_current_status_snapshot`` depends on ``session_manager`` /
    # ``task_manager``; stub those so the snapshot-building path is
    # deterministic and local.
    agent.session_manager = None
    agent.task_manager = None
    return agent


def _make_mock_netref(on_message_func):
    """Create a MagicMock whose ``____conn__`` evaluates
    ``async_request(8, _, 'on_message', args, kwargs)`` by calling
    ``on_message_func(args[0])`` — matching the real
    ``asyncreq(subscriber, HANDLE_CALLATTR, 'on_message', ...)``
    call in production code."""
    class _Conn:
        async def async_request(self, handler, _proxy, name, args, kwargs):
            if handler == 8 and name == "on_message":
                return await on_message_func(args[0])
            raise NotImplementedError(
                f"unexpected async_request handler={handler}"
            )
    netref = MagicMock()
    object.__setattr__(netref, "____conn__", _Conn())
    return netref


@pytest.mark.asyncio
async def test_dispatch_status_event_does_not_wrap_with_rpyc_async() -> None:
    agent = _make_bare_agent_process()
    on_message_mock = AsyncMock()
    cb = _make_mock_netref(on_message_mock)
    agent._message_subscribers.add(cb)

    # Patch ``rpyc.async_`` at the source. If the implementation
    # reaches for it — either directly or via a lazy ``import rpyc``
    # inside the method — this call counter goes up and we fail.
    import rpyc as rpyc_mod

    with patch.object(rpyc_mod, "async_") as async_wrapper:
        await agent._dispatch_status_event({"type": "status_change"})
        # ``_dispatch_status_event`` schedules one task per subscriber;
        # give the loop a real tick so those tasks run.
        await asyncio.sleep(0.02)

    assert async_wrapper.call_count == 0, (
        "_dispatch_status_event must not wrap on_message with rpyc.async_; "
        "with rpyc_async the async netref already picks HANDLE_ASYNC_CALL"
    )
    assert on_message_mock.await_count == 1
    (arg,) = on_message_mock.await_args.args
    payload = json.loads(arg)
    assert payload["type"] == "status_change"
    # status defaults to "idle" when session_manager is None (see
    # _current_status_snapshot fallback).
    assert payload["status"] == "idle"


@pytest.mark.asyncio
async def test_dispatch_status_event_awaits_async_callback_coroutine() -> None:
    """The dispatched push must drive the subscriber's coroutine
    to completion — no stranded "coroutine was never awaited" warnings.

    Since 2026-05-20 the dispatch goes through ``fire_and_forget_async``
    (see ``test_dispatch_status_event_fire_and_forget.py``), but the
    contract is the same from the subscriber's perspective: when the
    subscriber's ``on_message`` returns normally, its body has run end
    to end and we have a JSON-decodable payload to inspect.
    """
    agent = _make_bare_agent_process()
    delivered: list[str] = []

    async def on_message(payload: str) -> None:
        await asyncio.sleep(0)
        delivered.append(payload)

    cb = _make_mock_netref(on_message)
    agent._message_subscribers.add(cb)

    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.sleep(0.02)

    assert len(delivered) == 1
    payload = json.loads(delivered[0])
    assert payload["type"] == "status_change"


@pytest.mark.asyncio
async def test_dispatch_status_event_survives_broken_subscriber() -> None:
    """A subscriber that raises inside its ``on_message`` must not take
    the whole notification down (observed via other subscribers still
    getting the payload)."""
    agent = _make_bare_agent_process()
    good_delivered: list[str] = []

    async def good_on_message(payload: str) -> None:
        good_delivered.append(payload)

    async def bad_on_message(payload: str) -> None:
        assert isinstance(payload, str)
        raise RuntimeError("peer gone")

    good = _make_mock_netref(good_on_message)
    bad = _make_mock_netref(bad_on_message)
    agent._message_subscribers.add(good)
    agent._message_subscribers.add(bad)

    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.sleep(0.02)

    assert len(good_delivered) == 1
