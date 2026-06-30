"""``AgentProcess._dispatch_status_event`` must not block on a hung
subscriber and must not leak unbounded tasks.

Same incident class as ``_notify_subscribers`` (see
``tests/rpyc_service/test_notify_subscribers_fire_and_forget.py``):
the agent's status push is wired through
``SessionManager.set_status`` → ``notifier=_dispatch_status_event``
→ ``await target.on_message(payload_str)``. If the web peer's
``MessageCallbackService.on_message`` hangs, every ``set_status``
call leaks a task that holds an ``AsyncResult`` slot on the
connection's ``_request_callbacks``. Long-running agents will
eventually trip the inbound-backpressure quarantine on web
(see ``rpyc_async/docs/DESIGN_INBOUND_BACKPRESSURE.md``).

Contract: ``_dispatch_status_event`` schedules each subscriber's
``on_message`` via ``fire_and_forget_async`` with a finite timeout,
returns immediately, and a failed/timed-out push is logged via
``error_callback``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess


def _make_bare_agent_process() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    agent._message_subscribers = set()
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
async def test_dispatch_status_event_returns_immediately_when_subscriber_hangs() -> None:
    """A hung subscriber must not block the SessionManager's setter."""
    hang_forever = asyncio.Event()

    async def hung_on_message(_payload: str) -> None:
        await hang_forever.wait()

    agent = _make_bare_agent_process()
    cb = _make_mock_netref(hung_on_message)
    agent._message_subscribers.add(cb)

    try:
        await asyncio.wait_for(
            agent._dispatch_status_event({"type": "status_change"}),
            timeout=1.0,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            "_dispatch_status_event blocked on a hung subscriber. "
            "Use fire_and_forget_async with a timeout."
        )


@pytest.mark.asyncio
async def test_dispatch_status_event_delivers_to_healthy_subscriber() -> None:
    """Happy path still works: payload reaches the subscriber."""
    delivered: list[str] = []
    delivered_event = asyncio.Event()

    async def good_on_message(payload: str) -> None:
        delivered.append(payload)
        delivered_event.set()

    agent = _make_bare_agent_process()
    cb = _make_mock_netref(good_on_message)
    agent._message_subscribers.add(cb)

    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.wait_for(delivered_event.wait(), timeout=1.0)

    assert len(delivered) == 1
    payload = json.loads(delivered[0])
    assert payload["type"] == "status_change"
    # status defaults to "idle" when session_manager is None
    assert payload["status"] == "idle"


@pytest.mark.asyncio
async def test_dispatch_status_event_logs_subscriber_failure(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A subscriber raising inside ``on_message`` must be logged (the
    legacy implementation prints to stderr; the new one routes the failure
    through Python logging — either is acceptable as long as the failure is
    visible)."""
    error_event = asyncio.Event()

    async def bad_on_message(_payload: str) -> None:
        # tick once so the failure happens in the dispatched task
        await asyncio.sleep(0)
        error_event.set()
        raise RuntimeError("peer gone")

    agent = _make_bare_agent_process()
    cb = _make_mock_netref(bad_on_message)
    agent._message_subscribers.add(cb)

    with caplog.at_level(logging.DEBUG):
        # MUST NOT raise out of the caller (SessionManager.set_status).
        await agent._dispatch_status_event({"type": "status_change"})
        await asyncio.wait_for(error_event.wait(), timeout=1.0)
        # Give the error_callback a chance to log.
        for _ in range(5):
            await asyncio.sleep(0.01)

    captured = capsys.readouterr()
    combined = (
        captured.err
        + captured.out
        + "\n".join(r.getMessage() for r in caplog.records)
    )
    # Failure must surface somewhere visible (stderr print OR Python logging).
    # We accept either the legacy substring or any reasonable variant.
    assert any(
        marker in combined.lower()
        for marker in ("status push", "subscriber", "peer gone", "runtimeerror")
    ), (
        f"Expected the broken subscriber to be logged; "
        f"captured/logs combined was: {combined!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_status_event_task_is_pinned_in_INFLIGHT() -> None:
    """Strong-ref pin via ``rpyc.utils.helpers._INFLIGHT`` — the same
    invariant we assert for ``_notify_subscribers``. See the 2026-05-08
    / 2026-05-11 GC-race incident reports.
    """
    from rpyc.utils import helpers as h

    sub_started = asyncio.Event()
    release = asyncio.Event()

    async def slow_on_message(_payload: str) -> None:
        sub_started.set()
        await release.wait()

    agent = _make_bare_agent_process()
    cb = _make_mock_netref(slow_on_message)
    agent._message_subscribers.add(cb)

    inflight_before = len(h._INFLIGHT)
    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.wait_for(sub_started.wait(), timeout=1.0)

    assert len(h._INFLIGHT) > inflight_before, (
        "fire_and_forget_async should have added the dispatched task to _INFLIGHT"
    )

    release.set()
    for _ in range(20):
        await asyncio.sleep(0.01)
        if len(h._INFLIGHT) == inflight_before:
            break
    assert len(h._INFLIGHT) == inflight_before, (
        "Task should have been auto-discarded from _INFLIGHT after completion"
    )


@pytest.mark.asyncio
async def test_dispatch_status_event_one_broken_subscriber_does_not_starve_others() -> None:
    """A broken subscriber must NOT prevent healthy subscribers from
    receiving the same status update.
    """
    good_delivered: list[str] = []
    good_event = asyncio.Event()
    bad_event = asyncio.Event()

    async def good_on_message(payload: str) -> None:
        good_delivered.append(payload)
        good_event.set()

    async def bad_on_message(_payload: str) -> None:
        await asyncio.sleep(0)
        bad_event.set()
        raise RuntimeError("broken")

    agent = _make_bare_agent_process()
    good = _make_mock_netref(good_on_message)
    bad = _make_mock_netref(bad_on_message)
    agent._message_subscribers.add(good)
    agent._message_subscribers.add(bad)

    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.wait_for(good_event.wait(), timeout=1.0)
    await asyncio.wait_for(bad_event.wait(), timeout=1.0)

    assert len(good_delivered) == 1


@pytest.mark.asyncio
async def test_dead_subscriber_is_evicted_after_push_error() -> None:
    """A subscriber whose ``on_message`` FAILS must be removed from
    ``_message_subscribers`` so the agent stops pushing status to a
    corpse.

    Root cause of the "stuck at idle during inter-agent
    conversation" bug (a related internal incident): the web client died/restarted
    without calling ``exposed_unsubscribe_messages``. The agent's
    ``on_disconnect`` deliberately does NOT clean up (it relies on a
    graceful unsubscribe that never came). The dead
    ``MessageCallbackService`` netref lingered in
    ``_message_subscribers`` forever; every ``set_status`` pushed only
    to the corpse, each push timed out (10 s), and the status NEVER
    reached the live UI — so it kept showing the stale value (idle)
    while the agent was actually working.

    Fix: evict the subscriber on push failure. A live web re-subscribes
    on its next ``connect_to_agent``, so eviction is safe.
    """
    agent = _make_bare_agent_process()
    error_seen = asyncio.Event()

    async def dead_on_message(_payload: str) -> None:
        await asyncio.sleep(0)
        error_seen.set()
        raise RuntimeError("peer gone")

    dead = _make_mock_netref(dead_on_message)
    agent._message_subscribers.add(dead)

    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.wait_for(error_seen.wait(), timeout=1.0)
    # Give the error_callback room to run the eviction.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if dead not in agent._message_subscribers:
            break

    assert dead not in agent._message_subscribers, (
        "a subscriber whose on_message failed must be evicted so the "
        "agent stops pushing status to a dead peer (a related internal incident)"
    )


@pytest.mark.asyncio
async def test_healthy_subscriber_is_not_evicted() -> None:
    """A subscriber whose ``on_message`` succeeds must STAY subscribed —
    eviction only happens on failure, never on a normal delivery."""
    agent = _make_bare_agent_process()
    delivered = asyncio.Event()

    async def good_on_message(_payload: str) -> None:
        delivered.set()

    good = _make_mock_netref(good_on_message)
    agent._message_subscribers.add(good)

    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.wait_for(delivered.wait(), timeout=1.0)
    for _ in range(10):
        await asyncio.sleep(0.01)

    assert good in agent._message_subscribers, (
        "a healthy subscriber must not be evicted on a successful push"
    )


@pytest.mark.asyncio
async def test_eviction_lets_live_subscriber_keep_receiving() -> None:
    """After the dead subscriber is evicted, a co-existing live one keeps
    receiving subsequent status pushes (the live UI recovers)."""
    agent = _make_bare_agent_process()
    live_count = {"n": 0}
    dead_err = asyncio.Event()

    async def live_on_message(_payload: str) -> None:
        live_count["n"] += 1

    async def dead_on_message(_payload: str) -> None:
        await asyncio.sleep(0)
        dead_err.set()
        raise RuntimeError("dead")

    live = _make_mock_netref(live_on_message)
    dead = _make_mock_netref(dead_on_message)
    agent._message_subscribers.add(live)
    agent._message_subscribers.add(dead)

    # First push: dead errors → evicted; live receives.
    await agent._dispatch_status_event({"type": "status_change"})
    await asyncio.wait_for(dead_err.wait(), timeout=1.0)
    for _ in range(20):
        await asyncio.sleep(0.01)
        if dead not in agent._message_subscribers:
            break
    assert dead not in agent._message_subscribers
    assert live in agent._message_subscribers

    # Second push: only the live subscriber remains and still gets it.
    before = live_count["n"]
    await agent._dispatch_status_event({"type": "status_change"})
    for _ in range(20):
        await asyncio.sleep(0.01)
        if live_count["n"] > before:
            break
    assert live_count["n"] > before
