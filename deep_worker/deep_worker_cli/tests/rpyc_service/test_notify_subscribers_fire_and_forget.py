"""``_notify_subscribers`` must not block on a hung subscriber.

Production incident: the agent received user
messages over RPyC, called ``_handle_invoke`` → ``_notify_subscribers``
→ ``await subscriber.on_message(payload_str)`` — and the web-side
``MessageCallbackService.on_message`` never returned. The agent
therefore could not run ``ainvoke`` at all: the very first pre-flight
broadcast on the new user message blocked forever.

Diagnosis: pyrasite introspection found three ``_dispatch_request_async``
tasks parked on ``agent_service.py:1219`` for 60+ minutes, each with a
hung ``subscriber.on_message`` AsyncResult on conn4. No quarantine, no
LLM hang, no codex hang — purely the outbound callback to web.

The fix is to dispatch the callback via ``fire_and_forget_async`` so
the broadcaster returns immediately. A bounded ``timeout`` plus an
``error_callback`` keep the outbound ``_request_callbacks`` from
growing unboundedly while a peer is sick.

Tests in this module assert the new contract:
    * ``_notify_subscribers`` returns in ~milliseconds even if every
      subscriber's ``on_message`` hangs forever.
    * The broadcast is actually delivered (the awaitable is scheduled
      and runs) when subscribers are healthy.
    * A failing ``on_message`` is logged via the ``error_callback``
      path, not re-raised into the caller.
    * The dispatched task is pinned in ``_INFLIGHT`` until it
      finishes (success / failure / timeout) — no GC race.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


class _FakeConn:
    """Minimal stand-in for a netref's ``____conn__`` (see
    ``tests/rpyc_service/conftest.py`` for the rationale). Production's
    ``asyncreq(subscriber, 8, "on_message", args, ())`` resolves to
    ``conn.async_request(8, subscriber, "on_message", args, ())``."""

    def __init__(self, on_message):
        self._on_message = on_message

    def async_request(self, handler, _proxy, name, args, kwargs):
        if handler == 8 and name == "on_message":
            return self._on_message(*args, **(dict(kwargs) if kwargs else {}))
        raise NotImplementedError(
            f"unexpected async_request handler={handler!r} name={name!r}"
        )


def _make_netref_subscriber(on_message):
    """Build a MagicMock that behaves like an rpyc_async netref: a real
    ``____conn__`` in ``__dict__`` so ``object.__getattribute__`` finds
    it, delegating ``on_message`` calls to ``on_message``."""
    subscriber = MagicMock()
    subscriber.on_message = on_message
    object.__setattr__(subscriber, "____conn__", _FakeConn(on_message))
    return subscriber


def _make_service_with_subscriber(on_message):
    """Build a minimal ``AgentService`` with a single subscriber.

    The subscriber is an rpyc_async-style netref (real ``____conn__``)
    so production's ``asyncreq(subscriber, 8, "on_message", ...)`` path
    resolves to ``on_message``.
    """
    service = AgentService()
    agent = MagicMock()
    subscriber = _make_netref_subscriber(on_message)
    agent._message_subscribers = {subscriber}
    service._agent_process = agent
    return service, subscriber


@pytest.mark.asyncio
async def test_notify_subscribers_returns_immediately_when_subscriber_hangs() -> None:
    """If the subscriber's ``on_message`` never resolves, the broadcaster
    must still return (within a small bound) — no more multi-hour hangs.
    """
    hang_forever = asyncio.Event()  # never set

    async def hung_on_message(_payload: str) -> None:
        await hang_forever.wait()

    service, _sub = _make_service_with_subscriber(hung_on_message)

    # Hard cap: a healthy notify call resolves in <100 ms. We allow
    # 1 s to absorb test scheduler noise and still catch a real hang.
    try:
        await asyncio.wait_for(
            service._notify_subscribers({"type": "user", "content": "hello"}),
            timeout=1.0,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            "_notify_subscribers blocked on a hung subscriber for >1s. "
            "The dispatched callback must not be awaited inline; use "
            "fire_and_forget_async instead."
        )


@pytest.mark.asyncio
async def test_notify_subscribers_returns_immediately_with_many_hung_subscribers() -> None:
    """Multiple hung subscribers must still not stack up serial waits."""
    hang_forever = asyncio.Event()

    async def hung_on_message(_payload: str) -> None:
        await hang_forever.wait()

    service = AgentService()
    agent = MagicMock()
    subs = []
    for _ in range(5):
        s = _make_netref_subscriber(hung_on_message)
        subs.append(s)
    agent._message_subscribers = set(subs)
    service._agent_process = agent

    try:
        await asyncio.wait_for(
            service._notify_subscribers({"type": "tool_call", "name": "x"}),
            timeout=1.0,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            "_notify_subscribers blocked when multiple subscribers hang. "
            "fire_and_forget_async should dispatch each independently."
        )


@pytest.mark.asyncio
async def test_notify_subscribers_actually_delivers_to_healthy_subscriber() -> None:
    """Sanity: when the subscriber's ``on_message`` is well-behaved, the
    JSON payload still reaches it — the fix doesn't break the happy path.
    """
    delivered: list[str] = []
    delivered_event = asyncio.Event()

    async def healthy_on_message(payload: str) -> None:
        delivered.append(payload)
        delivered_event.set()

    service, _sub = _make_service_with_subscriber(healthy_on_message)

    await service._notify_subscribers({"type": "user", "content": "hi"})
    # The dispatch may be on a separate task; give it a small window.
    await asyncio.wait_for(delivered_event.wait(), timeout=1.0)

    assert len(delivered) == 1
    parsed = json.loads(delivered[0])
    assert parsed == {"type": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_notify_subscribers_logs_subscriber_error_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A subscriber raising inside ``on_message`` must:
        * not propagate the exception out of ``_notify_subscribers``
        * leave a record in the agent_service logger so operators
          can see which subscriber was broken.
    """
    logged_event = asyncio.Event()

    async def bad_on_message(_payload: str) -> None:
        # tick once so the error happens in the dispatched task,
        # not during the synchronous call() that creates AsyncResult.
        await asyncio.sleep(0)
        logged_event.set()
        raise RuntimeError("peer gone")

    service, _sub = _make_service_with_subscriber(bad_on_message)

    import logging
    with caplog.at_level(logging.WARNING, logger="deep_worker_cli.rpyc_service.agent_service"):
        # MUST NOT raise.
        await service._notify_subscribers({"type": "user", "content": "x"})
        # Give the dispatched task room to run and log.
        try:
            await asyncio.wait_for(logged_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("subscriber's on_message was never even called")
        # Drain any remaining tasks so the error_callback can log.
        for _ in range(5):
            await asyncio.sleep(0.01)

    # At least one warning/error log line must mention the failed subscriber.
    matching = [
        rec for rec in caplog.records
        if "subscriber" in rec.getMessage().lower()
        and ("peer gone" in rec.getMessage() or "RuntimeError" in rec.getMessage()
             or "fail" in rec.getMessage().lower())
    ]
    assert matching, (
        f"Expected at least one log record about the failed subscriber; "
        f"got: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_notify_subscribers_dispatched_task_is_pinned_in_INFLIGHT() -> None:
    """The fire-and-forget Task must be strong-ref pinned via the
    rpyc_async ``_INFLIGHT`` set so a stray GC pass can't kill it
    while ``on_message`` is mid-flight.

    Regression guard for past GC-race incidents:
    a dispatched Task with only a weak ref dies silently and pins the
    AsyncResult chain in ``Connection._request_callbacks``.
    """
    from rpyc_async.utils import helpers as h

    sub_call_started = asyncio.Event()
    release_subscriber = asyncio.Event()

    async def slow_on_message(_payload: str) -> None:
        sub_call_started.set()
        await release_subscriber.wait()

    service, _sub = _make_service_with_subscriber(slow_on_message)

    inflight_before = len(h._INFLIGHT)
    await service._notify_subscribers({"type": "user", "content": "x"})
    await asyncio.wait_for(sub_call_started.wait(), timeout=1.0)

    # While the subscriber is still working, the dispatched task must
    # be alive AND pinned in _INFLIGHT (otherwise GC could drop it).
    assert len(h._INFLIGHT) > inflight_before, (
        "fire_and_forget_async should have added the dispatched task to _INFLIGHT"
    )

    # Now let the subscriber finish. The task must be removed from
    # _INFLIGHT (auto-discard via add_done_callback).
    release_subscriber.set()
    for _ in range(20):
        await asyncio.sleep(0.01)
        if len(h._INFLIGHT) == inflight_before:
            break
    assert len(h._INFLIGHT) == inflight_before, (
        "Task should have been removed from _INFLIGHT after completion "
        "(auto-discard via add_done_callback)"
    )


@pytest.mark.asyncio
async def test_notify_subscribers_enforces_timeout_on_hung_subscriber() -> None:
    """The dispatched task must time out if the subscriber's
    ``on_message`` never resolves, so ``Connection._request_callbacks``
    cannot grow unboundedly while a peer is sick.

    We don't assert on wall-clock — just that the task eventually
    completes (with a TimeoutError) rather than living forever.
    """
    from rpyc_async.utils import helpers as h

    hang_forever = asyncio.Event()

    async def hung_on_message(_payload: str) -> None:
        await hang_forever.wait()

    service, _sub = _make_service_with_subscriber(hung_on_message)

    inflight_before = len(h._INFLIGHT)
    await service._notify_subscribers({"type": "user", "content": "x"})

    # The dispatched task should now be in _INFLIGHT.
    assert len(h._INFLIGHT) > inflight_before

    # Find the task we just dispatched and verify it has a timeout —
    # i.e. it eventually becomes done() even though hang_forever never fires.
    # We use the small test-only timeout configured by the implementation
    # (overridable via env). The default production timeout is 10s; in
    # tests we set it to a small value via DWCLI_NOTIFY_TIMEOUT_SEC.
    import os
    os.environ["DWCLI_NOTIFY_TIMEOUT_SEC"] = "0.2"
    try:
        # Re-dispatch with the env var set so this test does not depend
        # on the previous (10s) one.
        await service._notify_subscribers({"type": "user", "content": "y"})
        # Wait until the dispatched task settles (timed out).
        for _ in range(50):
            await asyncio.sleep(0.05)
            if len(h._INFLIGHT) <= inflight_before + 1:  # +1 = first hung dispatch
                break
        # At least one of the dispatched tasks must have settled.
        assert len(h._INFLIGHT) < inflight_before + 2, (
            f"Tasks should have settled after timeout; _INFLIGHT size = "
            f"{len(h._INFLIGHT)}, before = {inflight_before}"
        )
    finally:
        os.environ.pop("DWCLI_NOTIFY_TIMEOUT_SEC", None)
        # Free the first hung task that was scheduled with the default timeout.
        # It will eventually time out on its own; mark it as not-our-concern.
        hang_forever.set()


@pytest.mark.asyncio
async def test_dead_message_subscriber_is_evicted() -> None:
    """A message subscriber whose ``on_message`` fails must be evicted
    from the agent's ``_message_subscribers`` set — same dead-peer
    hazard as status pushes (observed in a past incident). on_disconnect does
    not clean up; without eviction the corpse soaks a 10 s timeout on
    every broadcast forever."""
    error_seen = asyncio.Event()

    async def dead_on_message(_payload: str) -> None:
        await asyncio.sleep(0)
        error_seen.set()
        raise RuntimeError("peer gone")

    service, sub = _make_service_with_subscriber(dead_on_message)
    subs = service._agent_process._message_subscribers

    await service._notify_subscribers({"type": "user", "content": "x"})
    await asyncio.wait_for(error_seen.wait(), timeout=1.0)
    for _ in range(20):
        await asyncio.sleep(0.01)
        if sub not in subs:
            break

    assert sub not in subs, (
        "a message subscriber whose on_message failed must be evicted"
    )


@pytest.mark.asyncio
async def test_healthy_message_subscriber_not_evicted() -> None:
    """A healthy message subscriber must stay subscribed."""
    delivered = asyncio.Event()

    async def good_on_message(_payload: str) -> None:
        delivered.set()

    service, sub = _make_service_with_subscriber(good_on_message)
    subs = service._agent_process._message_subscribers

    await service._notify_subscribers({"type": "user", "content": "x"})
    await asyncio.wait_for(delivered.wait(), timeout=1.0)
    for _ in range(10):
        await asyncio.sleep(0.01)

    assert sub in subs, "a healthy subscriber must not be evicted"
