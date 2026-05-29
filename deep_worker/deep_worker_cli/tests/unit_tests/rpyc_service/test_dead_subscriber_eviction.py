"""Regression: evicting a DEAD RPyC-netref subscriber must not crash the
notify path (and must not leak the corpse).

Incident (observed 2026-05-22): a web subscriber's RPyC channel died. In
``_notify_subscribers``:

  1. ``subscriber.on_message(...)`` raised ``EOFError('stream has been
     closed')`` (send over dead socket);
  2. the defensive ``except`` tried to evict it with
     ``subscriber_set.discard(subscriber)``;
  3. ``set.discard`` needs ``hash(subscriber)`` → ``netref.__hash__`` →
     a SYNC RPyC request over the SAME dead channel → ``EOFError`` AGAIN,
     this time INSIDE the except handler;
  4. that second EOFError escaped ``_notify_subscribers`` into
     ``_handle_invoke`` → ``set_error("Agent invocation failed: stream has
     been closed")`` → the agent went to ERROR over a dead UI socket.

The fix must evict by OBJECT IDENTITY (no wire-crossing ``__hash__`` on the
netref) and never let a secondary eviction error reach the invoke.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService


class _DeadishNetref:
    """Mimics an RPyC netref that was hashable when added on a LIVE channel,
    then the channel died: from that point ``on_message`` AND ``__hash__``
    do a wire round-trip and raise ``EOFError``.

    ``go_dead()`` flips it after it has been added to the subscriber set
    (so the set add — which needs a working hash — succeeds, exactly like
    production).
    """

    def __init__(self) -> None:
        self._dead = False
        self._live_hash = id(self)

    def go_dead(self) -> None:
        self._dead = True

    def on_message(self, _payload: str) -> Any:
        raise EOFError("stream has been closed")

    def __hash__(self) -> int:
        if self._dead:
            raise EOFError("stream has been closed")
        return self._live_hash

    def __eq__(self, other: object) -> bool:
        if self._dead:
            raise EOFError("stream has been closed")
        return self is other


class _FakeAgentProcess:
    def __init__(self) -> None:
        self._message_subscribers: set[Any] = set()


def _make_service(agent: _FakeAgentProcess) -> AgentService:
    svc = AgentService.__new__(AgentService)
    svc._agent_process = agent  # type: ignore[attr-defined]
    return svc


def _members(s: set) -> list:
    """Snapshot set members WITHOUT triggering __hash__/__eq__ on them
    (``x in s`` would). Iteration is safe — it doesn't hash elements."""
    return list(s)


@pytest.mark.asyncio
async def test_dead_subscriber_eviction_does_not_raise_and_evicts() -> None:
    agent = _FakeAgentProcess()
    dead = _DeadishNetref()
    agent._message_subscribers.add(dead)  # add while "live" (hash works)
    dead.go_dead()  # channel dies — hash now raises

    svc = _make_service(agent)

    # Must NOT raise even though on_message AND __hash__ both explode.
    await svc._notify_subscribers(
        {"type": "user", "role": "user", "content": "hi"}
    )
    for _ in range(10):
        await asyncio.sleep(0.01)

    # The corpse must be gone (checked by identity, not membership test).
    assert all(m is not dead for m in _members(agent._message_subscribers)), (
        "dead subscriber must be evicted by identity after a failed push"
    )


@pytest.mark.asyncio
async def test_live_subscriber_still_receives() -> None:
    """A healthy subscriber must keep getting messages (no over-eviction)."""

    class _LiveSub:
        def __init__(self) -> None:
            self.msgs: list[str] = []
            self._h = id(self)

        async def on_message(self, payload: str) -> None:
            self.msgs.append(payload)

        def __hash__(self) -> int:
            return self._h

        def __eq__(self, other: object) -> bool:
            return self is other

    agent = _FakeAgentProcess()
    live = _LiveSub()
    agent._message_subscribers.add(live)
    svc = _make_service(agent)

    await svc._notify_subscribers(
        {"type": "user", "role": "user", "content": "hello"}
    )
    for _ in range(10):
        await asyncio.sleep(0.01)
        if live.msgs:
            break

    assert live.msgs, "live subscriber must still receive the message"
    assert any(m is live for m in _members(agent._message_subscribers)), (
        "live subscriber must NOT be evicted"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
