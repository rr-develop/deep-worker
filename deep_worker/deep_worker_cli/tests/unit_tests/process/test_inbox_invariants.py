"""The four inbox invariants, asserted together on a live process.

They constrain each other, which is why they are pinned in one place:

1. every record in the inbox is delivered into the agent's context;
2. delivery follows arrival order;
3. a duplicate is refused at publish, so it never enters the inbox at all;
4. the consumer is event-driven — no polling, no timed wakeups.

(3) is the one that could quietly break (1): a ban that refused too eagerly
would drop mail instead of duplicates. Asserting them jointly is what makes
"no duplicates" and "nothing is lost" verifiable as one property rather than
two that can be traded against each other.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.process import inbox_spool

_BOUND = 5.0


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent_data"
    d.mkdir(mode=0o700)
    return d


def _make_proc(data_dir: Path, tmp_path: Path, name: str = "test_invariants"):
    """A real AgentProcess with a real spool; only the turn body is replaced.

    ``data_dir`` is ALWAYS explicit: an AgentProcess built without it would
    default to the real ``deep_worker_home()`` and write into production state.
    """
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path,
        provider="claude-code-sdk",
        model="claude-3-5-sonnet-20241022",
    )
    proc = AgentProcess(
        name=name, config=config, data_dir=data_dir, rpyc_port=False,
    )
    proc.init_inbox_spool()
    return proc


async def _wait_for(predicate, timeout: float = _BOUND, what: str = "condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _collect_turns(proc, delivered: list) -> None:
    async def fake_owned(command, **_kwargs):
        delivered.append(command["message"])
        return {"success": True}

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_every_queued_message_reaches_the_context_in_arrival_order(
    data_dir: Path, tmp_path: Path
) -> None:
    """Nothing in the inbox is skipped, and order is arrival order.

    Ten distinct messages are published before the consumer starts, so the
    drain sees them as one batch and its ORDERING is what is under test, not
    the order in which they happened to be awaited.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    expected = [f"peer -> me :> message {i:02d}" for i in range(10)]
    for text in expected:
        await proc.publish_inbox_message(text, True)

    assert len(inbox_spool.pending_records(data_dir)) == 10

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(delivered) >= 10, what="all ten deliveries"),
            _BOUND,
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert delivered == expected, "delivery must follow arrival (seq) order"
    assert not inbox_spool.pending_records(data_dir), (
        "every delivered record must be unlinked"
    )


@pytest.mark.asyncio
async def test_duplicate_never_enters_the_inbox_and_originals_still_arrive(
    data_dir: Path, tmp_path: Path
) -> None:
    """The ban removes only the duplicate — the queue itself stays intact.

    This is the interaction that matters: refusing a resend must not cost the
    original, nor any unrelated message queued around it.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("peer -> me :> alpha", True)
    await proc.publish_inbox_message("peer -> me :> beta", True)

    # The duplicate is refused BEFORE it becomes a record.
    with pytest.raises(inbox_spool.InboxDuplicateMessageError):
        await proc.publish_inbox_message("peer -> me :> alpha", True)

    assert len(inbox_spool.pending_records(data_dir)) == 2, (
        "the refused duplicate must not be queued, and must not evict anything"
    )

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(delivered) >= 2, what="both deliveries"), _BOUND
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert delivered == ["peer -> me :> alpha", "peer -> me :> beta"]


@pytest.mark.asyncio
async def test_consumer_is_event_driven_not_polling(
    data_dir: Path, tmp_path: Path
) -> None:
    """An idle drain must not wake up on its own, and must wake on arrival.

    Polling is not disproved by reading the loop — a timed wakeup would still
    pass a "does it deliver" test. So this asserts the two halves that
    distinguish the mechanisms: while idle the consumer performs NO directory
    scans at all, and it reacts to a publish without anyone advancing a clock.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    scans = 0
    real_pending = inbox_spool.pending_records

    def counting_pending(d: Path) -> Any:
        nonlocal scans
        scans += 1
        return real_pending(d)

    # agent_process imports inbox_spool INSIDE its functions, so the patch has
    # to land on the source module — that is the object those imports resolve.
    original = inbox_spool.pending_records
    inbox_spool.pending_records = counting_pending  # type: ignore[assignment]
    try:
        task = proc.start_inbox_drain()
        try:
            # Idle: give the loop real time to misbehave if it polls.
            await asyncio.sleep(0.3)
            assert scans == 0, (
                f"an idle drain scanned the spool {scans} times — that is polling"
            )

            # Arrival alone must wake it; nothing else is touched.
            await proc.publish_inbox_message("peer -> me :> wake up", True)
            await asyncio.wait_for(
                _wait_for(lambda: delivered, what="the event-driven delivery"),
                _BOUND,
            )
            assert delivered == ["peer -> me :> wake up"]
            scans_after = scans

            # And it goes back to sleep instead of spinning.
            await asyncio.sleep(0.3)
            assert scans == scans_after, (
                "the drain kept scanning after the queue drained — that is a spin"
            )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    finally:
        inbox_spool.pending_records = original  # type: ignore[assignment]


def test_drain_loop_has_no_timed_wakeup_primitives() -> None:
    """The loop body must contain no sleep/timeout, structurally.

    A behavioural test can only prove a fast poll absent; a slow one (say every
    30s) would slip through any bounded test. Reading the source closes that
    gap permanently.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    source = inspect.getsource(AgentProcess._inbox_drain_loop)
    for banned in ("asyncio.sleep", "wait_for", "call_later", "call_at"):
        assert banned not in source, (
            f"{banned} in the drain loop would make it timed, not event-driven"
        )
    assert "await self._inbox_event.wait()" in source, (
        "the loop must block on the arrival event"
    )
