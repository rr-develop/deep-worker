"""Unit tests for the durable-inbox drain (design §5, tests 9-19).

These pin the behaviours the design review fought hardest over, each of which
was a real defect caught before implementation:

* test 9  — the drain must hold NO lock (deadlock guard)
* test 10 — wakeup protocol: clear-before-scan, no self-rearm
* test 12 — a user interrupt must kill the TURN, never the consumer
* test 13 — a failing record must not spin (hot retry loop)
* test 16 — attempt order, with overtake explicitly ALLOWED
* test 17 — task creation and callback injection are separate concerns
* test 18 — per-record fault containment (no head-of-line stranding)
* test 19 — lifecycle: shutdown cancels+awaits before the checkpointer closes

See docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md (signed off, r6).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.process import inbox_spool

pytestmark = pytest.mark.asyncio

# Every wait is bounded: a drain test that hangs is worse than one that fails.
_BOUND = 5.0


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent_data"
    d.mkdir(mode=0o700)
    return d


def _make_proc(data_dir: Path, tmp_path: Path, name: str = "test_drain"):
    """A real AgentProcess with a real spool; only the turn body is replaced."""
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


def _messages_on_disk(data_dir: Path) -> list[str]:
    return [
        json.loads(p.read_text())["message"]
        for p in inbox_spool.pending_records(data_dir)
    ]


# --- test 9: the drain must hold no lock ---------------------------------

async def test_drain_holds_no_lock_and_turn_acquires_it(
    data_dir: Path, tmp_path: Path
) -> None:
    """The turn runs via the PUBLIC _handle_invoke, which takes _invoke_lock
    itself. If the drain also held it, this would deadlock — hence the bound.

    Mutation guard: wrapping the child in `async with _invoke_lock` makes this
    time out instead of hanging production.
    """
    proc = _make_proc(data_dir, tmp_path)
    seen_locked: list[bool] = []

    async def fake_owned(command, **kwargs):
        # _handle_invoke acquired the lock before delegating here.
        seen_locked.append(proc._invoke_lock.locked())
        return {"success": True}

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]
    await proc.publish_inbox_message("m", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: seen_locked, what="the turn to run"), _BOUND
        )
        assert seen_locked == [True], "the public path must hold the lock"
        assert not proc._invoke_lock.locked(), "drain must not retain the lock"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 10: wakeup protocol / tail race --------------------------------

async def test_tail_arrival_is_processed_because_producer_sets_the_event(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.4.1: a record published right after the drain took its
    snapshot is still processed — because the PRODUCER sets the event, not
    because the consumer re-arms on finding a pending file.

    Formulated this way deliberately: an unconditional self-rearm (the removed
    hot-loop defect) cannot satisfy it.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    snapshot_taken = asyncio.Event()
    release = asyncio.Event()
    first = True

    async def fake_owned(command, **kwargs):
        nonlocal first
        delivered.append(command["message"])
        if first:
            first = False
            # The pass has its snapshot; publish the tail arrival now.
            snapshot_taken.set()
            await release.wait()
        return {"success": True}

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]
    await proc.publish_inbox_message("first", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(snapshot_taken.wait(), _BOUND)
        # Tail arrival: published AFTER the current snapshot was taken.
        await proc.publish_inbox_message("tail", True)
        release.set()

        await asyncio.wait_for(
            _wait_for(lambda: "tail" in delivered, what="the tail arrival"),
            _BOUND,
        )
        assert delivered == ["first", "tail"]
        assert _messages_on_disk(data_dir) == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_drain_parks_when_spool_is_empty(
    data_dir: Path, tmp_path: Path
) -> None:
    """No polling: with nothing pending the consumer blocks on the event."""
    proc = _make_proc(data_dir, tmp_path)
    task = proc.start_inbox_drain()
    try:
        await asyncio.sleep(0.05)
        assert not task.done()
        assert not proc._inbox_event.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 11: delete boundary --------------------------------------------

async def test_record_is_unlinked_only_on_success(
    data_dir: Path, tmp_path: Path
) -> None:
    proc = _make_proc(data_dir, tmp_path)

    async def ok(command, **kwargs):
        return {"success": True}

    proc._handle_invoke_owned = ok  # type: ignore[assignment]
    await proc.publish_inbox_message("delivered", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: _messages_on_disk(data_dir) == [],
                      what="the record to be unlinked"),
            _BOUND,
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 13: no hot retry loop ------------------------------------------

async def test_failed_record_is_attempted_once_per_wake_and_does_not_spin(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.3.1a / review BLOCKER: a deterministic ``success: false``
    must produce EXACTLY ONE attempt and then park — never a self-sustaining
    loop of LLM turns.

    Mutation guard: restoring the unconditional self-rearm makes the attempt
    count grow without bound.
    """
    proc = _make_proc(data_dir, tmp_path)
    attempts: list[str] = []

    async def always_fails(command, **kwargs):
        attempts.append(command["message"])
        return {"success": False, "error": "deterministic failure"}

    proc._handle_invoke_owned = always_fails  # type: ignore[assignment]
    await proc.publish_inbox_message("doomed", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: attempts, what="the first attempt"), _BOUND
        )
        # Give a spin every chance to manifest.
        await asyncio.sleep(0.3)

        assert attempts == ["doomed"], f"expected one attempt, got {attempts}"
        assert not proc._inbox_event.is_set(), "must not self-rearm"
        assert _messages_on_disk(data_dir) == ["doomed"], "record must be kept"
        assert not task.done(), "consumer must stay alive"

        # A NEW arrival is a legitimate external edge: it retries the old
        # record too.
        #
        # ORDER UPDATED by DESIGN_INBOX_STALE_REDELIVERY.md §3.3: the fresh
        # record is attempted BEFORE the previously-attempted one. Walking
        # plain seq order meant a retained record always went first, so a stale
        # message perpetually overtook current traffic — the defect that design
        # fixes. What THIS test guards is unchanged and still asserted: exactly
        # one attempt per wake (no spin), and the retained record is retried on
        # a genuine external edge rather than being dropped.
        await proc.publish_inbox_message("fresh", True)
        await asyncio.wait_for(
            _wait_for(lambda: len(attempts) >= 3, what="retry on new arrival"),
            _BOUND,
        )
        assert attempts == ["doomed", "fresh", "doomed"], (
            "fresh mail must not queue behind a known-failing record"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 16: attempt order, overtake allowed -----------------------------

async def test_attempts_begin_in_seq_order_and_overtake_is_allowed(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.3.0a: FIFO is NOT promised. A retained failure may be
    overtaken by a later record; this test asserts that as ALLOWED so it can
    never drift back to claiming FIFO delivery."""
    proc = _make_proc(data_dir, tmp_path)
    attempts: list[str] = []

    async def first_fails(command, **kwargs):
        msg = command["message"]
        attempts.append(msg)
        return {"success": msg != "seq1"}

    proc._handle_invoke_owned = first_fails  # type: ignore[assignment]
    await proc.publish_inbox_message("seq1", True)
    await proc.publish_inbox_message("seq2", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(attempts) >= 2, what="both attempts"), _BOUND
        )
        # Attempts began in seq order...
        assert attempts == ["seq1", "seq2"]
        # ...but seq2 was DELIVERED while seq1 is still queued: overtake.
        assert _messages_on_disk(data_dir) == ["seq1"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 17: ownership seam ---------------------------------------------

async def test_drain_starts_without_callback_and_late_injection_is_used(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.3.3 / review MAJOR 1 (round 3): task creation and callback
    injection are SEPARATE, so the idempotence guard can never discard a late
    callback, and the drain re-reads it per record."""
    proc = _make_proc(data_dir, tmp_path)
    seen: list[Any] = []

    async def capture(command, **kwargs):
        seen.append(kwargs.get("notify_callback"))
        return {"success": True}

    proc._handle_invoke_owned = capture  # type: ignore[assignment]
    await proc.publish_inbox_message("before", True)

    task = proc.start_inbox_drain()
    try:
        # 1. Started with no callback at all — still processes the record.
        await asyncio.wait_for(
            _wait_for(lambda: len(seen) >= 1, what="the pre-injection record"),
            _BOUND,
        )
        assert seen[0] is None

        # 2. Late injection is picked up by the NEXT record, no restart.
        sentinel = object()
        proc.set_inbox_notify(sentinel)
        await proc.publish_inbox_message("after", True)
        await asyncio.wait_for(
            _wait_for(lambda: len(seen) >= 2, what="the post-injection record"),
            _BOUND,
        )
        assert seen[1] is sentinel

        # 3. Task identity unchanged, and a second start is a no-op that does
        #    NOT clear the callback.
        again = proc.start_inbox_drain()
        assert again is task
        assert proc._inbox_notify is sentinel
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 18: per-record fault containment --------------------------------

async def test_one_raising_record_does_not_strand_the_rest_of_the_batch(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.3.4 / review MAJOR 3 (round 3): an unexpected exception on
    seq1 must not abandon seq2..N from the same already-triggered snapshot.

    Mutation guard: moving the try/except from per-record to per-pass strands
    seq2 here.
    """
    proc = _make_proc(data_dir, tmp_path)
    attempts: list[str] = []

    async def boom_on_first(command, **kwargs):
        msg = command["message"]
        attempts.append(msg)
        if msg == "seq1":
            raise RuntimeError("unexpected explosion")
        return {"success": True}

    proc._handle_invoke_owned = boom_on_first  # type: ignore[assignment]
    await proc.publish_inbox_message("seq1", True)
    await proc.publish_inbox_message("seq2", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(attempts) >= 2, what="both records tried"),
            _BOUND,
        )
        assert attempts == ["seq1", "seq2"]
        # seq1 retained, seq2 delivered, consumer alive and parked.
        assert _messages_on_disk(data_dir) == ["seq1"]
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 15: startup edge ------------------------------------------------

async def test_records_pending_at_startup_are_drained_without_new_arrival(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.4 goal 4: whatever survived the restart is processed on the
    startup edge, with no inbound message to trigger it."""
    # A previous life left a record behind.
    seed = _make_proc(data_dir, tmp_path, name="test_startup_seed")
    await seed.publish_inbox_message("left over", True)

    # A brand-new process over the same spool.
    proc = _make_proc(data_dir, tmp_path, name="test_startup_seed")
    delivered: list[str] = []

    async def ok(command, **kwargs):
        delivered.append(command["message"])
        return {"success": True}

    proc._handle_invoke_owned = ok  # type: ignore[assignment]

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: delivered, what="the recovered record"), _BOUND
        )
        assert delivered == ["left over"]
        assert _messages_on_disk(data_dir) == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_schema_invalid_record_is_quarantined_never_delivered(
    data_dir: Path, tmp_path: Path
) -> None:
    """A record that fails schema validation must be quarantined, not run.

    Covers the future-version case specifically: delivering a record written
    by a newer agent (and unlinking it on success) would silently destroy a
    message the current code cannot correctly interpret.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []

    async def capture(command, **kwargs):
        delivered.append(command["message"])
        return {"success": True}

    proc._handle_invoke_owned = capture  # type: ignore[assignment]

    inbox = inbox_spool.ensure_spool(data_dir)
    future = inbox / f"{0:012d}-{'a' * 32}.json"
    future.write_text(json.dumps({
        "v": inbox_spool.RECORD_VERSION + 1,
        "delivery_id": "f" * 32,
        "ts": "2026-08-03T12:00:00+00:00",
        "message": "from the future",
        "reply_back": True,
    }))
    # A valid record behind it must still be delivered.
    await proc.publish_inbox_message("current", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: delivered, what="the valid record"), _BOUND
        )
        await asyncio.sleep(0.1)

        assert delivered == ["current"], "future-version record must not run"
        assert not future.exists(), "invalid record must leave the pending set"
        quarantined = inbox_spool.corrupt_dir(data_dir)
        assert quarantined.is_dir() and list(quarantined.iterdir()), (
            "the record must be preserved in quarantine, not deleted"
        )
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 14: delivery_id reaches the message ----------------------------

async def test_delivery_id_is_threaded_into_the_turn(
    data_dir: Path, tmp_path: Path
) -> None:
    """A duplicate produced by a crash between checkpoint and unlink must be
    linkable to its original spool record."""
    proc = _make_proc(data_dir, tmp_path)
    commands: list[dict] = []

    async def capture(command, **kwargs):
        commands.append(command)
        return {"success": True}

    proc._handle_invoke_owned = capture  # type: ignore[assignment]
    delivery_id = await proc.publish_inbox_message("m", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: commands, what="the turn"), _BOUND
        )
        assert commands[0]["delivery_id"] == delivery_id
        assert commands[0]["inter_agent_delivery"] is True
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
