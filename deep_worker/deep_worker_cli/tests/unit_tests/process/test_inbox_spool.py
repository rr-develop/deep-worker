"""Unit tests for the durable inter-agent spool (design §5, tests 1-8).

Covers the storage layer in isolation: permissions, atomic publish, ordering,
quarantine, tmp cleanup, seq allocation.

See docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md and
a related internal incident analysis (not included here).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from deep_worker_cli.process import inbox_spool


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent_data"
    d.mkdir(mode=0o700)
    return d


def _publish(data_dir: Path, message: str, seq: int) -> Path:
    record = inbox_spool.build_record(message, True)
    return inbox_spool.publish(data_dir, record, seq)


# --- test 1: permissions -------------------------------------------------

def test_published_record_and_dir_are_private(data_dir: Path) -> None:
    """Project policy: never 0666/0777. Messages can carry sensitive text."""
    path = _publish(data_dir, "hello", 0)

    assert path.stat().st_mode & 0o777 == 0o600
    assert inbox_spool.inbox_dir(data_dir).stat().st_mode & 0o777 == 0o700


# --- test 2: atomic publish ----------------------------------------------

def test_tmp_file_is_never_visible_as_a_queue_entry(data_dir: Path) -> None:
    """A torn write must never be picked up as a pending record."""
    inbox = inbox_spool.ensure_spool(data_dir)
    # Simulate an interrupted write: a tmp file with garbage in it.
    (inbox / f".tmp-{'a' * 32}").write_text("{not json")

    assert inbox_spool.pending_records(data_dir) == []


def test_publish_leaves_no_tmp_behind(data_dir: Path) -> None:
    _publish(data_dir, "hello", 0)
    inbox = inbox_spool.inbox_dir(data_dir)

    assert [p.name for p in inbox.iterdir() if p.name.startswith(".tmp-")] == []


# --- test 3: ordering ----------------------------------------------------

def test_pending_records_are_returned_in_seq_order(data_dir: Path) -> None:
    """Lexicographic order is arrival order — the zero-padding must hold
    across a decimal-width boundary (9 → 10)."""
    for seq, msg in enumerate(["first", "second", "third"]):
        _publish(data_dir, msg, seq)
    _publish(data_dir, "tenth", 10)

    bodies = [
        json.loads(p.read_text())["message"]
        for p in inbox_spool.pending_records(data_dir)
    ]
    assert bodies == ["first", "second", "third", "tenth"]


# --- test 4: corrupt quarantine ------------------------------------------

def test_corrupt_record_is_quarantined_and_stops_being_pending(
    data_dir: Path,
) -> None:
    """Design §3.3.5: a corrupt record must not block later records nor wake
    the drain forever."""
    inbox = inbox_spool.ensure_spool(data_dir)
    bad = inbox / f"{0:012d}-{'b' * 32}.json"
    bad.write_text("{definitely not json")
    good = _publish(data_dir, "good", 1)

    assert inbox_spool.read_record(bad) is None
    moved = inbox_spool.quarantine(data_dir, bad)

    assert moved is not None and moved.exists()
    assert not bad.exists()
    # The good record is still pending and unaffected.
    assert inbox_spool.pending_records(data_dir) == [good]


def test_record_without_message_is_treated_as_corrupt(data_dir: Path) -> None:
    """Delivering an empty message is worse than quarantining the record."""
    inbox = inbox_spool.ensure_spool(data_dir)
    path = inbox / f"{0:012d}-{'c' * 32}.json"
    path.write_text(json.dumps({"v": 1, "delivery_id": "x"}))

    assert inbox_spool.read_record(path) is None


# --- schema is ENFORCED, not merely declared -----------------------------

def _valid_body(**overrides) -> dict:
    body = {
        "v": inbox_spool.RECORD_VERSION,
        "delivery_id": "f" * 32,
        "ts": "2026-08-03T12:00:00+00:00",
        "message": "hello",
        "reply_back": True,
    }
    body.update(overrides)
    return body


def _write(data_dir: Path, body: object, seq: int = 0) -> Path:
    inbox = inbox_spool.ensure_spool(data_dir)
    path = inbox / f"{seq:012d}-{'a' * 32}.json"
    path.write_text(json.dumps(body))
    return path


def test_valid_v1_record_is_accepted(data_dir: Path) -> None:
    assert inbox_spool.read_record(_write(data_dir, _valid_body())) is not None


def test_record_from_a_future_version_is_never_delivered(
    data_dir: Path,
) -> None:
    """``v`` exists so a later change can MIGRATE rather than misparse.

    A record written by a newer version must be left alone — guessing at its
    shape and then unlinking it on success would destroy the evidence exactly
    when it matters.
    """
    path = _write(data_dir, _valid_body(v=inbox_spool.RECORD_VERSION + 1))
    assert inbox_spool.read_record(path) is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"v": None}, id="missing-version"),
        pytest.param({"delivery_id": None}, id="missing-delivery-id"),
        pytest.param({"delivery_id": "not-a-uuid"}, id="malformed-delivery-id"),
        pytest.param({"delivery_id": ""}, id="empty-delivery-id"),
        pytest.param({"ts": 12345}, id="non-string-ts"),
        pytest.param({"reply_back": "yes"}, id="non-bool-reply-back"),
        pytest.param({"message": None}, id="null-message"),
    ],
)
def test_schema_invalid_records_are_rejected(
    data_dir: Path, overrides: dict
) -> None:
    """Fail closed. A record that does not match the v1 schema is quarantined
    by the drain rather than delivered — otherwise the stable-delivery_id
    guarantee that makes duplicates diagnosable would be unenforceable."""
    body = _valid_body()
    for key, value in overrides.items():
        if value is None and key in ("v", "delivery_id"):
            body.pop(key, None)
        else:
            body[key] = value

    assert inbox_spool.read_record(_write(data_dir, body)) is None


def test_non_dict_record_is_rejected(data_dir: Path) -> None:
    assert inbox_spool.read_record(_write(data_dir, ["not", "a", "dict"])) is None


# --- test 5: survives a fresh reader --------------------------------------

def test_record_is_readable_by_a_completely_fresh_reader(
    data_dir: Path,
) -> None:
    """Stand-in for abrupt death: nothing is held in memory."""
    path = _publish(data_dir, "payload", 7)
    body = inbox_spool.read_record(Path(str(path)))

    assert body is not None
    assert body["message"] == "payload"
    assert body["v"] == inbox_spool.RECORD_VERSION
    assert body["delivery_id"]
    # No sender field: it is not on the wire (design §3.1.2).
    assert "sender" not in body


# --- test 6: tmp cleanup at startup --------------------------------------

def test_cleanup_tmp_removes_leftovers_only(data_dir: Path) -> None:
    inbox = inbox_spool.ensure_spool(data_dir)
    (inbox / f".tmp-{'d' * 32}").write_text("partial")
    keep = _publish(data_dir, "keep", 0)

    removed = inbox_spool.cleanup_tmp(data_dir)

    assert removed == 1
    assert keep.exists()
    assert inbox_spool.pending_records(data_dir) == [keep]


# --- test 7: next_seq spans pending AND quarantined ------------------------

def test_next_seq_accounts_for_quarantined_records(data_dir: Path) -> None:
    """A quarantined record must never have its seq reused, or a later
    arrival could collide with it by name."""
    inbox = inbox_spool.ensure_spool(data_dir)
    bad = inbox / f"{41:012d}-{'e' * 32}.json"
    bad.write_text("{nope")
    inbox_spool.quarantine(data_dir, bad)
    _publish(data_dir, "pending", 3)

    # max(pending=3, quarantined=41) + 1
    assert inbox_spool.next_seq(data_dir) == 42


def test_next_seq_on_empty_spool_is_zero(data_dir: Path) -> None:
    assert inbox_spool.next_seq(data_dir) == 0


# --- test 8: concurrent allocation ----------------------------------------

@pytest.mark.asyncio
async def test_concurrent_enqueues_never_overwrite_each_other(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.1.1 / review MAJOR 1: allocate+publish is serialised, so
    concurrent arrivals get distinct seqs and no record is clobbered.

    Uses the REAL AgentProcess publish path and asserts the lock-ownership
    invariant directly (see the comment in the body for why provoking a race
    from outside is not possible).
    """
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path,
        provider="claude-code",
        model="claude-3-5-sonnet-20241022",
    )
    proc = AgentProcess(
        name="test_seq_alloc", config=config, data_dir=data_dir,
        rpyc_port=False,
    )
    proc.init_inbox_spool()

    # Design §3.1.1: allocation of _inbox_next_seq and the publish that
    # consumes it must be ONE critical section under _inbox_seq_lock.
    #
    # We assert the INVARIANT directly — the lock is held at the moment
    # publish() runs — rather than trying to provoke a race. Provoking one is
    # not possible from the outside: publish() is synchronous, so a task never
    # yields between reading the counter and writing the file, and any
    # interleaving would have to be introduced by the implementation itself.
    # An outcome-only check (all seqs distinct) would therefore pass for a
    # lock-less implementation too, which is exactly the weakness the
    # `seq_outside_lock` mutation exposed in the previous version of this test.
    real_publish = inbox_spool.publish
    lock_state: list[bool] = []
    arrived: list[int] = []

    def checking_publish(dd, record, seq):
        lock_state.append(proc._inbox_seq_lock.locked())
        arrived.append(seq)
        return real_publish(dd, record, seq)

    inbox_spool.publish = checking_publish  # type: ignore[assignment]

    messages = [f"msg-{i}" for i in range(25)]
    try:
        ids = await asyncio.gather(
            *(proc.publish_inbox_message(m, True) for m in messages)
        )
    finally:
        inbox_spool.publish = real_publish  # type: ignore[assignment]

    assert lock_state and all(lock_state), (
        "publish ran without _inbox_seq_lock held — allocation and publish "
        "must be one critical section (design §3.1.1)"
    )
    assert len(set(arrived)) == len(messages), (
        f"the same seq was handed to more than one publisher: {sorted(arrived)}"
    )

    assert len(set(ids)) == len(messages), "delivery_ids must be unique"
    records = inbox_spool.pending_records(data_dir)
    assert len(records) == len(messages), (
        f"a record was overwritten: {len(records)} of {len(messages)} survived"
    )
    stored = {json.loads(p.read_text())["message"] for p in records}
    assert stored == set(messages)
    # Every seq is distinct — the allocation itself must be serialised.
    seqs = [p.name.split("-", 1)[0] for p in records]
    assert len(set(seqs)) == len(messages), f"duplicate seq allocated: {seqs}"


@pytest.mark.asyncio
async def test_failed_priming_cannot_publish_seq_zero_over_survivors(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transiently failed startup scan must never yield a misordered ack.

    Regression for the round-3 review finding: ``start()`` logs and swallows an
    ``OSError`` from the startup prime so a broken spool cannot stop the agent
    booting. If ``publish`` then allocated from the unprimed default, a new
    arrival would be written as seq 0 while a survivor sits at seq 7 — silently
    jumping ahead of it in attempt order while the sender is told
    ``success: true``.

    Contract: prime on demand inside the lock; on success allocate correctly,
    on failure raise so the ack is ``success: false``.
    """
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path, provider="claude-code",
        model="claude-3-5-sonnet-20241022",
    )
    # A record from a previous life survives at seq 7.
    _publish(data_dir, "survivor", 7)

    proc = AgentProcess(
        name="test_prime_failure", config=config, data_dir=data_dir,
        rpyc_port=False,
    )

    def broken_scan(_data_dir):
        raise OSError("transient filesystem failure")

    monkeypatch.setattr(inbox_spool, "next_seq", broken_scan)
    # start() swallows this, exactly as production does.
    with pytest.raises(OSError):
        proc.init_inbox_spool()
    assert proc._inbox_seq_primed is False
    assert proc._inbox_next_seq == 0

    # While the spool is still broken the enqueue must FAIL LOUD, not ack.
    with pytest.raises(OSError):
        await proc.publish_inbox_message("must not be acked", True)
    assert len(inbox_spool.pending_records(data_dir)) == 1, (
        "no record may be written while the counter is unprimed"
    )

    # Once the filesystem recovers, the late prime must allocate ABOVE the
    # survivor — never seq 0.
    monkeypatch.undo()
    await proc.publish_inbox_message("new arrival", True)

    names = [p.name for p in inbox_spool.pending_records(data_dir)]
    assert len(names) == 2
    assert not any(n.startswith("000000000000-") for n in names), (
        f"a new arrival was published as seq 0 over a survivor: {names}"
    )
    assert names[1].startswith("000000000008-"), names
    assert proc._inbox_seq_primed is True


@pytest.mark.asyncio
async def test_second_init_does_not_reset_a_live_counter(
    data_dir: Path, tmp_path: Path
) -> None:
    """``init_inbox_spool`` runs twice in production (once before the RPyC
    server starts, once from ``bring_up_inbox``). The second call must not
    rescan and reset a counter that has already handed out seqs, or a later
    arrival could reuse a filename."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path, provider="claude-code",
        model="claude-3-5-sonnet-20241022",
    )
    proc = AgentProcess(
        name="test_double_init", config=config, data_dir=data_dir,
        rpyc_port=False,
    )
    proc.init_inbox_spool()
    await proc.publish_inbox_message("first", True)
    counter_after_publish = proc._inbox_next_seq

    # Second production call — must be a no-op for the counter.
    proc.init_inbox_spool()
    assert proc._inbox_next_seq == counter_after_publish

    await proc.publish_inbox_message("second", True)
    names = [p.name for p in inbox_spool.pending_records(data_dir)]
    assert len(names) == 2, f"a filename was reused: {names}"


@pytest.mark.asyncio
async def test_seq_continues_after_restart(data_dir: Path, tmp_path: Path) -> None:
    """A new process must not reuse seqs of surviving records."""
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path,
        provider="claude-code",
        model="claude-3-5-sonnet-20241022",
    )

    def _make() -> AgentProcess:
        return AgentProcess(
            name="test_seq_restart", config=config, data_dir=data_dir,
            rpyc_port=False,
        )

    first = _make()
    first.init_inbox_spool()
    await first.publish_inbox_message("before restart", True)

    # New process over the same spool.
    second = _make()
    second.init_inbox_spool()
    await second.publish_inbox_message("after restart", True)

    records = inbox_spool.pending_records(data_dir)
    assert len(records) == 2, "the surviving record was clobbered"
    assert [json.loads(p.read_text())["message"] for p in records] == [
        "before restart",
        "after restart",
    ]
