"""Stale inter-agent re-delivery: dedup ledger, attempt cap, TTL, ordering.

Design: ``docs/DESIGN_INBOX_STALE_REDELIVERY.md`` (signed off, r4).
See also: a related internal incident analysis (not included here).

The defect these pin: the spool guaranteed at-least-once with NO bound on the
"more than once" half. A record that was delivered but lost its ``unlink``
replayed verbatim, ahead of fresh traffic, forever. Observed live on
a production agent: two records from 2026-08-14 whose ``delivery_id``s were already in
``threads.db`` — the turns had run; only the unlink was lost.

Every test drives the PUBLIC path (``publish_inbox_message`` + the real drain);
none reaches into private ledger helpers to assert. Every wait is bounded — a
drain test that hangs is worse than one that fails.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.process import inbox_spool

pytestmark = pytest.mark.asyncio

_BOUND = 5.0


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent_data"
    d.mkdir(mode=0o700)
    return d


def _make_proc(data_dir: Path, tmp_path: Path, name: str = "test_redelivery"):
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


async def _drain_once(proc, expect: int, delivered: list, what: str) -> None:
    """Run the drain until ``expect`` turns have happened, then stop it."""
    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(delivered) >= expect, what=what), _BOUND
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _collect_turns(proc, delivered: list, result: Any = None) -> None:
    """Replace only the turn body, recording each message handed to a turn."""
    async def fake_owned(command, **kwargs):
        delivered.append(command["message"])
        return {"success": True} if result is None else result

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]


def _pending(data_dir: Path) -> list[Path]:
    return inbox_spool.pending_records(data_dir)


def _expired(data_dir: Path) -> list[Path]:
    d = inbox_spool.inbox_dir(data_dir) / "expired"
    return sorted(d.iterdir()) if d.is_dir() else []


def _restore(path: Path, body: dict[str, Any]) -> None:
    """Re-create a record file, simulating an unlink that never reached disk."""
    path.write_text(json.dumps(body), encoding="utf-8")
    os.chmod(path, 0o600)


# --- test 1: the core defect ---------------------------------------------

async def test_delivered_record_is_not_replayed(
    data_dir: Path, tmp_path: Path
) -> None:
    """A record whose unlink was lost must NOT be delivered a second time.

    This is the observed production failure exactly: the turn ran, the record survived.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("verdict A", True)
    record_path = _pending(data_dir)[0]
    body = json.loads(record_path.read_text())

    await _drain_once(proc, 1, delivered, "the first delivery")
    assert delivered == ["verdict A"]
    assert not _pending(data_dir), "a delivered record must be unlinked"

    # Simulate the lost unlink, then wake the drain with a fresh arrival.
    _restore(record_path, body)
    await proc.publish_inbox_message("verdict B", True)
    await _drain_once(proc, 2, delivered, "the second delivery")

    assert delivered == ["verdict A", "verdict B"], (
        "the resurrected record must be suppressed, not replayed"
    )
    assert not _pending(data_dir), "the suppressed record must be unlinked"


# --- test 2: dedup is on disk, not in memory ------------------------------

async def test_dedup_survives_restart(data_dir: Path, tmp_path: Path) -> None:
    """A fresh process over the same data_dir must still suppress."""
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("once", True)
    record_path = _pending(data_dir)[0]
    body = json.loads(record_path.read_text())
    await _drain_once(proc, 1, delivered, "the first delivery")
    assert delivered == ["once"]

    _restore(record_path, body)

    proc2 = _make_proc(data_dir, tmp_path, name="test_redelivery_2")
    delivered2: list[str] = []
    _collect_turns(proc2, delivered2)
    await proc2.publish_inbox_message("wake", True)
    await _drain_once(proc2, 1, delivered2, "the post-restart delivery")

    assert delivered2 == ["wake"], "the ledger must be read from disk"


# --- test 3: the production shape -------------------------------------------

async def test_stale_record_retired_not_delivered(
    data_dir: Path, tmp_path: Path, caplog
) -> None:
    """A 13-day-old record (no counter keys) is retired, never delivered."""
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("ancient CHANGES REQUESTED", True)
    path = _pending(data_dir)[0]
    body = json.loads(path.read_text())
    body["ts"] = (
        datetime.now(timezone.utc) - timedelta(days=13)
    ).isoformat()
    body.pop("attempted", None)
    body.pop("failure_attempts", None)
    _restore(path, body)

    with caplog.at_level("WARNING"):
        proc2 = _make_proc(data_dir, tmp_path, name="test_redelivery_sweep")
        _collect_turns(proc2, delivered)
        await proc2.publish_inbox_message("fresh", True)
        await _drain_once(proc2, 1, delivered, "the fresh delivery")

    assert delivered == ["fresh"], "a stale record must never reach a turn"
    assert len(_expired(data_dir)) == 1, "it must be retired to expired/"
    text = caplog.text
    assert "RETIRED" in text
    assert body["delivery_id"] in text, "log must carry the delivery_id"


# --- test 4/5: the two counters are independent ---------------------------

async def test_failure_cap_retires_after_max_failures(
    data_dir: Path, tmp_path: Path
) -> None:
    """success=False five times retires the record instead of keeping it."""
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered, result={"success": False})

    await proc.publish_inbox_message("doomed", True)
    for i in range(inbox_spool.MAX_FAILURES):
        await proc.publish_inbox_message(f"wake {i}", True)
        task = proc.start_inbox_drain()
        try:
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert len(_expired(data_dir)) >= 1, (
        "a record failing MAX_FAILURES times must be retired, not retained"
    )


async def test_interrupt_sets_attempted_but_not_failure_attempts(
    data_dir: Path, tmp_path: Path
) -> None:
    """A user interrupt marks the record attempted WITHOUT charging retirement.

    Merging the two counters (the r1 design defect) breaks one half or the
    other: either an interrupt expires a message nobody read, or an interrupted
    record stays 'fresh' forever and overtakes new traffic.
    """
    proc = _make_proc(data_dir, tmp_path)

    async def interrupted_turn(command, **kwargs):
        raise asyncio.CancelledError()

    proc._handle_invoke_owned = interrupted_turn  # type: ignore[assignment]
    await proc.publish_inbox_message("interrupted", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(
                lambda: json.loads(
                    _pending(data_dir)[0].read_text()
                ).get("attempted"),
                what="the record to be marked attempted",
            ),
            _BOUND,
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    body = json.loads(_pending(data_dir)[0].read_text())
    assert body["attempted"] is True, "an interrupt must demote from the fresh lane"
    assert body["failure_attempts"] == 0, (
        "an interrupt must NOT count toward retirement"
    )


# --- test 6/12: ordering --------------------------------------------------

async def test_interrupted_record_does_not_overtake_new_message(
    data_dir: Path, tmp_path: Path
) -> None:
    """The r1 design defect, as a test: interrupt A, publish B, B goes first."""
    proc = _make_proc(data_dir, tmp_path)

    async def interrupted_turn(command, **kwargs):
        raise asyncio.CancelledError()

    proc._handle_invoke_owned = interrupted_turn  # type: ignore[assignment]
    await proc.publish_inbox_message("A (interrupted, older seq)", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(
                lambda: json.loads(
                    _pending(data_dir)[0].read_text()
                ).get("attempted"),
                what="A to be attempted",
            ),
            _BOUND,
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    delivered: list[str] = []
    _collect_turns(proc, delivered)
    await proc.publish_inbox_message("B (fresh, newer seq)", True)
    await _drain_once(proc, 2, delivered, "both records")

    assert delivered[0].startswith("B ("), (
        "a fresh record must be attempted before a previously-attempted one, "
        f"got {delivered!r}"
    )


# --- test 7: foreign files ------------------------------------------------

async def test_foreign_file_is_reported_and_left_in_place(
    data_dir: Path, tmp_path: Path, caplog
) -> None:
    """A file another component wrote into the spool is logged, not quarantined."""
    proc = _make_proc(data_dir, tmp_path)
    foreign = inbox_spool.inbox_dir(data_dir) / "review_request.md"
    foreign.write_text("# REVIEW REQUEST\n", encoding="utf-8")
    os.chmod(foreign, 0o644)

    delivered: list[str] = []
    _collect_turns(proc, delivered)

    with caplog.at_level("WARNING"):
        await proc.publish_inbox_message("real", True)
        await _drain_once(proc, 1, delivered, "the real delivery")

    assert delivered == ["real"], "a foreign file must never reach a turn"
    assert foreign.exists(), "a foreign file must be LEFT IN PLACE, not moved"
    corrupt = inbox_spool.corrupt_dir(data_dir)
    assert not (corrupt / "review_request.md").exists(), (
        "a foreign file must not be silently quarantined"
    )
    assert "foreign file" in caplog.text.lower()


# --- test 8: stale marking ------------------------------------------------

async def test_stale_delivery_is_marked_in_body(
    data_dir: Path, tmp_path: Path
) -> None:
    """A delivered-but-old record reaches the turn with a staleness prefix."""
    proc = _make_proc(data_dir, tmp_path)
    await proc.publish_inbox_message("agent_a -> me :> SIGN OFF", True)
    path = _pending(data_dir)[0]
    body = json.loads(path.read_text())
    body["ts"] = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    _restore(path, body)

    delivered: list[str] = []
    proc2 = _make_proc(data_dir, tmp_path, name="test_redelivery_stale")
    _collect_turns(proc2, delivered)
    await _drain_once(proc2, 1, delivered, "the stale delivery")

    assert delivered, "an old-but-not-expired record must still be delivered"
    assert "STALE MESSAGE" in delivered[0], (
        "the model must be able to see the message is stale"
    )
    assert "agent_a -> me :> SIGN OFF" in delivered[0], (
        "the original body must be preserved after the prefix"
    )


# --- test 15: the prefix must not break reply routing ---------------------

@pytest.mark.asyncio(loop_scope="function")
async def test_stale_prefix_preserves_reply_routing() -> None:
    """Sender extraction must survive the stale prefix.

    inter_agent_reply.py scans PER LINE, so a prefix line is tolerated — but
    both helpers keep a ``^``-anchored whole-content fallback. If a refactor
    ever promotes that to the only path, reply routing breaks silently and the
    peer conversation dies with no error. This pins the seam.
    """
    from langchain_core.messages import HumanMessage

    from deep_worker_cli.middleware.inter_agent_reply import (
        _extract_sender,
        _is_inter_agent_message,
    )

    prefixed = (
        "[STALE MESSAGE — queued 13d2h ago, delivered late; may be obsolete]\n"
        "agent_a -> agent_b :> DESIGN SIGN OFF"
    )
    msg = HumanMessage(content=prefixed)

    assert _is_inter_agent_message(msg, "agent_b") is True
    assert _extract_sender(prefixed) == "agent_a"


# --- tests 9/10/16/17: ledger durability ----------------------------------

async def test_initial_ledger_create_fsyncs_directory_before_unlink(
    data_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """FIRST delivery: the inbox DIRECTORY must be fsynced before the unlink.

    fsync on a newly created file does not make its directory entry durable.
    Without the dir fsync, `create -> write+fsync -> unlink -> crash` recovers
    with the record gone and the ledger pathname absent: SILENT LOSS.
    """
    proc = _make_proc(data_dir, tmp_path)
    inbox = inbox_spool.inbox_dir(data_dir)

    events: list[str] = []
    real_fsync = os.fsync
    real_unlink = os.unlink

    def spy_fsync(fd: int) -> None:
        try:
            if os.path.samestat(os.fstat(fd), os.stat(inbox)):
                events.append("fsync_dir")
        except OSError:
            pass
        real_fsync(fd)

    def spy_unlink(path, *a, **kw):
        # Scope to THIS test's own inbox. os.unlink is global, so an unrelated
        # test's still-running drain task can unlink its own record while we
        # are patched — an event from another spool would corrupt the ordering
        # assertion below (and only under full-suite ordering, which is the
        # worst kind of flake to debug).
        p = Path(path)
        if p.suffix == ".json" and p.parent == inbox:
            events.append("unlink_record")
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "unlink", spy_unlink)

    delivered: list[str] = []
    _collect_turns(proc, delivered)
    await proc.publish_inbox_message("first ever", True)
    events.clear()  # ignore the publish chain; we care about the delivery
    await _drain_once(proc, 1, delivered, "the first delivery")

    assert "fsync_dir" in events, (
        "creating delivered.log must fsync the inbox directory"
    )
    assert "unlink_record" in events
    assert events.index("fsync_dir") < events.index("unlink_record"), (
        "the directory entry must be durable BEFORE the record is unlinked"
    )


async def test_existing_ledger_append_does_not_fsync_directory(
    data_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """Steady state: the ledger already exists, so no dir fsync is needed.

    Pins the create/open branches as genuinely distinct. Without this, an
    implementation that unconditionally dir-fsyncs (or never does) passes the
    create-branch test on its own.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    inbox = inbox_spool.inbox_dir(data_dir)
    dir_fsyncs: list[str] = []
    real_fsync = os.fsync
    phase = "create"

    def spy_fsync(fd: int) -> None:
        try:
            if os.path.samestat(os.fstat(fd), os.stat(inbox)):
                dir_fsyncs.append(phase)
        except OSError:
            pass
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)

    # Phase 1 — the ledger does not exist yet; the create branch MUST dir-fsync.
    await proc.publish_inbox_message("creates the ledger", True)
    dir_fsyncs.clear()  # ignore the publish chain's own dir fsync
    await _drain_once(proc, 1, delivered, "the first delivery")
    assert "create" in dir_fsyncs, (
        "the create branch must fsync the directory (guards against a test "
        "that would pass simply because nothing ever fsyncs)"
    )

    # Phase 2 — the ledger now exists; the append branch must NOT dir-fsync.
    phase = "append"
    await proc.publish_inbox_message("second", True)
    dir_fsyncs.clear()
    await _drain_once(proc, 2, delivered, "the second delivery")

    assert "append" not in dir_fsyncs, (
        "appending to an EXISTING ledger must not fsync the directory"
    )


async def test_incomplete_ledger_append_leaves_record_pending(
    data_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """If the append cannot be confirmed, the record must NOT be unlinked.

    Failing this way costs one visible duplicate; failing the other way loses
    the message silently.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    real_fsync = os.fsync
    ledger = inbox_spool.ledger_path(data_dir)

    def failing_fsync(fd: int) -> None:
        # Fail ONLY this test's ledger. os.fsync is global: matching broadly
        # (any non-empty non-directory fd) also breaks writes belonging to
        # other tests running against their own spools, which corrupts this
        # test's state under full-suite ordering.
        try:
            same = ledger.exists() and os.path.samestat(
                os.fstat(fd), os.stat(ledger)
            )
        except OSError:
            same = False
        if same:
            raise OSError("simulated fsync failure")
        real_fsync(fd)

    await proc.publish_inbox_message("must survive", True)
    monkeypatch.setattr(os, "fsync", failing_fsync)
    await _drain_once(proc, 1, delivered, "the attempted delivery")

    assert _pending(data_dir), (
        "an unconfirmed ledger append must leave the record pending"
    )


async def test_torn_final_ledger_line_is_dropped(
    data_dir: Path, tmp_path: Path
) -> None:
    """A torn last line is discarded — safe only because its record is pending.

    An incomplete append means the unlink never ran, so dropping the id costs
    at most one replay of a record that still exists, never a loss.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("a", True)
    await _drain_once(proc, 1, delivered, "a delivery to create the ledger")

    ledger = inbox_spool.ledger_path(data_dir)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write("deadbeef" * 3)  # 24 chars, no newline: torn

    proc2 = _make_proc(data_dir, tmp_path, name="test_redelivery_torn")
    delivered2: list[str] = []
    _collect_turns(proc2, delivered2)
    await proc2.publish_inbox_message("b", True)
    await _drain_once(proc2, 1, delivered2, "delivery after a torn ledger")

    assert delivered2 == ["b"], "a torn line must not poison the ledger"


# --- tests 14/18: compaction keeps suppressing ids ------------------------

async def test_ledger_compaction_keeps_only_pending_ids(
    data_dir: Path, tmp_path: Path
) -> None:
    """After the sweep the ledger holds exactly the still-pending ids."""
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    for i in range(3):
        await proc.publish_inbox_message(f"m{i}", True)
    await _drain_once(proc, 3, delivered, "three deliveries")
    assert not _pending(data_dir)

    # Constructing the process IS the action under test: init_inbox_spool runs
    # the sweep and then compacts the ledger.
    _make_proc(data_dir, tmp_path, name="test_redelivery_compact")
    ledger = inbox_spool.ledger_path(data_dir)
    remaining = ledger.read_text().split() if ledger.exists() else []
    assert remaining == [], (
        "ids whose records are gone can suppress nothing and must be dropped"
    )


async def test_compaction_keeps_exactly_the_delivered_pending_intersection(
    data_dir: Path, tmp_path: Path
) -> None:
    """``keep`` is ledger ∩ pending — never merely ``pending``.

    The case above only exercises a ledger with ZERO pending records, so it
    passes even if compaction writes every pending id in. Here both kinds are
    present at startup: a delivered record whose unlink was lost, and a
    never-delivered one. The ledgered record is unlinked by the sweep (its turn
    already ran), so its id is legitimately no longer relevant and drops out.
    What must NEVER happen is the other direction — the never-delivered id
    being written in as though delivered, which arms a later sweep to unlink a
    message nobody has seen. ``keep`` may only ever REMOVE ids.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("delivered but not unlinked", True)
    lost_path = _pending(data_dir)[0]
    lost_body = json.loads(lost_path.read_text())
    await _drain_once(proc, 1, delivered, "the delivery that loses its unlink")
    _restore(lost_path, lost_body)

    # Never handed to a turn: published while no drain is running.
    await proc.publish_inbox_message("never delivered", True)
    fresh_id = next(
        json.loads(p.read_text())["delivery_id"]
        for p in _pending(data_dir)
        if p != lost_path
    )

    _make_proc(data_dir, tmp_path, name="test_redelivery_intersection")

    ledger = inbox_spool.ledger_path(data_dir)
    remaining = ledger.read_text().split() if ledger.exists() else []
    assert fresh_id not in remaining, (
        "a NEVER-DELIVERED record's id in the ledger marks it delivered; the "
        f"next sweep would unlink it unseen — silent loss; got {remaining!r}"
    )
    assert set(remaining) <= {lost_body["delivery_id"]}, (
        f"compaction may only remove ids, never introduce them; got {remaining!r}"
    )
    # The never-delivered record must still be there to be delivered.
    assert any(
        json.loads(p.read_text())["delivery_id"] == fresh_id
        for p in _pending(data_dir)
    ), "the undelivered record must survive the sweep"


async def test_undelivered_record_survives_startup_compaction(
    data_dir: Path, tmp_path: Path
) -> None:
    """The end-to-end shape of the same defect: it must still be DELIVERED.

    Startup (load ledger -> sweep -> compact) runs before any drain. If that
    path marks a fresh pending record as delivered, the message is lost with no
    trace. Restarting again and draining must still hand it to a turn.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)
    # Give startup a ledger to compact, so the compaction branch is live.
    await proc.publish_inbox_message("first", True)
    await _drain_once(proc, 1, delivered, "a delivery to create the ledger")

    await proc.publish_inbox_message("must not be swallowed", True)

    # Two restarts: the first compacts, the second sweeps against whatever the
    # first wrote. A record wrongly recorded as delivered dies in between.
    _make_proc(data_dir, tmp_path, name="test_redelivery_survive_1")
    assert _pending(data_dir), "startup must not unlink an undelivered record"

    proc3 = _make_proc(data_dir, tmp_path, name="test_redelivery_survive_2")
    delivered3: list[str] = []
    _collect_turns(proc3, delivered3)
    await _drain_once(proc3, 1, delivered3, "the surviving delivery")

    assert delivered3 == ["must not be swallowed"], (
        f"the undelivered record must still be delivered; got {delivered3!r}"
    )


# --- tests 19/20: ledger integrity on the append and load paths -----------

async def test_partial_append_is_repaired_and_retry_succeeds(
    data_dir: Path, tmp_path: Path
) -> None:
    """A torn tail must be repaired ON DISK before the next append.

    Appending onto ``<partial>`` fuses it with the new id into one junk line
    that the parser rejects — so the id never suppresses anything even though
    the record was unlinked. That converts a survivable duplicate into a
    permanent replay. Repair must happen without waiting for a restart.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("a", True)
    await _drain_once(proc, 1, delivered, "a delivery to create the ledger")

    ledger = inbox_spool.ledger_path(data_dir)
    intact = ledger.read_text()
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write("0123456789abcdef")  # partial id, no newline

    # Same process, no restart: the next delivery must repair and append.
    await proc.publish_inbox_message("b", True)
    b_path = _pending(data_dir)[0]
    b_body = json.loads(b_path.read_text())
    await _drain_once(proc, 2, delivered, "the delivery after the torn append")

    lines = ledger.read_text().splitlines()
    assert all(len(line) == 32 for line in lines), (
        f"the torn fragment must be gone, not fused into an id; got {lines!r}"
    )
    assert b_body["delivery_id"] in lines, (
        "the retried append must actually record the id it unlinked for"
    )
    assert intact.split() == lines[:1], "the earlier id must not be disturbed"

    # And the recorded id must really suppress after a restart.
    _restore(b_path, b_body)
    proc2 = _make_proc(data_dir, tmp_path, name="test_redelivery_partial")
    delivered2: list[str] = []
    _collect_turns(proc2, delivered2)
    await proc2.publish_inbox_message("wake", True)
    await _drain_once(proc2, 1, delivered2, "the post-restart delivery")
    await asyncio.sleep(0.2)

    assert delivered2 == ["wake"], (
        f"the repaired ledger entry must suppress the replay; got {delivered2!r}"
    )


async def test_symlinked_ledger_is_refused_and_record_kept(
    data_dir: Path, tmp_path: Path
) -> None:
    """A planted/symlinked ledger must never drive the sweep.

    ``load_delivered`` feeds the suppression set that the sweep unlinks
    against. If it follows a symlink or trusts a file that is not ours, an
    attacker-planted (or merely corrupt) id set silently deletes real,
    undelivered messages. Fail closed: refuse, and keep the record.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)
    await proc.publish_inbox_message("real message", True)
    pending_id = json.loads(_pending(data_dir)[0].read_text())["delivery_id"]

    # A ledger that claims the pending record was already delivered, reachable
    # only through a symlink — the shape that must not be trusted.
    planted = tmp_path / "planted_ledger"
    planted.write_text(f"{pending_id}\n", encoding="utf-8")
    os.chmod(planted, 0o600)
    ledger = inbox_spool.ledger_path(data_dir)
    if ledger.exists():
        ledger.unlink()
    ledger.symlink_to(planted)

    with pytest.raises(OSError):
        _make_proc(data_dir, tmp_path, name="test_redelivery_symlink")

    assert _pending(data_dir), (
        "a ledger that is not our own regular file must never authorise the "
        "sweep to unlink an undelivered record"
    )


async def test_bad_ledger_leaves_process_unprimed_on_production_path(
    data_dir: Path, tmp_path: Path
) -> None:
    """The REAL recovery path must refuse, not ack against an unswept spool.

    ``test_symlinked_ledger_is_refused_and_record_kept`` only proves the raise
    escapes ``init_inbox_spool``. Production does not call that directly:
    ``bring_up_inbox`` logs and swallows the ``OSError`` so a broken spool
    cannot stop the agent booting. If priming was already marked done by then,
    ``publish_inbox_message`` skips its late-prime retry and publishes — the
    sender is ack'd against a ledger we explicitly refused to validate, and the
    sweep that would have suppressed a replay never ran. Fail closed instead:
    stay unprimed, retry inside the lock, and raise so the sender sees
    ``success: false``.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)
    await proc.publish_inbox_message("real message", True)
    pending_before = _pending(data_dir)
    assert pending_before

    planted = tmp_path / "planted_ledger2"
    planted.write_text("", encoding="utf-8")
    os.chmod(planted, 0o600)
    ledger = inbox_spool.ledger_path(data_dir)
    if ledger.exists():
        ledger.unlink()
    ledger.symlink_to(planted)

    # Built explicitly, NOT via _make_proc: that helper calls init_inbox_spool
    # directly, which is exactly the path this test must avoid.
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    fresh = AgentProcess(
        name="test_redelivery_unprimed",
        config=AgentConfig(
            working_dir=tmp_path,
            provider="claude-code-sdk",
            model="claude-3-5-sonnet-20241022",
        ),
        data_dir=data_dir,
        rpyc_port=False,
    )
    # The production entry point: it swallows the OSError by design and starts
    # the drain anyway, so the agent still boots on a broken spool.
    _collect_turns(fresh, [])
    recovered = fresh.bring_up_inbox()
    try:
        assert recovered == 0, "a failed prime cannot claim recovered records"
        assert not fresh._inbox_seq_primed, (
            "a spool whose ledger failed validation must NOT count as primed"
        )
        # The late prime inside the lock re-runs init and fails again, so the
        # sender is told success: false instead of being ack'd.
        with pytest.raises(OSError):
            await fresh.publish_inbox_message("must not be acked", True)
    finally:
        task = fresh._inbox_drain_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert _pending(data_dir) == pending_before, (
        "the refused publish must not add a record, and the sweep must not "
        "have unlinked the existing one"
    )


async def test_retirement_fsyncs_both_directories(
    data_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """``retire`` renames ACROSS directories, so both entries must be durable.

    Syncing only ``expired/`` leaves the removal from ``inbox/`` non-durable:
    after a power loss the record can resurrect as pending and be delivered,
    defeating the bounded-retirement guarantee.
    """
    synced: list[Path] = []
    real_fsync_dir = inbox_spool._fsync_dir

    def recording_fsync_dir(path: Path) -> None:
        synced.append(Path(path))
        real_fsync_dir(path)

    monkeypatch.setattr(inbox_spool, "_fsync_dir", recording_fsync_dir)

    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)
    await proc.publish_inbox_message("too old", True)

    path = _pending(data_dir)[0]
    body = json.loads(path.read_text())
    body["ts"] = (
        datetime.now(timezone.utc) - timedelta(days=30)
    ).isoformat()
    _restore(path, body)

    synced.clear()
    assert inbox_spool.retire(data_dir, path, body, "ttl") is not None

    assert inbox_spool.expired_dir(data_dir) in synced, "expired/ must be synced"
    assert inbox_spool.inbox_dir(data_dir) in synced, (
        "inbox/ must be synced too — the rename REMOVED an entry from it, and "
        f"an unsynced removal can resurrect the record; synced={synced!r}"
    )


async def test_oldest_pending_id_beyond_cap_is_retained(
    data_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """A suppressing id BEYOND the cap by position must survive compaction.

    Truncating by recency would evict it and re-arm the replay it exists to
    prevent — the r2 design defect.
    """
    monkeypatch.setattr(inbox_spool, "LEDGER_CAP", 8, raising=False)

    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("the survivor", True)
    survivor_path = _pending(data_dir)[0]
    survivor_body = json.loads(survivor_path.read_text())
    await _drain_once(proc, 1, delivered, "the survivor delivery")
    _restore(survivor_path, survivor_body)  # its unlink was "lost"

    for i in range(12):  # > LEDGER_CAP further deliveries
        await proc.publish_inbox_message(f"filler {i}", True)
    await _drain_once(proc, 13, delivered, "the filler deliveries")

    proc2 = _make_proc(data_dir, tmp_path, name="test_redelivery_cap")
    delivered2: list[str] = []
    _collect_turns(proc2, delivered2)
    await proc2.publish_inbox_message("wake", True)
    await _drain_once(proc2, 1, delivered2, "the post-restart delivery")
    # Let any SECOND (unsuppressed) delivery land before asserting: stopping at
    # the first turn would make this pass whichever record won the race.
    await asyncio.sleep(0.2)

    assert "the survivor" not in delivered2, (
        "the old ledgered id must still suppress, however far back it sits; "
        f"got {delivered2!r}"
    )
    assert delivered2 == ["wake"], f"expected only the fresh wake, got {delivered2!r}"
    assert not _pending(data_dir), "the suppressed record must be unlinked"


# --- isolation guard ------------------------------------------------------

@pytest.mark.asyncio(loop_scope="function")
async def test_suite_never_touches_production_state() -> None:
    """Tests must never write into the real ~/.deep_worker (hard policy)."""
    from deep_worker_cli.storage_paths import deep_worker_home

    home = deep_worker_home()
    marker = home / "inbox"
    assert not marker.exists(), (
        f"a test wrote a spool into production state at {marker}"
    )
