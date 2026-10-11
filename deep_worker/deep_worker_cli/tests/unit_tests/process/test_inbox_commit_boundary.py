"""The inbox delete boundary is the checkpoint commit, not the turn result.

Design: ``docs/DESIGN_DELIVERY_BARRIER_ON_CHECKPOINT.md``.

The defect these pin: the drain unlinked a spool record as soon as the turn
returned ``{"success": True}``. That is a claim about control flow, not about
the model's context. A process that dies between the turn's return and the
checkpointer's ``commit()`` loses the message outright — while the sender was
already told the delivery had been accepted. The only honest criterion is the
one the self-restart barrier already uses: the record was recorded in a
committed checkpoint.

Every wait is bounded; a drain test that hangs is worse than one that fails.
"""

from __future__ import annotations

import asyncio
import json
import uuid
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


def _make_proc(data_dir: Path, tmp_path: Path, name: str = "test_commit_boundary"):
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
    # A committing checkpointer is present in production; the tests that want
    # the legacy fallback turn this back off explicitly.
    proc._inbox_commit_signal_available = True
    return proc


async def _wait_for(predicate, timeout: float = _BOUND, what: str = "condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if value := predicate():
            return value
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


async def _drain_once(proc, expect: int, delivered: list, what: str) -> None:
    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(delivered) >= expect, what=what), _BOUND
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _turns_without_commit(proc, delivered: list) -> None:
    """A turn that returns success but never reaches a checkpoint commit."""
    async def fake_owned(command, **kwargs):
        delivered.append(command["message"])
        return {"success": True}

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]


def _turns_with_commit(proc, delivered: list) -> None:
    """A turn that returns success AFTER its message is committed.

    Mirrors production ordering: the checkpointer commits, fires the delivery
    listener, and only then does the turn return.
    """
    async def fake_owned(command, **kwargs):
        delivered.append(command["message"])
        delivery_id = command.get("delivery_id")
        if isinstance(delivery_id, str):
            proc._on_delivery_committed(frozenset({delivery_id}))
        return {"success": True}

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]


def _pending(data_dir: Path) -> list[Path]:
    return inbox_spool.pending_records(data_dir)


async def test_record_survives_a_turn_that_returned_before_the_commit(
    data_dir: Path, tmp_path: Path
) -> None:
    """THE case: success returned, commit never happened → keep the record.

    This is the kill window. Unlinking here is exactly the lost message the
    durable spool exists to prevent, and it is invisible: the sender was told
    ``success: true``, so nothing anywhere reports the loss.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _turns_without_commit(proc, delivered)

    await proc.publish_inbox_message("unconfirmed", True)
    await _drain_once(proc, 1, delivered, "the delivery attempt")

    assert delivered == ["unconfirmed"]
    assert _pending(data_dir), (
        "a turn that returned before its commit must NOT drop the record — "
        "a kill in that window would lose the message silently"
    )


async def test_record_is_dropped_once_the_commit_confirms_it(
    data_dir: Path, tmp_path: Path
) -> None:
    """The positive half: a committed message really is removed."""
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _turns_with_commit(proc, delivered)

    await proc.publish_inbox_message("confirmed", True)
    await _drain_once(proc, 1, delivered, "the delivery")

    assert delivered == ["confirmed"]
    await _wait_for(
        lambda: not _pending(data_dir), what="the committed record to be removed"
    )


async def test_retained_record_is_dropped_on_a_later_commit(
    data_dir: Path, tmp_path: Path
) -> None:
    """A record held back for want of a commit is released by a later one.

    Retention must be a delay, not a leak: once the proof arrives the record
    goes away rather than being redelivered forever.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _turns_without_commit(proc, delivered)

    await proc.publish_inbox_message("slow", True)
    await _drain_once(proc, 1, delivered, "the first attempt")
    assert _pending(data_dir), "not yet committed → retained"

    # The commit lands late (the real listener fires from the checkpointer).
    record = json.loads(_pending(data_dir)[0].read_text())
    proc._on_delivery_committed(frozenset({record["delivery_id"]}))

    _turns_with_commit(proc, delivered)
    await proc.publish_inbox_message("wake", True)
    await _drain_once(proc, 2, delivered, "the retry after the commit")

    await _wait_for(
        lambda: not _pending(data_dir),
        what="the now-committed record to be removed",
    )


async def test_a_foreign_commit_does_not_release_our_record(
    data_dir: Path, tmp_path: Path
) -> None:
    """Correlation is by delivery_id, exactly as the barrier correlates ids.

    Some other message reaching the context says nothing about ours; treating
    any commit as proof would drop records that were never recorded.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _turns_without_commit(proc, delivered)

    await proc.publish_inbox_message("ours", True)
    await _drain_once(proc, 1, delivered, "the delivery attempt")

    proc._on_delivery_committed(frozenset({uuid.uuid4().hex}))

    await proc.publish_inbox_message("wake", True)
    await _drain_once(proc, 3, delivered, "the retry")

    assert _pending(data_dir), (
        "an unrelated delivery_id must not release our record"
    )


async def test_memory_only_falls_back_to_the_turn_result(
    data_dir: Path, tmp_path: Path
) -> None:
    """No committing saver → no proof can ever arrive; do not retain forever.

    Under ``--memory-only`` there is no commit to wait for. Retaining every
    record would replay the whole spool on every wake, so the turn result stays
    the boundary in that configuration — and only that one.
    """
    proc = _make_proc(data_dir, tmp_path)
    proc._inbox_commit_signal_available = False
    delivered: list[str] = []
    _turns_without_commit(proc, delivered)

    await proc.publish_inbox_message("memory only", True)
    await _drain_once(proc, 1, delivered, "the delivery")

    await _wait_for(
        lambda: not _pending(data_dir),
        what="the record to be removed on the fallback path",
    )


async def test_uncommitted_record_is_not_ledgered(
    data_dir: Path, tmp_path: Path
) -> None:
    """A retained record must stay replayable.

    Ledgering an id suppresses it forever. Doing that for a message that never
    reached the context would convert "retained for retry" into a silent drop.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _turns_without_commit(proc, delivered)

    await proc.publish_inbox_message("unconfirmed", True)
    await _drain_once(proc, 1, delivered, "the delivery attempt")

    record = json.loads(_pending(data_dir)[0].read_text())
    assert record["delivery_id"] not in inbox_spool.load_delivered(data_dir), (
        "an uncommitted delivery must not be ledgered — that would suppress "
        "the retry and lose the message"
    )


async def test_retained_record_still_ages_out(
    data_dir: Path, tmp_path: Path
) -> None:
    """Retention is bounded by the TTL, so it cannot become a replay loop.

    Without this the new boundary would trade a lost message for an immortal
    one: a commit proof that never arrives would keep the record forever.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _turns_without_commit(proc, delivered)

    await proc.publish_inbox_message("ancient", True)
    await _drain_once(proc, 1, delivered, "the delivery attempt")
    assert _pending(data_dir), "retained, pending the commit that never comes"

    # Backdate past the TTL, the way a record stranded over a weekend would be.
    path = _pending(data_dir)[0]
    body: dict[str, Any] = json.loads(path.read_text())
    body["ts"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(body), encoding="utf-8")

    await proc.publish_inbox_message("wake", True)
    await _drain_once(proc, 2, delivered, "the wake")

    assert not any(p.name == path.name for p in _pending(data_dir)), (
        "a record retained for want of a commit must still retire on TTL"
    )
    expired = inbox_spool.expired_dir(data_dir)
    assert expired.is_dir() and any(expired.iterdir()), (
        "the retired record must be MOVED to expired/, never silently deleted"
    )


async def test_suite_never_touches_production_state() -> None:
    """Tests must never write into the real ~/.deep_worker (hard policy)."""
    from deep_worker_cli.storage_paths import deep_worker_home

    marker = deep_worker_home() / "inbox"
    assert not marker.exists(), (
        f"a test wrote a spool into production state at {marker}"
    )
