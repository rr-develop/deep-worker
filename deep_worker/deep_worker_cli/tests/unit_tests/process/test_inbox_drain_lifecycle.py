"""Interrupt-safety and lifecycle of the durable-inbox drain.

Design §5 tests 12 and 19; §3.3.0b and §3.3.4.

The BLOCKER these pin (design review round 4): ``_handle_invoke_owned`` sets
``current_worker_task = asyncio.current_task()`` and ``exposed_interrupt_agent``
cancels exactly that task. If the drain awaited the turn INLINE, a routine user
interrupt of an inter-agent turn would cancel the ONLY consumer — every later
arrival would publish durably and set the event with nobody listening, until
the process restarted. A silent processing outage, i.e. the very failure class
this whole change removes.

Running each turn in a short-lived CHILD task makes ``current_worker_task``
point at the turn, so an interrupt cancels one delivery and the drain survives.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deep_worker_cli.process import inbox_spool

pytestmark = pytest.mark.asyncio

_BOUND = 5.0


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent_data"
    d.mkdir(mode=0o700)
    return d


def _make_proc(data_dir: Path, tmp_path: Path):
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path,
        provider="claude-code-sdk",
        model="claude-3-5-sonnet-20241022",
    )
    proc = AgentProcess(
        name="test_drain_lifecycle", config=config, data_dir=data_dir,
        rpyc_port=False,
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


# --- test 12: a user interrupt kills the TURN, not the consumer -----------

async def test_user_interrupt_cancels_the_turn_and_the_drain_survives(
    data_dir: Path, tmp_path: Path
) -> None:
    """The exact BLOCKER scenario, driven through the REAL interrupt entry
    point (``cancel_current_worker_task``), not a synthetic cancel.

    Mutation guard: awaiting the turn inline (so ``current_worker_task`` is the
    consumer) makes the drain die here and the final assertions fail.
    """
    proc = _make_proc(data_dir, tmp_path)
    started = asyncio.Event()
    delivered: list[str] = []
    # The record is RETAINED after the interrupt (that is the contract), so it
    # is legitimately retried on the next wake. Only the FIRST attempt hangs;
    # otherwise the retry would block the pass and we would never observe the
    # later arrival — a harness artefact, not product behaviour.
    hang_once = {"pending": True}

    async def hang_then_record(command, **kwargs):
        # Mirror what the real _handle_invoke_owned does on entry
        # (agent_process.py:3275-3276) — registering the RUNNING TURN as the
        # interrupt target. That registration is the whole point of the test,
        # so the stub must not skip it.
        proc.is_running = True
        proc.current_worker_task = asyncio.current_task()
        try:
            msg = command["message"]
            if msg == "interrupted" and hang_once["pending"]:
                hang_once["pending"] = False
                started.set()
                await asyncio.sleep(3600)  # cancelled by the interrupt
            delivered.append(msg)
            return {"success": True}
        finally:
            if proc.current_worker_task is asyncio.current_task():
                proc.is_running = False
                proc.current_worker_task = None

    proc._handle_invoke_owned = hang_then_record  # type: ignore[assignment]
    await proc.publish_inbox_message("interrupted", True)

    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(started.wait(), _BOUND)

        # The turn — not the drain — must be the registered interrupt target.
        assert proc.current_worker_task is not None
        assert proc.current_worker_task is not task, (
            "the consumer must never be current_worker_task"
        )

        assert proc.cancel_current_worker_task() is True
        await asyncio.sleep(0.05)

        # Consumer survived; the interrupted record was retained.
        assert not task.done(), "user interrupt must not kill the consumer"
        assert _messages_on_disk(data_dir) == ["interrupted"]

        # And it still works: a new arrival is processed normally.
        await proc.publish_inbox_message("after interrupt", True)
        await asyncio.wait_for(
            _wait_for(lambda: "after interrupt" in delivered,
                      what="delivery after the interrupt"),
            _BOUND,
        )
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --- test 19: lifecycle ---------------------------------------------------

async def test_start_is_idempotent(data_dir: Path, tmp_path: Path) -> None:
    proc = _make_proc(data_dir, tmp_path)
    first = proc.start_inbox_drain()
    try:
        assert proc.start_inbox_drain() is first
        assert proc.start_inbox_drain() is first
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


async def test_stop_cancels_and_awaits_an_active_child(
    data_dir: Path, tmp_path: Path
) -> None:
    """Shutdown must leave NO orphan turn running against a checkpointer that
    is about to close (design §3.3.4).

    Mutation guard: skipping the child cancel-and-drain leaves ``child_done``
    False here.
    """
    proc = _make_proc(data_dir, tmp_path)
    started = asyncio.Event()
    child_done = asyncio.Event()

    async def long_turn(command, **kwargs):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            child_done.set()
            raise
        return {"success": True}

    proc._handle_invoke_owned = long_turn  # type: ignore[assignment]
    await proc.publish_inbox_message("in flight", True)

    task = proc.start_inbox_drain()
    await asyncio.wait_for(started.wait(), _BOUND)

    # Production ordering: flag first, then cancel+await.
    proc._shutting_down = True
    await asyncio.wait_for(proc.stop_inbox_drain(), _BOUND)

    assert task.done(), "the drain must be finished after stop"
    assert child_done.is_set(), "the active child must be cancelled and awaited"
    # An interrupted delivery keeps its record.
    assert _messages_on_disk(data_dir) == ["in flight"]
    assert proc._inbox_drain_task is None


async def test_shutdown_stops_drain_before_checkpointer_cleanup(
    data_dir: Path, tmp_path: Path
) -> None:
    """Ordering proof: the drain is stopped BEFORE ``cleanup()`` closes the
    checkpointer — otherwise a turn could run against a closed database.

    Mutation guard: moving the stop after ``cleanup()`` (or dropping it)
    reverses this recorded order.
    """
    proc = _make_proc(data_dir, tmp_path)
    order: list[str] = []

    real_stop = proc.stop_inbox_drain

    async def traced_stop() -> None:
        order.append("stop_drain")
        await real_stop()

    async def traced_cleanup() -> None:
        order.append("cleanup")

    proc.stop_inbox_drain = traced_stop  # type: ignore[assignment]
    proc.cleanup = traced_cleanup  # type: ignore[assignment]
    proc.running = True
    proc.rpyc_server = None
    proc.task_manager = None
    proc.session_manager = None
    proc._unregister = _noop  # type: ignore[assignment]
    proc._remove_pid_file = _noop  # type: ignore[assignment]

    proc.start_inbox_drain()
    await asyncio.wait_for(proc.shutdown(), _BOUND)

    assert order == ["stop_drain", "cleanup"], order
    assert proc._shutting_down is True


async def _noop(*args, **kwargs) -> None:
    return None


async def test_drain_survives_an_unexpected_scan_error(
    data_dir: Path, tmp_path: Path
) -> None:
    """A defective pass must not kill the only consumer while durable records
    are pending — and must not spin either (no re-arm on that path)."""
    proc = _make_proc(data_dir, tmp_path)
    calls: list[int] = []

    async def exploding_pass() -> None:
        calls.append(1)
        raise RuntimeError("scan machinery broke")

    proc._drain_inbox_once = exploding_pass  # type: ignore[assignment]

    task = proc.start_inbox_drain()
    try:
        await proc.publish_inbox_message("m", True)
        await asyncio.wait_for(
            _wait_for(lambda: calls, what="the failing pass"), _BOUND
        )
        await asyncio.sleep(0.2)

        assert not task.done(), "consumer must survive a pass-level error"
        assert len(calls) == 1, f"must not spin; got {len(calls)} passes"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
