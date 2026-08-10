"""LIVE end-to-end proof for the inter-agent message-loss incident. No mocks.

Incident: a related internal incident analysis (not included here).
Design:   ``docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md`` (signed off r6).

``exposed_enqueue_message`` used to return ``{"success": true}`` immediately
after ``asyncio.create_task(_bg_turn())``. At that instant the payload existed
ONLY as a captured local inside a coroutine that had not started. ``_bg_turn``
parked on the non-reentrant ``_invoke_lock``, and the message was not persisted
until ``astream`` ran. Kill the process in that window and the message was gone
— while the sender had already been told it arrived.

What is REAL here (design §2 goal 5 — no mocks on the delivery path):

* the recipient runs as a **separate OS process** with a real ``AgentProcess``
  and a real ``AsyncioServer`` on a real RPyC port;
* it registers in a real ``ProcessRegistry`` (isolated to the test's tmp_path,
  so production agents are never touched);
* the sender is a real ``AgentClient`` that **discovers the recipient by name
  through the registry** and performs a real RPyC ``enqueue_message`` over the
  socket — the exact path ``send_message_to_agent`` uses, not a direct call;
* the recipient is ``SIGKILL``ed (equivalent to ``restart_self``'s
  ``os._exit``) and then **actually restarted** over the same ``data_dir``;
* the restarted process is observed to PROCESS the recovered message with the
  same ``delivery_id`` — the property goal 4 promises.

The restarted recipient drives the REAL production startup wiring
(``bring_up_inbox()`` — the same hook ``AgentProcess.start()`` calls), so
deleting that call from ``start()`` fails this test. The only stub is the LLM
turn body; spool recovery, the startup edge, the drain, the child-task invoke
and the delete boundary are all production code. Real LLM turns are exercised
by ``tests/integration_ai/``.

Ordering is proven by OBSERVED EVENTS the child emits on stdout at precisely
defined points, consumed with a bounded ``readline`` — never by a polling loop
or ``sleep``. A test that can go green by timing luck is worthless.

Budget discipline: every wait is bounded, the module is capped, and every
spawned process is reaped (SIGTERM → SIGKILL) in ``finally``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_BUDGET_SECONDS = 120.0
_SIGNAL_TIMEOUT = 40.0
_REAP_TIMEOUT = 10.0

_MESSAGE = "adam -> eva :> durable delivery probe"
_CLI_ROOT = str(Path(__file__).resolve().parents[2])


def _inbox_dir(data_dir: Path) -> Path:
    return data_dir / "inbox"


def _pending_records(data_dir: Path) -> list[Path]:
    inbox = _inbox_dir(data_dir)
    if not inbox.is_dir():
        return []
    return sorted(
        p for p in inbox.iterdir()
        if p.is_file() and not p.name.startswith(".tmp-")
    )


async def _next_signal(proc: asyncio.subprocess.Process, timeout: float) -> dict:
    """Read one JSON event line from the child — an OBSERVED barrier.

    The child emits at precisely defined points in its own execution; we block
    on that. No polling, no sleep-derived ordering.
    """
    assert proc.stdout is not None
    while True:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            raise AssertionError(
                f"child emitted no signal within {timeout}s "
                f"(rc={proc.returncode})"
            ) from None
        if not line:
            err = b""
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    err = await asyncio.wait_for(proc.stderr.read(), 5.0)
            raise AssertionError(
                f"child closed stdout without signalling "
                f"(rc={proc.returncode}); stderr:\n{err.decode(errors='replace')}"
            )
        text = line.decode(errors="replace").strip()
        if not text.startswith("{"):
            # Unrelated stdout chatter from startup logging — skip it.
            continue
        try:
            return json.loads(text)
        except ValueError:
            continue


async def _reap(proc: asyncio.subprocess.Process | None) -> None:
    """Guaranteed cleanup: SIGTERM, then SIGKILL. Never leak a child."""
    if proc is None or proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), _REAP_TIMEOUT)
        return
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), _REAP_TIMEOUT)


# ---------------------------------------------------------------- child code

_PRELUDE = """
import asyncio, json, os, sys
sys.path.insert(0, {cli_root!r})

def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\\n")
    sys.stdout.flush()

from pathlib import Path
from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry

DATA_DIR = Path({data_dir!r})
REGISTRY = ProcessRegistry(
    registry_file=Path({registry_file!r}), lock_file=Path({lock_file!r}),
)
CONFIG = AgentConfig(
    working_dir=DATA_DIR, provider="claude-code",
    model="claude-3-5-sonnet-20241022",
)
"""

# Recipient with a REAL RPyC server, registered by name, deliberately BUSY
# (its _invoke_lock is held) so an inbound delivery cannot be processed yet.
_BUSY_RECIPIENT = _PRELUDE + """
async def main():
    proc = AgentProcess(
        name={name!r}, config=CONFIG, data_dir=DATA_DIR, rpyc_port=0,
        registry=REGISTRY,
    )
    proc.init_inbox_spool()
    await proc._start_rpyc_server()
    await proc._register()
    # Busy recipient: hold the turn gate for the rest of our life.
    await proc._invoke_lock.acquire()
    emit(event="listening", port=proc.rpyc_port, pid=os.getpid())
    await asyncio.sleep(3600)

asyncio.run(main())
"""

# Restarted recipient over the SAME data_dir: not busy, must recover and
# process whatever the previous life durably accepted.
_RECOVERING_RECIPIENT = _PRELUDE + """
async def main():
    from deep_worker_cli.process import inbox_spool

    proc = AgentProcess(
        name={name!r}, config=CONFIG, data_dir=DATA_DIR, rpyc_port=0,
        registry=REGISTRY,
    )
    finished = asyncio.Event()

    async def record_turn(command, **kwargs):
        # Stands in for the LLM turn ONLY. Spool recovery, the startup edge,
        # the drain, the child-task invoke and the delete boundary above/below
        # this point are all production code.
        emit(event="processed",
             message=command.get("message"),
             delivery_id=command.get("delivery_id"),
             inter_agent=command.get("inter_agent_delivery"))
        return {{"success": True}}

    proc._handle_invoke_owned = record_turn

    # EVENT-DRIVEN completion barrier: wrap the PRODUCTION delivery method so
    # the signal fires strictly AFTER it returns — i.e. after the real delete
    # boundary has run. No polling, no sleep. The production method itself is
    # what executes; this only observes its completion.
    real_deliver = proc._deliver_inbox_record

    async def observed_deliver(path, record):
        await real_deliver(path, record)
        emit(event="drained",
             remaining=len(inbox_spool.pending_records(DATA_DIR)))
        finished.set()

    proc._deliver_inbox_record = observed_deliver

    # Observe the production startup wiring rather than re-implementing it.
    # The spool bring-up MUST come from AgentProcess.start(); if we called
    # bring_up_inbox() ourselves, deleting that call from start() would not
    # fail this test and the wiring would be unproven.
    real_bring_up = proc.bring_up_inbox

    def observed_bring_up():
        count = real_bring_up()
        emit(event="recovered", count=count)
        return count

    proc.bring_up_inbox = observed_bring_up

    # Run the REAL start(): it blocks on shutdown_event, so drive it as a
    # background task and shut down cleanly once the delivery is observed.
    start_task = asyncio.create_task(proc.start())
    try:
        await asyncio.wait_for(finished.wait(), 60)
    finally:
        proc.shutdown_event.set()
        try:
            await asyncio.wait_for(start_task, 20)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            start_task.cancel()

asyncio.run(main())
"""


@pytest.fixture
def registry_paths(tmp_path: Path) -> tuple[Path, Path]:
    """An ISOLATED registry — a misconfigured test can never reach a real
    agent (same guard discipline as the A2A two-agent fixture)."""
    return tmp_path / "registry.json", tmp_path / "registry.lock"


def _script(template: str, data_dir: Path, name: str,
            registry_paths: tuple[Path, Path]) -> str:
    return template.format(
        cli_root=_CLI_ROOT,
        data_dir=str(data_dir),
        registry_file=str(registry_paths[0]),
        lock_file=str(registry_paths[1]),
        name=name,
    )


async def _spawn(script: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _send_over_rpyc(name: str, registry_paths: tuple[Path, Path]) -> dict:
    """Real client → real registry lookup → real RPyC → real service."""
    from deep_worker_cli.client.agent_client import AgentClient
    from deep_worker_cli.process.registry import ProcessRegistry

    client = AgentClient(
        agent_name=name,
        registry=ProcessRegistry(
            registry_file=registry_paths[0], lock_file=registry_paths[1]
        ),
    )
    await asyncio.wait_for(client.connect(), _SIGNAL_TIMEOUT)
    try:
        return await asyncio.wait_for(
            client._send_command({
                "command": "enqueue_message",
                "message": _MESSAGE,
                "reply_back": True,
            }),
            _SIGNAL_TIMEOUT,
        )
    finally:
        with contextlib.suppress(Exception):
            await client.disconnect()


# ---------------------------------------------------------------- the tests

@pytest.mark.timeout(_BUDGET_SECONDS)
async def test_real_rpyc_send_to_busy_recipient_is_durable_before_ack(
    tmp_path: Path, registry_paths: tuple[Path, Path]
) -> None:
    """Design §5 test 23 over the REAL transport.

    A real client discovers the recipient by name and sends over RPyC while
    the recipient's turn gate is held. The ack must not arrive before the
    record is on disk.
    """
    data_dir = tmp_path / "recipient_data"
    data_dir.mkdir(mode=0o700)
    name = f"test_inbox_busy_{os.getpid()}"

    child = await _spawn(
        _script(_BUSY_RECIPIENT, data_dir, name, registry_paths)
    )
    try:
        ready = await _next_signal(child, _SIGNAL_TIMEOUT)
        assert ready["event"] == "listening"
        # Nothing durable yet — the recipient has only started.
        assert _pending_records(data_dir) == []

        resp = await _send_over_rpyc(name, registry_paths)
        assert resp.get("success") is True, resp
        assert resp.get("delivery_id"), resp

        # The ack has returned, so by contract the record MUST already be
        # durable — even though the recipient is busy and cannot process it.
        records = _pending_records(data_dir)
        assert records, (
            "ack returned success=true but no durable record exists — the "
            "message lives only in memory and dies with the process "
            "(incident root cause, design §1 D1)"
        )
        body = json.loads(records[0].read_text())
        assert body["message"] == _MESSAGE
        assert body["delivery_id"] == resp["delivery_id"]
        assert records[0].stat().st_mode & 0o777 == 0o600
        assert _inbox_dir(data_dir).stat().st_mode & 0o777 == 0o700
    finally:
        await _reap(child)


@pytest.mark.timeout(_BUDGET_SECONDS)
async def test_durably_acked_message_is_processed_after_recipient_restart(
    tmp_path: Path, registry_paths: tuple[Path, Path]
) -> None:
    """Design §5 test 24 / goal 4: THE incident scenario, end to end.

    Replies #3 and #4 were lost because the recipient was mid-turn and then
    hard-exited. Here:

      1. a real recipient listens on real RPyC, registered by name, and is BUSY;
      2. a real client discovers it and sends over the socket, observing
         ``success: true``;
      3. the record is confirmed on disk BEFORE any kill, so the test cannot
         pass by timing luck;
      4. the recipient is SIGKILLed — the equivalent of ``os._exit``;
      5. a NEW process starts over the same ``data_dir`` and is observed to
         PROCESS the recovered message with the same ``delivery_id``.

    Step 5 is what makes the ack honest: durably acked before death ⇒
    processed after restart. Without it the test would only prove that a file
    survived a kill.
    """
    data_dir = tmp_path / "recipient_data"
    data_dir.mkdir(mode=0o700)
    name = f"test_inbox_restart_{os.getpid()}"

    first: asyncio.subprocess.Process | None = None
    second: asyncio.subprocess.Process | None = None
    try:
        first = await _spawn(
            _script(_BUSY_RECIPIENT, data_dir, name, registry_paths)
        )
        ready = await _next_signal(first, _SIGNAL_TIMEOUT)
        assert ready["event"] == "listening"

        resp = await _send_over_rpyc(name, registry_paths)
        assert resp.get("success") is True, resp
        delivery_id = resp["delivery_id"]

        # OBSERVED: the record exists. Only now may we kill — the ordering is
        # established by this assertion, not by elapsed time.
        assert _pending_records(data_dir), "acked message was never persisted"

        first.send_signal(signal.SIGKILL)
        await asyncio.wait_for(first.wait(), _REAP_TIMEOUT)
        assert first.returncode is not None

        # The message outlived the abrupt death...
        assert _pending_records(data_dir), (
            "the acknowledged message vanished when the busy recipient was "
            "killed — this is the 2026-08-03 incident (design §1 D2)"
        )

        # ...and the restarted process must actually PROCESS it.
        second = await _spawn(
            _script(_RECOVERING_RECIPIENT, data_dir, name, registry_paths)
        )
        recovered = await _next_signal(second, _SIGNAL_TIMEOUT)
        assert recovered["event"] == "recovered"
        assert recovered["count"] == 1, recovered

        processed = await _next_signal(second, _SIGNAL_TIMEOUT)
        assert processed["event"] == "processed"
        assert processed["message"] == _MESSAGE
        assert processed["inter_agent"] is True
        assert processed["delivery_id"] == delivery_id, (
            "the recovered turn must carry the original delivery_id so a "
            "duplicate is diagnosable"
        )

        drained = await _next_signal(second, _SIGNAL_TIMEOUT)
        assert drained["event"] == "drained"
        assert drained["remaining"] == 0, "delivered record must be unlinked"
    finally:
        await _reap(first)
        await _reap(second)
