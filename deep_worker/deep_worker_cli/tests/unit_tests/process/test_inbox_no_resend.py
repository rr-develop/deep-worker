"""No automatic resend of a message already sent to an agent.

A message counts as SENT once it is in the recipient's inbox — that is what the
durable acknowledgement promises the sender — so the ban is armed at publish,
not after the turn has run.

The defect these pin, observed live on a deployed agent: the SAME review verdict from
a peer agent entered one conversation NINE times. The receiver-side machinery
was working perfectly — inbox empty, ledger correct, delete boundary honoured.
It could not help, because ``build_record`` mints ``delivery_id`` per send
(``uuid4``), so five resends of byte-identical text were five unrelated records
by identity. Dedup on ``delivery_id`` is structurally incapable of seeing a
resend; only identity derived from CONTENT can.

The rule enforced here: a message already sent to an agent is never accepted
again, and the refusal is HARD (``success: false``) so a component that resends
automatically stays visible instead of being silently absorbed.

Every test drives the PUBLIC path (``publish_inbox_message`` + the real drain).
Every wait is bounded.
"""

from __future__ import annotations

import asyncio
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


def _make_proc(data_dir: Path, tmp_path: Path, name: str = "test_no_resend"):
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
    task = proc.start_inbox_drain()
    try:
        await asyncio.wait_for(
            _wait_for(lambda: len(delivered) >= expect, what=what), _BOUND
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _collect_turns(proc, delivered: list, result: Any = None) -> None:
    async def fake_owned(command, **kwargs):
        delivered.append(command["message"])
        return {"success": True} if result is None else result

    proc._handle_invoke_owned = fake_owned  # type: ignore[assignment]


# --- the core rule --------------------------------------------------------

@pytest.mark.asyncio
async def test_resend_of_received_message_is_refused(
    data_dir: Path, tmp_path: Path
) -> None:
    """Byte-identical text, sent again after delivery, must be REFUSED.

    This is the live failure reduced to its essentials. Note the assertion
    is on the second *publish*, not on the drain: the duplicate has to be
    stopped before it ever becomes a record, because once it is in the
    conversation the damage is already done.
    """
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    text = "agent_1 -> agent_2 :> SIGN-OFF — POLICIES §14"

    await proc.publish_inbox_message(text, True)
    await _drain_once(proc, 1, delivered, "the first delivery")
    assert delivered == [text]

    with pytest.raises(inbox_spool.InboxDuplicateMessageError):
        await proc.publish_inbox_message(text, True)

    assert not inbox_spool.pending_records(data_dir), (
        "a refused resend must not leave a record behind"
    )
    assert delivered == [text], "the refused resend must never reach a turn"


@pytest.mark.asyncio
async def test_refusal_survives_restart(data_dir: Path, tmp_path: Path) -> None:
    """A fresh process over the same data_dir must still refuse.

    "Already in context" is a property of the persisted conversation, not of
    the process that happened to receive it. The live incident replayed across
    several PIDs precisely because nothing about the ban outlived a restart.
    """
    first = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(first, delivered)

    text = "peer -> me :> the same verdict"
    await first.publish_inbox_message(text, True)
    await _drain_once(first, 1, delivered, "the first delivery")
    assert delivered == [text]

    second = _make_proc(data_dir, tmp_path)
    _collect_turns(second, delivered)

    with pytest.raises(inbox_spool.InboxDuplicateMessageError):
        await second.publish_inbox_message(text, True)


# --- the ban must not over-reach -----------------------------------------

@pytest.mark.asyncio
async def test_queued_but_undelivered_message_is_already_banned(
    data_dir: Path, tmp_path: Path
) -> None:
    """A message sitting in the inbox counts as SENT: a resend is refused.

    From the sender's side delivery is complete once the record is in the
    recipient's inbox — that is exactly what the durable ack promises. Arming
    the ban only after the turn ran would leave a window where the message is
    queued, already acknowledged, and still resendable; a sender that retried
    inside that window would double-queue the same text.
    """
    proc = _make_proc(data_dir, tmp_path)

    text = "peer -> me :> never drained"
    await proc.publish_inbox_message(text, True)
    # No drain: the record is pending, the turn has NOT run.
    assert len(inbox_spool.pending_records(data_dir)) == 1

    with pytest.raises(inbox_spool.InboxDuplicateMessageError):
        await proc.publish_inbox_message(text, True)

    assert len(inbox_spool.pending_records(data_dir)) == 1, (
        "the refused resend must not add a second copy to the spool"
    )


@pytest.mark.asyncio
async def test_distinct_messages_are_unaffected(
    data_dir: Path, tmp_path: Path
) -> None:
    """Different text, and the same text with a different reply_back, pass."""
    proc = _make_proc(data_dir, tmp_path)
    delivered: list[str] = []
    _collect_turns(proc, delivered)

    await proc.publish_inbox_message("first", True)
    await _drain_once(proc, 1, delivered, "the first delivery")

    # Different payload — unrelated fingerprint.
    await proc.publish_inbox_message("second", True)
    await _drain_once(proc, 2, delivered, "the second delivery")

    # Same payload, different ASK: reply_back flips, so this is a different
    # request and must not be collapsed into the previous one.
    await proc.publish_inbox_message("first", False)
    await _drain_once(proc, 3, delivered, "the third delivery")

    assert delivered == ["first", "second", "first"]


# --- identity is content-derived, which is the whole point ----------------

def test_fingerprint_is_stable_across_sends() -> None:
    """Two sends of identical text fingerprint the same; delivery_id does not.

    This is the asymmetry the incident turned on, asserted directly: the field
    the old dedup relied on differs every time, so no amount of ledgering it
    could ever recognise a resend.
    """
    text = "a -> b :> identical"

    assert inbox_spool.content_fingerprint(text, True) == (
        inbox_spool.content_fingerprint(text, True)
    )

    first = inbox_spool.build_record(text, True)
    second = inbox_spool.build_record(text, True)
    assert first["delivery_id"] != second["delivery_id"], (
        "delivery_id is minted per send — that is why content identity is needed"
    )


def test_fingerprint_separates_sender_and_receiver() -> None:
    """Origin is part of identity, via the prefix the sender already applied.

    ``build_record`` deliberately keeps no ``sender`` field, so identity has to
    come from the prefixed display text. The same body from a different sender,
    or to a different recipient, must not collide.
    """
    body = "status please"
    assert inbox_spool.content_fingerprint(f"alice -> bob :> {body}", True) != (
        inbox_spool.content_fingerprint(f"carol -> bob :> {body}", True)
    )
    assert inbox_spool.content_fingerprint(f"alice -> bob :> {body}", True) != (
        inbox_spool.content_fingerprint(f"alice -> dave :> {body}", True)
    )
