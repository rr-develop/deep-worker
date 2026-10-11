"""``exposed_enqueue_message`` must not lie (design §5, tests 20-21).

The 2026-08-03 incident was silent because the ack was produced before the
message was durable. These tests pin the two halves of the honest-ack
contract: failure is LOUD, and success is ORDERED after durability.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


def _make_service(proc):
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    service = AgentService.__new__(AgentService)
    service._agent_process = proc
    service._subscribers = []
    return service


def _make_proc(data_dir: Path, tmp_path: Path):
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    config = AgentConfig(
        working_dir=tmp_path,
        provider="claude-code-sdk",
        model="claude-3-5-sonnet-20241022",
    )
    proc = AgentProcess(
        name="test_enqueue_ack", config=config, data_dir=data_dir,
        rpyc_port=False,
    )
    proc.init_inbox_spool()
    return proc


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent_data"
    d.mkdir(mode=0o700)
    return d


# --- test 20: loss must be loud ------------------------------------------

async def test_enqueue_reports_failure_when_the_record_cannot_be_persisted(
    data_dir: Path, tmp_path: Path
) -> None:
    """Goal 2: if we cannot persist, the sender MUST see success=false.

    Against the pre-fix code this was impossible — the method returned
    success=true unconditionally after create_task.
    """
    proc = _make_proc(data_dir, tmp_path)
    service = _make_service(proc)

    async def boom(message: str, reply_back: bool) -> str:
        raise OSError("disk full")

    proc.publish_inbox_message = boom  # type: ignore[assignment]

    resp = json.loads(await service.exposed_enqueue_message("m", True))

    assert resp["success"] is False
    assert "disk full" in resp["error"]


# --- test 21: ack ordering -----------------------------------------------

async def test_ack_is_produced_only_after_the_record_is_durable(
    data_dir: Path, tmp_path: Path
) -> None:
    """Ordering: durable publish → event.set() → ack.

    Proven by observing state AT THE MOMENT the ack is produced, not by
    elapsed time.
    """
    from deep_worker_cli.process import inbox_spool

    proc = _make_proc(data_dir, tmp_path)
    service = _make_service(proc)

    observed: dict[str, object] = {}
    real_publish = proc.publish_inbox_message

    async def traced(message: str, reply_back: bool) -> str:
        delivery_id = await real_publish(message, reply_back)
        # At this instant the ack has NOT been produced yet.
        observed["records_at_publish"] = len(
            inbox_spool.pending_records(data_dir)
        )
        observed["event_set_at_publish"] = proc._inbox_event.is_set()
        return delivery_id

    proc.publish_inbox_message = traced  # type: ignore[assignment]

    resp = json.loads(await service.exposed_enqueue_message("payload", True))

    assert resp["success"] is True
    assert resp["delivery_id"]
    # The record and the wakeup both existed before the caller was told yes.
    assert observed["records_at_publish"] == 1
    assert observed["event_set_at_publish"] is True

    body = json.loads(inbox_spool.pending_records(data_dir)[0].read_text())
    assert body["message"] == "payload"
    assert body["delivery_id"] == resp["delivery_id"]


async def test_unprimed_counter_yields_a_loud_failure_not_a_misordered_ack(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 review regression, at the RPyC boundary.

    ``start()`` swallows a failed startup prime so a broken spool cannot stop
    the agent booting. The sender must then see ``success: false`` — never a
    cheerful ack for a record silently written as seq 0 ahead of survivors.
    """
    from deep_worker_cli.process import inbox_spool

    proc = _make_proc(data_dir, tmp_path)
    service = _make_service(proc)

    # A survivor from a previous life.
    inbox_spool.publish(data_dir, inbox_spool.build_record("survivor", True), 7)
    proc._inbox_seq_primed = False
    proc._inbox_next_seq = 0

    monkeypatch.setattr(
        inbox_spool, "next_seq",
        lambda _d: (_ for _ in ()).throw(OSError("still broken")),
    )

    resp = json.loads(await service.exposed_enqueue_message("m", True))

    assert resp["success"] is False, resp
    assert len(inbox_spool.pending_records(data_dir)) == 1, (
        "a record was published despite the failure being reported"
    )


async def test_enqueue_without_agent_process_is_reported(tmp_path: Path) -> None:
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    service = AgentService.__new__(AgentService)
    service._agent_process = None

    resp = json.loads(await service.exposed_enqueue_message("m", True))
    assert resp["success"] is False


async def test_set_agent_process_injects_notify_without_starting_a_task(
    data_dir: Path, tmp_path: Path
) -> None:
    """Design §3.3.3: attaching a service injects the callback ONLY; the drain
    task is owned by AgentProcess."""
    proc = _make_proc(data_dir, tmp_path)
    service = _make_service(proc)

    assert proc._inbox_notify is None
    service.set_agent_process(proc)

    assert proc._inbox_notify == service._notify_subscribers
    assert proc._inbox_drain_task is None, (
        "attaching a service must not create the drain task"
    )
