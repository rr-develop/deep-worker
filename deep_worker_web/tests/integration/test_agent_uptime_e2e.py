"""Integration test for the per-agent uptime reconcile path.

Uses the REAL ``AgentService`` and REAL ``ProcessRegistry`` (backed by an
isolated tmp registry file — no interference with running agents). The test
captures ``AgentService._broadcast_message`` to verify the new event payload;
WebSocket transport is covered separately by existing integration tests.

Flow:
  1. Register a fake agent record in the real registry (name, pid, port,
     started_at=T0).
  2. Start the real AgentService against that registry and verify the
     initial ``started_at`` loaded into memory.
  3. Overwrite the registry entry with a new ``started_at=T1``
     (simulates ``ProcessRegistry.register`` on graceful restart — the
     production code overwrites entries in-place).
  4. Trigger reconcile; assert:
       * ``agent_started_at_changed`` is broadcast with the new ISO
         ``started_at``,
       * ``service._agents[name].started_at`` was updated in-memory.

Design doc: ``docs/DESIGN_AGENT_UPTIME_UI.md``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app as main_app
from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from deep_worker_cli.types import AgentProcessInfo  # type: ignore[import-not-found]


def _isolated_registry() -> tuple[ProcessRegistry, Path]:
    """Return a ProcessRegistry backed by a fresh tmp file (no interference
    with real agents running on the host)."""
    tmp = tempfile.mkdtemp(prefix="dw_web_uptime_it_")
    reg_file = Path(tmp) / "registry.json"
    lock_file = reg_file.with_suffix(".lock")
    reg_file.write_text("{}")
    return ProcessRegistry(registry_file=reg_file, lock_file=lock_file), reg_file


def _register(reg: ProcessRegistry, name: str, started_at: datetime) -> None:
    # Use the current process's PID: the registry's list_all() filters
    # out dead PIDs (a real-world safety net), so an obviously-dead PID
    # like 999999 would be silently skipped and our test would see zero
    # agents. os.getpid() is guaranteed alive throughout the test.
    #
    # On (re)register: the registry refuses to overwrite a live entry
    # with the SAME name (AgentAlreadyExistsError), which matches the
    # real-daemon protocol where the old process must unregister before
    # the new one registers. Emulate that here.
    try:
        reg.unregister(name)
    except Exception:  # noqa: BLE001 — first-call has nothing to remove
        pass
    reg.register(
        AgentProcessInfo(
            name=name,
            pid=os.getpid(),
            pid_file=Path(f"/tmp/{name}.pid"),
            working_dir="/tmp",
            started_at=started_at,
            rpyc_port=48861,
        )
    )


@pytest.mark.integration
def test_agent_started_at_flows_and_broadcasts_on_restart() -> None:
    name = f"test_uptime_{os.getpid()}"
    reg, reg_file = _isolated_registry()

    T0 = datetime.now().replace(microsecond=0)
    _register(reg, name, T0)

    service = AgentService(registry=reg)
    # Prevent AgentService from attempting real RPyC connects to our fake
    # entry — background_connect against port 48861 with a nonexistent PID
    # would fail loudly but not fatally; short-circuit it to keep the
    # integration test hermetic.
    service._start_background_connect = lambda name: None  # type: ignore[assignment]

    async def scenario() -> tuple[list[dict], list[dict]]:
        # initialize() already seeds _agents from the registry, so we
        # verify the loaded started_at right away.
        await service.initialize()
        assert service._agents[name].started_at == T0

        discovered: list[dict] = []
        changed: list[dict] = []

        original_broadcast = service._broadcast_message

        async def capture(msg: dict) -> None:
            t = msg.get("type")
            if t == "agent_discovered":
                discovered.append(msg)
            elif t == "agent_started_at_changed":
                changed.append(msg)
            await original_broadcast(msg)

        service._broadcast_message = capture  # type: ignore[assignment]

        # First reconcile: agent is already known & started_at unchanged
        # → NO broadcast expected. (Sanity: our fix must not spam.)
        await service._reconcile_from_registry()

        # Simulate graceful restart: same name, NEW started_at.
        T1 = T0 + timedelta(seconds=7)
        _register(reg, name, T1)
        await service._reconcile_from_registry()

        return discovered, changed

    discovered, changed = asyncio.run(scenario())

    # (a) First reconcile with unchanged registry: NO change/discovered
    #     broadcast — agent is known, nothing moved.
    assert not any(d["agent"]["name"] == name for d in discovered), (
        f"unexpected agent_discovered for unchanged agent: {discovered}"
    )

    # (b) Restart triggers exactly ONE agent_started_at_changed with the
    #     new ISO started_at.
    assert len(changed) == 1, f"expected exactly one change event, got {changed}"
    assert changed[0]["agent_name"] == name
    T1 = T0 + timedelta(seconds=7)
    assert changed[0]["started_at"] == T1.isoformat()

    # (c) in-memory _agents map reflects the new started_at.
    assert service._agents[name].started_at == T1

    # Cleanup
    asyncio.run(service.shutdown())
    reg_file.unlink(missing_ok=True)
    reg_file.with_suffix(".lock").unlink(missing_ok=True)


# Note: WS transport delivery of ``agent_started_at_changed`` is not
# re-tested here — it uses the same broadcast pipeline as every other
# WS event already covered by ``test_websocket_integration.py``. Adding
# a WebSocket dance around ``TestClient`` in a sync test just to
# re-verify that pipeline led to a cross-event-loop deadlock and no new
# signal (the reconcile ran on one asyncio loop; the WS receiver on
# another). Our new coverage focuses on the actual delta: does reconcile
# emit the event with the right payload and update the in-memory map.
