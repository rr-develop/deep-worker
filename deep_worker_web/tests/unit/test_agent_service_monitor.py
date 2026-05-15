"""The agent-discovery monitor must be inotify-driven, not polling.

Pinned invariants:

* No periodic ``asyncio.sleep(5)``-style loop that scans the registry.
* Exactly one inotify watch on the registry-file directory, wired via
  ``loop.add_reader``.
* When the registry file is rewritten atomically (rename-in-place),
  the reader callback reconciles the in-memory agent map and emits
  the same ``agent_discovered`` / ``agent_terminated`` broadcasts the
  old loop used to emit.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent_service import AgentService


class _RegistryStub:
    """Minimal stand-in for deep_worker_cli ProcessRegistry."""

    def __init__(self) -> None:
        self._records: list[MagicMock] = []

    def list_all(self) -> list[MagicMock]:
        return list(self._records)

    def lookup(self, name: str) -> MagicMock | None:
        for r in self._records:
            if r.name == name:
                return r
        return None

    def set(self, names: list[str]) -> None:
        recs: list[MagicMock] = []
        for n in names:
            m = MagicMock()
            m.name = n
            m.pid = 10000 + len(recs)
            m.rpyc_port = 20000 + len(recs)
            m.working_dir = "/tmp"
            from datetime import datetime
            m.started_at = datetime.now()
            recs.append(m)
        self._records = recs


@pytest.fixture
def service() -> AgentService:
    return AgentService(registry=_RegistryStub())


async def test_monitor_does_not_sleep_poll(service: AgentService) -> None:
    """monitor_agents must not invoke asyncio.sleep for polling."""
    sleep_calls: list[float] = []

    async def spy_sleep(delay: float, *a: Any, **kw: Any) -> None:
        sleep_calls.append(delay)

    # Start monitor in the background; cancel after a tick.
    task = asyncio.create_task(service.monitor_agents())
    # Let the loop schedule the watcher.
    with patch("app.services.agent_service.asyncio.sleep", new=spy_sleep):
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Polling is defined as any sleep > 0; cooperative yields (sleep(0))
    # are tolerated.
    polling_sleeps = [d for d in sleep_calls if d > 0]
    assert polling_sleeps == [], f"polling detected: {polling_sleeps}"


async def test_monitor_wires_inotify_loop_reader(service: AgentService) -> None:
    """The monitor must use ``os.read`` + ``loop.add_reader`` — not
    ``run_in_executor(registry.list_all)`` in a loop."""
    loop = asyncio.get_running_loop()
    add_reader = MagicMock()
    add_reader_orig = loop.add_reader

    def fake_add_reader(fd: int, cb: Any, *args: Any) -> None:
        add_reader(fd, cb, *args)
        # still wire it so shutdown works
        add_reader_orig(fd, cb, *args)

    with patch.object(loop, "add_reader", side_effect=fake_add_reader):
        task = asyncio.create_task(service.monitor_agents())
        await asyncio.sleep(0.05)  # let it install the watcher
        service._shutdown_event.set()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()
    assert add_reader.call_count >= 1, "monitor must register an inotify reader"


async def test_registry_change_is_reconciled_without_sleep(
    service: AgentService,
) -> None:
    """When the watcher fires, agents are discovered/terminated
    without any intervening polling sleep."""
    broadcasts: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        broadcasts.append(msg)

    service._broadcast_message = capture  # type: ignore[method-assign]

    reg: _RegistryStub = service._registry  # type: ignore[assignment]

    # 1) start monitor (no agents yet)
    task = asyncio.create_task(service.monitor_agents())
    await asyncio.sleep(0.05)

    # 2) inject new agent + trigger one reconcile pass
    reg.set(["alpha"])
    await service._reconcile_from_registry()  # type: ignore[attr-defined]
    await asyncio.sleep(0)

    assert any(
        b.get("type") == "agent_discovered"
        and (b.get("agent") or {}).get("name") == "alpha"
        for b in broadcasts
    ), f"expected agent_discovered for alpha; got {broadcasts}"

    # 3) remove the agent + reconcile
    reg.set([])
    await service._reconcile_from_registry()  # type: ignore[attr-defined]

    assert any(
        b.get("type") == "agent_terminated" and b.get("agent_name") == "alpha"
        for b in broadcasts
    ), f"expected agent_terminated for alpha; got {broadcasts}"

    service._shutdown_event.set()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        task.cancel()
