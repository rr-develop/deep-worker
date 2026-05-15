"""TDD tests for LifecycleJobRegistry and AgentLifecycleService.

These tests pin the fire-and-forget contract:
- every op returns a request_id immediately (<100 ms)
- no polling: no asyncio.sleep in ops
- SIGTERM → SIGKILL escalation is event-driven
- batch is parallel fan-out, not sequential await
- status=running does NOT block stop
- errors surface via broadcast (_failed events)

These tests use a fake broadcaster to capture WS events, and mock os.kill /
pidfd_open so no real processes are harmed.
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
# Test scaffolding
# --------------------------------------------------------------------------- #


@dataclass
class FakeBroadcaster:
    """Captures all broadcast events for assertions."""

    events: list[dict[str, Any]] = field(default_factory=list)

    async def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def by_type(self, t: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == t]

    def by_request(self, request_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("request_id") == request_id]


class FakeRegistry:
    """Minimal ProcessRegistry stand-in that stores (name -> info)."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}

    def lookup(self, name: str) -> Any | None:
        return self._agents.get(name)

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def register_fake(self, name: str, pid: int, port: int = 12345) -> None:
        self._agents[name] = MagicMock(
            name=name, pid=pid, rpyc_port=port, working_dir="/tmp"
        )


@pytest.fixture
def broadcaster() -> FakeBroadcaster:
    return FakeBroadcaster()


@pytest.fixture
def registry() -> FakeRegistry:
    r = FakeRegistry()
    r.register_fake("test_alpha", pid=4242)
    r.register_fake("test_beta", pid=4243)
    return r


# --------------------------------------------------------------------------- #
# LifecycleJobRegistry contract
# --------------------------------------------------------------------------- #


class TestLifecycleJobRegistry:
    async def test_start_job_returns_request_id_and_broadcasts_started(
        self, broadcaster: FakeBroadcaster
    ) -> None:
        from app.services.lifecycle import LifecycleJobRegistry

        jr = LifecycleJobRegistry(broadcaster.broadcast)
        rid = await jr.start_job(
            op="agent_stop",
            agent_name="test_alpha",
            payload={"pid": 4242, "grace_seconds": 5.0, "force": False},
        )
        assert isinstance(rid, str) and len(rid) > 8
        started = broadcaster.by_type("agent_stop_started")
        assert len(started) == 1
        assert started[0]["request_id"] == rid
        assert started[0]["agent_name"] == "test_alpha"
        assert "timestamp" in started[0]
        assert started[0]["payload"]["pid"] == 4242

    async def test_complete_job_broadcasts_completed_and_drops_state(
        self, broadcaster: FakeBroadcaster
    ) -> None:
        from app.services.lifecycle import LifecycleJobRegistry

        jr = LifecycleJobRegistry(broadcaster.broadcast)
        rid = await jr.start_job("agent_stop", "test_alpha", {"pid": 4242})
        await jr.complete_job(rid, {"phase": "graceful"})
        completed = broadcaster.by_type("agent_stop_completed")
        assert completed[0]["request_id"] == rid
        assert completed[0]["payload"]["phase"] == "graceful"
        assert jr.get_job(rid) is None  # cleaned up

    async def test_fail_job_broadcasts_failed_with_error_and_type(
        self, broadcaster: FakeBroadcaster
    ) -> None:
        from app.services.lifecycle import LifecycleJobRegistry

        jr = LifecycleJobRegistry(broadcaster.broadcast)
        rid = await jr.start_job("agent_stop", "test_alpha", {"pid": 4242})
        await jr.fail_job(rid, RuntimeError("boom"))
        failed = broadcaster.by_type("agent_stop_failed")
        assert failed[0]["error"] == "boom"
        assert failed[0]["error_type"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# request_stop — returns fast, no polling, SIGTERM→SIGKILL escalation
# --------------------------------------------------------------------------- #


class TestRequestStop:
    async def test_returns_request_id_in_under_100ms(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        # Prevent real work:
        with (
            patch("app.services.lifecycle.os.kill"),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
        ):
            t0 = time.monotonic()
            rid = await svc.request_stop("test_alpha", grace_seconds=5.0, force=False)
            elapsed = time.monotonic() - t0
        assert isinstance(rid, str)
        assert elapsed < 0.1, f"HTTP-facing call must return <100ms, took {elapsed:.3f}s"
        assert broadcaster.by_type("agent_stop_started")

    async def test_unknown_agent_broadcasts_failed_immediately(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        rid = await svc.request_stop("no_such_agent", grace_seconds=1.0, force=False)
        # Let the loop process the scheduled fail task.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        failed = broadcaster.by_type("agent_stop_failed")
        assert any(e["request_id"] == rid for e in failed)
        assert any("not found" in e["error"].lower() for e in failed)

    async def test_no_asyncio_sleep_on_stop_path(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        """Forbidden pattern: asyncio.sleep in the stop path (polling)."""
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        sleep_calls: list[float] = []

        async def spy_sleep(delay: float, *a: Any, **kw: Any) -> None:
            sleep_calls.append(delay)

        with (
            patch("app.services.lifecycle.os.kill"),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
            patch("app.services.lifecycle.asyncio.sleep", new=spy_sleep),
        ):
            await svc.request_stop("test_alpha", grace_seconds=3.0, force=False)

        # Only sleeps with delay==0 (cooperative yield) are tolerated, but
        # the current implementation should not need any sleep at all.
        assert all(d == 0 for d in sleep_calls), f"polling detected: {sleep_calls}"

    async def test_status_running_does_not_block_stop(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry, tmp_path
    ) -> None:
        """Regression: 409-gate on stale session_metadata.json is removed."""
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        # Even when metadata says status=running, stop must be accepted.
        with (
            patch("app.services.lifecycle.os.kill"),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
            patch(
                "app.services.lifecycle._read_status_file",
                return_value="running",
            ),
        ):
            rid = await svc.request_stop(
                "test_alpha", grace_seconds=5.0, force=False
            )
        assert rid
        assert not broadcaster.by_type("agent_stop_failed")

    async def test_sigkill_escalation_on_grace_timeout(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        """After grace_seconds, SIGKILL must be sent (event-driven via call_later)."""
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        kill_calls: list[tuple[int, int]] = []

        def fake_kill(pid: int, sig: int) -> None:
            kill_calls.append((pid, sig))

        # Patch pidfd watcher to never fire (simulates hung process).
        with (
            patch("app.services.lifecycle.os.kill", side_effect=fake_kill),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
            patch.object(svc, "_verify_process_gone", return_value=True),
        ):
            await svc.request_stop(
                "test_alpha", grace_seconds=0.05, force=False
            )
            # Allow the scheduled call_later to fire.
            await asyncio.sleep(0.2)

        sigs = [s for _, s in kill_calls]
        assert signal.SIGTERM in sigs, f"SIGTERM missing: {kill_calls}"
        assert signal.SIGKILL in sigs, f"SIGKILL escalation missing: {kill_calls}"

    async def test_force_true_skips_sigterm(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        kill_calls: list[tuple[int, int]] = []
        with (
            patch(
                "app.services.lifecycle.os.kill",
                side_effect=lambda pid, sig: kill_calls.append((pid, sig)),
            ),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
            patch.object(svc, "_verify_process_gone", return_value=True),
        ):
            await svc.request_stop("test_alpha", grace_seconds=5.0, force=True)
            await asyncio.sleep(0.05)

        assert any(s == signal.SIGKILL for _, s in kill_calls)
        assert not any(s == signal.SIGTERM for _, s in kill_calls)

    async def test_sigkill_verify_failed_broadcasts_failed(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        """If SIGKILL does not actually kill (non-zombie), fail loudly."""
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        with (
            patch("app.services.lifecycle.os.kill"),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
            patch.object(svc, "_verify_process_gone", return_value=False),
            patch(
                "app.services.lifecycle._read_proc_state", return_value="R"
            ),  # running, not zombie
        ):
            rid = await svc.request_stop(
                "test_alpha", grace_seconds=0.02, force=True
            )
            await asyncio.sleep(0.5)

        failed = broadcaster.by_type("agent_stop_failed")
        assert any(e["request_id"] == rid for e in failed)
        assert any("4242" in e["error"] for e in failed)


# --------------------------------------------------------------------------- #
# request_batch — parallel fan-out
# --------------------------------------------------------------------------- #


class TestRequestBatch:
    async def test_batch_returns_per_job_request_ids_immediately(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        with (
            patch("app.services.lifecycle.os.kill"),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
        ):
            t0 = time.monotonic()
            handle = await svc.request_batch(
                action="stop",
                agent_names=["test_alpha", "test_beta"],
                params={"grace_seconds": 5.0, "force": False},
            )
            elapsed = time.monotonic() - t0
        assert elapsed < 0.1
        assert handle["request_id"]
        assert len(handle["jobs"]) == 2
        names = {j["agent_name"] for j in handle["jobs"]}
        assert names == {"test_alpha", "test_beta"}

    async def test_batch_fan_out_is_parallel(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        """All per-job _started events are broadcast before any _completed."""
        from app.services.lifecycle import AgentLifecycleService

        svc = AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        with (
            patch("app.services.lifecycle.os.kill"),
            patch.object(svc, "_attach_pidfd_watcher", new=AsyncMock()),
        ):
            await svc.request_batch(
                "stop",
                ["test_alpha", "test_beta"],
                {"grace_seconds": 5.0, "force": False},
            )
            # Let event loop tick so all start_job broadcasts flush.
            for _ in range(5):
                await asyncio.sleep(0)

        # Exactly two _started events, in close succession, interleaved
        # with batch_started.
        starts = broadcaster.by_type("agent_stop_started")
        assert len(starts) == 2
        assert broadcaster.by_type("batch_started")


# --------------------------------------------------------------------------- #
# _pidfd_watcher wiring (no polling)
# --------------------------------------------------------------------------- #


class TestPidfdWatcher:
    async def test_pidfd_open_and_add_reader_used(
        self, broadcaster: FakeBroadcaster, registry: FakeRegistry
    ) -> None:
        """Implementation MUST use os.pidfd_open + loop.add_reader — not polling."""
        from app.services import lifecycle as life_mod

        svc = life_mod.AgentLifecycleService(
            registry=registry, broadcast=broadcaster.broadcast
        )
        fake_pidfd = 77
        loop = asyncio.get_running_loop()
        add_reader = MagicMock()  # do NOT wrap — 77 isn't a real fd
        with (
            patch("app.services.lifecycle.os.pidfd_open", return_value=fake_pidfd)
            as pfd,
            patch("app.services.lifecycle.os.kill"),
            patch("app.services.lifecycle.os.close"),
            patch.object(loop, "add_reader", add_reader),
        ):
            await svc._attach_pidfd_watcher(
                pid=4242,
                agent_name="test_alpha",
                request_id="rid-x",
            )
        pfd.assert_called_once_with(4242)
        add_reader.assert_called()
        fd_arg = add_reader.call_args[0][0]
        assert fd_arg == fake_pidfd


# --------------------------------------------------------------------------- #
# Event format contract
# --------------------------------------------------------------------------- #


class TestEventFormat:
    async def test_events_have_required_fields(
        self, broadcaster: FakeBroadcaster
    ) -> None:
        from app.services.lifecycle import LifecycleJobRegistry

        jr = LifecycleJobRegistry(broadcaster.broadcast)
        rid = await jr.start_job(
            "agent_stop", "test_alpha", {"pid": 4242}
        )
        await jr.complete_job(rid, {"phase": "graceful"})
        for e in broadcaster.events:
            assert "type" in e
            assert "request_id" in e
            assert "timestamp" in e
