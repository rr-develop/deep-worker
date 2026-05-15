"""Wave 3 — no task, no callback, no subscriber error may vanish
without a log.

Wave 1 added a catch-all for the whole app. Wave 2 hardened the WS
and REST surfaces. Wave 3 cleans up the remaining silent-failure
pockets: loose ``asyncio.create_task`` calls whose exceptions can be
lost at GC time, the broadcast gather that collected
``return_exceptions=True`` results and then dropped them on the
floor, a stale RPyC shutdown path that returned ``False`` with no
trace, and a block of unreachable code in
``get_agent_dialog_formatted``.

Contracts pinned here
---------------------

* ``_broadcast_message`` logs every per-callback exception at
  ERROR level under ``app.services.agent_service`` — one broken
  subscriber must not hide that it was broken.
* Every internal ``asyncio.create_task`` site in ``AgentService``
  is wrapped so a crash in the spawned coroutine lands in the log
  under ``app.utils.supervisor`` rather than as a GC-time
  warning.
* ``AgentLifecycleService._try_rpyc_shutdown``, when it falls back
  to SIGTERM because the RPyC RPC failed, broadcasts a structured
  ``agent_shutdown_rpc_failed`` event so the UI can show what
  actually happened.
* The body of ``get_agent_dialog_formatted`` contains no
  unreachable code after the ``return`` statement — unreachable
  code either works or it doesn't, and here it definitely didn't.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.agent_service import AgentService


class _FakeRegistry:
    def __init__(self) -> None:
        self.registry_file = "/tmp/_nonexistent_registry.json"

    def list_all(self) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_broadcast_message_logs_callback_exceptions(caplog) -> None:
    """When a registered message callback raises, the broadcast
    loop collects the exceptions via ``return_exceptions=True`` —
    but it MUST log them too, otherwise a broken subscriber stays
    invisible until a user complains about missing events."""
    svc = AgentService(registry=_FakeRegistry())

    async def good(_msg: dict[str, Any]) -> None:
        return None

    async def bad(_msg: dict[str, Any]) -> None:
        raise RuntimeError("subscriber exploded")

    svc.register_message_callback(good)
    svc.register_message_callback(bad)

    with caplog.at_level(logging.ERROR, logger="app.services.agent_service"):
        await svc._broadcast_message({"type": "probe"})

    matches = [
        rec for rec in caplog.records
        if rec.levelno >= logging.ERROR
        and "subscriber exploded" in (rec.exc_text or rec.getMessage())
    ]
    assert matches, (
        "broadcast must log per-callback exceptions; caplog: "
        + "; ".join(rec.getMessage() for rec in caplog.records)
    )


@pytest.mark.asyncio
async def test_reconcile_spawn_is_supervised(monkeypatch, caplog) -> None:
    """``_reconcile_from_registry`` spawns ``connect_to_agent`` as a
    background task for each newly-discovered agent. If that task
    crashes, the exception must reach ``app.utils.supervisor`` —
    never disappear at GC time."""
    svc = AgentService(registry=_FakeRegistry())

    class _Agent:
        name = "ghost"
        pid = 111
        rpyc_port = 18999
        working_dir = "/tmp"
        started_at = None

    monkeypatch.setattr(
        svc._registry, "list_all", lambda: [_Agent()]
    )

    # Make connect_to_agent blow up so the spawned task fails.
    async def boom(name: str) -> bool:
        raise RuntimeError("synthetic connect failure")

    svc.connect_to_agent = boom  # type: ignore[assignment]

    # ``_reconcile_from_registry`` also calls
    # ``AgentInfo.model_dump(mode="json")`` on discovered agents.
    # Stub that out using a minimal info dict via monkeypatching
    # the constructor of AgentInfo -> but simpler: patch the
    # broadcast so we don't care about message shape.
    svc._broadcast_message = AsyncMock(return_value=None)  # type: ignore[assignment]

    # Need ``started_at`` to be a datetime for AgentInfo(..., started_at=...).
    from datetime import datetime, timezone
    _Agent.started_at = datetime.now(timezone.utc)  # type: ignore[attr-defined]

    with caplog.at_level(logging.ERROR, logger="app.utils.supervisor"):
        await svc._reconcile_from_registry()
        # The spawned task is fire-and-forget; give the loop a
        # tick so its done-callback fires.
        await asyncio.sleep(0.02)

    # The record MUST come from ``app.utils.supervisor`` — that's
    # the marker that the create_task site was wrapped with the
    # ``supervise()`` helper. Default asyncio emits "Task exception
    # was never retrieved" under the ``asyncio`` logger at GC time;
    # that is what we want to eliminate, so matching on it would
    # give false positives.
    matches = [
        rec for rec in caplog.records
        if rec.name == "app.utils.supervisor"
        and "synthetic connect failure" in (rec.exc_text or "")
    ]
    assert matches, (
        "the create_task(self.connect_to_agent(...)) site in "
        "_reconcile_from_registry must be wrapped with supervise(); "
        "records by logger name: "
        + "; ".join(f"{rec.name}:{rec.getMessage()}" for rec in caplog.records)
    )


def test_get_agent_dialog_formatted_has_no_unreachable_tail() -> None:
    """A stale ``fire_and_forget_async`` block lived after the
    ``return`` in ``get_agent_dialog_formatted``. It never ran and
    confused readers. Wave 3 removes it."""
    src = inspect.getsource(AgentService.get_agent_dialog_formatted)
    # Count top-level return statements: there should be exactly
    # one in the function body (there are inner returns inside
    # ``on_success``/``on_error`` nested functions; we don't care
    # about those). The simplest invariant is that the function
    # must not reference ``fire_and_forget_async`` — the orphaned
    # tail is the only reason it would.
    assert "fire_and_forget_async" not in src, (
        "get_agent_dialog_formatted should no longer contain the "
        "unreachable fire_and_forget_async tail"
    )


@pytest.mark.asyncio
async def test_rpyc_shutdown_failure_broadcasts_agent_shutdown_rpc_failed(
    caplog,
) -> None:
    """``_try_rpyc_shutdown`` returns False when the RPC path
    fails — historically the caller falls back to SIGTERM, but
    the UI never sees the failure reason. Wave 3 broadcasts a
    structured event so the UI can show it."""
    from app.services.lifecycle import AgentLifecycleService

    broadcasts: list[dict[str, Any]] = []

    async def capture_broadcast(msg: dict[str, Any]) -> None:
        broadcasts.append(msg)

    # Minimal rpyc_resolver that returns an object whose
    # ``shutdown`` attribute access itself raises — simulates a
    # connection that died mid-call.
    class _ExplodingRoot:
        @property
        def shutdown(self) -> Any:  # noqa: D401
            raise ConnectionError("connection torn down")

    def resolver(_name: str) -> Any:
        return _ExplodingRoot()

    svc = AgentLifecycleService(
        registry=MagicMock(),
        broadcast=capture_broadcast,
        rpyc_resolver=resolver,
    )

    with caplog.at_level(logging.WARNING):
        ok = await svc._try_rpyc_shutdown(
            "vc_x", grace_seconds=2.0, force=False
        )

    assert ok is False, "RPC path must signal failure to the caller"

    types_seen = [msg.get("type") for msg in broadcasts]
    assert "agent_shutdown_rpc_failed" in types_seen, (
        "caller falling back to SIGTERM must see a structured event; "
        "got: " + repr(types_seen)
    )
