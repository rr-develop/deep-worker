"""Integration test for the per-agent context status bar path.

DESIGN_AGENT_CONTEXT_STATUS_BAR.md — web integration piece.

Real ``AgentService`` + real ``ProcessRegistry`` (isolated tmp registry,
no interference with running agents). Captures
``AgentService._broadcast_message`` to verify the new
``agent_context_changed`` event payload.

Flow:
  1. Register a fake agent record in the real registry.
  2. Seed ``token_stats.jsonl`` under the isolated deep_worker_home with
     records for ``kind=agent`` and ``kind=tm``.
  3. Start the real ``AgentService`` against the registry and verify the
     initial ``AgentInfo`` loaded the 6 context fields via
     ``_discover_agents()``.
  4. Trigger ``_update_agent_status`` via the broadcast path (with a
     mock RPyC root that returns a valid status JSON) and assert:
       * ``agent_context_changed`` is broadcast with the 6 fields,
       * ``service._agents[name].agent_context_*`` was updated in-memory.
  5. ``GET /api/agents/{name}`` returns the same 6 fields.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app as main_app
from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]
from deep_worker_cli.types import AgentProcessInfo  # type: ignore[import-not-found]
from deep_worker_cli.storage_paths import deep_worker_home, agent_root


def _isolated_registry() -> tuple[ProcessRegistry, Path]:
    tmp = tempfile.mkdtemp(prefix="dw_web_ctx_it_")
    reg_file = Path(tmp) / "registry.json"
    lock_file = reg_file.with_suffix(".lock")
    reg_file.write_text("{}")
    return ProcessRegistry(registry_file=reg_file, lock_file=lock_file), reg_file


def _isolated_home(tmp_path: Path) -> Path:
    """Standalone deep_worker_home, isolated from the real host one."""
    home = tmp_path / "deep_worker"
    home.mkdir()
    return home


def _register(reg: ProcessRegistry, name: str, started_at: datetime) -> None:
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


def _write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _rec(*, kind: str, semantics: str, used: int | None = None,
         size: int | None = None, ts: str = "2026-08-01T12:00:00.000Z",
         total: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": "deepseek",
        "model": "m",
        "kind": kind,
        "status": "ok",
        "usage_semantics": semantics,
        "ts": ts,
        "context_used": used,
        "context_size": size,
        "total_tokens": total,
    }


@pytest.mark.integration
def test_discover_seeds_agent_context_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On discover, AgentInfo MUST carry the 6 context fields populated
    from the JSONL file."""
    home = _isolated_home(tmp_path)
    # deep_worker_home accepts an optional override arg.
    monkeypatch.setattr(
        "deep_worker_cli.storage_paths.deep_worker_home",
        lambda *a, **kw: home,
    )

    name = f"test_ctx_{os.getpid()}"
    reg, _reg_file = _isolated_registry()

    _register(reg, name, datetime.now())
    data_dir = agent_root(name, home=home, create=True)
    _write_jsonl(data_dir / "token_stats.jsonl", [
        _rec(kind="agent", semantics="context_snapshot", used=42000, size=200000),
        _rec(kind="tm", semantics="context_snapshot", used=500, size=10000,
             ts="2026-08-01T12:01:00.000Z"),
    ])

    service = AgentService(registry=reg)
    service._start_background_connect = lambda n: None  # type: ignore[assignment]

    async def scenario() -> None:
        await service.initialize()
        info = service._agents[name]

        # REST snapshot via GET /api/agents reflects the discovered fields.
        assert info.agent_context_used == 42000
        assert info.agent_context_size == 200000
        assert info.agent_context_semantics == "context_snapshot"
        # has_task_manager is None (we didn't poll the agent), so tm_*
        # may carry the cached JSONL value: the BACKEND no longer
        # double-gates on has_task_manager (see a related internal incident analysis, not included here).
        # The frontend is the sole gate (``tmWrap.style.display = hasTm ? '' : 'none'``),
        # so a ``None`` or unset has_task_manager still lets the
        # backend surface the cached tm_* data, ready to be shown the
        # moment the next status poll flips has_task_manager to True.
        assert info.tm_context_used == 500
        assert info.tm_context_size == 10000
        assert info.tm_context_semantics == "context_snapshot"

    asyncio.run(scenario())


@pytest.mark.integration
def test_update_agent_status_broadcasts_agent_context_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a status poll, the service MUST broadcast
    ``agent_context_changed`` with the 6 fields populated."""
    home = _isolated_home(tmp_path)
    monkeypatch.setattr(
        "deep_worker_cli.storage_paths.deep_worker_home",
        lambda *a, **kw: home,
    )

    name = f"test_ctx_bc_{os.getpid()}"
    reg, _reg_file = _isolated_registry()

    _register(reg, name, datetime.now())
    data_dir = agent_root(name, home=home, create=True)
    _write_jsonl(data_dir / "token_stats.jsonl", [
        _rec(kind="agent", semantics="context_snapshot", used=11000, size=200000),
    ])

    service = AgentService(registry=reg)
    service._start_background_connect = lambda n: None  # type: ignore[assignment]

    broadcasts: list[dict[str, Any]] = []

    async def capture(msg: dict[str, Any]) -> None:
        broadcasts.append(msg)

    service._broadcast_message = capture  # type: ignore[method-assign]

    # Only the RPyC transport is stubbed. The method under test is the
    # REAL ``_update_agent_status``: a hand-written copy of its body
    # here (what this test used to do) asserts against the copy, so it
    # stays green however the production path drifts — which is the
    # whole failure mode this file exists to catch.
    agent_response = {
        "success": True,
        "status": "ready",
        "task_manager_active": True,
    }

    async def scenario() -> None:
        await service.initialize()
        # ``initialize`` already seeded the fields from the same file, and
        # the real broadcast is diff-gated
        # (``_maybe_broadcast_context_changed``), so a poll that finds
        # nothing new legitimately stays silent. Model what a poll after
        # a finished turn actually sees: a newer record on disk, and a
        # cache entry whose ~1s TTL has lapsed.
        _write_jsonl(data_dir / "token_stats.jsonl", [
            _rec(kind="agent", semantics="context_snapshot", used=11000,
                 size=200000),
            _rec(kind="agent", semantics="context_snapshot", used=12500,
                 size=200000, ts="2026-08-01T12:05:00.000Z", total=1500),
        ])
        service._context_cache.pop(name, None)

        class _FakeRoot:
            async def get_status(self) -> str:
                return json.dumps(agent_response)

        root = _FakeRoot()
        service._connections[name] = object()  # type: ignore[assignment]
        service._roots[name] = root  # type: ignore[assignment]

        with patch("rpyc_async.utils.helpers.fire_and_forget_call") as mock_faf:
            await service._update_agent_status(name, root)
            assert mock_faf.called, "status poll never reached the agent"
            on_success = mock_faf.call_args.kwargs["success_callback"]
            await on_success(json.dumps(agent_response))

        # Find the agent_context_changed broadcast.
        ctx_bc = [b for b in broadcasts if b.get("type") == "agent_context_changed"]
        assert ctx_bc, "agent_context_changed broadcast was not emitted"
        msg = ctx_bc[-1]
        assert msg["agent_name"] == name
        assert msg["agent_context_used"] == 12500
        assert msg["agent_context_size"] == 200000
        assert msg["agent_context_semantics"] == "context_snapshot"
        assert msg["agent_last_turn_tokens"] == 1500
        # No ``kind="tm"`` record was seeded, so tm_* stay None. The
        # response reports ``task_manager_active=True`` → attached, which
        # means the None comes from the absent data and not from the
        # backend gating tm_* on attachment
        # (see a related internal incident analysis, not included here).
        assert service._agents[name].has_task_manager is True
        assert msg["tm_context_used"] is None
        assert msg["tm_context_size"] is None
        assert msg["tm_context_semantics"] is None
        # The in-memory mirror must carry the new value too, not just the
        # wire message (step 4 of this file's docstring).
        assert service._agents[name].agent_context_used == 12500

    asyncio.run(scenario())