"""Tests for ``AgentInfo.context_limit`` mirror (operator-set -l value).

The web UI's status bar shows the operator's enforced context limit
("-l 400k") alongside the current usage — this test pins the backend
plumbing that reads ``launch_config.context_limit`` from
``session_metadata.json`` and exposes it through the AgentInfo schema.

Without this mirror, the UI cannot tell the user "the guard is
holding the limit at 400k" — the operator would have to read JSON
or run ``deep-worker --info`` to see it.

DESIGN_AGENT_CONTEXT_STATUS_BAR.md §1 (operator-set limit display).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService

from deep_worker_cli.storage_paths import agent_root


def _make_info(name: str = "ctx_limit_web") -> AgentInfo:
    return AgentInfo(
        name=name,
        pid=12345,
        status=AgentStatus.IDLE,
        provider="claude-code",
        model="fable",
        rpyc_port=42000,
        working_dir="/tmp/x",
        started_at=datetime(2026, 8, 18, 12, 0, 0),
    )


@pytest.mark.asyncio
async def test_agent_info_schema_has_context_limit_field() -> None:
    """AgentInfo must declare a ``context_limit`` field (int | None)."""
    info = _make_info()
    # Default None — no metadata seeded yet.
    assert info.context_limit is None
    # Round-trip via model_dump (the route serializer uses this).
    data = info.model_dump()
    assert "context_limit" in data
    assert data["context_limit"] is None


@pytest.mark.asyncio
async def test_context_limit_set_explicit_value() -> None:
    """Setting context_limit persists via the schema."""
    info = _make_info()
    info.context_limit = 400_000
    data = info.model_dump()
    assert data["context_limit"] == 400_000


@pytest.mark.asyncio
async def test_fill_context_limit_from_session_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_fill_context_limit_fields`` reads ``launch_config.context_limit``.

    This is the agent_a 2026-08-18 path: operator passed
    ``-l 400k``, session_metadata.json persisted it, and the web UI
    must surface that value — not the provider default.
    """
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    _seed_session_metadata(
        agent_name="ctx_limit_web",
        home=tmp_path,
        launch_config={
            "provider": "claude-code",
            "model": "fable",
            "context_limit": 400_000,
        },
    )

    svc = AgentService(registry=object())
    info = _make_info(name="ctx_limit_web")
    svc._fill_context_limit_fields(info)
    assert info.context_limit == 400_000, (
        "AgentInfo.context_limit must mirror launch_config.context_limit "
        "from session_metadata.json — without this, the web UI shows "
        "the WRONG limit and the operator cannot verify the guard."
    )


@pytest.mark.asyncio
async def test_fill_context_limit_none_when_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default behaviour: no ``-l`` → ``context_limit is None``."""
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    _seed_session_metadata(
        agent_name="ctx_limit_web",
        home=tmp_path,
        launch_config={
            "provider": "claude-code",
            "model": "fable",
            # No context_limit — operator used default.
        },
    )

    svc = AgentService(registry=object())
    info = _make_info(name="ctx_limit_web")
    svc._fill_context_limit_fields(info)
    assert info.context_limit is None


def _seed_session_metadata(
    *, agent_name: str, home: Path, launch_config: dict[str, Any]
) -> Path:
    """Plant a session_metadata.json for the given agent under ``home``.

    Returns the path to the file. Mirrors the shape used by
    ``agent_command.py::build_launch_config``:
      { "launch_config": { ... }, "status": "ready", "started_at": ... }
    """
    data_dir = agent_root(agent_name, home=home, create=True)
    payload = {
        "launch_config": launch_config,
        "status": "ready",
        "started_at": "2026-08-18T12:00:00+00:00",
        "last_activity": "2026-08-18T12:00:00+00:00",
    }
    target = data_dir / "session_metadata.json"
    target.write_text(json.dumps(payload))
    return target


@pytest.mark.asyncio
async def test_fill_context_limit_missing_metadata_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No session_metadata.json → field stays None (no exception).

    The fill helper must never raise — it runs on the hot path of
    ``_update_agent_status`` and a missing file is the cold-boot case
    (agent registered before its first write to disk).
    """
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    svc = AgentService(registry=object())
    info = _make_info(name="ghost_agent")
    # Must not raise even though no metadata exists.
    svc._fill_context_limit_fields(info)
    assert info.context_limit is None


@pytest.mark.asyncio
async def test_fill_context_limit_corrupt_metadata_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt JSON in session_metadata.json → field stays None.

    Same as above — corruption must NEVER crash the status update.
    """
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    # Plant a corrupt session_metadata.json.
    data_dir = agent_root("ctx_limit_web", home=tmp_path, create=True)
    (data_dir / "session_metadata.json").write_text("{not valid json")
    svc = AgentService(registry=object())
    info = _make_info(name="ctx_limit_web")
    svc._fill_context_limit_fields(info)
    assert info.context_limit is None
