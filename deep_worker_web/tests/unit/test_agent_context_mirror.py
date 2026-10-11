"""Tests for AgentInfo context fields and ``_fill_context_fields`` helper.

DESIGN_AGENT_CONTEXT_STATUS_BAR.md — web mirror piece.

Covers mutations M13 (backend gate tm_* by has_task_manager), and pins
the AgentInfo schema (6 new fields) for round-trip through /api/agents.

Each test fails RED until the impl adds:
  * 6 fields to AgentInfo (agent_context_used/size/semantics,
    tm_context_used/size/semantics).
  * ``AgentService._fill_context_fields(info)`` helper that reads
    ``token_stats.jsonl`` and fills the 6 fields.
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
from deep_worker_cli.stats.token_stats import ReadStats


def _make_agent_info(name: str = "ctxagent") -> AgentInfo:
    return AgentInfo(
        name=name,
        pid=1,
        status=AgentStatus.IDLE,
        provider="claude-code-sdk",
        model="opus",
        rpyc_port=1,
        working_dir="/tmp",
        started_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# AgentInfo schema round-trip (6 fields)
# ---------------------------------------------------------------------------


def test_agent_info_accepts_agent_context_fields() -> None:
    """AgentInfo MUST accept agent_context_used/size/semantics."""
    info = AgentInfo(
        name="a",
        pid=1,
        status=AgentStatus.IDLE,
        provider="x",
        model="y",
        rpyc_port=1,
        working_dir="/tmp",
        started_at=datetime.now(),
        agent_context_used=42100,
        agent_context_size=200000,
        agent_context_semantics="context_snapshot",
    )
    assert info.agent_context_used == 42100
    assert info.agent_context_size == 200000
    assert info.agent_context_semantics == "context_snapshot"

    payload = json.loads(info.model_dump_json())
    assert payload["agent_context_used"] == 42100
    assert payload["agent_context_size"] == 200000
    assert payload["agent_context_semantics"] == "context_snapshot"


def test_agent_info_accepts_tm_context_fields() -> None:
    """AgentInfo MUST accept tm_context_used/size/semantics."""
    info = AgentInfo(
        name="a",
        pid=1,
        status=AgentStatus.IDLE,
        provider="x",
        model="y",
        rpyc_port=1,
        working_dir="/tmp",
        started_at=datetime.now(),
        tm_context_used=500,
        tm_context_size=10000,
        tm_context_semantics="tokens",
    )
    payload = json.loads(info.model_dump_json())
    assert payload["tm_context_used"] == 500
    assert payload["tm_context_size"] == 10000
    assert payload["tm_context_semantics"] == "tokens"


def test_agent_info_context_fields_default_none() -> None:
    """All 6 new fields default to None when not provided (page-load before
    any model run)."""
    info = _make_agent_info()
    assert info.agent_context_used is None
    assert info.agent_context_size is None
    assert info.agent_context_semantics is None
    assert info.tm_context_used is None
    assert info.tm_context_size is None
    assert info.tm_context_semantics is None


# ---------------------------------------------------------------------------
# _fill_context_fields helper
# ---------------------------------------------------------------------------


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


def test_fill_context_fields_reads_agent_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_fill_context_fields`` MUST populate agent_context_* from
    the latest kind=agent snapshot record."""
    # Redirect storage_paths to tmp_path.
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    name = "ctxagent"
    data_dir = agent_root(name, home=tmp_path, create=True)
    _write_jsonl(data_dir / "token_stats.jsonl", [
        _rec(kind="agent", semantics="context_snapshot", used=42000, size=200000,
             ts="2026-08-01T12:00:00.000Z"),
    ])

    info = _make_agent_info(name)
    svc = AgentService(registry=object())
    svc._fill_context_fields(info)  # type: ignore[attr-defined]

    assert info.agent_context_used == 42000
    assert info.agent_context_size == 200000
    assert info.agent_context_semantics == "context_snapshot"


def test_fill_context_fields_reads_tm_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_fill_context_fields`` MUST populate tm_context_* when a kind=tm record exists."""
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    name = "ctxagent"
    data_dir = agent_root(name, home=tmp_path, create=True)
    _write_jsonl(data_dir / "token_stats.jsonl", [
        _rec(kind="agent", semantics="context_snapshot", used=42000, size=200000),
        _rec(kind="tm", semantics="context_snapshot", used=500, size=10000,
             ts="2026-08-01T12:01:00.000Z"),
    ])

    info = _make_agent_info(name)
    info.has_task_manager = True
    svc = AgentService(registry=object())
    svc._fill_context_fields(info)  # type: ignore[attr-defined]

    assert info.tm_context_used == 500
    assert info.tm_context_size == 10000
    assert info.tm_context_semantics == "context_snapshot"


def test_fill_context_fields_clears_tm_when_no_task_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M13: even if kind=tm records exist in JSONL, the helper MUST NOT
    surface them when ``has_task_manager is False`` (REST snapshot stays
    clean; UI gate alone is not enough)."""
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    name = "ctxagent"
    data_dir = agent_root(name, home=tmp_path, create=True)
    # A tm record exists in JSONL (from a previous run).
    _write_jsonl(data_dir / "token_stats.jsonl", [
        _rec(kind="tm", semantics="context_snapshot", used=500, size=10000),
    ])

    info = _make_agent_info(name)
    # Agent has NO TaskManager (e.g. legacy data, or provider without TM).
    info.has_task_manager = False

    svc = AgentService(registry=object())
    svc._fill_context_fields(info)  # type: ignore[attr-defined]

    assert info.tm_context_used is None
    assert info.tm_context_size is None
    assert info.tm_context_semantics is None


def test_fill_context_fields_swallows_broken_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M11: corrupt JSONL must not raise — the helper silently no-ops."""
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    name = "ctxagent"
    data_dir = agent_root(name, home=tmp_path, create=True)
    (data_dir / "token_stats.jsonl").write_text("{not-json}\n")

    info = _make_agent_info(name)
    svc = AgentService(registry=object())
    # MUST NOT raise.
    svc._fill_context_fields(info)  # type: ignore[attr-defined]

    # No data → all None.
    assert info.agent_context_used is None
    assert info.tm_context_used is None


def test_fill_context_fields_missing_jsonl_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing JSONL (new agent, no runs yet) → all None."""
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    info = _make_agent_info("freshagent")
    svc = AgentService(registry=object())
    svc._fill_context_fields(info)  # type: ignore[attr-defined]

    assert info.agent_context_used is None
    assert info.tm_context_used is None


def test_fill_context_fields_caches_read_stats_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M12 — per-agent cache: 100 broadcasts in < cache-window should
    hit read_stats_file at most once. Without cache, every broadcast
    triggers a disk read — too expensive.

    RED until impl adds a per-agent cache around _fill_context_fields.
    """
    monkeypatch.setattr("deep_worker_cli.storage_paths.deep_worker_home",
                        lambda *a, **kw: tmp_path)
    monkeypatch.setattr("app.services.agent_service.deep_worker_home",
                        lambda *a, **kw: tmp_path, raising=False)

    from app.services import agent_service

    calls: list[Any] = []
    fake = ReadStats(records=[], skipped_records=0,
                     duplicate_records=0, partial_tail=False)

    def fake_read_stats_file(path: Any) -> ReadStats:
        calls.append(path)
        return fake

    monkeypatch.setattr(agent_service, "read_stats_file", fake_read_stats_file)

    name = "cacheagent"
    info = _make_agent_info(name)
    svc = AgentService(registry=object())
    svc._agents[name] = info

    # 100 calls in a tight loop — must hit the disk at most once.
    for _ in range(100):
        svc._fill_context_fields(info)  # type: ignore[attr-defined]

    assert len(calls) <= 1, (
        f"M12: per-agent cache must debounce disk reads, got {len(calls)} "
        f"read_stats_file calls for 100 broadcasts within one cache window"
    )