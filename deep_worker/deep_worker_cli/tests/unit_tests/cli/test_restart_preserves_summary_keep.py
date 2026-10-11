"""``--summary-keep`` must survive ``--restart`` (lesson e7e6eb5).

Observed at the ``run_agent(effective)`` seam, NOT by reading the persisted
file back: a value that round-trips to disk but is dropped while rebuilding
the effective namespace would still be lost in production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deep_worker_cli.cli import agent_command
from deep_worker_cli.cli.agent_command import (
    _normalize_summary_keep,
    build_launch_config,
)


def _write_saved_config(tmp_path: Path, name: str, extra: dict) -> Path:
    data_dir = tmp_path / "agents" / name
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir()
    launch_config = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "working_dir": str(tmp_path),
        "skip_permissions": False,
        "task_manager": False,
        "memory_only": False,
        "history_memory": False,
        "use_sqlite": False,
        "clean": False,
        "auto_approve_send_message": False,
        "dev_tree": False,
        "web_search_policy": "auto",
        "preferred_provider": "deepseek",
        "preferred_model": None,
        "mcp_config_paths": [],
        "mcp_server_inline": [],
        "turn_ending_tools": [],
        "context_limit": None,
        "hot_cache_ttl": {},
        "summary_keep": None,
        "storage_version": 5,
        "tree_root": "/tmp",
        "started_at": "2026-08-18T00:00:00",
        "pid": 1234,
        "cli_args": "",
    }
    launch_config.update(extra)
    metadata = {
        "status": "idle",
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": "2026-08-18T00:00:00",
        "storage_version": 5,
        "launch_config": launch_config,
    }
    (data_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2))
    return data_dir


def _restart_args(name: str, **cli_overrides) -> argparse.Namespace:
    """Namespace for a bare ``--restart``.

    ``summary_keep`` is deliberately ABSENT unless a caller passes it: the
    real parser uses ``default=argparse.SUPPRESS`` for this flag, so on a bare
    restart the attribute does not exist and the value can only come from the
    saved config. Seeding it as ``None`` here would let a dropped restore line
    masquerade as success (the copy-back would re-supply ``None`` and the
    assertion would compare against a value the restore never produced).
    """
    ns = argparse.Namespace(
        name=name,
        restart=True,
        provider="deepseek",
        model=None,
        dir=None,
        skip_permissions=False,
        task_manager=False,
        memory_only=False,
        history_memory=False,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        dev_tree=False,
        web_search_policy="auto",
        preferred_provider=None,
        preferred_model=None,
        no_update_preferred=False,
        mcp_config_paths=[],
        mcp_server_inline=[],
        turn_ending_tools=[],
        config=None,
        summarize=False,
        fork=None,
        fork_from=None,
        context_limit=None,
        hot_cache_ttl=None,
    )
    for key, value in cli_overrides.items():
        setattr(ns, key, value)
    return ns


def _no_registered_agent(monkeypatch) -> None:
    class _EmptyRegistry:
        def __init__(self, *a, **k):
            pass

        def lookup(self, name):
            return None

    monkeypatch.setattr(
        "deep_worker_cli.process.registry.ProcessRegistry",
        _EmptyRegistry,
    )


@pytest.fixture
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "deep_worker_cli.cli.agent_command.deep_worker_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "deep_worker_cli.storage_paths.deep_worker_home",
        lambda h=None: tmp_path,
    )
    return tmp_path


def test_build_launch_config_persists_summary_keep() -> None:
    args = argparse.Namespace(
        provider="deepseek",
        model="deepseek-chat",
        dir=Path("/tmp"),
        skip_permissions=False,
        task_manager=False,
        memory_only=False,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        summary_keep=("tokens", 0),
    )
    cfg = build_launch_config(args)
    assert cfg["summary_keep"] == ("tokens", 0)


def test_normalize_accepts_saved_list_form() -> None:
    """JSON round-trips tuples as lists — both must normalize."""
    assert _normalize_summary_keep(["tokens", 0]) == ("tokens", 0)
    assert _normalize_summary_keep(("tokens", 0)) == ("tokens", 0)
    assert _normalize_summary_keep(["fraction", 0.1]) == ("fraction", 0.1)
    assert _normalize_summary_keep(["messages", 20]) == ("messages", 20)
    assert _normalize_summary_keep("10k") == ("tokens", 10_000)
    assert _normalize_summary_keep(None) is None


def test_normalize_treats_unrecognised_input_as_unset() -> None:
    """Not-a-spec must degrade to ``None``, never raise.

    ``merge_restart_config`` reads this attribute off namespaces that some
    callers build as ``MagicMock``; a missing attribute yields a mock object,
    and raising on it would break an otherwise valid restart. Operator input
    is still validated strictly at argparse time.
    """
    from unittest.mock import MagicMock

    assert _normalize_summary_keep(MagicMock()) is None
    assert _normalize_summary_keep(object()) is None
    assert _normalize_summary_keep(["tokens", "not-a-number"]) is None
    assert _normalize_summary_keep(["nonsense", 1]) is None
    assert _normalize_summary_keep(["tokens"]) is None


@pytest.mark.asyncio
async def test_restart_preserves_summary_keep_zero(_sandbox, monkeypatch):
    """``0`` is falsy — a truthiness guard would silently drop it."""
    _write_saved_config(_sandbox, "a", {"summary_keep": ["tokens", 0]})
    _no_registered_agent(monkeypatch)
    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["summary_keep"] = getattr(effective_ns, "summary_keep", "MISSING")
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)
    rc = await agent_command.restart_agent(_restart_args("a"))
    assert rc == 0
    assert captured.get("summary_keep") == ("tokens", 0), (
        f"summary_keep dropped on bare --restart: {captured.get('summary_keep')!r}"
    )


@pytest.mark.asyncio
async def test_restart_preserves_summary_keep_ceiling(_sandbox, monkeypatch):
    _write_saved_config(_sandbox, "a", {"summary_keep": ["tokens", 10000]})
    _no_registered_agent(monkeypatch)
    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["summary_keep"] = getattr(effective_ns, "summary_keep", "MISSING")
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)
    rc = await agent_command.restart_agent(_restart_args("a"))
    assert rc == 0
    assert captured.get("summary_keep") == ("tokens", 10000)


@pytest.mark.asyncio
async def test_restart_explicit_summary_keep_overrides_saved(_sandbox, monkeypatch):
    _write_saved_config(_sandbox, "a", {"summary_keep": ["fraction", 0.10]})
    _no_registered_agent(monkeypatch)
    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["summary_keep"] = getattr(effective_ns, "summary_keep", "MISSING")
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)
    rc = await agent_command.restart_agent(
        _restart_args("a", summary_keep=("tokens", 0))
    )
    assert rc == 0
    assert captured.get("summary_keep") == ("tokens", 0)
