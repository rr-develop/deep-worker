"""``--hot-cache-ttl`` must survive ``--restart`` (e7e6eb5 / DESIGN §4).

Observed at ``run_agent(effective)``, not by reading the persisted file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deep_worker_cli.cli import agent_command
from deep_worker_cli.cli.agent_command import (
    _normalize_hot_cache_ttl,
    build_launch_config,
    parse_hot_cache_ttl,
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


def test_parse_hot_cache_ttl_forms() -> None:
    assert parse_hot_cache_ttl("240") == ("*", 240)
    assert parse_hot_cache_ttl("claude-code=180") == ("claude-code", 180)
    assert parse_hot_cache_ttl("0") == ("*", 0)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_hot_cache_ttl("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_hot_cache_ttl("=30")


def test_build_launch_config_persists_ttl_dict() -> None:
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
        hot_cache_ttl=[("claude-code", 180), ("*", 0)],
    )
    cfg = build_launch_config(args)
    assert cfg["hot_cache_ttl"] == {"claude-code": 180, "*": 0}


def test_normalize_accepts_saved_dict() -> None:
    assert _normalize_hot_cache_ttl({"Claude-Code": "0"}) == {"claude-code": 0}


@pytest.mark.asyncio
async def test_restart_preserves_hot_cache_ttl(_sandbox, monkeypatch):
    _write_saved_config(_sandbox, "a", {"hot_cache_ttl": {"*": 0, "grok": 120}})
    _no_registered_agent(monkeypatch)
    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["hot_cache_ttl"] = getattr(effective_ns, "hot_cache_ttl", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)
    rc = await agent_command.restart_agent(_restart_args("a"))
    assert rc == 0
    assert captured.get("hot_cache_ttl") == {"*": 0, "grok": 120}, (
        f"hot_cache_ttl dropped on bare --restart: {captured.get('hot_cache_ttl')!r}"
    )


@pytest.mark.asyncio
async def test_restart_explicit_ttl_overrides_saved(_sandbox, monkeypatch):
    _write_saved_config(_sandbox, "a", {"hot_cache_ttl": {"*": 240}})
    _no_registered_agent(monkeypatch)
    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["hot_cache_ttl"] = getattr(effective_ns, "hot_cache_ttl", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)
    rc = await agent_command.restart_agent(
        _restart_args("a", hot_cache_ttl=[("claude-code", 0)])
    )
    assert rc == 0
    assert captured.get("hot_cache_ttl") == {"claude-code": 0}
