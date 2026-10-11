from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from deep_worker_cli.types import AgentProcessInfo


def _running_info(name: str = "ag") -> AgentProcessInfo:
    from pathlib import Path

    return AgentProcessInfo(
        name=name,
        pid=12345,
        pid_file=Path("/tmp/ag.pid"),
        working_dir="/tmp",
        started_at=datetime.now(),
        last_activity=datetime.now(),
        rpyc_port=18861,
    )


class _Registry:
    def __init__(self, result=None):
        self.result = result

    def lookup(self, name: str):
        return self.result


class _Client:
    def __init__(self, name: str):
        self.name = name
        self.connected = False
        self.reindex_called = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def get_status(self):
        return SimpleNamespace(status="idle")

    async def reindex_history(self):
        self.reindex_called = True
        return {"success": True, "files": 2, "lines": 3, "chunks": 4, "errors": 0}


class _EnumLike:
    value = "idle"


def test_reindex_history_flag_parses() -> None:
    from deep_worker_cli.cli.agent_command import create_parser

    args = create_parser().parse_args(["ag", "--reindex-history"])
    assert args.name == "ag"
    assert args.reindex_history is True


def test_reindex_history_requires_name(monkeypatch, capsys) -> None:
    from deep_worker_cli.cli import agent_command

    monkeypatch.setattr(sys, "argv", ["deep-worker", "--reindex-history"])
    rc = agent_command.main()
    assert rc == 2
    assert "--reindex-history requires an agent name" in capsys.readouterr().err


def test_reindex_history_dispatches_command(monkeypatch) -> None:
    from deep_worker_cli.cli import agent_command

    seen: list[str] = []

    async def fake_command(name: str) -> int:
        seen.append(name)
        return 37

    monkeypatch.setattr(sys, "argv", ["deep-worker", "ag", "--reindex-history"])
    monkeypatch.setattr(agent_command, "reindex_history_command", fake_command, raising=False)

    assert agent_command.main() == 37
    assert seen == ["ag"]


@pytest.mark.asyncio
async def test_reindex_offline_calls_clear_then_index(monkeypatch, tmp_path) -> None:
    from deep_worker_cli.cli import agent_command
    from deep_worker_cli.storage_paths import agent_root

    summaries = agent_root("ag", home=tmp_path) / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "one.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[object, bool]] = []

    class Client:
        async def index_history(self, path, clear_first=False):
            calls.append((path, clear_first))
            return SimpleNamespace(
                files_indexed=1,
                chunks_added=1,
                lines_read=1,
                errors=0,
            )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))
    import deep_worker_cli.semantic_memory.client as client_mod
    monkeypatch.setattr(client_mod, "SemanticMemoryClient", Client)

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 0
    assert calls == [(summaries, True)]


@pytest.mark.asyncio
async def test_reindex_offline_no_summaries_dir_exit_1(monkeypatch, tmp_path, capsys) -> None:
    from deep_worker_cli.cli import agent_command

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 1
    assert "no summaries directory" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_reindex_offline_no_jsonl_files_exit_0(monkeypatch, tmp_path, capsys) -> None:
    from deep_worker_cli.cli import agent_command
    from deep_worker_cli.storage_paths import agent_root

    (agent_root("ag", home=tmp_path) / "summaries").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 0
    assert "nothing to reindex" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_reindex_offline_stats_errors_exit_1(monkeypatch, tmp_path) -> None:
    from deep_worker_cli.cli import agent_command
    from deep_worker_cli.storage_paths import agent_root

    summaries = agent_root("ag", home=tmp_path) / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "one.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))
    class Client:
        async def index_history(self, path, clear_first=False):
            return SimpleNamespace(
                files_indexed=1,
                chunks_added=0,
                lines_read=1,
                errors=1,
            )

    import deep_worker_cli.semantic_memory.client as client_mod
    monkeypatch.setattr(client_mod, "SemanticMemoryClient", Client)

    assert await agent_command.reindex_history_command("ag") == 1


@pytest.mark.asyncio
async def test_reindex_running_busy_refuses_exit_1(monkeypatch, capsys) -> None:
    from deep_worker_cli.cli import agent_command

    class BusyClient(_Client):
        async def get_status(self):
            return SimpleNamespace(status="running")

    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(_running_info()))
    monkeypatch.setattr(agent_command, "AgentClient", BusyClient, raising=False)

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 1
    assert "busy" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_reindex_running_idle_delegates_via_rpyc(monkeypatch) -> None:
    from deep_worker_cli.cli import agent_command

    client = _Client("ag")
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(_running_info()))
    monkeypatch.setattr(agent_command, "AgentClient", lambda name: client, raising=False)

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 0
    assert client.reindex_called is True


@pytest.mark.asyncio
async def test_reindex_status_guard_accepts_enum_like_idle(monkeypatch) -> None:
    from deep_worker_cli.cli import agent_command

    class EnumClient(_Client):
        async def get_status(self):
            return SimpleNamespace(status=_EnumLike())

    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(_running_info()))
    monkeypatch.setattr(agent_command, "AgentClient", EnumClient, raising=False)

    assert await agent_command.reindex_history_command("ag") == 0


@pytest.mark.asyncio
async def test_reindex_status_guard_refuses_unknown_status(monkeypatch) -> None:
    from deep_worker_cli.cli import agent_command

    class UnknownClient(_Client):
        async def get_status(self):
            return SimpleNamespace(status="mystery")

    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(_running_info()))
    monkeypatch.setattr(agent_command, "AgentClient", UnknownClient, raising=False)

    assert await agent_command.reindex_history_command("ag") == 1
