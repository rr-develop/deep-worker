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

    summaries = tmp_path / Path("~/.deep_worker").name / "ag" / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "one.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))
    monkeypatch.setattr(agent_command, "clear_history_index", lambda path: calls.append("clear"), raising=False)
    monkeypatch.setattr(
        agent_command,
        "index_all_history_files",
        lambda path, on_progress=None, on_message_progress=None,
        on_analyze_progress=None: calls.append("index")
        or SimpleNamespace(files_processed=1, lines_read=1, chunks_indexed=1, errors=0),
        raising=False,
    )

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 0
    assert calls == ["clear", "index"]


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

    (tmp_path / Path("~/.deep_worker").name / "ag" / "summaries").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))

    rc = await agent_command.reindex_history_command("ag")

    assert rc == 0
    assert "nothing to reindex" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_reindex_offline_stats_errors_exit_1(monkeypatch, tmp_path) -> None:
    from deep_worker_cli.cli import agent_command

    summaries = tmp_path / Path("~/.deep_worker").name / "ag" / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "one.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(agent_command, "ProcessRegistry", lambda: _Registry(None))
    monkeypatch.setattr(agent_command, "clear_history_index", lambda path: None, raising=False)
    monkeypatch.setattr(
        agent_command,
        "index_all_history_files",
        lambda path, on_progress=None, on_message_progress=None,
        on_analyze_progress=None: SimpleNamespace(
            files_processed=1, lines_read=1, chunks_indexed=0, errors=1
        ),
        raising=False,
    )

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


def test_run_reindex_with_progress_wires_analyze_and_message(monkeypatch) -> None:
    """`_run_reindex_with_progress` must forward BOTH an analyze callback
    (Phase 1) and a message callback (Phase 2) to the indexer, and drive them
    through a single rich Progress task. Regression guard: the "Analyzing
    history…" phase must exist and precede the "Indexing" phase.
    """
    from pathlib import Path

    from deep_worker_cli.cli import agent_command

    # Stub the indexer func: exercise the callbacks it was handed exactly like
    # the real one would — analyze ticks first, then message ticks.
    order: list[str] = []

    def fake_index(summaries_dir, *, on_progress, on_message_progress,
                   on_analyze_progress):
        on_progress("thread_x.jsonl")
        on_analyze_progress(0, 3)
        on_analyze_progress(3, 3)
        on_message_progress(0, 5)
        on_message_progress(5, 5)
        return SimpleNamespace(files_processed=1, lines_read=5,
                               chunks_indexed=7, errors=0)

    monkeypatch.setattr(
        agent_command, "_history_indexer_funcs",
        lambda: (None, fake_index),
    )

    # Capture description updates to assert the phase labels appear in order.
    descriptions: list[str] = []
    import rich.progress as rp

    real_update = rp.Progress.update

    def spy_update(self, task_id, **kwargs):
        desc = kwargs.get("description")
        if desc is not None:
            descriptions.append(desc)
        return real_update(self, task_id, **kwargs)

    monkeypatch.setattr(rp.Progress, "update", spy_update)

    stats = agent_command._run_reindex_with_progress(Path("/tmp/summaries"))
    assert stats.files_processed == 1

    analyzing = [d for d in descriptions if d.startswith("Analyzing")]
    indexing = [d for d in descriptions if d.startswith("Indexing")]
    assert analyzing, "Phase 1 'Analyzing …' must drive the bar"
    assert indexing, "Phase 2 'Indexing …' must drive the bar"
    # first Analyzing appears before first Indexing
    assert descriptions.index(analyzing[0]) < descriptions.index(indexing[0])
