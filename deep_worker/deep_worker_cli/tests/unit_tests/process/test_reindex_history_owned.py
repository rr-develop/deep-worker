from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

@pytest.mark.asyncio
async def test_reindex_history_owned_runs_under_invoke_lock(monkeypatch, tmp_path) -> None:
    from deep_worker_cli.process import agent_process as mod
    from deep_worker_cli.process.agent_process import AgentProcess

    proc = AgentProcess.__new__(AgentProcess)
    proc.name = "ag"
    proc._invoke_lock = asyncio.Lock()
    entered_while_locked: list[bool] = []

    class Client:
        async def index_history(self, path, clear_first=False):
            entered_while_locked.append(proc._invoke_lock.locked())
            assert clear_first is True
            return SimpleNamespace(
                files_indexed=1,
                chunks_added=3,
                lines_read=2,
                errors=0,
            )

    import deep_worker_cli.summaries as summaries_mod
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir()
    monkeypatch.setattr(summaries_mod, "get_summaries_dir", lambda name: summaries_dir)
    import deep_worker_cli.semantic_memory.client as client_mod
    monkeypatch.setattr(client_mod, "SemanticMemoryClient", Client)

    result = await proc.reindex_history_owned()

    assert entered_while_locked == [True]
    assert result == {"files": 1, "lines": 2, "chunks": 3, "errors": 0}


@pytest.mark.asyncio
async def test_reindex_history_owned_uses_index_history_rpc(monkeypatch, tmp_path) -> None:
    from deep_worker_cli.process import agent_process as mod
    from deep_worker_cli.process.agent_process import AgentProcess

    proc = AgentProcess.__new__(AgentProcess)
    proc.name = "ag"
    proc._invoke_lock = asyncio.Lock()
    calls: list[tuple[object, bool]] = []

    class Client:
        async def index_history(self, path, clear_first=False):
            calls.append((path, clear_first))
            return SimpleNamespace(
                files_indexed=1,
                chunks_added=3,
                lines_read=2,
                errors=0,
            )

    import deep_worker_cli.summaries as summaries_mod
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir()
    monkeypatch.setattr(summaries_mod, "get_summaries_dir", lambda name: summaries_dir)
    import deep_worker_cli.semantic_memory.client as client_mod
    monkeypatch.setattr(client_mod, "SemanticMemoryClient", Client)

    await proc.reindex_history_owned()

    assert calls == [(summaries_dir, True)]


@pytest.mark.asyncio
async def test_exposed_reindex_history_clear_failure_returns_rpc_error() -> None:
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    class Proc:
        session_manager = SimpleNamespace(status="idle")

        async def reindex_history_owned(self):
            raise RuntimeError("clear failed")

    service = AgentService.__new__(AgentService)
    service._agent_process = Proc()

    payload = json.loads(await service.exposed_reindex_history())

    assert payload["success"] is False
    assert "clear failed" in payload["error"]


@pytest.mark.asyncio
async def test_exposed_reindex_history_returns_stats() -> None:
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    class Proc:
        session_manager = SimpleNamespace(status="idle")

        async def reindex_history_owned(self):
            return {"files": 1, "lines": 2, "chunks": 3, "errors": 0}

    service = AgentService.__new__(AgentService)
    service._agent_process = Proc()

    payload = json.loads(await service.exposed_reindex_history())

    assert payload == {
        "success": True,
        "files": 1,
        "lines": 2,
        "chunks": 3,
        "errors": 0,
    }


@pytest.mark.asyncio
async def test_exposed_reindex_history_refuses_non_idle_agent() -> None:
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    class Proc:
        session_manager = SimpleNamespace(status="running")

        async def reindex_history_owned(self):
            raise AssertionError("must not reindex when busy")

    service = AgentService.__new__(AgentService)
    service._agent_process = Proc()

    payload = json.loads(await service.exposed_reindex_history())

    assert payload["success"] is False
    assert "busy" in payload["error"]


@pytest.mark.asyncio
async def test_reindex_history_owned_missing_summaries_dir_raises(
    monkeypatch, tmp_path
) -> None:
    """Fail-loud invariant: absent summaries dir must raise, not return zero stats.

    Mirrors the offline CLI path guard for the running+idle agent-owned path;
    exposed_reindex_history converts the raised RuntimeError to
    {success: False, error: ...} so the CLI running branch exits non-zero.
    """
    from deep_worker_cli.process import agent_process as mod
    from deep_worker_cli.process.agent_process import AgentProcess

    proc = AgentProcess.__new__(AgentProcess)
    proc.name = "no_such_agent"
    proc._invoke_lock = asyncio.Lock()

    import deep_worker_cli.summaries as summaries_mod
    missing_dir = tmp_path / "definitely" / "does_not_exist"
    monkeypatch.setattr(summaries_mod, "get_summaries_dir", lambda name: missing_dir)

    called = False

    async def fake_to_thread(func, *args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(
            files_processed=0, lines_read=0, chunks_indexed=0, errors=0
        )

    monkeypatch.setattr(mod.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="no summaries directory"):
        await proc.reindex_history_owned()

    assert called is False, (
        "clear+index must NOT run when summaries dir is absent"
    )


@pytest.mark.asyncio
async def test_exposed_reindex_history_missing_summaries_dir_returns_rpc_error() -> None:
    """End-to-end at the RPyC boundary: missing dir surfaces as {success:false, error:...}."""
    from deep_worker_cli.rpyc_service.agent_service import AgentService

    class Proc:
        session_manager = SimpleNamespace(status="idle")

        async def reindex_history_owned(self):
            raise RuntimeError(
                "no summaries directory for agent 'ag' (/nope)"
            )

    service = AgentService.__new__(AgentService)
    service._agent_process = Proc()

    payload = json.loads(await service.exposed_reindex_history())

    assert payload["success"] is False
    assert "no summaries directory" in payload["error"]
