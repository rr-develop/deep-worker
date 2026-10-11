from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest

from deep_worker_cli.semantic_memory import client
from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb


class _RpcError(grpc.aio.AioRpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


def test_resolve_project_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = SimpleNamespace(state={"working_dir": str(tmp_path / "runtime")})
    monkeypatch.chdir(tmp_path)
    assert client.resolve_project_path(runtime, str(tmp_path / "kwarg")) == (
        tmp_path / "runtime"
    ).resolve()
    assert client.resolve_project_path(None, str(tmp_path / "kwarg")) == (
        tmp_path / "kwarg"
    ).resolve()
    assert client.resolve_project_path(None, None) == tmp_path.resolve()


def test_resolve_explicit_project_path_wins(tmp_path: Path) -> None:
    runtime = SimpleNamespace(state={"working_dir": "/wrong"})
    assert client.resolve_project_path(runtime, "/also-wrong", tmp_path) == tmp_path.resolve()


def test_resolve_history_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path / "runtime-summaries"
    runtime = SimpleNamespace(state={"summaries_dir": str(runtime_dir)})
    monkeypatch.setattr(client, "deep_worker_home", lambda: tmp_path / "home")
    assert client.resolve_history_target(runtime, None) == (runtime_dir.resolve(), "current agent")
    assert client.resolve_history_target(runtime, "alice") == (
        (tmp_path / "home/agents/alice/summaries").resolve(),
        "alice",
    )


@pytest.mark.parametrize("top_k", [0, 21])
def test_top_k_is_rejected_before_rpc(top_k: int) -> None:
    assert client.clamp_top_k(top_k) == "Error: top_k must be between 1 and 20"


@pytest.mark.asyncio
async def test_unavailable_retries_once_after_dropping_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = SimpleNamespace(
        ensure_started=Mock(return_value=Path("/tmp/memory.sock")),
        registry=SimpleNamespace(drop=Mock()),
    )
    memory = client.SemanticMemoryClient(supervisor)
    rpc = AsyncMock(side_effect=[_RpcError(grpc.StatusCode.UNAVAILABLE), "ok"])
    stub = SimpleNamespace(SearchProject=rpc)
    channel = AsyncMock()
    channel.__aenter__.return_value = SimpleNamespace()
    monkeypatch.setattr(client.grpc.aio, "insecure_channel", Mock(return_value=channel))
    monkeypatch.setattr(client.pb2_grpc, "SemanticMemoryStub", Mock(return_value=stub))

    result = await memory._call("SearchProject", pb.ProjectSearchRequest(), 30.0)

    assert result == "ok"
    assert supervisor.ensure_started.call_count == 2
    supervisor.registry.drop.assert_called_once_with()


@pytest.mark.asyncio
async def test_non_unavailable_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = SimpleNamespace(
        ensure_started=Mock(return_value=Path("/tmp/memory.sock")),
        registry=SimpleNamespace(drop=Mock()),
    )
    memory = client.SemanticMemoryClient(supervisor)
    rpc = AsyncMock(side_effect=_RpcError(grpc.StatusCode.DEADLINE_EXCEEDED))
    channel = AsyncMock()
    channel.__aenter__.return_value = SimpleNamespace()
    monkeypatch.setattr(client.grpc.aio, "insecure_channel", Mock(return_value=channel))
    monkeypatch.setattr(
        client.pb2_grpc,
        "SemanticMemoryStub",
        Mock(return_value=SimpleNamespace(SearchProject=rpc)),
    )

    with pytest.raises(grpc.aio.AioRpcError):
        await memory._call("SearchProject", pb.ProjectSearchRequest(), 30.0)
    assert supervisor.ensure_started.call_count == 1


def test_deadlines_and_tenant_suffix() -> None:
    assert (client.HEALTH_TIMEOUT_S, client.SEARCH_TIMEOUT_S) == (5.0, 30.0)
    assert (client.INDEX_TIMEOUT_S, client.STATUS_TIMEOUT_S) == (120.0, 5.0)
    reply = SimpleNamespace(tenants=["a", "b"], total=5)
    assert client.format_tenants(reply) == "a, b … (+3 more)"
    assert client.format_tenants(SimpleNamespace(tenants=["a"], total=1)) == "a"


@pytest.mark.asyncio
async def test_shutdown_event_respawns_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = SimpleNamespace(
        ensure_started=Mock(return_value=Path("/tmp/memory.sock")),
        registry=SimpleNamespace(drop=Mock()),
    )
    memory = client.SemanticMemoryClient(supervisor)
    async def stream(event_type: int):
        yield pb.SessionEvent(event_type=event_type)

    streams = [
        stream(pb.SESSION_EVENT_SHUTDOWN),
        stream(pb.SESSION_EVENT_ATTACHED),
    ]

    class Stub:
        def AttachSession(self, request: object, timeout: object = None) -> object:
            return streams.pop(0)

    channel = AsyncMock()
    channel.__aenter__.return_value = SimpleNamespace()
    monkeypatch.setattr(client.grpc.aio, "insecure_channel", Mock(return_value=channel))
    monkeypatch.setattr(client.pb2_grpc, "SemanticMemoryStub", Mock(return_value=Stub()))
    stop = asyncio.Event()

    events = []
    async for event in memory.session_events("agent:1", stop):
        events.append(event.event_type)
        if event.event_type == pb.SESSION_EVENT_ATTACHED:
            stop.set()

    assert events == [pb.SESSION_EVENT_ATTACHED]
    supervisor.registry.drop.assert_called_once_with()
    assert supervisor.ensure_started.call_count == 2
