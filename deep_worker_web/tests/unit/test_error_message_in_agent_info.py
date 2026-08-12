"""Regression: the agent's error_message must reach the UI on PAGE LOAD,
not only via a live WebSocket push.

User report (2026-05-20): an agent was in ``error`` but the web UI
showed no error detail. The persistent error panel reads
``agents[name].error_message`` (set from the snapshot the page loads),
but:

  * the ``AgentInfo`` schema had NO ``error_message`` field, so
    ``GET /api/agents`` / the single-agent endpoint / the
    ``agent_discovered`` snapshot never carried it; and
  * the status-poll path broadcast ``error_message`` live but never
    wrote it into the ``AgentInfo`` mirror, so a browser that loaded
    the page AFTER the error transition had no way to learn the cause.

Contract under test:
  * ``AgentInfo`` accepts and round-trips ``error_message``.
  * The status poll writes the agent's error into
    ``self._agents[name].error_message`` (so ``GET /api/agents`` and a
    re-select read it), and clears it when the agent recovers.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest

from app.schemas.agent_schemas import AgentInfo, AgentStatus
from app.services.agent_service import AgentService

from deep_worker_cli.storage_paths import deep_worker_home
def test_agent_info_accepts_error_message() -> None:
    info = AgentInfo(
        name="a",
        pid=1,
        status=AgentStatus.ERROR,
        provider="claude-code",
        model="opus",
        rpyc_port=1,
        working_dir="/tmp",
        started_at=datetime.now(),
        error_message="boom",
    )
    assert info.error_message == "boom"
    # Round-trips through serialization (the /api/agents response).
    assert json.loads(info.model_dump_json())["error_message"] == "boom"


def test_agent_info_error_message_defaults_none() -> None:
    info = AgentInfo(
        name="a",
        pid=1,
        status=AgentStatus.IDLE,
        provider="x",
        model="y",
        rpyc_port=1,
        working_dir="/tmp",
        started_at=datetime.now(),
    )
    assert info.error_message is None


def _make_service(name: str = "erragent") -> AgentService:
    svc = AgentService(registry=object())
    svc._agents[name] = AgentInfo(
        name=name,
        pid=1,
        status=AgentStatus.IDLE,
        provider="claude-code",
        model="opus",
        rpyc_port=1,
        working_dir="/tmp",
        started_at=datetime.now(),
    )
    return svc


class _FakeRoot:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def get_status(self) -> str:
        return json.dumps(self._payload)


async def _poll(svc: AgentService, name: str, root: Any) -> None:
    async def capture(_msg: dict[str, Any]) -> None:
        pass
    svc._broadcast_message = capture  # type: ignore[method-assign]
    await svc._update_agent_status(name, root)
    for _ in range(20):
        await asyncio.sleep(0.01)
        if svc._agents[name].status == AgentStatus.ERROR or svc._agents[name].error_message:
            break


@pytest.mark.asyncio
async def test_poll_writes_error_message_into_agent_info() -> None:
    """When the agent reports status=error, the poll must store the cause
    in the AgentInfo mirror so /api/agents (page load) carries it."""
    svc = _make_service("erragent")
    err = "Agent invocation failed: Claude CLI subprocess was hung."
    root = _FakeRoot({"success": True, "status": "error", "error": err})

    await _poll(svc, "erragent", root)

    assert svc._agents["erragent"].error_message == err, (
        "the status poll must write error_message into the AgentInfo "
        "mirror so a page load shows the error detail (incident observed 2026-05-20)"
    )


@pytest.mark.asyncio
async def test_poll_clears_error_message_on_recovery() -> None:
    """When the agent is healthy again, error_message must be cleared so
    a stale error panel doesn't linger after recovery."""
    svc = _make_service("erragent")
    svc._agents["erragent"].error_message = "old error"
    svc._agents["erragent"].status = AgentStatus.ERROR
    root = _FakeRoot({"success": True, "status": "idle", "error": None})

    await _poll(svc, "erragent", root)

    assert svc._agents["erragent"].error_message is None


@pytest.mark.asyncio
async def test_discover_seeds_error_from_persisted_metadata(tmp_path, monkeypatch) -> None:
    """On discover (which feeds the page-load snapshot), the persisted
    ``session_metadata.json`` is the DURABLE source of the error: even
    when the agent's live in-memory status has diverged to idle (the
    incident observed 2026-05-20 case), the UI must still show the error from
    disk.
    """
    import app.services.agent_service as asvc

    # Fake registry returning one agent.
    class _ProcInfo:
        name = "diverged_agent"
        pid = 4321
        rpyc_port = 9999
        working_dir = "/tmp/wd"
        started_at = datetime.now()
        pid_file = "/tmp/x.pid"

    class _Registry:
        def list_all(self):
            return [_ProcInfo()]

    # Persisted metadata says error with a cause.
    data_dir = deep_worker_home(tmp_path) / "diverged_agent"
    data_dir.mkdir(parents=True)
    (data_dir / "session_metadata.json").write_text(json.dumps({
        "status": "error",
        "error_message": "Claude CLI subprocess was hung.",
    }))

    # Point the agent-data-dir resolver at our tmp home.
    monkeypatch.setattr(asvc.Path, "home", staticmethod(lambda: tmp_path))

    svc = AgentService(registry=_Registry())
    await svc._discover_agents()

    info = svc._agents["diverged_agent"]
    assert info.status == AgentStatus.ERROR, (
        "discover must seed status from persisted session_metadata.json"
    )
    assert info.error_message == "Claude CLI subprocess was hung.", (
        "discover must seed error_message from disk so a page load shows "
        "the error even if the live agent status diverged (incident observed 2026-05-20)"
    )
