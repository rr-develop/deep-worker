"""Context status updates are driven by every model call."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.services.agent_service import AgentService
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


def _setup() -> tuple[AgentService, str, list[dict[str, Any]]]:
    tmp = tempfile.mkdtemp(prefix="dw_web_ctxrefresh_")
    reg_file = Path(tmp) / "registry.json"
    reg_file.write_text("{}")
    registry = ProcessRegistry(
        registry_file=reg_file, lock_file=reg_file.with_suffix(".lock")
    )
    name = f"ctxrefresh_{os.getpid()}"
    registry.register(
        AgentProcessInfo(
            name=name, pid=os.getpid(), pid_file=Path(f"/tmp/{name}.pid"),
            working_dir="/tmp", started_at=datetime.now(), rpyc_port=48871,
        )
    )
    service = AgentService(registry=registry)
    service._start_background_connect = lambda n: None
    sent: list[dict[str, Any]] = []

    async def capture(message: dict[str, Any]) -> None:
        sent.append(message)

    service._broadcast_message = capture
    return service, name, sent


def _model_call(*, kind: str, prompt: int, total: int) -> str:
    return json.dumps({
        "type": "model_context_changed", "schema_version": 1,
        "provider": "claude-code", "model": "fable", "kind": kind,
        "status": "ok", "usage_semantics": "tokens",
        "context_used": None, "context_size": None,
        "prompt_tokens": prompt, "completion_tokens": total - prompt,
        "total_tokens": total,
    })


@pytest.mark.integration
def test_every_model_call_emits_context_frame() -> None:
    service, name, sent = _setup()

    async def scenario() -> None:
        await service.initialize()
        sent.clear()
        await service._dispatch_subscriber_payload(
            name, _model_call(kind="agent", prompt=10_000, total=10_100)
        )
        await service._dispatch_subscriber_payload(
            name, _model_call(kind="agent", prompt=10_500, total=10_650)
        )
        frames = [m for m in sent if m.get("type") == "agent_context_changed"]
        assert len(frames) == 2
        assert [m["agent_context_used"] for m in frames] == [10_000, 10_500]
        assert [m["agent_last_turn_tokens"] for m in frames] == [10_100, 10_650]

    asyncio.run(scenario())


@pytest.mark.integration
def test_tm_model_call_updates_tm_context_without_status_change() -> None:
    service, name, sent = _setup()

    async def scenario() -> None:
        await service.initialize()
        sent.clear()
        await service._dispatch_subscriber_payload(
            name, _model_call(kind="tm", prompt=20_000, total=20_250)
        )
        frames = [m for m in sent if m.get("type") == "agent_context_changed"]
        assert len(frames) == 1
        assert frames[0]["tm_context_used"] == 20_000
        assert frames[0]["tm_last_turn_tokens"] == 20_250
        assert not [m for m in sent if m.get("type") == "agent_status"]

    asyncio.run(scenario())
