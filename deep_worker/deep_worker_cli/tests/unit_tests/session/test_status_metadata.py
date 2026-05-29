from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


@pytest.mark.asyncio
async def test_set_status_persists_and_notifies_metadata(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    async def notify(payload: dict[str, Any]) -> None:
        events.append(payload)

    sm = SessionManager(data_dir=tmp_path, notifier=notify)

    await sm.set_status(
        AgentStatus.IDLE,
        metadata={
            "substatus": "waiting_agent",
            "source_tool": "send_message_to_agent",
            "reason": "tool_call",
        },
    )

    saved = json.loads((tmp_path / "session_metadata.json").read_text())
    assert saved["status"] == "idle"
    assert saved["status_metadata"] == {
        "substatus": "waiting_agent",
        "source_tool": "send_message_to_agent",
        "reason": "tool_call",
    }
    assert events[-1]["status_metadata"] == saved["status_metadata"]


@pytest.mark.asyncio
async def test_set_status_without_metadata_clears_stale_metadata(tmp_path: Path) -> None:
    sm = SessionManager(data_dir=tmp_path)
    await sm.set_status(AgentStatus.IDLE, metadata={"substatus": "waiting_agent"})

    await sm.set_status(AgentStatus.RUNNING)
    await sm.set_status(AgentStatus.IDLE)

    saved = json.loads((tmp_path / "session_metadata.json").read_text())
    assert "status_metadata" not in saved
    assert sm.status_metadata is None


def test_loads_existing_status_metadata(tmp_path: Path) -> None:
    (tmp_path / "session_metadata.json").write_text(
        json.dumps(
            {
                "status": "idle",
                "tasks_completed": 0,
                "message_count": 0,
                "error_message": None,
                "last_activity": "2026-05-26T10:00:00",
                "status_metadata": {"substatus": "waiting_agent", "extra": "kept"},
            }
        )
    )

    sm = SessionManager(data_dir=tmp_path)

    assert sm.status_metadata == {"substatus": "waiting_agent", "extra": "kept"}


def test_restart_clears_status_metadata(tmp_path: Path) -> None:
    (tmp_path / "session_metadata.json").write_text(
        json.dumps(
            {
                "status": "idle",
                "tasks_completed": 0,
                "message_count": 0,
                "error_message": None,
                "last_activity": "2026-05-26T10:00:00",
                "status_metadata": {"substatus": "waiting_agent"},
            }
        )
    )

    sm = SessionManager(data_dir=tmp_path, launch_config={"provider": "codex"})

    assert sm.status_metadata is None
    saved = json.loads((tmp_path / "session_metadata.json").read_text())
    assert "status_metadata" not in saved
