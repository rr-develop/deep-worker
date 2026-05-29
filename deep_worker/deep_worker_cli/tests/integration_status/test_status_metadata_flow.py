from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


@pytest.mark.asyncio
async def test_status_metadata_real_session_manager_flow(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    async def notifier(payload: dict[str, Any]) -> None:
        events.append(payload)

    sm = SessionManager(data_dir=tmp_path, notifier=notifier)

    await sm.set_status(AgentStatus.RUNNING)
    await sm.set_status(
        AgentStatus.IDLE,
        metadata={
            "substatus": "waiting_agent",
            "source_tool": "send_message_to_agent",
            "reason": "tool_call",
        },
    )
    waiting_saved = json.loads((tmp_path / "session_metadata.json").read_text())
    assert waiting_saved["status"] == "idle"
    assert waiting_saved["status_metadata"]["substatus"] == "waiting_agent"
    assert events[-1]["status_metadata"]["source_tool"] == "send_message_to_agent"

    await sm.set_status(AgentStatus.RUNNING)
    await sm.set_status(AgentStatus.IDLE)

    final_saved = json.loads((tmp_path / "session_metadata.json").read_text())
    assert final_saved["status"] == "idle"
    assert "status_metadata" not in final_saved
    assert "status_metadata" not in events[-1]
