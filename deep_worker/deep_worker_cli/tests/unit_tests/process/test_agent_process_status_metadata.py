from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


def _make_agent() -> AgentProcess:
    agent = AgentProcess.__new__(AgentProcess)
    agent.name = "alpha"
    agent.session_manager = SessionManager(data_dir=Path(tempfile.mkdtemp()))
    agent.task_manager = None
    return agent


@pytest.mark.asyncio
async def test_status_snapshot_includes_status_metadata() -> None:
    agent = _make_agent()
    await agent.session_manager.set_status(
        AgentStatus.IDLE,
        metadata={"substatus": "waiting_agent", "source_tool": "send_message_to_agent"},
    )

    snapshot = agent._current_status_snapshot()

    assert snapshot["status"] == "idle"
    assert snapshot["status_metadata"] == {
        "substatus": "waiting_agent",
        "source_tool": "send_message_to_agent",
    }


def test_idle_metadata_for_send_message_echo() -> None:
    agent = _make_agent()

    metadata = agent._idle_metadata_for_response("alpha -> beta :> hello")

    assert metadata == {
        "substatus": "waiting_agent",
        "source_tool": "send_message_to_agent",
        "reason": "tool_call",
    }


def test_idle_metadata_for_plain_final_answer_is_none() -> None:
    agent = _make_agent()

    assert agent._idle_metadata_for_response("Done.") is None
