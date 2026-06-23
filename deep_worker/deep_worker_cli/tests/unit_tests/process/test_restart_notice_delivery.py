"""Unit test: agent reads restart-notice on start, delivers it as a
SystemMessage in the next ainvoke, and DOES NOT go idle.

This is the mechanism that must be true regardless of which model or
provider is in use.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _write_notice(data_dir: Path, *, cause: str = "self") -> Path:
    """Helper: place a restart-notice on disk like restart_self would."""
    data_dir.mkdir(parents=True, exist_ok=True)
    notice = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cause": cause,
        "message": "test notice",
    }
    notice_path = data_dir / ".restart_notice.json"
    notice_path.write_text(json.dumps(notice))
    return notice_path


def test_pending_restart_notice_loaded_on_start(tmp_path: Path) -> None:
    """AgentProcess.start() must consume `.restart_notice.json` if present.

    The notice file is REMOVED after reading (so next start sees a clean
    slate) and stored on the AgentProcess instance for later delivery.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    data_dir = tmp_path / "agent_data"
    notice_path = _write_notice(data_dir)

    # The helper that loads & consumes the notice should be on
    # AgentProcess. We don't want to instantiate a real AgentProcess
    # (it pulls in registry, sqlite, etc.) — but the loader must be
    # importable and side-effect-free except for unlinking the file.
    assert hasattr(AgentProcess, "_load_and_consume_restart_notice"), (
        "AgentProcess must expose a method to load the restart notice "
        "from data_dir and remove the file in one shot."
    )

    fake_self = MagicMock()
    fake_self.data_dir = data_dir
    fake_self.name = "test_agent"

    notice = AgentProcess._load_and_consume_restart_notice(fake_self)

    # File consumed.
    assert not notice_path.exists(), "Notice file should be removed after read"
    # Notice returned with expected keys.
    assert notice is not None
    assert notice["cause"] == "self"
    assert "ts" in notice


def test_no_notice_means_no_action(tmp_path: Path) -> None:
    """If no notice file exists, the loader returns None silently."""
    from deep_worker_cli.process.agent_process import AgentProcess

    data_dir = tmp_path / "agent_data"
    data_dir.mkdir()

    fake_self = MagicMock()
    fake_self.data_dir = data_dir
    fake_self.name = "test_agent"

    notice = AgentProcess._load_and_consume_restart_notice(fake_self)
    assert notice is None


def test_corrupt_notice_returns_none_and_removes_file(tmp_path: Path) -> None:
    """Corrupt JSON must not crash — file is removed, returns None."""
    from deep_worker_cli.process.agent_process import AgentProcess

    data_dir = tmp_path / "agent_data"
    data_dir.mkdir()
    bad = data_dir / ".restart_notice.json"
    bad.write_text("{not valid json")

    fake_self = MagicMock()
    fake_self.data_dir = data_dir
    fake_self.name = "test_agent"

    notice = AgentProcess._load_and_consume_restart_notice(fake_self)
    assert notice is None
    assert not bad.exists(), "Bad notice file should still be removed"


def test_notice_to_system_message_format() -> None:
    """The loaded notice must convert to a SystemMessage with an
    informative — not interrogative — content string.

    The agent MUST NOT be asked to confirm; the notice is purely
    informational.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    notice = {
        "ts": "2026-06-12T15:00:00+00:00",
        "cause": "self",
        "message": "restart_self called",
    }
    sys_msg = AgentProcess._format_restart_notice_message(None, notice)  # type: ignore[arg-type]

    from langchain_core.messages import SystemMessage
    assert isinstance(sys_msg, SystemMessage), type(sys_msg)
    text = str(sys_msg.content)

    # Must mention the timestamp and the cause.
    assert "2026-06-12T15:00:00" in text, text
    # cause=self translates to a human-readable phrase.
    assert "self-initiated" in text.lower() or "self" in text.lower(), text

    # Must NOT ask for confirmation.
    forbidden = [
        "please confirm",
        "respond with",
        "acknowledge",
        "?",
    ]
    for tok in forbidden:
        assert tok.lower() not in text.lower(), (
            f"Restart notice must be informational only. "
            f"Found forbidden token {tok!r} in: {text!r}"
        )
