"""Tests for SessionManager status reset on agent restart.

BUG: When agent crashes with status=error and restarts, SessionManager.__init__()
loads the old error status from session_metadata.json but does NOT reset it.
The agent appears stuck in error even though it's fully functional.

These tests verify the fix: providing launch_config (= new agent start) MUST
reset status to idle and clear error_message.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


class TestStatusResetOnRestart:
    """Verify that status resets to idle when launch_config is provided."""

    def test_restart_clears_error_status(self, tmp_path: Path) -> None:
        """Agent restart (new launch_config) must clear error status."""
        # GIVEN: Previous session crashed with error
        metadata = {
            "status": "error",
            "tasks_completed": 5,
            "message_count": 42,
            "error_message": "Agent invocation failed: database is locked",
            "last_activity": "2026-04-18T22:00:00",
            "launch_config": {"provider": "claude-code-sdk", "model": "opus"},
        }
        (tmp_path / "session_metadata.json").write_text(json.dumps(metadata))

        # WHEN: Agent restarts with new launch_config
        sm = SessionManager(
            data_dir=tmp_path,
            launch_config={"provider": "claude-code-sdk", "model": "opus", "started_at": "2026-04-18T23:00:00"},
        )

        # THEN: Status must be idle, error must be cleared
        assert sm.status == AgentStatus.IDLE, (
            f"Status should be IDLE after restart, got {sm.status}"
        )
        assert sm.error_message is None, (
            f"Error message should be None after restart, got {sm.error_message}"
        )

    def test_restart_preserves_counters(self, tmp_path: Path) -> None:
        """Restart resets status but preserves task/message counters."""
        metadata = {
            "status": "error",
            "tasks_completed": 5,
            "message_count": 42,
            "error_message": "some error",
            "last_activity": "2026-04-18T22:00:00",
        }
        (tmp_path / "session_metadata.json").write_text(json.dumps(metadata))

        sm = SessionManager(
            data_dir=tmp_path,
            launch_config={"provider": "claude-code-sdk"},
        )

        assert sm.tasks_completed == 5
        assert sm.message_count == 42

    def test_no_launch_config_preserves_error(self, tmp_path: Path) -> None:
        """Without launch_config (e.g. dwcli connecting), status stays as-is."""
        metadata = {
            "status": "error",
            "tasks_completed": 0,
            "message_count": 0,
            "error_message": "previous error",
            "last_activity": "2026-04-18T22:00:00",
        }
        (tmp_path / "session_metadata.json").write_text(json.dumps(metadata))

        # No launch_config = not a restart, just reading state
        sm = SessionManager(data_dir=tmp_path)

        assert sm.status == AgentStatus.ERROR
        assert sm.error_message == "previous error"

    def test_restart_persists_reset_status(self, tmp_path: Path) -> None:
        """After restart, the persisted file must also show idle status."""
        metadata = {
            "status": "error",
            "tasks_completed": 0,
            "message_count": 0,
            "error_message": "old crash",
            "last_activity": "2026-04-18T22:00:00",
        }
        (tmp_path / "session_metadata.json").write_text(json.dumps(metadata))

        SessionManager(
            data_dir=tmp_path,
            launch_config={"provider": "claude-code-sdk"},
        )

        # Verify persisted file
        saved = json.loads((tmp_path / "session_metadata.json").read_text())
        assert saved["status"] == "idle"
        assert saved["error_message"] is None

    def test_fresh_start_no_metadata_file(self, tmp_path: Path) -> None:
        """First start (no metadata file) should be idle."""
        sm = SessionManager(
            data_dir=tmp_path,
            launch_config={"provider": "claude-code-sdk"},
        )
        assert sm.status == AgentStatus.IDLE
        assert sm.error_message is None
