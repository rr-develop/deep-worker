"""Tests for SessionManager launch_config persistence.

TDD tests: launch_config should be saved to session_metadata.json at startup
and preserved across status updates and manager reloads.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for session data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_launch_config():
    """Sample launch_config dict matching real CLI args."""
    return {
        "provider": "claude-code-sdk",
        "model": "opus",
        "working_dir": "/path/to/project",
        "skip_permissions": True,
        "task_manager": True,
        "memory_only": False,
        "use_sqlite": False,
        "clean": True,
        "started_at": "2026-04-11T16:59:41.692129",
        "pid": 3726921,
        "cli_args": "agent_1 -sp -tm --dir /path/to/project --provider claude-code-sdk --model opus --clean",
    }


class TestLaunchConfigSavedOnInit:
    """launch_config is written to disk when SessionManager is created."""

    def test_launch_config_saved_to_metadata_file(
        self, temp_data_dir, sample_launch_config
    ):
        """launch_config appears in session_metadata.json after init."""
        manager = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert "launch_config" in data
        assert data["launch_config"]["provider"] == "claude-code-sdk"
        assert data["launch_config"]["model"] == "opus"
        assert data["launch_config"]["working_dir"] == "/path/to/project"
        assert data["launch_config"]["skip_permissions"] is True
        assert data["launch_config"]["task_manager"] is True
        assert data["launch_config"]["pid"] == 3726921

    def test_launch_config_stored_as_attribute(
        self, temp_data_dir, sample_launch_config
    ):
        """launch_config is accessible as manager.launch_config."""
        manager = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        assert manager.launch_config is not None
        assert manager.launch_config["provider"] == "claude-code-sdk"
        assert manager.launch_config["model"] == "opus"

    def test_no_launch_config_is_ok(self, temp_data_dir):
        """SessionManager works without launch_config (backward compat)."""
        manager = SessionManager(data_dir=temp_data_dir)

        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        # launch_config should not appear if not provided
        assert "launch_config" not in data
        assert manager.launch_config is None

    def test_cli_args_saved(self, temp_data_dir, sample_launch_config):
        """Raw CLI args string is preserved."""
        manager = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert "cli_args" in data["launch_config"]
        assert "--provider claude-code-sdk" in data["launch_config"]["cli_args"]


class TestLaunchConfigPreservedAcrossStatusUpdates:
    """launch_config must NOT be overwritten when status changes."""

    @pytest.mark.asyncio
    async def test_set_status_preserves_launch_config(
        self, temp_data_dir, sample_launch_config
    ):
        """Setting status should not remove launch_config."""
        manager = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        # Change status multiple times
        await manager.set_status(AgentStatus.RUNNING)
        await manager.set_status(AgentStatus.WAITING_INPUT)
        await manager.set_status(AgentStatus.IDLE)

        # Verify launch_config is still there
        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert "launch_config" in data
        assert data["launch_config"]["provider"] == "claude-code-sdk"
        assert data["launch_config"]["model"] == "opus"

    @pytest.mark.asyncio
    async def test_set_error_preserves_launch_config(
        self, temp_data_dir, sample_launch_config
    ):
        """Setting error should not remove launch_config."""
        manager = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        await manager.set_error("Something went wrong")

        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert "launch_config" in data
        assert data["launch_config"]["provider"] == "claude-code-sdk"

    @pytest.mark.asyncio
    async def test_increment_tasks_preserves_launch_config(
        self, temp_data_dir, sample_launch_config
    ):
        """Incrementing task counter should not remove launch_config."""
        manager = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        await manager.increment_tasks_completed()
        await manager.increment_message_count()

        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert "launch_config" in data
        assert data["tasks_completed"] == 1
        assert data["message_count"] == 1


class TestLaunchConfigModelUpdate:
    @pytest.mark.asyncio
    async def test_update_launch_config_model_persists_model(self, temp_data_dir):
        manager = SessionManager(
            data_dir=temp_data_dir,
            launch_config={"provider": "codex", "model": None},
        )

        await manager.update_launch_config_model("gpt-dynamic-default")

        data = json.loads((temp_data_dir / "session_metadata.json").read_text())
        assert data["launch_config"]["model"] == "gpt-dynamic-default"
        assert manager.launch_config["model"] == "gpt-dynamic-default"

    @pytest.mark.asyncio
    async def test_update_launch_config_model_rejects_unknown(self, temp_data_dir):
        manager = SessionManager(
            data_dir=temp_data_dir,
            launch_config={"provider": "codex", "model": None},
        )

        await manager.update_launch_config_model("unknown")

        data = json.loads((temp_data_dir / "session_metadata.json").read_text())
        assert data["launch_config"]["model"] is None

    @pytest.mark.asyncio
    async def test_update_launch_config_model_notifies(self, temp_data_dir):
        events = []

        async def notify(payload):
            events.append(payload)

        manager = SessionManager(
            data_dir=temp_data_dir,
            launch_config={"provider": "codex", "model": None},
            notifier=notify,
        )

        await manager.update_launch_config_model("gpt-dynamic-default")

        assert events == [
            {
                "type": "status_change",
                "status": "idle",
                "error_message": None,
                "provider": "codex",
                "model": "gpt-dynamic-default",
            }
        ]


class TestLaunchConfigSurvivesReload:
    """launch_config must persist when a new SessionManager loads from disk."""

    def test_reload_preserves_launch_config(
        self, temp_data_dir, sample_launch_config
    ):
        """New SessionManager instance loads launch_config from disk."""
        # First instance writes launch_config
        manager1 = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        # Second instance should load it
        manager2 = SessionManager(data_dir=temp_data_dir)

        assert manager2.launch_config is not None
        assert manager2.launch_config["provider"] == "claude-code-sdk"
        assert manager2.launch_config["model"] == "opus"
        assert manager2.launch_config["working_dir"] == "/path/to/project"
        assert manager2.launch_config["skip_permissions"] is True

    def test_new_launch_config_overwrites_old(
        self, temp_data_dir, sample_launch_config
    ):
        """When agent restarts with new config, old launch_config is replaced."""
        # First launch
        manager1 = SessionManager(
            data_dir=temp_data_dir, launch_config=sample_launch_config
        )

        # Second launch with different params
        new_config = {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "working_dir": "/path/to/project",
            "skip_permissions": False,
            "task_manager": False,
            "memory_only": True,
            "use_sqlite": False,
            "clean": False,
            "started_at": "2026-04-12T10:00:00.000000",
            "pid": 9999,
            "cli_args": "vc_test --provider deepseek --memory-only",
        }
        manager2 = SessionManager(
            data_dir=temp_data_dir, launch_config=new_config
        )

        assert manager2.launch_config["provider"] == "deepseek"
        assert manager2.launch_config["model"] == "deepseek-chat"
        assert manager2.launch_config["pid"] == 9999

        # Verify on disk too
        metadata_file = temp_data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["launch_config"]["provider"] == "deepseek"


class TestLaunchConfigBackwardCompatibility:
    """Old metadata files without launch_config should still work."""

    def test_load_old_metadata_without_launch_config(self, temp_data_dir):
        """Loading a pre-existing metadata file that has no launch_config."""
        # Write old-format metadata
        metadata_file = temp_data_dir / "session_metadata.json"
        old_metadata = {
            "status": "running",
            "tasks_completed": 5,
            "message_count": 10,
            "error_message": None,
            "last_activity": datetime.now().isoformat(),
        }
        metadata_file.write_text(json.dumps(old_metadata))

        # Load — should not crash
        manager = SessionManager(data_dir=temp_data_dir)

        assert manager.status == AgentStatus.RUNNING
        assert manager.tasks_completed == 5
        assert manager.message_count == 10
        assert manager.launch_config is None

    @pytest.mark.asyncio
    async def test_status_update_on_old_metadata_no_launch_config(
        self, temp_data_dir
    ):
        """Status updates on old metadata should not inject launch_config."""
        metadata_file = temp_data_dir / "session_metadata.json"
        old_metadata = {
            "status": "idle",
            "tasks_completed": 0,
            "message_count": 0,
            "error_message": None,
            "last_activity": datetime.now().isoformat(),
        }
        metadata_file.write_text(json.dumps(old_metadata))

        manager = SessionManager(data_dir=temp_data_dir)
        await manager.set_status(AgentStatus.RUNNING)

        data = json.loads(metadata_file.read_text())
        assert "launch_config" not in data
