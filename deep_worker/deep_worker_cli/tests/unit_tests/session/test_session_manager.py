"""Tests for SessionManager."""

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
def session_manager(temp_data_dir):
    """Create a SessionManager instance."""
    return SessionManager(data_dir=temp_data_dir)


class TestSessionManagerInitialization:
    """Tests for SessionManager initialization."""

    def test_create_session_manager(self, temp_data_dir):
        """Test creating a SessionManager."""
        manager = SessionManager(data_dir=temp_data_dir)

        assert manager.data_dir == temp_data_dir
        assert manager.status == AgentStatus.IDLE
        assert manager.tasks_completed == 0
        assert manager.error_message is None

    def test_metadata_file_created(self, session_manager):
        """Test that metadata file is created on initialization."""
        metadata_file = session_manager.data_dir / "session_metadata.json"
        assert metadata_file.exists()

    def test_load_existing_metadata(self, temp_data_dir):
        """Test loading existing metadata from file."""
        # Create metadata file
        metadata_file = temp_data_dir / "session_metadata.json"
        metadata = {
            "status": "running",
            "tasks_completed": 5,
            "error_message": "Test error",
            "last_activity": datetime.now().isoformat(),
        }
        metadata_file.write_text(json.dumps(metadata))

        # Load manager
        manager = SessionManager(data_dir=temp_data_dir)

        assert manager.status == AgentStatus.RUNNING
        assert manager.tasks_completed == 5
        assert manager.error_message == "Test error"


class TestSessionManagerStatusManagement:
    """Tests for session status management."""

    def test_set_status(self, session_manager):
        """Test setting session status."""
        session_manager.set_status(AgentStatus.RUNNING)
        assert session_manager.status == AgentStatus.RUNNING

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["status"] == "running"

    def test_status_transitions(self, session_manager):
        """Test various status transitions."""
        # IDLE -> RUNNING
        session_manager.set_status(AgentStatus.RUNNING)
        assert session_manager.status == AgentStatus.RUNNING

        # RUNNING -> WAITING_APPROVAL
        session_manager.set_status(AgentStatus.WAITING_APPROVAL)
        assert session_manager.status == AgentStatus.WAITING_APPROVAL

        # WAITING_APPROVAL -> RUNNING
        session_manager.set_status(AgentStatus.RUNNING)
        assert session_manager.status == AgentStatus.RUNNING

        # RUNNING -> IDLE
        session_manager.set_status(AgentStatus.IDLE)
        assert session_manager.status == AgentStatus.IDLE

    def test_set_error_status(self, session_manager):
        """Test setting error status with message."""
        session_manager.set_error("Something went wrong")

        assert session_manager.status == AgentStatus.ERROR
        assert session_manager.error_message == "Something went wrong"

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["status"] == "error"
        assert data["error_message"] == "Something went wrong"

    def test_clear_error(self, session_manager):
        """Test clearing error status."""
        # Set error
        session_manager.set_error("Test error")
        assert session_manager.error_message is not None

        # Clear error
        session_manager.set_status(AgentStatus.IDLE)
        session_manager.clear_error()

        assert session_manager.error_message is None

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["error_message"] is None


class TestSessionManagerTaskTracking:
    """Tests for task completion tracking."""

    def test_initial_tasks_completed(self, session_manager):
        """Test initial tasks completed count."""
        assert session_manager.tasks_completed == 0

    def test_increment_tasks_completed(self, session_manager):
        """Test incrementing tasks completed counter."""
        session_manager.increment_tasks_completed()
        assert session_manager.tasks_completed == 1

        session_manager.increment_tasks_completed()
        assert session_manager.tasks_completed == 2

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["tasks_completed"] == 2

    def test_multiple_increments(self, session_manager):
        """Test multiple task completions."""
        for _ in range(10):
            session_manager.increment_tasks_completed()

        assert session_manager.tasks_completed == 10


class TestSessionManagerPersistence:
    """Tests for metadata persistence."""

    def test_persist_on_status_change(self, session_manager):
        """Test that metadata is persisted on status change."""
        session_manager.set_status(AgentStatus.RUNNING)

        # Load metadata file
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert data["status"] == "running"

    def test_persist_on_task_completion(self, session_manager):
        """Test that metadata is persisted on task completion."""
        session_manager.increment_tasks_completed()

        # Load metadata file
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert data["tasks_completed"] == 1

    def test_last_activity_updated(self, session_manager):
        """Test that last_activity timestamp is updated."""
        before = datetime.now()
        session_manager.set_status(AgentStatus.RUNNING)
        after = datetime.now()

        # Load metadata
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        last_activity = datetime.fromisoformat(data["last_activity"])
        assert before <= last_activity <= after

    def test_metadata_survives_reload(self, temp_data_dir):
        """Test that metadata persists across manager instances."""
        # Create first manager
        manager1 = SessionManager(data_dir=temp_data_dir)
        manager1.set_status(AgentStatus.RUNNING)
        manager1.increment_tasks_completed()
        manager1.increment_tasks_completed()

        # Create second manager (should load existing metadata)
        manager2 = SessionManager(data_dir=temp_data_dir)

        assert manager2.status == AgentStatus.RUNNING
        assert manager2.tasks_completed == 2


class TestSessionManagerMessageCount:
    """Tests for message counting."""

    def test_initial_message_count(self, session_manager):
        """Test initial message count."""
        assert session_manager.message_count == 0

    def test_increment_message_count(self, session_manager):
        """Test incrementing message count."""
        session_manager.increment_message_count()
        assert session_manager.message_count == 1

        session_manager.increment_message_count()
        assert session_manager.message_count == 2

    def test_message_count_persisted(self, session_manager):
        """Test that message count is persisted."""
        session_manager.increment_message_count()
        session_manager.increment_message_count()
        session_manager.increment_message_count()

        # Load metadata
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert data["message_count"] == 3
