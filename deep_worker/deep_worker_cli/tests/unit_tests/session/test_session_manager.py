"""Tests for SessionManager."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

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


def test_launch_config_storage_version_is_used_for_new_metadata(temp_data_dir):
    manager = SessionManager(
        data_dir=temp_data_dir,
        launch_config={"storage_version": 2},
    )

    metadata_file = temp_data_dir / "session_metadata.json"
    data = json.loads(metadata_file.read_text())
    assert manager.storage_version == 2
    assert data["storage_version"] == 2


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

    @pytest.mark.asyncio
    async def test_set_status(self, session_manager):
        """Test setting session status."""
        await session_manager.set_status(AgentStatus.RUNNING)
        assert session_manager.status == AgentStatus.RUNNING

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_status_transitions(self, session_manager):
        """Test various status transitions."""
        # IDLE -> RUNNING
        await session_manager.set_status(AgentStatus.RUNNING)
        assert session_manager.status == AgentStatus.RUNNING

        # RUNNING -> WAITING_INPUT
        await session_manager.set_status(AgentStatus.WAITING_INPUT)
        assert session_manager.status == AgentStatus.WAITING_INPUT

        # WAITING_INPUT -> RUNNING
        await session_manager.set_status(AgentStatus.RUNNING)
        assert session_manager.status == AgentStatus.RUNNING

        # RUNNING -> IDLE
        await session_manager.set_status(AgentStatus.IDLE)
        assert session_manager.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_set_error_status(self, session_manager):
        """Test setting error status with message."""
        await session_manager.set_error("Something went wrong")

        assert session_manager.status == AgentStatus.ERROR
        assert session_manager.error_message == "Something went wrong"

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["status"] == "error"
        assert data["error_message"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_clear_error(self, session_manager):
        """Test clearing error status."""
        # Set error
        await session_manager.set_error("Test error")
        assert session_manager.error_message is not None

        # Clear error
        await session_manager.set_status(AgentStatus.IDLE)
        await session_manager.clear_error()

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

    @pytest.mark.asyncio
    async def test_increment_tasks_completed(self, session_manager):
        """Test incrementing tasks completed counter."""
        await session_manager.increment_tasks_completed()
        assert session_manager.tasks_completed == 1

        await session_manager.increment_tasks_completed()
        assert session_manager.tasks_completed == 2

        # Verify persisted
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["tasks_completed"] == 2

    @pytest.mark.asyncio
    async def test_multiple_increments(self, session_manager):
        """Test multiple task completions."""
        for _ in range(10):
            await session_manager.increment_tasks_completed()

        assert session_manager.tasks_completed == 10


class TestSessionManagerPersistence:
    """Tests for metadata persistence."""

    @pytest.mark.asyncio
    async def test_persist_on_status_change(self, session_manager):
        """Test that metadata is persisted on status change."""
        await session_manager.set_status(AgentStatus.RUNNING)

        # Load metadata file
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_persist_on_task_completion(self, session_manager):
        """Test that metadata is persisted on task completion."""
        await session_manager.increment_tasks_completed()

        # Load metadata file
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert data["tasks_completed"] == 1

    @pytest.mark.asyncio
    async def test_last_activity_updated(self, session_manager):
        """Test that last_activity timestamp is updated."""
        before = datetime.now()
        await session_manager.set_status(AgentStatus.RUNNING)
        after = datetime.now()

        # Load metadata
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        last_activity = datetime.fromisoformat(data["last_activity"])
        assert before <= last_activity <= after

    @pytest.mark.asyncio
    async def test_metadata_survives_reload(self, temp_data_dir):
        """Test that metadata persists across manager instances."""
        # Create first manager
        manager1 = SessionManager(data_dir=temp_data_dir)
        await manager1.set_status(AgentStatus.RUNNING)
        await manager1.increment_tasks_completed()
        await manager1.increment_tasks_completed()

        # Create second manager (should load existing metadata)
        manager2 = SessionManager(data_dir=temp_data_dir)

        assert manager2.status == AgentStatus.RUNNING
        assert manager2.tasks_completed == 2


class TestSessionManagerMessageCount:
    """Tests for message counting."""

    def test_initial_message_count(self, session_manager):
        """Test initial message count."""
        assert session_manager.message_count == 0

    @pytest.mark.asyncio
    async def test_increment_message_count(self, session_manager):
        """Test incrementing message count."""
        await session_manager.increment_message_count()
        assert session_manager.message_count == 1

        await session_manager.increment_message_count()
        assert session_manager.message_count == 2

    @pytest.mark.asyncio
    async def test_message_count_persisted(self, session_manager):
        """Test that message count is persisted."""
        await session_manager.increment_message_count()
        await session_manager.increment_message_count()
        await session_manager.increment_message_count()

        # Load metadata
        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())

        assert data["message_count"] == 3


class TestRecoverableErrorVisibility:
    """A failure that settles in IDLE must still be visible.

    POLICIES.md §1 requires an exceeded iteration cap (and friends) to
    land in IDLE, not ERROR — the user can simply type "continue". But
    the web UI reads the status, not the error log, so an IDLE carrying
    no ``error_message`` is indistinguishable from a clean turn. That is
    the defect these tests pin.

    See docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §4.
    """

    @pytest.mark.asyncio
    async def test_set_status_without_the_kwarg_leaves_error_message_alone(
        self, session_manager
    ):
        """The default must be a no-op, or every existing caller changes."""
        session_manager.error_message = "previous failure"

        await session_manager.set_status(AgentStatus.RUNNING)

        assert session_manager.error_message == "previous failure"

    @pytest.mark.asyncio
    async def test_recoverable_failure_is_recorded_without_the_error_status(
        self, session_manager
    ):
        await session_manager.set_status(
            AgentStatus.IDLE,
            error_message="MaxIterationsExceeded: hit the cap",
        )

        assert session_manager.status == AgentStatus.IDLE, (
            "a recoverable failure must not flip the agent into ERROR"
        )
        assert session_manager.error_message == "MaxIterationsExceeded: hit the cap"

        metadata_file = session_manager.data_dir / "session_metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["status"] == "idle"
        assert data["error_message"] == "MaxIterationsExceeded: hit the cap"

    @pytest.mark.asyncio
    async def test_the_marker_is_cleared_when_a_new_turn_starts(
        self, session_manager
    ):
        """Otherwise one cap hit shows an error in the UI forever.

        ``clear_error`` had no production callers at all, so an earlier
        draft of this feature would have made an invisible failure into
        an unclearable one.
        """
        await session_manager.set_status(
            AgentStatus.IDLE, error_message="MaxIterationsExceeded: hit the cap"
        )

        # What a starting turn does (agent_process): clear, then RUNNING.
        await session_manager.clear_error()
        await session_manager.set_status(AgentStatus.RUNNING)

        assert session_manager.error_message is None

    @pytest.mark.asyncio
    async def test_set_status_alone_does_not_clear_the_marker(
        self, session_manager
    ):
        """``set_status`` must stay non-mutating for the untouched field.

        This pins the property. Whether the *wiring* respects it is a
        different question — see the status_callback tests below, which
        drive the real callback.
        """
        await session_manager.set_status(
            AgentStatus.IDLE, error_message="MaxIterationsExceeded: hit the cap"
        )

        await session_manager.set_status(AgentStatus.RUNNING)

        assert session_manager.error_message == "MaxIterationsExceeded: hit the cap"

    @pytest.mark.asyncio
    async def test_a_finished_tool_does_not_clear_the_marker(
        self, session_manager
    ):
        """The reset belongs to a STARTING turn, not to every RUNNING.

        ``StatusTrackingCallback`` returns the agent to RUNNING after each
        tool finishes — inside the same turn. Clearing there would erase
        the marker the failing turn just produced, restoring the defect in
        the shape of its own fix.

        This drives the REAL callback rather than calling ``set_status``
        by hand: the wrong wiring lives in ``status_callback``, so a test
        that never runs that module cannot observe it. Verified by
        mutation — adding ``clear_error()`` beside either
        ``set_status(RUNNING)`` there fails this test.
        """
        from deep_worker_cli.process.status_callback import StatusTrackingCallback

        await session_manager.set_status(
            AgentStatus.IDLE, error_message="MaxIterationsExceeded: hit the cap"
        )
        callback = StatusTrackingCallback(session_manager)

        await callback.on_tool_end("some tool output")

        assert session_manager.status == AgentStatus.RUNNING
        assert session_manager.error_message == "MaxIterationsExceeded: hit the cap"

    @pytest.mark.asyncio
    async def test_a_recoverable_tool_error_does_not_clear_the_marker(
        self, session_manager
    ):
        """The other mid-turn return to RUNNING, same reasoning."""
        from deep_worker_cli.process.status_callback import StatusTrackingCallback

        await session_manager.set_status(
            AgentStatus.IDLE, error_message="MaxIterationsExceeded: hit the cap"
        )
        callback = StatusTrackingCallback(session_manager)

        await callback.on_tool_error(ValueError("bad argument"), run_id=uuid4())

        assert session_manager.status == AgentStatus.RUNNING
        assert session_manager.error_message == "MaxIterationsExceeded: hit the cap"
