"""Tests for BaseSession protocol types."""

from datetime import datetime

import pytest

from deep_worker_cli.basesession.protocol_types import (
    SessionData,
    SessionInfo,
    SessionMetadata,
    SessionStatus,
    SessionTypeId,
)


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all required statuses are defined."""
        assert SessionStatus.READY.value == "ready"
        assert SessionStatus.WORKING.value == "working"
        assert SessionStatus.WAITING_INPUT.value == "waiting_input"
        assert SessionStatus.ERROR.value == "error"
        assert SessionStatus.UNKNOWN.value == "unknown"

    def test_status_can_be_compared(self):
        """Test that statuses can be compared."""
        assert SessionStatus.READY == SessionStatus.READY
        assert SessionStatus.READY != SessionStatus.WORKING


class TestSessionTypeId:
    """Tests for SessionTypeId type."""

    def test_valid_session_types(self):
        """Test that all valid session types are accepted."""
        # These should be valid literal values
        valid_types: list[SessionTypeId] = [
            "cco",
            "csdk",
            "persistent_csdk",
            "langchain",
        ]
        assert len(valid_types) == 4


class TestSessionData:
    """Tests for SessionData model."""

    def test_create_minimal(self):
        """Test creating SessionData with minimal fields."""
        data = SessionData(
            message_count=10,
            process_id=12345,
        )

        assert data.message_count == 10
        assert data.process_id == 12345

    def test_create_with_optional_fields(self):
        """Test creating SessionData with optional fields."""
        data = SessionData(
            message_count=5,
            process_id=99999,
            custom_field="test",  # type: ignore[call-arg]
        )

        assert data.message_count == 5
        assert hasattr(data, "custom_field")


class TestSessionMetadata:
    """Tests for SessionMetadata model."""

    def test_create_with_environment(self):
        """Test creating SessionMetadata with environment."""
        metadata = SessionMetadata(
            environment={"PATH": "/usr/bin", "HOME": "/home/user"},
            properties={"key": "value"},
        )

        assert metadata.environment["PATH"] == "/usr/bin"
        assert metadata.properties["key"] == "value"

    def test_create_empty(self):
        """Test creating empty SessionMetadata."""
        metadata = SessionMetadata(
            environment={},
            properties={},
        )

        assert metadata.environment == {}
        assert metadata.properties == {}


class TestSessionInfo:
    """Tests for SessionInfo model."""

    def test_create_minimal_session_info(self):
        """Test creating SessionInfo with required fields only."""
        now = datetime.now()
        info = SessionInfo(
            name="test-agent",
            session_type="langchain",
            status=SessionStatus.READY,
            working_dir="/tmp/work",
            created_at=now,
            last_activity=None,
            tasks_completed=0,
            error_message=None,
            data=None,
            metadata=None,
        )

        assert info.name == "test-agent"
        assert info.session_type == "langchain"
        assert info.status == SessionStatus.READY
        assert info.working_dir == "/tmp/work"
        assert info.created_at == now
        assert info.tasks_completed == 0

    def test_create_full_session_info(self):
        """Test creating SessionInfo with all fields."""
        now = datetime.now()
        data = SessionData(
            message_count=42,
            process_id=12345,
        )
        metadata = SessionMetadata(
            environment={"PATH": "/usr/bin"},
            properties={"model": "claude-sonnet-4-5-20250929"},
        )

        info = SessionInfo(
            name="full-agent",
            session_type="langchain",
            status=SessionStatus.WORKING,
            working_dir="/home/user/project",
            created_at=now,
            last_activity=now,
            tasks_completed=10,
            error_message=None,
            data=data,
            metadata=metadata,
        )

        assert info.name == "full-agent"
        assert info.status == SessionStatus.WORKING
        assert info.tasks_completed == 10
        assert info.data is not None
        assert info.data.message_count == 42
        assert info.metadata is not None
        assert info.metadata.properties["model"] == "claude-sonnet-4-5-20250929"

    def test_session_info_with_error(self):
        """Test SessionInfo with error status and message."""
        now = datetime.now()
        info = SessionInfo(
            name="error-agent",
            session_type="langchain",
            status=SessionStatus.ERROR,
            working_dir="/tmp/work",
            created_at=now,
            last_activity=now,
            tasks_completed=5,
            error_message="Connection failed",
            data=None,
            metadata=None,
        )

        assert info.status == SessionStatus.ERROR
        assert info.error_message == "Connection failed"
