"""Tests for Pydantic schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.agent_schemas import (
    AgentInfo,
    AgentStatus,
    AgentStatusInfo,
    MessageInfo,
    MessageRequest,
    MessageRole,
    StopTaskManagerRequest,
    WebSocketMessage,
)


class TestAgentStatus:
    """Test AgentStatus enum."""

    def test_all_status_values(self) -> None:
        """Test that all expected status values are defined."""
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.WAITING_INPUT == "waiting_input"
        assert AgentStatus.WAITING_APPROVAL == "waiting_approval"
        assert AgentStatus.INTERRUPTED == "interrupted"
        assert AgentStatus.ERROR == "error"
        assert AgentStatus.SHUTDOWN == "shutdown"


class TestMessageRole:
    """Test MessageRole enum."""

    def test_all_role_values(self) -> None:
        """Test that all expected role values are defined."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL_CALL == "tool_call"
        assert MessageRole.TOOL_RESULT == "tool_result"


class TestAgentInfo:
    """Test AgentInfo schema."""

    def test_valid_agent_info(self) -> None:
        """Test creating valid AgentInfo."""
        now = datetime.now()
        info = AgentInfo(
            name="test-agent",
            pid=12345,
            status=AgentStatus.IDLE,
            provider="claude",
            model="claude-sonnet-4",
            rpyc_port=18861,
            working_dir="/tmp/test",
            started_at=now,
        )

        assert info.name == "test-agent"
        assert info.pid == 12345
        assert info.status == AgentStatus.IDLE
        assert info.provider == "claude"
        assert info.model == "claude-sonnet-4"
        assert info.rpyc_port == 18861
        assert info.working_dir == "/tmp/test"
        assert info.started_at == now

    def test_agent_info_missing_required_fields(self) -> None:
        """Test that AgentInfo requires all fields."""
        with pytest.raises(ValidationError):
            AgentInfo(name="test")  # type: ignore

    def test_agent_info_serialization(self) -> None:
        """Test that AgentInfo can be serialized to JSON."""
        now = datetime.now()
        info = AgentInfo(
            name="test-agent",
            pid=12345,
            status=AgentStatus.IDLE,
            provider="claude",
            model="claude-sonnet-4",
            rpyc_port=18861,
            working_dir="/tmp/test",
            started_at=now,
        )

        json_data = info.model_dump()
        assert json_data["name"] == "test-agent"
        assert json_data["status"] == "idle"


class TestAgentStatusInfo:
    """Test AgentStatusInfo schema."""

    def test_valid_agent_status_info(self) -> None:
        """Test creating valid AgentStatusInfo with optional fields."""
        now = datetime.now()
        info = AgentStatusInfo(
            name="test-agent",
            pid=12345,
            status=AgentStatus.RUNNING,
            provider="claude",
            model="claude-sonnet-4",
            rpyc_port=18861,
            working_dir="/tmp/test",
            started_at=now,
            current_thread_id="thread-123",
            message_count=5,
            last_activity=now,
            task_manager_active=True,
        )

        assert info.current_thread_id == "thread-123"
        assert info.message_count == 5
        assert info.last_activity == now
        assert info.task_manager_active is True

    def test_agent_status_info_default_optional_fields(self) -> None:
        """Test that optional fields have correct defaults."""
        now = datetime.now()
        info = AgentStatusInfo(
            name="test-agent",
            pid=12345,
            status=AgentStatus.IDLE,
            provider="claude",
            model="claude-sonnet-4",
            rpyc_port=18861,
            working_dir="/tmp/test",
            started_at=now,
        )

        assert info.current_thread_id is None
        assert info.message_count == 0
        assert info.last_activity is None
        assert info.task_manager_active is False


class TestMessageInfo:
    """Test MessageInfo schema."""

    def test_valid_message_info(self) -> None:
        """Test creating valid MessageInfo."""
        now = datetime.now()
        msg = MessageInfo(
            role=MessageRole.USER,
            content="Hello",
            timestamp=now,
        )

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp == now

    def test_message_info_with_metadata(self) -> None:
        """Test MessageInfo with optional metadata."""
        now = datetime.now()
        msg = MessageInfo(
            role=MessageRole.TOOL_CALL,
            content="call_tool",
            timestamp=now,
            metadata={"tool": "read_file", "args": {"path": "/test"}},
        )

        assert msg.metadata is not None
        assert msg.metadata["tool"] == "read_file"


class TestMessageRequest:
    """Test MessageRequest schema."""

    def test_valid_message_request(self) -> None:
        """Test creating valid MessageRequest."""
        req = MessageRequest(message="Hello, agent!")
        assert req.message == "Hello, agent!"

    def test_message_request_empty_string(self) -> None:
        """Test that empty message is allowed (validation happens at API level)."""
        req = MessageRequest(message="")
        assert req.message == ""


class TestStopTaskManagerRequest:
    """Test StopTaskManagerRequest schema."""

    def test_valid_stop_request_default(self) -> None:
        """Test creating StopTaskManagerRequest with default force value."""
        req = StopTaskManagerRequest()
        assert req.force is False

    def test_valid_stop_request_with_force(self) -> None:
        """Test creating StopTaskManagerRequest with force=True."""
        req = StopTaskManagerRequest(force=True)
        assert req.force is True


class TestWebSocketMessage:
    """Test WebSocketMessage schema."""

    def test_valid_websocket_message_agent_message(self) -> None:
        """Test WebSocket message for agent message event."""
        now = datetime.now()
        msg = WebSocketMessage(
            type="agent_message",
            agent_name="test-agent",
            message=MessageInfo(
                role=MessageRole.ASSISTANT,
                content="Hello!",
                timestamp=now,
            ),
        )

        assert msg.type == "agent_message"
        assert msg.agent_name == "test-agent"
        assert msg.message is not None
        assert msg.message.content == "Hello!"

    def test_valid_websocket_message_agent_status(self) -> None:
        """Test WebSocket message for agent status event."""
        msg = WebSocketMessage(
            type="agent_status",
            agent_name="test-agent",
            status=AgentStatus.RUNNING,
            details="Processing request",
        )

        assert msg.type == "agent_status"
        assert msg.status == AgentStatus.RUNNING
        assert msg.details == "Processing request"

    def test_valid_websocket_message_agent_discovered(self) -> None:
        """Test WebSocket message for agent discovered event."""
        now = datetime.now()
        agent_info = AgentInfo(
            name="new-agent",
            pid=54321,
            status=AgentStatus.IDLE,
            provider="deepseek",
            model="deepseek-coder",
            rpyc_port=18862,
            working_dir="/tmp/new",
            started_at=now,
        )

        msg = WebSocketMessage(
            type="agent_discovered",
            agent=agent_info,
        )

        assert msg.type == "agent_discovered"
        assert msg.agent is not None
        assert msg.agent.name == "new-agent"

    def test_valid_websocket_message_agent_terminated(self) -> None:
        """Test WebSocket message for agent terminated event."""
        msg = WebSocketMessage(
            type="agent_terminated",
            agent_name="old-agent",
        )

        assert msg.type == "agent_terminated"
        assert msg.agent_name == "old-agent"

    def test_valid_websocket_message_pong(self) -> None:
        """Test WebSocket message for pong response."""
        msg = WebSocketMessage(type="pong")
        assert msg.type == "pong"

    def test_websocket_message_serialization(self) -> None:
        """Test that WebSocketMessage can be serialized to JSON."""
        now = datetime.now()
        msg = WebSocketMessage(
            type="agent_status",
            agent_name="test-agent",
            status=AgentStatus.IDLE,
        )

        json_data = msg.model_dump(exclude_none=True)
        assert json_data["type"] == "agent_status"
        assert json_data["agent_name"] == "test-agent"
        assert json_data["status"] == "idle"
        # Fields that are None should be excluded
        assert "message" not in json_data
        assert "agent" not in json_data
