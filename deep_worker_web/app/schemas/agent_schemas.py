"""Pydantic schemas for deep-worker agents.

This module defines all data models used for API requests/responses and
internal data representation following the architecture design.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Status of a deep-worker agent.
    
    These statuses represent the current state of an agent process:
    - IDLE: Agent is waiting for user input
    - RUNNING: Agent is actively processing a task
    - WAITING_INPUT: Agent is waiting for user input (interactive prompt)
    - WAITING_APPROVAL: Agent is waiting for user approval (e.g., tool execution)
    - ERROR: Agent encountered an error
    - SHUTDOWN: Agent is shutting down or has shut down
    """

    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    WAITING_APPROVAL = "waiting_approval"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessageRole(str, Enum):
    """Role of a message in agent conversation.
    
    Follows the standard conversational AI message format:
    - USER: Message from the user
    - ASSISTANT: Message from the AI assistant
    - TOOL_CALL: Agent requesting to execute a tool
    - TOOL_RESULT: Result from a tool execution
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentInfo(BaseModel):
    """Basic information about a deep-worker agent.
    
    This represents the core metadata about an agent process discovered
    from the ProcessRegistry or queried via RPyC.
    """

    name: str = Field(..., description="Agent name (typically project name)")
    pid: int = Field(..., description="Process ID of the agent")
    status: AgentStatus = Field(..., description="Current status of the agent")
    provider: str = Field(..., description="AI provider (e.g., 'claude', 'deepseek')")
    model: str = Field(..., description="Model name (e.g., 'claude-sonnet-4')")
    rpyc_port: int = Field(..., description="RPyC server port for this agent")
    working_dir: str = Field(..., description="Working directory of the agent")
    started_at: datetime = Field(..., description="Timestamp when agent was started")


class AgentStatusInfo(BaseModel):
    """Detailed status information for a deep-worker agent.
    
    Extends AgentInfo with additional runtime information about the agent's
    current conversation state and activity.
    """

    # Core info (same as AgentInfo)
    name: str
    pid: int
    status: AgentStatus
    provider: str
    model: str
    rpyc_port: int
    working_dir: str
    started_at: datetime

    # Extended status information
    current_thread_id: str | None = Field(
        None, description="Current conversation thread ID"
    )
    message_count: int = Field(
        0, description="Total messages in current thread"
    )
    last_activity: datetime | None = Field(
        None, description="Timestamp of last activity"
    )
    task_manager_active: bool = Field(
        False, description="Whether TaskManager is currently running"
    )


class MessageInfo(BaseModel):
    """Information about a single message in agent conversation.
    
    Represents a message exchanged between user and agent, including
    tool calls and results.
    """

    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(..., description="Timestamp of the message")
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata (e.g., tool parameters)"
    )


class MessageRequest(BaseModel):
    """Request to send a message to an agent.
    
    Used by the API endpoint to accept user messages for delivery to agents.
    """

    message: str = Field(..., description="Message content to send to the agent")


class StopTaskManagerRequest(BaseModel):
    """Request to stop an agent's TaskManager.
    
    The TaskManager is the internal component that handles background task
    processing within a deep-worker agent.
    """

    force: bool = Field(
        False,
        description="Whether to force stop (terminate immediately) or graceful shutdown",
    )


class RestartRequest(BaseModel):
    """Request to restart an agent with optional parameter overrides."""

    model: str | None = Field(None, description="Override model name")
    provider: str | None = Field(None, description="Override provider")
    clean: bool = Field(False, description="Clean history before restart")


class BatchRequest(BaseModel):
    """Request for batch operations on multiple agents."""

    action: str = Field(..., description="Action: 'stop' or 'restart'")
    agent_names: list[str] = Field(..., description="List of agent names")


class WebSocketMessage(BaseModel):
    """WebSocket message sent from server to client.
    
    This is the unified message format for all WebSocket events. Different
    event types use different subsets of fields.
    
    Event types:
    - agent_message: New message from agent (uses agent_name, message)
    - agent_status: Agent status change (uses agent_name, status, details)
    - agent_discovered: New agent detected (uses agent)
    - agent_terminated: Agent shut down (uses agent_name)
    - pong: Response to ping (no extra fields)
    """

    type: str = Field(..., description="Event type")
    
    # For agent_message events
    agent_name: str | None = Field(None, description="Name of the agent")
    message: MessageInfo | None = Field(None, description="Message information")
    
    # For agent_status events
    status: AgentStatus | None = Field(None, description="New status")
    details: str | None = Field(None, description="Additional status details")
    
    # For agent_discovered events
    agent: AgentInfo | None = Field(None, description="Newly discovered agent info")
