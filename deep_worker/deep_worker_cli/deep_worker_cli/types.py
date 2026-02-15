"""Type definitions for deep_worker."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AgentStatus(str, Enum):
    """Status of an agent process."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class ClientMode(str, Enum):
    """Mode for client connections."""

    INTERACTIVE = "interactive"  # Can invoke agent
    OBSERVER = "observer"  # Read-only
    CONTROL = "control"  # Can interrupt


class EventType(str, Enum):
    """Types of events from agent."""

    STARTED = "started"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUIRED = "approval_required"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentProcessInfo:
    """Information about a running agent process.

    RPyC is the ONLY communication method - no Unix sockets.
    """

    name: str
    pid: int
    pid_file: Path
    working_dir: str
    started_at: datetime
    rpyc_port: int  # RPyC server port (REQUIRED)
    last_activity: datetime | None = None


@dataclass
class AgentEvent:
    """Event from agent execution."""

    type: EventType
    content: str | None = None
    data: dict[str, Any] | None = None
    timestamp: datetime | None = None
    partial: bool = False


@dataclass
class ApprovalRequest:
    """Request for human approval."""

    approval_id: str
    tool_name: str
    tool_args: dict[str, Any]
    timeout: float = 300.0  # 5 minutes


@dataclass
class ApprovalResult:
    """Result of approval request."""

    approved: bool
    modifications: dict[str, Any] | None = None
    reason: str | None = None
