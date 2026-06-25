"""Type definitions for deep_worker."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AgentStatus(str, Enum):
    """Status of an agent process.

    Values:
        IDLE          — Turn finished cleanly, agent ready for the next
                        user message.
        RUNNING       — Agent actively processing a turn (LLM thinking
                        or LangGraph routing between nodes).
        WAITING_INPUT — Inside a running turn, currently awaiting a
                        tool's output. Despite the name, this is NOT
                        "waiting for the user" — it is "waiting for an
                        in-flight tool to return". (Set by
                        ``StatusTrackingCallback.on_tool_start``.)
        INTERRUPTED   — User cancelled the current turn via the
                        Interrupt control. Agent is alive and waiting
                        for the user's next message — typically a
                        correction. Distinct from WAITING_INPUT so the
                        UI can render it in a "needs user attention"
                        colour. (Set by ``AgentProcess._handle_invoke``
                        in the ``except asyncio.CancelledError`` branch.)
        ERROR         — Agent invocation failed; ``error_message`` set.
        SHUTDOWN      — Agent process is shutting down or has shut down.
    """

    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    INTERRUPTED = "interrupted"
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
    # Optional A2A (Agent2Agent) protocol endpoint, populated only when the
    # agent runs its A2A server (deepworker_a2a). Backward-compatible: absent
    # for agents without A2A enabled.
    a2a_port: int | None = None
    a2a_url: str | None = None


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
