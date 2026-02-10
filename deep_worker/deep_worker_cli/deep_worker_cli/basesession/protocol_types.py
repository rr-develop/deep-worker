"""BaseSession protocol types for cco daemon compatibility.

These types define the unified session protocol compatible with cco daemon.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


# Session type identifier
SessionTypeId = Literal["cco", "csdk", "persistent_csdk", "langchain"]


class SessionStatus(str, Enum):
    """Session status enumeration compatible with BaseSession protocol."""

    READY = "ready"
    WORKING = "working"
    WAITING_INPUT = "waiting_input"
    ERROR = "error"
    UNKNOWN = "unknown"


class SessionData(BaseModel):
    """Additional session data.

    Contains extended information about the session that doesn't fit
    into the main SessionInfo fields.
    """

    message_count: int
    process_id: int
    socket_path: str

    model_config = {"extra": "allow"}  # Allow additional fields


class SessionMetadata(BaseModel):
    """Session metadata for extensibility.

    Contains environment variables and custom properties.
    """

    environment: dict[str, str]
    properties: dict[str, Any]

    model_config = {"extra": "allow"}  # Allow additional fields


class SessionInfo(BaseModel):
    """Unified session information compatible with BaseSession protocol.

    This model represents the state of an agent session in a format
    compatible with cco daemon and other BaseSession-aware tools.
    """

    name: str
    session_type: SessionTypeId
    status: SessionStatus
    working_dir: str
    created_at: datetime
    last_activity: datetime | None
    tasks_completed: int
    error_message: str | None
    data: SessionData | None
    metadata: SessionMetadata | None

    model_config = {"extra": "allow"}  # Allow additional fields
