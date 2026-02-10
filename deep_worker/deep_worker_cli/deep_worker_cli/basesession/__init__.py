"""BaseSession protocol integration for cco daemon compatibility."""

from deep_worker_cli.basesession.detector import DeepWorkerDetector
from deep_worker_cli.basesession.protocol_types import (
    SessionData,
    SessionInfo,
    SessionMetadata,
    SessionStatus,
    SessionTypeId,
)
from deep_worker_cli.basesession.session_adapter import (
    get_session_info,
    map_agent_status,
)

__all__ = [
    "SessionInfo",
    "SessionStatus",
    "SessionData",
    "SessionMetadata",
    "SessionTypeId",
    "get_session_info",
    "map_agent_status",
    "DeepWorkerDetector",
]
