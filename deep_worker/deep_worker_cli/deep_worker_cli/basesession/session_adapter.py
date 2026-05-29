"""Session adapter for converting AgentProcess to BaseSession SessionInfo.

This adapter provides compatibility with the BaseSession protocol used by
cco daemon and other tools.
"""

import os
from typing import TYPE_CHECKING

from deep_worker_cli.basesession.protocol_types import (
    SessionData,
    SessionInfo,
    SessionMetadata,
    SessionStatus,
)
from deep_worker_cli.types import AgentStatus

if TYPE_CHECKING:
    from deep_worker_cli.process.agent_process import AgentProcess


def map_agent_status(agent_status: AgentStatus) -> SessionStatus:
    """Map internal agent status to BaseSession SessionStatus.

    Args:
        agent_status: Internal agent status

    Returns:
        Corresponding SessionStatus for BaseSession protocol
    """
    mapping = {
        AgentStatus.IDLE: SessionStatus.READY,
        AgentStatus.RUNNING: SessionStatus.WORKING,
        AgentStatus.WAITING_INPUT: SessionStatus.WAITING_INPUT,
        AgentStatus.INTERRUPTED: SessionStatus.INTERRUPTED,
        AgentStatus.ERROR: SessionStatus.ERROR,
        AgentStatus.SHUTDOWN: SessionStatus.UNKNOWN,
    }
    return mapping.get(agent_status, SessionStatus.UNKNOWN)


def get_session_info(agent_process: "AgentProcess") -> SessionInfo:
    """Convert agent process state to BaseSession SessionInfo.

    This allows cco daemon and other tools to track deep-worker processes
    using the unified BaseSession protocol.

    Args:
        agent_process: The agent process instance

    Returns:
        SessionInfo with current agent state

    Raises:
        ValueError: If session_manager or started_at are not initialized
    """
    if agent_process.session_manager is None:
        msg = "Agent process session_manager is not initialized"
        raise ValueError(msg)

    if agent_process.started_at is None:
        msg = "Agent process started_at is not set"
        raise ValueError(msg)

    return SessionInfo(
        name=agent_process.name,
        session_type="langchain",
        status=map_agent_status(agent_process.session_manager.status),
        working_dir=str(agent_process.config.working_dir),
        created_at=agent_process.started_at,
        last_activity=agent_process.session_manager.last_activity,
        tasks_completed=agent_process.session_manager.tasks_completed,
        error_message=agent_process.session_manager.error_message,
        data=SessionData(
            message_count=agent_process.session_manager.message_count,
            process_id=os.getpid(),
        ),
        metadata=SessionMetadata(
            environment=dict(os.environ),
            properties={
                "agent_type": "deep_worker",
                "model": agent_process.config.model,
            },
        ),
    )
