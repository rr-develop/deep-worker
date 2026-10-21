"""DeepWorkerDetector for cco daemon integration.

This module provides the DeepWorkerDetector class that allows cco daemon
to discover and query all running deep-worker agent processes.
"""

import logging
from typing import TYPE_CHECKING

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry

if TYPE_CHECKING:
    from deep_worker_cli.basesession.protocol_types import SessionInfo

logger = logging.getLogger(__name__)


class DeepWorkerDetector:
    """Detector for deep-worker processes for cco daemon integration.

    This class provides methods to discover all running deep-worker agents
    and query their status, returning SessionInfo objects compatible with
    the BaseSession protocol.

    Example:
        detector = DeepWorkerDetector()
        sessions = await detector.detect_sessions()
        for session in sessions:
            print(f"Agent: {session.name}, Status: {session.status}")
    """

    def __init__(self, registry: ProcessRegistry | None = None):
        """Initialize the detector.

        Args:
            registry: Optional ProcessRegistry instance. If None, uses default.
        """
        self.registry = registry or ProcessRegistry()

    async def detect_sessions(self) -> list["SessionInfo"]:
        """Detect all running deep-worker agent processes.

        Queries the ProcessRegistry for all registered agents and retrieves
        their status information via IPC.

        Returns:
            List of SessionInfo objects for all running agents

        Note:
            If an agent fails to respond (e.g., crashed but not cleaned up),
            it will be skipped and a warning will be logged.
        """
        # ``list_all`` raises RegistryReadError when the registry file exists
        # but cannot be read. That propagates on purpose: this detector is a
        # library entry point for the cco daemon, not a user interface, so it
        # has no business deciding how the failure is presented. Swallowing it
        # into an empty list would reintroduce the exact silent-"0 sessions"
        # defect the loud registry was added to kill, and the daemon could not
        # tell "no agents" from "I cannot see the agents". Callers that do own
        # a UI (see ``cli/client_command.py``) catch it and print the remedy.
        agents = self.registry.list_all()
        sessions: list[SessionInfo] = []

        for agent in agents:
            try:
                # Create client and query status
                client = AgentClient(agent.name, registry=self.registry)
                await client.connect()

                session_info = await client.get_status()
                sessions.append(session_info)

                await client.disconnect()

            except Exception as e:
                logger.warning(
                    f"Failed to query agent {agent.name}: {e}",
                    exc_info=True,
                )
                # Continue with other agents

        return sessions
