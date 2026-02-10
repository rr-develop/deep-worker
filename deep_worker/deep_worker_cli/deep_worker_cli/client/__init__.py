"""Client SDK for connecting to deep_worker agents.

This module provides a high-level SDK for connecting to and communicating
with deep-worker agent processes via Unix socket IPC.

Features:
    - Agent discovery via ProcessRegistry
    - Automatic socket connection
    - Command execution (attach, detach, status, invoke)
    - SessionInfo queries (BaseSession protocol)
    - Error handling and reconnection support

Example:
    >>> from deep_worker_cli.client import AgentClient
    >>>
    >>> # Connect to agent
    >>> client = AgentClient("my-agent")
    >>> await client.connect()
    >>>
    >>> # Query status
    >>> session_info = await client.get_status()
    >>> print(f"Status: {session_info.status}")
    >>> print(f"Tasks: {session_info.tasks_completed}")
    >>>
    >>> # Disconnect
    >>> await client.disconnect()

Usage Notes:
    - Agents must be started first with 'deep-worker' command
    - AgentClient uses ProcessRegistry to locate agents by name
    - All methods are async and require await
    - Always disconnect after use to free resources
"""

from deep_worker_cli.client.agent_client import AgentClient

__all__ = ["AgentClient"]
