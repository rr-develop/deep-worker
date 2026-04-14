"""Client SDK for connecting to deep_worker agents via RPyC.

This module provides a high-level SDK for connecting to and communicating
with deep-worker agent processes via RPyC (AsyncioServer from the local rpyc_async checkout).

IMPORTANT: Only RPyC connections are supported. Unix sockets are NOT supported.

Features:
    - Agent discovery via ProcessRegistry
    - Automatic RPyC connection
    - Command execution (status, invoke, get_history)
    - SessionInfo queries (BaseSession protocol)
    - Error handling

Example:
    >>> from deep_worker_cli.client import AgentClient
    >>>
    >>> # Connect to agent via RPyC
    >>> client = AgentClient("my-agent")
    >>> await client.connect()  # Connects to RPyC port
    >>>
    >>> # Send a prompt
    >>> response = await client.invoke_agent("Hello, world!")
    >>> print(response["response"]["content"])
    >>>
    >>> # Query status
    >>> session_info = await client.get_status()
    >>> print(f"Status: {session_info.status}")
    >>>
    >>> # Disconnect
    >>> await client.disconnect()

Usage Notes:
    - Agents must be started with 'deep-worker' command
    - RPyC port must be configured (automatic)
    - AgentClient uses ProcessRegistry to locate agents by name
    - All methods are async and require await
    - Always disconnect after use to free resources
"""

from deep_worker_cli.client.agent_client import AgentClient

__all__ = ["AgentClient"]
