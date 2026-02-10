"""IPC communication for deep_worker agents.

This module provides Unix socket-based IPC for agent-client communication.

Protocol:
    - JSON newline-delimited messages
    - Commands: attach, detach, status, invoke
    - Request-response pattern
    - Async I/O with asyncio

Components:
    - IPCProtocol: Message encoding/decoding (JSON + newline)
    - IPCServer: Unix socket server with concurrent connection handling
    - IPCConnection: Per-client connection wrapper with reader/writer

Socket Location:
    /tmp/deep_worker_{agent_name}.sock

Example:
    >>> from deep_worker_cli.ipc import IPCServer
    >>>
    >>> async def handle_command(command: dict, connection: IPCConnection) -> dict:
    ...     return {"success": True, "result": "ok"}
    >>>
    >>> server = IPCServer(
    ...     socket_path=Path("/tmp/my_agent.sock"),
    ...     handler=handle_command,
    ... )
    >>> await server.start()
"""

from deep_worker_cli.ipc.protocol import IPCProtocol
from deep_worker_cli.ipc.server import IPCConnection, IPCServer

__all__ = ["IPCProtocol", "IPCConnection", "IPCServer"]
