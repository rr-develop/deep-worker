"""Agent process implementation for deep_worker."""

import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.exceptions import IPCError
from deep_worker_cli.ipc.server import IPCConnection, IPCServer
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentProcessInfo, AgentStatus


class AgentProcess:
    """
    Main class for a deep-worker agent process.

    Responsibilities:
    - Register in ProcessRegistry
    - Create PID file
    - Initialize agent (LangGraph)
    - Setup IPC server
    - Handle signals
    - Manage lifecycle
    """

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        registry: ProcessRegistry | None = None,
        socket_path: Path | None = None,
        pid_file: Path | None = None,
        data_dir: Path | None = None,
        skip_permissions: bool = False,
    ):
        """Initialize the agent process.

        Args:
            name: Unique name for this agent
            config: Agent configuration
            registry: Optional ProcessRegistry instance
            socket_path: Optional custom socket path
            pid_file: Optional custom PID file path
            data_dir: Optional custom data directory
            skip_permissions: Skip permission checks (auto-approve)
        """
        self.name = name
        self.config = config
        self.skip_permissions = skip_permissions

        # Paths
        self.socket_path = socket_path or Path(f"/tmp/deep_worker_{name}.sock")
        self.pid_file = pid_file or Path(f"/tmp/deep_worker_{name}.pid")
        self.data_dir = data_dir or Path(f"~/.deep_worker/{name}").expanduser()

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.registry = registry or ProcessRegistry()
        self.agent = None  # Will be initialized in start()
        self.ipc_server: IPCServer | None = None
        self.session_manager: SessionManager | None = None

        # State
        self.running = False
        self.clients: dict[str, IPCConnection] = {}
        self.started_at: datetime | None = None  # Set when agent starts

    async def start(self) -> None:
        """Start the agent process.

        This is the main entry point for the agent.
        """
        try:
            # 0. Set started_at timestamp
            self.started_at = datetime.now()

            # 1. Register in registry
            await self._register()

            # 2. Create PID file
            await self._create_pid_file()

            # 3. Setup signal handlers
            self._setup_signal_handlers()

            # 4. Initialize session manager
            self.session_manager = SessionManager(data_dir=self.data_dir)

            # 5. Initialize agent (placeholder for now)
            # TODO: Initialize LangGraph agent with create_deep_agent
            self.agent = None  # Will be implemented in full version

            # 6. Start IPC server
            self.ipc_server = IPCServer(
                socket_path=self.socket_path,
                handler=self._handle_ipc_command,
            )
            await self.ipc_server.start()

            # 7. Enter event loop
            self.running = True
            print(f"Agent '{self.name}' started (PID {os.getpid()})")
            print(f"Socket: {self.socket_path}")
            print(f"Working directory: {self.config.working_dir}")

            # Event loop would run here
            # For now, just wait for shutdown signal

        except Exception as e:
            print(f"Failed to start agent: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Graceful shutdown of the agent."""
        if not self.running:
            return

        print(f"Shutting down agent '{self.name}'...")
        self.running = False

        # 1. Disconnect clients
        for client in list(self.clients.values()):
            await client.close()
        self.clients.clear()

        # 2. Stop IPC server
        if self.ipc_server:
            await self.ipc_server.stop()

        # 3. Persist state
        if self.session_manager:
            self.session_manager.set_status(AgentStatus.SHUTDOWN)

        # 4. Unregister
        await self._unregister()

        # 5. Cleanup files
        await self._remove_pid_file()

        print(f"Agent '{self.name}' stopped")

    async def _register(self) -> None:
        """Register agent in ProcessRegistry."""
        if self.started_at is None:
            self.started_at = datetime.now()

        process_info = AgentProcessInfo(
            name=self.name,
            pid=os.getpid(),
            socket_path=self.socket_path,
            pid_file=self.pid_file,
            working_dir=str(self.config.working_dir),
            started_at=self.started_at,
        )
        self.registry.register(process_info)

    async def _unregister(self) -> None:
        """Unregister agent from ProcessRegistry."""
        self.registry.unregister(self.name)

    async def _create_pid_file(self) -> None:
        """Create PID file."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

    async def _remove_pid_file(self) -> None:
        """Remove PID file."""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False

    async def _handle_ipc_command(
        self,
        command: dict[str, Any],
        connection: IPCConnection,
    ) -> dict[str, Any]:
        """Handle IPC command from client.

        Args:
            command: Command dictionary
            connection: Client connection

        Returns:
            Response dictionary
        """
        cmd_type = command.get("command")

        if cmd_type == "attach":
            return await self._handle_attach(command, connection)
        elif cmd_type == "detach":
            return await self._handle_detach(command, connection)
        elif cmd_type == "status":
            return self._handle_status(command)
        elif cmd_type == "invoke":
            return await self._handle_invoke(command, connection)
        else:
            return {"success": False, "error": f"Unknown command: {cmd_type}"}

    async def _handle_attach(
        self, command: dict[str, Any], connection: IPCConnection
    ) -> dict[str, Any]:
        """Handle client attach command."""
        client_id = command.get("client_id", "unknown")
        self.clients[client_id] = connection

        return {
            "success": True,
            "agent_name": self.name,
            "status": self.session_manager.status.value if self.session_manager else "unknown",
        }

    async def _handle_detach(
        self, command: dict[str, Any], connection: IPCConnection
    ) -> dict[str, Any]:
        """Handle client detach command."""
        client_id = command.get("client_id")
        if client_id and client_id in self.clients:
            del self.clients[client_id]

        return {"success": True}

    def _handle_status(self, command: dict[str, Any]) -> dict[str, Any]:
        """Handle status query command.

        Returns full session information compatible with BaseSession protocol.
        """
        if not self.session_manager:
            return {"success": False, "error": "Session manager not initialized"}

        # Import here to avoid circular dependency
        from deep_worker_cli.basesession.session_adapter import get_session_info

        try:
            session_info = get_session_info(self)
            return {
                "success": True,
                "session_info": session_info.model_dump(mode="json"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_invoke(
        self, command: dict[str, Any], connection: IPCConnection
    ) -> dict[str, Any]:
        """Handle agent invocation command.

        This is a placeholder - full implementation will integrate with LangGraph.
        """
        # TODO: Implement actual agent invocation with LangGraph
        return {
            "success": False,
            "error": "Agent invocation not yet implemented",
        }
