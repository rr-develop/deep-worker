# Deep Worker Agent Separation Architecture Design (Revised)

**Author**: Design Document for Review (v2)
**Date**: 2025-02-10
**Status**: Draft for review
**Goal**: Separating the deep_worker agent from the CLI process with support for multiple named agents

---

## Table of Contents

1. [Overview of Changes](#1-overview-of-changes)
2. [Analysis of Reference Systems](#2-analysis-of-reference-systems)
3. [Requirements and Goals](#3-requirements-and-goals)
4. [Target Architecture](#4-target-architecture)
5. [Process Management](#5-process-management)
6. [System Components](#6-system-components)
7. [Interaction Protocols](#7-interaction-protocols)
8. [Session Management](#8-session-management)
9. [Middleware Design](#9-middleware-design)
10. [BaseSession Protocol Integration](#10-basesession-protocol-integration)
11. [CLI Usage Examples](#11-cli-usage-examples)
12. [Migration Plan](#12-migration-plan)
13. [Open Questions](#13-open-questions)

---

## 1. Overview of Changes

### 1.1 Key changes from v1

**Removed:**
- ❌ Embedded mode (agent in the same process as the CLI)
- ❌ The name `deep-worker-server`
- ❌ Centralized server approach (one server for all sessions)
- ❌ WebSocket URL in the CLI (`--server ws://...`)

**Added:**
- ✅ Named agent processes (each agent = a separate process with a unique name)
- ✅ Process Registry for discovery
- ✅ Unix Socket IPC (as in comparable session-daemon tools)
- ✅ PID files for tracking processes
- ✅ Simplified CLI: `deep-worker-cli <agent_name>`
- ✅ BaseSession protocol compatibility for integration with the cco daemon

### 1.2 Architectural approach

Instead of a centralized server with many sessions, now:
- **Each agent = a separate process** with a unique name
- **One agent = one process** (by default)
- **Multiple processes** can run in parallel
- **Process Registry** for discovery and connection

```
OLD (v1):                          NEW (v2):
┌────────────────────────┐        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Agent Server Process  │        │ deep-worker     │  │ deep-worker     │  │ deep-worker     │
│  ┌──────────────────┐  │        │ process         │  │ process         │  │ process         │
│  │  Session A       │  │        │ "project-alpha" │  │ "data-analysis" │  │ "code-review"   │
│  ├──────────────────┤  │        │                 │  │                 │  │                 │
│  │  Session B       │  │        │ [Agent]         │  │ [Agent]         │  │ [Agent]         │
│  ├──────────────────┤  │        │ .sock + .pid    │  │ .sock + .pid    │  │ .sock + .pid    │
│  │  Session C       │  │        └─────────────────┘  └─────────────────┘  └─────────────────┘
│  └──────────────────┘  │                ▲                    ▲                    ▲
└────────────────────────┘                │                    │                    │
          ▲                               │                    │                    │
          │                      ┌────────┴─────────┬──────────┴─────────┬─────────┘
          │                      │                  │                    │
     [Clients]            [CLI Client]        [CLI Client]        [CLI Client]
```

---

## 2. Analysis of Reference Systems

### 2.1 Session daemon pattern

**Architecture:**
```python
# Daemon
class SessionDaemon:
    PID_FILE = Path("/tmp/session_daemon.pid")
    SOCKET_PATH = "/tmp/session_daemon.sock"

    tracked_sessions: Dict[int, TrackedSession]  # PID → Session
    session_logs: Dict[str, Path]                # session_name → log file
```

**IPC Protocol:**
```python
# Client → Daemon
{
    "command": "get_status",
    "session_name": "my-session"
}

# Daemon → Client
{
    "success": true,
    "data": {
        "status": "ready",
        "message_count": 42,
        ...
    }
}
```

**Key features:**
- Unix socket for IPC (fast, local only)
- PID file to check whether the daemon is running
- session_name for identification
- JSON-based protocol

### 2.2 BaseSession Protocol (ai_lib)

**SessionInfo model:**
```python
class SessionInfo(BaseModel):
    name: str                      # Unique identifier
    session_type: SessionTypeId    # "cco" | "csdk" | "persistent_csdk" | "langchain"
    status: SessionStatus          # READY | WORKING | WAITING_INPUT | ERROR | ...
    working_dir: str
    created_at: datetime
    last_activity: Optional[datetime]
    tasks_completed: int
    error_message: Optional[str]
    data: Optional[SessionData]    # message_count, process_id, socket_path, ...
    metadata: Optional[SessionMetadata]
```

**Key features:**
- A unified protocol for all session types
- Clear statuses (READY, WORKING, WAITING_INPUT, etc.)
- Metadata for extensibility
- Support for cco daemon integration

---

## 3. Requirements and Goals

### 3.1 Functional requirements

#### Mandatory (Must Have)
- **REQ-1**: Each agent runs as a separate named process
- **REQ-2**: The agent name is unique; name conflicts are detected at startup
- **REQ-3**: The CLI connects to an agent by name: `deep-worker-cli <agent_name>`
- **REQ-4**: Multiple parallel agents can run at the same time
- **REQ-5**: The CLI can disconnect/reconnect without stopping the agent
- **REQ-6**: Process registry for discovery of all running agents
- **REQ-7**: BaseSession protocol compatible (for future cco daemon integration)

#### Desirable (Should Have)
- **REQ-8**: Multiple simultaneous clients for one agent (read-only observers)
- **REQ-9**: Hot reload of the agent without losing state
- **REQ-10**: Health checks and monitoring

#### Optional (Could Have)
- **REQ-11**: Multi-threading inside a single agent process (single-threaded by default)
- **REQ-12**: WebUI as an alternative client
- **REQ-13**: Agent-to-agent communication

### 3.2 Non-functional requirements

- **NFR-1**: Local execution only (Unix sockets, not network)
- **NFR-2**: No authentication (trusted local environment)
- **NFR-3**: Latency < 50ms for local IPC
- **NFR-4**: Zero data loss on reconnection
- **NFR-5**: Simple deployment (single binary, no external deps)

---

## 4. Target Architecture

### 4.1 High-level architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PROCESS LAYER                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────┐ │
│  │ deep-worker        │  │ deep-worker        │  │ deep-worker   │ │
│  │ "project-alpha"    │  │ "data-analysis"    │  │ "code-review" │ │
│  │                    │  │                    │  │               │ │
│  │ PID: 12345         │  │ PID: 12346         │  │ PID: 12347    │ │
│  │ Socket: .sock      │  │ Socket: .sock      │  │ Socket: .sock │ │
│  │                    │  │                    │  │               │ │
│  │ ┌────────────────┐ │  │ ┌────────────────┐ │  │ ┌───────────┐ │ │
│  │ │ Agent Instance │ │  │ │ Agent Instance │ │  │ │ Agent     │ │ │
│  │ │ (LangGraph)    │ │  │ │ (LangGraph)    │ │  │ │ Instance  │ │ │
│  │ └────────────────┘ │  │ └────────────────┘ │  │ └───────────┘ │ │
│  │ ┌────────────────┐ │  │ ┌────────────────┐ │  │ ┌───────────┐ │ │
│  │ │ IPC Server     │ │  │ │ IPC Server     │ │  │ │ IPC Server│ │ │
│  │ │ (Unix Socket)  │ │  │ │ (Unix Socket)  │ │  │ │(UnixSocket│ │ │
│  │ └────────────────┘ │  │ └────────────────┘ │  │ └───────────┘ │ │
│  │ ┌────────────────┐ │  │ ┌────────────────┐ │  │ ┌───────────┐ │ │
│  │ │ Session State  │ │  │ │ Session State  │ │  │ │ Session   │ │ │
│  │ │ Manager        │ │  │ │ Manager        │ │  │ │ State Mgr │ │ │
│  │ └────────────────┘ │  │ └────────────────┘ │  │ └───────────┘ │ │
│  └────────────────────┘  └────────────────────┘  └───────────────┘ │
│           ▲                       ▲                      ▲           │
└───────────┼───────────────────────┼──────────────────────┼───────────┘
            │                       │                      │
            │ Unix Socket IPC       │                      │
            │                       │                      │
┌───────────▼───────────────────────▼──────────────────────▼───────────┐
│                         CLIENT LAYER                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │ deep-worker-  │  │ deep-worker-  │  │ deep-worker-  │          │
│  │ cli "project- │  │ cli "data-    │  │ cli "code-    │          │
│  │ alpha"        │  │ analysis"     │  │ review"       │          │
│  │               │  │               │  │               │          │
│  │ [Textual UI]  │  │ [Textual UI]  │  │ [Textual UI]  │          │
│  └───────────────┘  └───────────────┘  └───────────────┘          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│                      REGISTRY & PERSISTENCE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Process Registry (/tmp/deep_worker_registry.json)             │ │
│  │  {                                                              │ │
│  │    "project-alpha": {                                           │ │
│  │      "pid": 12345,                                              │ │
│  │      "socket_path": "/tmp/deep_worker_project-alpha.sock",     │ │
│  │      "pid_file": "/tmp/deep_worker_project-alpha.pid",         │ │
│  │      "working_dir": "/path/to/project",                        │ │
│  │      "started_at": "2025-02-10T12:00:00Z",                     │ │
│  │      "last_activity": "2025-02-10T12:30:00Z"                   │ │
│  │    },                                                           │ │
│  │    ...                                                          │ │
│  │  }                                                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Per-Agent State Persistence                                   │ │
│  │  - Checkpointer (SqliteSaver per agent)                        │ │
│  │  - Session metadata                                            │ │
│  │  - Message history                                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 File Structure per Agent

```
/tmp/
├── deep_worker_<agent_name>.pid          # PID file
├── deep_worker_<agent_name>.sock         # Unix socket
├── deep_worker_registry.json             # Registry (shared)
└── deep_worker_registry.lock             # Lock file for atomic updates

~/.deep_worker/<agent_name>/
├── checkpoints.db                        # LangGraph checkpointer
├── session_metadata.json                 # Session metadata
├── config.yaml                           # Agent configuration
└── logs/
    └── agent.log                         # Agent logs
```

### 4.3 Process Lifecycle

```
START:
  ┌──────────────────────────────────────────────────────────┐
  │ 1. Check name uniqueness in registry                     │
  │    - If exists → Error: "Agent '<name>' already running" │
  │ 2. Create PID file                                       │
  │ 3. Create Unix socket                                    │
  │ 4. Register in registry                                  │
  │ 5. Initialize agent (create_deep_agent)                  │
  │ 6. Start IPC server                                      │
  │ 7. Enter event loop                                      │
  └──────────────────────────────────────────────────────────┘

RUNNING:
  ┌──────────────────────────────────────────────────────────┐
  │ - Listen for IPC connections                             │
  │ - Process commands (invoke, status, etc.)                │
  │ - Update last_activity in registry                       │
  │ - Persist state via checkpointer                         │
  └──────────────────────────────────────────────────────────┘

SHUTDOWN:
  ┌──────────────────────────────────────────────────────────┐
  │ 1. Graceful shutdown signal (SIGTERM/SIGINT)             │
  │ 2. Disconnect all clients                                │
  │ 3. Persist final state                                   │
  │ 4. Unregister from registry                              │
  │ 5. Remove Unix socket                                    │
  │ 6. Remove PID file                                       │
  │ 7. Exit                                                  │
  └──────────────────────────────────────────────────────────┘
```

---

## 5. Process Management

### 5.1 ProcessRegistry

```python
class ProcessRegistry:
    """
    Centralized registry for tracking all running deep-worker processes.

    Responsibilities:
    - Register/unregister agent processes
    - Lookup agent by name
    - List all running agents
    - Cleanup stale entries (dead processes)
    """

    REGISTRY_FILE = Path("/tmp/deep_worker_registry.json")
    LOCK_FILE = Path("/tmp/deep_worker_registry.lock")

    def __init__(self):
        self.registry_file = self.REGISTRY_FILE
        self.lock_file = self.LOCK_FILE

    def register(self, agent_info: AgentProcessInfo) -> None:
        """
        Register a new agent process.

        Raises:
            AgentAlreadyExistsError: If agent with this name already exists
        """
        with self._lock():
            registry = self._load_registry()

            # Check uniqueness
            if agent_info.name in registry:
                existing = registry[agent_info.name]
                # Verify process still alive
                if self._is_process_alive(existing["pid"]):
                    raise AgentAlreadyExistsError(
                        f"Agent '{agent_info.name}' is already running (PID {existing['pid']})"
                    )
                # Dead process, cleanup
                self._cleanup_stale_entry(agent_info.name)

            # Register new entry
            registry[agent_info.name] = {
                "pid": agent_info.pid,
                "socket_path": str(agent_info.socket_path),
                "pid_file": str(agent_info.pid_file),
                "working_dir": agent_info.working_dir,
                "started_at": agent_info.started_at.isoformat(),
                "last_activity": datetime.now().isoformat(),
            }

            self._save_registry(registry)

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the registry"""
        with self._lock():
            registry = self._load_registry()
            if agent_name in registry:
                del registry[agent_name]
                self._save_registry(registry)

    def lookup(self, agent_name: str) -> Optional[AgentProcessInfo]:
        """
        Find an agent by name.

        Returns:
            AgentProcessInfo if found and alive, otherwise None
        """
        registry = self._load_registry()
        if agent_name not in registry:
            return None

        entry = registry[agent_name]

        # Verify process still alive
        if not self._is_process_alive(entry["pid"]):
            # Cleanup stale entry
            self._cleanup_stale_entry(agent_name)
            return None

        return AgentProcessInfo(
            name=agent_name,
            pid=entry["pid"],
            socket_path=Path(entry["socket_path"]),
            pid_file=Path(entry["pid_file"]),
            working_dir=entry["working_dir"],
            started_at=datetime.fromisoformat(entry["started_at"]),
        )

    def list_all(self) -> List[AgentProcessInfo]:
        """Get the list of all running agents"""
        registry = self._load_registry()
        agents = []

        for name, entry in registry.items():
            if self._is_process_alive(entry["pid"]):
                agents.append(AgentProcessInfo(
                    name=name,
                    pid=entry["pid"],
                    socket_path=Path(entry["socket_path"]),
                    pid_file=Path(entry["pid_file"]),
                    working_dir=entry["working_dir"],
                    started_at=datetime.fromisoformat(entry["started_at"]),
                ))
            else:
                # Cleanup stale entry
                self._cleanup_stale_entry(name)

        return agents

    def _is_process_alive(self, pid: int) -> bool:
        """Check if process is alive via /proc"""
        proc_path = Path(f"/proc/{pid}")
        if not proc_path.exists():
            return False

        # Verify it's our process (deep-worker)
        try:
            cmdline = (proc_path / "cmdline").read_text()
            return "deep-worker" in cmdline or "deep_worker" in cmdline
        except (PermissionError, OSError):
            return False

    def _lock(self):
        """Context manager for the file lock"""
        return FileLock(self.lock_file)

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from JSON file"""
        if not self.registry_file.exists():
            return {}
        return json.loads(self.registry_file.read_text())

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save registry to JSON file"""
        self.registry_file.write_text(json.dumps(registry, indent=2))


@dataclass
class AgentProcessInfo:
    """Information about a running agent process"""
    name: str
    pid: int
    socket_path: Path
    pid_file: Path
    working_dir: str
    started_at: datetime
```

### 5.2 Agent Process Main

```python
class AgentProcess:
    """
    Main class for the deep-worker process.

    Responsibilities:
    - Initialize agent
    - Setup IPC server
    - Handle signals
    - Manage lifecycle
    """

    def __init__(
        self,
        name: str,
        working_dir: str | Path,
        config: AgentConfig,
    ):
        self.name = name
        self.working_dir = Path(working_dir).resolve()
        self.config = config

        # Paths
        self.socket_path = Path(f"/tmp/deep_worker_{name}.sock")
        self.pid_file = Path(f"/tmp/deep_worker_{name}.pid")
        self.data_dir = Path(f"~/.deep_worker/{name}").expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.registry = ProcessRegistry()
        self.agent: Optional[CompiledStateGraph] = None
        self.ipc_server: Optional[IPCServer] = None
        self.session_manager: Optional[SessionManager] = None

        # State
        self.running = False
        self.clients: Dict[str, IPCClient] = {}

    def start(self) -> None:
        """Start the agent process"""

        # 1. Register in registry (checks uniqueness)
        try:
            process_info = AgentProcessInfo(
                name=self.name,
                pid=os.getpid(),
                socket_path=self.socket_path,
                pid_file=self.pid_file,
                working_dir=str(self.working_dir),
                started_at=datetime.now(),
            )
            self.registry.register(process_info)
        except AgentAlreadyExistsError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # 2. Create PID file
        self.pid_file.write_text(str(os.getpid()))

        # 3. Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # 4. Initialize agent
        self.agent = create_deep_agent(
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            tools=self.config.tools,
            backend=self.config.backend,
            checkpointer=SqliteSaver.from_conn_string(
                str(self.data_dir / "checkpoints.db")
            ),
        )

        # 5. Initialize session manager
        self.session_manager = SessionManager(
            agent=self.agent,
            data_dir=self.data_dir,
        )

        # 6. Start IPC server
        self.ipc_server = IPCServer(
            socket_path=self.socket_path,
            handler=self._handle_ipc_command,
        )
        self.ipc_server.start()

        # 7. Enter event loop
        self.running = True
        print(f"Agent '{self.name}' started (PID {os.getpid()})")
        print(f"Socket: {self.socket_path}")
        print(f"Working directory: {self.working_dir}")

        try:
            self._event_loop()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown"""
        if not self.running:
            return

        print(f"Shutting down agent '{self.name}'...")
        self.running = False

        # 1. Disconnect clients
        for client in list(self.clients.values()):
            client.disconnect()

        # 2. Stop IPC server
        if self.ipc_server:
            self.ipc_server.stop()

        # 3. Persist state
        if self.session_manager:
            self.session_manager.persist()

        # 4. Unregister
        self.registry.unregister(self.name)

        # 5. Cleanup files
        if self.socket_path.exists():
            self.socket_path.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()

        print(f"Agent '{self.name}' stopped")

    def _event_loop(self) -> None:
        """Main event loop"""
        while self.running:
            # Update last_activity in registry
            # (IPC server handles actual events)
            time.sleep(1)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False

    async def _handle_ipc_command(
        self,
        command: Dict[str, Any],
        client: IPCClient,
    ) -> Dict[str, Any]:
        """Handle IPC command from client"""
        cmd_type = command.get("command")

        if cmd_type == "invoke":
            return await self._handle_invoke(command, client)
        elif cmd_type == "status":
            return self._handle_status(command)
        elif cmd_type == "attach":
            return self._handle_attach(command, client)
        elif cmd_type == "detach":
            return self._handle_detach(command, client)
        else:
            return {"success": False, "error": f"Unknown command: {cmd_type}"}


def main():
    """Entry point for deep-worker"""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Worker Agent Process")
    parser.add_argument("name", help="Unique name for this agent")
    parser.add_argument("--working-dir", default=".", help="Working directory")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Model to use")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = AgentConfig.from_file(args.config)
    else:
        config = AgentConfig(
            model=args.model,
            working_dir=args.working_dir,
        )

    # Start agent
    agent = AgentProcess(
        name=args.name,
        working_dir=args.working_dir,
        config=config,
    )
    agent.start()


if __name__ == "__main__":
    main()
```

---

## 6. System Components

### 6.1 AgentClient (SDK for the CLI)

```python
class AgentClient:
    """
    Client SDK for connecting to the deep-worker process.

    Usage:
        client = AgentClient("project-alpha")
        await client.connect()

        async for event in client.stream("Hello, agent!"):
            print(event)

        await client.disconnect()
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.registry = ProcessRegistry()
        self.socket: Optional[socket.socket] = None
        self.connected = False

    async def connect(self) -> None:
        """Connect to agent process"""

        # 1. Lookup agent in registry
        agent_info = self.registry.lookup(self.agent_name)
        if not agent_info:
            raise AgentNotFoundError(
                f"Agent '{self.agent_name}' not found. "
                f"Start it with: deep-worker {self.agent_name}"
            )

        # 2. Connect to Unix socket
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(str(agent_info.socket_path))
            self.connected = True
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to agent: {e}")

        # 3. Send attach command
        response = await self._send_command({
            "command": "attach",
            "client_id": str(uuid.uuid4()),
        })

        if not response.get("success"):
            raise ConnectionError(f"Failed to attach: {response.get('error')}")

    async def disconnect(self) -> None:
        """Disconnect from agent"""
        if self.connected and self.socket:
            await self._send_command({"command": "detach"})
            self.socket.close()
            self.connected = False

    async def invoke(self, input_data: str | dict) -> dict:
        """Synchronous invocation"""
        response = await self._send_command({
            "command": "invoke",
            "input": input_data,
            "stream": False,
        })
        return response["data"]

    async def stream(self, input_data: str | dict) -> AsyncIterator[AgentEvent]:
        """Streaming invocation"""
        await self._send_command({
            "command": "invoke",
            "input": input_data,
            "stream": True,
        })

        # Receive streaming events
        while True:
            event = await self._receive_event()
            if event["type"] == "completed":
                break
            yield AgentEvent(**event)

    async def get_status(self) -> SessionInfo:
        """Get agent status (BaseSession compatible)"""
        response = await self._send_command({"command": "status"})
        return SessionInfo(**response["data"])

    async def _send_command(self, command: dict) -> dict:
        """Send command and receive response"""
        if not self.connected:
            raise RuntimeError("Not connected to agent")

        # Send JSON command
        command_json = json.dumps(command) + "\n"
        self.socket.sendall(command_json.encode('utf-8'))

        # Receive response
        response_data = b""
        while True:
            chunk = self.socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
            if b"\n" in response_data:
                break

        return json.loads(response_data.decode('utf-8'))

    async def _receive_event(self) -> dict:
        """Receive streaming event"""
        event_data = b""
        while True:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed")
            event_data += chunk
            if b"\n" in event_data:
                break

        return json.loads(event_data.decode('utf-8'))
```

### 6.2 CLI Application Integration

```python
class DeepWorkerCLI(TextualApp):
    """
    CLI application for connecting to the deep-worker process.

    Modified from original app.py to use AgentClient instead of in-process agent.
    """

    def __init__(
        self,
        agent_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.client: Optional[AgentClient] = None
        # ... other initialization

    async def on_mount(self) -> None:
        """Connect to agent on mount"""
        try:
            self.client = AgentClient(self.agent_name)
            await self.client.connect()

            # Show connection success
            self._add_app_message(f"Connected to agent '{self.agent_name}'")

        except AgentNotFoundError as e:
            self._add_error_message(str(e))
            self.exit()
        except ConnectionError as e:
            self._add_error_message(f"Connection error: {e}")
            self.exit()

    async def _submit_message(self, user_input: str) -> None:
        """Submit message to agent"""
        if not self.client:
            self._add_error_message("Not connected to agent")
            return

        # Add user message to UI
        self._add_user_message(user_input)

        # Stream response from agent
        try:
            async for event in self.client.stream(user_input):
                if event.type == "message":
                    self._add_assistant_message(event.content)
                elif event.type == "tool_call":
                    self._add_tool_call_message(event.tool_name, event.tool_args)
                elif event.type == "approval_required":
                    await self._handle_approval(event)
                # ... handle other events
        except Exception as e:
            self._add_error_message(f"Error: {e}")


def main():
    """Entry point for deep-worker-cli"""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Worker CLI Client")
    parser.add_argument("agent_name", help="Name of agent to connect to")
    parser.add_argument("--list", action="store_true", help="List all running agents")

    args = parser.parse_args()

    if args.list:
        # List all running agents
        registry = ProcessRegistry()
        agents = registry.list_all()
        if not agents:
            print("No running agents found")
        else:
            print("Running agents:")
            for agent in agents:
                print(f"  - {agent.name} (PID {agent.pid}, started {agent.started_at})")
        return

    # Start CLI
    app = DeepWorkerCLI(agent_name=args.agent_name)
    app.run()


if __name__ == "__main__":
    main()
```

---

## 7. Interaction Protocols

### 7.1 IPC Protocol (Unix Socket + JSON)

**Transport**: Unix Domain Socket
**Encoding**: JSON (newline-delimited)
**Pattern**: Request-Response + Streaming

#### 7.1.1 Client → Agent Messages

```python
# ATTACH - connect to the agent
{
    "command": "attach",
    "client_id": "uuid-1234",
    "mode": "interactive"  # interactive | observer
}

# Response
{
    "success": true,
    "data": {
        "agent_name": "project-alpha",
        "status": "ready",
        "clients_count": 1
    }
}

# DETACH - disconnect
{
    "command": "detach",
    "client_id": "uuid-1234"
}

# INVOKE - call the agent (sync)
{
    "command": "invoke",
    "input": "What is LangGraph?",
    "stream": false,
    "config": {}
}

# Response
{
    "success": true,
    "data": {
        "output": "LangGraph is...",
        "message_count": 5
    }
}

# INVOKE - call the agent (streaming)
{
    "command": "invoke",
    "input": "What is LangGraph?",
    "stream": true,
    "config": {}
}

# Streaming events (multiple responses)
{"type": "started", "timestamp": "..."}
{"type": "message", "content": "LangGraph is", "partial": true}
{"type": "message", "content": " a framework", "partial": true}
{"type": "tool_call", "tool_name": "search", "tool_args": {...}}
{"type": "tool_result", "result": {...}}
{"type": "message", "content": "...", "partial": false}
{"type": "completed", "timestamp": "..."}

# STATUS - get the agent status
{
    "command": "status"
}

# Response (BaseSession compatible)
{
    "success": true,
    "data": {
        "name": "project-alpha",
        "session_type": "langchain",
        "status": "ready",
        "working_dir": "/path/to/project",
        "created_at": "2025-02-10T12:00:00Z",
        "last_activity": "2025-02-10T12:30:00Z",
        "tasks_completed": 5,
        "data": {
            "message_count": 42,
            "process_id": 12345,
            "socket_path": "/tmp/deep_worker_project-alpha.sock"
        }
    }
}

# INTERRUPT - interrupt execution
{
    "command": "interrupt"
}

# APPROVE - respond to a Human-in-the-Loop request
{
    "command": "approve",
    "approval_id": "approval-456",
    "approved": true,
    "modifications": {}
}
```

---

## 8. Session Management

### 8.1 Single-Session per Process

In the current design:
- **One agent process = One session**
- Session ID = agent name
- Thread ID for LangGraph = agent name or default

This simplifies the architecture and matches the requirement "one thread per agent process by default".

### 8.2 Multi-Threading Support (Optional)

If multitasking support is required:

```python
class AgentProcess:
    def __init__(self, name: str, config: AgentConfig):
        # ...
        self.multi_threading = config.multi_threading
        self.sessions: Dict[str, Session] = {}  # thread_id → Session
        self.default_session: Session  # Main session (= agent name)

    async def _handle_invoke(self, command: dict, client: IPCClient):
        thread_id = command.get("thread_id", self.name)  # default = agent name

        if thread_id not in self.sessions:
            if not self.multi_threading:
                return {
                    "success": False,
                    "error": "Multi-threading not enabled"
                }
            # Create new thread/session
            self.sessions[thread_id] = Session(
                thread_id=thread_id,
                agent=self.agent,
            )

        session = self.sessions[thread_id]
        # ... execute
```

Configuration:
```yaml
# ~/.deep_worker/project-alpha/config.yaml
multi_threading:
  enabled: false  # Disabled by default
  max_threads: 5
```

---

## 9. Middleware Design

The middleware stays the same, but with a few adaptations:

### 9.1 New: ClientConnectionMiddleware

```python
class ClientConnectionMiddleware(AgentMiddleware):
    """
    Middleware for managing client connections.

    Adds context about the connected clients to the agent state.
    """

    def __init__(self, agent_process: AgentProcess):
        self.agent_process = agent_process

    async def before_agent(
        self,
        state: AgentState,
        runtime: ToolRuntime,
    ) -> AgentState:
        """Inject client context"""
        state["_client_context"] = {
            "agent_name": self.agent_process.name,
            "clients": list(self.agent_process.clients.keys()),
            "client_count": len(self.agent_process.clients),
        }
        return state
```

### 9.2 Enhanced: HumanInTheLoopMiddleware

```python
class EnhancedHumanInTheLoopMiddleware(HumanInTheLoopMiddleware):
    """
    HITL middleware with support for IPC clients.

    Sends approval requests to all connected clients.
    """

    def __init__(self, agent_process: AgentProcess, **kwargs):
        super().__init__(**kwargs)
        self.agent_process = agent_process

    async def request_approval(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> ApprovalResult:
        """Request approval from the connected clients"""
        approval_id = str(uuid.uuid4())

        # Broadcast the approval request to all clients
        event = {
            "type": "approval_required",
            "approval_id": approval_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

        for client in self.agent_process.clients.values():
            client.send_event(event)

        # Wait for response (first come first served)
        try:
            result = await asyncio.wait_for(
                self._wait_for_approval(approval_id),
                timeout=300,  # 5 minutes
            )
            return result
        except asyncio.TimeoutError:
            return ApprovalResult(approved=False, reason="timeout")
```

---

## 10. BaseSession Protocol Integration

### 10.1 SessionInfo Mapping

```python
def get_session_info(agent_process: AgentProcess) -> SessionInfo:
    """
    Convert agent process state to BaseSession SessionInfo.

    This will allow the cco daemon to track deep-worker processes.
    """
    return SessionInfo(
        name=agent_process.name,              # agent_name = session_name
        session_type="langchain",             # Fixed type
        status=_map_agent_status(agent_process.session_manager.status),
        working_dir=str(agent_process.working_dir),
        created_at=agent_process.started_at,
        last_activity=agent_process.last_activity,
        tasks_completed=agent_process.session_manager.tasks_completed,
        error_message=agent_process.session_manager.error_message,
        data=SessionData(
            message_count=agent_process.session_manager.message_count,
            process_id=os.getpid(),
            socket_path=str(agent_process.socket_path),
        ),
        metadata=SessionMetadata(
            environment=dict(os.environ),
            properties={
                "agent_type": "deep_worker",
                "model": agent_process.config.model,
                "clients_count": len(agent_process.clients),
            },
        ),
    )


def _map_agent_status(agent_status: str) -> SessionStatus:
    """Map internal agent status to BaseSession SessionStatus"""
    mapping = {
        "idle": SessionStatus.READY,
        "running": SessionStatus.WORKING,
        "waiting_approval": SessionStatus.WAITING_INPUT,
        "error": SessionStatus.ERROR,
    }
    return mapping.get(agent_status, SessionStatus.UNKNOWN)
```

### 10.2 Future: cco daemon Integration

In the future, the cco daemon will be able to discover deep-worker processes:

```python
# In the cco daemon
class DeepWorkerDetector:
    """Detector for deep-worker processes"""

    def detect_sessions(self) -> List[SessionInfo]:
        """Discover all running deep-worker processes"""
        registry = ProcessRegistry()
        agents = registry.list_all()

        sessions = []
        for agent in agents:
            # Query status via IPC
            try:
                client = AgentClient(agent.name)
                await client.connect()
                session_info = await client.get_status()
                await client.disconnect()
                sessions.append(session_info)
            except Exception as e:
                logger.warning(f"Failed to query agent {agent.name}: {e}")

        return sessions
```

---

## 11. CLI Usage Examples

### 11.1 Starting an Agent

```bash
# Start agent with default settings
$ deep-worker my-project
Agent 'my-project' started (PID 12345)
Socket: /tmp/deep_worker_my-project.sock
Working directory: /home/user/projects/my-project

# Start with specific model
$ deep-worker data-analysis --model claude-sonnet-4-5-20250929 --working-dir /data

# Start with config file
$ deep-worker code-review --config ~/.deep_worker/code-review.yaml

# Error: name already in use
$ deep-worker my-project
Error: Agent 'my-project' is already running (PID 12345)
```

### 11.2 Connecting with CLI

```bash
# Connect to agent
$ deep-worker-cli my-project
Connected to agent 'my-project'
/home/user/projects/my-project $

# Error: agent not found
$ deep-worker-cli nonexistent
Error: Agent 'nonexistent' not found.
Start it with: deep-worker nonexistent

# List all running agents
$ deep-worker-cli --list
Running agents:
  - my-project (PID 12345, started 2025-02-10 12:00:00)
  - data-analysis (PID 12346, started 2025-02-10 12:05:00)
  - code-review (PID 12347, started 2025-02-10 12:10:00)
```

### 11.3 Managing Agents

```bash
# Stop agent gracefully
$ kill -TERM $(cat /tmp/deep_worker_my-project.pid)
Agent 'my-project' shutting down...
Agent 'my-project' stopped

# Or via helper command
$ deep-worker-cli --stop my-project
Stopping agent 'my-project'...
Agent stopped successfully

# Check agent status
$ deep-worker-cli --status my-project
Agent: my-project
Status: ready
Working directory: /home/user/projects/my-project
Uptime: 2 hours 15 minutes
Tasks completed: 42
Connected clients: 1
```

### 11.4 Multiple Agents & Clients

```bash
# Terminal 1: Start agent
$ deep-worker project-alpha
Agent 'project-alpha' started...

# Terminal 2: Connect CLI
$ deep-worker-cli project-alpha
Connected to agent 'project-alpha'
project-alpha $ What is LangGraph?

# Terminal 3: Start another agent
$ deep-worker project-beta --working-dir /path/to/beta
Agent 'project-beta' started...

# Terminal 4: Connect to second agent
$ deep-worker-cli project-beta
Connected to agent 'project-beta'
project-beta $ Help me with data analysis

# Terminal 5: Observer mode (future feature)
$ deep-worker-cli project-alpha --mode observer
Connected to agent 'project-alpha' (observer mode)
[watching agent activity...]
```

---

## 12. Migration Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Goals:**
- Process registry
- Unix socket IPC
- Basic agent process
- CLI client SDK

**Deliverables:**
1. `ProcessRegistry` class with file-based storage
2. `IPCServer` and `IPCClient` with Unix socket
3. `AgentProcess` main class
4. `AgentClient` SDK
5. Name uniqueness enforcement
6. PID file management

**Milestone:** It is possible to start `deep-worker my-agent`, connect with `deep-worker-cli my-agent`, and send a simple command

### Phase 2: Agent Integration (Week 3)

**Goals:**
- Integrate LangGraph agent
- Session management
- State persistence
- Graceful shutdown

**Deliverables:**
1. Agent initialization in process
2. SessionManager with checkpointer
3. Signal handling (SIGTERM/SIGINT)
4. Registry cleanup on shutdown
5. State persistence across restarts

**Milestone:** A fully functional agent with persistence

### Phase 3: CLI Application (Week 4)

**Goals:**
- Refactor existing Textual UI
- Use AgentClient instead of in-process agent
- Streaming support
- HITL integration

**Deliverables:**
1. Modified `DeepWorkerCLI` app
2. Streaming event handling
3. Approval UI via IPC
4. Error handling & reconnection
5. Observer mode (read-only)

**Milestone:** The CLI fully works with a remote agent

### Phase 4: BaseSession Integration (Week 5)

**Goals:**
- SessionInfo mapping
- Status protocol implementation
- cco daemon compatibility

**Deliverables:**
1. `get_session_info()` implementation
2. Status command in the IPC protocol
3. SessionStatus mapping
4. Documentation for cco integration

**Milestone:** deep-worker processes can be discovered by the cco daemon

### Phase 5: Polish & Documentation (Week 6)

**Goals:**
- Helper commands (--list, --stop, --status)
- Error messages & UX
- Documentation
- Testing

**Deliverables:**
1. Helper commands implementation
2. Comprehensive error messages
3. User documentation
4. Integration tests
5. Deployment guide

**Milestone:** A production-ready system

---

## 13. Open Questions

### 13.1 Resolved Questions

1. **Session Timeout**: 1 hour for client connections ✓
2. **Max Concurrent Sessions**: Not applicable (each process = one session) ✓
3. **Authentication**: Not required (local only) ✓
4. **Agent Execution**: Separate processes, optionally multi-threading ✓
5. **WebUI**: Not required for now ✓

### 13.2 Pending Technical Decisions

1. **Registry Persistence Format**
   - Q: Is a JSON file enough, or is SQLite needed?
   - Options: JSON (simple), SQLite (queryable), Redis (distributed)
   - Recommendation: JSON for the MVP, SQLite for the future if queries are needed

2. **Socket Permission Model**
   - Q: Who may connect to the socket?
   - Options: Owner only (600), group (660), all (666)
   - Recommendation: Owner only (600) for security

3. **Stale Entry Cleanup Strategy**
   - Q: When should dead process entries be removed from the registry?
   - Options: On lookup, periodic cleanup, on new registration
   - Recommendation: On lookup + periodic cleanup (every 5 minutes)

4. **Multi-Threading Default**
   - Q: Should multi-threading be enabled by default?
   - Options: Disabled (simple), Enabled (flexible), Configurable
   - Recommendation: Disabled by default, configurable per agent

5. **Log Rotation Strategy**
   - Q: How should agent logs be managed?
   - Options: No rotation, size-based, time-based
   - Recommendation: Time-based daily rotation with a maximum of 7 days

### 13.3 UX Decisions

1. **Agent Name Validation**
   - Q: Which characters are allowed in names?
   - Options: Alphanumeric only, allow dash/underscore, allow all
   - Recommendation: `[a-zA-Z0-9_-]+` (alphanumeric + dash + underscore)

2. **Default Working Directory**
   - Q: Which directory should be used if none is specified?
   - Options: CWD, home dir, specific ~/.deep_worker/<name>
   - Recommendation: CWD (the current directory at startup)

3. **Agent Restart Behavior**
   - Q: What should happen to the data on restart?
   - Options: Keep all, clean logs, prompt user
   - Recommendation: Keep all data, rotate logs

---

## Appendix A: Configuration File Format

```yaml
# ~/.deep_worker/<agent_name>/config.yaml

agent:
  name: "project-alpha"
  working_dir: "/path/to/project"

model:
  name: "claude-sonnet-4-5-20250929"
  max_tokens: 20000
  temperature: 1.0

system_prompt: |
  You are a helpful AI assistant working on project-alpha.
  Your role is to help with software development tasks.

tools:
  - name: "custom_tool"
    module: "my_tools.custom"

backend:
  type: "filesystem"  # filesystem | sandbox
  root_dir: "/path/to/project"

persistence:
  checkpointer: "sqlite"
  path: "~/.deep_worker/project-alpha/checkpoints.db"

multi_threading:
  enabled: false
  max_threads: 5

logging:
  level: "INFO"
  file: "~/.deep_worker/project-alpha/logs/agent.log"
  rotation: "daily"
  max_days: 7

ipc:
  socket_path: "/tmp/deep_worker_project-alpha.sock"
  max_clients: 10
  client_timeout: 3600  # 1 hour
```

---

## Appendix B: Code Structure

```
deep_worker/
├── __init__.py
├── __main__.py                    # Entry point: deep-worker
├── process/
│   ├── __init__.py
│   ├── agent_process.py           # AgentProcess main class
│   ├── registry.py                # ProcessRegistry
│   └── ipc_server.py              # IPCServer
├── client/
│   ├── __init__.py
│   ├── agent_client.py            # AgentClient SDK
│   └── cli_app.py                 # CLI application (modified from app.py)
├── session/
│   ├── __init__.py
│   ├── session_manager.py         # SessionManager
│   └── basesession_adapter.py     # BaseSession protocol adapter
├── middleware/
│   ├── __init__.py
│   ├── client_connection.py       # ClientConnectionMiddleware
│   └── enhanced_hitl.py           # EnhancedHumanInTheLoopMiddleware
├── config/
│   ├── __init__.py
│   └── agent_config.py            # AgentConfig dataclass
└── exceptions.py                  # Custom exceptions

deep_worker_cli/
├── __init__.py
└── __main__.py                    # Entry point: deep-worker-cli
```

---

## Conclusion

The reworked design meets all the requirements:

✅ **Embedded mode removed** - the agent is always in a separate process
✅ **Named processes** - each agent has a unique name
✅ **Process registry** - discovery and management of processes
✅ **Unix Socket IPC** - fast local communication
✅ **Simplified CLI** - `deep-worker-cli <agent_name>`
✅ **Multiple processes** - many parallel agents
✅ **BaseSession compatible** - for future cco daemon integration
✅ **Local only** - without network exposure
✅ **No auth** - trusted environment

The architecture is inspired by comparable session-daemon tools and follows proven Unix IPC patterns.

**Next step**: Review and sign-off before implementation begins.
