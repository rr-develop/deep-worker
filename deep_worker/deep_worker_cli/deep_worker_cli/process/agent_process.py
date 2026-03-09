"""Agent process implementation for deep_worker."""

import asyncio
import os
import signal
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import rpyc  # type: ignore[import-untyped]
from rpyc.core.protocol import DEFAULT_CONFIG  # type: ignore[import-untyped]
from rpyc.utils.server import ThreadedServer  # type: ignore[import-untyped]

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentProcessInfo, AgentStatus


def _find_free_port(start_port: int = 18861, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port.

    Args:
        start_port: Starting port number (default: 18861)
        max_attempts: Maximum number of ports to try (default: 100)

    Returns:
        Free port number

    Raises:
        RuntimeError: If no free port found in range
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port to check if it's free
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                # If binding succeeds, port is free
                return port
        except OSError:
            # Port is in use, try next one
            continue

    raise RuntimeError(
        f"No free ports available in range {start_port}-{start_port + max_attempts - 1}"
    )


class AgentProcess:
    """
    Main class for a deep-worker agent process.

    Responsibilities:
    - Register in ProcessRegistry
    - Create PID file
    - Initialize agent (LangGraph)
    - Setup RPyC server
    - Handle signals
    - Manage lifecycle
    """

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        registry: ProcessRegistry | None = None,
        pid_file: Path | None = None,
        data_dir: Path | None = None,
        skip_permissions: bool = False,
        enable_task_manager: bool = False,
        rpyc_port: int | bool | None = True,
    ):
        """Initialize the agent process.

        Args:
            name: Unique name for this agent
            config: Agent configuration
            registry: Optional ProcessRegistry instance
            pid_file: Optional custom PID file path
            data_dir: Optional custom data directory
            skip_permissions: Skip permission checks (auto-approve)
            enable_task_manager: Enable TaskManager for task evaluation
            rpyc_port: RPyC server port configuration:
                - True (default): Auto-assign free port starting from 18861
                - int: Use specific port number
                - False/None: Disable RPyC server (NOT RECOMMENDED)
        """
        self.name = name
        self.config = config
        self.skip_permissions = skip_permissions
        self.enable_task_manager = enable_task_manager

        # Auto-assign RPyC port if enabled
        if rpyc_port is True:
            self.rpyc_port = _find_free_port()
        elif isinstance(rpyc_port, int):
            self.rpyc_port = rpyc_port
        else:
            self.rpyc_port = None

        # Paths
        self.pid_file = pid_file or Path(f"/tmp/deep_worker_{name}.pid")
        self.data_dir = data_dir or (Path("~/.deep_worker").expanduser() / name)

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.registry = registry or ProcessRegistry()
        self.agent = None  # Will be initialized in start()
        self.backend = None  # Will be initialized in start()
        self.session_manager: SessionManager | None = None
        self.task_manager = None  # Will be initialized in start() if enabled

        # RPyC components
        self.rpyc_server: ThreadedServer | None = None
        self.rpyc_service_instance: Any = None
        self._agent_event_loop: Any = None  # Event loop for RPyC callbacks

        # State
        self.running = False
        self.started_at: datetime | None = None  # Set when agent starts

    async def start(self) -> None:
        """Start the agent process.

        This is the main entry point for the agent.
        """
        try:
            # 0. Set started_at timestamp
            self.started_at = datetime.now()

            # 1. Create PID file
            await self._create_pid_file()

            # 2. Setup signal handlers
            self._setup_signal_handlers()

            # 3. Initialize session manager
            self.session_manager = SessionManager(data_dir=self.data_dir)

            # 4. Initialize agent with create_cli_agent
            try:
                import dotenv
                from pathlib import Path as PathLib

                msg = f"[AGENT_INIT] Initializing agent '{self.name}'...\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)

                # Load environment variables from ~/.deepagents/.env BEFORE importing config
                # This is critical because config.py initializes Settings at module level
                # and needs OPENAI_MODEL, OPENAI_BASE_URL, etc. to be set before import
                deepagents_env = PathLib.home() / ".deepagents" / ".env"
                if deepagents_env.exists():
                    dotenv.load_dotenv(deepagents_env)
                    msg = f"[AGENT_INIT] Loaded environment from {deepagents_env}\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)

                # Now import AFTER environment is loaded
                from deep_worker_cli.agent import create_cli_agent
                from deep_worker_cli.config import create_model

                # Create model using the same configuration as deepagents
                # This will use OPENAI_API_KEY + OPENAI_BASE_URL from ~/.deepagents/.env
                # Don't pass model name to create_model() - let it auto-detect from environment
                model = create_model()

                # Create the agent with auto-approve for testing
                self.agent, self.backend = create_cli_agent(
                    model=model,
                    assistant_id=self.name,
                    auto_approve=self.skip_permissions,
                    enable_memory=True,
                    enable_skills=True,
                    enable_shell=True,
                    project_memory_dir=Path(self.config.working_dir),
                )
                msg = f"[AGENT_INIT] Agent '{self.name}' initialized successfully (agent={self.agent})\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)

                # DEBUG: Log immediately after successful initialization
                msg = f"[AGENT_INIT] DEBUG: After agent init, checking task_manager (enabled={self.enable_task_manager})...\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)

                # Initialize TaskManager if enabled and agent is available
                if self.enable_task_manager and self.agent:
                    from deep_worker_cli.task_manager import TaskManager

                    msg = f"[AGENT_INIT] Initializing TaskManager for agent '{self.name}'...\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)

                    self.task_manager = TaskManager(
                        agent_name=self.name,
                        config=self.config,
                        session_manager=self.session_manager
                    )
                    await self.task_manager.start()

                    msg = f"[AGENT_INIT] TaskManager started (thread: {self.name}:tm)\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)

            except Exception as e:
                msg = f"[AGENT_INIT] Error initializing agent: {e}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                # Agent remains None, fallback to echo mode
                self.agent = None

            # DEBUG: Log before RPyC server start
            msg = f"[AGENT_INIT] Reached line before RPyC server start...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            # 5. Start RPyC server if port is configured
            msg = f"[AGENT_INIT] About to start RPyC server (port={self.rpyc_port})...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            if self.rpyc_port:
                self._start_rpyc_server()
            else:
                raise RuntimeError("RPyC port must be configured - IPC server has been removed")

            msg = f"[AGENT_INIT] RPyC server started successfully\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            # 6. Register in registry (ONLY after RPyC server is ready!)
            # This ensures clients can connect immediately after seeing the agent in registry
            msg = f"[AGENT_INIT] About to register in registry...\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            await self._register()

            msg = f"[AGENT_INIT] Registered successfully\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            print(msg, file=sys.stderr)

            # 7. Enter event loop
            self.running = True
            print(f"Agent '{self.name}' started (PID {os.getpid()})")
            print(f"RPyC port: {self.rpyc_port}")
            print(f"Working directory: {self.config.working_dir}")

            # Main event loop: wait for shutdown signal
            while self.running:
                await asyncio.sleep(1.0)

            print(f"Agent '{self.name}' event loop exiting...")

            # Graceful shutdown after event loop exits
            await self.shutdown()

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

        # 1. Stop TaskManager if enabled
        if self.task_manager:
            await self.task_manager.stop()

        # 2. Stop RPyC server
        if self.rpyc_server:
            print(f"[RPyC] Stopping RPyC server...")
            try:
                self.rpyc_server.close()
            except Exception as e:
                print(f"[RPyC] Error stopping RPyC server: {e}")

        # 3. Persist state
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.SHUTDOWN)

        # 4. Unregister
        await self._unregister()

        # 5. Cleanup files
        await self._remove_pid_file()

        print(f"Agent '{self.name}' stopped")

    async def _register(self) -> None:
        """Register agent in ProcessRegistry."""
        if self.started_at is None:
            self.started_at = datetime.now()

        # RPyC port is required for communication
        if self.rpyc_port is None:
            raise RuntimeError("RPyC port must be configured - RPyC is required for agent communication")

        process_info = AgentProcessInfo(
            name=self.name,
            pid=os.getpid(),
            pid_file=self.pid_file,
            working_dir=str(self.config.working_dir),
            started_at=self.started_at,
            rpyc_port=self.rpyc_port,  # RPyC port (REQUIRED for communication)
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

    def _start_rpyc_server(self) -> None:
        """Start RPyC server in a background thread.

        This method initializes the RPyC service and starts a ThreadedServer
        that listens on the configured rpyc_port.

        CRITICAL: This function MUST NOT return until the server is fully initialized
        and ready to accept connections. Returning prematurely will cause connection
        failures and race conditions in tests and client code.

        SECURITY REQUIREMENTS:
        ====================
        1. Server MUST listen ONLY on localhost (127.0.0.1), NEVER on 0.0.0.0
        2. No external network access is allowed - local trusted environment only
        3. Authentication is NOT implemented - relies on localhost isolation
        4. allow_pickle is enabled - safe ONLY because of localhost-only binding
        5. This is NOT suitable for untrusted networks or remote access

        The security model assumes:
        - All connections come from the local machine
        - The local environment is trusted
        - No malicious processes can connect to localhost
        """
        if not self.rpyc_port:
            return

        from deep_worker_cli.rpyc_service.agent_service import AgentService
        import time

        # SECURITY VALIDATION: Ensure we're binding to localhost only
        SAFE_HOSTNAME = "127.0.0.1"

        print(f"[RPyC] Starting RPyC server on {SAFE_HOSTNAME}:{self.rpyc_port}...")

        # Create service class factory that captures agent_process reference
        agent_process = self
        # Store the event loop for RPyC callbacks
        # Use get_event_loop() instead of get_running_loop() to ensure compatibility
        # with sync code that may run in different contexts
        self._agent_event_loop = asyncio.get_running_loop()

        class BoundAgentService(AgentService):
            """AgentService bound to specific AgentProcess instance."""

            def on_connect(self, conn: Any) -> None:
                """Called when client connects."""
                super().on_connect(conn)
                # Set agent_process and event_loop references for this connection
                self.set_agent_process(agent_process)
                self.set_event_loop(agent_process._agent_event_loop)

        # Extend safe_attrs with standard Python magic methods for transparent proxies
        # These methods are used by built-in Python operations (len(), reversed(), bytes(), etc.)
        # and must work transparently through RPyC NetRef proxies for dwcli interactive mode
        extended_safe_attrs = set(DEFAULT_CONFIG["safe_attrs"])
        extended_safe_attrs.update([
            # Iteration & sequence operations
            "__reversed__",      # reversed(obj) - used in interactive.py:_load_more_history()
            "__missing__",       # dict[key] when key not found (defaultdict)

            # Type conversion & representation
            "__bytes__",         # bytes(obj)
            "__fspath__",        # os.fspath(obj) for path-like objects
            "__complex__",       # complex(obj)

            # Introspection (safe, read-only)
            "__dir__",           # dir(obj)
            "__sizeof__",        # sys.getsizeof(obj)

            # Context managers (already have __enter__/__exit__)
            "__aenter__",        # async with obj
            "__aexit__",         # async with obj

            # Async iteration
            "__aiter__",         # async for
            "__anext__",         # async for

            # Descriptors (safe for data access)
            "__get__",           # descriptor protocol
            "__set__",           # descriptor protocol
            "__delete__",        # descriptor protocol

            # Copy operations
            "__copy__",          # copy.copy(obj)
            "__deepcopy__",      # copy.deepcopy(obj)

            # Pickling (already allow_pickle=True, so expose these)
            "__reduce__",        # pickle support
            "__reduce_ex__",     # pickle support
            "__getnewargs__",    # pickle support
            "__getnewargs_ex__", # pickle support
            "__getstate__",      # pickle support
            "__setstate__",      # pickle support
        ])

        # Create ThreadedServer with bound service
        # SECURITY: Listen ONLY on localhost (127.0.0.1), NEVER on external interfaces!
        # This prevents remote access and ensures the server is only accessible locally.
        self.rpyc_server = ThreadedServer(
            BoundAgentService,
            hostname=SAFE_HOSTNAME,  # CRITICAL: localhost only, no external access
            port=self.rpyc_port,
            protocol_config={
                # SECURITY WARNING: These settings are ONLY safe because we bind to localhost!
                # allow_public_attrs: Allows access to public attributes without authentication
                # allow_pickle: Enables pickle serialization (unsafe if exposed externally)
                # safe_attrs: Extended list of safe magic methods for transparent proxy operations
                # DO NOT use these settings if the server is accessible from untrusted networks!
                "allow_public_attrs": True,
                "allow_pickle": True,
                "safe_attrs": extended_safe_attrs,
            }
        )

        # Start server in background thread
        import threading
        server_thread = threading.Thread(
            target=self.rpyc_server.start,
            daemon=True,
            name=f"{self.name}-rpyc-server"
        )
        server_thread.start()

        # CRITICAL: Wait for ThreadedServer to be fully initialized
        # The server_thread.start() returns immediately and starts the server
        # in a background daemon thread. However, ThreadedServer.start() needs
        # time to bind() and listen() on the socket before accepting connections.
        #
        # ThreadedServer.start() is a BLOCKING call that runs the accept() loop,
        # but we're running it in a daemon thread, so we need to give it time to
        # actually reach the point where it's accepting connections.
        #
        # Simple approach: small fixed delay for thread to start accepting.
        # This is MORE RELIABLE than trying to connect, because connecting might
        # interfere with the server's ability to handle the first real client.
        print(f"[RPyC] Waiting for server thread to initialize on port {self.rpyc_port}...")
        time.sleep(0.2)  # 200ms should be more than enough for bind() + listen()
        print(f"[RPyC] Server thread initialized")

        print(f"[RPyC] RPyC server started on {SAFE_HOSTNAME}:{self.rpyc_port} (localhost only)")

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

    async def _handle_get_history(
        self, command: dict[str, Any], connection: Any = None
    ) -> dict[str, Any]:
        """Handle get_history command to retrieve conversation history.

        Args:
            command: Command dictionary with optional 'limit' and 'offset' keys
            connection: Legacy parameter (unused, kept for compatibility)

        Returns:
            Response dictionary with history data:
            {
                "success": True,
                "history": [list of message dicts],
                "total": total message count,
                "has_more": bool indicating if more messages exist
            }
        """
        import logging
        logger = logging.getLogger(__name__)

        limit = command.get("limit", 50)
        offset = command.get("offset", 0)

        logger.info(f"[GET_HISTORY] Starting get_history (limit={limit}, offset={offset})")

        if not self.agent:
            logger.error("[GET_HISTORY] Agent not initialized")
            return {
                "success": False,
                "error": "Agent not initialized",
            }

        logger.info(f"[GET_HISTORY] Agent is initialized: {self.agent}")

        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
            from langchain_core.runnables import RunnableConfig

            # Use agent name as thread_id (same as invoke)
            config: RunnableConfig = {"configurable": {"thread_id": self.name}}
            logger.info(f"[GET_HISTORY] Config created: {config}")

            # Get state from agent checkpoint
            logger.info(f"[GET_HISTORY] About to call agent.aget_state()...")
            state = await self.agent.aget_state(config)
            logger.info(f"[GET_HISTORY] Successfully got state: {state}")
            if not state or not state.values:
                return {
                    "success": True,
                    "history": [],
                    "total": 0,
                    "has_more": False,
                }

            messages = state.values.get("messages", [])
            total_count = len(messages)

            # Apply offset and limit for pagination
            start_idx = max(0, total_count - limit - offset)
            end_idx = total_count - offset if offset > 0 else total_count
            selected_messages = messages[start_idx:end_idx]

            # Convert LangChain messages to JSON-serializable format
            history = []
            for msg in selected_messages:
                msg_dict: dict[str, Any] = {
                    "type": msg.__class__.__name__,
                    "id": getattr(msg, "id", None),
                }

                if isinstance(msg, HumanMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    msg_dict["content"] = content

                elif isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, str):
                        msg_dict["content"] = content
                    elif isinstance(content, list):
                        # Handle content blocks
                        text_content = ""
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_content += block.get("text", "")
                            elif isinstance(block, str):
                                text_content += block
                        msg_dict["content"] = text_content
                    else:
                        msg_dict["content"] = str(content)

                    # Include tool calls if present
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        msg_dict["tool_calls"] = [
                            {
                                "id": tc.get("id"),
                                "name": tc.get("name"),
                                "args": tc.get("args", {}),
                            }
                            for tc in tool_calls
                        ]

                elif isinstance(msg, ToolMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    msg_dict["content"] = content
                    msg_dict["tool_call_id"] = getattr(msg, "tool_call_id", None)
                    msg_dict["name"] = getattr(msg, "name", "unknown")
                    msg_dict["status"] = getattr(msg, "status", "success")

                history.append(msg_dict)

            has_more = start_idx > 0

            return {
                "success": True,
                "history": history,
                "total": total_count,
                "has_more": has_more,
            }

        except Exception as e:
            import traceback
            print(f"[GET_HISTORY] Exception: {e}", file=sys.stderr)
            print(f"[GET_HISTORY] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            return {
                "success": False,
                "error": f"Failed to retrieve history: {str(e)}",
            }

    async def _handle_invoke(
        self,
        command: dict[str, Any],
        connection: Any = None,
        notify_callback: Any = None,
    ) -> dict[str, Any]:
        """Handle agent invocation command.

        Args:
            command: Command dictionary with 'message' and optional 'stream' keys
            connection: IPC connection for streaming responses
            notify_callback: Optional callable(msg: dict) that broadcasts a message
                to all subscribers (e.g., AgentService._notify_subscribers).
                When provided, the TaskManager evaluation loop calls this for each
                intermediate assistant response and each corrective prompt so that
                ALL connected dwcli instances see the full conversation in real time.
                This upholds the "agent is single source of truth" architecture
                principle (see ARCHITECTURE.md).

        Returns:
            Response dictionary with assistant message
        """
        message = command.get("message")
        stream = command.get("stream", False)

        if not message:
            return {
                "success": False,
                "error": "No message provided",
            }

        # Set status to RUNNING before execution
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.RUNNING)

        # ============================================================
        # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
        # ============================================================
        # Notify user message at the start (unless in streaming mode,
        # which handles this separately to ensure proper ordering).
        # This ensures ALL messages flow through the agent and are
        # broadcast to ALL connected clients.
        # ============================================================
        if not stream and notify_callback is not None:
            notify_callback({
                "type": "user",
                "role": "user",
                "content": message,
            })

        try:
            print(f"[INVOKE] Checking agent: self.agent={self.agent}", file=sys.stderr)
            if self.agent:
                from langchain_core.runnables import RunnableConfig
                from deep_worker_cli.process.status_callback import StatusTrackingCallback

                # Create status tracking callback
                status_callback = StatusTrackingCallback(session_manager=self.session_manager)

                # Configure thread_id for checkpointer (use agent name as thread_id)
                config: RunnableConfig = {
                    "configurable": {"thread_id": self.name},
                    "callbacks": [status_callback]
                }

                # Use actual LangGraph agent if available
                if stream:
                    # ============================================================
                    # STREAMING MODE: Full message streaming with all types
                    # ============================================================
                    # Uses LangGraph's stream_mode=["messages", "updates"] and
                    # subgraphs=True to capture ALL message types in real-time:
                    # - Text chunks from assistant
                    # - Tool calls
                    # - Tool results
                    # All messages are broadcast via notify_callback to uphold
                    # the "agent as single source of truth" architecture.
                    # ============================================================
                    from deep_worker_cli.process.stream_processor import process_message_chunk

                    # Notify user message at start (agent as source of truth)
                    if notify_callback is not None:
                        notify_callback({
                            "type": "user",
                            "role": "user",
                            "content": message,
                        })

                    # Tool call accumulation buffer for streaming chunks
                    tool_call_buffer: dict[str, Any] = {}
                    full_response = ""

                    # Stream with full message visibility
                    async for chunk in self.agent.astream(
                        {"messages": [{"role": "user", "content": message}]},
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                        config=config,
                    ):
                        # LangGraph stream format: (namespace, stream_mode, data)
                        if not isinstance(chunk, tuple) or len(chunk) != 3:
                            continue

                        namespace, current_stream_mode, data = chunk

                        # Handle MESSAGES stream - for actual message content
                        if current_stream_mode == "messages":
                            message_obj, metadata = data

                            # Process message with stream_processor
                            if notify_callback is not None:
                                await process_message_chunk(
                                    message=message_obj,
                                    metadata=metadata,
                                    notify_callback=notify_callback,
                                    tool_call_buffer=tool_call_buffer,
                                )

                            # Track full response for return value
                            if hasattr(message_obj, "content"):
                                content = message_obj.content
                                if isinstance(content, str):
                                    full_response = content

                        # Handle UPDATES stream - for interrupts and state changes
                        # (Currently we don't process updates, but this is where they would go)

                    # Set status back to IDLE after successful completion
                    if self.session_manager:
                        await self.session_manager.set_status(AgentStatus.IDLE)

                    # Send final response
                    return {
                        "success": True,
                        "response": {
                            "role": "assistant",
                            "content": full_response,
                        },
                    }
                else:
                    # Non-streaming mode: wait for complete response
                    # If TaskManager enabled, wrap in evaluation loop
                    if self.task_manager:
                        # Initialize task with user's message
                        await self.task_manager.init_task(
                            initial_request=message,
                            project_path=str(self.config.working_dir),
                            session_file_path=None  # Could add session file tracking
                        )

                        # TaskManager evaluation loop
                        max_iterations = 10  # Prevent infinite loops
                        iteration = 0
                        current_message = message

                        while iteration < max_iterations:
                            iteration += 1
                            print(f"[INVOKE] TaskManager iteration {iteration}/{max_iterations}", file=sys.stderr)

                            # Invoke agent
                            result = await self.agent.ainvoke(
                                {"messages": [{"role": "user", "content": current_message}]},
                                config
                            )

                            # Extract response
                            messages = result.get("messages", [])
                            if not messages:
                                break

                            last_msg = messages[-1]
                            content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")

                            # ============================================================
                            # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
                            # ============================================================
                            # Broadcast this intermediate assistant response to ALL
                            # subscribers (including dwcli TUI instances) so the full
                            # conversation is visible in real time.
                            # This is the fix for the TaskManager loop bypassing
                            # _notify_subscribers() for intermediate messages.
                            # ============================================================
                            if notify_callback is not None:
                                notify_callback(
                                    {"type": "assistant", "content": content, "role": "assistant"}
                                )

                            # Send to TaskManager for evaluation
                            corrective_prompt = await self.task_manager.on_subordinate_response(
                                response=content,
                                send_to_subordinate_callback=None  # Not used in loop approach
                            )

                            # If no corrective prompt, task is done or unable
                            if not corrective_prompt:
                                print(f"[INVOKE] TaskManager evaluation complete (done/unable)", file=sys.stderr)
                                break

                            # Broadcast TaskManager corrective prompt to ALL subscribers
                            # before sending it back to the agent. This ensures dwcli TUI
                            # shows the "TaskManager:> ..." message in real time.
                            if notify_callback is not None:
                                notify_callback(
                                    {
                                        "type": "user",
                                        "content": corrective_prompt,
                                        "role": "user",
                                    }
                                )

                            # Continue with corrective prompt
                            print(f"[INVOKE] TaskManager requesting corrections: {corrective_prompt[:100]}...", file=sys.stderr)
                            current_message = corrective_prompt

                        # Return final response
                        messages = result.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
                            role = getattr(last_msg, "type", "assistant")

                            # Notify final assistant response (TaskManager mode)
                            if notify_callback is not None:
                                notify_callback({
                                    "type": "assistant",
                                    "role": role,
                                    "content": content,
                                })

                            if self.session_manager:
                                await self.session_manager.set_status(AgentStatus.IDLE)

                            return {
                                "success": True,
                                "response": {
                                    "role": role,
                                    "content": content,
                                },
                            }
                        else:
                            return {
                                "success": False,
                                "error": "No messages returned from agent",
                            }

                    else:
                        # Standard invocation without TaskManager
                        result = await self.agent.ainvoke(
                            {"messages": [{"role": "user", "content": message}]},
                            config
                        )
                        print(f"[INVOKE] Got result: {result}", file=sys.stderr)
                        messages = result.get("messages", [])
                        print(f"[INVOKE] Extracted messages: {messages}", file=sys.stderr)
                        if messages:
                            last_msg = messages[-1]
                            # Handle both dict and LangChain Message objects
                            if isinstance(last_msg, dict):
                                role = last_msg.get("role", "assistant")
                                content = last_msg.get("content", "")

                                # Notify assistant response
                                if notify_callback is not None:
                                    notify_callback({
                                        "type": "assistant",
                                        "role": role,
                                        "content": content,
                                    })

                                # Set status back to IDLE after successful completion
                                if self.session_manager:
                                    await self.session_manager.set_status(AgentStatus.IDLE)

                                return {
                                    "success": True,
                                    "response": {
                                        "role": role,
                                        "content": content,
                                    },
                                }
                            elif hasattr(last_msg, "content"):
                                # LangChain Message object (AIMessage, HumanMessage, etc.)
                                role = getattr(last_msg, "type", "assistant")
                                content = last_msg.content
                                print(f"[INVOKE] Returning LangChain message: role={role}, content={content}", file=sys.stderr)

                                # Notify assistant response
                                if notify_callback is not None:
                                    notify_callback({
                                        "type": "assistant",
                                        "role": role,
                                        "content": content,
                                    })

                                # Set status back to IDLE after successful completion
                                if self.session_manager:
                                    await self.session_manager.set_status(AgentStatus.IDLE)

                                return {
                                    "success": True,
                                    "response": {
                                        "role": role,
                                        "content": content,
                                    },
                                }
                            else:
                                print(f"[INVOKE] last_msg is not dict or Message: {type(last_msg)}", file=sys.stderr)
                                return {
                                    "success": False,
                                    "error": f"Invalid message format: expected dict or Message, got {type(last_msg)}",
                                }
                        else:
                            print(f"[INVOKE] No messages in result", file=sys.stderr)
                            return {
                                "success": False,
                                "error": "No messages returned from agent",
                            }
            else:
                # Fallback: simple echo response if no agent available
                print(f"[INVOKE] Returning fallback echo (self.agent={self.agent})", file=sys.stderr)

                # Set status back to IDLE (fallback doesn't change status)
                if self.session_manager:
                    await self.session_manager.set_status(AgentStatus.IDLE)

                return {
                    "success": True,
                    "response": {
                        "role": "assistant",
                        "content": f"Echo: {message} (Agent not fully initialized)",
                    },
                }

        except Exception as e:
            import traceback
            print(f"[INVOKE] Exception occurred: {e}", file=sys.stderr)
            print(f"[INVOKE] Traceback:\n{traceback.format_exc()}", file=sys.stderr)

            # Set error status on exception
            if self.session_manager:
                await self.session_manager.set_error(f"Agent invocation failed: {str(e)}")

            return {
                "success": False,
                "error": f"Agent invocation failed: {str(e)}",
            }
