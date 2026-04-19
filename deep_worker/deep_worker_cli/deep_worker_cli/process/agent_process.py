"""Agent process implementation for deep_worker."""

import asyncio
import os
import signal
import socket
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import rpyc
from rpyc.core.protocol import DEFAULT_CONFIG
from rpyc.utils.async_server import AsyncioServer  # AsyncioServer from rpyc_async!

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentProcessInfo, AgentStatus

def _find_free_port(start_port: int = 0, max_attempts: int = 100) -> int:
    """Find a free port dynamically.

    ⚠️ CRITICAL: NEVER USE FIXED PORTS IN TESTS! ⚠️
    ============================================
    Using fixed ports (like 18861) causes race conditions and test failures
    when multiple agents/tests run concurrently. The port may be in use by:
    - Another test running in parallel
    - A previous test that didn't clean up
    - System services

    This function uses port 0 by default, which tells the OS to automatically
    select a free ephemeral port. This is the ONLY CORRECT approach for testing.

    DO NOT change start_port to a fixed value unless you have a very good reason
    and understand the consequences!

    Args:
        start_port: Starting port number (default: 0 = OS chooses automatically)
                   ⚠️ MUST be 0 for tests to avoid port conflicts!
        max_attempts: Maximum number of ports to try (default: 100)

    Returns:
        Free port number assigned by the OS

    Raises:
        RuntimeError: If no free port found (extremely rare with port=0)
    """
    # ⚠️ CRITICAL: Use port 0 for automatic OS port selection
    # This is the ONLY way to avoid port conflicts in tests!
    if start_port == 0:
        # Let the OS choose a free ephemeral port (RECOMMENDED for tests)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            # Get the port that OS assigned
            port = s.getsockname()[1]
            return port

    # Fallback: sequential search (NOT RECOMMENDED for tests)
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

    @staticmethod
    def _is_outgoing_inter_agent_echo(agent_name: str, response: Any) -> bool:
        """Return True iff ``response`` is this agent's own outgoing
        inter-agent echo (produced by ``SendMessageCompletionMiddleware``).

        An outgoing echo has the exact shape ``"agent:<self.name>:> ..."``
        at the very start of the string — it's not a reply to the user
        but a message addressed to a peer agent. TaskManager must skip
        evaluation of such turns because they aren't task responses.

        Args:
            agent_name: This agent's own name (typically ``self.name``).
            response: The ``full_response`` captured from the agent's
                stream / ainvoke output. Non-string inputs (``None``,
                tool-block lists, etc.) return False safely.

        Returns:
            True iff the string starts with the exact outgoing marker.
        """
        if not isinstance(response, str):
            return False
        marker = f"agent:{agent_name}:> "
        return response.startswith(marker)

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
        memory_only: bool = False,
        use_sqlite: bool = False,
        launch_config: dict | None = None,
        auto_approve_send_message: bool = False,
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
            memory_only: Use in-memory storage only (no persistence to database)
            use_sqlite: Force SQLite persistence (even in test environment)
            launch_config: Optional launch config dict persisted to
                session_metadata.json for --restart.
            auto_approve_send_message: If True, auto-approve
                `send_message_to_agent` tool calls without prompting,
                while keeping HITL gates for all other side-effect tools.
                Has no effect when ``skip_permissions`` is True
                (everything is already auto-approved).

        Raises:
            ValueError: If both memory_only and use_sqlite are True
        """
        # Validation: both flags cannot be True simultaneously
        if memory_only and use_sqlite:
            raise ValueError("Cannot specify both --memory-only and --use-sqlite")

        self.name = name
        self.config = config
        self.skip_permissions = skip_permissions
        self.auto_approve_send_message = auto_approve_send_message
        self.enable_task_manager = enable_task_manager
        self.memory_only = memory_only
        self.use_sqlite = use_sqlite
        self.launch_config = launch_config
        self._checkpointer_ctx = None  # For lifecycle management

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
        self.rpyc_server: AsyncioServer | None = None
        self.rpyc_service_instance: Any = None
        self._agent_event_loop: Any = None  # Event loop for RPyC callbacks

        # State
        self.running = False
        self.started_at: datetime | None = None  # Set when agent starts

        # Track how many messages we've broadcasted per thread
        # This prevents re-broadcasting old messages from LangGraph checkpointer
        # Similar to sent_count pattern in ClaudeCodeChatModel
        # Key: thread_id, Value: number of messages already broadcasted
        self._last_broadcasted_count: dict[str, int] = {}

        # Agent-side command state
        # These are used by slash commands (/clear, /tokens, etc.)
        self.current_thread_id: str = "default"  # Current conversation thread
        self.last_token_count: int = 0  # Token count from last agent response

        # Agent interruption state
        # These are used by exposed_interrupt_agent to cancel running tasks
        self.is_running: bool = False  # True when agent is executing
        self.current_worker_task: Any = None  # Current asyncio task for cancellation

        # Approval decisions cache (for backward compatibility)
        # Used by AgentService to store approval decisions when not using HITL framework
        # Maps approval_id -> decision dict (with "type" field: "approve", "reject", etc.)
        self._approval_decisions: dict[str, dict[str, Any]] = {}

        # Message subscribers (shared across ALL RPyC connections)
        # Each connection (TUI, headless, web UI) gets its own AgentService instance,
        # but they all share this set of subscribers to ensure ALL clients receive
        # messages regardless of which connection triggered the agent invocation.
        # This upholds the "agent is single source of truth" architecture principle.
        self._message_subscribers: set[Any] = set()

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

            # 3. Initialize session manager (with launch_config for post-mortem inspection)
            self.session_manager = SessionManager(
                data_dir=self.data_dir,
                launch_config=self.launch_config,
            )

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

                # deepagents is installed as a sibling package (see
                # pyproject.toml `[tool.uv.sources]` / packaging config).
                # Now import AFTER environment is loaded
                from deep_worker_cli.agent import create_cli_agent
                from deep_worker_cli.config import create_model
                from deep_worker_cli.tools import (
                    http_request,
                    fetch_url,
                    make_send_message_to_agent_tool,
                    web_search_delegation_tool,
                    read_image,
                )

                # Create model using configuration from AgentConfig
                # Pass provider and model explicitly from config
                model = create_model(
                    provider=self.config.provider,
                    model_name=self.config.model
                )

                # Set working_dir on model for MCP direct tool execution
                # (e.g. project_memory needs to know the project root)
                # Uses setattr() because create_model() returns BaseChatModel
                # but only ClaudeCodeChatModel has _working_dir attribute.
                if hasattr(model, '_working_dir'):
                    setattr(model, '_working_dir', str(self.config.working_dir))

                # Prepare web tools
                # - http_request and fetch_url work without API keys
                # - web_search_delegation_tool uses Claude Code SDK for universal web search
                # - send_message_to_agent: inter-agent messaging; sender name
                #   is baked into the factory from self.name so the LLM
                #   cannot forge it. See docs/DESIGN_SEND_MESSAGE_TO_AGENT_TOOL.md
                web_tools = [
                    http_request,
                    fetch_url,
                    web_search_delegation_tool,
                    read_image,
                    make_send_message_to_agent_tool(self.name),
                ]

                # Determine checkpointer based on flags and environment
                # Priority order:
                # 1. --use-sqlite flag: Use SQLite (overrides test detection)
                # 2. --memory-only flag: Use memory (overrides test detection)
                # 3. Test environment: Use memory (default for tests)
                # 4. Normal run: Use SQLite (default for production)
                from deep_worker_cli.utils import is_running_tests
                from deep_worker_cli.sessions import get_checkpointer

                checkpointer = None
                if self.use_sqlite:
                    # Explicit --use-sqlite flag: force SQLite even in tests
                    # Per-agent DB to avoid "database is locked" with concurrent agents
                    self._checkpointer_ctx = get_checkpointer(agent_name=self.name)
                    checkpointer = await self._checkpointer_ctx.__aenter__()
                elif self.memory_only:
                    # Explicit --memory-only flag: force memory even in production
                    checkpointer = None  # Will fall back to InMemorySaver
                elif is_running_tests():
                    # Test environment: default to memory (unless --use-sqlite specified)
                    checkpointer = None  # Will fall back to InMemorySaver
                else:
                    # Normal production run: use per-agent SQLite DB
                    # Each agent gets its own file (sessions_{name}.db) to avoid
                    # "database is locked" errors from concurrent writes.
                    self._checkpointer_ctx = get_checkpointer(agent_name=self.name)
                    checkpointer = await self._checkpointer_ctx.__aenter__()

                # Create the agent with auto-approve for testing

                self.agent, self.backend = create_cli_agent(
                    model=model,
                    assistant_id=self.name,
                    tools=web_tools,
                    auto_approve=self.skip_permissions,
                    auto_approve_send_message=self.auto_approve_send_message,
                    enable_memory=True,
                    enable_skills=True,
                    enable_shell=True,
                    project_memory_dir=Path(self.config.working_dir),
                    checkpointer=checkpointer,
                )

            except Exception as e:
                msg = f"[AGENT_INIT] Error initializing agent: {e}\n"
                sys.stdout.write(msg)
                sys.stdout.flush()
                print(msg, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                # Agent remains None, fallback to echo mode
                self.agent = None

            # Initialize TaskManager if enabled and agent is available
            # This is done OUTSIDE the agent creation try/except so that
            # TaskManager failures don't prevent the main agent from working
            if self.enable_task_manager and self.agent:
                try:
                    from deep_worker_cli.task_manager import TaskManager

                    # Pass the actual checkpointer instance (not context manager)
                    # TaskManager MUST use same checkpointer type as main agent:
                    # - InMemorySaver if main agent uses InMemorySaver (test mode)
                    # - AsyncSqliteSaver if main agent uses AsyncSqliteSaver (production)
                    self.task_manager = TaskManager(
                        agent_name=self.name,
                        config=self.config,
                        session_manager=self.session_manager,
                        checkpointer=checkpointer,  # Same as main agent
                    )
                    await self.task_manager.start()

                    msg = f"[AGENT_INIT] TaskManager started (manager_id: {self.name}:tm)\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)

                except Exception as e:
                    msg = f"[AGENT_INIT] Error initializing TaskManager: {e}\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    print(msg, file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    # TaskManager remains None, agent continues without TaskManager
                    self.task_manager = None

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

            if self.rpyc_port is not None:
                await self._start_rpyc_server()  # Now async!
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

    async def cleanup(self) -> None:
        """Clean up resources, including checkpointer.

        This method should be called on agent shutdown to properly close
        database connections and release resources.
        """
        if self._checkpointer_ctx is not None:
            try:
                await self._checkpointer_ctx.__aexit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error closing checkpointer: {e}", file=sys.stderr)
            finally:
                self._checkpointer_ctx = None

    async def shutdown(self) -> None:
        """Graceful shutdown of the agent."""
        if not self.running:
            return

        print(f"Shutting down agent '{self.name}'...")
        self.running = False

        # 1. Stop TaskManager if enabled
        if self.task_manager:
            await self.task_manager.stop()

        # 2. Stop AsyncioServer
        if self.rpyc_server:
            print(f"[RPyC] Stopping AsyncioServer...")
            try:
                await self.rpyc_server.close()  # AsyncioServer.close() is async
            except Exception as e:
                print(f"[RPyC] Error stopping AsyncioServer: {e}")

        # 3. Persist state
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.SHUTDOWN)

        # 4. Cleanup checkpointer
        await self.cleanup()

        # 5. Unregister
        await self._unregister()

        # 6. Cleanup files
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

    async def _start_rpyc_server(self) -> None:
        """Start RPyC AsyncioServer (native asyncio, no threads!).

        MIGRATED FROM ThreadedServer TO AsyncioServer for:
        - True bidirectional async support (no deadlocks!)
        - 156x better performance (event-driven vs polling)
        - Single event loop (no thread synchronization needed)
        - Native async/await integration

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

        # SECURITY VALIDATION: Ensure we're binding to localhost only
        SAFE_HOSTNAME = "127.0.0.1"

        print(f"[RPyC] Starting AsyncioServer on {SAFE_HOSTNAME}:{self.rpyc_port}...")

        # Create service class factory that captures agent_process reference
        agent_process = self
        # Store the event loop for RPyC callbacks (we're already in the loop!)
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

        # Create AsyncioServer (native asyncio - no threads!)
        # SECURITY: Listen ONLY on localhost (127.0.0.1), NEVER on external interfaces!
        # This prevents remote access and ensures the server is only accessible locally.
        # ============================================================================
        # CRITICAL: NEVER ADD TIMEOUTS TO RPyC SERVER!
        # ============================================================================
        # deep-worker agents work with dwcli for DAYS or WEEKS continuously.
        # RPyC connection lifetime timeouts are ABSOLUTELY FORBIDDEN.
        #
        # FORBIDDEN:
        # - "sync_request_timeout": <any value>  # NEVER ADD THIS
        # - Any other timeout on connection lifetime
        #
        # Agent may be busy for extended periods with legitimate work:
        # - TaskManager evaluation loops (minutes to hours)
        # - Complex multi-step operations
        # - Large file processing
        #
        # If you add a timeout, the connection WILL drop during normal operation.
        # This is a BUG, not a feature!
        # ============================================================================
        self.rpyc_server = AsyncioServer(
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
                # NO TIMEOUT - connection can live for days/weeks
            }
        )

        # Start AsyncioServer (non-blocking, returns immediately)
        await self.rpyc_server.start()

        # Update port if it was 0 (auto-assigned)
        if self.rpyc_port == 0:
            self.rpyc_port = self.rpyc_server.listener.getsockname()[1]
            print(f"[RPyC] Auto-assigned port: {self.rpyc_port}")

        print(f"[RPyC] AsyncioServer started on {SAFE_HOSTNAME}:{self.rpyc_port} (localhost only)")
        print(f"[RPyC] ✅ Native asyncio - no threads, true bidirectional async, 156x faster!")

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

        # ⚠️ NO ARTIFICIAL LIMITS POLICY (per design doc
        # DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §6).
        # The default is "return everything" — the only ceiling is the
        # agent's own context window. ``limit`` may be:
        #   - absent / None / 0 / negative → return all messages
        #   - positive int N → return last N messages (legacy contract)
        raw_limit = command.get("limit", None)
        try:
            int_limit = int(raw_limit) if raw_limit is not None else 0
        except (TypeError, ValueError):
            int_limit = 0
        # 0 or negative ⇒ no cap
        limit: int | None = int_limit if int_limit > 0 else None
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
            config: RunnableConfig = {
                "configurable": {"thread_id": self.name},
                "metadata": {
                    "assistant_id": self.name,
                    "agent_name": self.name,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            }
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

            # Apply offset and limit for pagination.
            # ``limit is None`` ⇒ return everything (no cap).
            end_idx = total_count - offset if offset > 0 else total_count
            if limit is None:
                start_idx = 0
            else:
                start_idx = max(0, end_idx - limit)
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

    async def _handle_get_last_response(self) -> dict[str, Any]:
        """Get the last assistant (AIMessage) response from conversation history.

        Returns only the last AIMessage with non-empty text content,
        skipping tool-only messages. This is much more efficient than
        loading full history when only the latest response is needed.

        Returns:
            Response dictionary:
            {
                "success": True,
                "last_response": {
                    "type": "AIMessage",
                    "id": "...",
                    "content": "...",
                    "tool_calls": [...]  # optional
                }
            }
            Or {"success": True, "last_response": None} if no AIMessage found.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.agent:
            return {"success": False, "error": "Agent not initialized"}

        try:
            from langchain_core.messages import AIMessage
            from langchain_core.runnables import RunnableConfig

            config: RunnableConfig = {
                "configurable": {"thread_id": self.name},
                "metadata": {
                    "assistant_id": self.name,
                    "agent_name": self.name,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            }

            state = await self.agent.aget_state(config)
            if not state or not state.values:
                return {"success": True, "last_response": None}

            messages = state.values.get("messages", [])

            # Walk backwards to find last AIMessage with text content
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue

                # Extract text content
                content = msg.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif isinstance(block, str):
                            text += block
                else:
                    text = str(content)

                # Skip AIMessages with empty text (tool-call-only messages)
                if not text.strip():
                    continue

                msg_dict: dict[str, Any] = {
                    "type": "AIMessage",
                    "id": getattr(msg, "id", None),
                    "content": text,
                }

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

                return {"success": True, "last_response": msg_dict}

            return {"success": True, "last_response": None}

        except Exception as e:
            logger.error(f"Error getting last response: {e}")
            return {"success": False, "error": f"Failed to get last response: {str(e)}"}

    async def _handle_invoke(
        self,
        command: dict[str, Any],
        connection: Any = None,
        notify_callback: Any = None,
        approval_callback: Any = None,
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
        stream = command.get("stream", True)  # Default to streaming to match deepagents

        if not message:
            return {
                "success": False,
                "error": "No message provided",
            }

        # Set status to RUNNING before execution
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.RUNNING)

        # Track execution state for interruption support
        self.is_running = True
        self.current_worker_task = asyncio.current_task()

        # ============================================================
        # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
        # ============================================================
        # Notify user message at the start (unless in streaming mode,
        # which handles this separately to ensure proper ordering).
        # This ensures ALL messages flow through the agent and are
        # broadcast to ALL connected clients.
        # ============================================================
        if not stream and notify_callback is not None:
            await notify_callback({
                "type": "user",
                "role": "user",
                "content": message,
            })

        try:
            if self.agent:
                from langchain_core.runnables import RunnableConfig
                from deep_worker_cli.process.status_callback import StatusTrackingCallback

                # Create status tracking callback
                status_callback = StatusTrackingCallback(session_manager=self.session_manager)

                # Configure thread_id for checkpointer (use agent name as thread_id)
                config: RunnableConfig = {
                    "configurable": {"thread_id": self.name},
                    "callbacks": [status_callback],
                    "metadata": {
                        "assistant_id": self.name,
                        "agent_name": self.name,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
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
                    #
                    # If TaskManager is enabled, wrap streaming in evaluation loop
                    # to allow corrective prompts.
                    # ============================================================
                    from deep_worker_cli.process.stream_processor import process_message_chunk

                    # Initialize TaskManager if enabled
                    if self.task_manager:
                        await self.task_manager.init_task(
                            initial_request=message,
                            project_path=str(self.config.working_dir),
                            session_file_path=None
                        )

                    # TaskManager evaluation loop or single iteration
                    # TaskManager has NO iteration limit — it runs until done/unable.
                    iteration = 0
                    current_message = message
                    full_response = ""

                    while True:
                        iteration += 1
                        if self.task_manager:
                            print(f"[STREAM] TaskManager iteration {iteration}", file=sys.stderr)

                        # Notify user message at start (agent as source of truth)
                        # For TaskManager: only send on first iteration, subsequent corrective
                        # prompts are already sent at the end of previous iteration (lines 1040-1044)
                        if notify_callback is not None and iteration == 1:
                            await notify_callback({
                                "type": "user",
                                "role": "user",
                                "content": current_message,
                            })

                        # Tool call accumulation buffer for streaming chunks
                        tool_call_buffer: dict[str, Any] = {}
                        full_response = ""

                        # DEBUG: Log to track chunks arriving from LangGraph
                        # ALWAYS enabled for debugging (no env var check)
                        import os
                        import time as time_mod
                        chunk_count = 0
                        debug_enabled = True  # ALWAYS log for debugging

                        if debug_enabled:
                            with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                f.write(f"\n{'='*80}\n")
                                f.write(f"[{time_mod.time():.3f}] STARTING astream loop\n")
                                if self.task_manager:
                                    f.write(f"  TaskManager iteration: {iteration}\n")
                                f.write(f"  stream_mode=['messages', 'updates']\n")
                                f.write(f"  durability='exit'\n")
                                f.write(f"  Env DWCLI_DEBUG_ASTREAM={os.environ.get('DWCLI_DEBUG_ASTREAM')}\n")
                                f.write(f"{'='*80}\n")

                        # Stream with full message visibility
                        async for chunk in self.agent.astream(
                            {"messages": [{"role": "user", "content": current_message}]},
                            stream_mode=["messages", "updates"],
                            subgraphs=True,
                            config=config,
                            durability="exit",  # Match deepagents - required for immediate streaming
                        ):
                            chunk_count += 1

                            # DEBUG: Log every chunk received
                            if debug_enabled:
                                with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                    f.write(f"\n[{time_mod.time():.3f}] CHUNK #{chunk_count} received:\n")
                                    f.write(f"  Type: {type(chunk)}\n")
                                    f.write(f"  Is tuple: {isinstance(chunk, tuple)}\n")
                                    if isinstance(chunk, tuple):
                                        f.write(f"  Length: {len(chunk)}\n")
                                        if len(chunk) >= 3:
                                            f.write(f"  namespace: {chunk[0]}\n")
                                            f.write(f"  stream_mode: {chunk[1]}\n")
                                            f.write(f"  data type: {type(chunk[2])}\n")

                            # LangGraph stream format: (namespace, stream_mode, data)
                            if not isinstance(chunk, tuple) or len(chunk) != 3:
                                if debug_enabled:
                                    with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                        f.write(f"  ⚠ SKIPPING: Invalid format\n")
                                continue

                            namespace, current_stream_mode, data = chunk

                            # DEBUG: Log stream mode
                            if debug_enabled:
                                with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                    f.write(f"  Stream mode: {current_stream_mode}\n")

                            # Handle MESSAGES stream - for actual message content
                            if current_stream_mode == "messages":
                                message_obj, metadata = data

                                # DEBUG: Log message details
                                if debug_enabled:
                                    with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                        f.write(f"  MESSAGE chunk:\n")
                                        f.write(f"    Message type: {type(message_obj)}\n")
                                        f.write(f"    Has content: {hasattr(message_obj, 'content')}\n")
                                        if hasattr(message_obj, 'content'):
                                            content = message_obj.content
                                            f.write(f"    Content type: {type(content)}\n")
                                            if isinstance(content, list):
                                                f.write(f"    Content blocks: {len(content)}\n")
                                                for i, block in enumerate(content):
                                                    if isinstance(block, dict):
                                                        block_type = block.get('type', 'unknown')
                                                        f.write(f"      Block {i}: type={block_type}\n")
                                                        if block_type in {'tool_call_chunk', 'tool_call'}:
                                                            f.write(f"        ✅ TOOL_CALL_CHUNK/CALL DETECTED!\n")
                                                            f.write(f"        id={block.get('id')}\n")
                                                            f.write(f"        name={block.get('name')}\n")
                                        f.write(f"    Calling process_message_chunk...\n")

                                # Process message with stream_processor
                                if notify_callback is not None:
                                    await process_message_chunk(
                                        message=message_obj,
                                        metadata=metadata,
                                        notify_callback=notify_callback,
                                        tool_call_buffer=tool_call_buffer,
                                    )

                                    if debug_enabled:
                                        with open("/tmp/dwcli_astream_debug.log", "a") as f:
                                            f.write(f"    ✓ process_message_chunk completed\n")

                                # Track full response for return value (accumulate text chunks)
                                if hasattr(message_obj, "content"):
                                    content = message_obj.content
                                    if isinstance(content, str):
                                        full_response += content

                            # Handle UPDATES stream - for interrupts and state changes
                            # (Currently we don't process updates, but this is where they would go)

                        # CRITICAL: Flush any remaining buffered text chunks
                        # This ensures all batched text is sent before returning
                        from deep_worker_cli.process.stream_processor import flush_all_text_buffers
                        await flush_all_text_buffers()

                        # If TaskManager enabled, evaluate response and check for corrective prompt
                        if self.task_manager:
                            # NOTE: In streaming mode, assistant response is already broadcast
                            # via stream_processor chunks. No need to send it again here.

                            # Skip TaskManager for outgoing inter-agent echoes:
                            # the turn's "final response" is a message addressed to another
                            # agent (produced by SendMessageCompletionMiddleware), not a
                            # reply to the user's task. Evaluating it would make the manager
                            # LLM complain that the subordinate didn't do the work.
                            # See docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md §5.4.
                            if self._is_outgoing_inter_agent_echo(self.name, full_response):
                                print(
                                    f"[STREAM] Skipping TaskManager: outgoing inter-agent message",
                                    file=sys.stderr,
                                )
                                break

                            corrective_prompt = await self.task_manager.on_subordinate_response(
                                response=full_response,
                                send_to_subordinate_callback=None
                            )

                            # If no corrective prompt, task is done or unable
                            if not corrective_prompt:
                                print(f"[STREAM] TaskManager evaluation complete (done/unable)", file=sys.stderr)
                                break

                            # Broadcast TaskManager corrective prompt to ALL subscribers
                            # BEFORE sending it back to the agent. This ensures dwcli TUI
                            # shows the "TaskManager:> ..." message in real time.
                            if notify_callback is not None:
                                try:
                                    await notify_callback({
                                        "type": "user",
                                        "content": corrective_prompt,
                                        "role": "user",
                                    })
                                except Exception as e:
                                    # Don't let notification failures stop TaskManager loop
                                    print(f"[STREAM] Warning: Failed to notify subscribers of TaskManager prompt: {e}", file=sys.stderr)

                            # Continue with corrective prompt
                            print(f"[STREAM] TaskManager requesting corrections: {corrective_prompt[:100]}...", file=sys.stderr)
                            current_message = corrective_prompt
                        else:
                            # No TaskManager - exit after first iteration
                            break

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
                        # NO iteration limit — TaskManager runs until done/unable.
                        iteration = 0
                        current_message = message

                        while True:
                            iteration += 1
                            print(f"[INVOKE] TaskManager iteration {iteration}", file=sys.stderr)

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
                                await notify_callback(
                                    {"type": "assistant", "content": content, "role": "assistant"}
                                )

                            # Skip TaskManager for outgoing inter-agent echoes:
                            # the turn's "final response" is a message addressed to another
                            # agent (produced by SendMessageCompletionMiddleware), not a
                            # reply to the user's task.
                            # See docs/DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md §5.4.
                            if self._is_outgoing_inter_agent_echo(self.name, content):
                                print(
                                    f"[INVOKE] Skipping TaskManager: outgoing inter-agent message",
                                    file=sys.stderr,
                                )
                                break

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
                                await notify_callback(
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
                                await notify_callback({
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
                        # CRITICAL FIX: In non-streaming mode, we must wait for FINAL response
                        # If agent has interrupts (tool approvals), ainvoke() may return early
                        # with empty content. We need to keep invoking until we get final answer.

                        max_iterations = 20  # Prevent infinite loops
                        iteration = 0
                        final_content = ""

                        # Initial ainvoke with user message
                        result = await self.agent.ainvoke(
                            {"messages": [{"role": "user", "content": message}]},
                            config
                        )
                        print(f"[INVOKE] Initial ainvoke: Got result with {len(result.get('messages', []))} messages", file=sys.stderr)

                        while iteration < max_iterations:
                            iteration += 1

                            messages = result.get("messages", [])
                            if not messages:
                                print(f"[INVOKE] No messages in result", file=sys.stderr)
                                break

                            # CRITICAL FIX: Notify ONLY NEW messages (not entire history)
                            # This ensures tool_call and tool_result messages are visible to users
                            # WITHOUT re-broadcasting old messages from previous queries
                            # Similar to streaming mode behavior in stream_processor.py
                            # Pattern: analogous to sent_count in ClaudeCodeChatModel
                            if notify_callback is not None:
                                # Extract thread_id from config
                                thread_id = config.get("configurable", {}).get("thread_id", "default")

                                # Determine which messages are NEW (not yet broadcasted)
                                last_broadcasted = self._last_broadcasted_count.get(thread_id, 0)
                                new_messages = messages[last_broadcasted:]

                                print(f"[INVOKE] Iteration {iteration}: Broadcasting {len(new_messages)} new messages (total={len(messages)}, last_broadcasted={last_broadcasted})", file=sys.stderr)

                                # Broadcast ONLY new messages
                                for msg in new_messages:
                                    # Extract message data
                                    if isinstance(msg, dict):
                                        msg_role = msg.get("role", "")
                                        msg_content = msg.get("content", "")
                                        msg_tool_calls = msg.get("tool_calls")
                                    elif hasattr(msg, "type"):
                                        msg_role = msg.type
                                        msg_content = getattr(msg, "content", "")
                                        msg_tool_calls = getattr(msg, "tool_calls", None)
                                    else:
                                        continue

                                    # Send tool_call notifications
                                    if msg_tool_calls:
                                        for tool_call in msg_tool_calls:
                                            if isinstance(tool_call, dict):
                                                await notify_callback({
                                                    "type": "tool_call",
                                                    "name": tool_call.get("name", ""),
                                                    "args": tool_call.get("args", {}),
                                                    "id": tool_call.get("id", ""),
                                                })
                                            else:
                                                # LangChain ToolCall object
                                                await notify_callback({
                                                    "type": "tool_call",
                                                    "name": getattr(tool_call, "name", ""),
                                                    "args": getattr(tool_call, "args", {}),
                                                    "id": getattr(tool_call, "id", ""),
                                                })

                                    # Send tool_result notifications
                                    if msg_role == "tool":
                                        # Extract status from ToolMessage (default: "success")
                                        # LangChain ToolMessage has status attribute ("success" or "error")
                                        if isinstance(msg, dict):
                                            msg_status = msg.get("status", "success")
                                        else:
                                            msg_status = getattr(msg, "status", "success")

                                        await notify_callback({
                                            "type": "tool_result",
                                            "content": msg_content,
                                            "status": msg_status,
                                            "tool_call_id": (
                                                msg.get("tool_call_id", "")
                                                if isinstance(msg, dict)
                                                else getattr(msg, "tool_call_id", "")
                                            ),
                                        })

                                # Update last_broadcasted_count AFTER broadcasting
                                # This ensures we don't re-broadcast these messages in future iterations
                                self._last_broadcasted_count[thread_id] = len(messages)

                            last_msg = messages[-1]

                            # Check if there are interrupts pending
                            has_interrupt = "__interrupt__" in result and result["__interrupt__"]

                            # Extract content from last message
                            if isinstance(last_msg, dict):
                                content = last_msg.get("content", "")
                                role = last_msg.get("role", "")
                            elif hasattr(last_msg, "content"):
                                content = last_msg.content
                                role = getattr(last_msg, "type", "")
                            else:
                                content = ""
                                role = ""

                            # Check if this message has tool calls
                            has_tool_calls = False
                            if isinstance(last_msg, dict):
                                has_tool_calls = bool(last_msg.get("tool_calls"))
                            elif hasattr(last_msg, "tool_calls"):
                                has_tool_calls = bool(last_msg.tool_calls)

                            print(f"[INVOKE] Iteration {iteration}: content='{content[:100] if content else '(empty)'}', role={role}, has_tool_calls={has_tool_calls}, has_interrupt={has_interrupt}", file=sys.stderr)

                            # If we have real content and no pending tool calls AND it's from assistant, this is final response
                            # CRITICAL: Must check role in ('assistant', 'ai') to avoid treating tool results as final response!
                            # LangGraph uses 'ai' for AIMessage.type, while dicts may use 'assistant'
                            if content and not has_tool_calls and role in ('assistant', 'ai'):
                                final_content = content
                                print(f"[INVOKE] Got final response: {content[:100]}...", file=sys.stderr)
                                break

                            # If there's an interrupt, handle it properly:
                            # 1. Get approval decision (via callback or auto-approve)
                            # 2. Resume with Command(resume=...) — do NOT re-send HumanMessage
                            if has_interrupt:
                                interrupts = result["__interrupt__"]
                                print(f"[INVOKE] Interrupt detected: {len(interrupts)} pending approvals", file=sys.stderr)

                                # NOTE: Do NOT broadcast here — exposed_request_approval()
                                # handles broadcasting with its own approval_id.

                                # Get decisions: either from callback or auto-approve
                                if approval_callback is not None:
                                    hitl_response = await approval_callback(interrupts)
                                else:
                                    # Auto-approve all (no callback = headless/web without approval UI)
                                    hitl_response = {}
                                    for intr in interrupts:
                                        actions = intr.value.get("action_requests", [])
                                        hitl_response[intr.id] = {
                                            "decisions": [{"type": "approve"} for _ in actions]
                                        }
                                    print(f"[INVOKE] Auto-approving {len(interrupts)} interrupts (no approval callback)", file=sys.stderr)

                                # Resume graph with decisions (NOT with HumanMessage!)
                                from langgraph.types import Command
                                resume_input = Command(resume=hitl_response)
                                result = await self.agent.ainvoke(resume_input, config)
                                print(f"[INVOKE] Resumed after interrupt, got {len(result.get('messages', []))} messages", file=sys.stderr)

                                # Re-evaluate the new result (loop continues from evaluation)
                                continue

                            # If we have tool calls but no interrupt, they should have been
                            # executed by ainvoke already. Break to avoid re-sending.
                            if has_tool_calls:
                                print(f"[INVOKE] Tool calls detected without interrupt — ainvoke should have handled them", file=sys.stderr)
                                break

                            # If we got here with empty content and no tool calls,
                            # the agent finished but has nothing to say (shouldn't happen normally)
                            if not content:
                                print(f"[INVOKE] Empty response with no tool calls at iteration {iteration}", file=sys.stderr)
                                break

                        if iteration >= max_iterations:
                            print(f"[INVOKE] Max iterations reached without final response", file=sys.stderr)
                            return {
                                "success": False,
                                "error": "Agent exceeded maximum iterations without producing final response"
                            }

                        if not final_content:
                            print(f"[INVOKE] No final content found after {iteration} iterations", file=sys.stderr)
                            return {
                                "success": False,
                                "error": "Agent did not produce a response"
                            }

                        # ============================================================
                        # ARCHITECTURE PRINCIPLE: AGENT IS THE SINGLE SOURCE OF TRUTH
                        # ============================================================
                        # In non-streaming mode WITHOUT TaskManager, we must explicitly
                        # broadcast the final assistant response to ensure ALL connected
                        # clients (TUI, web UI, etc.) see it via real-time callbacks.
                        # This is critical for maintaining consistency across all subscribers.
                        # ============================================================
                        if notify_callback is not None:
                            await notify_callback({
                                "type": "assistant",
                                "role": "assistant",
                                "content": final_content,
                            })

                        # Set status back to IDLE after successful completion
                        if self.session_manager:
                            await self.session_manager.set_status(AgentStatus.IDLE)

                        return {
                            "success": True,
                            "response": {
                                "role": "assistant",
                                "content": final_content,
                            },
                        }
            else:
                # STRICT FAILURE: Agent MUST be initialized, NO SILENT FALLBACKS
                error_msg = (
                    "CRITICAL: Agent not initialized! Cannot process requests.\n"
                    "This indicates a fatal initialization error that must be fixed.\n"
                    f"Agent state: {self.agent}\n"
                    "Check agent process logs for initialization errors."
                )
                print(f"[INVOKE] {error_msg}", file=sys.stderr)

                if self.session_manager:
                    await self.session_manager.set_error("Agent not initialized")

                raise RuntimeError(error_msg)

        except asyncio.CancelledError:
            # Task was cancelled (interrupted) — this is a NORMAL state
            # transition initiated by the user, not an error. The agent
            # process is fully alive and ready to accept the next user
            # message, so we land in WAITING_INPUT (not ERROR).
            #
            # Putting the agent in ERROR here was a bug: clients refuse
            # to send new messages to an agent in error state, requiring
            # a manual restart after every interrupt.
            print("[INVOKE] Agent task cancelled (interrupted)", file=sys.stderr)

            # Notify cancellation via callback (if provided)
            if notify_callback is not None:
                await notify_callback({
                    "type": "agent_interrupted",
                    "role": "system",
                    "content": "Interrupted",
                })

            # Transition to WAITING_INPUT — interrupted, awaiting user input
            if self.session_manager:
                await self.session_manager.set_status(AgentStatus.WAITING_INPUT)

            # Re-raise to properly handle cancellation
            raise

        except Exception as e:
            import traceback
            print(f"[INVOKE] Exception occurred: {e}", file=sys.stderr)
            print(f"[INVOKE] Traceback:\n{traceback.format_exc()}", file=sys.stderr)

            # CRITICAL: Notify error to TUI so users see what went wrong
            # This matches deepagents architecture where errors are displayed via ErrorMessage
            error_msg = f"Agent error: {str(e)}"
            if notify_callback is not None:
                try:
                    await notify_callback({
                        "type": "error",
                        "role": "system",
                        "content": error_msg,
                    })
                except Exception as notify_err:
                    print(f"[INVOKE] Warning: Failed to notify error to TUI: {notify_err}", file=sys.stderr)

            # Set error status on exception
            if self.session_manager:
                await self.session_manager.set_error(f"Agent invocation failed: {str(e)}")

            return {
                "success": False,
                "error": f"Agent invocation failed: {str(e)}",
            }

        finally:
            # Always reset execution state
            self.is_running = False
            self.current_worker_task = None
