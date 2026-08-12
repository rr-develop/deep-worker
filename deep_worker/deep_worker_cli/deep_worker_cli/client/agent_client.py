"""Agent client SDK for connecting to deep_worker agents via RPyC.

This client ONLY supports RPyC connections. Unix sockets are NOT supported.
All communication is done via native rpyc with AsyncioServer.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deep_worker_cli.exceptions import AgentNotFoundError, IPCConnectionError
from deep_worker_cli.process.registry import ProcessRegistry

# ============================================================================
# rpyc-async is a regular PyPI dependency (>=1.0.0) providing the
# AsyncioServer-based protocol (rpyc_async.core.async_connect and
# rpyc_async.utils.classic.obtain). The previous sys.path injection of a
# local rpyc_async checkout was removed: on a developer machine it shadowed
# the PyPI package with sources, making "clean env" wheel installs test the
# wrong code.
# ============================================================================

import rpyc_async as rpyc
from rpyc_async.core.async_connect import async_connect
from rpyc_async.utils.classic import obtain

if TYPE_CHECKING:
    from deep_worker_cli.basesession.protocol_types import SessionInfo


class AgentClient:
    """
    Client SDK for connecting to deep_worker agent processes via RPyC.

    ONLY RPyC connections are supported. Unix sockets are NOT supported.

    Usage:
        client = AgentClient("project-alpha")
        await client.connect()  # Connects via RPyC

        response = await client.invoke_agent("Hello")

        await client.disconnect()
    """

    # Bound for the ENTIRE connect (TCP sock_connect + eager RPyC
    # HANDLE_GETROOT handshake). MUST cover the handshake, not just the TCP
    # step — see the banner in ``connect()`` and
    # test_agent_client_connect_bounded.py.
    _CONNECT_TIMEOUT: float = 10.0

    def __init__(
        self,
        agent_name: str,
        registry: ProcessRegistry | None = None,
    ):
        """Initialize the agent client.

        Args:
            agent_name: Name of the agent to connect to
            registry: Optional ProcessRegistry instance. If None, uses default.
        """
        self.agent_name = agent_name
        self.registry = registry or ProcessRegistry()
        self.client_id = str(uuid.uuid4())
        self.connected = False

        # RPyC is the ONLY connection method
        self._conn: rpyc.Connection | None = None
        self._root: Any = None  # RPyC root reference (obtained before enable_asyncio_serving)

    async def connect(self) -> None:
        """Connect to the agent process via RPyC.

        RPyC is the ONLY supported connection method.

        Raises:
            AgentNotFoundError: If agent is not found in registry
            IPCConnectionError: If connection fails or RPyC port not configured
        """
        # Lookup agent in registry
        agent_info = self.registry.lookup(self.agent_name)
        if not agent_info:
            raise AgentNotFoundError(
                f"Agent '{self.agent_name}' not found. "
                f"Start it with: deep-worker {self.agent_name}"
            )

        # RPyC port is REQUIRED
        if not agent_info.rpyc_port:
            raise IPCConnectionError(
                f"RPyC port not configured for agent '{self.agent_name}'. "
                f"Agent must be started with RPyC support."
            )

        # Connect via RPyC (ONLY method)
        try:
            # Get event loop
            loop = asyncio.get_running_loop()

            # ⚠️ BOUND THE WHOLE CONNECT — DO NOT REGRESS. ⚠️
            # ``async_connect``'s OWN ``timeout=`` kwarg wraps ONLY the
            # ``sock_connect`` step. The eager ``HANDLE_GETROOT`` handshake
            # inside async_connect (``await conn.async_request(HANDLE_GETROOT)``)
            # is UNBOUNDED. Against a busy/deadlocked peer TCP connects fine but
            # the handshake never gets answered → this ``connect()`` blocks
            # forever. That is exactly what hung an agent's
            # ``send_message_to_agent`` in a past incident: bounded only by the outer
            # 120s tool timeout, the send parked as a background job and never
            # returned ``{"success": true}``, so SendMessageCompletionMiddleware
            # never ended the turn and the agent WRONGLY kept making tool calls
            # after the send instead of going idle. Same family as a related
            # internal incident analysis (not included here). The OUTER
            # ``wait_for`` bounds the ENTIRE connect (TCP + eager handshake),
            # mirroring the web AgentService.
            # See tests/unit_tests/client/test_agent_client_connect_bounded.py.
            self._conn = await asyncio.wait_for(
                async_connect(
                    "127.0.0.1",
                    agent_info.rpyc_port,
                    config={"allow_public_attrs": True},
                    timeout=self._CONNECT_TIMEOUT,
                    loop=loop,
                ),
                timeout=self._CONNECT_TIMEOUT,
            )

            # NOTE: async_connect() already enables asyncio serving internally
            # AND eager-fetches ``conn._remote_root`` via the event-driven async
            # handshake, so the first ``self._conn.root`` access is non-blocking
            # (cached). ``_ensure_root`` therefore never triggers a blocking
            # ``sync_request(HANDLE_GETROOT)`` on a healthy conn.

            self.connected = True

        except asyncio.TimeoutError as e:
            raise IPCConnectionError(
                f"Connection to agent '{self.agent_name}' timed out after "
                f"{self._CONNECT_TIMEOUT}s (TCP + RPyC root handshake)"
            ) from e
        except Exception as e:
            raise IPCConnectionError(f"Failed to connect via RPyC: {e}") from e

    def _ensure_root(self) -> None:
        """Lazy initialization of _root.

        CRITICAL: This MUST be called before any RPC method that uses self._root.
        First call will access self._conn.root which triggers RPC handshake (may block once).
        Subsequent calls use cached self._root (fast).
        """
        if self._root is None and self._conn is not None:
            self._root = self._conn.root

    async def disconnect(self) -> None:
        """Disconnect from the agent (closes RPyC session)."""
        if not self.connected:
            return

        # Close RPyC connection
        if self._conn:
            try:
                self._conn.disable_asyncio_serving()
                self._conn.close()
            except Exception:
                pass  # Ignore errors during disconnect
            self._conn = None
            self._root = None

        self.connected = False

    async def _send_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Send a command via RPyC.

        OPTIMIZATION: Uses low-level HANDLE_ASYNC_CALLATTR to eliminate extra
        RPyC messages. Instead of:
          1. SEND getattr(root, 'method') -> RECV method_netref -> SEND call() -> RECV result
        We do:
          1. SEND callattr(root, 'method', args) -> RECV result

        This reduces RPyC protocol messages from 6 to 2 per call (see docs/RPYC_SERIALIZATION_DESIGN.md).

        Args:
            command: Command dictionary to send

        Returns:
            Response dictionary

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        if not self.connected or not self._conn:
            raise IPCConnectionError("Not connected to agent")

        # Lazy root initialization (first call triggers RPC handshake)
        self._ensure_root()

        try:
            command_type = command.get("command")

            # Import RPyC consts for low-level protocol handlers
            from rpyc_async.core import consts

            # Map commands to RPyC service methods using low-level HANDLE_ASYNC_CALLATTR
            # This bypasses Python's attribute access (which creates intermediate netrefs)
            # and directly calls the remote method in a single RPC roundtrip.

            if command_type == "status":
                # Use low-level async_request with HANDLE_ASYNC_CALLATTR
                # NOTE: Must use "exposed_" prefix for actual method name
                # Format: (handler, root_netref, method_name, args_tuple, kwargs_list)
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,              # root object
                    "exposed_get_status",    # method name (with exposed_ prefix!)
                    (),                      # args tuple (empty for get_status)
                    ()                       # kwargs list (empty)
                )
                # Wait for result (still JSON string from server)
                response_json_str = await async_result
                # Deserialize JSON string to local dict (no RPC overhead for field access)
                response = json.loads(response_json_str)
                return response

            elif command_type == "summarize":
                prompt = command.get("prompt")
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_summarize",
                    (prompt,),
                    (),
                )
                response_json_str = await async_result
                return json.loads(response_json_str)

            elif command_type == "invoke":
                message = command.get("message", "")
                stream = command.get("stream", False)
                # Optional startup-query send-through (§4.5). When present,
                # forwarded positionally to exposed_invoke_agent(message,
                # stream, sum_query, query).
                sum_query = command.get("sum_query")
                query = command.get("query")
                # Use low-level async_request with HANDLE_ASYNC_CALLATTR
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_invoke_agent",  # method name (with exposed_ prefix!)
                    (message, stream, sum_query, query),  # args tuple
                    ()                       # kwargs list (empty)
                )
                response_json_str = await async_result
                # Deserialize JSON string to local dict
                response = json.loads(response_json_str)
                return response

            elif command_type == "enqueue_message":
                # Fire-and-forget delivery: server enqueues the message and
                # returns immediately without waiting for the agent's LLM
                # turn to complete. Used by send_message_to_agent.
                message = command.get("message", "")
                reply_back = command.get("reply_back", True)
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_enqueue_message",
                    (message, reply_back),
                    ()
                )
                response_json_str = await async_result
                response = json.loads(response_json_str)
                return response

            elif command_type == "get_history":
                limit = command.get("limit", 50)
                offset = command.get("offset", 0)
                # Use low-level async_request with HANDLE_ASYNC_CALLATTR
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_get_history",   # method name (with exposed_ prefix!)
                    (limit, offset),         # args tuple
                    ()                       # kwargs list (empty)
                )
                response_json_str = await async_result
                # Deserialize JSON string to local dict
                response = json.loads(response_json_str)
                return response

            elif command_type == "metadata":
                # Exposes agent-side ``exposed_get_metadata``. Unlike
                # ``status`` this call returns ONLY what the agent knows
                # for a fact — ``directory``, ``started``, ``pid``,
                # ``rpyc_port``, ``model``, ``provider``. The caller must
                # not invent any field that is missing; if the RPC fails
                # the helper returns ``{"success": False, ...}`` and the
                # UI is responsible for printing "N/A" rather than
                # fabricating a value.
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_get_metadata",
                    (),
                    (),
                )
                response_json_str = await async_result
                response = json.loads(response_json_str)
                return response

            elif command_type == "reindex_history":
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_reindex_history",
                    (),
                    (),
                )
                response_json_str = await async_result
                response = json.loads(response_json_str)
                return response

            elif command_type == "status_raw":
                # Same as ``status`` but returns the raw dict from
                # ``exposed_get_status`` (with ``task_manager_active`` /
                # ``task_manager_disabled`` keys present when the agent
                # configured a TaskManager). Used by ``--status`` to
                # display honest TM state without reshaping it through
                # ``SessionInfo`` (which has no TM fields).
                async_result = self._conn.async_request(
                    consts.HANDLE_ASYNC_CALLATTR,
                    self._root,
                    "exposed_get_status",
                    (),
                    (),
                )
                response_json_str = await async_result
                response = json.loads(response_json_str)
                return response

            elif command_type in ("attach", "detach"):
                # RPyC doesn't need explicit attach/detach
                return {"success": True}

            else:
                raise IPCConnectionError(f"Unknown command type: {command_type}")

        except Exception as e:
            # Connection is dead — reset state so subsequent calls fail
            # fast at the "Not connected to agent" guard (line 171)
            # instead of cascading more opaque RPyC errors.
            self.connected = False
            self._conn = None
            self._root = None
            raise IPCConnectionError(f"RPyC communication error: {e}") from e

    async def get_status(self) -> "SessionInfo":
        """Get agent status as SessionInfo.

        This method queries the agent's status and returns a SessionInfo object
        compatible with the BaseSession protocol.

        OPTIMIZATION: Agent now returns minimal status response (status + error only)
        to reduce serialization overhead. We reconstruct minimal SessionInfo locally.

        Returns:
            SessionInfo with current agent state

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        response = await self._send_command({"command": "status"})

        if not response.get("success"):
            raise IPCConnectionError(
                f"Failed to get status: {response.get('error', 'Unknown error')}"
            )

        # Import here to avoid circular dependency
        from deep_worker_cli.basesession.protocol_types import SessionInfo
        from datetime import datetime

        # OPTIMIZATION: Handle both old format (session_info) and new format (minimal)
        session_data = response.get("session_info")

        if session_data:
            # Old format - full SessionInfo (backward compatibility)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[CLIENT] Using old session_info format")
            result = SessionInfo.model_validate(session_data)
            logger.info(f"[CLIENT] model_validate() completed, returning SessionInfo")
            return result
        else:
            # New format - minimal status response
            # Reconstruct minimal SessionInfo with only status field populated
            status_str = response.get("status", "unknown")
            error = response.get("error")

            return SessionInfo(
                name=self.agent_name,
                session_type="langchain",
                status=status_str,
                working_dir="/tmp",  # Placeholder (not used)
                created_at=datetime.now(),  # Placeholder (not used)
                last_activity=None,
                tasks_completed=0,
                error_message=error,
                data=None,
                metadata=None,
            )

    async def get_agent_metadata(self) -> dict[str, Any] | None:
        """Return raw agent metadata (directory, started, model, …) or None.

        This is a read-only query of ``exposed_get_metadata``. Unlike
        :meth:`get_status`, this method does NOT invent placeholder
        values: if the RPC failed or the agent doesn't expose the
        method, we return ``None`` and let the caller (e.g.
        ``deep-worker-cli --status``) print "N/A" explicitly.

        Returns:
            The ``metadata`` dict from the agent (keys: ``directory``,
            ``pid``, ``started``, ``rpyc_port``, ``agent_type``,
            ``model``, ``provider``), or ``None`` on error.
        """
        try:
            response = await self._send_command({"command": "metadata"})
        except IPCConnectionError:
            return None
        if not response.get("success"):
            return None
        meta = response.get("metadata")
        return meta if isinstance(meta, dict) else None

    async def reindex_history(self) -> dict[str, Any]:
        response = await self._send_command({"command": "reindex_history"})
        if not isinstance(response, dict):
            return {"success": False, "error": "Invalid reindex_history response"}
        return response

    async def get_task_manager_info(self) -> dict[str, Any] | None:
        """Return TaskManager state from the agent, or ``None`` on error.

        Agent's ``exposed_get_status`` includes
        ``task_manager_active`` / ``task_manager_disabled`` ONLY when
        the agent was launched with ``--task-manager`` (see
        ``rpyc_service/agent_service.py``). We surface three distinct
        states to the caller:

        * ``{"configured": True, "active": bool, "disabled": bool}``
          — agent has a TaskManager and told us its state;
        * ``{"configured": False, "active": None, "disabled": None}``
          — agent has no TaskManager (fields absent in the response);
        * ``None`` — we could not reach the agent / RPC error.

        No guessing, no defaults: the UI decides what to display.
        """
        try:
            response = await self._send_command({"command": "status_raw"})
        except IPCConnectionError:
            return None
        if not response.get("success"):
            return None
        has_tm_fields = (
            "task_manager_active" in response
            or "task_manager_disabled" in response
        )
        if not has_tm_fields:
            return {"configured": False, "active": None, "disabled": None}
        return {
            "configured": True,
            "active": bool(response.get("task_manager_active")),
            "disabled": bool(response.get("task_manager_disabled")),
        }

    async def invoke_agent(
        self, message: str, stream: bool = False
    ) -> dict[str, Any]:
        """Send a message to the agent and get response.

        Args:
            message: User message to send to the agent
            stream: Whether to stream the response (default: False)

        Returns:
            Response dictionary with format:
                {
                    "success": bool,
                    "response": {
                        "role": "assistant",
                        "content": str
                    }
                }

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        response = await self._send_command({
            "command": "invoke",
            "message": message,
            "stream": stream,
        })

        if not response.get("success"):
            raise IPCConnectionError(
                f"Failed to invoke agent: {response.get('error', 'Unknown error')}"
            )

        return response

    async def get_history(
        self, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Get conversation history from the agent.

        Args:
            limit: Maximum number of messages to retrieve (default: 50)
            offset: Number of recent messages to skip for pagination (default: 0)

        Returns:
            Response dictionary with format:
                {
                    "success": bool,
                    "history": [list of message dicts],
                    "total": total message count,
                    "has_more": bool indicating if more messages exist
                }

        Raises:
            IPCConnectionError: If not connected or communication fails
        """
        response = await self._send_command({
            "command": "get_history",
            "limit": limit,
            "offset": offset,
        })

        if not response.get("success"):
            raise IPCConnectionError(
                f"Failed to get history: {response.get('error', 'Unknown error')}"
            )

        return response
