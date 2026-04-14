"""AgentService for managing deep-worker agent connections.

This is the core service that:
1. Discovers agents from ProcessRegistry
2. Manages RPyC connections to agents
3. Subscribes to agent message callbacks
4. Broadcasts agent events to WebSocket clients

CRITICAL ARCHITECTURAL PRINCIPLES:
==================================
1. SINGLE EVENT LOOP: All async operations run in ONE event loop (no threads)
2. DIRECT RPyC: Connect directly to agents (no DwSession abstraction)
3. AGENT IS SOURCE OF TRUTH: All messages come from agent via callbacks
4. LOCALHOST ONLY: Only connect to 127.0.0.1 (security requirement)
"""

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.schemas.agent_schemas import AgentInfo, AgentStatus, AgentStatusInfo

# ============================================================================
# CRITICAL: RPYC MUST BE IMPORTED FROM SOURCE CODE, NOT FROM PIP!
# ============================================================================
# Import rpyc from a local rpyc_async source checkout to ensure AsyncioServer support
# ============================================================================
_rpyc_async_path = Path("rpyc_async")
if _rpyc_async_path.exists() and str(_rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(_rpyc_async_path))

import rpyc  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing connections to deep-worker agents.
    
    This service uses ProcessRegistry to discover agents and RPyC AsyncioServer
    to communicate with them directly. All operations are async and run in the
    main event loop.
    
    Architecture:
    - Discovers agents from /tmp/deep_worker_registry.json
    - Establishes RPyC connections to agent processes
    - Subscribes to agent message callbacks
    - Broadcasts events to registered callbacks (WebSocket connections)
    """

    def __init__(
        self,
        registry: Any | None = None,
    ) -> None:
        """Initialize AgentService.
        
        Args:
            registry: ProcessRegistry instance (or mock for testing).
                     If None, will create real ProcessRegistry from deep_worker_cli.
        """
        # Import ProcessRegistry only if not provided (allows testing with mocks)
        if registry is None:
            import sys
            from pathlib import Path

            # Add the CLI package to path if not already there
            dw_cli_source_path = (
                Path(__file__).resolve().parents[3]
                / "deep_worker" / "deep_worker_cli"
            )
            if str(dw_cli_source_path) not in sys.path:
                sys.path.insert(0, str(dw_cli_source_path))

            from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]

            self._registry: Any = ProcessRegistry()
        else:
            self._registry = registry

        # Agent tracking
        # IMPORTANT: We store basic agent info from registry, not full connections yet
        # Full RPyC connections will be established on-demand or during monitoring
        self._agents: dict[str, AgentInfo] = {}

        # RPyC connections (agent_name -> RPyC connection object)
        # These are established when we connect to agents
        self._connections: dict[str, Any] = {}
        
        # RPyC root objects (agent_name -> root service proxy)
        # Used to call exposed methods on the agent
        self._roots: dict[str, Any] = {}
        
        # Message callback services for RPyC subscriptions
        # (agent_name -> callback service instance)
        self._callback_services: dict[str, Any] = {}

        # Message callbacks (for WebSocket broadcasting)
        # Each callback is an async function that receives message dict
        self._message_callbacks: list[Callable[[dict[str, Any]], Awaitable[None]]] = []

        # Auto-approve agents (agents where user enabled auto-approve)
        self._auto_approve_agents: set[str] = set()

        # Background monitoring task
        self._monitor_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        logger.info("AgentService initialized")

    async def initialize(self) -> None:
        """Initialize service and discover agents.
        
        This discovers all agents currently registered in ProcessRegistry
        and prepares the service for operation.
        """
        logger.info("Initializing AgentService - discovering agents")
        await self._discover_agents()
        logger.info(f"AgentService initialized with {len(self._agents)} agents")

    async def _discover_agents(self) -> None:
        """Discover agents from ProcessRegistry.
        
        This reads the ProcessRegistry and updates our internal agent tracking.
        Runs in async context but ProcessRegistry operations are sync (file-based).
        """
        # ProcessRegistry is sync (file-based), so we run it in executor
        # to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        agent_process_infos = await loop.run_in_executor(
            None, self._registry.list_all
        )

        logger.info(f"Discovered {len(agent_process_infos)} agents from registry")

        # Convert to AgentInfo (our internal format)
        # NOTE: We don't know provider/model yet - will query via RPyC when needed
        for proc_info in agent_process_infos:
            agent_info = AgentInfo(
                name=proc_info.name,
                pid=proc_info.pid,
                status=AgentStatus.IDLE,  # Default, will update from agent
                provider="unknown",  # Will query from agent
                model="unknown",  # Will query from agent
                rpyc_port=proc_info.rpyc_port,
                working_dir=proc_info.working_dir,
                started_at=proc_info.started_at,
            )
            self._agents[proc_info.name] = agent_info
            logger.debug(f"Registered agent: {proc_info.name} (PID {proc_info.pid}, port {proc_info.rpyc_port})")

    async def list_agents(self) -> list[AgentInfo]:
        """Get list of all known agents.
        
        Returns:
            List of AgentInfo for all discovered agents.
        """
        return list(self._agents.values())

    async def get_agent(self, agent_name: str) -> AgentInfo | None:
        """Get info for a specific agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            AgentInfo if found, None otherwise.
        """
        return self._agents.get(agent_name)

    async def is_connected(self, agent_name: str) -> bool:
        """Check if we have an active RPyC connection to an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            True if connected, False otherwise.
        """
        return agent_name in self._connections and self._connections[agent_name] is not None

    def get_root(self, agent_name: str) -> Any | None:
        """Get RPyC root proxy for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            RPyC root proxy if connected, None otherwise.
        """
        return self._roots.get(agent_name)

    async def connect_to_agent(self, agent_name: str) -> bool:
        """Establish RPyC connection to an agent.
        
        This connects to the agent's RPyC server and subscribes to message callbacks.
        
        Args:
            agent_name: Name of the agent to connect to.
            
        Returns:
            True if connection successful, False otherwise.
        """
        # Check if already connected
        if await self.is_connected(agent_name):
            logger.debug(f"Already connected to {agent_name}")
            return True
        
        # Get agent info
        agent = self._agents.get(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found in registry")
            return False
        
        try:
            logger.info(f"Connecting to agent {agent_name} on port {agent.rpyc_port}")
            
            # SECURITY: Connect only to localhost (127.0.0.1)
            conn = rpyc.connect(
                "127.0.0.1",
                agent.rpyc_port,
                config={"allow_public_attrs": True}
            )
            
            # CRITICAL: Enable asyncio serving BEFORE accessing root
            # With AsyncioServer, we must enable asyncio serving first
            loop = asyncio.get_running_loop()
            conn.enable_asyncio_serving(loop=loop)
            
            # Now safe to get root (after asyncio serving is enabled)
            root = conn.root
            
            # Store connection and root
            self._connections[agent_name] = conn
            self._roots[agent_name] = root
            
            # Subscribe to agent messages
            await self._subscribe_to_agent(agent_name, root)

            # Fetch metadata (provider, model) — once at connect time
            await self._fetch_agent_metadata(agent_name, root)

            # Query and update agent status
            await self._update_agent_status(agent_name, root)
            
            logger.info(f"Successfully connected to agent {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to agent {agent_name}: {e}")
            # Cleanup partial connection
            if agent_name in self._connections:
                del self._connections[agent_name]
            if agent_name in self._roots:
                del self._roots[agent_name]
            return False

    async def disconnect_from_agent(self, agent_name: str) -> None:
        """Disconnect from an agent using fire-and-forget for cleanup.

        Args:
            agent_name: Name of the agent to disconnect from.
        """
        if not await self.is_connected(agent_name):
            return

        from rpyc.utils.helpers import fire_and_forget_async

        try:
            # Unsubscribe via fire-and-forget (don't block on cleanup)
            if agent_name in self._callback_services and agent_name in self._roots:
                root = self._roots[agent_name]
                callback = self._callback_services[agent_name]

                async def on_success(result: Any) -> None:
                    logger.info(f"Unsubscribed from {agent_name}")

                async def on_error(exc: BaseException) -> None:
                    logger.warning(f"Error unsubscribing from {agent_name}: {exc}")
                    # Don't fail disconnect on unsubscribe error

                # Fire and forget with short timeout (cleanup)
                fire_and_forget_async(
                    root.unsubscribe_messages(callback),
                    timeout=2.0,
                    success_callback=on_success,
                    error_callback=on_error,
                    name=f"unsubscribe_{agent_name}",
                )

                # Don't wait - continue with disconnect immediately

            # Close connection (do this synchronously)
            conn = self._connections[agent_name]
            conn.disable_asyncio_serving()
            conn.close()

            logger.info(f"Disconnected from agent {agent_name}")

        except Exception as e:
            logger.error(f"Error disconnecting from {agent_name}: {e}")
        finally:
            # Cleanup
            if agent_name in self._connections:
                del self._connections[agent_name]
            if agent_name in self._roots:
                del self._roots[agent_name]
            if agent_name in self._callback_services:
                del self._callback_services[agent_name]

    async def _subscribe_to_agent(self, agent_name: str, root: Any) -> None:
        """Subscribe to message callbacks from an agent using fire-and-forget pattern.

        Args:
            agent_name: Name of the agent.
            root: RPyC root proxy for the agent.
        """
        from rpyc.utils.helpers import fire_and_forget_async

        # Create callback service for this agent
        agent_service = self

        class MessageCallbackService(rpyc.Service):  # type: ignore[misc]
            """RPyC service for receiving messages from agent."""

            async def exposed_on_message(self, msg: dict[str, Any]) -> None:
                """Receive message from agent (async exposed for RPyC).

                This callback is invoked by the agent for EVERY message:
                - user messages (type="user")
                - assistant messages (type="assistant" - final responses)
                - assistant_chunk messages (type="assistant_chunk" - streaming)
                - tool_call messages (type="tool_call")
                - tool_result messages (type="tool_result")
                - token_update messages (type="token_update")

                CRITICAL ARCHITECTURE:
                We wrap all messages in "agent_message" type so the frontend
                handleWebSocketMessage() can route them correctly via switch/case.

                Frontend expects:
                {
                    type: "agent_message",      // ← Frontend routing type
                    agent_name: "test_agent",   // ← Which agent sent this
                    message: {...}              // ← Original message from agent (nested)
                }

                Args:
                    msg: Message dict from agent (raw format from deep_worker_cli).
                         NOTE: This might be an RPyC NetRef, so we need to convert to plain dict.
                """
                import json

                # ============================================================
                # Convert RPyC NetRef to plain Python dict
                # ============================================================
                # RPyC sends NetRef proxy objects that look like dicts but aren't.
                # We need to recursively convert them to plain Python types.
                def convert_netref(obj: Any) -> Any:
                    """Recursively convert RPyC NetRef objects to plain Python types."""
                    if isinstance(obj, dict):
                        return {k: convert_netref(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_netref(item) for item in obj]
                    elif isinstance(obj, (str, int, float, bool, type(None))):
                        return obj
                    else:
                        # For any other type (including NetRef), convert to string
                        return str(obj)

                # Convert message to plain Python types
                plain_msg = convert_netref(msg)

                # ============================================================
                # Handle approval_request specially
                # ============================================================
                if plain_msg.get("type") == "approval_request":
                    # Auto-approve if enabled for this agent
                    if agent_name in agent_service._auto_approve_agents:
                        approval_id = plain_msg.get("approval_id")
                        if approval_id:
                            await agent_service.send_approval(
                                agent_name, approval_id, "approve"
                            )
                        return

                    # Broadcast as top-level type (not wrapped in agent_message)
                    await agent_service._broadcast_message({
                        "type": "approval_request",
                        "agent_name": agent_name,
                        "approval_id": plain_msg.get("approval_id"),
                        "actions": plain_msg.get("actions", []),
                    })
                    return

                if plain_msg.get("type") == "approval_result":
                    await agent_service._broadcast_message({
                        "type": "approval_result",
                        "agent_name": agent_name,
                        "approval_id": plain_msg.get("approval_id"),
                        "decision": plain_msg.get("decision"),
                    })
                    return

                # ============================================================
                # WRAP regular messages in standard WebSocket format
                # ============================================================
                wrapped_message = {
                    "type": "agent_message",        # ← Frontend switch/case routing
                    "agent_name": agent_name,        # ← Which agent sent this
                    "message": plain_msg,            # ← Original message (nested, converted)
                }

                # Broadcast wrapped message to all WebSocket clients
                await agent_service._broadcast_message(wrapped_message)

        # Create and store callback instance
        callback = MessageCallbackService()
        self._callback_services[agent_name] = callback

        async def on_success(success: bool) -> None:
            """Handle successful subscription."""
            if success:
                logger.info(f"Subscribed to messages from {agent_name}")
                await self._broadcast_message({
                    "type": "subscription_success",
                    "agent_name": agent_name,
                })
            else:
                logger.error(f"Failed to subscribe to messages from {agent_name}")
                await self._broadcast_message({
                    "type": "subscription_failed",
                    "agent_name": agent_name,
                    "error": "Agent returned False",
                })

        async def on_error(exc: BaseException) -> None:
            """Handle subscription errors."""
            logger.error(f"Error subscribing to {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "subscription_error",
                "agent_name": agent_name,
                "error": str(exc),
            })
            # Clean up callback service on error
            if agent_name in self._callback_services:
                del self._callback_services[agent_name]

        # Fire and forget with 10s timeout
        fire_and_forget_async(
            root.subscribe_messages(callback),
            timeout=10.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"subscribe_{agent_name}",
        )

    async def _fetch_agent_metadata(self, agent_name: str, root: Any) -> None:
        """Fetch agent metadata (provider, model) once at connect time.

        Args:
            agent_name: Name of the agent.
            root: RPyC root proxy for the agent.
        """
        from rpyc.utils.helpers import fire_and_forget_async

        async def on_success(response_str: str) -> None:
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse metadata response: {response_str}")
                return

            if response.get("success"):
                metadata = response.get("metadata", {})
                provider = metadata.get("provider")
                model = metadata.get("model")

                if agent_name in self._agents:
                    if provider:
                        self._agents[agent_name].provider = provider
                    if model:
                        self._agents[agent_name].model = model
                    logger.info(f"Fetched metadata for {agent_name}: {provider}/{model}")

                    # Broadcast metadata update so frontend picks it up
                    await self._broadcast_message({
                        "type": "agent_metadata",
                        "agent_name": agent_name,
                        "provider": self._agents[agent_name].provider,
                        "model": self._agents[agent_name].model,
                    })

        async def on_error(exc: BaseException) -> None:
            logger.error(f"Error fetching metadata for {agent_name}: {exc}")

        try:
            fire_and_forget_async(
                root.get_metadata(),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"get_metadata_{agent_name}",
            )
        except Exception as e:
            logger.error(f"Failed to call get_metadata on {agent_name}: {e}")

    async def _update_agent_status(self, agent_name: str, root: Any) -> None:
        """Query agent status and update AgentInfo using fire-and-forget pattern.

        Args:
            agent_name: Name of the agent.
            root: RPyC root proxy for the agent.
        """
        from rpyc.utils.helpers import fire_and_forget_async

        async def on_success(response_str: str) -> None:
            """Handle successful status query.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse status response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            if response.get("success"):
                # Agent's exposed_get_status() returns only status:
                #   {"success": true, "status": "working", "error": null}
                # Provider/model are fetched once via get_metadata() at connect time.
                status_str = response.get("status", "idle")

                if agent_name in self._agents:
                    self._agents[agent_name].status = self._convert_status(status_str)

                    logger.info(f"Updated status for {agent_name}: {status_str}")

                    # Broadcast status update
                    broadcast_msg: dict[str, Any] = {
                        "type": "agent_status",
                        "agent_name": agent_name,
                        "status": status_str,
                    }
                    # Include task_manager fields only if present in response
                    if "task_manager_active" in response:
                        broadcast_msg["task_manager_active"] = response["task_manager_active"]
                    if "task_manager_disabled" in response:
                        broadcast_msg["task_manager_disabled"] = response["task_manager_disabled"]
                    await self._broadcast_message(broadcast_msg)

        async def on_error(exc: BaseException) -> None:
            """Handle status query errors."""
            logger.error(f"Error updating status for {agent_name}: {exc}")
            # Set defaults on error
            if agent_name in self._agents:
                self._agents[agent_name].provider = "error"
                self._agents[agent_name].model = "unavailable"

        # Fire and forget with 5s timeout (status queries should be fast)
        fire_and_forget_async(
            root.get_status(),
            timeout=5.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"get_status_{agent_name}",
        )

    async def send_message(self, agent_name: str, message: str) -> dict[str, Any]:
        """Send a message to an agent using fire-and-forget pattern.

        Returns immediately. Results delivered via WebSocket callbacks.

        Args:
            agent_name: Name of the agent.
            message: Message to send.

        Returns:
            Immediate response indicating message was sent. Actual agent response
            will be delivered via WebSocket with type="agent_response" or "agent_error".
        """
        # Connect if not connected
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        from rpyc.utils.helpers import fire_and_forget_async

        root = self._roots[agent_name]

        # Define async callbacks
        async def on_success(response_str: str) -> None:
            """Handle successful agent response.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse agent response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            logger.info(f"Agent {agent_name} responded successfully")
            logger.debug(f"Response type: {type(response)}, value: {response}")

            # Extract text from response (handle different formats)
            response_text = response
            if isinstance(response, dict):
                # Try common keys for response text
                response_text = (
                    response.get("text") or
                    response.get("content") or
                    response.get("response") or
                    str(response)
                )

            # Ensure response_text is JSON serializable
            if not isinstance(response_text, (str, int, float, bool, type(None))):
                response_text = str(response_text)

            logger.debug(f"Extracted response_text: {response_text[:200]}")

            # Broadcast success to WebSocket clients
            await self._broadcast_message({
                "type": "agent_response",
                "agent_name": agent_name,
                "success": True,
                "response": response_text,  # Send only serializable text
            })

        async def on_error(exc: BaseException) -> None:
            """Handle RPC errors (timeout, connection, remote exception)."""
            logger.error(f"Agent {agent_name} error: {exc}")
            # Broadcast error to WebSocket clients
            await self._broadcast_message({
                "type": "agent_error",
                "agent_name": agent_name,
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            })

        # ================================================================
        # DO NOT ADD A TIMEOUT HERE!
        # ================================================================
        # Agent invocation can take minutes or even tens of minutes:
        # - Complex tasks with multiple tool calls
        # - TaskManager evaluation loops (multiple LLM iterations)
        # - Interrupt/approval waiting (blocks until user decides)
        # - Web search and other slow tools
        #
        # A timeout here does NOT cancel the agent — it only makes the
        # web UI show a false error while the agent keeps working.
        # Connection errors (agent crash, network) are already handled
        # by RPyC raising EOFError/ConnectionError → on_error callback.
        # ================================================================
        fire_and_forget_async(
            root.invoke_agent(message, stream=False),
            timeout=None,
            success_callback=on_success,
            error_callback=on_error,
            name=f"invoke_agent_{agent_name}",
        )

        # Return immediately - actual response comes via WebSocket
        return {
            "success": True,
            "message": "Message sent to agent (response will arrive via WebSocket)",
            "agent_name": agent_name,
        }

    async def send_approval(
        self,
        agent_name: str,
        approval_id: str,
        decision: str,
    ) -> dict[str, Any]:
        """Send approval/rejection decision for a tool interrupt.

        Args:
            agent_name: Agent name
            approval_id: Unique ID of the approval request
            decision: "approve", "reject", or "auto_approve_all"

        Returns:
            Response dict with success status.
        """
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        root = self._roots[agent_name]

        decision_dict = {"type": decision}
        if decision == "reject":
            decision_dict["message"] = "Rejected by user via web UI"

        # Handle auto_approve_all: approve this one and set flag
        if decision == "auto_approve_all":
            decision_dict = {"type": "auto_approve_all"}
            self._auto_approve_agents.add(agent_name)

        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response: Any) -> None:
                logger.info(f"Approval sent for {agent_name}: {decision}")

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error sending approval for {agent_name}: {exc}")

            fire_and_forget_async(
                root.respond_to_approval(approval_id, decision_dict),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"approve_{agent_name}",
            )
        except Exception as e:
            logger.error(f"Failed to send approval to {agent_name}: {e}")
            return {"success": False, "error": str(e)}

        return {"success": True, "message": f"Decision '{decision}' sent"}

    async def set_task_manager_enabled(
        self, agent_name: str, enabled: bool
    ) -> dict[str, Any]:
        """Enable or disable TaskManager for an agent."""
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        root = self._roots[agent_name]
        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response_str: str) -> None:
                import json
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
                logger.info(f"TaskManager {'enabled' if enabled else 'disabled'} for {agent_name}")

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error toggling TaskManager for {agent_name}: {exc}")

            fire_and_forget_async(
                root.set_task_manager_enabled(enabled),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"tm_toggle_{agent_name}",
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": True, "enabled": enabled}

    async def reset_task_manager(self, agent_name: str) -> dict[str, Any]:
        """Reset TaskManager for an agent."""
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        root = self._roots[agent_name]
        try:
            from rpyc.utils.helpers import fire_and_forget_async

            async def on_success(response_str: str) -> None:
                logger.info(f"TaskManager reset for {agent_name}")

            async def on_error(exc: BaseException) -> None:
                logger.error(f"Error resetting TaskManager for {agent_name}: {exc}")

            fire_and_forget_async(
                root.reset_task_manager(),
                timeout=10.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"tm_reset_{agent_name}",
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": True, "message": "TaskManager reset requested"}

    async def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
        offset: int = 0,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Get agent history using fire-and-forget pattern.

        Args:
            agent_name: Agent name
            limit: Max messages
            offset: Skip recent messages
            request_id: Optional ID to correlate request/response

        Returns:
            Immediate response. Actual history delivered via WebSocket with
            type="history_response" or "history_error".
        """
        # Connect if not connected
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        from rpyc.utils.helpers import fire_and_forget_async
        import uuid

        # Generate request ID if not provided
        req_id = request_id or str(uuid.uuid4())

        root = self._roots[agent_name]

        async def on_success(response_str: str) -> None:
            """Handle successful history retrieval.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse history response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            logger.info(f"Got history for {agent_name}: {len(response.get('history', []))} messages")
            await self._broadcast_message({
                "type": "history_response",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": True,
                "history": response.get("history", []),
            })

        async def on_error(exc: BaseException) -> None:
            """Handle history retrieval errors."""
            logger.error(f"Error getting history from {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "history_error",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": False,
                "error": str(exc),
            })

        # Fire and forget with 15s timeout (history can be large)
        fire_and_forget_async(
            root.get_history(limit, offset),
            timeout=15.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"get_history_{agent_name}",
        )

        return {
            "success": True,
            "message": "History request sent (response will arrive via WebSocket)",
            "request_id": req_id,
        }

    async def get_agent_last_response(
        self,
        agent_name: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Get last assistant response using fire-and-forget pattern.

        Args:
            agent_name: Agent name
            request_id: Optional ID to correlate request/response

        Returns:
            Immediate response. Actual data delivered via WebSocket with
            type="last_response_result" or "last_response_error".
        """
        # Connect if not connected
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        from rpyc.utils.helpers import fire_and_forget_async
        import uuid

        req_id = request_id or str(uuid.uuid4())
        root = self._roots[agent_name]

        async def on_success(response_str: str) -> None:
            """Handle successful last response retrieval."""
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse last response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            logger.info(f"Got last response for {agent_name}")
            await self._broadcast_message({
                "type": "last_response_result",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": True,
                "last_response": response.get("last_response"),
            })

        async def on_error(exc: BaseException) -> None:
            """Handle last response retrieval errors."""
            logger.error(f"Error getting last response from {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "last_response_error",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": False,
                "error": str(exc),
            })

        try:
            fire_and_forget_async(
                root.get_last_response(),
                timeout=5.0,
                success_callback=on_success,
                error_callback=on_error,
                name=f"get_last_response_{agent_name}",
            )
        except Exception as e:
            logger.error(f"Failed to call get_last_response on {agent_name}: {e}")
            return {"success": False, "error": f"Agent does not support get_last_response: {e}"}

        return {
            "success": True,
            "message": "Last response request sent (response will arrive via WebSocket)",
            "request_id": req_id,
        }

    async def get_agent_detailed_status(
        self,
        agent_name: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed agent status using fire-and-forget pattern.

        Args:
            agent_name: Name of the agent.
            request_id: Optional ID to correlate request/response

        Returns:
            Immediate response. Actual status delivered via WebSocket with
            type="detailed_status_response" or "detailed_status_error".
        """
        # Connect if not connected
        if not await self.is_connected(agent_name):
            if not await self.connect_to_agent(agent_name):
                return {"success": False, "error": "Failed to connect to agent"}

        from rpyc.utils.helpers import fire_and_forget_async
        import uuid

        req_id = request_id or str(uuid.uuid4())
        root = self._roots[agent_name]

        async def on_success(response_str: str) -> None:
            """Handle successful status retrieval.

            Note: Agent returns JSON-serialized string, not dict!
            """
            import json
            try:
                response = json.loads(response_str) if isinstance(response_str, str) else response_str
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse detailed status response: {response_str}")
                await on_error(Exception(f"Invalid response format: {response_str}"))
                return

            if not response.get("success"):
                await on_error(Exception("Agent returned success=False"))
                return

            session_info = response.get("session_info", {})
            agent = self._agents.get(agent_name)

            if not agent:
                await on_error(Exception("Agent not found in tracking"))
                return

            # Build status info dict
            status_info = {
                "name": agent.name,
                "pid": agent.pid,
                "status": self._convert_status(session_info.get("status", "idle")).value,
                "provider": session_info.get("provider", agent.provider),
                "model": session_info.get("model", agent.model),
                "rpyc_port": agent.rpyc_port,
                "working_dir": agent.working_dir,
                "started_at": agent.started_at.isoformat(),
                "current_thread_id": session_info.get("current_thread_id"),
                "message_count": session_info.get("message_count", 0),
                "last_activity": agent.started_at.isoformat(),
                "task_manager_active": session_info.get("task_manager_active", False),
            }

            logger.info(f"Got detailed status for {agent_name}")
            await self._broadcast_message({
                "type": "detailed_status_response",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": True,
                "status": status_info,
            })

    async def get_agent_dialog_formatted(self, agent_name: str) -> str:
        """Get formatted dialog text for an agent.

        This method retrieves the conversation history and formats it as plain text
        for display in the Full Dialog view, similar to the general-purpose web
        application's DwSession implementation.

        Args:
            agent_name: Name of the agent.

        Returns:
            Formatted dialog text as plain string in format:
            --- User ---
            <user message>

            --- Assistant ---
            <assistant response>

        Raises:
            ConnectionError: If agent is not connected.
            ValueError: If history retrieval fails.
            asyncio.TimeoutError: If agent doesn't respond within timeout.
        """
        if agent_name not in self._roots:
            raise ConnectionError(f"Agent '{agent_name}' is not connected")

        root = self._roots[agent_name]

        import asyncio
        import json

        try:
            # Get full history via RPyC (use large limit to get complete dialog)
            # This follows the DwSession.get_history_formatted_async() pattern
            history_call = root.get_history(limit=1000, offset=0)

            # Await the RPyC AsyncResult with timeout
            result = await asyncio.wait_for(history_call, timeout=10.0)

            # If result is a string, parse it as JSON
            if isinstance(result, str):
                result = json.loads(result)

            if not result.get('success'):
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"Failed to get history for '{agent_name}': {error_msg}")
                raise ValueError(f"Agent returned error: {error_msg}")

            history = result.get('history', [])

            # Build dialog pairs (user_query, assistant_response)
            # This follows the same logic as DwSession
            dialog_pairs = []
            current_user_query = None

            for message in history:
                msg_type = message.get('type')
                content = message.get('content', '')

                if msg_type == 'HumanMessage':
                    # New user query - start new pair
                    current_user_query = content

                elif msg_type == 'AIMessage' and current_user_query is not None:
                    # Assistant response - complete the pair
                    # Skip if content is empty (tool-only response)
                    if content and content.strip():
                        dialog_pairs.append((current_user_query, content))
                        current_user_query = None  # Reset for next pair

                # Skip ToolMessage - don't show in dialog

            # Format dialog text (same format as DwSession)
            if not dialog_pairs:
                return "No conversation history yet."

            parts = []
            for user_query, assistant_response in dialog_pairs:
                parts.append(f"--- User ---\n{user_query}\n")
                parts.append(f"--- Assistant ---\n{assistant_response}\n")

            return '\n'.join(parts)

        except asyncio.TimeoutError:
            logger.error(f"Timeout getting history for '{agent_name}' after 10s")
            raise asyncio.TimeoutError(f"Agent '{agent_name}' did not respond within 10s")

        except Exception as e:
            logger.error(f"Error getting formatted history for '{agent_name}': {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve dialog: {e}")

        async def on_error(exc: BaseException) -> None:
            """Handle status retrieval errors."""
            logger.error(f"Error getting detailed status from {agent_name}: {exc}")
            await self._broadcast_message({
                "type": "detailed_status_error",
                "agent_name": agent_name,
                "request_id": req_id,
                "success": False,
                "error": str(exc),
            })

        # Fire and forget with 10s timeout
        fire_and_forget_async(
            root.get_status(),
            timeout=10.0,
            success_callback=on_success,
            error_callback=on_error,
            name=f"get_detailed_status_{agent_name}",
        )

        return {
            "success": True,
            "message": "Status request sent (response will arrive via WebSocket)",
            "request_id": req_id,
        }

    def register_message_callback(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a callback for agent messages.
        
        Callbacks are called when messages are received from agents.
        Typically used to broadcast to WebSocket clients.
        
        Args:
            callback: Async function that receives message dict.
        """
        self._message_callbacks.append(callback)
        logger.debug(f"Registered message callback: {callback}")

    def unregister_message_callback(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Unregister a message callback.
        
        Args:
            callback: The callback to remove.
        """
        if callback in self._message_callbacks:
            self._message_callbacks.remove(callback)
            logger.debug(f"Unregistered message callback: {callback}")

    async def _broadcast_message(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all registered callbacks.

        This is called when we receive messages from agents via RPyC callbacks.

        Args:
            message: Message dict to broadcast.
        """
        import json

        # Add uptime data to every message
        from app.utils.uptime import get_uptime_data
        message["uptime"] = get_uptime_data()

        # CRITICAL: Ensure message is JSON serializable before broadcasting
        # This handles RPyC proxies, datetime objects, and other non-serializable types
        try:
            # Try to serialize and deserialize to catch any issues early
            json_str = json.dumps(message, default=str)
            serializable_message = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to serialize message for broadcast: {e}")
            logger.error(f"Message keys: {list(message.keys())}")
            # Fallback: convert all values to serializable types
            serializable_message = {}
            for key, value in message.items():
                try:
                    json.dumps({key: value})
                    serializable_message[key] = value
                except (TypeError, ValueError):
                    serializable_message[key] = str(value)
                    logger.warning(f"Converted non-serializable field '{key}' to string")

        logger.debug(f"Broadcasting message to {len(self._message_callbacks)} callbacks")

        # Call all callbacks concurrently with serializable message
        if self._message_callbacks:
            await asyncio.gather(
                *[callback(serializable_message) for callback in self._message_callbacks],
                return_exceptions=True,  # Don't let one failure stop others
            )

    def _convert_status(self, status_str: str) -> AgentStatus:
        """Convert string status to AgentStatus enum.

        Args:
            status_str: Status string from agent.

        Returns:
            AgentStatus enum value.
        """
        # Map agent SessionStatus values to web AgentStatus
        status_lower = status_str.lower()
        if status_lower == "ready":
            return AgentStatus.IDLE
        if status_lower == "working":
            return AgentStatus.RUNNING

        # Try direct mapping
        try:
            return AgentStatus(status_lower)
        except ValueError:
            # Unknown status - default to IDLE
            logger.warning(f"Unknown agent status: {status_str}, defaulting to IDLE")
            return AgentStatus.IDLE

    async def monitor_agents(self) -> None:
        """Background task to monitor ProcessRegistry for agent changes.
        
        This runs continuously and:
        - Discovers new agents
        - Removes terminated agents
        - Reconnects to restarted agents
        """
        logger.info("Starting agent monitoring task")
        
        while not self._shutdown_event.is_set():
            try:
                # Check for new/changed agents every 5 seconds
                await asyncio.sleep(5)
                
                # Discover current agents from registry
                loop = asyncio.get_event_loop()
                current_agents = await loop.run_in_executor(
                    None, self._registry.list_all
                )
                
                current_names = {agent.name for agent in current_agents}
                known_names = set(self._agents.keys())
                
                # Find new agents
                new_agents = current_names - known_names
                for name in new_agents:
                    logger.info(f"Discovered new agent: {name}")
                    # Add to tracking
                    proc_info = next(a for a in current_agents if a.name == name)
                    agent_info = AgentInfo(
                        name=proc_info.name,
                        pid=proc_info.pid,
                        status=AgentStatus.IDLE,
                        provider="unknown",
                        model="unknown",
                        rpyc_port=proc_info.rpyc_port,
                        working_dir=proc_info.working_dir,
                        started_at=proc_info.started_at,
                    )
                    self._agents[name] = agent_info
                    
                    # Broadcast agent discovered event
                    await self._broadcast_message({
                        "type": "agent_discovered",
                        "agent": agent_info.model_dump(mode="json"),
                    })
                    
                    # Try to connect
                    await self.connect_to_agent(name)
                
                # Find terminated agents
                terminated_agents = known_names - current_names
                for name in terminated_agents:
                    logger.info(f"Agent terminated: {name}")
                    
                    # Disconnect if connected
                    if await self.is_connected(name):
                        await self.disconnect_from_agent(name)
                    
                    # Remove from tracking
                    if name in self._agents:
                        del self._agents[name]
                    
                    # Broadcast agent terminated event
                    await self._broadcast_message({
                        "type": "agent_terminated",
                        "agent_name": name,
                    })
                
                # Check existing agents: reconnect or poll status
                for name in known_names & current_names:
                    if not await self.is_connected(name):
                        logger.debug(f"Attempting to reconnect to {name}")
                        await self.connect_to_agent(name)
                    else:
                        # Poll status of connected agents
                        root = self._roots.get(name)
                        if root:
                            await self._update_agent_status(name, root)
                
            except asyncio.CancelledError:
                logger.info("Agent monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}", exc_info=True)
                # Continue monitoring despite errors
                continue
        
        logger.info("Agent monitoring task stopped")

    async def shutdown(self) -> None:
        """Shutdown the service and cleanup resources.
        
        This:
        - Stops monitoring task
        - Closes all RPyC connections
        - Clears callbacks and agent tracking
        """
        logger.info("Shutting down AgentService")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop monitoring task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Disconnect from all agents
        for agent_name in list(self._connections.keys()):
            await self.disconnect_from_agent(agent_name)

        # Clear state
        self._agents.clear()
        self._message_callbacks.clear()

        logger.info("AgentService shutdown complete")
