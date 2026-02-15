"""RPyC service for deep-worker agent.

This module provides AgentService - an RPyC service that exposes agent
functionality via RPC calls and supports real-time message streaming via callbacks.

SECURITY WARNING:
================
This service is designed for LOCAL TRUSTED ENVIRONMENT ONLY!

- NO authentication or authorization
- Allows pickle serialization (unsafe if exposed externally)
- Allows public attribute access
- Suitable ONLY for localhost (127.0.0.1) binding

DO NOT expose this service to external networks!
See SECURITY.md for full security requirements.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import rpyc  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from deep_worker_cli.process.agent_process import AgentProcess

logger = logging.getLogger(__name__)


class AgentService(rpyc.Service):
    """
    RPyC service for deep-worker agent.

    Exposes:
    - invoke_agent: Execute agent with message
    - get_history: Retrieve conversation history
    - get_status: Get agent status
    - subscribe_messages: Subscribe to real-time messages
    - unsubscribe_messages: Unsubscribe from messages
    """

    def on_connect(self, conn: Any) -> None:
        """Called when client connects."""
        self._agent_process: AgentProcess | None = None
        self._event_loop: Any = None  # asyncio.AbstractEventLoop
        self._message_subscribers: set[Any] = set()
        self._subscribers_lock = threading.Lock()

        logger.info(f"RPyC client connected: {conn}")

    def on_disconnect(self, conn: Any) -> None:
        """Called when client disconnects."""
        logger.info(f"RPyC client disconnected: {conn}")

        # Clear subscribers for this connection
        with self._subscribers_lock:
            self._message_subscribers.clear()

    def set_agent_process(self, agent_process: AgentProcess) -> None:
        """Set reference to agent process."""
        self._agent_process = agent_process

    def set_event_loop(self, event_loop: Any) -> None:
        """Set reference to event loop."""
        self._event_loop = event_loop

    def exposed_get_status(self) -> dict[str, Any]:
        """Get agent status."""
        if not self._agent_process:
            return {"success": False, "error": "Agent not initialized"}

        try:
            if not self._agent_process.session_manager:
                return {"success": False, "error": "Session manager not initialized"}

            from deep_worker_cli.basesession.session_adapter import get_session_info

            session_info = get_session_info(self._agent_process)
            return {
                "success": True,
                "session_info": session_info.model_dump(mode="json"),
            }
        except Exception as e:
            logger.exception("Error getting status: %s", e)
            return {"success": False, "error": str(e)}

    def exposed_invoke_agent(
        self, message: str, stream: bool = False
    ) -> dict[str, Any]:
        """
        Invoke agent with message.

        Args:
            message: User message
            stream: Whether to stream response

        Returns:
            Response dict with success status and agent response
        """
        if not self._agent_process:
            return {"success": False, "error": "Agent not initialized"}

        if not self._event_loop:
            return {"success": False, "error": "Event loop not initialized"}

        try:
            # Notify subscribers about user message
            self._notify_subscribers(
                {"type": "user", "content": message, "role": "user"}
            )

            # Run async invoke in event loop
            import asyncio
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(
                self._agent_process._handle_invoke(
                    {"command": "invoke", "message": message, "stream": stream},
                    connection=None,  # type: ignore[arg-type]
                ),
                self._event_loop
            )

            # Wait for result with timeout
            response = future.result(timeout=300.0)  # 5 minute timeout

            # Notify subscribers about assistant response
            if response.get("success"):
                assistant_msg = response.get("response", {}).get("content", "")
                self._notify_subscribers(
                    {"type": "assistant", "content": assistant_msg, "role": "assistant"}
                )

            return response

        except concurrent.futures.TimeoutError:
            logger.error("Agent invocation timed out")
            return {"success": False, "error": "Agent invocation timed out"}
        except Exception as e:
            logger.exception("Error invoking agent: %s", e)
            return {"success": False, "error": str(e)}

    def exposed_get_history(
        self, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Get conversation history."""
        if not self._agent_process:
            return {"success": False, "error": "Agent not initialized"}

        if not self._event_loop:
            return {"success": False, "error": "Event loop not initialized"}

        try:
            import asyncio
            import concurrent.futures

            # Run async get_history in event loop
            future = asyncio.run_coroutine_threadsafe(
                self._agent_process._handle_get_history(
                    {"command": "get_history", "limit": limit, "offset": offset},
                    connection=None,  # type: ignore[arg-type]
                ),
                self._event_loop
            )

            # Wait for result with timeout
            return future.result(timeout=30.0)

        except concurrent.futures.TimeoutError:
            logger.error("Get history timed out")
            return {"success": False, "error": "Get history timed out"}
        except Exception as e:
            logger.exception("Error getting history: %s", e)
            return {"success": False, "error": str(e)}

    def exposed_subscribe_messages(self, callback: Any) -> bool:
        """
        Subscribe to real-time messages.

        Args:
            callback: Callback object with on_message(msg: dict) method

        Returns:
            True if subscription successful
        """
        try:
            with self._subscribers_lock:
                self._message_subscribers.add(callback)
            logger.info(f"Client subscribed to messages: {callback}")
            return True
        except Exception as e:
            logger.exception("Error subscribing: %s", e)
            return False

    def exposed_unsubscribe_messages(self, callback: Any) -> bool:
        """
        Unsubscribe from messages.

        Args:
            callback: Callback object to remove

        Returns:
            True if unsubscription successful
        """
        try:
            with self._subscribers_lock:
                self._message_subscribers.discard(callback)
            logger.info(f"Client unsubscribed from messages: {callback}")
            return True
        except Exception as e:
            logger.exception("Error unsubscribing: %s", e)
            return False

    def exposed_get_metadata(self) -> dict[str, Any]:
        """
        Get session metadata.

        Returns metadata including working_dir, pid, started_at, rpyc_port, model.
        Needed for DwSession compatibility.

        Returns:
            Response dict with metadata
        """
        if not self._agent_process:
            return {"success": False, "error": "Agent not initialized"}

        try:
            import os

            # Get working_dir from config
            working_dir = (
                self._agent_process.config.working_dir
                if self._agent_process.config
                else None
            )

            metadata = {
                "directory": str(working_dir) if working_dir else None,
                "pid": os.getpid(),  # Current process PID
                "started": (
                    self._agent_process.started_at.isoformat()
                    if self._agent_process.started_at
                    else None
                ),
                "rpyc_port": self._agent_process.rpyc_port,
                "agent_type": "deep-worker",
                "model": (
                    self._agent_process.config.model
                    if self._agent_process.config
                    else None
                ),
            }

            return {"success": True, "metadata": metadata}

        except Exception as e:
            logger.exception("Error getting metadata: %s", e)
            return {"success": False, "error": str(e)}

    def exposed_shutdown(self) -> dict[str, Any]:
        """
        Request graceful shutdown of agent process.

        Returns immediately with success status. Actual shutdown happens asynchronously.
        The RPyC server will be closed after a short delay to allow the response to be sent.

        Returns:
            Response dict with success status
        """
        if not self._agent_process:
            return {"success": False, "error": "Agent not initialized"}

        try:
            # Signal agent to shutdown gracefully
            self._agent_process.running = False

            # Close RPyC server after response is sent
            def delayed_close() -> None:
                import time

                time.sleep(0.5)  # Give time for response to be sent
                if self._agent_process and self._agent_process.rpyc_server:
                    try:
                        self._agent_process.rpyc_server.close()
                    except Exception:
                        pass

            threading.Thread(target=delayed_close, daemon=True).start()

            return {"success": True, "message": "Shutdown initiated"}

        except Exception as e:
            logger.exception("Error during shutdown: %s", e)
            return {"success": False, "error": str(e)}

    def _notify_subscribers(self, message: dict[str, Any]) -> None:
        """
        Notify all subscribers about a new message.

        Args:
            message: Message dict to send to subscribers
        """
        with self._subscribers_lock:
            subscribers = list(self._message_subscribers)

        for subscriber in subscribers:
            try:
                subscriber.on_message(message)
            except Exception as e:
                logger.exception(f"Error notifying subscriber {subscriber}: {e}")
