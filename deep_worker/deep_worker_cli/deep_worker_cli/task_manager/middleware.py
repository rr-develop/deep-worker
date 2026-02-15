"""TaskManager middleware for LangGraph integration.

This module provides a LangGraph callback handler that intercepts subordinate
agent responses and sends them to TaskManager for evaluation.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.callbacks import AsyncCallbackHandler

if TYPE_CHECKING:
    from deep_worker_cli.task_manager.task_manager import TaskManager

logger = logging.getLogger(__name__)


class TaskManagerCallback(AsyncCallbackHandler):
    """
    LangGraph callback handler for TaskManager integration.

    This callback intercepts when the subordinate agent completes a response
    and sends it to TaskManager for evaluation. If TaskManager determines the
    work is incomplete, it can send a corrective prompt back to the agent.

    Architecture:
        1. Subordinate agent completes response → on_chain_end() triggered
        2. Extract last assistant message
        3. Send to TaskManager.on_subordinate_response()
        4. TaskManager evaluates via manager thread
        5. If needed, TaskManager returns corrective prompt
        6. Callback sends corrective prompt back to agent

    Example:
        >>> tm = TaskManager(agent_name="my_agent", ...)
        >>> callback = TaskManagerCallback(task_manager=tm)
        >>> config = {"callbacks": [callback]}
        >>> result = await agent.ainvoke({"messages": [...]}, config)
        >>> # Callback automatically handles TaskManager integration
    """

    def __init__(
        self,
        task_manager: 'TaskManager',
        send_message_func: Optional[Any] = None
    ):
        """
        Initialize TaskManager callback.

        Args:
            task_manager: TaskManager instance
            send_message_func: Optional function to send messages to agent
                             Signature: async def send(message: str) -> None
        """
        super().__init__()
        self.task_manager = task_manager
        self.send_message_func = send_message_func
        self._processing = False

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Called when the agent (chain) completes execution.

        Extracts the last assistant message and sends it to TaskManager
        for evaluation.

        Args:
            outputs: Chain outputs containing messages
            **kwargs: Additional callback parameters
        """
        # Avoid recursive processing
        if self._processing:
            return

        # Only process if TaskManager has an active task
        if self.task_manager.state == 'inactive':
            return

        try:
            self._processing = True

            # Extract messages from outputs
            messages = outputs.get('messages', [])
            if not messages:
                logger.debug("[TaskManagerCallback] No messages in chain output")
                return

            # Get last message
            last_msg = messages[-1]

            # Extract content based on message type
            if isinstance(last_msg, dict):
                role = last_msg.get('role', '')
                content = last_msg.get('content', '')
            else:
                # LangChain Message object
                role = getattr(last_msg, 'type', '')
                content = getattr(last_msg, 'content', '')

            # Only process assistant messages
            if role not in ('assistant', 'ai'):
                logger.debug(f"[TaskManagerCallback] Ignoring non-assistant message (role={role})")
                return

            if not content:
                logger.debug("[TaskManagerCallback] Empty assistant message")
                return

            logger.info(f"[TaskManagerCallback] Subordinate response received ({len(content)} chars)")

            # Send to TaskManager for evaluation
            corrective_prompt = await self.task_manager.on_subordinate_response(
                response=content,
                send_to_subordinate_callback=self._send_to_subordinate
            )

            # If TaskManager returned a corrective prompt, send it to agent
            if corrective_prompt:
                logger.info(f"[TaskManagerCallback] Sending corrective prompt to subordinate")
                await self._send_to_subordinate(corrective_prompt)

        except Exception as e:
            logger.exception(f"[TaskManagerCallback] Error in on_chain_end: {e}")

        finally:
            self._processing = False

    async def _send_to_subordinate(self, message: str) -> None:
        """
        Send message to subordinate agent.

        This is passed to TaskManager as a callback so it can send
        corrective prompts.

        Args:
            message: Message to send to agent
        """
        if not self.send_message_func:
            logger.error("[TaskManagerCallback] No send_message_func configured, cannot send corrective prompt")
            return

        try:
            await self.send_message_func(message)
        except Exception as e:
            logger.exception(f"[TaskManagerCallback] Error sending message to subordinate: {e}")
