"""TaskManager class for deep-worker agent task evaluation and correction.

This module provides TaskManager functionality similar to ai_lib TaskManager,
adapted to run within the same process as deep-worker agent using a separate thread.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from deep_worker_cli.task_manager.manager_thread import ManagerThread
from deep_worker_cli.task_manager.prompts import (
    format_evaluation_prompt,
    format_followup_prompt,
    parse_manager_response,
)

if TYPE_CHECKING:
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class TaskManager:
    """
    TaskManager for deep-worker agent task evaluation and correction.

    This class manages task completion with AI-powered evaluation using a
    separate manager thread. Similar to ai_lib TaskManager but adapted for
    deep-worker architecture.

    Architecture:
        - Main thread: Subordinate agent executes tasks
        - Manager thread ({agent_name}:tm): Evaluates task completion
        - Communication: asyncio.Queue between threads

    Example:
        >>> tm = TaskManager(agent_name="my_agent", config=config, session_manager=sm)
        >>> await tm.start()
        >>> await tm.init_task("Create function with tests", project_path="/tmp/project")
        >>> # TaskManager will automatically evaluate responses via middleware
        >>> await tm.stop()
    """

    def __init__(
        self,
        agent_name: str,
        config: 'AgentConfig',
        session_manager: 'SessionManager'
    ):
        """
        Initialize TaskManager.

        Args:
            agent_name: Name of the subordinate agent (for thread naming)
            config: Agent configuration for creating manager agent
            session_manager: For accessing conversation history
        """
        self.agent_name = agent_name
        self.config = config
        self.session_manager = session_manager

        # Inter-thread communication queues
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()

        # Manager thread
        self.manager_thread: Optional[ManagerThread] = None

        # Task state
        self.state: Literal['inactive', 'waiting', 'evaluating'] = 'inactive'
        self.initial_request: Optional[str] = None
        self.project_path: Optional[str] = None
        self.session_file_path: Optional[str] = None
        self.is_first_evaluation = True
        self.evaluation_count = 0

        # Callback for sending corrective prompts to subordinate
        self.send_to_subordinate_callback = None

    async def start(self) -> None:
        """
        Start the TaskManager and initialize manager thread.

        This creates and starts the manager thread that will process
        evaluation requests.
        """
        if self.manager_thread and self.manager_thread.is_alive():
            logger.warning(f"[TaskManager] Manager thread already running")
            return

        logger.info(f"[TaskManager] Starting manager thread for agent '{self.agent_name}'")

        # Create and start manager thread
        self.manager_thread = ManagerThread(
            agent_name=self.agent_name,
            task_manager=self
        )
        self.manager_thread.start()

        # Wait a bit for thread to initialize
        await asyncio.sleep(0.1)

        if not self.manager_thread.is_alive():
            raise RuntimeError("Manager thread failed to start")

        logger.info(f"[TaskManager] Manager thread '{self.manager_thread.name}' started")

    async def init_task(
        self,
        initial_request: str,
        project_path: Optional[str] = None,
        session_file_path: Optional[str] = None
    ) -> None:
        """
        Initialize a new task for evaluation.

        Args:
            initial_request: The user's original task request
            project_path: Optional project directory path
            session_file_path: Optional path to subordinate's session file
        """
        logger.info(f"[TaskManager] Initializing task: {initial_request[:100]}...")

        self.initial_request = initial_request
        self.project_path = project_path
        self.session_file_path = session_file_path
        self.is_first_evaluation = True
        self.evaluation_count = 0
        self.state = 'waiting'

        logger.debug(f"[TaskManager] Task initialized, waiting for subordinate response")

    async def on_subordinate_response(
        self,
        response: str,
        send_to_subordinate_callback
    ) -> Optional[str]:
        """
        Handle subordinate agent response and evaluate it.

        This is called by middleware when subordinate completes a response.

        Args:
            response: The subordinate's response text
            send_to_subordinate_callback: Callback function to send corrective prompts

        Returns:
            Corrective prompt if needed, None if task is done/unable
        """
        if self.state == 'inactive':
            # No active task, ignore
            return None

        if self.state == 'evaluating':
            # Already evaluating, ignore (shouldn't happen)
            logger.warning(f"[TaskManager] Already evaluating, ignoring response")
            return None

        # Store callback for sending corrections
        self.send_to_subordinate_callback = send_to_subordinate_callback

        # Start evaluation
        self.state = 'evaluating'
        self.evaluation_count += 1

        logger.info(f"[TaskManager] Evaluation #{self.evaluation_count} starting...")

        try:
            # Prepare evaluation prompt
            if self.is_first_evaluation:
                if not self.initial_request:
                    logger.error(f"[TaskManager] No initial request set")
                    self.state = 'inactive'
                    return None

                eval_prompt = format_evaluation_prompt(
                    initial_request=self.initial_request,
                    last_response=response,
                    session_file_path=self.session_file_path,
                    project_path=self.project_path
                )
                self.is_first_evaluation = False
            else:
                eval_prompt = format_followup_prompt(response)

            # Send evaluation request to manager thread
            await self.task_queue.put({
                'type': 'evaluate',
                'prompt': eval_prompt
            })

            # Wait for evaluation result
            result = await self.result_queue.get()

            if not result.get('success'):
                error = result.get('error', 'Unknown error')
                logger.error(f"[TaskManager] Evaluation failed: {error}")
                self.state = 'inactive'
                return None

            # Parse manager response
            manager_response = result.get('content', '')
            status, message = parse_manager_response(manager_response)

            logger.info(f"[TaskManager] Evaluation result: status={status}")

            # Handle evaluation result
            if status == 'done':
                logger.info(f"[TaskManager] Task completed successfully!")
                self.state = 'inactive'
                return None

            elif status == 'unable':
                logger.warning(f"[TaskManager] Task unable: {message}")
                self.state = 'inactive'
                return None

            elif status == 'in_progress':
                preview = message[:100] if message else ""
                logger.info(f"[TaskManager] Sending corrective prompt: {preview}...")
                self.state = 'waiting'  # Wait for next response
                # Return corrective prompt with TaskManager prefix
                return f"TaskManager:> {message}"

        except Exception as e:
            logger.exception(f"[TaskManager] Error during evaluation: {e}")
            self.state = 'inactive'
            return None

    async def stop(self) -> None:
        """
        Stop the TaskManager and shutdown manager thread.
        """
        logger.info(f"[TaskManager] Stopping...")

        self.state = 'inactive'

        if self.manager_thread and self.manager_thread.is_alive():
            self.manager_thread.stop()

            # Wait for thread to finish (with timeout)
            self.manager_thread.join(timeout=5.0)

            if self.manager_thread.is_alive():
                logger.warning(f"[TaskManager] Manager thread did not stop within timeout")
            else:
                logger.info(f"[TaskManager] Manager thread stopped")

        self.manager_thread = None

    def get_state(self) -> str:
        """
        Get current TaskManager state.

        Returns:
            Current state: 'inactive', 'waiting', or 'evaluating'
        """
        return self.state
