"""TaskManager class for deep-worker agent task evaluation and correction.

This module provides TaskManager functionality similar to ai_lib TaskManager,
adapted to run within the same process as deep-worker agent using async tasks
in the main event loop (single-thread architecture).
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel

from deep_worker_cli.task_manager.prompts import (
    TASK_MANAGER_SYSTEM_PROMPT,
    format_evaluation_prompt,
    format_followup_prompt,
    parse_manager_response,
)

if TYPE_CHECKING:
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.session.session_manager import SessionManager
    from deepagents.backends.composite import CompositeBackend

logger = logging.getLogger(__name__)


class TaskManager:
    """
    TaskManager for deep-worker agent task evaluation and correction.

    This class manages task completion with AI-powered evaluation using async tasks
    in the main event loop. Similar to ai_lib TaskManager but adapted for
    single-thread, single event loop architecture.

    Architecture:
        - Main event loop: Subordinate agent executes tasks
        - Background async task: Manager agent evaluates task completion
        - Communication: Direct async calls (no queues, no threads)

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
        session_manager: 'SessionManager',
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        """
        Initialize TaskManager.

        Args:
            agent_name: Name of the subordinate agent
            config: Agent configuration for creating manager agent
            session_manager: For accessing conversation history
            checkpointer: Checkpointer for manager agent persistence.
                         MUST be same type as main agent:
                         - InMemorySaver if main agent uses InMemorySaver
                         - AsyncSqliteSaver if main agent uses AsyncSqliteSaver
                         If None, defaults to InMemorySaver (test mode)
        """
        self.agent_name = agent_name
        self.config = config
        self.session_manager = session_manager
        self.checkpointer = checkpointer

        # Manager agent components (initialized in start())
        self.manager_agent: Pregel | None = None
        self.manager_backend: 'CompositeBackend | None' = None

        # Task state
        self.state: Literal['inactive', 'waiting', 'evaluating'] = 'inactive'
        self.initial_request: str | None = None
        self.project_path: str | None = None
        self.session_file_path: str | None = None
        self.is_first_evaluation = True
        self.evaluation_count = 0

        # Callback for sending corrective prompts to subordinate
        self.send_to_subordinate_callback = None

        # Running flag for lifecycle management
        self._running = False

    async def start(self) -> None:
        """
        Start the TaskManager and initialize manager agent.

        This initializes the manager LangGraph agent that will evaluate
        the subordinate agent's work. No threads are created - the manager
        runs in the same event loop as the main agent.
        """
        if self._running:
            logger.warning(f"[TaskManager] Already running")
            return

        logger.info(f"[TaskManager] Starting for agent '{self.agent_name}'")

        # Initialize manager agent
        await self._initialize_manager()

        self._running = True
        logger.info(f"[TaskManager] Started successfully (manager_id: {self.agent_name}:tm)")

    async def _initialize_manager(self) -> None:
        """
        Initialize manager LangGraph agent.

        Creates a LangGraph agent with TaskManager system prompt for
        evaluating subordinate agent's work. This runs in the same event
        loop as the main agent.
        """
        try:
            from deep_worker_cli.agent import create_cli_agent
            from deep_worker_cli.config import create_model

            logger.info(f"[TaskManager] Initializing manager agent...")

            # Create model using same configuration as subordinate
            model = create_model()

            # Get project_memory_dir from config (required by create_cli_agent)
            project_memory_dir = Path(self.config.working_dir)

            # Create manager agent with TaskManager system prompt
            # CRITICAL: Use colon ':' not underscore '_' to avoid conflicts
            # with agent names that contain '_tm' suffix
            self.manager_agent, self.manager_backend = create_cli_agent(
                model=model,
                assistant_id=f"{self.agent_name}:tm",  # Colon separator
                system_prompt=TASK_MANAGER_SYSTEM_PROMPT,
                auto_approve=True,  # Manager doesn't need approvals
                enable_memory=False,  # Manager doesn't need memory
                enable_skills=False,  # Manager doesn't use skills
                enable_shell=False,  # Manager doesn't execute shell commands
                project_memory_dir=project_memory_dir,
                checkpointer=self.checkpointer,  # Same as main agent
            )

            logger.info(f"[TaskManager] Manager agent initialized")

        except Exception as e:
            logger.exception(f"[TaskManager] Failed to initialize manager agent: {e}")
            raise

    async def init_task(
        self,
        initial_request: str,
        project_path: str | None = None,
        session_file_path: str | None = None
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
        send_to_subordinate_callback: Any,
    ) -> str | None:
        """
        Handle subordinate agent response and evaluate it.

        This is called by middleware when subordinate completes a response.
        Evaluation happens directly in this method (no queues, no threads).

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

            # Invoke manager agent directly (no thread, no queue)
            manager_response = await self._evaluate_with_manager(eval_prompt)

            if not manager_response:
                logger.error(f"[TaskManager] Evaluation failed")
                self.state = 'inactive'
                return None

            # Parse manager response
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

    async def _evaluate_with_manager(self, prompt: str) -> str | None:
        """
        Evaluate a prompt using the manager agent.

        This method directly invokes the manager agent in the current event loop.
        No threads, no queues - just a direct async call.

        Args:
            prompt: The evaluation prompt to send to manager

        Returns:
            Manager agent's response content, or None on error
        """
        try:
            if self.manager_agent is None:
                logger.error(f"[TaskManager] Manager agent not initialized")
                return None

            logger.debug(f"[TaskManager] Invoking manager agent with prompt length={len(prompt)}")

            # Invoke manager agent
            # CRITICAL: Metadata field meanings:
            # - thread_id: Unique identifier for this conversation thread (agent_name:tm)
            # - assistant_id: Technical ID for the assistant/agent instance (agent_name:tm)
            # - agent_name: The MAIN agent name (WITHOUT :tm suffix) - identifies which agent this thread belongs to
            # - updated_at: ISO 8601 UTC timestamp of last update
            #
            # This ensures TaskManager threads are associated with the parent agent in --list output
            result = await self.manager_agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {
                    "configurable": {"thread_id": f"{self.agent_name}:tm"},
                    "metadata": {
                        "assistant_id": f"{self.agent_name}:tm",
                        "agent_name": self.agent_name,  # Parent agent name (without :tm)
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                }
            )

            # Extract response
            messages = result.get("messages", [])
            if not messages:
                logger.error(f"[TaskManager] Manager agent returned no messages")
                return None

            last_msg = messages[-1]

            # Handle both dict and LangChain Message objects
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            else:
                content = last_msg.content if hasattr(last_msg, 'content') else ""

            logger.debug(f"[TaskManager] Manager agent responded with {len(content)} chars")

            return content

        except Exception as e:
            logger.exception(f"[TaskManager] Error invoking manager agent: {e}")
            return None

    async def stop(self) -> None:
        """
        Stop the TaskManager and cleanup manager agent.

        This is a graceful shutdown that doesn't need to wait for threads
        since everything runs in the same event loop.
        """
        if not self._running:
            return

        logger.info(f"[TaskManager] Stopping...")

        self.state = 'inactive'
        self._running = False

        # Cleanup manager backend if it has a cleanup method
        if self.manager_backend and hasattr(self.manager_backend, 'cleanup'):
            with contextlib.suppress(Exception):
                cleanup_method = getattr(self.manager_backend, 'cleanup')
                if callable(cleanup_method):
                    await cleanup_method()

        self.manager_agent = None
        self.manager_backend = None

        logger.info(f"[TaskManager] Stopped successfully")

    def get_state(self) -> str:
        """
        Get current TaskManager state.

        Returns:
            Current state: 'inactive', 'waiting', or 'evaluating'
        """
        return self.state
