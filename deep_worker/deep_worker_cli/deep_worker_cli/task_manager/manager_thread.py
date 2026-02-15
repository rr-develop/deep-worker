"""Manager thread implementation for TaskManager.

This module provides a separate thread that runs the manager LangGraph agent
for task evaluation. The thread communicates with the main thread via asyncio queues.
"""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from deep_worker_cli.task_manager.task_manager import TaskManager

logger = logging.getLogger(__name__)


class ManagerThread(threading.Thread):
    """
    Separate thread for running TaskManager evaluation agent.

    This thread:
    - Creates its own asyncio event loop
    - Initializes a LangGraph agent for task evaluation
    - Processes evaluation requests from main thread via queue
    - Returns results back to main thread via queue

    Thread naming: If agent is named "my_agent", this thread is named "my_agent:tm"
    """

    def __init__(self, agent_name: str, task_manager: 'TaskManager'):
        """
        Initialize manager thread.

        Args:
            agent_name: Name of the subordinate agent (for thread naming)
            task_manager: TaskManager instance (for accessing queues and config)
        """
        super().__init__(name=f"{agent_name}:tm", daemon=True)
        self.agent_name = agent_name
        self.task_manager = task_manager
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.manager_agent = None
        self.manager_backend = None
        self.running = False

    def run(self) -> None:
        """
        Main thread entry point.

        Creates a new event loop, initializes manager agent, and processes
        evaluation requests until stopped.
        """
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            logger.info(f"[{self.name}] Manager thread started")

            # Initialize manager agent
            self.loop.run_until_complete(self._initialize_manager())

            # Process evaluation requests
            self.loop.run_until_complete(self._process_evaluations())

        except Exception as e:
            logger.exception(f"[{self.name}] Error in manager thread: {e}")
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            logger.info(f"[{self.name}] Manager thread stopped")

    async def _initialize_manager(self) -> None:
        """
        Initialize manager LangGraph agent.

        Creates a LangGraph agent with TaskManager system prompt for
        evaluating subordinate agent's work.
        """
        try:
            from deep_worker_cli.agent import create_cli_agent
            from deep_worker_cli.config import create_model
            from deep_worker_cli.task_manager.prompts import TASK_MANAGER_SYSTEM_PROMPT

            logger.info(f"[{self.name}] Initializing manager agent...")

            # Create model using same configuration as subordinate
            model = create_model()

            # Create manager agent with TaskManager system prompt
            self.manager_agent, self.manager_backend = create_cli_agent(
                model=model,
                assistant_id=f"{self.agent_name}_tm",
                system_prompt=TASK_MANAGER_SYSTEM_PROMPT,
                auto_approve=True,  # Manager doesn't need approvals
                enable_memory=False,  # Manager doesn't need memory
                enable_skills=False,  # Manager doesn't use skills
                enable_shell=False,  # Manager doesn't execute shell commands
            )

            logger.info(f"[{self.name}] Manager agent initialized")

        except Exception as e:
            logger.exception(f"[{self.name}] Failed to initialize manager agent: {e}")
            raise

    async def _process_evaluations(self) -> None:
        """
        Main loop: process evaluation requests from queue.

        Receives evaluation requests from main thread, invokes manager agent,
        and sends results back.
        """
        self.running = True
        logger.info(f"[{self.name}] Starting evaluation processing loop")

        try:
            while self.running:
                try:
                    # Get request from queue (with timeout to allow periodic checks)
                    request = await asyncio.wait_for(
                        self.task_manager.task_queue.get(),
                        timeout=1.0
                    )

                except asyncio.TimeoutError:
                    # No request, continue loop
                    continue

                # Handle request
                request_type = request.get('type')

                if request_type == 'stop':
                    logger.info(f"[{self.name}] Received stop request")
                    self.running = False
                    break

                elif request_type == 'evaluate':
                    logger.debug(f"[{self.name}] Received evaluation request")
                    await self._handle_evaluation(request)

                else:
                    logger.warning(f"[{self.name}] Unknown request type: {request_type}")

        except Exception as e:
            logger.exception(f"[{self.name}] Error in evaluation processing loop: {e}")
            self.running = False

    async def _handle_evaluation(self, request: Dict[str, Any]) -> None:
        """
        Handle evaluation request from main thread.

        Args:
            request: Request dictionary with 'prompt' key
        """
        try:
            prompt = request.get('prompt')
            if not prompt:
                logger.error(f"[{self.name}] Evaluation request missing 'prompt'")
                await self.task_manager.result_queue.put({
                    'success': False,
                    'error': 'Missing prompt in evaluation request'
                })
                return

            logger.debug(f"[{self.name}] Invoking manager agent with prompt length={len(prompt)}")

            # Invoke manager agent
            result = await self.manager_agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": f"{self.agent_name}_tm"}}
            )

            # Extract response
            messages = result.get("messages", [])
            if not messages:
                logger.error(f"[{self.name}] Manager agent returned no messages")
                await self.task_manager.result_queue.put({
                    'success': False,
                    'error': 'Manager agent returned no messages'
                })
                return

            last_msg = messages[-1]

            # Handle both dict and LangChain Message objects
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            else:
                content = last_msg.content if hasattr(last_msg, 'content') else ""

            logger.debug(f"[{self.name}] Manager agent responded with {len(content)} chars")

            # Send result back to main thread
            await self.task_manager.result_queue.put({
                'success': True,
                'content': content
            })

        except Exception as e:
            logger.exception(f"[{self.name}] Error handling evaluation: {e}")
            await self.task_manager.result_queue.put({
                'success': False,
                'error': str(e)
            })

    def stop(self) -> None:
        """
        Stop the manager thread.

        This is a synchronous method that can be called from main thread.
        """
        logger.info(f"[{self.name}] Stopping manager thread...")
        self.running = False

        # Put stop request in queue (if loop exists)
        if self.loop and not self.loop.is_closed():
            try:
                # Schedule stop request in thread's event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.task_manager.task_queue.put({'type': 'stop'}),
                    self.loop
                )
                # Wait for it to complete (with timeout)
                future.result(timeout=2.0)
            except Exception as e:
                logger.warning(f"[{self.name}] Error sending stop request: {e}")
