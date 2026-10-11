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


def assemble_manager_web_tools(config: "AgentConfig") -> list[Any]:
    """Production TaskManager web-tools assembly (``_initialize_manager`` call-site).

    Returns ``[fetch_url]`` plus the universal web-search delegation tool when
    the tri-state policy resolves it. Extracted so the exact production
    assembly entry point is directly exercisable (DESIGN §5.5).
    """
    from deep_worker_cli.config import WebSearchPolicy
    from deep_worker_cli.tools import (
        fetch_url,
        resolve_web_search_tool,
    )

    manager_tools: list[Any] = [fetch_url]
    _web_search_avail = resolve_web_search_tool(
        WebSearchPolicy(getattr(config, "web_search_policy", "auto"))
    )
    if _web_search_avail.tool is not None:
        manager_tools.insert(0, _web_search_avail.tool)
    return manager_tools


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

    Manager bound tool set (all providers):
        - ``web_search`` (universal ``web_search_delegation_tool``):
          read-only web search via Claude Code SDK; no API key.
        - ``fetch_url``: fetch a known URL and return its content as
          markdown. Both are advertised by name in
          ``TASK_MANAGER_SYSTEM_PROMPT`` so the model knows when to
          reach for them.
        - When ``config.provider == "codex"``: also the eight
          ``codex_provider.native_tools.NATIVE_TOOLS`` passthrough
          tools, so a native Codex tool call during a manager turn
          does not crash it. See
          ``codex_provider/docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md``.

    Manager bound tool set deliberately omits:
        - ``http_request`` (POST/PUT/DELETE — unnecessary write surface
          for an evaluator).
        - ``send_message_to_agent`` (autonomous peer messaging — outside
          the evaluator role).
        - ``read_image`` (not a web tool; image inspection out of scope).

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
        token_stats_recorder: Any | None = None,
        auto_approve: bool = False,
        commit: str | None = None,
        dirty: bool | None = None,
        disabled: bool = False,
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
            token_stats_recorder: Shared token-stats recorder (owned by the
                         main AgentProcess) so TM invocations land in the same
                         ``token_stats.jsonl`` with ``kind="tm"``
                         (token-stats design §5.1).
            auto_approve: Mirrors the operator's ``-sp`` flag (owner-gate
                         Q2/Q5): permission prompts are auto-approved ONLY
                         when the OPERATOR opted in. The TaskManager never
                         grants auto-approval on its own — it inherits the
                         main agent's explicit flag.
            commit: Git SHA of the tree this process was started from, or
                         None when it could not be determined. Forwarded so
                         infrastructure feedback filed by the TaskManager
                         carries the same code revision as the main agent's.
            dirty: Whether that tree had uncommitted changes at startup.
                         Tri-state: False = clean, True = modified,
                         None = undetermined (never fabricate "clean").
            disabled: The operator's persisted off switch, replayed on start.
                         This is NOT a launch flag: whether a TaskManager
                         exists at all is ``launch_config.task_manager``. This
                         argument restores a ``set_task_manager(enabled=False)``
                         that the operator issued in an earlier process, so a
                         restart does not silently resurrect the manager
                         (reported 2026-10-10). Default False
                         applies only when the switch was never set.
        """
        self.agent_name = agent_name
        self.config = config
        self.session_manager = session_manager
        self.checkpointer = checkpointer
        self.token_stats_recorder = token_stats_recorder
        self.auto_approve = auto_approve
        self.commit = commit
        self.dirty = dirty

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

        # Runtime control flags
        self._running = False
        # Seeded from the persisted operator switch, not hardcoded False:
        # a restart must not re-enable a manager the operator turned off.
        self.disabled = disabled

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

    def disable(self) -> None:
        """Temporarily disable TaskManager evaluation."""
        self.disabled = True
        logger.info(f"[TaskManager] Disabled for '{self.agent_name}'")

    def enable(self) -> None:
        """Re-enable TaskManager evaluation."""
        self.disabled = False
        logger.info(f"[TaskManager] Enabled for '{self.agent_name}'")

    async def reset(self) -> None:
        """Reset TaskManager to initial state.

        Clears evaluation history, reinitializes manager agent.
        After reset, TM behaves as if just started.
        The disabled flag is preserved.
        """
        import contextlib

        logger.info(f"[TaskManager] Resetting...")

        # 1. Reset task state
        self.state = 'inactive'
        self.initial_request = None
        self.project_path = None
        self.session_file_path = None
        self.is_first_evaluation = True
        self.evaluation_count = 0

        # 2. Clear checkpoint history for TM thread
        if self.checkpointer is not None:
            thread_id = f"{self.agent_name}:tm"
            await self._clear_thread_history(thread_id)

        # 3. Cleanup old backend
        if self.manager_backend and hasattr(self.manager_backend, 'cleanup'):
            with contextlib.suppress(Exception):
                cleanup_method = getattr(self.manager_backend, 'cleanup')
                if callable(cleanup_method):
                    await cleanup_method()

        # 4. Reinitialize manager agent (fresh instance)
        self.manager_agent = None
        self.manager_backend = None
        await self._initialize_manager()

        logger.info(f"[TaskManager] Reset complete")

    async def _clear_thread_history(self, thread_id: str) -> None:
        """Clear all checkpoint data for a specific thread."""
        from typing import Any
        conn: Any = getattr(self.checkpointer, 'conn', None)
        if conn is not None:
            # AsyncSqliteSaver — clear via SQL
            await conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
            await conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
            )
            await conn.commit()
            logger.info(f"[TaskManager] Cleared checkpoint history for thread '{thread_id}'")
        # InMemorySaver — reinitializing manager_agent is sufficient

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
            from deep_worker_cli.stats.callback import (
                make_stats_callback_factory,
                make_usage_event_callback_factory,
            )

            logger.info(f"[TaskManager] Initializing manager agent...")

            # Create model using same provider and model as subordinate agent.
            # This ensures TaskManager uses the same AI backend (e.g. claude-code-sdk)
            # as the main agent, not auto-detected DeepSeek.
            # Pass user=f"{self.agent_name}:tm" for KV-cache isolation so the
            # TaskManager thread does NOT share a cache with the main agent
            # thread — they have different system prompts and conversation
            # prefixes, so sharing the cache would cause mutual eviction.
            stats_kwargs: dict[str, Any] = {}
            if self.token_stats_recorder is not None:
                stats_kwargs = {
                    "kind": "tm",
                    "stats_callback_factory": make_stats_callback_factory(
                        self.token_stats_recorder
                    ),
                    "usage_event_callback_factory": make_usage_event_callback_factory(
                        self.token_stats_recorder
                    ),
                }
            model = create_model(
                provider=self.config.provider,
                model_name=self.config.model,
                user=f"{self.agent_name}:tm",
                auto_approve=self.auto_approve,
                **stats_kwargs,
            )

            # Set working_dir on model for providers that need it (e.g. claude-code-sdk
            # uses it for MCP tool execution with project_memory)
            # Uses setattr() because create_model() returns BaseChatModel
            # but only ClaudeCodeChatModel has _working_dir attribute.
            if hasattr(model, '_working_dir'):
                setattr(model, '_working_dir', str(self.config.working_dir))

            # Get project_memory_dir from config (required by create_cli_agent)
            project_memory_dir = Path(self.config.working_dir)

            # ---- Web tools (all providers) ----
            # The manager needs to verify factual claims the subordinate
            # made and to read documentation/specs by URL when the
            # subordinate's solution should match an external reference.
            # Two read-only web tools are enough for that:
            #
            #   web_search (delegation) — universal web search via
            #       Claude Code SDK; no extra API key required, works
            #       with every provider. Resolved lazily under the
            #       tri-state policy (auto/enabled/disabled).
            #   fetch_url                  — fetch a known URL and
            #       return its content as markdown.
            #
            # We deliberately omit ``http_request`` (supports POST/PUT/
            # DELETE — unnecessary write surface for an evaluator role)
            # and ``send_message_to_agent`` (would give the manager
            # autonomous peer-to-peer agent messaging, outside its
            # role). The TASK_MANAGER_SYSTEM_PROMPT advertises these
            # tools by name so the model knows to use them when needed.
            manager_tools = assemble_manager_web_tools(self.config)

            # ---- Provider-specific tool wiring (codex native passthrough) ----
            # Mirrors the main-agent path in ``agent_process.py``: when
            # the subordinate runs against Codex CLI, the manager runs
            # against it too (same ``create_model`` call above), and the
            # same set of native passthrough tools is required. Codex
            # hard-codes a handful of native tools that no ``--disable``
            # flag strips (``apply_patch``, ``update_plan``,
            # ``view_image``, ``request_user_input``,
            # ``list_mcp_resources*``). If the model invokes any of them
            # in the manager turn and the matching BaseTool is not in
            # the bound tool set, LangGraph's ToolNode raises
            # ``Tool '<name>' not found`` and the manager turn dies.
            # See codex_provider/docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md
            # and ``deep_worker_cli/process/agent_process.py:475-477``.
            if self.config.provider == "codex":
                # Resolve the vendored provider through provider_paths (source
                # root on sys.path in source mode; no-op in installed mode).
                # See deep_worker_cli/provider_paths.py.
                from deep_worker_cli.provider_paths import ensure_provider_importable
                from deep_worker_cli.exceptions import CapabilityError

                resolution = ensure_provider_importable(
                    "deep_worker.providers.codex_provider"
                )
                if resolution.unavailable:
                    raise CapabilityError(
                        capability="codex",
                        remedy=resolution.remedy or resolution.unavailable,
                    )
                from deep_worker.providers.codex_provider.native_tools import NATIVE_TOOLS
                manager_tools.extend(NATIVE_TOOLS)

            # Same rationale for grok: its built-ins always run locally and
            # the model may invoke one in the manager turn. The matching
            # passthrough BaseTool must be bound or ToolNode raises
            # ``Tool '<name>' not found``. See
            # grok_provider/docs/DESIGN_GROK_PROVIDER.md §11A.4.
            if self.config.provider == "grok":
                # Resolve the vendored provider through provider_paths (source
                # root on sys.path in source mode; no-op in installed mode).
                # See deep_worker_cli/provider_paths.py.
                from deep_worker_cli.provider_paths import ensure_provider_importable
                from deep_worker_cli.exceptions import CapabilityError

                resolution = ensure_provider_importable(
                    "deep_worker.providers.grok_provider"
                )
                if resolution.unavailable:
                    raise CapabilityError(
                        capability="grok",
                        remedy=resolution.remedy or resolution.unavailable,
                    )
                from deep_worker.providers.grok_provider.builtin_tool_passthrough import (
                    build_passthrough_tools,
                )
                manager_tools.extend(build_passthrough_tools())

            # Every OTHER ACP provider: same channel-B recording, selected by
            # CAPABILITY rather than by provider name. The model carries the
            # profile, and the profile declares which built-ins the CLI runs
            # locally, so nothing here needs to know who the provider is.
            #
            # Without this the manager thread loses every native tool result
            # the same way the main thread did (2026-09-28):
            # ToolNode takes its unknown-name path and checkpoints
            # ``ToolMessage(status="error", "Error: Bash is not a valid
            # tool...")`` WITHOUT running anything, while the CLI had already
            # executed the command.
            #
            # grok is handled above by its own pinned entry point (it must keep
            # the ``GrokBuiltin`` name, a public contract), so it is excluded
            # here. codex is NOT excluded by name: it reaches this branch and
            # binds nothing because it declares no ACP built-ins. Its native
            # tools were already added via ``NATIVE_TOOLS`` above.
            elif self.config.provider != "grok":
                from deep_worker.providers.acp.builtin_tool_passthrough import (
                    build_passthrough_tools_for_profile,
                )

                # ``_profile`` FIRST, and it is load-bearing.
                #
                # ``AcpChatModel`` exposes the profile as a ``@property`` while
                # ALSO inheriting ``profile`` as a pydantic field. Pydantic
                # strips the property off the class and the instance
                # ``__dict__`` shadows it, so ``getattr(model, "profile")``
                # hands back the raw ``property`` OBJECT — not the profile. It
                # is not None, so the guard below passed, and
                # ``build_passthrough_tools_for_profile`` then read no
                # ``known_builtins`` off it and returned ``()``: ZERO recorders
                # in the manager thread too, and ToolNode resumed fabricating
                # "Error: Bash is not a valid tool" without running anything.
                # See the twin fix in ``process/agent_process.py``.
                #
                # NB: ``profile`` is also a field on langchain's BaseChatModel,
                # so every model HAS the attribute — it is the VALUE that
                # discriminates. Non-ACP providers leave it at its ``None``
                # default, and even a populated upstream ``ModelProfile``
                # (a plain dict, no ``known_builtins``) yields ``()``.
                _profile = getattr(model, "_profile", None) or getattr(
                    model, "profile", None
                )
                if _profile is not None:
                    manager_tools.extend(
                        build_passthrough_tools_for_profile(_profile)
                    )

            # Create manager agent with TaskManager system prompt
            # CRITICAL: Use colon ':' not underscore '_' to avoid conflicts
            # with agent names that contain '_tm' suffix
            self.manager_agent, self.manager_backend = create_cli_agent(
                model=model,
                assistant_id=f"{self.agent_name}:tm",  # Colon separator
                system_prompt=TASK_MANAGER_SYSTEM_PROMPT,
                tools=manager_tools or None,
                auto_approve=True,  # Manager doesn't need approvals
                enable_memory=False,  # Manager doesn't need memory
                enable_skills=False,  # Manager doesn't use skills
                enable_shell=False,  # Manager doesn't execute shell commands
                project_memory_dir=project_memory_dir,
                checkpointer=self.checkpointer,  # Same as main agent
                commit=self.commit,
                dirty=self.dirty,
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
        if self.disabled:
            return

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
        if self.disabled or self.state == 'inactive':
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
            # Let a fatal error (any provider's usage/session-limit, etc.)
            # propagate (see _evaluate_with_manager): the agent must surface
            # ERROR + failover, not silently end the TaskManager loop.
            # (observed live); codex regression 2026-06-17 (see a related
            # internal incident analysis (not included here)).
            # Provider-agnostic via is_fatal_exception.
            from deep_worker_cli.exceptions import is_fatal_exception
            if is_fatal_exception(e):
                self.state = 'inactive'
                raise
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

            # Log token usage with cache info for DeepSeek monitoring
            try:
                from deep_worker_cli.utils import log_token_usage
                last = messages[-1]
                rm = getattr(last, 'response_metadata', None) or {}
                log_token_usage(logger, f"{self.agent_name}:tm", rm, "tm")
            except Exception:
                pass

            return content

        except Exception as e:
            # A fatal hit (any provider's usage/session-limit, auth-fatal,
            # checkpoint-corrupt, …) must NOT be swallowed into None — that
            # made the evaluation fail silently and the agent settle to idle
            # with no error (observed live; codex regression
            # 2026-06-17, see a related internal incident analysis (not
            # included here)). Let it propagate so the invoke
            # path (agent_process.py) runs set_error + _schedule_failover. We
            # reuse the SINGLE provider-agnostic registry is_fatal_exception
            # instead of hard-coding one provider's class name.
            from deep_worker_cli.exceptions import is_fatal_exception
            if is_fatal_exception(e):
                raise
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
