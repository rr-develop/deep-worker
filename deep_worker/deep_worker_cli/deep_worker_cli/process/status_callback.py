"""Callback handler for tracking agent status during tool execution."""

import sys
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from deep_worker_cli.exceptions import is_fatal_exception
from deep_worker_cli.types import AgentStatus


class StatusTrackingCallback(AsyncCallbackHandler):
    """Callback handler that tracks tool execution and updates agent status.

    This handler listens to tool lifecycle events and updates the session manager
    status accordingly:
    - on_tool_start: Set status to WAITING_INPUT (tool is executing)
    - on_tool_end: Set status back to RUNNING (tool completed)
    - on_tool_error: Classify per the No-Agent-Error-State Policy —
        * fatal (``is_fatal_exception`` → True): persist error via
          ``session_manager.set_error(str(exc))`` — this is the ONLY path
          that surfaces ``AgentStatus.ERROR`` from this callback.
        * recoverable (default): set status back to ``RUNNING``. The
          LangChain runtime already converts the exception into a
          ``ToolMessage(status="error")`` visible to the model, which will
          think and retry. Flipping the agent to ERROR here would violate
          POLICIES.md §1 (see
          docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md and
          docs/DESIGN_FIX_TOOL_VALIDATION_ERROR_STATE_AND_GROK_ARG_COMPAT.md).

    This enables proper status tracking: ready → working → waiting_input → working → ready.
    """

    def __init__(self, session_manager: Optional[Any] = None):
        """Initialize the callback handler.

        Args:
            session_manager: SessionManager instance to update status
        """
        super().__init__()
        self.session_manager = session_manager

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Called when tool starts executing.

        Args:
            serialized: Serialized tool info
            input_str: Input to the tool
            **kwargs: Additional arguments
        """
        if self.session_manager:
            await self.session_manager.set_status(AgentStatus.WAITING_INPUT)

    async def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """Called when tool finishes executing.

        Args:
            output: Tool output
            **kwargs: Additional arguments
        """
        if self.session_manager:
            # Return to RUNNING status (model will process tool output)
            await self.session_manager.set_status(AgentStatus.RUNNING)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Called when tool execution fails.

        Per the No-Agent-Error-State Policy (POLICIES.md §1) the agent
        MUST NOT flip to ``AgentStatus.ERROR`` for recoverable tool
        failures (pydantic validation, sloppy tool exceptions, MCP
        errors, etc.) — LangChain surfaces them to the model as a
        ``ToolMessage(status="error")`` and the model retries. Only
        genuinely fatal conditions (``is_fatal_exception``) transition
        the agent to ERROR here.

        Args:
            error: Exception that occurred
            run_id: Run ID
            parent_run_id: Parent run ID
            tags: Tags
            **kwargs: Additional arguments
        """
        # 1) Post-mortem log line — format MUST stay bit-for-bit stable
        #    (grep patterns / post-mortem tooling depend on it).
        fields = [f"error={error}", f"run_id={run_id}"]
        if parent_run_id is not None:
            fields.append(f"parent_run_id={parent_run_id}")
        print(
            f"[{datetime.now(UTC).isoformat(timespec='microseconds')}] [TOOL_ERROR] {' '.join(fields)}",
            file=sys.stderr,
        )
        # 2) Status transition — fatal vs recoverable per policy.
        if not self.session_manager:
            return
        if is_fatal_exception(error):
            # Fatal → persist error message + flip to ERROR (matches the
            # fatal path in agent_process._handle_invoke).
            await self.session_manager.set_error(str(error))
        else:
            # Recoverable → the model will process a ToolMessage and
            # retry; mirror on_tool_end and return to RUNNING.
            await self.session_manager.set_status(AgentStatus.RUNNING)
