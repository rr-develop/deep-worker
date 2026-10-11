"""Middleware for conditional history memory tool registration.

Registers the ``history_memory`` tool only when:
1. The agent was started with ``--history-memory`` flag, AND
2. A history semantic index exists in the summaries directory.

Injects a cognitive hint into the system prompt to guide the agent's usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from typing_extensions import TypedDict

from deep_worker_cli.summaries import get_summaries_dir
from deep_worker_cli.history_memory_tool import history_memory

# Lazy import for append_to_system_message to avoid circular imports
_append_to_system_message = None


class HistoryMemoryState(AgentState):
    """State schema for HistoryMemoryMiddleware.

    Attributes:
        summaries_dir: Summaries directory path for history search.
    """

    summaries_dir: NotRequired[str]


COGNITIVE_HINT = """### Conversation History Memory

**CRITICAL**: This agent has a semantic index of past conversation history. Use `history_memory` to recall information from turns that were summarized and removed from context.

**PRIORITY**: Use `history_memory` when:
- You need information discussed BEFORE the current context window
- Recalling past technical decisions, requirements, or user preferences
- Finding specific details mentioned in earlier conversation turns
- Understanding context that was lost due to summarization

**When to use this tool:**
- ANY question about "what was discussed earlier" or "what did we decide"
- Recovering details about past bug reports, feature requests, or discussions
- Before making decisions that depend on earlier conversation context

**Example queries:**
- "What did we decide about the database schema?"
- "Find the error message from the earlier debugging session"
- "What were the requirements for feature X?"
- "Remind me about the authentication approach we discussed"

**Note**: Do NOT use this tool for the current conversation context (messages still visible to you) — use it only for information that was summarized and is no longer in your visible history."""

MISSING_INDEX_HINT = """### Conversation History Memory

This agent has no local history index. Use `history_memory` only with an explicit `agent_name` that appears below:

{tenants}"""


class HistoryMemoryMiddleware(AgentMiddleware):
    """Conditionally enable history memory based on flag and index availability.

    If the agent was started with ``--history-memory`` AND a history index
    exists:
    - Registers the ``history_memory`` tool
    - Injects a cognitive hint into the system prompt

    Otherwise, does nothing (tool remains unavailable).

    Args:
        assistant_id: Agent identifier (used to locate summaries dir).
        enabled: Whether the ``--history-memory`` flag was passed.
    """

    state_schema = HistoryMemoryState

    def __init__(
        self,
        assistant_id: str,
        enabled: bool = False,
        client: Any = None,
        owner_loop: Any = None,
    ):
        """Initialize middleware with agent identity and enable flag.

        Args:
            assistant_id: Agent identifier used to find summaries directory
                at ``~/.deep_worker/{assistant_id}/summaries/``.
            enabled: Whether the user passed ``--history-memory`` CLI flag.
            client: Semantic-memory client (injected in tests).
            owner_loop: Holder for the agent's event loop. The
                post-summarization callback runs on a worker thread and
                cannot look the loop up itself, so this middleware — which
                the graph awaits ON the loop — records it here.
        """
        self.assistant_id = assistant_id
        self._enabled = enabled
        self._owner_loop = owner_loop
        self._summaries_dir: Path | None = None
        self._has_index = False
        self._client = client

        if enabled:
            try:
                self._summaries_dir = get_summaries_dir(assistant_id)
            except Exception:  # noqa: BLE001
                self._summaries_dir = None

        self._cognitive_hint: str | None = None
        self.tools = [history_memory] if enabled else []

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, str] | None:
        """Set summaries_dir in state for the history_memory tool.

        This allows the history_memory tool to locate the summaries directory
        via ``runtime.state``.

        Args:
            state: Current agent state.
            runtime: Tool runtime for state access.

        Returns:
            Dict with summaries_dir if enabled, or ``None``.
        """
        if self._owner_loop is not None:
            self._owner_loop.bind()
        if self._enabled and self._summaries_dir:
            return {"summaries_dir": str(self._summaries_dir)}
        return None

    async def abefore_agent(
        self, state: AgentState, runtime: Any
    ) -> dict[str, str] | None:
        # Runs on the agent loop: the only place that can hand the
        # worker-thread callback a loop it may schedule on.
        if self._owner_loop is not None:
            self._owner_loop.bind()
        if not self._enabled or self._summaries_dir is None:
            return None
        from deep_worker_cli.semantic_memory import client as memory_client
        from deep_worker_cli.semantic_memory.service.proto import semantic_memory_pb2 as pb

        client = self._client or memory_client.SemanticMemoryClient()
        try:
            status = await client.index_status(
                pb.INDEX_KIND_HISTORY, self._summaries_dir.resolve()
            )
            self._has_index = bool(status.exists and status.complete)
            if self._has_index:
                self._cognitive_hint = COGNITIVE_HINT
            else:
                tenants = await client.list_tenants(pb.INDEX_KIND_HISTORY)
                listed = memory_client.format_tenants(tenants) or "(none)"
                self._cognitive_hint = MISSING_INDEX_HINT.format(tenants=listed)
        except Exception:
            self._has_index = False
            self._cognitive_hint = MISSING_INDEX_HINT.format(
                tenants="(service unavailable)"
            )
        return self.before_agent(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject cognitive hint into system prompt if index exists.

        Appends the cognitive hint to the system prompt to guide
        the agent's usage of the history_memory tool.
        """
        if not self._has_index or self._cognitive_hint is None:
            return handler(request)

        # Inject cognitive hint into system prompt
        global _append_to_system_message
        if _append_to_system_message is None:
            from deepagents.middleware._utils import append_to_system_message as _ats  # noqa: PLC0415
            _append_to_system_message = _ats

        new_system_message = _append_to_system_message(
            request.system_message,
            self._cognitive_hint,
        )

        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version: Inject cognitive hint into system prompt if index exists.

        Appends the cognitive hint to the system prompt to guide
        the agent's usage of the history_memory tool.
        """
        if not self._has_index or self._cognitive_hint is None:
            return await handler(request)

        # Inject cognitive hint into system prompt
        global _append_to_system_message
        if _append_to_system_message is None:
            from deepagents.middleware._utils import append_to_system_message as _ats  # noqa: PLC0415
            _append_to_system_message = _ats

        new_system_message = _append_to_system_message(
            request.system_message,
            self._cognitive_hint,
        )

        return await handler(request.override(system_message=new_system_message))
