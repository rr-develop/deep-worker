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


def _history_index_exists(summaries_dir: Path) -> bool:
    """Check if a history index exists under *summaries_dir*.

    Auto path (DESIGN_PROVIDERS_FROM_WHEEL §3.6): when the semantic-search
    extra is unavailable the probe resolves lazily and reports ``False``
    (no raise) — the middleware simply stays disabled.
    """
    try:
        from deep_worker_cli.provider_paths import ensure_provider_importable

        resolution = ensure_provider_importable("deep_worker.providers.semantic_search")
        if resolution.unavailable:
            return False
        from deep_worker.providers.semantic_search.history_indexer import history_index_exists  # noqa: E402
        return history_index_exists(summaries_dir)
    except ImportError:
        return False


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

    def __init__(self, assistant_id: str, enabled: bool = False):
        """Initialize middleware with agent identity and enable flag.

        Args:
            assistant_id: Agent identifier used to find summaries directory
                at ``~/.deep_worker/{assistant_id}/summaries/``.
            enabled: Whether the user passed ``--history-memory`` CLI flag.
        """
        self.assistant_id = assistant_id
        self._enabled = enabled
        self._summaries_dir: Path | None = None
        self._has_index = False

        if enabled:
            try:
                self._summaries_dir = get_summaries_dir(assistant_id)
                self._has_index = _history_index_exists(self._summaries_dir)
            except Exception:  # noqa: BLE001
                self._has_index = False

        self._cognitive_hint = COGNITIVE_HINT if (enabled and self._has_index) else None

        # Create tools list (empty if not enabled or no index)
        if enabled and self._has_index:
            self.tools = [history_memory]
        else:
            self.tools = []

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
        if self._enabled and self._summaries_dir:
            return {"summaries_dir": str(self._summaries_dir)}
        return None

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
