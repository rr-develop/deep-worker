"""Middleware for conditional project memory tool registration.

Registers the project_memory tool only when a semantic search index
exists in the working directory, and injects a cognitive hint to
guide the agent's usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from typing_extensions import TypedDict

from deepagents.middleware._utils import append_to_system_message
from deepagents.tools.project_memory import project_memory
from deepagents.utils.project_index import has_semantic_index


class ProjectMemoryState(AgentState):
    """State schema for ProjectMemoryMiddleware.

    Attributes:
        working_dir: Working directory for semantic search.
    """

    working_dir: NotRequired[str]


COGNITIVE_HINT = """### Project Knowledge Memory

**CRITICAL**: This project has a semantic search index. ALWAYS check `project_memory` FIRST before using other tools or delegating to subagents.

**PRIORITY**: `project_memory` has HIGHEST PRIORITY for:
- Technical terms, concepts, or names (e.g., "What is X?")
- How features/systems are implemented
- Where functionality is located
- Project architecture and patterns
- Code examples and usage

**When to use this tool:**
- ANY question about project-specific terminology, concepts, or systems
- Understanding implementation details
- Finding code locations or examples
- Before using `task`, `web_search`, or other external tools

**Example queries:**
- "What is SAVANT?" → Use `project_memory` FIRST (might be project-specific)
- "How is authentication implemented?" → Use `project_memory`
- "Where is HTTP client configured?" → Use `project_memory`

**Note**: Only use `grep`/`read_file` for unindexed files or very recent changes. For everything else, START with `project_memory`."""


class ProjectMemoryMiddleware(AgentMiddleware):
    """Conditionally enable project memory based on index availability.

    If a semantic search index exists in the working directory:
    - Registers the `project_memory` tool
    - Injects a cognitive hint into the system prompt

    Otherwise, does nothing (tool remains unavailable).

    Args:
        working_dir: Directory to check for semantic index.
                     Defaults to Path.cwd() if None.

    Example:
        >>> middleware = ProjectMemoryMiddleware(working_dir=Path("/path/to/your/project"))
        >>> # If index exists, project_memory tool will be registered
    """

    state_schema = ProjectMemoryState

    def __init__(self, working_dir: Path | None = None):
        """Initialize middleware with working directory.

        Args:
            working_dir: Directory to check for semantic index.
                        If None, defaults to Path.cwd().
        """
        self.working_dir = working_dir if working_dir is not None else Path.cwd()
        self._has_index = has_semantic_index(self.working_dir)
        self._cognitive_hint = COGNITIVE_HINT if self._has_index else None

        # Create tools list (empty if no index)
        if self._has_index:
            self.tools = [project_memory]
        else:
            self.tools = []

    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, str] | None:
        """Set working_dir in state before agent execution.

        This allows the project_memory tool to access the working directory
        via runtime.state.

        CRITICAL FIX: working_dir MUST be set in state regardless of whether
        semantic index exists. This ensures tools and agent can always determine
        the correct working directory, even when no index is present.

        Without this, agents fall back to Path.cwd() which returns the process's
        current directory (often "/" for daemon processes), not the configured
        working directory.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with working_dir.
        """
        # ALWAYS set working_dir in state, regardless of index existence
        # This is critical for tools that need to know the working directory
        return {"working_dir": str(self.working_dir)}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject cognitive hint into system prompt if index exists.

        Appends the cognitive hint to the system prompt to guide
        the agent's usage of the project_memory tool.
        """
        if not self._has_index or self._cognitive_hint is None:
            return handler(request)

        # Inject cognitive hint into system prompt
        new_system_message = append_to_system_message(
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
        the agent's usage of the project_memory tool.
        """
        if not self._has_index or self._cognitive_hint is None:
            return await handler(request)

        # Inject cognitive hint into system prompt
        new_system_message = append_to_system_message(
            request.system_message,
            self._cognitive_hint,
        )

        return await handler(request.override(system_message=new_system_message))
