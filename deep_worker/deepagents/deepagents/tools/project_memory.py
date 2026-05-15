"""Project memory tool for deep-worker agents.

Provides semantic search capabilities over the project's codebase
when a vector index is available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

# Ensure semantic_search package is importable
# Navigate from deepagents/tools/project_memory.py to project root
_pkg_path = Path(__file__).parent.parent.parent.parent.parent / "semantic_search"
if _pkg_path.exists() and str(_pkg_path) not in sys.path:
    sys.path.insert(0, str(_pkg_path))

# Lazy import — heavy dependencies (llama_index, torch, lancedb)
# are loaded only on the FIRST real call to project_memory.
_semantic_search = None  # (search_func, IndexNotFoundError)


def _ensure_semantic_search() -> tuple[Any, type[Exception]]:
    """One-time lazy import of semantic_search."""
    global _semantic_search
    if _semantic_search is None:
        # noqa — function-level import, ordering is safe
        from semantic_search.exceptions import IndexNotFoundError  # noqa: E402
        from semantic_search.retriever import search  # noqa: E402

        _semantic_search = (search, IndexNotFoundError)
    return _semantic_search


@tool
def project_memory(
    query: str,
    top_k: int = 5,
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Recall relevant knowledge from the project's codebase.

    Search the project's semantic index to find code, documentation,
    and implementation details related to your query.

    Use this tool to:
    - Understand how features are implemented
    - Find where specific functionality is located
    - Discover related code patterns across files
    - Learn about project architecture and conventions
    - Find usage examples or similar implementations

    Args:
        query: Natural language question about the project.
               Examples:
               - "How is user authentication implemented?"
               - "Where is the HTTP client configured?"
               - "Find async database query examples"
        top_k: Number of results to return (default: 5, max: 20).
        runtime: Tool runtime context (injected automatically by LangGraph).
        working_dir: Working directory override (used by MCP direct execution
                     when runtime is not available).

    Returns:
        Formatted search results with file paths and code excerpts,
        or an error message if search fails.

    Example:
        >>> project_memory("semantic search indexing", top_k=3)
        Project Memory: 'semantic search indexing'

        --- Result 1 (score=0.892) [python] ---
        File: semantic_search/semantic_search/indexer.py
        Type: parent

        def index_repository(
            repo_path: Path,
            config: IndexConfig | None = None,
            ...
    """
    # Determine working directory: runtime state > working_dir kwarg > cwd
    resolved_dir: Path
    if runtime is not None:
        working_dir_value = runtime.state.get("working_dir")
        resolved_dir = Path(working_dir_value) if working_dir_value else Path.cwd()
    elif working_dir is not None:
        resolved_dir = Path(working_dir)
    else:
        resolved_dir = Path.cwd()

    # Validate top_k
    if not 1 <= top_k <= 20:
        return "Error: top_k must be between 1 and 20"

    # Execute search (lazy import — dependencies loaded only on first call)
    try:
        _search, _IndexNotFoundError = _ensure_semantic_search()
    except Exception as exc:  # noqa: BLE001
        return f"Error: Failed to load semantic search: {exc}"
    try:
        results = _search(query, repo_path=resolved_dir, top_k=top_k)
    except _IndexNotFoundError:
        return (
            f"Error: Project memory not available.\n"
            f"The directory '{resolved_dir}' does not have a semantic search index.\n"
            f"This usually means the index was not created or is corrupted."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: Search failed with exception: {exc}"

    # Handle no results
    if not results:
        return (
            f"Project Memory: No results found for '{query}'\n\n"
            f"The semantic index may not contain information about this topic.\n"
            f"Try rephrasing your query or use other tools like grep or read_file."
        )

    # Format results for agent consumption
    lines: list[str] = [f"Project Memory: '{query}'\n"]
    for i, result in enumerate(results, 1):
        lang = f" [{result.language}]" if result.language else ""
        lines.append(
            f"--- Result {i} (score={result.score:.3f}){lang} ---\n"
            f"File: {result.file_path}\n"
            f"Type: {result.node_type}\n"
            f"\n{result.content}\n"
        )

    return "\n".join(lines)
