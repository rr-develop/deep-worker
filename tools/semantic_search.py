"""LangChain tools for semantic code search.

Thin wrappers around the ``semantic_search`` package that expose
``index_repository`` and ``semantic_search`` as LangChain ``@tool`` functions
for use by agents.

Usage::

    from tools.semantic_search import index_repository_tool, semantic_search_tool

    agent = create_agent(tools=[index_repository_tool, semantic_search_tool, ...])
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool

# Ensure the semantic_search package is importable when running from the
# project root without a venv (consistent with the project's conventions).
_pkg_path = Path(__file__).parent.parent / "semantic_search"
if _pkg_path.exists() and str(_pkg_path) not in sys.path:
    sys.path.insert(0, str(_pkg_path))

from semantic_search import IndexConfig  # noqa: E402
from semantic_search.exceptions import IndexNotFoundError  # noqa: E402
from semantic_search.indexer import index_repository  # noqa: E402
from semantic_search.retriever import search  # noqa: E402


@tool
def index_repository_tool(repo_path: str) -> str:
    """Index a code repository for semantic search.

    Creates or incrementally updates a vector index at
    ``{repo_path}/.vector_index/``. Only changed files are reprocessed;
    unchanged files are skipped automatically.

    Call this tool before using ``semantic_search_tool`` on a repository.
    Safe to call repeatedly — subsequent calls are fast (incremental).

    Args:
        repo_path: Absolute path to the repository root directory.

    Returns:
        A summary string describing how many files were added, updated,
        deleted, and skipped, along with the total number of indexed nodes.
    """
    path = Path(repo_path)
    try:
        stats = index_repository(path)
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Indexing failed: {exc}"

    return (
        f"Indexed {repo_path}\n"
        f"  Added:   {stats.added} files\n"
        f"  Updated: {stats.updated} files\n"
        f"  Deleted: {stats.deleted} files\n"
        f"  Skipped: {stats.skipped} files (unchanged)\n"
        f"  Total nodes in index: {stats.total_nodes}"
    )


@tool
def semantic_search_tool(query: str, repo_path: str, top_k: int = 5) -> str:
    """Search a code repository using semantic (vector) similarity.

    Finds the most relevant source-code snippets and documentation chunks
    for the given natural-language query. The repository must have been
    indexed first with ``index_repository_tool``.

    Args:
        query:     Natural-language search query (e.g. "HTTP authentication
                   middleware", "database connection pooling").
        repo_path: Absolute path to the repository root directory.
        top_k:     Maximum number of results to return (default 5).

    Returns:
        Formatted search results with file paths, relevance scores, and
        content excerpts, or a message if no results were found.
    """
    path = Path(repo_path)
    try:
        results = search(query, repo_path=path, top_k=top_k)
    except IndexNotFoundError:
        return (
            f"Repository '{repo_path}' has not been indexed yet.\n"
            "Call index_repository_tool first."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Search failed: {exc}"

    if not results:
        return f"No results found for query: {query!r}"

    lines: list[str] = [f"Search results for: {query!r}\n"]
    for i, result in enumerate(results, 1):
        lang = f" [{result.language}]" if result.language else ""
        lines.append(
            f"--- Result {i} (score={result.score:.3f}){lang} ---\n"
            f"File: {result.file_path}\n"
            f"Type: {result.node_type}\n"
            f"{result.content}\n"
        )
    return "\n".join(lines)


# Convenience list for agent construction
SEMANTIC_SEARCH_TOOLS = [index_repository_tool, semantic_search_tool]
