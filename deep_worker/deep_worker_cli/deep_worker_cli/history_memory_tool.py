"""History memory tool for deep-worker agents.

Provides semantic search over summarised conversation history stored in JSONL
files. Uses the same LanceDB + LlamaIndex infrastructure as ``project_memory``
but indexes message-level chunks instead of code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

# Lazy import — heavy dependencies loaded only on first real call
_history_search = None  # (search_func, FileNotFoundError)


def _ensure_history_search() -> tuple[Any, type[Exception]]:
    """One-time lazy import of history_retriever.

    Resolves the semantic_search provider through provider_paths (source
    root on sys.path in source mode; no-op in installed mode) — no sys.path
    hacks. Raises ImportError when the semantic-search extra is missing; the
    caller converts that into a user-facing error string.
    """
    global _history_search
    if _history_search is None:
        from deep_worker_cli.provider_paths import ensure_provider_importable

        resolution = ensure_provider_importable("deep_worker.providers.semantic_search")
        if resolution.unavailable:
            raise ImportError(resolution.remedy or "deep-worker[semantic-search] not installed")
        from deep_worker.providers.semantic_search.history_retriever import search_history  # noqa: E402

        _history_search = (search_history, FileNotFoundError)
    return _history_search


def _run_history_search(query: str, summaries_dir: str, top_k: int) -> str:
    """Heavy, BLOCKING body of ``history_memory`` — runs semantic search
    over LanceDB + a (cold) torch embedding model.

    ⚠️ MUST be executed OFF the agent's event-loop thread (via
    ``asyncio.to_thread`` — see ``history_memory`` below). LanceDB bridges
    to its own background event loop with ``run_coroutine_threadsafe(...)
    .result()``; running this inline on the agent loop blocks that thread
    while holding the GIL and deadlocks LanceDB's loop (an observed live
    deadlock). See docs/DESIGN_NO_LOOP_BLOCKING_TOOLS.md.
    """
    # Validate top_k
    if not 1 <= top_k <= 20:
        return "Error: top_k must be between 1 and 20"

    try:
        search_func, _not_found = _ensure_history_search()
    except ImportError as exc:
        return (
            f"Error: semantic search dependencies not available: {exc}\n"
            "Install with: pip install deep-worker[semantic-search]"
        )

    try:
        results = search_func(
            query=query,
            summaries_dir=summaries_dir,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        return (
            f"History not indexed: {exc}\n"
            "Run the agent with --history-memory flag and allow a "
            "summarization cycle to complete before searching history."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Search error: {type(exc).__name__}: {exc}"

    if not results:
        return (
            f"History Memory: '{query}'\n\n"
            "No matching results found in conversation history."
        )

    parts: list[str] = [f"History Memory: '{query}'\n"]
    for idx, r in enumerate(results, 1):
        lines_info = f"Lines: {r.line_range}" if r.line_range else "Position: unknown"
        time_info = f" | When: {r.timestamp}" if r.timestamp else ""
        type_info = f" | Types: {r.message_types}" if r.message_types else ""
        parts.append(
            f"--- Result {idx} (score={r.score:.3f}) ---\n"
            f"Thread: {r.thread_id}\n"
            f"{lines_info}{time_info}{type_info}\n\n"
            f"{r.content}\n"
        )

    return "\n".join(parts)


@tool
async def history_memory(
    query: str,
    top_k: int = 5,
    runtime: Any = None,
    working_dir: str | None = None,
) -> str:
    """Recall relevant knowledge from past conversation history.

    Search the summarised conversation history (JSONL files) to find messages,
    decisions, and context from earlier turns that have been removed from
    the current context window via summarization.

    Use this tool to:
    - Recall what was discussed or decided before summarization
    - Find specific technical details mentioned in past turns
    - Recover context about project decisions, requirements, or user preferences
    - Look up implementation details that were discussed earlier

    Args:
        query: Natural language question about past conversation history.
               Examples:
               - "What did we decide about the database schema?"
               - "What error was the user reporting earlier?"
               - "Find the discussion about authentication tokens"
        top_k: Number of results to return (default: 5, max: 20).
        runtime: Tool runtime context (injected automatically by LangGraph).
        working_dir: Working directory override (used by MCP direct execution
                     when runtime is not available).

    Returns:
        Formatted search results with thread context and message excerpts,
        or an error message if search fails.

    Example:
        >>> history_memory("authentication implementation", top_k=3)
        History Memory: 'authentication implementation'

        --- Result 1 (score=0.892) ---
        Thread: thread_abc123
        Lines: 142-145 | When: 2026-07-17T10:30:00
        Types: human,ai

        User: How is authentication implemented?
        Assistant: We use JWT tokens with...
    """
    # Determine summaries directory
    summaries_dir: Path
    if runtime is not None:
        summaries_dir_value = runtime.state.get("summaries_dir")
        summaries_dir = (
            Path(summaries_dir_value) if summaries_dir_value else Path.home() / Path("~/.deep_worker").name
        )
    elif working_dir is not None:
        # With explicit working_dir, find summaries relative to agent dir
        summaries_dir = Path(working_dir)
    else:
        summaries_dir = Path.home() / Path("~/.deep_worker").name

    # ⚠️ MANDATORY OFFLOAD — DO NOT run the search inline here.
    # ``_run_history_search`` loads a torch embedding model and queries
    # LanceDB, which bridges to its own background event loop via a
    # blocking ``run_coroutine_threadsafe(...).result()``. Executed on the
    # agent's event-loop thread it blocks that thread while holding the
    # GIL → LanceDB's loop starves on the GIL → permanent deadlock (an
    # observed live hang). ``asyncio.to_thread`` runs it on a
    # worker thread whose blocking wait RELEASES the GIL, so both loops
    # progress. This tool is ``async`` for exactly this reason — keep it
    # async and keep the ``to_thread``. See
    # docs/DESIGN_NO_LOOP_BLOCKING_TOOLS.md +
    # tests/test_no_loop_blocking_summarization.py.
    return await asyncio.to_thread(
        _run_history_search, query, str(summaries_dir), top_k
    )
