"""Regression guard: heavy semantic-search tools must NOT block the agent
event loop — they must execute via an async/offloaded path.

``history_memory`` and ``project_memory`` load a torch embedding model and
query LanceDB (which bridges to its own background event loop with a
blocking ``run_coroutine_threadsafe(...).result()``). Run inline on the
agent's event-loop thread, that blocks the loop while holding the GIL and
deadlocks LanceDB's loop (the live mj1_init hang, 2026-06-15).

These tools are therefore defined as ``async`` ``@tool``s that offload the
heavy body via ``asyncio.to_thread``. If someone reverts them to a plain
sync ``@tool`` (relying on LangChain's implicit, fragile ``run_in_executor``
that any ``.invoke()`` bypasses), these guards fail.

See deep_worker_cli/docs/DESIGN_NO_LOOP_BLOCKING_TOOLS.md.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Ensure semantic_search is importable for the tool modules' lazy import
# paths (they only import it on first call, so import here is not required,
# but keep paths consistent with production).
_SS = Path(__file__).resolve().parents[4] / "semantic_search"
if _SS.exists() and str(_SS) not in sys.path:
    sys.path.insert(0, str(_SS))

from deep_worker_cli.history_memory_tool import history_memory  # noqa: E402


def test_history_memory_is_async_offloaded() -> None:
    """history_memory must run via an async coroutine (offloaded), with NO
    inline sync execution path."""
    # As an async @tool, LangChain stores the impl in ``.coroutine`` and
    # leaves ``.func`` None — so EVERY invocation (even sync .invoke()) goes
    # through the async/offloaded path, never inline on the loop.
    assert history_memory.coroutine is not None, (
        "history_memory must be an async tool (offloaded). A sync @tool "
        "relies on LangChain's implicit run_in_executor, which any "
        ".invoke() call bypasses → inline LanceDB call → GIL deadlock."
    )
    assert inspect.iscoroutinefunction(history_memory.coroutine)
    assert history_memory.func is None, (
        "history_memory must NOT expose a sync func path"
    )
    # The heavy body must be offloaded via asyncio.to_thread.
    src = inspect.getsource(history_memory.coroutine)
    assert "to_thread" in src, (
        "history_memory must offload its search via asyncio.to_thread"
    )


def test_project_memory_is_async_offloaded() -> None:
    # Imported lazily so a missing deepagents path doesn't break collection
    # of the history_memory guard above.
    _da = Path(__file__).resolve().parents[3] / "deepagents"
    if _da.exists() and str(_da) not in sys.path:
        sys.path.insert(0, str(_da))
    from deepagents.tools.project_memory import project_memory

    assert project_memory.coroutine is not None, (
        "project_memory must be an async tool (offloaded) — see "
        "history_memory rationale."
    )
    assert inspect.iscoroutinefunction(project_memory.coroutine)
    assert project_memory.func is None
    src = inspect.getsource(project_memory.coroutine)
    assert "to_thread" in src, (
        "project_memory must offload its search via asyncio.to_thread"
    )
