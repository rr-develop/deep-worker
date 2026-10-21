"""Regression guard: heavy semantic-search tools must NOT block the agent
event loop — they must execute via an async/offloaded path.

``history_memory`` and ``project_memory`` used to load a torch embedding
model and query LanceDB in-process (LanceDB bridges to its own background
event loop with a blocking ``run_coroutine_threadsafe(...).result()``). Run
inline on the agent's event-loop thread, that blocks the loop while holding
the GIL and deadlocks LanceDB's loop (the live agent_c hang, 2026-06-15).

Since ``2567038`` the heavy work lives in a separate semantic-memory service
and the tools reach it over async gRPC (``grpc.aio``), so what they ``await``
no longer blocks the loop. The invariant is unchanged — nothing blocking may
run on the loop thread — but the shape that satisfies it is: wrapping an
``await`` of a ``grpc.aio`` call in ``asyncio.to_thread`` would be wrong (it
would hand a coroutine to a worker thread with no running loop).

What must still hold: these are ``async`` ``@tool``s with no sync ``func``
path, so every invocation — including a sync ``.invoke()``, which bypasses
LangChain's implicit ``run_in_executor`` — goes through the async path; and
every service call is awaited rather than invoked inline.

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


def _assert_every_service_call_is_awaited(tool: object, name: str) -> None:
    """Every semantic-memory RPC in ``tool`` must be awaited.

    Walks the AST rather than grepping: a substring check cannot tell
    ``await client.search_project(...)`` from a bare
    ``client.search_project(...)``, and it is exactly the bare form — a sync
    client, or a dropped ``await`` — that puts a blocking call back on the
    loop thread.
    """
    import ast

    # Methods that cross the process boundary. Each is a coroutine, so a
    # bare call is either a forgotten await or a sync client sneaking back.
    service_calls = {
        "index_status",
        "list_tenants",
        "search_history",
        "search_project",
        "index_history",
        "index_project",
    }

    fn = tool.coroutine
    source_file = inspect.getsourcefile(fn)
    assert source_file is not None, f"cannot locate the source of {name}"
    with open(source_file, encoding="utf-8") as fh:
        module = ast.parse(fh.read())

    body = None
    for node in ast.walk(module):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == fn.__name__:
            body = node
            break
    assert body is not None, (
        f"{name} is not an async def in {source_file}; the checks below "
        f"would pass vacuously"
    )

    awaited = {
        id(node.value)
        for node in ast.walk(body)
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
    }
    bare = [
        node.func.attr
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in service_calls
        and id(node) not in awaited
    ]
    assert not bare, (
        f"{name} calls {', '.join(bare)} without awaiting — a blocking call "
        f"on the agent's event-loop thread (see the module docstring)"
    )


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
    # The heavy work happens in the semantic-memory service; every call
    # into it must be awaited, never run inline on the loop thread.
    _assert_every_service_call_is_awaited(history_memory, "history_memory")


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
    _assert_every_service_call_is_awaited(project_memory, "project_memory")
