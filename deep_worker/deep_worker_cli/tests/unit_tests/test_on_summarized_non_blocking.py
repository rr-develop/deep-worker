from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from deep_worker_cli.agent import OwnerLoop, _make_on_summarized


class Loop:
    def __init__(self) -> None:
        self.callbacks = []

    def call_soon_threadsafe(self, callback) -> None:
        self.callbacks.append(callback)


def _bound(loop):
    """An OwnerLoop already holding *loop*, as `abefore_agent` leaves it."""
    holder = OwnerLoop()
    holder.bind(loop)
    return holder


def test_callback_only_enqueues_work() -> None:
    class Client:
        async def index_history(self, path):
            raise AssertionError("must not run inline")

    loop = Loop()
    callback = _make_on_summarized(_bound(loop), Client(), Path("/summaries"))
    callback("ignored")
    assert len(loop.callbacks) == 1


def test_callback_contains_no_blocking_bridge() -> None:
    source = inspect.getsource(_make_on_summarized)
    assert "asyncio.run" not in source
    assert "ThreadPoolExecutor" not in source
    assert ".result()" not in source


@pytest.mark.asyncio
async def test_background_failure_never_escapes(caplog) -> None:
    class Client:
        async def index_history(self, path):
            raise RuntimeError("service down")

    loop = Loop()
    callback = _make_on_summarized(_bound(loop), Client(), Path("/summaries"))
    callback("ignored")
    loop.callbacks.pop()()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert "Failed to index conversation history" in caplog.text


def test_callback_builds_without_a_running_loop() -> None:
    """`create_cli_agent` is sync and is called with no loop running.

    Binding `asyncio.get_running_loop()` at build time made the CLI path
    die with "no running event loop" before the agent existed.
    """

    class Client:
        async def index_history(self, path):
            raise AssertionError("must not run inline")

    callback = _make_on_summarized(OwnerLoop(), Client(), Path("/summaries"))
    assert callable(callback)


def test_unbound_owner_loop_never_raises_into_summarization() -> None:
    """No loop was ever bound: log and move on, never raise."""

    class Client:
        async def index_history(self, path):
            raise AssertionError("must not run inline")

    callback = _make_on_summarized(OwnerLoop(), Client(), Path("/summaries"))
    callback("ignored")  # must not raise


@pytest.mark.asyncio
async def test_indexing_is_scheduled_when_callback_runs_on_worker_thread() -> None:
    """The production shape: summarization calls us via `to_thread`.

    `_aoffload_to_backend` deliberately offloads this callback to a worker
    thread to avoid a GIL deadlock with LanceDB, so the callback executes
    where `asyncio.get_running_loop()` RAISES. A callback that looked the
    loop up itself would silently drop every indexing request — the agent
    would summarize forever and never update its index.

    Here the owner loop is bound the way `abefore_agent` binds it, then
    the callback is invoked from a real worker thread, and the indexing
    coroutine must still be scheduled on the owner loop.
    """
    indexed = asyncio.Event()

    class Client:
        async def index_history(self, path):
            indexed.set()

    holder = OwnerLoop()
    holder.bind()  # we are on the agent loop here, exactly like abefore_agent
    callback = _make_on_summarized(holder, Client(), Path("/summaries"))

    def worker() -> None:
        # Proof the production context really has no loop of its own.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:  # pragma: no cover - would mean the test lost its premise
            raise AssertionError("worker thread unexpectedly has a running loop")
        callback("/summaries/thread.jsonl")

    await asyncio.to_thread(worker)
    await asyncio.wait_for(indexed.wait(), timeout=5)


@pytest.mark.asyncio
async def test_middleware_binds_the_owner_loop() -> None:
    """`abefore_agent` is what supplies the loop the callback needs."""
    from deep_worker_cli.middleware.history_memory import HistoryMemoryMiddleware

    holder = OwnerLoop()
    middleware = HistoryMemoryMiddleware(
        "alice", enabled=False, owner_loop=holder
    )
    assert holder.get() is None

    await middleware.abefore_agent({}, None)
    assert holder.get() is asyncio.get_running_loop()
