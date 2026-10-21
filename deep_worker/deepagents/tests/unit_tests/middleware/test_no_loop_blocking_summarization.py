"""Regression guard: the summarization ``on_summarized`` callback must NOT
block the agent's event loop.

Background — the live agent_c deadlock (2026-06-15): the production
``on_summarized`` callback is ``index_all_history_files``, a fully
synchronous LanceDB write. ``_aoffload_to_backend`` is an ``async`` hook
that runs on the agent's event-loop thread. Calling the callback INLINE
there blocks the loop thread while holding the GIL; LanceDB's own
background event loop then can never acquire the GIL → permanent GIL
deadlock, the agent's loop stops ticking forever.

The fix offloads the callback via ``asyncio.to_thread``. These tests fail
if anyone reverts to an inline call.

See deep_worker_cli/docs/DESIGN_NO_LOOP_BLOCKING_TOOLS.md.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable

import pytest

from deepagents.middleware.summarization import SummarizationMiddleware


def _make_middleware(callback: Callable[[str], None]) -> SummarizationMiddleware:
    """A SummarizationMiddleware with just the state ``_aoffload_to_backend``
    touches for the callback dispatch, bypassing heavy __init__."""
    mw = SummarizationMiddleware.__new__(SummarizationMiddleware)
    mw._on_summarized = callback  # type: ignore[attr-defined]
    return mw


def _gil_busy(seconds: float) -> None:
    """Busy-loop that HOLDS the GIL for ``seconds`` — models a native
    blocking call (LanceDB/torch) that does not release the GIL. Run inline
    on the loop, this freezes the loop; offloaded, the loop keeps ticking
    (the worker releases the GIL on bytecode boundaries / blocking I/O)."""
    end = time.monotonic() + seconds
    x = 0
    while time.monotonic() < end:
        x += 1  # pure-Python work: holds the GIL


async def _dispatch_like_aoffload(mw: SummarizationMiddleware, path: str) -> None:
    """Invoke ``_on_summarized`` the way ``_aoffload_to_backend`` MUST — i.e.
    via the exact statement present in its source. We extract and execute
    that statement so the runtime guard tracks the production dispatch: if
    the source is reverted to an inline call, this runs inline too and the
    loop stalls below."""
    src = inspect.getsource(SummarizationMiddleware._aoffload_to_backend)
    if "await asyncio.to_thread(self._on_summarized" in src:
        await asyncio.to_thread(mw._on_summarized, path)  # type: ignore[attr-defined]
    else:
        # Mirrors a reverted inline call — must make the guard FAIL.
        mw._on_summarized(path)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_summarized_dispatch_does_not_block_event_loop() -> None:
    """The production dispatch of ``_on_summarized`` must keep the event loop
    ticking while a GIL-holding callback runs (i.e. it is offloaded). If the
    source is reverted to an inline call, ``_dispatch_like_aoffload`` runs it
    inline and the loop stalls → this fails."""
    BLOCK_S = 1.0

    def blocking_callback(path: str) -> None:
        _gil_busy(BLOCK_S)

    mw = _make_middleware(blocking_callback)

    loop = asyncio.get_running_loop()
    ticks: list[float] = []
    stop = False

    async def probe() -> None:
        last = loop.time()
        while not stop:
            await asyncio.sleep(0.02)
            now = loop.time()
            ticks.append(now - last)
            last = now

    probe_task = asyncio.create_task(probe())

    t0 = loop.time()
    await _dispatch_like_aoffload(mw, "/tmp/x")
    elapsed = loop.time() - t0

    stop = True
    await probe_task

    assert elapsed >= BLOCK_S * 0.5, "the blocking callback should have run"
    max_gap = max(ticks) if ticks else 0.0
    assert max_gap < BLOCK_S * 0.5, (
        f"event loop stalled {max_gap:.2f}s while on_summarized ran — the "
        f"callback was NOT offloaded (regression). A GIL-holding inline "
        f"callback freezes the loop and deadlocks LanceDB (agent_c "
        f"2026-06-15). See DESIGN_NO_LOOP_BLOCKING_TOOLS.md"
    )


def test_aoffload_source_offloads_on_summarized() -> None:
    """Static guard (works without heavy deps): the async
    ``_aoffload_to_backend`` must dispatch ``_on_summarized`` via
    ``asyncio.to_thread`` — never inline."""
    src = inspect.getsource(SummarizationMiddleware._aoffload_to_backend)
    assert "self._on_summarized" in src, "callback wiring missing"
    assert "await asyncio.to_thread(self._on_summarized" in src, (
        "_aoffload_to_backend must offload _on_summarized via "
        "`await asyncio.to_thread(self._on_summarized, path)`. Running the "
        "LanceDB index write inline on the agent event loop deadlocks the "
        "GIL (agent_c 2026-06-15). See DESIGN_NO_LOOP_BLOCKING_TOOLS.md."
    )
