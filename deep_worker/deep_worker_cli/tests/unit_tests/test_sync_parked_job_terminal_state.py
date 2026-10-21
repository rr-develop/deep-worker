"""A thread-backed (sync) parked job must reach a TERMINAL, REPORTABLE state.

Incident 2026-08-28 (agent_a): ``report_infrastructure_feedback`` is a sync
tool, so it parks through the thread path. Its report WAS written to disk, but
the agent could never confirm it: ``wait_for_job`` answered "STILL running on a
worker thread" three times (120s, 300s, 600s) while ``py-spy`` showed every
worker thread idle. The work had finished; only the ANSWER was lost.

Two independent defects on the sync path produced that:

  1. ``bounded_tool_call`` parks the future and attaches
     ``future.add_done_callback(lambda _f: registry.drop(job_id))``. Finishing
     therefore ERASES the registry entry instead of moving it to a terminal
     state, so the completed result becomes unreachable — the caller is told
     "no background job with id", never the result.

  2. ``_wait_parked_tool`` ignores the caller's ``timeout`` on the sync branch:
     it reads ``future.done()`` once and returns immediately. A caller asking
     for 30s got an answer in ~0.000s that claimed the job was still running.

Both are the same class of defect the delivery work closed elsewhere: the work
completes but the confirmation never reaches the model.

Every test here is bounded (well under the 60s integration budget), fully
event-driven (no sleep-polling), and shuts its executor down in ``finally``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from deep_worker_cli.backends import CLIShellBackend
from deepagents.middleware.bounded_tool_call import BoundedToolCallMiddleware
from deepagents.middleware.parked_job_registry import (
    ParkedJob,
    get_parked_job_registry,
)

_BUDGET = 10.0


def _backend() -> CLIShellBackend:
    return CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)


def _advertised_job_id(notice: ToolMessage) -> str:
    """The ``job_N`` the middleware handed the model in its in-progress notice."""
    match = re.search(r"job_\d+", str(notice.content))
    assert match is not None, f"no job id advertised in: {notice.content!r}"
    return match.group(0)


@pytest.mark.asyncio
async def test_finished_sync_job_still_reports_its_result() -> None:
    """Defect 1: the completed result must remain reachable.

    The registry entry must not be erased on completion — a finished job has
    to be able to hand the caller what it produced.
    """
    registry = get_parked_job_registry()
    registry.clear()
    middleware = BoundedToolCallMiddleware(wait_seconds=1)
    release = threading.Event()

    def _handler(_request: object) -> ToolMessage:
        release.wait(_BUDGET)
        return ToolMessage(
            content="REPORT-SAVED", tool_call_id="toolu_probe", status="success"
        )

    request = SimpleNamespace(
        tool_call={
            "id": "toolu_probe",
            "name": "report_infrastructure_feedback",
            "args": {},
        },
        runtime=None,
    )
    try:
        notice = middleware.wrap_tool_call(request, _handler)
        job_id = _advertised_job_id(notice)
        assert registry.resolve(job_id) is not None, "parked job must resolve"

        # Event-driven: the production done-callback is already attached, so
        # add our own AFTER it and block on that — callbacks fire in order, so
        # when ours runs the registry has settled. No polling.
        settled = threading.Event()
        parked_job = registry.resolve(job_id)
        assert parked_job is not None
        handle = parked_job.handle
        handle.add_done_callback(lambda _f: settled.set())

        release.set()
        assert settled.wait(_BUDGET), "sync job did not settle within budget"

        parked = registry.resolve(job_id)
        assert parked is not None, (
            "finished sync job was erased from the registry — its result is "
            "unreachable and the agent is told 'no background job with id'"
        )
    finally:
        release.set()
        registry.clear()


@pytest.mark.asyncio
async def test_wait_for_sync_job_honours_the_requested_timeout() -> None:
    """Defect 2: the sync branch must actually WAIT for the requested time.

    A caller asking for a bounded wait must not get an instant "STILL running".
    """
    backend = _backend()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    try:
        future = executor.submit(lambda: release.wait(_BUDGET) and "DONE")
        parked = ParkedJob(
            job_id="job_1",
            kind="tool",
            handle=future,
            tool_name="report_infrastructure_feedback",
        )

        started = time.monotonic()
        response = await asyncio.wait_for(
            backend._wait_parked_tool(parked, 2), timeout=_BUDGET
        )
        waited = time.monotonic() - started

        assert waited >= 1.5, (
            f"wait_for_job(timeout=2) returned after {waited:.3f}s without "
            f"waiting; output was: {response.output[:120]}"
        )
    finally:
        release.set()
        executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_wait_for_sync_job_returns_the_result_when_it_finishes() -> None:
    """The bounded wait must return the tool's own result, not a status line."""
    backend = _backend()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    try:
        future = executor.submit(lambda: release.wait(_BUDGET) and "REPORT-SAVED")
        parked = ParkedJob(
            job_id="job_1",
            kind="tool",
            handle=future,
            tool_name="report_infrastructure_feedback",
        )

        loop = asyncio.get_running_loop()
        loop.call_later(0.2, release.set)
        response = await asyncio.wait_for(
            backend._wait_parked_tool(parked, 5), timeout=_BUDGET
        )

        assert "REPORT-SAVED" in response.output, (
            f"finished sync job did not deliver its result: {response.output[:200]}"
        )
        assert response.exit_code == 0
    finally:
        release.set()
        executor.shutdown(wait=False)
