"""A finished ASYNC parked job must stay reachable until its result is served.

Second half of the agent_a incident (2026-08-28). The sync path was fixed in
5110f23; the async parking site (``BoundedToolCallMiddleware._park_task``) still
called ``registry.drop`` from its done-callback, so completing the work ERASED
the entry. A caller that returns AFTER the work finished — the exact shape of
the incident — is told "no background job with id" and the produced result is
gone for good.

Note on scope, established empirically before writing these tests: the async
path has only ONE of the two sync defects. Its bounded wait is correct — it
awaits the ``asyncio.Task`` under the requested timeout and returns the tool's
own result (measured: 0.30s, exit_code 0). So there is nothing here mirroring
sync defect 2, and these tests deliberately do not pin one.

The three tests in ``test_sync_parked_job_terminal_state`` pin the SYNC
(thread-backed) defects; nothing here duplicates them — this file drives
``awrap_tool_call`` and the ``asyncio.Task`` branch.

Bounded well under the 60s budget, event-driven (no sleep-polling), registry
cleared in ``finally``.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from deep_worker_cli.backends import CLIShellBackend
from deepagents.middleware.bounded_tool_call import BoundedToolCallMiddleware
from deepagents.middleware.parked_job_registry import get_parked_job_registry

_BUDGET = 10.0


def _advertised_job_id(notice: ToolMessage) -> str:
    """The ``job_N`` the middleware handed the model in its in-progress notice."""
    match = re.search(r"job_\d+", str(notice.content))
    assert match is not None, f"no job id advertised in: {notice.content!r}"
    return match.group(0)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={"id": "toolu_async", "name": "slow_tool", "args": {}},
        runtime=None,
    )


async def _park_a_slow_tool(
    middleware: BoundedToolCallMiddleware,
    gate: "asyncio.Future[None]",
) -> str:
    """Drive the real async parking site and return the advertised job id."""

    async def _handler(_request: object) -> ToolMessage:
        await gate
        return ToolMessage(
            content="ASYNC-PAYLOAD", tool_call_id="toolu_async", status="success"
        )

    notice = await middleware.awrap_tool_call(_request(), _handler)
    return _advertised_job_id(notice)


@pytest.mark.asyncio
async def test_finished_async_job_is_still_reachable() -> None:
    """Completion must not erase the entry.

    This is the incident shape: the work finishes while nobody is waiting, and
    the agent comes back afterwards to collect what it produced.
    """
    registry = get_parked_job_registry()
    registry.clear()
    middleware = BoundedToolCallMiddleware(wait_seconds=1)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    try:
        job_id = await _park_a_slow_tool(middleware, gate)
        assert registry.resolve(job_id) is not None, "parked job must resolve"

        parked = registry.resolve(job_id)
        assert parked is not None
        settled = asyncio.Event()
        parked.handle.task.add_done_callback(lambda _t: settled.set())

        gate.set_result(None)
        await asyncio.wait_for(settled.wait(), timeout=_BUDGET)

        assert registry.resolve(job_id) is not None, (
            "finished async job was erased from the registry — its result is "
            "unreachable and the agent is told 'no background job with id'"
        )
    finally:
        if not gate.done():
            gate.set_result(None)
        registry.clear()


@pytest.mark.asyncio
async def test_finished_async_job_delivers_its_result_to_a_late_caller() -> None:
    """A caller arriving AFTER completion must still receive the result."""
    registry = get_parked_job_registry()
    registry.clear()
    middleware = BoundedToolCallMiddleware(wait_seconds=1)
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    try:
        job_id = await _park_a_slow_tool(middleware, gate)
        parked = registry.resolve(job_id)
        assert parked is not None
        settled = asyncio.Event()
        parked.handle.task.add_done_callback(lambda _t: settled.set())

        gate.set_result(None)
        await asyncio.wait_for(settled.wait(), timeout=_BUDGET)

        response = await asyncio.wait_for(
            backend.wait_for_job(job_id, timeout=5), timeout=_BUDGET
        )
        assert "ASYNC-PAYLOAD" in response.output, (
            f"late caller lost the finished result: {response.output[:200]}"
        )
        assert response.exit_code == 0
    finally:
        if not gate.done():
            gate.set_result(None)
        registry.clear()


@pytest.mark.asyncio
async def test_served_async_job_is_dropped_and_does_not_leak() -> None:
    """Once the result has been served, the entry is erased.

    Keeping finished jobs reachable forever would leak one record per parked
    call for the lifetime of the process, so ``drop`` must still happen — just
    at the point the caller has been served rather than at completion.
    """
    registry = get_parked_job_registry()
    registry.clear()
    middleware = BoundedToolCallMiddleware(wait_seconds=1)
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    try:
        job_id = await _park_a_slow_tool(middleware, gate)
        parked = registry.resolve(job_id)
        assert parked is not None
        settled = asyncio.Event()
        parked.handle.task.add_done_callback(lambda _t: settled.set())

        gate.set_result(None)
        await asyncio.wait_for(settled.wait(), timeout=_BUDGET)

        await asyncio.wait_for(
            backend.wait_for_job(job_id, timeout=5), timeout=_BUDGET
        )

        assert registry.resolve(job_id) is None, (
            "a served job must be dropped, otherwise the registry grows by one "
            "record per parked tool call and never shrinks"
        )
        assert job_id not in middleware.get_job_registry(), (
            "the middleware's own bookkeeping must not outlive the served job"
        )
    finally:
        if not gate.done():
            gate.set_result(None)
        registry.clear()
