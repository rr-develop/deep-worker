"""LIVE proof that an advertised job_id is always honourable. No mocks.

Incident: the tool-call middleware handed the model a ``toolu_…`` as
``job_id`` while ``wait_for_job``/``kill_job`` only knew ``job_N``, so the
model permanently lost control of live background work.

This test drives the REAL objects the agent uses — a real
``BoundedToolCallMiddleware`` bound, a real ``CLIShellBackend`` running a real
subprocess, and the real ``wait_for_job``/``kill_job`` tool bodies — with the
production bound lowered via ``DW_LONG_RUNNING_WAIT_SECONDS``. Nothing is
patched or faked: the only thing configured is the timeout, through its
documented environment variable.

Budget discipline (integration tests must never hang): every wait is bounded,
the whole module is capped, and every spawned process is killed in ``finally``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path

import pytest
from langchain.tools.tool_node import ToolCallRequest, ToolRuntime
from langchain_core.messages import ToolMessage

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_JOB_RE = re.compile(r"\bjob_\d+\b")

# Hard ceiling for the whole scenario. A parked-job test that hangs is worse
# than one that fails.
_BUDGET_SECONDS = 60.0
_BOUND_SECONDS = 3.0


class _Runtime:
    """Minimal stand-in for ToolRuntime on the plain backend path."""


def _tool(mw, name: str):
    for t in mw.tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found")


@pytest.fixture
def _short_bound(monkeypatch: pytest.MonkeyPatch) -> float:
    """Lower the production bound through its documented env var only."""
    monkeypatch.setenv("DW_LONG_RUNNING_WAIT_SECONDS", str(_BOUND_SECONDS))
    return _BOUND_SECONDS


async def test_advertised_job_id_is_honoured_end_to_end(_short_bound: float) -> None:
    """The id the model is shown must resolve through the real tools.

    Runs a real ``sleep`` subprocess past a real bound and then recovers it by
    the advertised id — the exact sequence that failed in production.
    """
    from deep_worker_cli.backends import CLIFilesystemMiddleware, CLIShellBackend
    from deepagents.middleware.parked_job_registry import get_parked_job_registry

    get_parked_job_registry().clear()
    started = time.monotonic()

    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    mw = CLIFilesystemMiddleware(backend=backend)
    job_id: str | None = None
    try:
        # Real subprocess that outlives the bound, then prints a marker.
        parked = await backend.execute(
            "sleep 4 && echo LIVE_MARKER", timeout=int(_BOUND_SECONDS)
        )
        match = _JOB_RE.search(parked.output)
        assert match is not None, (
            f"execute must advertise a canonical job_N, got: {parked.output!r}"
        )
        job_id = match.group(0)

        # The advertised id must resolve in the shared registry...
        assert get_parked_job_registry().resolve(job_id) is not None

        # ...and through the REAL tool the agent calls.
        wait_tool = _tool(mw, "wait_for_job")
        out = await asyncio.wait_for(
            wait_tool.coroutine(job_id=job_id, timeout=20, runtime=_Runtime()),
            timeout=_BUDGET_SECONDS - (time.monotonic() - started),
        )
        assert "no background job" not in out, (
            f"the advertised id {job_id} was not honoured: {out!r}"
        )
        assert "LIVE_MARKER" in out, f"lost the command's real output: {out!r}"
        job_id = None  # finished and drained; nothing to clean up
    finally:
        if job_id is not None:
            with_suppress = getattr(backend, "kill_job", None)
            if with_suppress is not None:
                try:
                    await asyncio.wait_for(backend.kill_job(job_id), timeout=10)
                except Exception:  # noqa: BLE001 — cleanup must never mask
                    pass
        for job in list(getattr(backend, "_jobs", {}).values()):
            try:
                job.proc.kill()
            except Exception:  # noqa: BLE001
                pass

    assert time.monotonic() - started < _BUDGET_SECONDS


async def test_middleware_parked_tool_is_recoverable_live(
    _short_bound: float,
) -> None:
    """A tool parked by the REAL middleware bound is reachable and killable.

    This is the branch that was completely unreachable before the fix: the
    middleware fires first (it also covers CLI stage-A), so the model only ever
    saw the middleware's id.
    """
    from deep_worker_cli.backends import CLIFilesystemMiddleware, CLIShellBackend
    from deepagents.middleware.bounded_tool_call import BoundedToolCallMiddleware
    from deepagents.middleware.parked_job_registry import get_parked_job_registry

    get_parked_job_registry().clear()
    started = time.monotonic()

    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    fs_mw = CLIFilesystemMiddleware(backend=backend)
    bound_mw = BoundedToolCallMiddleware()
    assert bound_mw.wait_seconds == pytest.approx(_BOUND_SECONDS), (
        "the bound must come from DW_LONG_RUNNING_WAIT_SECONDS — no patching"
    )

    state = {"messages": []}
    runtime = ToolRuntime(
        state=state,
        context=None,
        tool_call_id="toolu_LIVE",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "toolu_LIVE", "name": "execute", "args": {}},
        state=state,
        tool=None,
    )

    slow_started = asyncio.Event()

    async def slow_handler(_req: ToolCallRequest) -> ToolMessage:
        slow_started.set()
        await asyncio.sleep(300)  # far beyond the bound; killed below
        return ToolMessage(content="unreachable", tool_call_id="toolu_LIVE")

    notice = await asyncio.wait_for(
        bound_mw.awrap_tool_call(request, slow_handler),
        timeout=_BUDGET_SECONDS - (time.monotonic() - started),
    )
    assert slow_started.is_set()
    assert notice.status != "error", "a timeout is NEVER an error"

    text = str(notice.content)
    match = _JOB_RE.search(text)
    assert match is not None, f"middleware must advertise job_N, got {text!r}"
    assert "toolu_LIVE" not in text, (
        "the middleware must not advertise the tool_use_id as a job id"
    )
    job_id = match.group(0)

    # The REAL kill_job tool must reach and stop it.
    kill_tool = _tool(fs_mw, "kill_job")
    out = await asyncio.wait_for(
        kill_tool.coroutine(job_id=job_id, runtime=_Runtime()),
        timeout=_BUDGET_SECONDS - (time.monotonic() - started),
    )
    assert "no background job" not in out, (
        f"kill_job could not honour the advertised id {job_id}: {out!r}"
    )
    assert get_parked_job_registry().resolve(job_id) is None
    assert time.monotonic() - started < _BUDGET_SECONDS


async def test_env_bound_is_the_single_source(_short_bound: float) -> None:
    """Guard: the bound really is configurable from one place, so this whole
    suite stays fast without touching production defaults."""
    from deepagents.middleware.bounded_tool_call import resolve_tool_wait_seconds

    assert resolve_tool_wait_seconds() == pytest.approx(_BOUND_SECONDS)
    assert os.environ["DW_LONG_RUNNING_WAIT_SECONDS"] == str(_BOUND_SECONDS)
