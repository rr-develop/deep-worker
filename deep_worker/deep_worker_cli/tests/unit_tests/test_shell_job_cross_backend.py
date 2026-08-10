"""Fifth focus: a shell job must resolve from ANY CLIShellBackend.

The ParkedJobRegistry already knows job_N (kind=shell), but wait_for_job/kill_job
only looked into the self._jobs of their own instance. Backend B, different from A
(where execute parked the job), answered "no background job".

See docs/DESIGN_FIX_SHELL_JOB_CROSS_BACKEND.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from deep_worker_cli.backends import CLIShellBackend
from deepagents.middleware.parked_job_registry import get_parked_job_registry


def _job_id(text: str) -> str:
    m = re.search(r"\bjob_\d+\b", text)
    assert m is not None, f"no job_N in: {text!r}"
    return m.group(0)


@pytest.fixture(autouse=True)
def _clean_registry():
    get_parked_job_registry().clear()
    yield
    get_parked_job_registry().clear()


@pytest.mark.asyncio
async def test_wait_for_job_on_other_backend_instance_finds_shell_job() -> None:
    """T1: A parks; B.wait_for_job(job_N) must NOT say 'no background job'."""
    a = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    b = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

    started = await a.execute("sleep 5 && echo cross", timeout=1)
    job_id = _job_id(started.output)
    assert job_id in a._jobs
    assert job_id not in b._jobs

    result = await b.wait_for_job(job_id, timeout=1)
    assert "no background job" not in result.output.lower()
    assert job_id in result.output or "STILL running" in result.output

    # Cleanup via the OTHER backend too (T2 partial).
    killed = await b.kill_job(job_id)
    assert "no background job" not in killed.output.lower()
    assert "killed" in killed.output.lower()
    assert job_id not in a._jobs
    assert get_parked_job_registry().resolve(job_id) is None


@pytest.mark.asyncio
async def test_kill_job_on_other_backend_stops_process() -> None:
    """T2: B.kill_job must terminate the process A parked."""
    a = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    b = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)

    started = await a.execute("sleep 60", timeout=1)
    job_id = _job_id(started.output)
    proc = a._jobs[job_id].proc

    killed = await b.kill_job(job_id)
    assert "killed" in killed.output.lower()
    assert proc.returncode is not None
    assert job_id not in a._jobs


@pytest.mark.asyncio
async def test_same_backend_wait_still_works() -> None:
    """T3: no regression on the owning instance."""
    a = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    started = await a.execute("sleep 2 && echo same", timeout=1)
    job_id = _job_id(started.output)

    result = await a.wait_for_job(job_id, timeout=10)
    assert "same" in result.output
    assert result.exit_code == 0
    assert a._jobs == {}


@pytest.mark.asyncio
async def test_tool_kind_still_resolves_via_registry() -> None:
    """T4: kind=tool via registry must keep working (no regression of loci 1–4).

    Shell cross-backend resolve must not shadow or break middleware-parked
    tool jobs. A fresh CLIShellBackend with empty _jobs must still wait/kill
    a kind=tool entry in ParkedJobRegistry.
    """
    import asyncio

    b = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    registry = get_parked_job_registry()

    async def slow_tool() -> str:
        await asyncio.sleep(0.15)
        return "tool-still-works"

    task = asyncio.ensure_future(slow_tool())
    job_id = registry.allocate()
    registry.register(
        job_id, kind="tool", handle=task, alias="toolu_T4", tool_name="execute"
    )

    assert b._jobs == {}
    result = await b.wait_for_job(job_id, timeout=10)
    assert "no background job" not in result.output.lower()
    assert "tool-still-works" in result.output
    # wait_parked_tool returns the result without drop (pre-existing);
    # kill_job is the path that drops kind=tool entries.


@pytest.mark.asyncio
async def test_tool_kind_kill_via_registry() -> None:
    """T4b: kill_job on kind=tool must cancel the task from any backend."""
    import asyncio

    b = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    registry = get_parked_job_registry()

    async def slow_tool() -> str:
        await asyncio.sleep(30)
        return "should-not-finish"

    task = asyncio.ensure_future(slow_tool())
    job_id = registry.allocate()
    registry.register(job_id, kind="tool", handle=task, tool_name="execute")

    killed = await b.kill_job(job_id)
    assert "no background job" not in killed.output.lower()
    assert "killed" in killed.output.lower() or "cancellation" in killed.output.lower()
    assert registry.resolve(job_id) is None
