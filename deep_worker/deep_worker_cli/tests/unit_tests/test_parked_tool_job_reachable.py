"""``wait_for_job`` / ``kill_job`` must reach jobs parked by EITHER layer.

A past incident (2026-07-31). The tool-call middleware parked long-running
tool work in its own registry under the ``tool_use_id``, while these two tools
only ever resolved the shell backend's ``job_N`` registry. The model was handed
an id nothing could honour:

    Tool 'execute' is still running after 120s … job_id: toolu_01ABC…
    wait_for_job(job_id="toolu_01ABC…") → Error: no background job …

Both layers now mint ids from ONE shared ``ParkedJobRegistry``, and these tools
resolve both kinds. This suite pins that end-to-end, through the real tool
objects the agent calls — no mocks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_worker_cli.backends import CLIFilesystemMiddleware, CLIShellBackend
from deepagents.middleware.parked_job_registry import get_parked_job_registry


class _Runtime:
    """Minimal stand-in for ToolRuntime (ignored by a plain backend)."""


def _middleware_and_backend() -> tuple[CLIFilesystemMiddleware, CLIShellBackend]:
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    mw = CLIFilesystemMiddleware(backend=backend)
    return mw, backend


def _tool(mw: CLIFilesystemMiddleware, name: str):
    for t in mw.tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found")


@pytest.fixture(autouse=True)
def _clean_registry():
    get_parked_job_registry().clear()
    yield
    get_parked_job_registry().clear()


# ── shell jobs keep working exactly as before ────────────────────────────


@pytest.mark.asyncio
async def test_shell_job_registers_in_shared_registry() -> None:
    _, backend = _middleware_and_backend()
    started = await backend.execute("sleep 5 && echo shell", timeout=1)
    assert "Background job id: job_" in started.output

    job_id = next(iter(backend._jobs))
    parked = get_parked_job_registry().resolve(job_id)
    assert parked is not None, "a shell job must be visible in the shared registry"
    assert parked.kind == "shell"

    await backend.kill_job(job_id)


# ── a job parked by the middleware is reachable by the agent's tools ─────


@pytest.mark.asyncio
async def test_wait_for_job_resolves_middleware_parked_tool_job() -> None:
    mw, _ = _middleware_and_backend()
    registry = get_parked_job_registry()

    async def slow_tool() -> str:
        await asyncio.sleep(0.2)
        return "tool-output"

    task = asyncio.ensure_future(slow_tool())
    job_id = registry.allocate()
    registry.register(
        job_id, kind="tool", handle=task, alias="toolu_MW", tool_name="execute"
    )

    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(job_id=job_id, timeout=10, runtime=_Runtime())

    assert "no background job" not in out, (
        "an id advertised by the middleware must resolve — this is the defect"
    )
    assert "tool-output" in out, "a finished tool job must hand back its result"


@pytest.mark.asyncio
async def test_middleware_parked_job_resolves_by_tool_use_alias() -> None:
    """The alias-fix contract: echoing the tool_use_id must still reach the job."""
    mw, _ = _middleware_and_backend()
    registry = get_parked_job_registry()

    async def slow_tool() -> str:
        await asyncio.sleep(0.2)
        return "via-alias"

    task = asyncio.ensure_future(slow_tool())
    job_id = registry.allocate()
    registry.register(
        job_id, kind="tool", handle=task, alias="toolu_ALIAS", tool_name="execute"
    )

    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(
        job_id="toolu_ALIAS", timeout=10, runtime=_Runtime()
    )
    assert "no background job" not in out
    assert "via-alias" in out


@pytest.mark.asyncio
async def test_kill_job_cancels_middleware_parked_task() -> None:
    mw, _ = _middleware_and_backend()
    registry = get_parked_job_registry()

    async def forever() -> str:
        await asyncio.sleep(10_000)
        return "never"

    task = asyncio.ensure_future(forever())
    await asyncio.sleep(0)
    job_id = registry.allocate()
    registry.register(
        job_id, kind="tool", handle=task, alias="toolu_KILL", tool_name="execute"
    )

    kill_tool = _tool(mw, "kill_job")
    out = await kill_tool.coroutine(job_id=job_id, runtime=_Runtime())

    assert "no background job" not in out
    assert task.cancelled() or task.done(), "kill_job must actually stop the task"
    assert "killed" in out.lower()
    # Killing purges the id from the shared registry — no stale alias, no leak.
    assert registry.resolve(job_id) is None
    assert registry.resolve("toolu_KILL") is None


# ── the two layers never collide on an id ────────────────────────────────


@pytest.mark.asyncio
async def test_shell_job_id_comes_from_the_shared_counter() -> None:
    """Both layers MUST draw ids from the one shared counter.

    A private per-backend counter would restart at ``job_1`` and eventually
    hand two different jobs the same id — the model would then wait on, or
    kill, the wrong work. Pinned by advancing the shared counter first and
    requiring the shell job to continue that sequence rather than restart.
    """
    _, backend = _middleware_and_backend()
    registry = get_parked_job_registry()

    # Advance the shared counter so a private counter would visibly diverge.
    # Absolute values are NOT asserted: the counter is process-wide and never
    # rewinds (reusing an id would let the model act on the wrong job).
    taken = [registry.allocate() for _ in range(3)]
    last = int(taken[-1].removeprefix("job_"))

    await backend.execute("sleep 5", timeout=1)
    shell_id = next(iter(backend._jobs))

    assert shell_id not in taken, (
        f"shell job reused an already-issued id ({shell_id}) — the backend is "
        "minting ids from a private counter instead of the shared registry"
    )
    assert int(shell_id.removeprefix("job_")) == last + 1, (
        f"expected the shared sequence to continue at job_{last + 1}, got "
        f"{shell_id} — a private counter would restart and collide"
    )
    await backend.kill_job(shell_id)


@pytest.mark.asyncio
async def test_shell_and_middleware_ids_are_unique() -> None:
    _, backend = _middleware_and_backend()
    registry = get_parked_job_registry()

    mw_first = registry.allocate()
    await backend.execute("sleep 5", timeout=1)
    shell_id = next(iter(backend._jobs))
    mw_second = registry.allocate()

    assert len({mw_first, shell_id, mw_second}) == 3, (
        f"ids must come from one counter: {mw_first}, {shell_id}, {mw_second}"
    )
    await backend.kill_job(shell_id)


@pytest.mark.asyncio
async def test_failed_tool_job_reports_nonzero_exit_code() -> None:
    """A job that FINISHED but whose tool reported failure must not look like
    a clean success — the content is returned either way, the code differs."""
    from langchain_core.messages import ToolMessage

    _, backend = _middleware_and_backend()
    registry = get_parked_job_registry()

    async def failing_tool() -> ToolMessage:
        await asyncio.sleep(0.05)
        return ToolMessage(
            content="boom", tool_call_id="toolu_FAIL", status="error"
        )

    task = asyncio.ensure_future(failing_tool())
    job_id = registry.allocate()
    registry.register(job_id, kind="tool", handle=task, tool_name="execute")

    result = await backend.wait_for_job(job_id, timeout=10)
    assert "boom" in result.output, "the tool's own output must survive"
    assert result.exit_code == 1, (
        "a failed tool must not be reported as exit code 0"
    )


@pytest.mark.asyncio
async def test_successful_tool_job_reports_zero_exit_code() -> None:
    from langchain_core.messages import ToolMessage

    _, backend = _middleware_and_backend()
    registry = get_parked_job_registry()

    async def ok_tool() -> ToolMessage:
        await asyncio.sleep(0.05)
        return ToolMessage(content="fine", tool_call_id="toolu_OK")

    task = asyncio.ensure_future(ok_tool())
    job_id = registry.allocate()
    registry.register(job_id, kind="tool", handle=task, tool_name="execute")

    result = await backend.wait_for_job(job_id, timeout=10)
    assert "fine" in result.output
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_unknown_id_still_reports_no_background_job() -> None:
    """The genuine 'not found' answer must survive the new lookup path."""
    mw, _ = _middleware_and_backend()
    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(
        job_id="toolu_NEVER_PARKED", timeout=1, runtime=_Runtime()
    )
    assert "no background job" in out
