"""Tests for the wait_for_job / kill_job tools exposed by
CLIFilesystemMiddleware.

When `execute` times out, the command keeps running in the background and
the agent gets a job_id (see test_background_jobs.py). These tools let the
agent act on that job. They must:
  * be present in the middleware's tool set (so they reach the model);
  * resolve the SAME CLIShellBackend instance that ran execute (shared job
    registry);
  * round-trip a real background job end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_worker_cli.backends import CLIFilesystemMiddleware, CLIShellBackend


def _middleware_and_backend() -> tuple[CLIFilesystemMiddleware, CLIShellBackend]:
    backend = CLIShellBackend(root_dir=Path("/tmp"), inherit_env=True)
    mw = CLIFilesystemMiddleware(backend=backend)
    return mw, backend


class _Runtime:
    """Minimal stand-in for ToolRuntime — only needs to be passed through to
    _get_backend, which (for a plain backend instance) ignores it."""


def _tool(mw: CLIFilesystemMiddleware, name: str):
    for t in mw.tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found; have {[t.name for t in mw.tools]}")


def test_job_tools_are_registered() -> None:
    mw, _ = _middleware_and_backend()
    names = {t.name for t in mw.tools}
    assert "execute" in names
    assert "wait_for_job" in names
    assert "kill_job" in names


def test_wait_for_job_tool_has_job_id_and_timeout_params() -> None:
    mw, _ = _middleware_and_backend()
    schema = _tool(mw, "wait_for_job").args
    assert "job_id" in schema
    assert "timeout" in schema


def test_kill_job_tool_has_job_id_param() -> None:
    mw, _ = _middleware_and_backend()
    schema = _tool(mw, "kill_job").args
    assert "job_id" in schema


@pytest.mark.asyncio
async def test_wait_for_job_tool_round_trip() -> None:
    """execute times out → tool waits → returns finished output.

    We call the tool's underlying coroutine directly, passing a fake runtime.
    Invoking through ``.ainvoke`` would require LangChain to inject the
    ``ToolRuntime`` (it rejects a hand-built one via pydantic); the coroutine
    itself only uses ``runtime`` to resolve the backend, which for a plain
    backend instance ignores it.
    """
    mw, backend = _middleware_and_backend()

    started = await backend.execute("sleep 2 && echo done", timeout=1)
    job_id = next(iter(backend._jobs))
    assert "job_" in started.output

    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(job_id=job_id, timeout=10, runtime=_Runtime())
    assert "done" in out
    assert "exit code 0" in out
    assert backend._jobs == {}


@pytest.mark.asyncio
async def test_kill_job_tool_round_trip() -> None:
    """execute times out → kill tool stops the job."""
    mw, backend = _middleware_and_backend()

    await backend.execute("sleep 30", timeout=1)
    job_id = next(iter(backend._jobs))

    kill_tool = _tool(mw, "kill_job")
    out = await kill_tool.coroutine(job_id=job_id, runtime=_Runtime())
    assert f"'{job_id}' killed" in out
    assert backend._jobs == {}


@pytest.mark.asyncio
async def test_wait_for_unknown_job_via_tool() -> None:
    mw, _ = _middleware_and_backend()
    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(job_id="job_404", timeout=1, runtime=_Runtime())
    assert "no background job" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
