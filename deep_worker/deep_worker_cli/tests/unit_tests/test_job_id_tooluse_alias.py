"""wait_for_job / kill_job must accept BOTH id namespaces.

Live incident (mj2_vac, 2026-06-17): `execute` timed out and parked a job
as ``job_1``; the model then called ``wait_for_job(job_id="toolu_018k…")``
— passing the originating *tool_use_id* instead of the advertised
``job_1`` — and got ``Error: no background job``. It looped on that
recoverable error until it went idle.

Fix (this suite pins it): a parked job is reachable by EITHER id —
its canonical ``job_N`` OR the ``tool_use_id`` of the ``execute`` call
that spawned it. The model can no longer "guess wrong". Finishing/killing
a job via one id removes BOTH keys (no stale alias, no leak).

The tool_use_id is captured at park time from the claude provider's
``_mcp_context`` ContextVar; when absent (e.g. no provider context) only
the ``job_N`` key exists and everything still works.
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
    """Minimal stand-in for ToolRuntime (ignored by a plain backend)."""


def _tool(mw: CLIFilesystemMiddleware, name: str):
    for t in mw.tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found")


# ── tool_use_id captured at park time ───────────────────────────────────


@pytest.mark.asyncio
async def test_parked_job_is_reachable_by_tool_use_id() -> None:
    from claude_agent_sdk._mcp_context import (
        reset_current_claude_tool_use_id,
        set_current_claude_tool_use_id,
    )

    mw, backend = _middleware_and_backend()
    token = set_current_claude_tool_use_id("toolu_TESTWAIT")
    try:
        started = await backend.execute("sleep 2 && echo done", timeout=1)
    finally:
        reset_current_claude_tool_use_id(token)

    # Canonical job_N still minted and advertised in the notice.
    assert "Background job id: job_" in started.output
    job_id = next(k for k in backend._jobs)

    # Waiting by the tool_use_id (what the model actually sent) must work.
    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(
        job_id="toolu_TESTWAIT", timeout=10, runtime=_Runtime()
    )
    assert "done" in out
    assert "exit code 0" in out
    # Both the canonical key AND the alias are gone — no leak.
    assert backend._jobs == {}
    assert getattr(backend, "_job_aliases", {}) == {}
    del job_id


@pytest.mark.asyncio
async def test_kill_job_by_tool_use_id() -> None:
    from claude_agent_sdk._mcp_context import (
        reset_current_claude_tool_use_id,
        set_current_claude_tool_use_id,
    )

    mw, backend = _middleware_and_backend()
    token = set_current_claude_tool_use_id("toolu_TESTKILL")
    try:
        await backend.execute("sleep 30", timeout=1)
    finally:
        reset_current_claude_tool_use_id(token)

    kill_tool = _tool(mw, "kill_job")
    out = await kill_tool.coroutine(job_id="toolu_TESTKILL", runtime=_Runtime())
    assert "killed" in out
    assert backend._jobs == {}
    assert getattr(backend, "_job_aliases", {}) == {}


@pytest.mark.asyncio
async def test_canonical_job_id_still_works_with_alias_present() -> None:
    """The job_N path must keep working even when an alias was registered."""
    from claude_agent_sdk._mcp_context import (
        reset_current_claude_tool_use_id,
        set_current_claude_tool_use_id,
    )

    mw, backend = _middleware_and_backend()
    token = set_current_claude_tool_use_id("toolu_BOTH")
    try:
        await backend.execute("sleep 2 && echo viacanon", timeout=1)
    finally:
        reset_current_claude_tool_use_id(token)

    job_id = next(k for k in backend._jobs)
    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(job_id=job_id, timeout=10, runtime=_Runtime())
    assert "viacanon" in out
    assert backend._jobs == {}
    assert getattr(backend, "_job_aliases", {}) == {}


@pytest.mark.asyncio
async def test_no_tool_use_id_context_only_canonical_key() -> None:
    """With no provider context set, only the job_N key exists; no crash."""
    mw, backend = _middleware_and_backend()
    await backend.execute("sleep 2 && echo plain", timeout=1)
    assert len(backend._jobs) == 1
    assert getattr(backend, "_job_aliases", {}) == {}
    job_id = next(iter(backend._jobs))
    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(job_id=job_id, timeout=10, runtime=_Runtime())
    assert "plain" in out


@pytest.mark.asyncio
async def test_unknown_id_still_reports_no_background_job() -> None:
    mw, _ = _middleware_and_backend()
    wait_tool = _tool(mw, "wait_for_job")
    out = await wait_tool.coroutine(
        job_id="toolu_NEVER_PARKED", timeout=1, runtime=_Runtime()
    )
    assert "no background job" in out


# ── the description the model sees must disambiguate the two id spaces ───


def test_job_id_description_disambiguates_from_tool_use_id() -> None:
    mw, _ = _middleware_and_backend()
    for tool_name in ("wait_for_job", "kill_job"):
        schema = _tool(mw, tool_name).args
        desc = (schema["job_id"].get("description") or "").lower()
        # Mentions the concrete form...
        assert "job_" in desc
        # ...and explicitly disclaims the tool_use_id confusion.
        assert "tool_use_id" in desc or "tool use id" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
