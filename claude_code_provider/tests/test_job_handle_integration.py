"""Integration: a never-resolving tool degrades to a job-handle instead of
hanging — with the REAL production wait (120s), proven without waiting
120s by mocking ONLY the clock.

No other mocks. The ProxyMCPServer, its handler, the asyncio Future, and
the wait/job-handle logic are all real. Only the event-loop clock is
advanced so the production ``long_running_wait_seconds`` (120s) deadline
fires instantly.

This pins the HARD POLICY (2026-06-16): EVERY tool gets the job-handle
mechanism by default; a Future that never resolves (the live ``edit_file``
hang, mj1_init) must return a non-error ``in_progress`` handle, never park
forever, never become an agent error.

See docs/POLICIES.md §7 and test_job_handle_by_default.py.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)

from claude_code_provider.custom_types import ToolMetadata
from claude_code_provider.proxy_mcp_server import ProxyMCPServer

pytestmark = pytest.mark.integration

# Policy default (~2 minutes). Asserted against the proxy's resolved value
# so a regression in the default is caught here.
_POLICY_DEFAULT_WAIT = 120.0


def _md(name: str, **extra: object) -> ToolMetadata:
    md: dict[str, object] = {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    md.update(extra)
    return md  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_edit_file_never_resolving_becomes_job_handle_at_120s() -> None:
    """A real ``edit_file``-shaped tool whose Future never resolves must,
    after the PRODUCTION 120s wait, return a non-error job-handle — proven
    by advancing ONLY the clock (no other mocks)."""
    # Real proxy with the real production default (no per-test override).
    proxy = ProxyMCPServer([_md("edit_file")])
    # The resolved default IS the policy value (~2 minutes) for EVERY tool.
    assert proxy._wait_policy["edit_file"] == (  # pyright: ignore[reportPrivateUsage]
        "job_handle", _POLICY_DEFAULT_WAIT
    )

    handler = proxy._create_tool_handler("edit_file")  # pyright: ignore[reportPrivateUsage]

    loop = asyncio.get_running_loop()
    real_time = loop.time
    # Skew added to the loop clock. asyncio.wait_for builds its deadline
    # from loop.time() + 120; once the handler has parked we bump the skew
    # past 120 so the SAME real timer fires "in the future" without us
    # waiting 120 real seconds. Only the clock is mocked.
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]

    token = set_current_claude_tool_use_id("toolu_EDIT")
    try:
        task = asyncio.ensure_future(handler({"path": "/f", "old": "a", "new": "b"}))
        # Let the handler register its pending call and enter the wait.
        await asyncio.sleep(0.05)
        assert not task.done(), "handler should be parked in the bounded wait"
        assert "toolu_EDIT" in proxy.pending_calls

        # Advance the clock past 120s and wake the loop so the wait_for
        # timer re-evaluates its (now-past) deadline.
        skew["v"] = _POLICY_DEFAULT_WAIT + 5.0
        # Nudge the loop a few times so the timer callback runs.
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        # Bounded real wait for the result (must be quick now).
        result = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]
        reset_current_claude_tool_use_id(token)

    # Non-error job-handle, never a hang, never an error.
    assert result.get("is_error") is not True
    text = result["content"][0]["text"]
    assert "in_progress" in text
    assert "toolu_EDIT" in text
    assert "wait_for_job" in text and "kill_job" in text

    # The real work is preserved (parked), not cancelled/lost.
    reg = proxy.get_job_registry()
    assert "toolu_EDIT" in reg and not reg["toolu_EDIT"]["future"].done()
    reg["toolu_EDIT"]["future"].cancel()
