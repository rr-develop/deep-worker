"""POLICY: every tool gets the job-handle mechanism BY DEFAULT.

Hard policy (2026-06-16, project owner):

* The job-handle mechanism (wait ``long_running_wait_seconds`` for the
  Future; if it has not resolved, park the live Future as a job and return
  an informational ``in_progress`` result with ``job_id`` +
  ``wait_for_job``/``kill_job`` actions) applies to **EVERY** tool BY
  DEFAULT. No bare unbounded ``await future`` is allowed.
* A tool may opt OUT only EXPLICITLY (``long_running=False``), and an
  explicit opt-out MUST carry its own ``timeout`` — opting out without a
  timeout is FORBIDDEN (it would reintroduce the eternal hang). On that
  timeout the handler returns an informational (NON-error) result so the
  agent simply continues; it is NEVER ``is_error=true``.
* No timeout ever turns into an agent ERROR. The agent is told what
  happened and keeps working (e.g. retries the tool).
* This is provider-agnostic and applies to existing AND future tools.

A previous agent claimed this mechanism was "implemented"; it was not —
``long_running`` was never set on any ToolMetadata, so the gate was always
False and every tool fell through to a bare ``await future`` that hung
forever (the live ``edit_file`` deadlock, mj1_init 2026-06-16). This suite
pins the corrected behaviour.

See docs/POLICIES.md §7 and the ProxyMCPServer handler.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)

from claude_code_provider.custom_types import ToolMetadata
from claude_code_provider.proxy_mcp_server import ProxyMCPServer

pytestmark = pytest.mark.asyncio


def _md(name: str, **extra: Any) -> ToolMetadata:
    md: dict[str, Any] = {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    md.update(extra)
    return md  # type: ignore[return-value]


async def _run_until_handle(proxy: ProxyMCPServer, tool: str, call_id: str):
    handler = proxy._create_tool_handler(tool)  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id(call_id)
    try:
        return await asyncio.wait_for(handler({"x": 1}), timeout=2.0)
    finally:
        reset_current_claude_tool_use_id(token)


# ── Default: ANY tool without a long_running flag → job-handle ──────────


async def test_unflagged_tool_degrades_to_job_handle_not_hang() -> None:
    """A tool with NO ``long_running`` in its metadata (the common case —
    e.g. ``edit_file``) must NOT hang forever. After the configured wait it
    returns a non-error job-handle and keeps the Future alive."""
    proxy = ProxyMCPServer([_md("edit_file")], long_running_wait_seconds=0.05)

    result = await _run_until_handle(proxy, "edit_file", "toolu_EDIT")

    # Non-error informational result.
    assert result.get("is_error") is not True
    text = result["content"][0]["text"]
    assert "in_progress" in text
    # ``job_id`` is a canonical ``job_N`` from the shared ParkedJobRegistry —
    # NOT the ``tool_use_id``. Advertising ``toolu_…`` was the observed defect
    # (2026-07-31): ``wait_for_job``/``kill_job`` could never resolve it.
    import re
    m = re.search(r"\bjob_\d+\b", text)
    assert m, f"notice must advertise a canonical job_N id: {text!r}"
    job_id = m.group(0)
    assert "toolu_EDIT" not in text, f"tool_use_id leaked into notice: {text!r}"
    assert "wait_for_job" in text and "kill_job" in text
    # The real Future is parked (under the canonical id), not cancelled.
    reg = proxy.get_job_registry()
    assert job_id in reg and not reg[job_id]["future"].done()
    reg[job_id]["future"].cancel()


async def test_unflagged_tool_resolves_normally_before_wait() -> None:
    """Fast path still works: if the Future resolves before the wait, the
    real result is returned (no job-handle)."""
    proxy = ProxyMCPServer([_md("read_file")], long_running_wait_seconds=5.0)
    handler = proxy._create_tool_handler("read_file")  # pyright: ignore[reportPrivateUsage]
    token = set_current_claude_tool_use_id("toolu_FAST")
    try:
        task = asyncio.ensure_future(handler({"path": "/x"}))
        await asyncio.sleep(0.02)
        proxy.resolve_call(
            "toolu_FAST", {"content": [{"type": "text", "text": "data"}]}
        )
        result = await asyncio.wait_for(task, timeout=1.0)
        assert "data" in result["content"][0]["text"]
        assert proxy.get_job_registry() == {}
    finally:
        reset_current_claude_tool_use_id(token)


# ── Explicit opt-out MUST carry a timeout; never bare-await ─────────────


async def test_explicit_optout_without_timeout_is_rejected() -> None:
    """Opting out of the job-handle mechanism (``long_running=False``)
    WITHOUT a ``timeout`` is forbidden — it would reintroduce the eternal
    hang. Construction must fail loudly."""
    with pytest.raises((ValueError, TypeError, AssertionError)):
        ProxyMCPServer([_md("noisy", long_running=False)])  # no timeout


async def test_explicit_optout_with_timeout_returns_notice_not_error() -> None:
    """A tool that explicitly opts out WITH a timeout: on timeout the
    handler returns a NON-error 'timed out, still running' notice (the
    agent continues), never is_error=true, and never hangs forever."""
    proxy = ProxyMCPServer([_md("quick", long_running=False, timeout=0.05)])

    result = await _run_until_handle(proxy, "quick", "toolu_Q")

    assert result.get("is_error") is not True, (
        "a timeout must NOT become an agent error — inform and continue"
    )
    text = result["content"][0]["text"]
    # Notice must advertise a canonical ``job_N`` (observed fix, 2026-07-31),
    # NOT the ``tool_use_id``.
    import re
    assert re.search(r"\bjob_\d+\b", text), (
        f"notice must advertise a canonical job_N id: {text!r}"
    )
    assert "toolu_Q" not in text, f"tool_use_id leaked into notice: {text!r}"
    # It must convey still-running info so the agent can decide.
    assert "still" in text.lower()
    # Cleanup: parked Future.
    for entry in proxy.get_job_registry().values():
        entry["future"].cancel()


async def test_no_tool_ever_waits_forever() -> None:
    """Belt-and-braces: neither a default tool nor an explicit-opt-out tool
    parks beyond its bound. Both return within the wait + slack."""
    proxy = ProxyMCPServer(
        [_md("a"), _md("b", long_running=False, timeout=0.05)],
        long_running_wait_seconds=0.05,
    )
    for name, cid in (("a", "toolu_a"), ("b", "toolu_b")):
        result = await _run_until_handle(proxy, name, cid)
        assert result.get("is_error") is not True
        assert isinstance(result["content"][0]["text"], str)
