"""Integration (codex provider): a never-resolving tool degrades to a
job-handle instead of hanging — with the REAL production wait (120s),
proven without waiting 120s by mocking ONLY the clock.

No other mocks. The ProxyMCPServer, its handler, the asyncio Future, and
the wait/job-handle logic are all real. Only the event-loop clock is
advanced so the production ``long_running_wait_seconds`` (120s) deadline
fires instantly. This mirrors
``claude_code_sdk_provider/tests/test_job_handle_integration.py`` — the
mechanism is COMMON to all providers (HARD POLICY, 2026-06-16).

Pins the policy: EVERY tool gets the job-handle mechanism by default; a
Future that never resolves (the live ``edit_file`` hang, mj1_init) must
return a non-error ``in_progress`` handle, never park forever, never
become an agent error.

See docs/POLICIES.md §7.1 and test_job_handle_by_default.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from deep_worker.providers.grok_provider.custom_types import ToolMetadata  # noqa: E402
from deep_worker.providers.grok_provider.proxy_mcp_server import ProxyMCPServer  # noqa: E402

pytestmark = pytest.mark.asyncio

# Policy default (~2 minutes). Asserted against the proxy's resolved value
# so a regression in the default is caught here.
_POLICY_DEFAULT_WAIT = 120.0


def _md(name: str, **extra: Any) -> ToolMetadata:
    md: dict[str, Any] = {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    md.update(extra)
    return md  # type: ignore[return-value]


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

    loop = asyncio.get_running_loop()
    real_time = loop.time
    # Skew added to the loop clock. asyncio.wait_for builds its deadline
    # from loop.time() + 120; once the handler has parked we bump the skew
    # past 120 so the SAME real timer fires "in the future" without us
    # waiting 120 real seconds. Only the clock is mocked.
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]

    try:
        task = asyncio.ensure_future(
            proxy.call_tool_for_test("edit_file", {"path": "/f", "old": "a", "new": "b"})
        )
        # Let the handler register its pending call and enter the wait.
        await asyncio.sleep(0.05)
        assert not task.done(), "handler should be parked in the bounded wait"
        assert proxy.has_pending_calls()

        # Advance the clock past 120s and wake the loop so the wait_for
        # timer re-evaluates its (now-past) deadline.
        skew["v"] = _POLICY_DEFAULT_WAIT + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        # Bounded real wait for the result (must be quick now).
        result = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]

    # Non-error job-handle, never a hang, never an error.
    assert not any(b.get("is_error") is True for b in result)
    text = " ".join(b.get("text", "") for b in result)
    assert "in_progress" in text
    assert "wait_for_job" in text and "kill_job" in text

    # The real work is preserved (parked), not cancelled/lost.
    reg = proxy.get_job_registry()
    assert len(reg) == 1
    job = next(iter(reg.values()))
    assert not job["future"].done()
    assert job["tool_name"] == "edit_file"
    job["future"].cancel()
