"""POLICY (codex provider): every tool gets the job-handle mechanism BY
DEFAULT — identical contract to claude_code_provider.

Hard policy (project owner): the mechanism that, after
``long_running_wait_seconds`` (default 120s, env-overridable via
``DW_LONG_RUNNING_WAIT_SECONDS``), parks a still-running tool's Future as a
job and returns a NON-error ``in_progress`` notice (``job_id`` +
``wait_for_job``/``kill_job``) applies to EVERY tool on EVERY provider —
no exceptions. A bare unbounded wait is forbidden. Opting out is EXPLICIT
only (``long_running=False`` AND a required ``timeout``); a timeout-less
opt-out is rejected at construction. A timeout NEVER becomes an error.

Before this change grok_provider used a fixed ``timeout`` that, on
expiry, returned an ``is_error`` block AND dropped the call (work lost) —
violating the policy. This suite pins the corrected behaviour.

See docs/POLICIES.md §7.1.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from grok_provider.custom_types import ToolMetadata  # noqa: E402
from grok_provider.proxy_mcp_server import ProxyMCPServer  # noqa: E402

pytestmark = pytest.mark.asyncio


def _md(name: str, **extra: Any) -> ToolMetadata:
    md: dict[str, Any] = {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    md.update(extra)
    return md  # type: ignore[return-value]


def _text(blocks: list[dict[str, Any]]) -> str:
    return " ".join(b.get("text", "") for b in blocks)


def _is_error(blocks: list[dict[str, Any]]) -> bool:
    return any(b.get("is_error") is True for b in blocks)


# ── Default: ANY tool without a long_running flag → job-handle ──────────


async def test_unflagged_tool_degrades_to_job_handle_not_hang() -> None:
    proxy = ProxyMCPServer([_md("edit_file")], long_running_wait_seconds=0.05)
    blocks = await proxy.call_tool_for_test("edit_file", {"x": 1})

    assert not _is_error(blocks)
    text = _text(blocks)
    assert "in_progress" in text
    assert "wait_for_job" in text and "kill_job" in text
    # Real Future parked, not cancelled.
    reg = proxy.get_job_registry()
    assert len(reg) == 1
    (job_id, job), = reg.items()
    assert job_id in text
    assert not job["future"].done()
    job["future"].cancel()


async def test_unflagged_tool_resolves_normally_before_wait() -> None:
    proxy = ProxyMCPServer([_md("read_file")], long_running_wait_seconds=5.0)
    import asyncio

    task = asyncio.ensure_future(proxy.call_tool_for_test("read_file", {"x": 1}))
    await asyncio.sleep(0.02)
    pend = proxy.get_pending_calls()
    assert len(pend) == 1
    proxy.resolve_call(
        pend[0]["id"], {"content": [{"type": "text", "text": "data"}]}
    )
    blocks = await asyncio.wait_for(task, timeout=1.0)
    assert _text(blocks) == "data"
    assert proxy.get_job_registry() == {}


# ── Explicit opt-out MUST carry a timeout; never bare-await ─────────────


async def test_explicit_optout_without_timeout_is_rejected() -> None:
    with pytest.raises((ValueError, TypeError, AssertionError)):
        ProxyMCPServer([_md("noisy", long_running=False)])  # no timeout


async def test_explicit_optout_with_timeout_returns_notice_not_error() -> None:
    proxy = ProxyMCPServer([_md("quick", long_running=False, timeout=0.05)])
    blocks = await proxy.call_tool_for_test("quick", {"x": 1})
    assert not _is_error(blocks), "a timeout must NOT be an error"
    text = _text(blocks)
    assert "timed out" in text.lower() or "still" in text.lower()
    assert "job_id" in text or "wait_for_job" in text


async def test_no_tool_ever_waits_forever() -> None:
    proxy = ProxyMCPServer(
        [_md("a"), _md("b", long_running=False, timeout=0.05)],
        long_running_wait_seconds=0.05,
    )
    for name in ("a", "b"):
        blocks = await proxy.call_tool_for_test(name, {"x": 1})
        assert not _is_error(blocks)
        assert isinstance(_text(blocks), str) and _text(blocks)
    # clean up parked jobs
    for _jid, job in list(proxy.get_job_registry().items()):
        job["future"].cancel()
