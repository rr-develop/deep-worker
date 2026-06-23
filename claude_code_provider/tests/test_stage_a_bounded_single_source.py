"""Stage A (the CLI dispatch wait) is bounded by the SINGLE timer source.

HARD POLICY (POLICIES.md §7.1, owner 2026-06-17, variant A): the wait for the
control-channel ``tools/call`` to register (``wait_for_pending_call_by_id``)
stays in ``ainvoke`` — it is structurally required there (sequential SDK
dispatch + authoritative args). But it must NOT be an unbounded ``await``: it
is bounded by the SAME source of truth as every other tool wait —
``deepagents.middleware.bounded_tool_call.resolve_tool_wait_seconds`` — and on
timeout it degrades to a NON-error ``in_progress`` handle, never hangs.

Before this: ``return await fut`` (no timeout) hung mj1_init / mj2_vac when a
control request never arrived. This suite pins the bound. Mocks ONLY the clock.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from claude_code_provider.custom_types import ToolMetadata  # noqa: E402
from claude_code_provider.proxy_mcp_server import (  # noqa: E402
    DispatchTimeout,
    ProxyMCPServer,
)

pytestmark = pytest.mark.asyncio


def _md(name: str) -> ToolMetadata:
    return {
        "name": name,
        "description": f"d:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


async def test_stage_a_uses_single_timer_source() -> None:
    """The bound comes from the shared resolve_tool_wait_seconds (not a private
    constant). A control request that never arrives → DispatchTimeout, bounded
    by that source — proven by mocking ONLY the clock."""
    from deepagents.middleware.bounded_tool_call import resolve_tool_wait_seconds

    proxy = ProxyMCPServer([_md("execute")])
    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
    try:
        task = asyncio.ensure_future(
            proxy.wait_for_pending_call_by_id("toolu_NEVER")
        )
        await asyncio.sleep(0.02)
        assert not task.done(), "must be bounded-waiting, not instant"
        # Advance the clock past the shared bound.
        skew["v"] = resolve_tool_wait_seconds() + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        with pytest.raises(DispatchTimeout):
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]


async def test_stage_a_returns_when_call_registers() -> None:
    """Happy path unchanged: when the call registers, the waiter returns it
    (no timeout)."""
    proxy = ProxyMCPServer([_md("read_file")])
    handler = proxy._create_tool_handler("read_file")  # pyright: ignore[reportPrivateUsage]

    from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
        reset_current_claude_tool_use_id,
        set_current_claude_tool_use_id,
    )

    token = set_current_claude_tool_use_id("toolu_OK")
    try:
        htask = asyncio.ensure_future(handler({"path": "/x"}))
        for _ in range(50):
            if "toolu_OK" in proxy.pending_calls:
                break
            await asyncio.sleep(0.001)
        pending = await asyncio.wait_for(
            proxy.wait_for_pending_call_by_id("toolu_OK"), timeout=1.0
        )
        assert pending["id"] == "toolu_OK"
        proxy.resolve_call("toolu_OK", {"content": [{"type": "text", "text": "ok"}]})
        await htask
    finally:
        reset_current_claude_tool_use_id(token)
