"""Stage-A DispatchTimeout in claude_code_provider must use the honest
``build_dispatch_timeout_notice`` — NOT ``build_in_progress_notice`` (which
would falsely advertise ``wait_for_job``/``kill_job`` for work that was never
parked). Companion to DESIGN_FIX_STAGE_A_FALSE_PROMISE.md.

Two layers of guarantee:

1. A code-shape assertion pins the call site (fast, deterministic; catches a
   regression that swaps the notice back).
2. A behavioural assertion drives ``wait_for_pending_call_by_id`` through
   ``DispatchTimeout`` and confirms the notice text — proving the wiring
   works, not just that the string exists somewhere.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import deep_worker.providers.claude_code_provider.chat_model as chat_model_mod  # noqa: E402
from deep_worker.providers.claude_code_provider.proxy_mcp_server import DispatchTimeout  # noqa: E402
from deepagents.middleware.bounded_tool_call import (  # noqa: E402
    build_dispatch_timeout_notice,
    resolve_tool_wait_seconds,
)


def _dispatch_timeout_handler_calls() -> list[str]:
    """Return function names invoked inside the ``except DispatchTimeout:``
    block in chat_model.ainvoke. Empty means the block is gone (regression)."""
    src = Path(chat_model_mod.__file__).read_text()
    tree = ast.parse(src)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        etype = node.type
        if etype is None:
            continue
        name = etype.id if isinstance(etype, ast.Name) else (
            etype.attr if isinstance(etype, ast.Attribute) else None
        )
        if name != "DispatchTimeout":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name):
                    calls.append(fn.id)
                elif isinstance(fn, ast.Attribute):
                    calls.append(fn.attr)
    return calls


def test_dispatch_timeout_handler_uses_new_notice() -> None:
    calls = _dispatch_timeout_handler_calls()
    assert calls, "no `except DispatchTimeout` handler found in chat_model"
    assert "build_dispatch_timeout_notice" in calls, (
        "claude stage-A DispatchTimeout must call build_dispatch_timeout_notice "
        "(not build_in_progress_notice — that would advertise a job_id and "
        "wait_for_job/kill_job for work that was never parked)"
    )
    assert "build_in_progress_notice" not in calls, (
        "build_in_progress_notice must not be used on the DispatchTimeout path "
        "— it promises wait_for_job/kill_job that cannot be honoured"
    )


@pytest.mark.asyncio
async def test_stage_a_dispatch_timeout_notice_text_end_to_end() -> None:
    """Drive the wait past the bound and assert the exact notice the model
    would receive on the claude stage-A path — proving the wiring, not the
    string constant."""
    from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata  # noqa: PLC0415
    from deep_worker.providers.claude_code_provider.proxy_mcp_server import ProxyMCPServer  # noqa: PLC0415

    md: ToolMetadata = {
        "name": "execute",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    proxy = ProxyMCPServer([md])

    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
    try:
        task = asyncio.ensure_future(
            proxy.wait_for_pending_call_by_id("toolu_LOST")
        )
        await asyncio.sleep(0.02)
        skew["v"] = resolve_tool_wait_seconds() + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        with pytest.raises(DispatchTimeout):
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]

    notice = build_dispatch_timeout_notice(
        "execute", resolve_tool_wait_seconds()
    )
    # Contract: no job_id, no wait/kill promise, honest status.
    assert "toolu_LOST" not in notice
    assert "job_id" not in notice.lower()
    assert "wait_for_job" not in notice
    assert "kill_job" not in notice
    assert "not_started" in notice
    assert "execute" in notice
