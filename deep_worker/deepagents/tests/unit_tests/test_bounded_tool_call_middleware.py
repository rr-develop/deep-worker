"""The ONE place every tool wait/timeout lives: BoundedToolCallMiddleware.

HARD POLICY (POLICIES.md §7.1, owner 2026-06-17): the 2-minute bound on a
tool call is enforced in EXACTLY ONE function — ``awrap_tool_call`` — for
EVERY provider (deepseek/openai/anthropic/google/vertexai/claude-code-sdk/codex)
and EVERY tool. No other place may wait on a tool. On the wait elapsing the
tool's work is parked in the background (NOT cancelled) and the model gets a
NON-error ``in_progress`` ToolMessage (job_id + wait_for_job/kill_job). A
timeout is NEVER ``status="error"``.

These tests mock ONLY the clock (loop.time) so the production 120s bound
fires instantly. No other mocks.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain.tools.tool_node import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.middleware.bounded_tool_call import (
    DEFAULT_TOOL_WAIT_SECONDS,
    IMAGINE_TOOL_WAIT_SECONDS,
    BoundedToolCallMiddleware,
    resolve_effective_tool_wait_seconds,
)

pytestmark = pytest.mark.asyncio


def _request(tool_call_id: str = "toolu_TEST", name: str = "execute") -> ToolCallRequest:
    state = {"messages": []}
    runtime = ToolRuntime(
        state=state, context=None, tool_call_id=tool_call_id,
        store=None, stream_writer=lambda _: None, config={},
    )
    return ToolCallRequest(
        runtime=runtime,
        tool_call={"id": tool_call_id, "name": name, "args": {"x": 1}},
        state=state,
        tool=None,
    )


# ── fast tool passes straight through ────────────────────────────────────


async def test_fast_tool_passes_through_unchanged() -> None:
    mw = BoundedToolCallMiddleware()
    expected = ToolMessage(content="done", tool_call_id="toolu_TEST")

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        return expected

    out = await mw.awrap_tool_call(_request(), handler)
    assert out is expected


# ── slow tool → non-error in_progress job-handle, work NOT cancelled ──────


async def test_slow_tool_becomes_inprogress_not_error() -> None:
    mw = BoundedToolCallMiddleware(wait_seconds=DEFAULT_TOOL_WAIT_SECONDS)
    started = asyncio.Event()
    cancelled = {"v": False}

    async def never_returns(_req: ToolCallRequest) -> ToolMessage:
        started.set()
        try:
            await asyncio.sleep(10_000)
        except asyncio.CancelledError:
            cancelled["v"] = True
            raise
        return ToolMessage(content="never", tool_call_id="toolu_TEST")

    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
    try:
        task = asyncio.ensure_future(mw.awrap_tool_call(_request(), never_returns))
        await started.wait()
        await asyncio.sleep(0.02)
        assert not task.done(), "must still be waiting under the bound"
        # Advance the clock past the production wait so the bound fires.
        skew["v"] = DEFAULT_TOOL_WAIT_SECONDS + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        out = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]

    assert isinstance(out, ToolMessage)
    # Never an error.
    assert out.status != "error"
    text = out.content if isinstance(out.content, str) else str(out.content)
    assert "in_progress" in text
    assert "wait_for_job" in text and "kill_job" in text
    # job_id is a CANONICAL job_N — the same namespace wait_for_job/kill_job
    # resolve. It must NOT be the tool_call_id: advertising a ``toolu_…`` as a
    # job id gave the model an unresolvable handle.
    # See tests/unit_tests/test_parked_job_id_namespace.py.
    import re as _re

    assert _re.search(r"\bjob_\d+\b", text), text
    assert "toolu_TEST" not in text
    # The tool_call_id round-trips so LangGraph pairs the message — that field
    # is the LangChain protocol, NOT a job id.
    assert out.tool_call_id == "toolu_TEST"
    # The real work was shielded — NOT cancelled by the timeout.
    await asyncio.sleep(0)
    assert cancelled["v"] is False


# ── stage A (CLI dispatch wait) is now UNDER the same bound ───────────────


async def test_stage_a_dispatch_wait_is_bounded() -> None:
    """A proxy provider's dispatch wait that NEVER completes (lost control
    request — the mj2_vac hole) must hit the SAME timer and degrade to a
    non-error in_progress handle, NOT hang forever."""
    from deepagents.middleware.tool_dispatch_bridge import (
        register_dispatch_waiter,
        unregister_dispatch_waiter,
    )

    tid = "agent_stageA"

    async def never_dispatches(_call_id: str):
        await asyncio.sleep(10_000)  # control request that never arrives
        return {"x": 1}

    register_dispatch_waiter(tid, never_dispatches)

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        # Must NEVER be reached — dispatch never completes.
        return ToolMessage(content="should-not-run", tool_call_id="toolu_TEST")

    state = {"messages": []}
    runtime = ToolRuntime(
        state=state, context=None, tool_call_id="toolu_TEST",
        store=None, stream_writer=lambda _: None,
        config={"configurable": {"thread_id": tid}},
    )
    req = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "toolu_TEST", "name": "execute", "args": {"x": 1}},
        state=state, tool=None,
    )

    mw = BoundedToolCallMiddleware()
    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"v": 0.0}
    loop.time = lambda: real_time() + skew["v"]  # type: ignore[method-assign]
    try:
        task = asyncio.ensure_future(mw.awrap_tool_call(req, handler))
        await asyncio.sleep(0.02)
        assert not task.done(), "stage-A wait must be bounded, not instant"
        skew["v"] = DEFAULT_TOOL_WAIT_SECONDS + 5.0
        for _ in range(5):
            await asyncio.sleep(0)
            if task.done():
                break
        out = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    finally:
        loop.time = real_time  # type: ignore[method-assign]
        unregister_dispatch_waiter(tid)

    assert isinstance(out, ToolMessage)
    assert out.status != "error"
    assert "in_progress" in str(out.content)


async def test_stage_a_dispatch_patches_authoritative_args() -> None:
    """When dispatch completes, the AUTHORITATIVE args from the provider
    replace the announced args before the tool runs."""
    from deepagents.middleware.tool_dispatch_bridge import (
        register_dispatch_waiter,
        unregister_dispatch_waiter,
    )

    tid = "agent_args"
    seen_args: dict[str, Any] = {}

    async def dispatch(_call_id: str):
        return {"pattern": "unsubscribe"}  # authoritative (announced was "..._messages")

    register_dispatch_waiter(tid, dispatch)

    async def handler(req: ToolCallRequest) -> ToolMessage:
        seen_args.update(req.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="toolu_TEST")

    state = {"messages": []}
    runtime = ToolRuntime(
        state=state, context=None, tool_call_id="toolu_TEST",
        store=None, stream_writer=lambda _: None,
        config={"configurable": {"thread_id": tid}},
    )
    req = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "toolu_TEST", "name": "grep", "args": {"pattern": "unsubscribe_messages"}},
        state=state, tool=None,
    )
    try:
        out = await BoundedToolCallMiddleware().awrap_tool_call(req, handler)
    finally:
        unregister_dispatch_waiter(tid)

    assert seen_args == {"pattern": "unsubscribe"}
    assert isinstance(out, ToolMessage) and out.content == "ok"


async def test_no_dispatch_waiter_is_noop_for_normal_providers() -> None:
    """A normal provider registers no waiter — tool runs immediately with the
    args the LLM returned."""
    seen: dict[str, Any] = {}

    async def handler(req: ToolCallRequest) -> ToolMessage:
        seen.update(req.tool_call["args"])
        return ToolMessage(content="normal", tool_call_id="toolu_TEST")

    out = await BoundedToolCallMiddleware().awrap_tool_call(_request(), handler)
    assert isinstance(out, ToolMessage) and out.content == "normal"
    assert seen == {"x": 1}


async def test_wait_seconds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DW_LONG_RUNNING_WAIT_SECONDS", "37")
    mw = BoundedToolCallMiddleware()
    assert mw.wait_seconds == 37.0


async def test_default_wait_is_120() -> None:
    assert DEFAULT_TOOL_WAIT_SECONDS == 120.0
    mw = BoundedToolCallMiddleware()
    assert mw.wait_seconds == 120.0


async def test_imagine_tool_wait_override() -> None:
    assert resolve_effective_tool_wait_seconds("GenerateImage") == IMAGINE_TOOL_WAIT_SECONDS
    assert resolve_effective_tool_wait_seconds("image_edit") == IMAGINE_TOOL_WAIT_SECONDS
    assert resolve_effective_tool_wait_seconds(
        "GrokBuiltin", {"_grok_original_title": "GenerateImage"}
    ) == IMAGINE_TOOL_WAIT_SECONDS
    assert resolve_effective_tool_wait_seconds(
        "GrokBuiltin", {"_grok_original_title": "Shell"}
    ) == DEFAULT_TOOL_WAIT_SECONDS


# ── static invariant: no second timer source, no bare await on a tool ─────


def _provider_sources() -> list[tuple[str, str]]:
    """Return ``(label, source)`` for every proxy provider under guard.

    The providers used to live at the repository root
    (``codex_provider/proxy_mcp_server.py``); they now live under
    ``deep_worker/providers/`` and the old root packages are deprecation
    shims with no ``proxy_mcp_server`` module. The guards below kept reading
    the old path, so they raised ``FileNotFoundError`` instead of checking
    anything — an AST guard that cannot find its target is not a guard.
    Resolving the path here, once, with an explicit existence assertion keeps
    a future move loud instead of silently unguarded.
    """
    from pathlib import Path

    providers = Path(__file__).resolve().parents[3] / "providers"
    sources: list[tuple[str, str]] = []
    for name in ("codex_provider",):
        path = providers / name / "proxy_mcp_server.py"
        assert path.is_file(), (
            f"{path} not found — the tool-wait guard has no target to check. "
            "If the providers moved again, update this resolver; do not let "
            "the invariant go unverified."
        )
        sources.append((f"{name}/proxy_mcp_server.py", path.read_text()))
    return sources


def test_providers_delegate_to_single_timer_source() -> None:
    """Every proxy provider under guard must source its tool-wait bound from
    resolve_tool_wait_seconds — NO independent 120s constant survives."""
    import re

    for rel, src in _provider_sources():
        # The shared source is referenced.
        assert "resolve_tool_wait_seconds" in src, rel
        # No independent timer constant assignment (a second source of truth).
        assert not re.search(
            r"_DEFAULT_LONG_RUNNING_WAIT_SECONDS\s*=\s*\d", src
        ), f"{rel} still defines an independent timer constant"


def test_no_bare_await_on_tool_future_in_providers() -> None:
    """No bare ``await fut`` / ``await future`` statement may exist on the tool
    path — every wait goes through asyncio.wait_for (the single bound). Matches
    only real statements (``await fut`` at end-of-line or before ``)``/``,``),
    not docstring prose like 'NO bare await future'."""
    import ast

    for rel, src in _provider_sources():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            # A bare ``await <Name>`` expression (not wrapped in a call like
            # asyncio.wait_for(...)) on an identifier named fut/future.
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Name):
                assert node.value.id not in ("fut", "future"), (
                    f"bare unbounded ``await {node.value.id}`` in {rel} "
                    f"(line {node.lineno}) — must go through asyncio.wait_for"
                )
