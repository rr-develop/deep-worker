"""Tests for ProxyMCPServer — in-process MCP server for tool delegation.

⚠ Contract after incident 2026-06-14 (POLICIES.md §7):

1. The handler key is the Claude ``tool_use_id`` from MCP
   ``_meta.claudecode/toolUseId``, propagated via the SDK's
   ``_mcp_context`` ContextVar. There is NO synthetic id, NO
   ``_call_counter``, NO ``id(args)``.

2. The handler waits HONESTLY on its Future. For short-running tools:
   bare ``await future``. For tools marked ``long_running=True`` in
   their ``ToolMetadata``: ``await asyncio.wait_for(shield(future),
   long_running_wait_seconds)`` and, on timeout, return an
   intermediate job-handle (NOT ``is_error``) WITHOUT cancelling the
   Future.

3. There is NO ``timeout`` parameter on ``ProxyMCPServer.__init__``.
   The former 120s default was a phantom that lied to the LLM. The
   per-long_running cap is explicit and operator-visible.

Tests use the ``_invoke_handler`` helper which sets the SDK ContextVar
before creating the handler task. Direct calls to
``proxy.tool_handlers[N].handler(args)`` raise ``RuntimeError`` by
design and any test that does so is incompatible with the new
contract (which is the whole point of the fix).
"""
from __future__ import annotations

import asyncio
import itertools
from contextlib import contextmanager
from typing import Any, Generator

import pytest

from claude_agent_sdk._mcp_context import (
    reset_current_claude_tool_use_id,
    set_current_claude_tool_use_id,
)
from claude_code_provider.proxy_mcp_server import ProxyMCPServer
from claude_code_provider.custom_types import MCPResult, ToolMetadata


# ─── Helpers ────────────────────────────────────────────────────────────────


_id_counter = itertools.count(1)


def _fresh_claude_id(prefix: str = "toolu_test") -> str:
    return f"{prefix}_{next(_id_counter):05d}"


@contextmanager
def _claude_id_ctx(claude_id: str) -> Generator[None, None, None]:
    token = set_current_claude_tool_use_id(claude_id)
    try:
        yield
    finally:
        reset_current_claude_tool_use_id(token)


async def _invoke_handler(
    proxy: ProxyMCPServer,
    handler_index: int,
    args: dict[str, Any],
    *,
    claude_id: str | None = None,
) -> tuple[asyncio.Task[MCPResult], str]:
    """Start a handler task with a Claude tool_use_id in the ContextVar.

    Returns ``(task, claude_id)``.
    """
    cid = claude_id or _fresh_claude_id()
    handler = proxy.tool_handlers[handler_index].handler

    async def _runner() -> MCPResult:
        return await handler(args)

    # Set ContextVar BEFORE create_task — Python copies the context
    # snapshot at create_task time.
    with _claude_id_ctx(cid):
        task = asyncio.create_task(_runner())
        # Yield once so the handler reaches its ContextVar read.
        await asyncio.sleep(0)
        return task, cid


async def _wait_for_pending(proxy: ProxyMCPServer, claude_id: str, *, max_steps: int = 50) -> None:
    """Yield the loop until the named pending call appears."""
    for _ in range(max_steps):
        if claude_id in proxy.pending_calls:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"pending call {claude_id!r} did not register")


# ─── Init / construction ────────────────────────────────────────────────────


class TestProxyMCPServerInit:
    def test_init_with_empty_tools(self) -> None:
        proxy = ProxyMCPServer([])
        assert proxy.mcp_server_config is not None
        assert proxy.mcp_server_config["type"] == "sdk"
        assert proxy.mcp_server_config["name"] == "deepworker"

    def test_init_with_single_tool(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        proxy = ProxyMCPServer([sample_tool_metadata[0]])
        assert len(proxy.tool_handlers) == 1

    def test_init_with_multiple_tools(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        assert len(proxy.tool_handlers) == len(sample_tool_metadata)


# ─── Pending-calls management (handler keyed by Claude tool_use_id) ─────────


class TestProxyMCPServerPendingCalls:
    def test_has_pending_calls_initially_false(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        assert proxy.has_pending_calls() is False

    def test_get_pending_calls_initially_empty(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        assert proxy.get_pending_calls() == []

    @pytest.mark.asyncio
    async def test_pending_call_after_handler_invocation(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "test_value"})
        await _wait_for_pending(proxy, cid)

        assert proxy.has_pending_calls() is True
        pending = proxy.get_pending_calls()
        assert len(pending) == 1
        assert pending[0]["name"] == "test_tool"
        assert pending[0]["args"] == {"arg": "test_value"}
        assert pending[0]["id"] == cid  # key IS the Claude tool_use_id

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ─── wait_for_pending_call_by_id (the new identity-keyed waiter) ────────────


class TestWaitForPendingCallById:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_pending_exists(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "x"})
        await _wait_for_pending(proxy, cid)

        pending = await asyncio.wait_for(
            proxy.wait_for_pending_call_by_id(cid), timeout=1.0
        )
        assert pending["id"] == cid
        assert pending["name"] == "test_tool"

        proxy.resolve_call(cid, {"content": [{"type": "text", "text": "ok"}]})
        await task


# ─── wait_for_any_pending_call / wait_for_pending_calls / _matching_* ──────
# Removed 2026-06-17 (variant A, single source of truth): those methods were
# production-dead and each carried an ad-hoc timer (or a bare await). The only
# live stage-A wait is wait_for_pending_call_by_id, bounded by the single
# source — see test_stage_a_bounded_single_source.py and
# docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md.


# ─── Handler blocking and resolve ──────────────────────────────────────────


class TestProxyMCPServerHandlerBlocking:
    @pytest.mark.asyncio
    async def test_handler_blocks_on_future(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, _cid = await _invoke_handler(proxy, 0, {"arg": "test"})
        await asyncio.sleep(0.05)
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_handler_returns_after_resolve(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "test"})
        await _wait_for_pending(proxy, cid)

        result: MCPResult = {"content": [{"type": "text", "text": "Success"}]}
        proxy.resolve_call(cid, result)
        assert (await task) == result


class TestProxyMCPServerResolveCall:
    @pytest.mark.asyncio
    async def test_resolve_with_valid_id(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "test"})
        await _wait_for_pending(proxy, cid)

        result: MCPResult = {"content": [{"type": "text", "text": "Result"}]}
        proxy.resolve_call(cid, result)
        assert (await task) == result

    def test_resolve_with_invalid_id(self, sample_tool_metadata: list[ToolMetadata]) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        proxy.resolve_call("nonexistent_id", {"content": []})  # must not raise

    @pytest.mark.asyncio
    async def test_resolve_removes_from_pending(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "test"})
        await _wait_for_pending(proxy, cid)

        assert proxy.has_pending_calls() is True
        proxy.resolve_call(cid, {"content": []})
        await task
        assert proxy.has_pending_calls() is False
        assert proxy.get_pending_calls() == []


# ─── Multiple simultaneous calls (different ids) ───────────────────────────


class TestProxyMCPServerMultipleCalls:
    @pytest.mark.asyncio
    async def test_multiple_pending_calls(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task1, cid1 = await _invoke_handler(proxy, 0, {"arg": "call1"})
        task2, cid2 = await _invoke_handler(proxy, 1, {"a": 1, "b": 2})
        await _wait_for_pending(proxy, cid1)
        await _wait_for_pending(proxy, cid2)

        assert len(proxy.get_pending_calls()) == 2
        proxy.resolve_call(cid1, {"content": [{"type": "text", "text": "Done"}]})
        proxy.resolve_call(cid2, {"content": [{"type": "text", "text": "Done"}]})
        await task1
        await task2
        assert proxy.has_pending_calls() is False

    @pytest.mark.asyncio
    async def test_resolve_calls_independently(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task1, cid1 = await _invoke_handler(proxy, 0, {"arg": "1"})
        task2, cid2 = await _invoke_handler(proxy, 0, {"arg": "2"})
        await _wait_for_pending(proxy, cid1)
        await _wait_for_pending(proxy, cid2)

        proxy.resolve_call(cid1, {"content": [{"type": "text", "text": "First"}]})
        result1 = await task1
        assert proxy.has_pending_calls() is True

        proxy.resolve_call(cid2, {"content": [{"type": "text", "text": "Second"}]})
        result2 = await task2

        assert result1["content"][0]["text"] == "First"
        assert result2["content"][0]["text"] == "Second"


class TestProxyMCPServerErrorHandling:
    @pytest.mark.asyncio
    async def test_resolve_with_error_flag(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "test"})
        await _wait_for_pending(proxy, cid)

        error_result: MCPResult = {
            "content": [{"type": "text", "text": "Error occurred"}],
            "is_error": True,
        }
        proxy.resolve_call(cid, error_result)
        result = await task
        assert result.get("is_error") is True
        assert "Error occurred" in result["content"][0]["text"]


class TestProxyMCPServerResolveIdempotency:
    @pytest.mark.asyncio
    async def test_second_resolve_is_noop(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "x"})
        await _wait_for_pending(proxy, cid)

        first: MCPResult = {"content": [{"type": "text", "text": "delivered"}]}
        second: MCPResult = {"content": [{"type": "text", "text": "LATE"}]}

        proxy.resolve_call(cid, first)
        proxy.resolve_call(cid, second)  # must be no-op, must not raise
        result = await task
        assert result["content"][0]["text"] == "delivered"

    @pytest.mark.asyncio
    async def test_resolve_unknown_id_is_noop(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        proxy = ProxyMCPServer(sample_tool_metadata)
        proxy.resolve_call("does-not-exist", {"content": [{"type": "text", "text": "x"}]})
        assert not proxy.has_pending_calls()


# ─── Regression: resolve_call must drop pending_calls SYNCHRONOUSLY ─────────
#
# INCIDENT 2026-06-15 (test_claude_code_mcp_prefix double dispatch).
#
# Before the fix, ``resolve_call`` only did ``future.set_result(...)`` and
# left ``pending_calls[call_id]`` in the dict until the spawned handler
# task reached its ``finally: pending_calls.pop(call_id, None)``. Between
# those two scheduling points, the calling code (``chat_model.ainvoke``
# section 8: "tool_resolved and not user_message_sent → check
# has_pending_calls") can run and observe the JUST-RESOLVED call_id as
# still pending — and return a second ``AIMessage(tool_calls=[same id])``
# to LangGraph. Symptom in the wild: `[ainvoke] Returning AIMessage with
# 1 tool_calls: ['ls']` printed twice in a row, identical tool_call_id,
# with no second SDK ``tool_handler_entry`` between them. Cluster:
# test_claude_code_mcp_prefix (both tests), test_restart_self_e2e
# (synchronous-with-notice).
#
# Fix: ``resolve_call`` pops ``pending_calls[call_id]`` BEFORE calling
# ``future.set_result`` — synchronous source-of-truth update. The
# handler ``finally: pending_calls.pop(call_id, None)`` is kept as a
# belt-and-braces no-op for cancellation / parked-job paths.


class TestProxyMCPServerResolveDropsPendingSynchronously:
    @pytest.mark.asyncio
    async def test_resolve_removes_id_before_yielding(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """After ``resolve_call(cid, ...)`` returns, ``cid`` MUST no longer
        appear in ``pending_calls`` — not after the next event-loop tick,
        not after the handler's ``finally`` block, but right now. Any code
        path that calls ``proxy.has_pending_calls()`` between ``resolve_call``
        and the handler task's wake-up MUST see the resolved call gone."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        task, cid = await _invoke_handler(proxy, 0, {"arg": "x"})
        await _wait_for_pending(proxy, cid)

        # Single pending call right now.
        assert cid in proxy.pending_calls
        assert proxy.has_pending_calls() is True

        proxy.resolve_call(cid, {"content": [{"type": "text", "text": "ok"}]})

        # POLICY: resolve_call is the source of truth for "call resolved".
        # The dict must reflect that synchronously, BEFORE the handler
        # task is awakened. Anything observing the proxy state without
        # an intervening ``await`` MUST see the call gone.
        assert cid not in proxy.pending_calls
        assert proxy.has_pending_calls() is False

        # Handler still wakes and returns; finally-pop is now a no-op.
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["content"][0]["text"] == "ok"

    @pytest.mark.asyncio
    async def test_two_pending_resolve_only_drops_resolved_one(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Resolving cid1 must NOT touch cid2 — only the exact resolved
        id is removed. The other still-pending call stays."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        task1, cid1 = await _invoke_handler(proxy, 0, {"arg": "1"})
        task2, cid2 = await _invoke_handler(proxy, 0, {"arg": "2"})
        await _wait_for_pending(proxy, cid1)
        await _wait_for_pending(proxy, cid2)

        proxy.resolve_call(cid1, {"content": [{"type": "text", "text": "1"}]})

        assert cid1 not in proxy.pending_calls
        assert cid2 in proxy.pending_calls
        assert proxy.has_pending_calls() is True
        pending = proxy.get_pending_calls()
        assert [p["id"] for p in pending] == [cid2]

        proxy.resolve_call(cid2, {"content": [{"type": "text", "text": "2"}]})
        assert cid2 not in proxy.pending_calls
        assert proxy.has_pending_calls() is False

        await asyncio.wait_for(asyncio.gather(task1, task2), timeout=1.0)


# ─── Direct invocation without ContextVar raises (D-guard) ─────────────────


class TestHandlerWithoutClaudeIdRaises:
    @pytest.mark.asyncio
    async def test_handler_raises_when_contextvar_empty(
        self, sample_tool_metadata: list[ToolMetadata]
    ) -> None:
        """Calling ``handler(args)`` without setting the SDK ContextVar
        MUST raise ``RuntimeError``. This guard makes silent fallback
        impossible (see POLICIES.md §7)."""
        proxy = ProxyMCPServer(sample_tool_metadata)
        handler = proxy.tool_handlers[0].handler
        with pytest.raises(RuntimeError, match=r"tool_use_id"):
            await handler({"arg": "test"})
