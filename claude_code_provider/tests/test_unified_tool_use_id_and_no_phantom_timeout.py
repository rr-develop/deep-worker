# pyright: reportPrivateUsage=false
"""Regression test: the new MCP-handler contract after the related
internal incident (analysis not included here).

Background — the related internal incident
==========================================
The "three identical copies of the same code block" symptom (observed
live) was caused by TWO defects in ProxyMCPServer + chat_model
that amplified each other:

 A) Two-worlds identity. ``_create_tool_handler`` synthesised its own
    ``call_id = f"{tool_name}_{counter}_{id(args)}"``, ignoring the
    ``_meta.claudecode/toolUseId`` that the MCP protocol carries with
    every ``tools/call``. ``chat_model.py`` held the visible-stream
    ``ToolUseBlock.id`` instead. The two were matched by ``(tool_name,
    args)`` — which the authors themselves admitted does not work
    ("args may diverge from the control-channel args", chat_model.py:
    1288-1290 in the pre-fix code).

 B) Phantom 120-second timeout. ``ProxyMCPServer.__init__(timeout=120.0)``
    bounded an INTERNAL plumbing Future (handler waiting for
    ``resolve_call`` from LangGraph). When it fired it sent the LLM
    a synthetic ``is_error=true, "Tool execution timeout after 120
    seconds"`` — a lie. Claude retried, triggering yet another real
    execution of the same logical call. The 120s cap also killed any
    legitimate long-running tool (build, pytest, indexer).

Fix plan — what THIS test pins down
====================================

POLICIES.md §7 (No-Arbitrariness) requires that the fix kill the root
cause, not paper over the symptoms with caches, TTLs, dedup sets, or
N-second windows. So:

 A) ONE identity end-to-end: the Claude ``tool_use_id`` from MCP
    ``_meta.claudecode/toolUseId`` flows through the SDK's
    ``_mcp_context`` ContextVar (already implemented in the vendored
    SDK), is read by ``_create_tool_handler`` and
    stored as the ``pending_calls`` key.

 B) No phantom timeout. ``ProxyMCPServer.__init__`` no longer accepts
    a ``timeout``. ``_create_tool_handler`` does plain ``await
    future`` for short-running tools. No silent ``is_error=true`` on
    plumbing waits.

 C) ``ToolMetadata`` gains a ``long_running: bool`` flag. For
    long-running tools, the handler may return an intermediate
    "job handle" (``status: in_progress``, ``job_id``,
    ``actions: [wait_for_job, kill_job]``) WITHOUT cancelling the
    real Future. The Future keeps running on the ToolNode side and is
    later picked up by ``wait_for_job``.

 D) Diagnostic guards: missing Claude id at handler start → raise
    (silent fallback is forbidden by §7).

⚠ READ POLICIES.md §7 BEFORE WEAKENING OR DELETING ANY OF THESE
TESTS. They pin INVARIANTS that prevent the next maintainer from
reintroducing the cluster (4 incidents in 3 weeks before this).
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from claude_code_provider.proxy_mcp_server import ProxyMCPServer
from claude_code_provider.custom_types import ToolMetadata


# ─── Fixture tools ───────────────────────────────────────────────────────────

def _md(name: str, **extra: Any) -> ToolMetadata:
    """ToolMetadata factory.

    Accepts arbitrary keyword args so tests can pass
    ``long_running=True`` without choking when the field is not yet
    in the TypedDict (the field is added in this PR — see Part C
    tests below).
    """
    md: dict[str, Any] = {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    md.update(extra)
    return md  # type: ignore[return-value]


def _short(name: str = "read_file") -> ToolMetadata:
    return _md(name)


def _long(name: str = "execute") -> ToolMetadata:
    return _md(name, long_running=True)


# ─── Part A — single ``tool_use_id`` end-to-end ──────────────────────────────


class TestPartAUnifiedToolUseId:
    """The handler uses the SDK-provided Claude ``tool_use_id`` as its
    ``pending_calls`` key. No synthetic counter+id(args) fallback.
    """

    @pytest.mark.asyncio
    async def test_handler_uses_claude_tool_use_id_as_pending_key(self) -> None:
        """When the SDK ContextVar carries a ``tool_use_id``, that exact
        string MUST be the ``pending_calls`` key — so ``chat_model.py``
        can resolve_call(block.id, ...) and hit the same entry.
        """
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        proxy = ProxyMCPServer([_short("read_file")])
        handler = proxy._create_tool_handler("read_file")

        # Simulate what the SDK does in Query._handle_sdk_mcp_request —
        # set the ContextVar before awaiting the handler.
        token = set_current_claude_tool_use_id("toolu_01ABC")
        try:
            handler_task = asyncio.create_task(handler({"path": "/x"}))
            # Wait for the handler to register its pending call.
            for _ in range(50):
                if proxy.pending_calls:
                    break
                await asyncio.sleep(0.001)
            assert "toolu_01ABC" in proxy.pending_calls, (
                f"handler did not use Claude tool_use_id as key — got "
                f"{list(proxy.pending_calls)!r}. Identity-A invariant"
                f" broken."
            )
            # Cleanup — resolve so the task can return.
            proxy.resolve_call("toolu_01ABC",
                               {"content": [{"type": "text", "text": "ok"}]})
            await handler_task
        finally:
            reset_current_claude_tool_use_id(token)

    @pytest.mark.asyncio
    async def test_handler_raises_when_no_claude_id_in_contextvar(self) -> None:
        """If the SDK ContextVar is empty (e.g. older CLI version,
        misconfiguration), the handler MUST raise immediately. Silent
        fallback to a synthetic ``f"{name}_{counter}_{id(args)}"`` is
        forbidden by §7 — that fallback is exactly what created the
        two-worlds identity bug.
        """
        proxy = ProxyMCPServer([_short("read_file")])
        handler = proxy._create_tool_handler("read_file")

        # ContextVar default is None — no set_current_claude_tool_use_id.
        with pytest.raises(RuntimeError, match=r"tool_use_id"):
            await handler({"path": "/x"})

    @pytest.mark.asyncio
    async def test_no_synthetic_call_counter_attribute(self) -> None:
        """The ``_call_counter`` is gone. Its presence in old code
        suggested it was the identity source; readers must not
        rediscover it and revive synthetic ids.
        """
        proxy = ProxyMCPServer([_short("read_file")])
        assert not hasattr(proxy, "_call_counter"), (
            "_call_counter is the legacy synthetic-id counter and MUST "
            "be removed."
        )


# ─── Part A (cont.) — wait_for_pending_call_by_id ──────────────────────────


class TestPartAWaitForPendingCallById:
    """chat_model.py needs an id-keyed wait method. Without it, the
    only way to "wait for the right pending call" remains by-name
    matching, which is the broken pre-fix behaviour.
    """

    @pytest.mark.asyncio
    async def test_returns_immediately_when_pending_already_exists(self) -> None:
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        proxy = ProxyMCPServer([_short("read_file")])
        handler = proxy._create_tool_handler("read_file")

        token = set_current_claude_tool_use_id("toolu_X")
        try:
            handler_task = asyncio.create_task(handler({"path": "/x"}))
            for _ in range(50):
                if "toolu_X" in proxy.pending_calls:
                    break
                await asyncio.sleep(0.001)

            # Already there — must return without timeout.
            pending = await asyncio.wait_for(
                proxy.wait_for_pending_call_by_id("toolu_X"), timeout=1.0,
            )
            assert pending["id"] == "toolu_X"
            assert pending["name"] == "read_file"

            proxy.resolve_call("toolu_X",
                               {"content": [{"type": "text", "text": "ok"}]})
            await handler_task
        finally:
            reset_current_claude_tool_use_id(token)

    @pytest.mark.asyncio
    async def test_returns_when_pending_appears_later(self) -> None:
        """Race-tolerant: visible ``ToolUseBlock`` may arrive before the
        SDK delivers the control-channel ``tools/call``. The waiter must
        park on an event-driven Future and wake the moment the matching
        id registers (NO polling, §2).
        """
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        proxy = ProxyMCPServer([_short("read_file")])
        handler = proxy._create_tool_handler("read_file")

        async def _arm_handler() -> None:
            await asyncio.sleep(0.02)
            token = set_current_claude_tool_use_id("toolu_LATE")
            try:
                await handler({"path": "/x"})
            finally:
                reset_current_claude_tool_use_id(token)

        async def _resolve_eventually() -> None:
            for _ in range(50):
                if "toolu_LATE" in proxy.pending_calls:
                    break
                await asyncio.sleep(0.001)
            proxy.resolve_call("toolu_LATE",
                               {"content": [{"type": "text", "text": "ok"}]})

        arm_task = asyncio.create_task(_arm_handler())
        resolve_task = asyncio.create_task(_resolve_eventually())

        pending = await asyncio.wait_for(
            proxy.wait_for_pending_call_by_id("toolu_LATE"), timeout=2.0,
        )
        assert pending["id"] == "toolu_LATE"

        await arm_task
        await resolve_task

    @pytest.mark.asyncio
    async def test_only_wakes_for_matching_id(self) -> None:
        """A pending call with a different id MUST NOT wake the waiter.
        This is the whole point of identity-keyed waits.
        """
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        proxy = ProxyMCPServer([_short("read_file"), _short("grep")])
        handler_rf = proxy._create_tool_handler("read_file")
        handler_gr = proxy._create_tool_handler("grep")

        async def _arm_wrong() -> None:
            token = set_current_claude_tool_use_id("toolu_WRONG")
            try:
                await handler_rf({"path": "/x"})
            finally:
                reset_current_claude_tool_use_id(token)

        async def _arm_right() -> None:
            await asyncio.sleep(0.05)
            token = set_current_claude_tool_use_id("toolu_RIGHT")
            try:
                await handler_gr({"pattern": "foo"})
            finally:
                reset_current_claude_tool_use_id(token)

        wrong_task = asyncio.create_task(_arm_wrong())
        right_task = asyncio.create_task(_arm_right())

        # Wait specifically for RIGHT — must not return when WRONG
        # registers first.
        pending = await asyncio.wait_for(
            proxy.wait_for_pending_call_by_id("toolu_RIGHT"), timeout=2.0,
        )
        assert pending["id"] == "toolu_RIGHT"
        assert pending["name"] == "grep"

        # Cleanup
        proxy.resolve_call("toolu_WRONG",
                           {"content": [{"type": "text", "text": "ok"}]})
        proxy.resolve_call("toolu_RIGHT",
                           {"content": [{"type": "text", "text": "ok"}]})
        await wrong_task
        await right_task


# ─── Part B — no phantom timeout ─────────────────────────────────────────────


class TestPartBNoPhantomTimeout:
    """``ProxyMCPServer.timeout`` is gone. No 120-second wait_for
    around the handler Future. Short-running tools wait as long as
    LangGraph ToolNode is honestly working.
    """

    def test_init_does_not_accept_timeout_kwarg(self) -> None:
        """Defensive: if a caller still passes ``timeout=``, fail loudly
        rather than silently accept an ignored value.
        """
        with pytest.raises(TypeError):
            ProxyMCPServer([_short()], timeout=120.0)  # type: ignore[call-arg]

    def test_proxy_has_no_timeout_attribute(self) -> None:
        proxy = ProxyMCPServer([_short()])
        assert not hasattr(proxy, "timeout"), (
            "ProxyMCPServer.timeout is the phantom 120s that lied to "
            "the LLM. It MUST NOT exist anymore. See §7."
        )

    @pytest.mark.asyncio
    async def test_no_phantom_error_on_timeout(self) -> None:
        """No-phantom-timeout means: a timeout NEVER reports ``is_error=
        true`` to the LLM (which caused retries + duplicate side effects in
        the related internal incident, not included here). It does NOT mean
        "wait forever".

        HARD POLICY: every tool is bounded — by default it
        degrades to a non-error job-handle (see
        ``test_job_handle_by_default.py``). This test pins the surviving
        part of the §7 contract: when the bound is hit, the result is
        informational, never an error, and the real Future is preserved.
        """
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        # Default (no flag) → job-handle path; tiny wait so the test is
        # fast. The Future never resolves here.
        proxy = ProxyMCPServer([_short("read_file")],
                               long_running_wait_seconds=0.05)
        handler = proxy._create_tool_handler("read_file")

        token = set_current_claude_tool_use_id("toolu_LONG")
        try:
            result = await asyncio.wait_for(handler({"path": "/x"}), timeout=1.0)
            # NEVER an error — informational handle.
            assert result.get("is_error") is not True, (
                "a timeout must NOT be is_error=true — that lie caused the "
                "retry/duplicate bug (§7)."
            )
            text = result["content"][0]["text"]
            assert "in_progress" in text and "toolu_LONG" in text
            # The real Future is parked, not cancelled/lost.
            reg = proxy.get_job_registry()
            assert "toolu_LONG" in reg and not reg["toolu_LONG"]["future"].done()
            reg["toolu_LONG"]["future"].cancel()

            # And the normal fast-resolve path still returns the real result.
            set_current_claude_tool_use_id("toolu_FAST")
            proxy2 = ProxyMCPServer([_short("read_file")],
                                    long_running_wait_seconds=5.0)
            h2 = proxy2._create_tool_handler("read_file")
            t2 = asyncio.ensure_future(h2({"path": "/x"}))
            await asyncio.sleep(0.02)
            proxy2.resolve_call(
                "toolu_FAST", {"content": [{"type": "text", "text": "ok"}]}
            )
            r2 = await asyncio.wait_for(t2, timeout=0.5)
            assert r2 == {"content": [{"type": "text", "text": "ok"}]}
        finally:
            reset_current_claude_tool_use_id(token)


# ─── Part C — long_running → job-handle policy ──────────────────────────────


class TestPartCLongRunningJobHandle:
    """For tools tagged ``long_running=True``, the handler may return
    an intermediate job-handle to Claude — NOT an ``is_error``. The
    real work keeps running on the ToolNode side and is later picked
    up by ``wait_for_job``.

    The cap on how long to wait before issuing the handle is policy,
    explicit, and per-tool. Default for long_running tools (when not
    configured): 60s (subject to a future per-tool override). This is
    not arbitrary; the value is documented and visible.
    """

    def test_tool_metadata_accepts_long_running_flag(self) -> None:
        """``ToolMetadata`` recognises ``long_running``. Without it the
        whole policy has nowhere to live.
        """
        md = _long("execute")
        assert md.get("long_running") is True

    @pytest.mark.asyncio
    async def test_long_running_handler_returns_job_handle_not_error(self) -> None:
        """When a long_running tool's Future does not resolve within
        the configured wait, the handler returns
        ``{"status": "in_progress", "job_id": <claude_tool_use_id>,
          "actions": [...]}`` and the underlying Future is NOT
        cancelled. NEVER ``is_error=true`` on a still-running job.
        """
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        # Use a very short long_running_wait_seconds so the test is fast.
        proxy = ProxyMCPServer(
            [_long("execute")],
            long_running_wait_seconds=0.05,
        )
        handler = proxy._create_tool_handler("execute")

        token = set_current_claude_tool_use_id("toolu_LR")
        try:
            result = await asyncio.wait_for(
                handler({"command": "sleep 999"}), timeout=1.0,
            )

            # Shape contract — see the related internal incident
            # analysis (not included here), Part C.
            assert "is_error" not in result or result.get("is_error") is not True
            text_payload = result["content"][0]["text"]
            assert "in_progress" in text_payload
            assert "toolu_LR" in text_payload  # job_id == tool_use_id
            assert "wait_for_job" in text_payload
            assert "kill_job" in text_payload

            # Real Future is still alive — JobRegistry holds it.
            registry = proxy.get_job_registry()
            assert "toolu_LR" in registry
            assert not registry["toolu_LR"]["future"].done()

            # Cleanup so test does not leak the task.
            registry["toolu_LR"]["future"].cancel()
        finally:
            reset_current_claude_tool_use_id(token)

    @pytest.mark.asyncio
    async def test_explicit_optout_tool_uses_its_own_timeout_notice(self) -> None:
        """A tool may opt OUT of the default job-handle path ONLY by
        EXPLICITLY setting ``long_running=False`` AND its own ``timeout``.
        It then uses the timeout-notice path: bounded wait, then a NON-error
        "still running" notice (NEVER a hang, NEVER is_error). The per-tool
        policy lives in metadata. See test_job_handle_by_default.py.
        """
        from claude_agent_sdk._mcp_context import (  # noqa: PLC0415
            set_current_claude_tool_use_id,
            reset_current_claude_tool_use_id,
        )

        proxy = ProxyMCPServer(
            [_md("read_file", long_running=False, timeout=0.05)],
        )
        handler = proxy._create_tool_handler("read_file")

        token = set_current_claude_tool_use_id("toolu_S")
        try:
            # Future never resolves → bounded by the explicit timeout.
            result = await asyncio.wait_for(handler({"path": "/x"}), timeout=1.0)
            assert result.get("is_error") is not True
            text = result["content"][0]["text"]
            assert "toolu_S" in text
            assert "timed out" in text.lower() or "still" in text.lower()
            # Work preserved (parked), not lost.
            assert "toolu_S" in proxy.get_job_registry()
            proxy.get_job_registry()["toolu_S"]["future"].cancel()

            # Fast-resolve path still returns the real result.
            set_current_claude_tool_use_id("toolu_S2")
            proxy2 = ProxyMCPServer(
                [_md("read_file", long_running=False, timeout=5.0)],
            )
            h2 = proxy2._create_tool_handler("read_file")
            t2 = asyncio.ensure_future(h2({"path": "/x"}))
            await asyncio.sleep(0.02)
            proxy2.resolve_call(
                "toolu_S2", {"content": [{"type": "text", "text": "ok"}]}
            )
            r2 = await asyncio.wait_for(t2, timeout=0.5)
            assert r2["content"][0]["text"] == "ok"
        finally:
            reset_current_claude_tool_use_id(token)


# ─── Part D — diagnostic guards (loud failures) ─────────────────────────────


class TestPartDDiagnosticGuards:
    """Silent fallbacks were the root cause across this whole cluster
    (incidents 06-12, 06-13, 06-14). Every new fallback is a §7
    violation. These tests pin a few hot guards.
    """

    @pytest.mark.asyncio
    async def test_missing_claude_id_raises_not_silent(self) -> None:
        """Same invariant as in Part A — repeated here because §7 elevates
        it from "implementation detail" to "non-negotiable policy".
        """
        proxy = ProxyMCPServer([_short("ls")])
        handler = proxy._create_tool_handler("ls")
        with pytest.raises(RuntimeError, match=r"tool_use_id"):
            await handler({})

    def test_no_legacy_completed_ids_cache(self) -> None:
        """``8d0ae52`` (on branch ``tasks``, not in master) introduced
        a ``_completed_ids`` FIFO 1000 cache to dedup replayed visible
        blocks. POLICIES.md §7 explicitly forbids this style of fix —
        with the unified id (Part A), there is nothing to dedup,
        because one id maps to one pending entry.
        """
        proxy = ProxyMCPServer([_short("ls")])
        assert not hasattr(proxy, "_completed_ids"), (
            "_completed_ids is the masking dedup-cache forbidden by §7."
        )
        assert not hasattr(proxy, "_completed_queue"), (
            "_completed_queue is the masking dedup-cache forbidden by §7."
        )
        assert not hasattr(proxy, "is_completed_call"), (
            "is_completed_call is the masking dedup API forbidden by §7."
        )
