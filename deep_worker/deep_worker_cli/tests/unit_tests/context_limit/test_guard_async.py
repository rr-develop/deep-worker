"""Async regression guard for ``ContextLimitGuardMiddleware``.

agent_a 2026-08-18 incident: Phase 1 commit ``1888357`` shipped
the guard with only ``def wrap_model_call``. Production code paths
on this project ALWAYS use ``agent.ainvoke`` / ``agent.astream``
(RPyC handler, TaskManager, full-dialog mode) — never the sync
``invoke``. LangChain's ``AgentMiddleware.awrap_model_call`` raises
``NotImplementedError`` unless the subclass overrides it, so the
agent's first ainvoke crashed with::

    NotImplementedError: Asynchronous implementation of awrap_model_call
    is not available. You are likely encountering this error because you
    defined only the sync version (wrap_model_call) and invoked your
    agent in an asynchronous context (e.g., using `astream()` or
    `ainvoke()`).

This file pins the async path. Every test below invokes
``awrap_model_call`` directly — same surface LangChain's factory
uses — so a regression that drops ``awrap_model_call`` is caught
before it reaches production.

CRITICAL policy (per the project's CLAUDE.md): async-first. Any
middleware that lacks ``awrap_model_call`` is a runtime fault on
every production invocation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_worker_cli.context_limit.guard import (
    ContextLimitExceeded,
    ContextLimitGuardMiddleware,
)


class _FakeRequest:
    """Minimal duck-typed ModelRequest for the guard tests.

    We avoid importing langchain.agents.middleware.ModelRequest because
    its signature drifts between versions; the guard only reads
    ``.messages``, ``.tools``, and ``.override(messages=...)``.
    """

    def __init__(self, messages, tools=None):
        self.messages = messages
        self.tools = tools if tools is not None else []

    def override(self, **kwargs):
        new = _FakeRequest(self.messages, self.tools)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new



def _reported(used: int, first_id: str = "h1"):
    """A history whose provider reported ``used`` tokens.

    Bands follow the PROVIDER's number (owner directive 2026-08-25); a
    character count no longer means anything to the guard.
    """
    from deep_worker_cli.context_limit.observed_context import (
        annotate_context_observation,
    )

    human = HumanMessage(content="task", id=first_id)
    stamped = AIMessage(content="ok", id="ai-1")
    annotate_context_observation(stamped, used=used, size=None, first_id=first_id)
    return [human, stamped]

def _request(messages=None, tools=None) -> _FakeRequest:
    return _FakeRequest(messages if messages is not None else [], tools=tools)


@pytest.mark.asyncio
async def test_async_no_op_when_limit_is_none() -> None:
    """Operator didn't set -l → guard is a passthrough.

    Regression guard: even with no limit, the async hook MUST exist
    and pass through. Otherwise LangChain's factory raises
    NotImplementedError on the first call.
    """
    guard = ContextLimitGuardMiddleware(context_limit=None, model_name="claude-sonnet-4.5")
    seen: dict[str, Any] = {}

    async def handler(req: Any) -> str:
        seen["req"] = req
        seen["called"] = True
        return "ok"

    result = await guard.awrap_model_call(_request(), handler)
    assert result == "ok"
    assert seen["called"] is True


@pytest.mark.asyncio
async def test_async_green_band_passes_through() -> None:
    """Green band: handler is awaited with the original request."""
    guard = ContextLimitGuardMiddleware(
        context_limit=400_000,
        model_name="claude-sonnet-4.5",
    )
    req = _request([HumanMessage(content="hello")])
    seen: dict[str, Any] = {}

    async def handler(r: Any) -> str:
        seen["called"] = True
        seen["messages"] = list(r.messages)
        return "ok"

    result = await guard.awrap_model_call(req, handler)
    assert result == "ok"
    assert seen["called"] is True
    # No hint or tool message injected.
    assert seen["messages"] == [HumanMessage(content="hello")]


@pytest.mark.asyncio
async def test_async_amber_band_prepends_hint_and_awaits() -> None:
    """Amber band: hint is prepended, handler is awaited with new_request.

    Verifies the async path is the canonical one — uses ``await``
    rather than blocking return. The hint is a SystemMessage and is
    marked with ``CONTEXT_GUARD_MARKER`` so the agent can recognise
    it (and not re-act to it).
    """
    # 2000 chars → ~1000 tokens at claude divisor 2.
    # Amber band is 0.85L ≤ est < 0.95L → with est=1000 we need
    # 1053 ≤ L ≤ 1176; L=1100 puts us at 90.9% (Amber).
    guard = ContextLimitGuardMiddleware(
        context_limit=1100,
        model_name="claude-sonnet-4.5",
    )
    msgs = _reported(1000)
    req = _request(msgs)
    seen: dict[str, Any] = {}

    async def handler(r: Any) -> str:
        seen["called"] = True
        seen["messages"] = list(r.messages)
        return "ok"

    result = await guard.awrap_model_call(req, handler)
    assert result == "ok"
    assert seen["called"] is True
    # First message is the hint (SystemMessage); original message follows.
    assert isinstance(seen["messages"][0], SystemMessage)
    assert seen["messages"][1:] == msgs


@pytest.mark.asyncio
async def test_async_red_band_injects_tool_message_and_awaits() -> None:
    """Red band: ToolMessage injected, handler is awaited.

    The model sees the warning on THIS turn. Critical: handler MUST
    be awaited, not skipped — "block + ToolMessage" alone would stall
    the graph (no AIMessage).
    """
    # 2000 chars → ~1000 tokens at claude divisor 2.
    # Red band is 0.95L ≤ est ≤ L → with est=1000 we need
    # 1000 ≤ L ≤ 1052; L=1020 puts us at 98% (Red).
    guard = ContextLimitGuardMiddleware(
        context_limit=1020,
        model_name="claude-sonnet-4.5",
    )
    msgs = _reported(1000)
    req = _request(msgs)
    seen: dict[str, Any] = {}

    async def handler(r: Any) -> str:
        seen["called"] = True
        seen["messages"] = list(r.messages)
        return "ok"

    result = await guard.awrap_model_call(req, handler)
    assert result == "ok"
    assert seen["called"] is True
    # Original messages preserved; the ToolMessage is appended.
    assert any(isinstance(m, ToolMessage) for m in seen["messages"])
    # Original HumanMessage still in the conversation.
    assert msgs[0] in seen["messages"]


@pytest.mark.asyncio
async def test_async_block_band_raises_context_limit_exceeded() -> None:
    """Block band: handler is NOT awaited, exception is raised.

    The caller (agent-process) catches this and transitions to IDLE.
    Regression guard: the async path must raise (not silently block
    the request).
    """
    guard = ContextLimitGuardMiddleware(
        context_limit=10,  # extreme limit → estimator > limit
        model_name="claude-sonnet-4.5",
    )
    msgs = _reported(500)
    req = _request(msgs)
    seen: dict[str, Any] = {"called": False}

    async def handler(r: Any) -> str:
        seen["called"] = True
        return "should not be called"

    with pytest.raises(ContextLimitExceeded):
        await guard.awrap_model_call(req, handler)
    assert seen["called"] is False


@pytest.mark.asyncio
async def test_async_await_is_actually_async() -> None:
    """Sanity: the handler is awaited, not blocking-called.

    Regression guard for the failure mode where someone delegates
    ``awrap_model_call`` to a sync method that calls
    ``handler(request)`` instead of ``await handler(request)``. The
    inner LLM call is async — calling it without ``await`` returns
    a coroutine object, not the result, and the test below detects
    that by running three concurrent calls (a sync-blocking call
    would not interleave).
    """
    guard = ContextLimitGuardMiddleware(
        context_limit=400_000,
        model_name="claude-sonnet-4.5",
    )

    async def handler(req: Any) -> str:
        await asyncio.sleep(0)  # yield to event loop
        return "ok"

    results = await asyncio.gather(
        guard.awrap_model_call(_request(), handler),
        guard.awrap_model_call(_request(), handler),
        guard.awrap_model_call(_request(), handler),
    )
    assert results == ["ok", "ok", "ok"]


@pytest.mark.asyncio
async def test_async_must_not_be_inherited_notimplemented() -> None:
    """Direct regression test for the agent_a failure.

    LangChain's ``AgentMiddleware.awrap_model_call`` raises
    ``NotImplementedError``. If our subclass did NOT override it
    (the original Phase 1 bug), this test would crash on the first
    ``await`` with that exact error.
    """
    guard = ContextLimitGuardMiddleware(context_limit=400_000)

    async def noop(req: Any) -> None:
        return None

    # Must NOT raise NotImplementedError.
    await asyncio.wait_for(
        guard.awrap_model_call(_request(), noop),
        timeout=1.0,
    )


# ── Sync path tests (kept for parity — must NOT diverge from async) ──


def test_sync_block_band_raises_too() -> None:
    """The sync hook raises the same exception type for Block band.

    This is a contract test — sync and async must fail the same way
    so the caller (agent-process) catches identically. If they
    diverged, an operator running sync tests would get a false sense
    of safety.
    """
    guard = ContextLimitGuardMiddleware(
        context_limit=10,
        model_name="claude-sonnet-4.5",
    )
    # 500 reported tokens against a limit of 10 → Block. The size is
    # stated, not inferred from a character count.
    req = _request(_reported(500))

    def handler(r: Any) -> str:
        return "should not be called"

    with pytest.raises(ContextLimitExceeded):
        guard.wrap_model_call(req, handler)
