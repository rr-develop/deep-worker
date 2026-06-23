"""Unit tests for the No-Agent-Error-State Policy.

See:
- ``docs/POLICIES.md`` §1
- ``docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md``
- ``deep_worker_cli/exceptions.py`` — ``FATAL_EXCEPTION_TYPES`` and
  ``is_fatal_exception``.
- ``deep_worker_cli/middleware/safe_tool_call.py`` — ``SafeToolCallMiddleware``.

These tests cover the failure-mode classification and the tool-dispatch
safety-net middleware.  They do NOT touch ``AgentProcess._handle_invoke``
directly — its behaviour is exercised through the integration test
(``tests/integration/test_no_error_state_policy_live.py``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from deep_worker_cli.exceptions import (
    AgentNotInitialisedError,
    CheckpointCorruptedError,
    FATAL_EXCEPTION_TYPES,
    ProviderAuthFatalError,
    is_fatal_exception,
)


# ---------------------------------------------------------------------------
# §4.1 — fatal-exception registry
# ---------------------------------------------------------------------------


def test_fatal_exception_types_set_is_frozen():
    assert isinstance(FATAL_EXCEPTION_TYPES, frozenset)


def test_fatal_exception_types_contains_expected_classes():
    assert "ClaudeUsageLimitError" in FATAL_EXCEPTION_TYPES
    assert "CodexUsageLimitError" in FATAL_EXCEPTION_TYPES
    assert "GrokUsageLimitError" in FATAL_EXCEPTION_TYPES
    assert "CheckpointCorruptedError" in FATAL_EXCEPTION_TYPES
    assert "ProviderAuthFatalError" in FATAL_EXCEPTION_TYPES
    assert "AgentNotInitialisedError" in FATAL_EXCEPTION_TYPES


def test_is_fatal_exception_for_fatal_classes():
    assert is_fatal_exception(CheckpointCorruptedError("x"))
    assert is_fatal_exception(ProviderAuthFatalError("x"))
    assert is_fatal_exception(AgentNotInitialisedError("x"))


def test_is_fatal_exception_for_recoverable_exceptions():
    # The policy default is "recoverable".  Everything that is not in
    # FATAL_EXCEPTION_TYPES must come back as False.
    assert not is_fatal_exception(RuntimeError("tool blew up"))
    assert not is_fatal_exception(ValueError("bad input"))
    assert not is_fatal_exception(BlockingIOError(11, "EAGAIN"))
    assert not is_fatal_exception(TimeoutError("tool took too long"))


def test_is_fatal_exception_matches_by_class_name_only():
    """Provider-agnostic match: a sibling class with the same name in any
    module must be treated as fatal so codex agents don't import
    claude-code symbols just for classification."""

    class ClaudeUsageLimitError(Exception):  # noqa: N818 — mimicking the real class
        pass

    assert is_fatal_exception(ClaudeUsageLimitError("local stub"))


# ---------------------------------------------------------------------------
# §4.4 — tool-dispatch safety net (SafeToolCallMiddleware)
# ---------------------------------------------------------------------------


@dataclass
class _FakeTool:
    name: str


@dataclass
class _FakeToolCallRequest:
    """Stub mirroring langchain.agents.middleware.types.ToolCallRequest.

    The middleware only reads ``tool_call`` and ``tool``; we keep the
    other fields out to avoid coupling to the langchain dataclass shape.
    """
    tool_call: dict[str, Any]
    tool: Any = None
    state: Any = None
    runtime: Any = None


@pytest.fixture
def safe_mw():
    # Import inside the fixture so a langchain-not-installed environment
    # surfaces the failure here, not at module collection time.
    from deep_worker_cli.middleware.safe_tool_call import SafeToolCallMiddleware
    return SafeToolCallMiddleware()


@pytest.mark.asyncio
async def test_safe_tool_call_passes_through_on_success(safe_mw):
    from langchain_core.messages import ToolMessage

    req = _FakeToolCallRequest(
        tool_call={"id": "call_ok", "name": "ping"},
        tool=_FakeTool(name="ping"),
    )

    async def handler(r):
        return ToolMessage(tool_call_id="call_ok", name="ping", content="pong")

    out = await safe_mw.awrap_tool_call(req, handler)
    assert isinstance(out, ToolMessage)
    assert out.content == "pong"
    assert getattr(out, "status", None) != "error"


@pytest.mark.asyncio
async def test_safe_tool_call_catches_generic_exception(safe_mw):
    from langchain_core.messages import ToolMessage

    req = _FakeToolCallRequest(
        tool_call={"id": "call_x", "name": "explode"},
        tool=_FakeTool(name="explode"),
    )

    async def handler(r):
        raise RuntimeError("kaboom")

    out = await safe_mw.awrap_tool_call(req, handler)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert out.tool_call_id == "call_x"
    assert out.name == "explode"
    assert "RuntimeError" in out.content
    assert "kaboom" in out.content


@pytest.mark.asyncio
async def test_safe_tool_call_catches_blocking_io_error(safe_mw):
    """The original failure mode: a stray print() inside a tool raising
    ``BlockingIOError`` used to propagate up to the agent invoke
    catch-all and flip the whole agent into AgentStatus.ERROR.  The
    safety-net must convert it into a tool-result error."""
    from langchain_core.messages import ToolMessage

    req = _FakeToolCallRequest(
        tool_call={"id": "call_io", "name": "spammy"},
        tool=_FakeTool(name="spammy"),
    )

    async def handler(r):
        raise BlockingIOError(11, "EAGAIN")

    out = await safe_mw.awrap_tool_call(req, handler)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "BlockingIOError" in out.content


@pytest.mark.asyncio
async def test_safe_tool_call_does_not_swallow_cancelled(safe_mw):
    req = _FakeToolCallRequest(
        tool_call={"id": "call_c", "name": "slow"},
        tool=_FakeTool(name="slow"),
    )

    async def handler(r):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await safe_mw.awrap_tool_call(req, handler)


@pytest.mark.asyncio
async def test_safe_tool_call_uses_unknown_id_when_missing(safe_mw):
    """Defensive: a malformed ToolCallRequest (no id) must still produce
    a ToolMessage rather than crashing the safety net itself."""
    from langchain_core.messages import ToolMessage

    req = _FakeToolCallRequest(tool_call={"name": "noid"}, tool=_FakeTool(name="noid"))

    async def handler(r):
        raise ValueError("oops")

    out = await safe_mw.awrap_tool_call(req, handler)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert out.tool_call_id == "unknown"


def test_safe_tool_call_sync_path():
    """The sync ``wrap_tool_call`` path must mirror the async one."""
    from deep_worker_cli.middleware.safe_tool_call import SafeToolCallMiddleware
    from langchain_core.messages import ToolMessage

    mw = SafeToolCallMiddleware()
    req = _FakeToolCallRequest(
        tool_call={"id": "call_s", "name": "sync"},
        tool=_FakeTool(name="sync"),
    )

    def handler(r):
        raise RuntimeError("sync kaboom")

    out = mw.wrap_tool_call(req, handler)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "RuntimeError" in out.content
    assert "sync kaboom" in out.content
