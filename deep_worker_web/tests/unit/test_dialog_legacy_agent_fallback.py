"""Tests for graceful fallback when the agent process is running an
older version of ``_handle_get_history`` that does not understand
``limit=None``.

Background
----------
The dialog-text endpoint sends ``limit=None`` to mean "return everything"
(see DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §3.6 RPyC). New agent
processes accept this. But agents that were started BEFORE this code
landed still have the old handler that does ``total - limit - offset``
and crashes with ``unsupported operand type(s) for -: 'int' and 'NoneType'``.

The fix MUST not require restarting every running agent. The web service
must detect that specific failure and retry with a very large positive
``limit`` (effectively "everything" — bigger than any plausible
conversation). This keeps Full Dialog reliable regardless of which
agent build is running.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


def _make_service_with_fake_root(fake_root: Any) -> AgentService:
    """Build a service whose ``_roots`` already contains a stubbed RPyC root.

    The stub is the only thing exercised by ``get_agent_dialog_formatted``;
    no real RPyC needed.
    """
    registry = MockProcessRegistry()
    registry.add_mock_agent("legacy-agent", pid=42, rpyc_port=9999)
    svc = AgentService(registry=registry)
    svc._roots["legacy-agent"] = fake_root  # type: ignore[attr-defined]
    return svc


def _hist_payload(messages: list[dict[str, Any]]) -> str:
    """JSON-serialised get_history success response."""
    return json.dumps({"success": True, "history": messages, "total": len(messages), "has_more": False})


def _legacy_error_payload() -> str:
    """The exact error message a pre-fix agent emits when it gets limit=None."""
    return json.dumps({
        "success": False,
        "error": "Failed to retrieve history: unsupported operand type(s) for -: 'int' and 'NoneType'",
    })


# ---------------------------------------------------------------------------
# Test fixtures: a legacy agent (rejects None) and a new agent (accepts None)
# ---------------------------------------------------------------------------


def _legacy_root_factory(messages: list[dict[str, Any]]):
    """Return a fake RPyC root that mimics a legacy agent.

    On first call (limit=None) returns the legacy error.
    On second call with positive int limit returns the messages.
    """
    calls: list[dict[str, Any]] = []

    async def get_history(limit: Any = 50, offset: int = 0) -> str:
        calls.append({"limit": limit, "offset": offset})
        # Legacy server cannot handle None: it returns success=False with the
        # exact arithmetic-error string we're trying to detect.
        if limit is None:
            return _legacy_error_payload()
        # Positive int — legacy contract works (return last `limit` messages).
        if isinstance(limit, int) and limit > 0:
            return _hist_payload(messages[-limit:])
        # Defensive: any other shape — empty.
        return _hist_payload([])

    root = MagicMock()
    root.get_history = MagicMock(side_effect=get_history)
    root._calls = calls  # for assertions
    return root


def _new_root_factory(messages: list[dict[str, Any]]):
    """Return a fake RPyC root that mimics a new agent (accepts limit=None)."""
    calls: list[dict[str, Any]] = []

    async def get_history(limit: Any = None, offset: int = 0) -> str:
        calls.append({"limit": limit, "offset": offset})
        # New server: any sentinel = everything.
        return _hist_payload(messages)

    root = MagicMock()
    root.get_history = MagicMock(side_effect=get_history)
    root._calls = calls
    return root


def _broken_root_factory(error_message: str):
    """Return a fake RPyC root that ALWAYS errors out (genuine failure)."""
    calls: list[dict[str, Any]] = []

    async def get_history(limit: Any = None, offset: int = 0) -> str:
        calls.append({"limit": limit, "offset": offset})
        return json.dumps({"success": False, "error": error_message})

    root = MagicMock()
    root.get_history = MagicMock(side_effect=get_history)
    root._calls = calls
    return root


HUMAN = {"type": "HumanMessage", "id": "h1", "content": "ping"}
AI_OK = {"type": "AIMessage", "id": "a1", "content": "pong"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLegacyAgentFallback:
    @pytest.mark.asyncio
    async def test_legacy_agent_falls_back_to_positive_limit(self) -> None:
        """When agent rejects limit=None with the legacy arithmetic error,
        the service must retry with a large positive int and succeed.
        """
        root = _legacy_root_factory([HUMAN, AI_OK])
        svc = _make_service_with_fake_root(root)

        out = await svc.get_agent_dialog_formatted("legacy-agent")

        # Two calls: first with None (rejected), second with positive int.
        assert len(root._calls) == 2, f"expected 2 calls, got {root._calls}"
        assert root._calls[0]["limit"] is None
        assert isinstance(root._calls[1]["limit"], int)
        assert root._calls[1]["limit"] >= 1_000_000, (
            "fallback limit must be huge — no artificial caps"
        )

        # Output must contain the rendered turn from the messages
        assert "pong" in out
        assert "--- User ---" in out

    @pytest.mark.asyncio
    async def test_new_agent_succeeds_on_first_call(self) -> None:
        """When agent accepts limit=None, no second call is made."""
        root = _new_root_factory([HUMAN, AI_OK])
        svc = _make_service_with_fake_root(root)

        out = await svc.get_agent_dialog_formatted("legacy-agent")

        assert len(root._calls) == 1, f"new agent should need only 1 call, got {root._calls}"
        assert root._calls[0]["limit"] is None
        assert "pong" in out

    @pytest.mark.asyncio
    async def test_genuine_error_not_retried(self) -> None:
        """If the agent returns a non-None-related error, do NOT retry —
        propagate the error so the user sees the real diagnostic.
        """
        root = _broken_root_factory("Database is on fire")
        svc = _make_service_with_fake_root(root)

        with pytest.raises(ValueError, match="Database is on fire"):
            await svc.get_agent_dialog_formatted("legacy-agent")

        # Only ONE call — no pointless retry.
        assert len(root._calls) == 1

    @pytest.mark.asyncio
    async def test_legacy_error_variants_all_trigger_retry(self) -> None:
        """The detector must match the family of NoneType arithmetic errors,
        not just the exact string. Be defensive against minor wording drift.
        """
        variants = [
            "Failed to retrieve history: unsupported operand type(s) for -: 'int' and 'NoneType'",
            "Failed to retrieve history: unsupported operand type(s) for -: 'NoneType' and 'int'",
            "unsupported operand type(s) for -: 'int' and 'NoneType'",
            # Hypothetical: someone reorders the operands or wraps differently
            "TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'",
        ]
        for err_msg in variants:
            calls: list[dict[str, Any]] = []

            async def get_history(limit: Any = None, offset: int = 0) -> str:
                calls.append({"limit": limit})
                if limit is None:
                    return json.dumps({"success": False, "error": err_msg})
                return _hist_payload([HUMAN, AI_OK])

            root = MagicMock()
            root.get_history = MagicMock(side_effect=get_history)
            svc = _make_service_with_fake_root(root)

            out = await svc.get_agent_dialog_formatted("legacy-agent")

            assert "pong" in out, f"variant {err_msg!r} did not trigger fallback"
            assert len(calls) == 2, f"variant {err_msg!r} did not retry"
