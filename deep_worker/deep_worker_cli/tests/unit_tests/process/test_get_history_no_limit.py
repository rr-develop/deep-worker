"""Tests for the no-limit contract on AgentProcess._handle_get_history.

Per design (DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §6/§3.6 RPyC),
the agent-side history retrieval must support "no limit" so the
Full Dialog endpoint can return the entire conversation. Three
sentinels mean "everything":

- ``limit`` key absent
- ``limit=None``
- ``limit=0`` or ``limit<0``

Positive integer limits keep working (legacy contract for dwcli).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _make_agent_process_with_messages(messages: list[Any]):
    """Build a minimal AgentProcess stub whose agent.aget_state returns the
    given message list. No RPyC, no real LangGraph compile.
    """
    from deep_worker_cli.process.agent_process import AgentProcess

    proc = AgentProcess.__new__(AgentProcess)
    # only the bits _handle_get_history reads
    proc.name = "test-agent"

    fake_state = MagicMock()
    fake_state.values = {"messages": messages}

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=fake_state)
    proc.agent = fake_agent  # type: ignore[assignment]

    return proc


def _msgs(n: int) -> list[Any]:
    out: list[Any] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(HumanMessage(content=f"q{i}"))
        else:
            out.append(AIMessage(content=f"a{i}"))
    return out


class TestNoLimitContract:
    @pytest.mark.asyncio
    async def test_no_limit_key_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"command": "get_history"})
        assert resp["success"] is True
        assert resp["total"] == 20
        assert len(resp["history"]) == 20, (
            "absent limit MUST return every message, no implicit cap"
        )
        assert resp["has_more"] is False

    @pytest.mark.asyncio
    async def test_explicit_none_limit_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": None})
        assert resp["total"] == 20
        assert len(resp["history"]) == 20

    @pytest.mark.asyncio
    async def test_zero_limit_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": 0})
        assert len(resp["history"]) == 20

    @pytest.mark.asyncio
    async def test_negative_limit_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": -1})
        assert len(resp["history"]) == 20

    @pytest.mark.asyncio
    async def test_positive_limit_still_works_legacy(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": 5})
        # legacy: last N
        assert len(resp["history"]) == 5
        # last message is q18 or a19 — we just check tail
        last_content = resp["history"][-1]["content"]
        assert last_content == "a19", f"expected last message 'a19', got {last_content!r}"
        assert resp["has_more"] is True
