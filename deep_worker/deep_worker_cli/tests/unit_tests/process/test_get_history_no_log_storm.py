"""Regression: ``_handle_get_history`` must not repr-log the full LangGraph
state / message history.

Incident (observed 2026-05-23, recurrence of the 2026-05-08
get_history repr-log-storm): the handler did

    logger.info(f"[GET_HISTORY] Successfully got state: {state}")
    logger.info(f"[GET_HISTORY] Agent is initialized: {self.agent}")

``get_history`` is polled at high frequency by web/CLI clients. Each call
``repr``'d the ENTIRE StateSnapshot — every HumanMessage / AIMessage /
ToolMessage with full content (file dumps, tool output) — at INFO level
(always on). The log grew to 5.2 GB at ~46 KB/s, and the per-call repr also
spikes RssAnon (the 2026-05-08 peak). The agent's own
``self.agent`` (a Pregel graph) repr is likewise unbounded.

Contract under test:
  * a single get_history call must NOT emit any log record whose message
    contains the conversation content (the message bodies);
  * it must NOT repr the full StateSnapshot or the Pregel ``agent`` object;
  * the functional result is unchanged (history still returned correctly).

A bounded breadcrumb (e.g. message COUNT) is fine — we only forbid dumping
the unbounded content.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _make_agent_process_with_messages(messages: list[Any]):
    from deep_worker_cli.process.agent_process import AgentProcess
    from deep_worker_cli.process.message_dict_cache import MessageDictCache

    proc = AgentProcess.__new__(AgentProcess)
    proc.name = "test-agent"

    fake_state = MagicMock()
    fake_state.values = {"messages": messages}
    # Make the StateSnapshot repr LOUD so the test fails hard if it is logged.
    fake_state.__repr__ = lambda self=fake_state: "STATE_REPR_" + "X" * 5000  # type: ignore[assignment]

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=fake_state)
    fake_agent.__repr__ = lambda self=fake_agent: "AGENT_REPR_" + "Y" * 5000  # type: ignore[assignment]
    proc.agent = fake_agent  # type: ignore[assignment]
    proc._message_dict_cache = MessageDictCache()
    return proc


def _msgs(n: int) -> list[Any]:
    out: list[Any] = []
    for i in range(n):
        # Unique, distinctive content so we can detect it leaking into logs.
        marker = f"SECRET_MSG_CONTENT_{i}_" + "Z" * 200
        if i % 2 == 0:
            out.append(HumanMessage(content=marker, id=f"h-{i}"))
        else:
            out.append(AIMessage(content=marker, id=f"a-{i}"))
    return out


@pytest.mark.asyncio
async def test_get_history_does_not_log_state_or_content(caplog) -> None:
    proc = _make_agent_process_with_messages(_msgs(10))

    with caplog.at_level(logging.DEBUG):  # capture EVERYTHING, even DEBUG
        resp = await proc._handle_get_history({"command": "get_history"})

    # Functional result unchanged.
    assert resp["success"] is True
    assert resp["total"] == 10
    assert len(resp["history"]) == 10

    all_log_text = "\n".join(r.getMessage() for r in caplog.records)

    # 1. Message CONTENT must never reach the log.
    assert "SECRET_MSG_CONTENT_" not in all_log_text, (
        "get_history logged conversation content — this is the 5.2 GB "
        "log-storm (observed 2026-05-23). Never repr the message bodies."
    )
    # 2. The full StateSnapshot repr must never be logged.
    assert "STATE_REPR_" not in all_log_text, (
        "get_history repr'd the full StateSnapshot — the exact incident "
        "from 2026-05-08 (a related internal incident analysis, not included here)."
    )
    # 3. The Pregel agent object repr must never be logged.
    assert "AGENT_REPR_" not in all_log_text, (
        "get_history repr'd the Pregel agent object — also unbounded."
    )


@pytest.mark.asyncio
async def test_get_history_log_volume_is_bounded(caplog) -> None:
    """A get_history call over a LARGE history must not emit a log whose size
    scales with the history (the storm). Total logged bytes for one call must
    stay small and roughly constant regardless of history size."""
    small = _make_agent_process_with_messages(_msgs(2))
    large = _make_agent_process_with_messages(_msgs(200))

    with caplog.at_level(logging.DEBUG):
        await small._handle_get_history({"command": "get_history"})
    small_bytes = sum(len(r.getMessage()) for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await large._handle_get_history({"command": "get_history"})
    large_bytes = sum(len(r.getMessage()) for r in caplog.records)

    # 100x more messages must NOT produce materially more log output.
    assert large_bytes <= small_bytes + 500, (
        f"get_history log volume scales with history size "
        f"(small={small_bytes}B, large={large_bytes}B) — that is the storm. "
        f"Per-call logging must be bounded."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
