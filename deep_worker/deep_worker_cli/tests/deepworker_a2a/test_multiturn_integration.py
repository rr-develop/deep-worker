"""Integration test (NO MOCKS): A2A multi-turn continuity over real HTTP.

A real uvicorn A2A server runs a stateful agent that records each turn's text
keyed by ``thread_id_override``. The outbound client sends two messages sharing
one ``context_id``; the second turn must see the first turn's text (same thread)
— proving RC-4 multi-turn history isolation end to end. A third message with a
different ``context_id`` must NOT see that history.
"""

from __future__ import annotations

import asyncio

import pytest

from deepworker_a2a.config import A2AConfig
from deepworker_a2a.client import A2AOutboundClient
from deepworker_a2a.server import start_a2a_server


class _StatefulAgent:
    def __init__(self, name="agentMT"):
        self.name = name
        self.history: dict[str, list[str]] = {}

    async def _handle_invoke(self, command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        text = command.get("message", "")
        hist = self.history.setdefault(thread_id_override, [])
        prior = list(hist)
        hist.append(text)
        # Echo back how many prior turns this thread had seen.
        answer = f"thread={thread_id_override} prior={len(prior)} text={text}"
        return {"success": True,
                "response": {"role": "assistant", "content": answer}}

    def cancel_current_worker_task(self) -> bool:
        return False


async def _wait_started(handle):
    for _ in range(250):
        if getattr(handle.server, "started", False):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_multiturn_context_continuity_over_http():
    agent = _StatefulAgent()
    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = await start_a2a_server(agent, cfg, agent_name="agentMT")
    try:
        assert await _wait_started(handle)
        url = handle.url
        out = A2AOutboundClient()

        r1 = await out.send(url, "first", context_id="conv-A")
        assert r1["success"], r1
        assert "prior=0" in r1["answer"]

        r2 = await out.send(url, "second", context_id="conv-A")
        assert r2["success"], r2
        # Same context -> same thread -> the prior turn is visible.
        assert "prior=1" in r2["answer"], r2
        assert r1["answer"].split()[0] == r2["answer"].split()[0]  # same thread=

        # Different context -> fresh thread, no shared history.
        r3 = await out.send(url, "elsewhere", context_id="conv-B")
        assert r3["success"], r3
        assert "prior=0" in r3["answer"], r3
        assert r3["answer"].split()[0] != r1["answer"].split()[0]
    finally:
        await handle.shutdown()
