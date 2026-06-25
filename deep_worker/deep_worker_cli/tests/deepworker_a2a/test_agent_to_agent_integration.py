"""Integration test (NO MOCKS): agent-to-agent over A2A.

Two real uvicorn A2A servers are started (agent B "server" + the outbound
client used by agent A). Agent A's ``A2AOutboundClient`` discovers and sends a
message to agent B over real HTTP; B's executor runs its (faked) turn engine
and replies. Validates the outbound path (stage 3) end to end.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from deepworker_a2a.config import A2AConfig
from deepworker_a2a.client import A2AOutboundClient
from deepworker_a2a.server import start_a2a_server


class _EchoAgentProcess:
    def __init__(self, name):
        self.name = name

    async def _handle_invoke(self, command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        text = command.get("message", "")
        if notify_callback is not None:
            await notify_callback({"type": "assistant", "role": "assistant",
                                   "content": f"B-reply: {text}"})
        return {"success": True,
                "response": {"role": "assistant", "content": f"B-reply: {text}"}}

    def cancel_current_worker_task(self) -> bool:
        return False


async def _wait_started(handle):
    for _ in range(250):
        if getattr(handle.server, "started", False):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_outbound_discover_and_send_to_real_agent():
    # Agent B: real A2A server.
    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = await start_a2a_server(_EchoAgentProcess("agentB"), cfg,
                                    agent_name="agentB", version="3.2.1")
    try:
        assert await _wait_started(handle), "agentB server did not start"
        b_url = handle.url

        # Agent A uses the outbound client to talk to B by URL.
        out = A2AOutboundClient()

        # discover -> structured dict
        disco = await out.discover(b_url)
        assert disco["success"] is True
        assert disco["name"] == "agentB"
        assert disco["version"] == "3.2.1"
        assert "url" in disco
        assert isinstance(disco["skills"], list)

        # send -> wait final answer
        res = await out.send(b_url, "ping")
        assert res["success"] is True, res
        assert "B-reply: ping" in res["answer"], res

        # streaming send -> chunks
        sres = await out.send_streaming(b_url, "stream-me")
        assert sres["success"] is True, sres
        assert "B-reply: stream-me" in sres["answer"], sres
    finally:
        await handle.shutdown()


@pytest.mark.asyncio
async def test_outbound_target_not_found_returns_structured_error():
    out = A2AOutboundClient()
    res = await out.send("no-such-agent-name", "hi")
    assert res["success"] is False
    assert res["error"] == "target_not_found"
