"""Tests for the outbound ``a2a_*`` LLM tools.

Tool wiring is unit-checked (names/schemas), and the discover/send tools are
exercised against a REAL A2A server (no mocks) to prove end-to-end routing.
"""

from __future__ import annotations

import asyncio

import pytest

from deep_worker.providers.deepworker_a2a.config import A2AConfig
from deep_worker.providers.deepworker_a2a.server import start_a2a_server
from deep_worker.providers.deepworker_a2a.tools import make_a2a_tools


class _EchoAgentProcess:
    def __init__(self, name="agentB"):
        self.name = name

    async def _handle_invoke(self, command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        text = command.get("message", "")
        return {"success": True,
                "response": {"role": "assistant", "content": f"echo: {text}"}}

    def cancel_current_worker_task(self) -> bool:
        return False


async def _wait_started(handle):
    for _ in range(250):
        if getattr(handle.server, "started", False):
            return True
        await asyncio.sleep(0.02)
    return False


def test_tool_names_and_count():
    tools = make_a2a_tools()
    names = [t.name for t in tools]
    assert names == [
        "a2a_discover",
        "a2a_send_message",
        "a2a_send_message_streaming",
        "a2a_get_task",
        "a2a_cancel_task",
    ]


def test_tools_are_coroutine_tools():
    tools = make_a2a_tools()
    for t in tools:
        # async tools expose a coroutine via ainvoke
        assert hasattr(t, "ainvoke")


@pytest.mark.asyncio
async def test_discover_and_send_tools_against_real_server():
    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = await start_a2a_server(_EchoAgentProcess(), cfg, agent_name="agentB")
    try:
        assert await _wait_started(handle)
        url = handle.url
        tools = {t.name: t for t in make_a2a_tools()}

        disco = await tools["a2a_discover"].ainvoke({"target": url})
        assert disco["success"] is True
        assert disco["name"] == "agentB"
        assert isinstance(disco["skills"], list)

        sent = await tools["a2a_send_message"].ainvoke(
            {"target": url, "text": "ping"}
        )
        assert sent["success"] is True
        assert "echo: ping" in sent["answer"]
    finally:
        await handle.shutdown()


@pytest.mark.asyncio
async def test_send_tool_unknown_target_is_structured_error():
    tools = {t.name: t for t in make_a2a_tools()}
    res = await tools["a2a_send_message"].ainvoke(
        {"target": "definitely-not-an-agent", "text": "x"}
    )
    assert res["success"] is False
    assert res["error"] == "target_not_found"
