"""Integration test (NO MOCKS of the A2A stack): a real Starlette+uvicorn A2A
server is started on localhost, and the REAL a2a-sdk client resolves the
AgentCard over HTTP and sends a message. Only the agent *turn engine* is faked
(a stand-in ``_handle_invoke``) so the test is hermetic — routes, JSON-RPC,
task lifecycle, HTTP transport and the SDK client are all real.

Validates the end-to-end inbound path of stage 1.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from deepworker_a2a.config import A2AConfig
from deepworker_a2a.server import start_a2a_server


class _FakeAgentProcess:
    """Minimal stand-in for AgentProcess: echoes the request text back."""

    def __init__(self):
        self.name = "itAgent"

    async def _handle_invoke(self, command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        text = command.get("message", "")
        return {
            "success": True,
            "response": {"role": "assistant", "content": f"echo: {text}"},
        }

    def cancel_current_worker_task(self) -> bool:
        return False


async def _wait_started(handle):
    for _ in range(250):
        if getattr(handle.server, "started", False):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_agent_card_discovery_over_http():
    import httpx

    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = await start_a2a_server(
        _FakeAgentProcess(), cfg, agent_name="itAgent", version="9.9.9"
    )
    try:
        assert await _wait_started(handle), "server did not start"
        base = handle.url.rstrip("/")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base}/.well-known/agent-card.json")
            assert resp.status_code == 200
            card = resp.json()
            assert card["name"] == "itAgent"
            assert card["version"] == "9.9.9"
    finally:
        await handle.shutdown()


@pytest.mark.asyncio
async def test_message_send_end_to_end_with_real_sdk_client():
    import httpx

    from a2a.client import create_client, ClientConfig
    from a2a.types import Message, Part, Role, SendMessageRequest

    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = await start_a2a_server(
        _FakeAgentProcess(), cfg, agent_name="itAgent", version="9.9.9"
    )
    try:
        assert await _wait_started(handle), "server did not start"

        async with httpx.AsyncClient(timeout=10) as httpx_client:
            client = await create_client(
                handle.url,
                ClientConfig(httpx_client=httpx_client, streaming=False),
            )
            msg = Message(
                message_id=uuid4().hex,
                role=Role.ROLE_USER,
                parts=[Part(text="hello")],
            )
            req = SendMessageRequest(message=msg)

            collected: list = []
            async for event in client.send_message(req):
                collected.append(event)

            # The fake turn echoes the input; assert it surfaced in the result.
            blob = repr(collected)
            assert "echo: hello" in blob, blob
    finally:
        await handle.shutdown()
