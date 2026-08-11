"""Integration test (NO MOCKS of the A2A stack): streaming message/stream over
real SSE. A real uvicorn A2A server is started; the real a2a-sdk client
subscribes in streaming mode and receives the artifact chunks the bridge emits
from the turn's ``notify_callback`` events.

Validates stage 2 (inbound streaming).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from deep_worker.providers.deepworker_a2a.config import A2AConfig
from deep_worker.providers.deepworker_a2a.server import start_a2a_server


class _StreamingAgentProcess:
    """Stand-in AgentProcess that streams several chunks then completes."""

    def __init__(self):
        self.name = "streamAgent"

    async def _handle_invoke(self, command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        # Emit a few streamed assistant chunks + a tool event, like a real turn.
        if notify_callback is not None:
            await notify_callback({"type": "assistant", "role": "assistant",
                                   "content": "Hello "})
            await notify_callback({"type": "tool_call", "name": "search",
                                   "args": {}, "id": "t1"})
            await notify_callback({"type": "assistant", "role": "assistant",
                                   "content": "world"})
        return {"success": True,
                "response": {"role": "assistant", "content": "Hello world"}}

    def cancel_current_worker_task(self) -> bool:
        return False


async def _wait_started(handle):
    for _ in range(250):
        if getattr(handle.server, "started", False):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_message_stream_emits_artifact_chunks_over_sse():
    import httpx

    from a2a.client import create_client, ClientConfig
    from a2a.types import Message, Part, Role, SendMessageRequest

    cfg = A2AConfig(enabled=True, host="127.0.0.1", port=0)
    handle = await start_a2a_server(
        _StreamingAgentProcess(), cfg, agent_name="streamAgent"
    )
    try:
        assert await _wait_started(handle), "server did not start"

        async with httpx.AsyncClient(timeout=15) as httpx_client:
            client = await create_client(
                handle.url,
                ClientConfig(httpx_client=httpx_client, streaming=True),
            )
            msg = Message(
                message_id=uuid4().hex,
                role=Role.ROLE_USER,
                parts=[Part(text="hi")],
            )
            req = SendMessageRequest(message=msg)

            events = []
            async for event in client.send_message(req):
                events.append(event)

            blob = repr(events)
            # The streamed chunks must have surfaced as artifact updates.
            assert "Hello" in blob, blob
            assert "world" in blob, blob
            # The tool event is folded into a [tool:...] artifact part.
            assert "[tool:search]" in blob, blob
            # More than one event => it actually streamed (not a single reply).
            assert len(events) >= 2, f"expected a stream, got {len(events)} events"
    finally:
        await handle.shutdown()
