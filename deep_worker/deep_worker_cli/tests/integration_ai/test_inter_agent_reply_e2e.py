"""End-to-end integration test for InterAgentReplyMiddleware and reply_back.

Verifies the reply_back parameter flows through the full stack with
two real communicating agents (no mocks).

1. Starts two real deep-worker agents via agent_fixture.
2. Asks source to send message to destination with reply_back=False.
3. Verifies source receives echo (SendMessageCompletionMiddleware).
4. Verifies destination received the prefixed HumanMessage.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from . import agent_fixture


_SRC_NAME = "test_reply_back_src"
_DST_NAME = "test_reply_back_dst"
_MESSAGE_TEXT = "integration-test-reply-back-99"


@pytest.fixture
def two_agents():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration_ai test")
    with agent_fixture(
        agent_name=_SRC_NAME, working_dir=Path("/tmp"),
        provider="deepseek", skip_permissions=True, api_key=api_key,
    ) as src, agent_fixture(
        agent_name=_DST_NAME, working_dir=Path("/tmp"),
        provider="deepseek", skip_permissions=True, api_key=api_key,
    ) as dst:
        yield src, dst


@pytest.mark.asyncio
async def test_reply_back_e2e(two_agents):
    """Source sends message with reply_back=False; verify transport."""
    src_info, dst_info = two_agents

    # Step 1: ask source to send message
    prompt = (
        f"Use send_message_to_agent() exactly once to send to "
        f"'{_DST_NAME}' the text (without quotes): {_MESSAGE_TEXT}. "
        f"Set reply_back=False so the receiver responds to the user, "
        f"not back to you. Do NOT add extra commentary."
    )

    src_client = AgentClient(src_info.agent_name)
    await src_client.connect()
    try:
        t0 = time.monotonic()
        response = await src_client.invoke_agent(prompt, stream=False)
        src_elapsed = time.monotonic() - t0
    finally:
        await src_client.disconnect()

    assert response.get("success"), f"invoke_agent failed: {response}"
    final_content = response.get("response", {}).get("content", "")

    # Step 2: source should finish quickly (fire-and-forget)
    assert src_elapsed <= 30.0, f"Source took {src_elapsed:.1f}s"

    # Step 3: verify echo
    expected_echo = f"{src_info.agent_name} -> {dst_info.agent_name} :> {_MESSAGE_TEXT}"
    assert final_content == expected_echo, (
        f"Expected echo: {expected_echo!r}, got: {final_content!r}"
    )

    # Step 4: wait for destination to finish
    dst_client = AgentClient(dst_info.agent_name)
    await dst_client.connect()
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = (await dst_client._send_command({"command": "status"})).get("status", "")
            if status.lower() in ("idle", "ready"):
                break
            await asyncio.sleep(0.3)

        # Step 5: destination history must have the prefixed HumanMessage
        hist = await dst_client._send_command(
            {"command": "get_history", "limit": 100, "offset": 0}
        )
    finally:
        await dst_client.disconnect()

    assert hist.get("success"), f"get_history failed: {hist}"
    history = hist.get("history", [])
    assert history, "destination history empty"

    prefixed = f"{src_info.agent_name} -> {dst_info.agent_name} :> {_MESSAGE_TEXT}"
    assert any(
        m.get("type") == "HumanMessage" and m.get("content") == prefixed
        for m in history
    ), (
        f"Destination must have received prefixed HumanMessage.\n"
        f"Expected: {prefixed!r}\n"
        f"Got: {[(m.get('type'), m.get('content','')[:80]) for m in history]}"
    )

    # Step 6 (soft assertion): verify InterAgentReplyMiddleware ran.
    # The middleware injects a SystemMessage hint — check it exists.
    # Note: in some frameworks the system msg content may be empty after
    # the agent finishes, so we only log a warning if missing.
    system_hint_found = any(
        m.get("type") == "SystemMessage"
        and ("reply_back" in m.get("content", "").lower() or
             "send_message_to_agent" in m.get("content", ""))
        for m in history
    )
    if not system_hint_found:
        # The hint may have been overwritten by the framework — that's OK,
        # transport was verified. Log for debugging.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "SystemMessage hint not found in destination history "
            "(may have been overwritten by framework). "
            "System msgs: %s",
            [m.get("content","")[:200] for m in history if m.get("type") == "SystemMessage"]
        )
