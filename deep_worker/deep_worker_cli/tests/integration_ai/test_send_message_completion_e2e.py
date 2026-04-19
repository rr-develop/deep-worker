"""End-to-end integration test for SendMessageCompletionMiddleware.

This test exists because a previous iteration shipped without verifying
that the whole `deep-worker agent -asm --dir /tmp --clean`-style flow
actually works in anger. Unit tests caught none of that because they
mocked the model, the tool transport, and the entire agent graph.

What this test does
-------------------
1. Starts two real deep-worker agents ("test_smc_src" and
   "test_smc_dst") with real RPyC servers in-process via the existing
   agent_fixture.
2. Connects to the source agent and asks it — with a simple,
   deterministic DeepSeek-style prompt — to call
   ``send_message_to_agent("test_smc_dst", "<text>")`` exactly once
   and then stop.
3. Verifies the source agent's history ends with an ``AIMessage``
   whose content is exactly ``"agent:test_smc_src:> <text>"``. That
   is the echo injected by ``SendMessageCompletionMiddleware``.
4. Verifies the destination agent's history received the prefixed
   message as a ``HumanMessage``.
5. Replays the exact pair-parser logic from the web UI's
   ``get_agent_dialog_formatted``
   and asserts the outgoing message appears as an assistant response
   in the parsed dialog pairs. This is the end state the user cares
   about.

This test would have caught a regression where the middleware is not
wired into ``create_cli_agent``, or where the echo content is wrong,
or where the pair parser still skips the outgoing turn.

Dependency note: requires ``DEEPSEEK_API_KEY`` (or a cheap model). Marked
as ``@pytest.mark.integration_ai`` / collected by integration_ai suite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.client.agent_client import AgentClient

from . import agent_fixture


# ---------------------------------------------------------------------------
# Helpers copied verbatim from the web UI for fidelity
# ---------------------------------------------------------------------------


def _dialog_pairs_from_history(history: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Re-implement the web UI's pair parser exactly.

    Keep this logic identical to the web UI's
    ``get_agent_dialog_formatted``
    — the whole point of the middleware is to produce messages the
    same parser renders correctly.
    """
    dialog_pairs: list[tuple[str, str]] = []
    current_user_query: str | None = None
    for message in history:
        msg_type = message.get("type")
        content = message.get("content", "")
        if msg_type == "HumanMessage":
            current_user_query = content
        elif msg_type == "AIMessage" and current_user_query is not None:
            if content and content.strip():
                dialog_pairs.append((current_user_query, content))
                current_user_query = None
    return dialog_pairs


# ---------------------------------------------------------------------------
# Fixtures: two agents, source and destination
# ---------------------------------------------------------------------------


_SRC_NAME = "test_smc_src"
_DST_NAME = "test_smc_dst"


@pytest.fixture
def two_agents():
    """Start a pair of agents (source + destination) both with
    skip_permissions so the tool runs without HITL prompts. We use
    ``skip_permissions=True`` rather than ``-asm`` because the fixture
    helper only exposes the former — the tool-level behaviour of
    auto-approving send_message_to_agent is identical either way."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration_ai test")

    with agent_fixture(
        agent_name=_SRC_NAME,
        working_dir=Path("/tmp"),
        model="deepseek-chat",
        provider="deepseek",
        skip_permissions=True,
        api_key=api_key,
    ) as src, agent_fixture(
        agent_name=_DST_NAME,
        working_dir=Path("/tmp"),
        model="deepseek-chat",
        provider="deepseek",
        skip_permissions=True,
        api_key=api_key,
    ) as dst:
        yield src, dst


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------


_MESSAGE_TEXT = "integration-test-message-42"


@pytest.mark.asyncio
async def test_send_message_to_agent_echoes_and_ends_turn(two_agents):
    """End-to-end: source agent sends to destination; both histories
    must reflect the exchange and the web-style pair parser must show
    the outgoing message as a non-empty assistant response on the
    source side."""
    src_info, dst_info = two_agents

    # --- Step 1: ask source agent to call send_message_to_agent ---
    prompt = (
        f"Use the send_message_to_agent() tool exactly once "
        f"to send the agent '{_DST_NAME}' exactly this text (without "
        f"quotes): {_MESSAGE_TEXT}\n\n"
        f"Do not comment or write anything else. Just call the tool."
    )

    src_client = AgentClient(src_info.agent_name)
    await src_client.connect()
    try:
        response = await src_client.invoke_agent(prompt, stream=False)
    finally:
        await src_client.disconnect()

    assert response.get("success"), f"invoke_agent failed: {response}"
    final_content = response.get("response", {}).get("content", "")

    # --- Step 2: final response should be exactly the echo ---
    expected_echo = f"agent:{src_info.agent_name}:> {_MESSAGE_TEXT}"
    assert final_content == expected_echo, (
        f"Final response from source agent must be the echo produced by "
        f"SendMessageCompletionMiddleware.\n"
        f"  expected: {expected_echo!r}\n"
        f"  got:      {final_content!r}"
    )

    # --- Step 3: inspect source agent history ---
    src_client = AgentClient(src_info.agent_name)
    await src_client.connect()
    try:
        hist_resp = await src_client._send_command(
            {"command": "get_history", "limit": 100, "offset": 0}
        )
    finally:
        await src_client.disconnect()

    assert hist_resp.get("success"), f"get_history failed: {hist_resp}"
    history = hist_resp.get("history", [])
    assert history, "source agent history must not be empty"

    # The last message MUST be the echo AIMessage (text, no tool_calls).
    last = history[-1]
    assert last.get("type") == "AIMessage", (
        f"Last message must be an AIMessage (the middleware echo); "
        f"got type={last.get('type')!r}"
    )
    assert last.get("content") == expected_echo, (
        f"Last AIMessage content must be the echo.\n"
        f"  expected: {expected_echo!r}\n"
        f"  got:      {last.get('content')!r}"
    )
    # Echo must be text-only, no tool_calls.
    assert not last.get("tool_calls"), (
        f"Echo AIMessage must NOT carry tool_calls; got {last.get('tool_calls')!r}"
    )

    # Structural check: right before the echo there must be the
    # ToolMessage from send_message_to_agent with success=True. That's
    # the trigger the middleware watched for.
    tool_msgs = [
        m for m in history
        if m.get("type") == "ToolMessage"
        and m.get("name") == "send_message_to_agent"
    ]
    assert tool_msgs, "source history must contain a send_message_to_agent ToolMessage"
    # The tool result body is a stringified dict; it must contain success and the text.
    latest_tool_content = str(tool_msgs[-1].get("content", ""))
    assert "success" in latest_tool_content and "true" in latest_tool_content.lower(), (
        f"send_message_to_agent ToolMessage must report success=true; got: "
        f"{latest_tool_content[:300]!r}"
    )
    assert _MESSAGE_TEXT in latest_tool_content, (
        f"ToolMessage content must contain the delivered text; got: "
        f"{latest_tool_content[:300]!r}"
    )

    # --- Step 4: dialog pair parser (web UI) renders it ---
    pairs = _dialog_pairs_from_history(history)
    assert pairs, (
        "pair-parser must find at least one (user, assistant) pair; "
        "otherwise Full Dialog will be empty"
    )
    # The last pair's assistant side must be our echo.
    last_user, last_assistant = pairs[-1]
    assert last_assistant == expected_echo, (
        f"the web UI's pair parser must render the outgoing message "
        f"as the assistant reply.\n"
        f"  expected assistant: {expected_echo!r}\n"
        f"  got assistant:      {last_assistant!r}\n"
        f"  (user side:         {last_user!r})"
    )

    # --- Step 5: destination agent received the message ---
    dst_client = AgentClient(dst_info.agent_name)
    await dst_client.connect()
    try:
        dst_hist = await dst_client._send_command(
            {"command": "get_history", "limit": 100, "offset": 0}
        )
    finally:
        await dst_client.disconnect()

    assert dst_hist.get("success"), f"dst get_history failed: {dst_hist}"
    dst_history = dst_hist.get("history", [])
    # Look for a HumanMessage with the prefixed text.
    incoming = [
        m for m in dst_history
        if m.get("type") == "HumanMessage"
        and m.get("content", "") == expected_echo
    ]
    assert incoming, (
        f"destination agent must have received the prefixed HumanMessage "
        f"{expected_echo!r}; "
        f"got history: {[(m.get('type'), (m.get('content') or '')[:60]) for m in dst_history]}"
    )
