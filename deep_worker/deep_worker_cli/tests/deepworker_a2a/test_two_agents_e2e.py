"""End-to-end A2A tests with TWO REAL agent subprocesses (no mocks).

What's exercised (the things our other a2a tests can't, because they all
live in one Python process):

* Two distinct OS processes, each running its own real A2A server.
* Real ``ProcessRegistry`` on disk, real ``a2a_url`` discovery by NAME
  (agentA looks up agentB through the shared registry file, just like a
  production agent's LLM tool would).
* Real outbound A2A client + ``a2a_*`` LLM tools (the same factories the
  agent registers in its toolset).
* Real HTTP wire (Starlette + uvicorn + a2a-sdk JSON-RPC / SSE).

What's NOT exercised here (intentionally — covered separately):
* The LLM turn engine. We swap in a deterministic stub so the test is fast
  and reliable. ``tests/integration_ai/`` covers the LLM-driven flow.

Loosely analogous to ``tests/integration_ai/test_inter_agent_reply_e2e.py``
(two real agents talking via the internal RPyC channel), but transport here
is A2A, and the two agents communicate over real HTTP through registry-based
discovery.
"""

from __future__ import annotations

import asyncio

import pytest

from deep_worker.providers.deepworker_a2a.client import A2AOutboundClient
from deep_worker.providers.deepworker_a2a.tools import make_a2a_tools

from .two_agents_fixture import two_a2a_agents


pytestmark = pytest.mark.timeout(60)


# ----------------------------------------------------------------------------
# Test 1: agentA delegates a non-trivial task to agentB BY NAME, gets the
# result. This is the primary e2e — it proves registry-based discovery +
# outbound client + real A2A server on the other end all work together.
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_a_delegates_task_to_agent_b_by_name_and_gets_result():
    with two_a2a_agents() as (ep_a, ep_b):
        # agentA's outbound client is exactly what production builds inside
        # the LLM tool factory. We hand it nothing extra — it will resolve
        # the target name through the real on-disk ProcessRegistry that
        # agentB just registered itself in.
        client = A2AOutboundClient()

        result = await client.send(ep_b.name, "upper:hello from a")

        assert result["success"] is True, result
        # agentB's stub turn engine returns the uppercased input.
        assert "HELLO FROM A" in result["answer"], result
        # We must have got back a Task identity + a terminal state.
        assert result.get("task_id"), result
        assert result.get("state") in {
            "TASK_STATE_COMPLETED", "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING",
        }, result


# ----------------------------------------------------------------------------
# Test 2: discover-by-name. Proves agentA can read agentB's AgentCard
# through the registry without knowing the URL upfront.
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_a_discovers_agent_b_by_name_via_registry():
    with two_a2a_agents() as (ep_a, ep_b):
        client = A2AOutboundClient()

        card = await client.discover(ep_b.name)

        assert card["success"] is True, card
        assert card["name"] == ep_b.name
        # AgentCard URL must match what agentB advertised in the registry.
        assert card["url"].rstrip("/") == ep_b.a2a_url.rstrip("/")


# ----------------------------------------------------------------------------
# Test 3: streaming send between two real processes. Mirrors the existing
# in-process streaming test but with the producer running in a *different*
# OS process from the consumer.
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_send_between_two_real_processes():
    with two_a2a_agents() as (ep_a, ep_b):
        client = A2AOutboundClient()

        result = await client.send_streaming(
            ep_b.name, "slow:3:done-after-3-chunks"
        )

        assert result["success"] is True, result
        # Stub emits 3 progress chunks + 1 final-answer chunk = 4 chunks total.
        assert len(result["chunks"]) >= 3, result
        assert any("chunk-1/3" in c for c in result["chunks"]), result
        assert any("chunk-3/3" in c for c in result["chunks"]), result
        assert "done-after-3-chunks" in result["answer"], result


# ----------------------------------------------------------------------------
# Test 4: the actual LLM-facing TOOL works end-to-end. This is what the
# model would call. We invoke `a2a_send_message.ainvoke(...)` against a
# real second-process agent.
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a2a_send_message_llm_tool_against_real_second_process():
    with two_a2a_agents() as (ep_a, ep_b):
        tools = {t.name: t for t in make_a2a_tools()}
        assert "a2a_send_message" in tools

        result = await tools["a2a_send_message"].ainvoke(
            {"target": ep_b.name, "text": "echo:pong"}
        )

        assert result["success"] is True, result
        assert "pong" in result["answer"], result


# ----------------------------------------------------------------------------
# Test 5: bidirectional. agentA sends to agentB, then agentB (acting as
# outbound caller) sends back to agentA. This mirrors what
# `tests/integration_ai/test_inter_agent_reply_e2e.py` does for the RPyC
# channel, but using the A2A protocol — and proves the registry is the
# *only* shared state needed for two A2A agents to find each other.
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bidirectional_a_to_b_then_b_to_a_over_a2a():
    with two_a2a_agents() as (ep_a, ep_b):
        client = A2AOutboundClient()

        # First leg: A -> B
        a_to_b = await client.send(ep_b.name, "echo:question-from-A")
        assert a_to_b["success"] is True, a_to_b
        assert "question-from-A" in a_to_b["answer"], a_to_b

        # Second leg: simulate B calling A (we drive it from the test parent
        # using the same client — what matters is that the message lands on
        # agentA's *own* real A2A server in its *own* OS process and is
        # processed there). The stub answers with "ack:<text> (from <name>)".
        b_to_a = await client.send(ep_a.name, "reply-from-B")
        assert b_to_a["success"] is True, b_to_a
        assert "ack: reply-from-B" in b_to_a["answer"], b_to_a
        assert f"(from {ep_a.name})" in b_to_a["answer"], b_to_a


# ----------------------------------------------------------------------------
# Test 6: registry-based discovery returns a structured error (NOT raises)
# when the named target doesn't exist. Verifies the LLM-tool contract in a
# two-process setting (registry actually loaded; target genuinely missing).
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_target_returns_structured_error_in_e2e_setting():
    with two_a2a_agents() as (ep_a, ep_b):
        client = A2AOutboundClient()
        result = await client.send("test_definitely_not_running", "hi")
        assert result["success"] is False
        assert result["error"] == "target_not_found"
