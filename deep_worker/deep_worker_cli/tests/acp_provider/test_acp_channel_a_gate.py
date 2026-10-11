"""Signed Channel-A integration gate (review item 2, round 2).

The FULL proxy-MCP tool round-trip against a REAL fake-CLI subprocess:

    fake CLI (session/prompt)
      ──tool_call update (mcp__<server>__<tool>)──▶ AcpChatModel
      ──AIMessage(tool_calls)──▶ LangGraph ToolNode
      ──ToolMessage──▶ ainvoke re-entry
      ──proxy_mcp.resolve_call──▶ MCP handler unblocks
      ──HTTP tools/call reply──▶ fake CLI ends the turn (end_turn).

This is the "signed" gate requested in review: the test FAILS if a second
``session/prompt`` is ever sent over the whole round trip (the fake CLI
writes every prompt to a count file; the assertion requires exactly one
line), and it FAILS if the proxy handler returns an intermediate
(in_progress job-handle) result instead of the real tool output (the
tools/call reply is captured and must contain the real tool text).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph._internal._constants import CONFIG_KEY_RUNTIME
from langgraph.prebuilt import ToolNode
from langgraph.runtime import DEFAULT_RUNTIME

from deep_worker.providers.acp.acp_client import AcpClient, reset_global_acp_clients
from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.profile import AcpProfile
from deep_worker.providers.acp.proxy_mcp_server import ProxyMCPServer

FAKE_CLI = Path(__file__).parent / "fake_acp_cli.py"


@tool
def acp_probe_tool(question: str) -> str:
    """Probe tool for the ACP channel-A signed integration gate."""
    return f"tool-result:{question}"


def _profile() -> AcpProfile:
    return AcpProfile(
        provider="claude-code",
        cmd=(sys.executable, str(FAKE_CLI)),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
        requires_authenticate=False,
    )


class TestChannelASignedGate:
    def test_proxy_tool_roundtrip_is_single_prompt(self, tmp_path: Path) -> None:
        count_file = tmp_path / "prompt_count.txt"
        result_file = tmp_path / "tool_result.json"

        env = dict(
            os.environ,
            FAKE_ACP_SCENARIO="tool-proxy",
            FAKE_ACP_TOOL_NAME="acp_probe_tool",
            FAKE_ACP_TOOL_ARGS=json.dumps({"question": "probe"}),
            FAKE_ACP_COUNT_FILE=str(count_file),
            FAKE_ACP_RESULT_FILE=str(result_file),
        )

        async def _run() -> None:
            reset_global_acp_clients()
            model = AcpChatModel(profile=_profile(), model="fake")
            # Bounded wait kept short so a broken correlation (no resolve_call)
            # makes the handler return its in_progress job-handle quickly
            # instead of hanging the test for the 120s default.
            model._proxy_mcp = ProxyMCPServer(
                [], long_running_wait_seconds=3.0
            )
            model._acp_client = AcpClient(
                model._client_state,
                cmd=(sys.executable, str(FAKE_CLI)),
                env=env,
                requires_authenticate=False,
            )
            model = model.bind_tools([acp_probe_tool])
            # Unique thread_id per run: AcpClientState persists session state
            # to disk and re-loads it for a repeated thread_id, which would
            # route the run through session/load and skip session/new (the
            # fresh fake CLI then never sees the mcpServers block).
            thread_id = f"e2e-gate-{uuid.uuid4().hex}"
            config = {"configurable": {"thread_id": thread_id}}

            try:
                # Turn 1: the model proposes the proxy tool call.
                ai1 = await model.ainvoke(
                    [HumanMessage(content="use the probe tool")], config
                )
                assert ai1.tool_calls, f"expected tool_calls, got {ai1!r}"
                assert ai1.tool_calls[0]["name"] == "acp_probe_tool"
                assert ai1.tool_calls[0]["args"] == {"question": "probe"}

                # The MCP handler must be PARKED (await-pending) — the fake
                # CLI is blocked on its HTTP tools/call, nobody resolved it.
                await asyncio.wait_for(
                    model._proxy_mcp.wait_for_pending_calls(
                        tool_name="acp_probe_tool", timeout=10.0
                    ),
                    timeout=10.0,
                )
                assert model._proxy_mcp.has_pending_calls()

                # LangGraph ToolNode executes the tool locally (in a real
                # graph Pregel injects the runtime into config automatically;
                # outside a graph we pass the default runtime explicitly).
                node = ToolNode([acp_probe_tool])
                out = await node.ainvoke(
                    {"messages": [ai1]},
                    {"configurable": {CONFIG_KEY_RUNTIME: DEFAULT_RUNTIME}},
                )
                tm = out["messages"][-1]
                assert isinstance(tm, ToolMessage), f"got {tm!r}"
                assert tm.content == "tool-result:probe"

                # Turn 2: mid-turn resume — the ToolMessage drives
                # resolve_call, which unblocks the parked MCP handler.
                ai2 = await model.ainvoke(
                    [
                        HumanMessage(content="use the probe tool"),
                        ai1,
                        tm,
                    ],
                    config,
                )
                assert "final" in (ai2.content or ""), f"got {ai2!r}"
                assert not model._proxy_mcp.has_pending_calls()
            finally:
                await model._acp_client.aclose()
                await model._proxy_mcp.stop()

        asyncio.run(_run())

        # THE SIGNED GATE — exactly ONE session/prompt over the whole trip.
        lines = count_file.read_text().strip().splitlines()
        assert lines == ["prompt"], (
            f"expected exactly 1 session/prompt, got {len(lines)}"
        )

        # The proxy handler returned the REAL tool output, not an
        # intermediate in_progress job-handle.
        reply = json.loads(result_file.read_text())
        assert "error" not in reply, f"tools/call error reply: {reply}"
        content = (reply.get("result") or {}).get("content") or []
        text = "".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
        assert text == "tool-result:probe", (
            f"proxy handler returned intermediate/incorrect result: {text!r}"
        )
