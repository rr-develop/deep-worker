"""Regression: a tool_call must survive a prompt that resolves in the same tick.

Follow-up slice from the hot/cold summarization review
("_finalize_response tool_call drop" — a separate issue, not to be folded in).

``_consume_turn`` waits on ``{get_task, response_fut}`` with
FIRST_COMPLETED. When the CLI streams a ``tool_call`` update and resolves
``session/prompt`` in the SAME event loop iteration, both futures are done
and the ``response_fut`` branch wins. That branch drains the queue into the
LOCAL ``pending_tool_calls`` list and then calls ``_finalize_response``,
which builds ``AIMessage(content=...)`` with no ``tool_calls`` — so the
mapped call is silently dropped and the tool never runs.

This is NOT about summarization: there the ACP session is reset right
after, so unfinished calls are discarded on purpose. This is the ordinary
agent turn, where losing the call means the user's request is never served.

The fake CLI (`tool-then-end` scenario) writes the tool_call notification
and the prompt reply back-to-back without blocking on tools/call, which is
exactly the same-tick condition.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deep_worker.providers.acp.acp_client import AcpClient, reset_global_acp_clients
from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.profile import AcpProfile
from deep_worker.providers.acp.proxy_mcp_server import ProxyMCPServer

FAKE_CLI = Path(__file__).parent / "fake_acp_cli.py"


@tool
def acp_probe_tool(question: str) -> str:
    """Probe tool for the same-tick tool_call regression test."""
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


class TestToolCallSameTickAsEndTurn:
    def test_tool_call_survives_prompt_resolved_in_same_tick(self) -> None:
        env = dict(
            os.environ,
            FAKE_ACP_SCENARIO="tool-then-end",
            FAKE_ACP_TOOL_NAME="acp_probe_tool",
            FAKE_ACP_TOOL_ARGS=json.dumps({"question": "probe"}),
        )

        async def _run() -> None:
            reset_global_acp_clients()
            model = AcpChatModel(profile=_profile(), model="fake")
            model._proxy_mcp = ProxyMCPServer([], long_running_wait_seconds=3.0)
            model._acp_client = AcpClient(
                model._client_state,
                cmd=(sys.executable, str(FAKE_CLI)),
                env=env,
                requires_authenticate=False,
            )
            model = model.bind_tools([acp_probe_tool])
            thread_id = f"same-tick-{uuid.uuid4().hex}"
            config = {"configurable": {"thread_id": thread_id}}

            try:
                ai = await asyncio.wait_for(
                    model.ainvoke(
                        [HumanMessage(content="use the probe tool")], config
                    ),
                    timeout=30.0,
                )
                # THE REGRESSION: the CLI asked for a tool. Dropping it here
                # means ToolNode never runs and the request is unserved.
                assert ai.tool_calls, (
                    "tool_call emitted in the same tick as the prompt reply "
                    f"was dropped; got {ai!r}"
                )
                assert ai.tool_calls[0]["name"] == "acp_probe_tool"
                assert ai.tool_calls[0]["args"] == {"question": "probe"}
            finally:
                await model._acp_client.aclose()
                await model._proxy_mcp.stop()

        asyncio.run(_run())
