"""Regression: a HumanMessage after a turn-ending tool_call must NOT be dropped.

rg1_worker 2026-08-22 reported the defect and produced 2 live captures on
his own process (10:23:24Z, 11:16:41Z) plus the repo_graph extractor.

Mechanism (chat_model.py):
  Turn N ends with a turn-ending tool_call (send_message_to_agent,
  submit_relations, any turn-ender). The CLI keeps ``session/prompt``
  UNRESOLVED because it is waiting for the tool result. AcpChatModel's
  mid-turn pause (~1284) returns AIMessage(tool_calls=...) and leaves
  ``state.pending_session_prompt_future`` alive.

  Turn N+1 arrives with tail ``[ToolMessage, HumanMessage("new question")]``:
    - step 7 sees the ToolMessage → ``resolve_call`` (proxy no-op for
      client-side tools) → ``tool_resolved=True``;
    - step 8 persists ``sent_count = len(state_messages)`` — the new
      HumanMessage is now marked "sent" on disk;
    - step 9 (line 696) sees ``tool_resolved and
      pending_session_prompt_future is not None`` → mid-turn resume →
      goes into ``_consume_turn`` on the OLD future. ``user_text_parts``
      (the fresh HumanMessage) is silently dropped: it is only used in
      the new-turn path further down.

Fix contract (step 9): when the tail carries a HumanMessage AND we would
take the mid-turn resume path, the resume must NOT swallow the new user
turn. The CLI must see a second ``session/prompt`` with the new prompt
text before this ainvoke returns.

Live signal on the recipient side: has_tool_calls=False, instant reply
that echoes the previous turn's tail — because ``_consume_turn`` finishes
on turn-1's response without ever sending turn-2's payload.

The fake CLI's ``turn-ending-open`` scenario emits a proxy-MCP tool_call
on turn 1 and leaves the prompt UNRESOLVED, then answers whatever turn-2
prompt the client sends with a fresh agent text + end_turn. Every prompt
payload is appended to FAKE_ACP_TURN_ENDING_PROMPTS so the assertions do
not depend on internal state.
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

from deep_worker.providers.acp.acp_client import AcpClient, reset_global_acp_clients
from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.profile import AcpProfile
from deep_worker.providers.acp.proxy_mcp_server import ProxyMCPServer

FAKE_CLI = Path(__file__).parent / "fake_acp_cli.py"


@tool
def send_message_to_agent(agent_name: str, message: str) -> str:
    """Stand-in for the real inter-agent messaging tool."""
    return json.dumps({"success": True, "target_agent": agent_name})


def _profile() -> AcpProfile:
    return AcpProfile(
        provider="claude-code",
        cmd=(sys.executable, str(FAKE_CLI)),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
        requires_authenticate=False,
    )


class TestTurnEndingMidTurnSwallow:
    def test_new_human_message_after_turn_ending_tool_call_is_delivered(
        self, tmp_path: Path
    ) -> None:
        prompts_file = tmp_path / "prompts.jsonl"
        env = dict(
            os.environ,
            FAKE_ACP_SCENARIO="turn-ending-open",
            FAKE_ACP_TOOL_NAME="send_message_to_agent",
            FAKE_ACP_TOOL_ARGS=json.dumps(
                {"agent_name": "peer", "message": "delivered"}
            ),
            FAKE_ACP_TURN_ENDING_PROMPTS=str(prompts_file),
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
            model = model.bind_tools([send_message_to_agent])
            thread_id = f"turn-ending-{uuid.uuid4().hex}"
            config = {"configurable": {"thread_id": thread_id}}

            try:
                # Turn 1: user asks something → CLI emits turn-ending
                # tool_call, keeps prompt UNRESOLVED.
                ai1 = await asyncio.wait_for(
                    model.ainvoke(
                        [HumanMessage(content="please send delivery")],
                        config,
                    ),
                    timeout=15.0,
                )
                assert ai1.tool_calls, f"turn 1 must return tool_calls: {ai1!r}"
                tcid = ai1.tool_calls[0]["id"]

                # LangGraph runs the tool locally; peer replies asynchronously
                # → a NEW HumanMessage arrives on top. Second ainvoke carries
                # the full tail: [initial human, AI(tc), ToolMessage, new human].
                turn2_messages = [
                    HumanMessage(content="please send delivery"),
                    ai1,
                    ToolMessage(
                        content='{"success": true, "target_agent": "peer"}',
                        tool_call_id=tcid,
                        name="send_message_to_agent",
                    ),
                    HumanMessage(content="brand new user question"),
                ]
                # Bounded turn 2: under the defect, mid-turn resume awaits
                # a response future that the fake CLI is holding open, and
                # there is no second session/prompt to unblock it — so the
                # coroutine deadlocks. A short timeout makes the RED case
                # explicit as a TimeoutError instead of a 15 s hang.
                try:
                    ai2 = await asyncio.wait_for(
                        model.ainvoke(turn2_messages, config),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError as exc:
                    prompt_lines = (
                        prompts_file.read_text().splitlines()
                        if prompts_file.exists() else []
                    )
                    raise AssertionError(
                        "turn 2 deadlocked: mid-turn resume swallowed the "
                        "new HumanMessage and no second session/prompt was "
                        f"fired; prompts recorded: {prompt_lines!r}"
                    ) from exc

                # Contract (the fix): the CLI must have received a SECOND
                # session/prompt whose payload contains the new HumanMessage.
                prompt_lines = (
                    prompts_file.read_text().splitlines()
                    if prompts_file.exists() else []
                )
                assert len(prompt_lines) >= 2, (
                    "second session/prompt was NEVER sent — the new "
                    f"HumanMessage was swallowed by mid-turn resume; "
                    f"prompts recorded: {prompt_lines!r}"
                )
                second_payload = json.loads(prompt_lines[1])
                second_text = json.dumps(second_payload)
                assert "brand new user question" in second_text, (
                    "second session/prompt did not carry the new user "
                    f"turn; payload was {second_payload!r}"
                )

                # Sanity: the model's response to turn 2 is the CLI's
                # answer to the NEW question, not the tail of turn 1.
                assert "answer to the new q" in (ai2.content or ""), (
                    f"turn 2 response is not from the new session/prompt: {ai2!r}"
                )
            finally:
                await model._acp_client.aclose()
                await model._proxy_mcp.stop()

        asyncio.run(_run())
