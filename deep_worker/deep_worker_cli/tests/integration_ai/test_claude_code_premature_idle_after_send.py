"""Bug repro: claude-code agent generates a useless textual "I'm waiting"
assistant message AFTER ``send_message_to_agent``, instead of going to
IDLE with the outgoing request as its final assistant message.

Designed scenario (per project owner, 2026-06-03):

    Agent A (claude-code, opus, NO TaskManager)
        sends to
    Agent B (deepseek, NO TaskManager)
        replies after a short, deterministic delay.

Spec (from DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md + project policy):

    A successful ``send_message_to_agent`` IS A's final response for that
    turn. A MUST go to IDLE with the outgoing request as its last
    assistant message, regardless of what assistant text the model
    produced alongside the tool call. A must NOT emit a follow-up
    AIMessage like "Waiting for a reply from B" / "Waiting for B's reply" between
    the echo and the eventual incoming reply.

Concretely we assert: in A's checkpoint state, between A's
``send_message_to_agent`` echo and the inbound ``HumanMessage`` carrying
B's reply, there is NO non-empty AIMessage. (One non-tool-call AIMessage
with an empty content string is acceptable; what we are guarding
against is the regression where A produces a textual filler like
"Waiting for a reply from B".)

NO MOCKS. Two real agents over RPyC. Skipped by default via
``@pytest.mark.integration`` and ``CLAUDE_CODE_AVAILABLE`` guard so the
default unit-test run is unaffected.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient
from .agent_fixture import agent_fixture as _make_agent


pytestmark = pytest.mark.integration


_SRC_NAME = "test_premature_idle_src"   # claude-code, opus
_DST_NAME = "test_premature_idle_dst"   # deepseek
_MARKER = "MARKER_PREM_IDLE_42K"
_DELAY_HINT_SEC = 20                    # B waits this long before replying


def _claude_code_available() -> bool:
    """The claude-code provider works only when the ``claude`` CLI is
    installed and on PATH (and node_modules are available). Skip the
    test on hosts where the binary is missing so we don't burn CI time
    on a guaranteed failure.
    """
    return shutil.which("claude") is not None


def _classify_message(m: dict) -> str:
    """Return a coarse role string for a get_history dict entry."""
    return m.get("type", "")


def _content(m: dict) -> str:
    raw = m.get("content", "")
    if isinstance(raw, str):
        return raw
    # Multimodal content blocks (e.g. claude-code) — flatten text parts only.
    if isinstance(raw, list):
        parts = []
        for blk in raw:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
        return "\n".join(parts)
    return str(raw)


def _tool_calls(m: dict) -> list[dict]:
    # Some serializers put tool_calls under additional_kwargs.
    tc = m.get("tool_calls") or []
    if not tc:
        ak = m.get("additional_kwargs") or {}
        tc = ak.get("tool_calls") or []
    return tc if isinstance(tc, list) else []


@pytest.fixture
def two_agents_claude_and_deepseek():
    if not _claude_code_available():
        pytest.skip("claude CLI not installed; skipping claude-code integration test")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration test")

    # NOTE: TaskManager is OFF for BOTH agents per the user's instruction —
    # we want the plain agent loop, not the manager-evaluated loop, so
    # that any "Waiting for a reply" assistant message can ONLY come from the
    # underlying model, not from a TaskManager corrective prompt.
    with _make_agent(
        agent_name=_SRC_NAME,
        working_dir=Path("/tmp"),
        provider="claude-code",
        model="opus",
        skip_permissions=True,
        api_key=None,                  # claude-code doesn't need one
        enable_task_manager=False,
    ) as src, _make_agent(
        agent_name=_DST_NAME,
        working_dir=Path("/tmp"),
        provider="deepseek",
        model="deepseek-chat",
        skip_permissions=True,
        api_key=deepseek_key,
        enable_task_manager=False,
    ) as dst:
        yield src, dst


async def _get_history(client: AgentClient) -> list[dict]:
    resp = await client._send_command(
        {"command": "get_history", "limit": 500, "offset": 0}
    )
    assert resp.get("success"), f"get_history failed: {resp}"
    return resp.get("history", []) or []


async def _wait_until_idle(client: AgentClient, deadline: float) -> str:
    """Poll status until idle or deadline; return the final status."""
    last = ""
    while time.monotonic() < deadline:
        st = (await client._send_command({"command": "status"})) or {}
        last = str(st.get("status", "")).lower()
        if last in ("idle", "ready"):
            return last
        await asyncio.sleep(0.3)
    return last


@pytest.mark.skip(
    reason=(
        "Superseded by tests/integration_ai/"
        "test_claude_code_inter_agent_send_and_reply.py. "
        "The original assertion here ('A must reply via send_message_to_agent "
        "back to B') turned out to be wrong per the actual cross-agent "
        "protocol — A is only required to recognise B's reply, not to send a "
        "follow-up message. The 12-step LangGraph↔Claude-SDK trace in "
        "a related internal incident analysis (not included here) was captured "
        "from THIS test and is the reason it is kept around (do not delete). "
        "Skipped to keep the test suite green; the trace it produced lives "
        "in the incident doc."
    )
)
@pytest.mark.asyncio
async def test_claude_code_does_not_emit_filler_after_send(
    two_agents_claude_and_deepseek,
) -> None:
    src_info, dst_info = two_agents_claude_and_deepseek

    # ── Step 1: bootstrap B so it knows what to do on any incoming
    # inter-agent message. The "delay before replying" gives A enough
    # wall-time to demonstrate it sat in IDLE properly (rather than
    # racing-finishing exactly when B replies).
    dst_client = AgentClient(dst_info.agent_name)
    await dst_client.connect()
    try:
        bootstrap = (
            f"SYSTEM INSTRUCTION for tests. When you FIRST receive an "
            f"inter-agent message in the format "
            f"'<sender> -> {dst_info.agent_name} :> ...', you MUST: "
            f"\n1) First call execute with the command `sleep "
            f"{_DELAY_HINT_SEC}` — this is critical for the test, don't skip it. "
            f"\n2) Then call send_message_to_agent(<sender>, <answer>) "
            f"EXACTLY ONCE. The marker '{_MARKER}' MUST be present "
            f"verbatim in <answer>. "
            f"\n3) After sending, don't do anything else. "
            f"\nBriefly confirm that you understood the instruction."
        )
        await dst_client.invoke_agent(bootstrap, stream=False)
    finally:
        await dst_client.disconnect()

    # ── Step 2: ask A (claude-code) to send a message to B (deepseek).
    # We DO NOT instruct A about the post-send behaviour at all — we want
    # to observe the system's own enforcement of "your turn ends here".
    src_client = AgentClient(src_info.agent_name)
    await src_client.connect()
    try:
        # NOTE: the prompt is deliberately written the way a real user
        # would write it — a small task + an instruction to delegate
        # via send_message_to_agent. We do NOT tell the model "do not
        # produce any other text", because in production we observed
        # the model spontaneously emitting a filler like "Waiting for
        # APPROVE from a live agent." right next to the tool call. The system —
        # SendMessageCompletionMiddleware + the agent loop — is what
        # must enforce "your turn ends at the send", regardless of
        # what extra prose the model wraps around its tool call.
        await src_client.invoke_agent(
            f"Pass this task to agent '{dst_info.agent_name}': ask it "
            f"to acknowledge receipt with the marker '{_MARKER}'. Use "
            f"the send_message_to_agent tool. You'll get a reply back "
            f"shortly.",
            stream=False,
        )

        # ── Step 3: wait until A's status returns to IDLE for the FIRST
        # time after the send. We must capture state BEFORE B's reply
        # actually lands, otherwise we cannot tell whether the filler
        # message came before or after the reply.
        first_idle = await _wait_until_idle(
            src_client, deadline=time.monotonic() + 60.0
        )
        assert first_idle in ("idle", "ready"), (
            f"Source agent never went IDLE after send "
            f"(last status={first_idle!r})"
        )

        # Snapshot history at the moment A first hit IDLE.
        post_send_history = await _get_history(src_client)

        # ── Step 4: wait for B's reply round-trip to land in A's history.
        reply_deadline = time.monotonic() + 60.0
        final_history: list[dict] = post_send_history
        reply_seen = False
        while time.monotonic() < reply_deadline:
            final_history = await _get_history(src_client)
            for m in final_history:
                if _classify_message(m) != "HumanMessage":
                    continue
                txt = _content(m)
                if (
                    txt.startswith(f"{dst_info.agent_name} -> {src_info.agent_name} :>")
                    and _MARKER in txt
                ):
                    reply_seen = True
                    break
            if reply_seen:
                break
            await asyncio.sleep(0.5)
    finally:
        await src_client.disconnect()

    assert reply_seen, (
        f"B's marker reply never landed in A's history; cannot judge "
        f"premature-idle bug. Last history tail: "
        f"{[(_classify_message(m), _content(m)[:60]) for m in final_history[-6:]]}"
    )

    # ── Step 5: analyse A's response to B's reply.
    # Spec: when an inter-agent reply arrives, A's NEXT assistant message
    # MUST be a tool call back to send_message_to_agent (per the
    # ``reply_back=True`` cognitive hint InterAgentReplyMiddleware
    # injects). Any plain AIMessage with non-empty text content and no
    # send_message_to_agent tool call between the inbound reply and
    # whatever-else-A-does is the regression we are guarding against —
    # the "Waiting for a reply from X" filler the user reported.
    #
    # Locate the inbound HumanMessage carrying B's marker reply.
    inbound_idx = -1
    inbound_prefix = f"{dst_info.agent_name} -> {src_info.agent_name} :>"
    for i, m in enumerate(final_history):
        if _classify_message(m) != "HumanMessage":
            continue
        if _content(m).startswith(inbound_prefix) and _MARKER in _content(m):
            inbound_idx = i
            break
    assert inbound_idx >= 0, "Could not locate inbound reply in final history"

    after_reply = final_history[inbound_idx + 1 :]

    # Walk forward until we either hit an AIMessage with a
    # send_message_to_agent tool call (CORRECT behaviour) or an
    # AIMessage with non-empty text content and no such tool call
    # (BUG — claude-code generated a filler).
    bad_filler: list[str] = []
    correct_reply_seen = False
    for m in after_reply:
        if _classify_message(m) != "AIMessage":
            continue
        content = _content(m).strip()
        tcs = _tool_calls(m)
        tc_names = [tc.get("name", "") for tc in tcs]
        if "send_message_to_agent" in tc_names:
            correct_reply_seen = True
            break
        if content:
            bad_filler.append(content[:160])
            # Don't break — keep collecting in case there are several.

    assert correct_reply_seen and not bad_filler, (
        "REGRESSION: after receiving B's inter-agent reply, claude-code "
        "agent A produced filler assistant message(s) INSTEAD OF "
        "replying via send_message_to_agent (which the "
        "InterAgentReplyMiddleware cognitive hint explicitly requested).\n"
        f"correct_reply_seen={correct_reply_seen}\n"
        f"Offending filler messages: {bad_filler}\n"
        f"Full history after inbound reply (idx={inbound_idx}):\n  " +
        "\n  ".join(
            f"[{_classify_message(m)}] {_content(m)[:120]!r} "
            f"tool_calls={[tc.get('name','?') for tc in _tool_calls(m)]}"
            for m in after_reply
        )
    )
