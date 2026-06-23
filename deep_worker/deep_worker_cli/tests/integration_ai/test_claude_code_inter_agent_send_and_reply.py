"""Bug reproduction: a claude-code agent A writes to agent B via
send_message_to_agent, and on its own screen after sending and
after receiving the reply from B shows something other than what it should.

Correct scenario (per the project owner's requirement, 2026-06-03):

  1. A calls send_message_to_agent("test_B", "<request>"). By design,
     A's request to B must be the LAST assistant message of A
     in its LangGraph state, and A goes IDLE. That is exactly the
     "final message" of this turn of A — literally the text of the request
     to B (the echo from SendMessageCompletionMiddleware).

  2. B replies through the same tool: send_message_to_agent("test_A",
     "<reply>"). The reply content arrives at A via RPyC
     enqueue_message → _bg_turn → _handle_invoke → astream and lands
     in A's state as a HumanMessage.

  3. A wakes up, recognises the reply and gives a coherent final answer.
     That is where the test ends.

The bug (observed live and now reproduced):

  (1) Instead of the request to B being A's final message, a text
      filler like "Waiting for a reply from test_B" /
      "Waiting for its reply" / "Message sent. Awaiting reply." appears
      in A's state — A emits this text itself, WITHOUT tool_calls, after a
      successful send. This violates the design: the final message of the
      turn must be the SendMessageCompletionMiddleware echo (i.e. the text
      of the request to B itself).

  (2) When B's reply arrives, A either does not wake up at all, or
      wakes up for a moment and does not recognise the reply: its answer
      does not mention the content of B's reply (e.g. the marker), and
      instead again contains something like "Waiting for a reply".

Test without mocks. Two real agents over RPyC. By default
deselected (`@pytest.mark.integration`) and gated on the presence of the `claude` CLI.
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


_SRC_NAME = "test_a"             # claude-code, opus
_DST_NAME = "test_b"             # deepseek
_MARKER = "MARK_42K"             # marker B includes in its reply
_DELAY_HINT_SEC = 20             # B waits this long before replying


def _claude_code_available() -> bool:
    return shutil.which("claude") is not None


def _classify(m: dict) -> str:
    return m.get("type", "")


def _content(m: dict) -> str:
    raw = m.get("content", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for blk in raw:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
        return "\n".join(parts)
    return str(raw)


@pytest.fixture
def two_agents_claude_and_deepseek():
    if not _claude_code_available():
        pytest.skip("claude CLI missing; skipping integration test")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration test")

    # Both agents WITHOUT TaskManager — we need a clean agent loop, without
    # corrective manager prompts, so that any text like
    # "Waiting for a reply" can come ONLY from the model.
    with _make_agent(
        agent_name=_SRC_NAME,
        working_dir=Path("/tmp"),
        provider="claude-code",
        model="opus",
        skip_permissions=True,
        api_key=None,
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
    last = ""
    while time.monotonic() < deadline:
        st = (await client._send_command({"command": "status"})) or {}
        last = str(st.get("status", "")).lower()
        if last in ("idle", "ready"):
            return last
        await asyncio.sleep(0.3)
    return last


@pytest.mark.asyncio
async def test_claude_code_send_to_b_and_recognise_reply(
    two_agents_claude_and_deepseek,
) -> None:
    src_info, dst_info = two_agents_claude_and_deepseek

    # === Prepare B: delay + reply with the marker. ============================
    dst_client = AgentClient(dst_info.agent_name)
    await dst_client.connect()
    try:
        bootstrap = (
            f"SYSTEM INSTRUCTION for tests. When you FIRST receive an "
            f"inter-agent message in the format '<sender> -> {dst_info.agent_name} :> ...', "
            f"you must: \n"
            f"1) First call execute with the command `sleep {_DELAY_HINT_SEC}` "
            f"(the test needs this — do not skip it). \n"
            f"2) Then call send_message_to_agent(<sender>, <reply>) EXACTLY ONCE, "
            f"where <reply> MUST contain the marker '{_MARKER}' verbatim. \n"
            f"3) After sending, do nothing else. \n"
            f"Briefly confirm that you understood the instruction."
        )
        await dst_client.invoke_agent(bootstrap, stream=False)
    finally:
        await dst_client.disconnect()

    # === Step 1: A writes to B and must go IDLE with the request to B as its
    # last assistant message. ==================================================
    src_client = AgentClient(src_info.agent_name)
    await src_client.connect()
    try:
        # The prompt is deliberately simple and natural — no "don't write
        # anything except the tool". Filler protection must be provided by the
        # framework itself (SendMessageCompletionMiddleware), not by the prompt.
        user_request_to_a = (
            f"Pass the task to agent '{dst_info.agent_name}': ask it to "
            f"confirm receipt of the task with the marker '{_MARKER}'. Use the "
            f"send_message_to_agent tool. The reply will come a bit later."
        )
        await src_client.invoke_agent(user_request_to_a, stream=False)

        # Wait for A's first transition to IDLE — this is the end of step 1.
        first_idle = await _wait_until_idle(
            src_client, deadline=time.monotonic() + 60.0
        )
        assert first_idle in ("idle", "ready"), (
            f"A did not go IDLE after send_message_to_agent (last "
            f"status={first_idle!r})"
        )

        # Snapshot of A's history at this point: B's reply must NOT have
        # arrived yet (B sleeps ~20 s). Verify that A's final assistant message
        # is the echo (the request to B), not a text filler.
        history_after_send = await _get_history(src_client)

        # === Steps 2 + 3: wait for the reply from B, then see how A
        # processed it. =========================================================
        reply_prefix = f"{dst_info.agent_name} -> {src_info.agent_name} :>"
        reply_seen_at: int | None = None

        deadline = time.monotonic() + 90.0
        history_final: list[dict] = history_after_send
        while time.monotonic() < deadline:
            history_final = await _get_history(src_client)
            for i, m in enumerate(history_final):
                if _classify(m) != "HumanMessage":
                    continue
                txt = _content(m)
                if txt.startswith(reply_prefix) and _MARKER in txt:
                    reply_seen_at = i
                    break
            if reply_seen_at is not None:
                # Give A time to finish forming its reply to the incoming reply.
                await _wait_until_idle(
                    src_client, deadline=time.monotonic() + 30.0
                )
                history_final = await _get_history(src_client)
                break
            await asyncio.sleep(0.5)
    finally:
        await src_client.disconnect()

    assert reply_seen_at is not None, (
        "B's reply never appeared in A's history — cannot verify "
        "reply recognition. History tail: "
        f"{[(_classify(m), _content(m)[:80]) for m in history_final[-6:]]}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Check (1): after STEP 1, A's final assistant message must be
    # the request to B (echo), NOT a "Waiting for a reply" filler.
    # ──────────────────────────────────────────────────────────────────────────
    # Find the last send_message_to_agent ToolMessage in the snapshot after
    # the send (it marks the end of step 1).
    last_send_tool_idx = -1
    for i, m in enumerate(history_after_send):
        if _classify(m) == "ToolMessage" and m.get("name") == "send_message_to_agent":
            last_send_tool_idx = i
    assert last_send_tool_idx >= 0, (
        "No ToolMessage from send_message_to_agent in A's history after step 1 — "
        "the send did not happen. History tail: "
        f"{[(_classify(m), _content(m)[:80]) for m in history_after_send[-6:]]}"
    )

    expected_echo_prefix = f"{src_info.agent_name} -> {dst_info.agent_name} :>"
    tail_after_send = history_after_send[last_send_tool_idx + 1 :]

    # Among the assistant messages AFTER the ToolMessage there must be exactly
    # one non-empty one — the echo with the request to B. Any other non-empty
    # AIMessage is a filler (bug 1).
    non_empty_ais: list[str] = []
    echo_found = False
    for m in tail_after_send:
        if _classify(m) != "AIMessage":
            continue
        text = _content(m).strip()
        if not text:
            continue
        if text.startswith(expected_echo_prefix):
            echo_found = True
        else:
            non_empty_ais.append(text[:200])

    assert echo_found, (
        "Echo with the request to B not found in A's final assistant message.\n"
        f"Tail after ToolMessage:\n  " +
        "\n  ".join(
            f"[{_classify(m)}] {_content(m)[:160]!r}" for m in tail_after_send
        )
    )

    assert not non_empty_ais, (
        "BUG (1): after send_message_to_agent, A emitted filler assistant "
        "messages instead of quietly going IDLE with the request to B as its final "
        "message.\n"
        f"Extra assistant messages: {non_empty_ais}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Check (2): after B's reply arrives (HumanMessage at index
    # reply_seen_at in the final history), A must give a coherent final
    # answer that mentions the marker.
    # ──────────────────────────────────────────────────────────────────────────
    after_reply = history_final[reply_seen_at + 1 :]

    next_ai_texts: list[str] = []
    for m in after_reply:
        if _classify(m) != "AIMessage":
            continue
        text = _content(m).strip()
        if not text:
            continue
        next_ai_texts.append(text)

    assert next_ai_texts, (
        "BUG (2): A gave NO non-empty assistant answer after "
        "receiving B's reply — it either never woke up or exited "
        "the turn immediately. History tail after reply:\n  " +
        "\n  ".join(
            f"[{_classify(m)}] {_content(m)[:160]!r}" for m in after_reply
        )
    )

    final_ai_text = next_ai_texts[-1]

    # A read the HumanMessage with the marker into its state and must give a
    # SUBSTANTIVE answer that directly acknowledges the FACT of having received
    # B's reply. Any phrasing in the vein of "waiting / reply will come later /
    # awaiting" means A thinks the reply has NOT arrived yet — that is exactly
    # the observed bug.
    #
    # The words "waiting" alone are not a sufficient filter — the model
    # may say "the reply WILL come a bit later" (future tense), which is
    # semantically the same bug. So we use an LLM judge: a small model is asked
    # to assess whether A acknowledges the fact that the reply arrived.
    judge_verdict = await _llm_judge_recognises_reply(
        agent_request=user_request_to_a,
        marker=_MARKER,
        a_final_assistant=final_ai_text,
    )
    assert judge_verdict["recognised"], (
        "BUG (2): A woke up after B's reply but does not acknowledge the fact "
        "of having received the reply in its final assistant message.\n"
        f"Final A: {final_ai_text[:300]!r}\n"
        f"Judge (deepseek): {judge_verdict['explanation']}"
    )


async def _llm_judge_recognises_reply(
    agent_request: str, marker: str, a_final_assistant: str
) -> dict:
    """Small LLM judge via the deepseek API: did A recognise that the reply arrived.

    Returns {"recognised": bool, "explanation": str}. On a judge error it
    returns recognised=True — we do NOT block the test over judge
    problems; such an error means we cannot prove the bug, not the
    opposite.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        return {"recognised": True, "explanation": "DEEPSEEK_API_KEY not set"}
    import json
    import urllib.request

    prompt = (
        "You are the judge of a test about inter-agent communication. Agent A "
        "asked agent B to confirm receipt of the task with the marker "
        f"'{marker}'. B has already sent a reply containing this marker. After "
        "receiving the reply, A wrote the following final assistant message:\n\n"
        f"<<<{a_final_assistant}>>>\n\n"
        "Does this message explicitly acknowledge the FACT that A has already "
        "received the reply from B? Answers like 'awaiting reply', 'waiting', "
        "'the reply will come later', 'waiting for reply', and so on are NOT "
        "acknowledgement; they mean A thinks the reply has not arrived yet. "
        "Clear confirmation like 'B confirmed receipt with the marker X' IS "
        "acknowledgement.\n\n"
        "Answer STRICTLY with valid JSON on a single line without markdown: "
        "{\"recognised\": true|false, \"explanation\": \"<brief explanation>\"}"
    )
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_key}",
        },
        method="POST",
    )
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=30).read()
        )
        data = json.loads(resp.decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        # Strip any possible markdown code fences.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        verdict = json.loads(text)
        return {
            "recognised": bool(verdict.get("recognised")),
            "explanation": str(verdict.get("explanation", "")),
        }
    except Exception as e:
        return {
            "recognised": True,
            "explanation": f"Judge returned an error: {e!r}",
        }
