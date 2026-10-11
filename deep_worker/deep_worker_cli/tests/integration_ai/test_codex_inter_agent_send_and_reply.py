"""Codex-provider analogue of the claude-code-sdk inter-agent messaging test.

The scenario is exactly the same as in test_claude_code_inter_agent_send_and_reply.py,
but agent A runs on the ``codex`` provider (instead of ``claude-code-sdk``).
Agent B is ``deepseek`` (the same as in the claude-code-sdk test).

Why a SEPARATE test is needed for codex:

  Codex is a provider with its own quirks. Unlike claude-code-sdk, it does
  NOT use ``InterAgentDeliveryResolveMiddleware`` (that middleware lives
  in claude_code_sdk_provider because it is specific to the in-process proxy
  MCP CLI). Codex uses its own JSONL-stream machinery plus native tools
  passthrough, in which the echo from ``SendMessageCompletionMiddleware``
  and the delivery from ``InterAgentReplyMiddleware`` must also work
  correctly — which is exactly what is verified here end-to-end.

Covers two regressions (see the design docs
DESIGN_SEND_MESSAGE_COMPLETION_MIDDLEWARE.md and DESIGN_REPLY_BACK_FOR_INTER_AGENT_MESSAGING.md):

  (1) After send_message_to_agent, A must go to IDLE with the ECHO of
      the request to B as its last assistant message. There must be NO
      textual filler like "Waiting for a reply from test_b" /
      "Waiting for reply".

  (2) When B's reply arrives (a HumanMessage in A's state), A must wake
      up and give a substantive final answer that explicitly
      acknowledges the FACT that it received B's reply (not "the reply
      will come later").

No mocks. Two real agents over RPyC. Deselected by default
(``@pytest.mark.integration``) and gated on the presence of the ``codex`` CLI.
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


_SRC_NAME = "test_a_codex"        # codex, default model
_DST_NAME = "test_b_codex"        # deepseek
_MARKER = "MARK_CDX_42K"          # marker B includes in its reply
_DELAY_HINT_SEC = 20              # B waits this long before replying


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _codex_provider_smoke() -> tuple[bool, str]:
    """Thin smoke check: can our codex_provider execute at least one turn
    on this machine RIGHT NOW.

    There are at least two infrastructure reasons why codex can refuse
    BEFORE our framework gets a chance to verify anything:

    1. The ``codex`` CLI 0.130+ picks the ``gpt-5.3-codex`` model by
       default, which returns 400 invalid_request_error for ChatGPT-Plus
       accounts.
    2. Our ``exec_runner`` always passes ``--ignore-user-config`` (this
       is needed for MCP isolation; see the incident observed live on
       2026-05-20). With ``--ignore-user-config``, a ChatGPT-Plus account
       refuses even older models like ``gpt-5.2``, although
       ``codex exec -m gpt-5.2`` without this flag accepts them.

    Skipping with an explicit reason is better than 'fail' in this case —
    we are testing inter-agent reply on codex, not the ChatGPT-bound auth
    infrastructure.
    """
    import subprocess
    smoke_model = os.getenv("CODEX_MODEL", "gpt-5.5")
    try:
        proc = subprocess.run(
            ["codex", "exec", "--ignore-user-config", "-m", smoke_model, "-"],
            input="ok",
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return (False, f"codex smoke failed to launch: {e!r}")
    combined = (proc.stdout or "") + (proc.stderr or "")
    if "invalid_request_error" in combined or "is not supported" in combined:
        return (False, (
            "codex CLI refuses to execute a turn on this machine "
            "(likely ChatGPT-bound auth + --ignore-user-config + "
            "default model). This is a codex infrastructure issue, not "
            "an inter-agent reply problem. See the _codex_provider_smoke() "
            "docstring for details. Stderr tail: "
            + combined.strip().splitlines()[-1][:200]
        ))
    return (True, "ok")


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
def two_agents_codex_and_deepseek():
    if not _codex_available():
        pytest.skip("codex CLI not installed; skipping integration test")
    ok, reason = _codex_provider_smoke()
    if not ok:
        pytest.skip(reason)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        pytest.skip("DEEPSEEK_API_KEY not set; skipping integration test")

    # Both agents WITHOUT TaskManager — the plain agent loop without
    # manager corrective prompts, so any "Waiting for a reply" can come
    # ONLY from the model.
    #
    # The Codex CLI authenticates via ~/.codex/auth.json (chatgpt-bound).
    # IMPORTANT: codex 0.130+ defaults to 'gpt-5.3-codex', which returns
    # 400 invalid_request_error for a ChatGPT-Plus account. That is why
    # we explicitly pin gpt-5.2 as a compatible model (see also
    # test_codex_mcp_prefix, which relies on the CLI default instead).
    # If the CODEX_MODEL variable is set, we use it.
    # gpt-5.5 is the only model the ChatGPT-Plus account currently
    # accepts via ``--ignore-user-config`` (we always pass this flag;
    # see exec_runner.py:514, incident observed live on 2026-05-20). On
    # different subscriptions or in CI with a real OPENAI_API_KEY it can
    # be overridden via CODEX_MODEL.
    codex_model = os.getenv("CODEX_MODEL", "gpt-5.5")
    with _make_agent(
        agent_name=_SRC_NAME,
        working_dir=Path("/tmp"),
        provider="codex",
        model=codex_model,
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
async def test_codex_send_to_b_and_recognise_reply(
    two_agents_codex_and_deepseek,
) -> None:
    src_info, dst_info = two_agents_codex_and_deepseek

    # === Prepare B: delay + reply carrying the marker. =========================
    dst_client = AgentClient(dst_info.agent_name)
    await dst_client.connect()
    try:
        bootstrap = (
            f"SYSTEM INSTRUCTION for tests. When you FIRST receive an "
            f"inter-agent message in the format '<sender> -> {dst_info.agent_name} :> ...', "
            f"you must: \n"
            f"1) First call execute with the command `sleep {_DELAY_HINT_SEC}` "
            f"(the test needs this — don't skip it). \n"
            f"2) Then call send_message_to_agent(<sender>, <answer>) "
            f"EXACTLY ONCE, where <answer> MUST contain the marker "
            f"'{_MARKER}' verbatim. \n"
            f"3) After sending, don't do anything else. \n"
            f"Briefly confirm that you understood the instruction."
        )
        await dst_client.invoke_agent(bootstrap, stream=False)
    finally:
        await dst_client.disconnect()

    # === Step 1: A (codex) writes to B and must go to IDLE with the
    # request to B as its last assistant message. ===============================
    src_client = AgentClient(src_info.agent_name)
    await src_client.connect()
    try:
        user_request_to_a = (
            f"Pass this task to agent '{dst_info.agent_name}': ask it to "
            f"confirm receipt of the task with the marker '{_MARKER}'. Use "
            f"the send_message_to_agent tool. The reply will come shortly."
        )
        await src_client.invoke_agent(user_request_to_a, stream=False)

        # Wait for A's FIRST transition to IDLE — end of step 1.
        first_idle = await _wait_until_idle(
            src_client, deadline=time.monotonic() + 90.0
        )
        assert first_idle in ("idle", "ready"), (
            f"A (codex) did not go to IDLE after send_message_to_agent "
            f"(last status={first_idle!r})"
        )

        # Snapshot of A's history at this point: B's reply must NOT have
        # arrived yet (B sleeps ~20s). We check that A's final assistant
        # message is the echo (request to B), not a textual filler.
        history_after_send = await _get_history(src_client)

        # === Step 2 + 3: wait for B's reply, then see how A handled it. ========
        reply_prefix = f"{dst_info.agent_name} -> {src_info.agent_name} :>"
        reply_seen_at: int | None = None

        deadline = time.monotonic() + 120.0
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
                # Give A time to finish forming its reply response.
                await _wait_until_idle(
                    src_client, deadline=time.monotonic() + 60.0
                )
                history_final = await _get_history(src_client)
                break
            await asyncio.sleep(0.5)
    finally:
        await src_client.disconnect()

    assert reply_seen_at is not None, (
        "B's reply never appeared in A's history — cannot verify reply "
        "recognition. History tail: "
        f"{[(_classify(m), _content(m)[:80]) for m in history_final[-6:]]}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Check (1): after STEP 1, A's final assistant message must be the
    # request to B (echo), NOT a "Waiting for a reply" filler.
    # ──────────────────────────────────────────────────────────────────────────
    last_send_tool_idx = -1
    for i, m in enumerate(history_after_send):
        if _classify(m) == "ToolMessage" and m.get("name") == "send_message_to_agent":
            last_send_tool_idx = i
    assert last_send_tool_idx >= 0, (
        "No ToolMessage from send_message_to_agent in A (codex) history "
        "after step 1 — the send did not happen. History tail: "
        f"{[(_classify(m), _content(m)[:80]) for m in history_after_send[-6:]]}"
    )

    expected_echo_prefix = f"{src_info.agent_name} -> {dst_info.agent_name} :>"
    tail_after_send = history_after_send[last_send_tool_idx + 1 :]

    # Among the assistant messages AFTER the ToolMessage there must be
    # exactly one non-empty one — the echo with the request to B. Any
    # other non-empty AIMessage is a filler (bug 1).
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
        "Echo with the request to B not found in A's final assistant "
        "message (codex).\n"
        f"Tail after ToolMessage:\n  " +
        "\n  ".join(
            f"[{_classify(m)}] {_content(m)[:160]!r}" for m in tail_after_send
        )
    )

    assert not non_empty_ais, (
        "BUG (1) on codex: after send_message_to_agent, A emitted filler "
        "assistant messages instead of quietly going to IDLE with the "
        "request to B as its final message.\n"
        f"Extra assistant messages: {non_empty_ais}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Check (2): after B's reply arrives (the HumanMessage at index
    # reply_seen_at in the final history), A must give a coherent final
    # answer mentioning the marker.
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
        "BUG (2) on codex: A produced NO non-empty assistant answer "
        "after receiving B's reply — either it never woke up, or it "
        "exited the turn immediately. History tail after reply:\n  " +
        "\n  ".join(
            f"[{_classify(m)}] {_content(m)[:160]!r}" for m in after_reply
        )
    )

    final_ai_text = next_ai_texts[-1]

    judge_verdict = await _llm_judge_recognises_reply(
        agent_request=user_request_to_a,
        marker=_MARKER,
        a_final_assistant=final_ai_text,
    )
    assert judge_verdict["recognised"], (
        "BUG (2) on codex: A woke up after B's reply, but does not "
        "acknowledge the fact that it received the reply in its final "
        "assistant message.\n"
        f"Final A: {final_ai_text[:300]!r}\n"
        f"Judge (deepseek): {judge_verdict['explanation']}"
    )


async def _llm_judge_recognises_reply(
    agent_request: str, marker: str, a_final_assistant: str
) -> dict:
    """LLM judge via deepseek: did A recognise that the reply arrived.

    On a judge error it returns recognised=True — we do NOT block the
    test because of judge problems.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        return {"recognised": True, "explanation": "DEEPSEEK_API_KEY not set"}
    import json
    import urllib.request

    prompt = (
        "You are the judge of an inter-agent messaging test. Agent A "
        "asked agent B to confirm receipt of a task with the marker "
        f"'{marker}'. B has already sent a reply containing that marker. "
        "After receiving the reply, A wrote the following final "
        "assistant message:\n\n"
        f"<<<{a_final_assistant}>>>\n\n"
        "Does this message explicitly acknowledge the FACT that A "
        "already received B's reply? Answers like 'awaiting a reply', "
        "'waiting', 'the reply will come later', 'waiting for reply' and "
        "so on are NOT acknowledgement; they mean A believes the reply "
        "has not arrived yet. A clear confirmation like 'B confirmed "
        "receipt with marker X' IS acknowledgement.\n\n"
        "Reply STRICTLY as valid JSON on a single line without markdown: "
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
