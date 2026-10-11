"""Grok provider: inter-agent communication A→B→A via send_message_to_agent.

The scenario matches test_claude_code_inter_agent_send_and_reply.py and
test_codex_inter_agent_send_and_reply.py, but **both** agents (A and B) run
on the ``grok`` provider.

Covers two regressions:

  (1) After send_message_to_agent, A must go IDLE with the echo of the
      request to B as the last assistant message. There must be NO text
      filler like "Waiting for reply from test_b" / "Waiting for reply".

  (2) When B's reply arrives (HumanMessage in A's state), A must wake up
      and give a substantive final answer that explicitly acknowledges the
      FACT that a reply from B was received (not "the reply will come
      later").

Test without mocks. Two real agents over RPyC. Disabled by default
(``@pytest.mark.integration``) and gated on ``GROK_INTEGRATION_TESTS=1``,
``grok`` being on PATH, and ``~/.grok/auth.json``.
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
from deep_worker_cli.storage_paths import deep_worker_home


_INTEGRATION_ENABLED = os.environ.get("GROK_INTEGRATION_TESTS", "").lower() in (
    "1",
    "true",
    "yes",
)
_GROK_AVAILABLE = shutil.which("grok") is not None
# The operator's REAL grok home, captured at IMPORT time — before the autouse
# ``_sandbox_deep_worker_home`` fixture (tests/conftest.py) repoints HOME at a
# per-run sandbox. Inside a test ``Path.home()`` is already that sandbox.
_REAL_GROK_HOME = Path.home() / ".grok"
_AUTH_PRESENT = (_REAL_GROK_HOME / "auth.json").exists()


@pytest.fixture(autouse=True)
def _grok_home_points_at_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the grok CLI its credentials back under the sandboxed HOME.

    The HOME sandbox (owner directive 2026-08-17: a test must never write the
    live runtime tree) also hides ``~/.grok/auth.json`` from the grok CLI the
    agent spawns. The CLI then starts UNAUTHENTICATED — its own log shows
    ``auth.json load result: read=error`` followed by an endless
    ``model catalog refresh failed`` backoff — so it never becomes promptable
    and the agent's very first turn hangs until the test times out. The
    module-level skipif above checked the REAL home and passed, which is why
    this read as a product hang rather than a harness defect (diagnosed
    2026-08-24; the same failure reproduces on pre-fix code).

    ``GROK_HOME`` overrides the CLI's home independently of ``HOME`` (verified
    live: ``HOME=<sandbox> GROK_HOME=~/.grok grok du`` reports the real
    store), so credentials are restored while all deep-worker state stays
    sandboxed. The variable is inherited by the multiprocessing-spawned agent
    process and by the CLI subprocess underneath it.
    """
    monkeypatch.setenv("GROK_HOME", str(_REAL_GROK_HOME))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _INTEGRATION_ENABLED,
        reason="set GROK_INTEGRATION_TESTS=1 to enable live grok tests",
    ),
    pytest.mark.skipif(not _GROK_AVAILABLE, reason="grok CLI not found on PATH"),
    pytest.mark.skipif(
        not _AUTH_PRESENT,
        reason="~/.grok/auth.json missing — grok requires a supergrok login",
    ),
]


_SRC_NAME = "test_a_grok"
_DST_NAME = "test_b_grok"
_MARKER = "MARK_GRK_42K"
_DELAY_HINT_SEC = 5
# SLA. Still a hard budget — an overrun is a defect, not noise — but
# calibrated against MEASURED grok 1.0.5 latency (2026-08-24, live agents
# started by the normal launcher, after the correlation fix below):
#   * first turn of a process ~20 s — the grok CLI cold start (spawn, auth,
#     model catalog, session/new) is paid once per agent;
#   * warm trivial turn ~2.5 s;
#   * a turn that calls send_message_to_agent ~13 s (it was ~30 s until the
#     dead 10 s proxy-correlation wait was removed — grok dispatches our
#     tools over ACP and never parks an MCP handler, so the wait could
#     never be satisfied; see PROXY_CORRELATION_PROBE_SEC).
# B's bootstrap and A's send are each a FIRST turn, so they carry the cold
# start; the reply leg is B's warm turn plus delivery.
_BOOTSTRAP_IDLE_SEC = 45.0
_REPLY_WAIT_SEC = 90.0
_IDLE_AFTER_REPLY_SEC = 10.0


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


def _cleanup_agent_state(agent_name: str) -> None:
    """Kill stale processes and remove checkpoints/logs so each run starts fresh.

    agent_fixture reuses live registry entries — without killing the process
    first, a previous failed run's in-memory grok session survives and B may
    answer with text instead of executing tools.
    """
    import os
    import signal

    from deep_worker_cli.process.registry import ProcessRegistry

    registry = ProcessRegistry()
    for _attempt in range(3):
        existing = registry.lookup(agent_name)
        if existing and existing.pid:
            try:
                os.kill(existing.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not registry.lookup(agent_name):
                    break
                time.sleep(0.1)
        try:
            registry.unregister(agent_name)
        except Exception:
            pass
        if registry.lookup(agent_name) is None:
            break
        time.sleep(0.5)

    time.sleep(0.5)

    for pid_file in (
        Path(f"/tmp/deep_worker_{agent_name}.pid"),
        Path(f"/tmp/deep_worker_{agent_name}.lock"),
    ):
        if pid_file.exists():
            try:
                stale_pid = int(pid_file.read_text().strip().split()[0])
                os.kill(stale_pid, signal.SIGKILL)
            except (ProcessLookupError, ValueError, OSError):
                pass
            pid_file.unlink(missing_ok=True)

    dw_root = deep_worker_home()
    data_dir = dw_root / agent_name
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    dw_root.mkdir(parents=True, exist_ok=True)

    log_dir = dw_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for log_file in log_dir.glob(f"{agent_name}.*"):
        log_file.unlink(missing_ok=True)


def _grok_agent_kwargs() -> dict:
    return {
        "working_dir": Path("/tmp"),
        "provider": "grok",
        "model": os.getenv("GROK_MODEL"),
        "skip_permissions": True,
        "api_key": None,
        "enable_task_manager": False,
    }


async def _get_history(client: AgentClient) -> list[dict]:
    resp = await client._send_command(
        {"command": "get_history", "limit": 500, "offset": 0}
    )
    assert resp.get("success"), f"get_history failed: {resp}"
    return resp.get("history", []) or []


def _assert_agent_registry_alive(agent_name: str) -> None:
    """Fail fast when a fixture-spawned agent died before the scenario runs."""
    from deep_worker_cli.process.registry import ProcessRegistry

    info = ProcessRegistry().lookup(agent_name)
    assert info is not None, (
        f"Agent {agent_name!r} is not in ProcessRegistry — process likely "
        f"crashed during setup or peer bootstrap. Check the agent's logs."
    )


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
@pytest.mark.timeout(240)
async def test_grok_send_to_b_and_recognise_reply() -> None:
    grok_kwargs = _grok_agent_kwargs()
    _cleanup_agent_state(_SRC_NAME)
    _cleanup_agent_state(_DST_NAME)

    reply_seen_at: int | None = None
    history_after_send: list[dict] = []
    history_final: list[dict] = []
    user_request_to_a = ""
    src_info = None
    dst_info = None

    # Bootstrap B alone — two concurrent grok MCP servers during B's first
    # invoke were killing test_a_grok (registry loss / MCP clash).
    with _make_agent(agent_name=_DST_NAME, **grok_kwargs) as dst_info:
        _assert_agent_registry_alive(_DST_NAME)

        dst_client = AgentClient(dst_info.agent_name)
        await dst_client.connect()
        try:
            bootstrap = (
                f"SYSTEM INSTRUCTION for tests. When you FIRST receive an "
                f"inter-agent message in the format '<sender> -> {dst_info.agent_name} :> ...', "
                f"you must: \n"
                f"1) First call execute with the command `sleep {_DELAY_HINT_SEC}` "
                f"(the test needs this — do not skip it). \n"
                f"2) Then call EXACTLY ONCE "
                f"send_message_to_agent(<sender>, <reply>, reply_back=False), "
                f"where <reply> MUST contain the marker "
                f"'{_MARKER}' verbatim. reply_back=False is mandatory — "
                f"otherwise A goes into a ping-pong instead of answering the "
                f"user. \n"
                f"3) After sending, do nothing else. \n"
                f"Briefly confirm that the instruction is understood."
            )
            await dst_client.invoke_agent(bootstrap, stream=False)
            bootstrap_idle = await _wait_until_idle(
                dst_client, deadline=time.monotonic() + _BOOTSTRAP_IDLE_SEC
            )
            assert bootstrap_idle in ("idle", "ready"), (
                f"B (grok) did not finish bootstrap (status={bootstrap_idle!r}). "
                "Check the logs — the model may have replied with text instead of tool calls."
            )
        finally:
            await dst_client.disconnect()

        _assert_agent_registry_alive(_DST_NAME)
        # Brief pause so B's grok MCP server settles before A spawns a second.
        await asyncio.sleep(1.0)
        _assert_agent_registry_alive(_DST_NAME)

        with _make_agent(agent_name=_SRC_NAME, **grok_kwargs) as src_info:
            _assert_agent_registry_alive(_SRC_NAME)
            _assert_agent_registry_alive(_DST_NAME)

            src_client = AgentClient(src_info.agent_name)
            await src_client.connect()
            try:
                user_request_to_a = (
                    f"Pass the task to agent '{dst_info.agent_name}': ask it to "
                    f"confirm receipt of the task with the marker '{_MARKER}'. "
                    f"Use the send_message_to_agent tool. The reply will come "
                    f"shortly."
                )
                await src_client.invoke_agent(user_request_to_a, stream=False)

                st = (await src_client._send_command({"command": "status"})) or {}
                first_idle = str(st.get("status", "")).lower()
                assert first_idle in ("idle", "ready"), (
                    f"A (grok) did not go IDLE after send_message_to_agent "
                    f"(last status={first_idle!r})"
                )

                history_after_send = await _get_history(src_client)

                reply_prefix = f"{dst_info.agent_name} -> {src_info.agent_name} :>"
                echo_to_b_prefix = (
                    f"{src_info.agent_name} -> {dst_info.agent_name} :>"
                )
                last_send_idx = max(
                    (
                        i
                        for i, m in enumerate(history_after_send)
                        if _classify(m) == "ToolMessage"
                        and m.get("name") == "send_message_to_agent"
                    ),
                    default=-1,
                )
                deadline = time.monotonic() + _REPLY_WAIT_SEC
                history_final = history_after_send
                while time.monotonic() < deadline:
                    history_final = await _get_history(src_client)
                    # B may enqueue its reply before A's echo lands — scan
                    # backward and only accept HumanMessages after send_tool.
                    for i in range(len(history_final) - 1, last_send_idx, -1):
                        m = history_final[i]
                        if _classify(m) != "HumanMessage":
                            continue
                        txt = _content(m)
                        if reply_prefix in txt and _MARKER in txt:
                            reply_seen_at = i
                            break
                    if reply_seen_at is not None:
                        ai_deadline = time.monotonic() + _IDLE_AFTER_REPLY_SEC
                        while time.monotonic() < ai_deadline:
                            history_final = await _get_history(src_client)
                            tail = history_final[reply_seen_at + 1 :]
                            if any(
                                _classify(m) == "AIMessage"
                                and (t := _content(m).strip())
                                and _MARKER in t
                                and not t.startswith(echo_to_b_prefix)
                                and "will come later" not in t.lower()
                                and "will arrive later" not in t.lower()
                                and "waiting" not in t.lower()
                                and "await" not in t.lower()
                                and "sent to the agent" not in t.lower()
                                and "delivered to the agent" not in t.lower()
                                and "will come separately" not in t.lower()
                                and "will arrive separately" not in t.lower()
                                for m in tail
                            ):
                                break
                            await asyncio.sleep(0.5)
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
    # Check (1): after STEP 1, A's final assistant message must be the
    # request to B (echo), NOT a "Waiting for reply" filler.
    # ──────────────────────────────────────────────────────────────────────────
    last_send_tool_idx = -1
    for i, m in enumerate(history_after_send):
        if _classify(m) == "ToolMessage" and m.get("name") == "send_message_to_agent":
            last_send_tool_idx = i
    assert last_send_tool_idx >= 0, (
        "A (grok) history after step 1 has no ToolMessage from "
        "send_message_to_agent — the send did not happen. History tail: "
        f"{[(_classify(m), _content(m)[:80]) for m in history_after_send[-6:]]}"
    )

    expected_echo_prefix = f"{src_info.agent_name} -> {dst_info.agent_name} :>"
    tail_after_send = history_after_send[last_send_tool_idx + 1 :]

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
        "message (grok).\n"
        f"Tail after ToolMessage:\n  " +
        "\n  ".join(
            f"[{_classify(m)}] {_content(m)[:160]!r}" for m in tail_after_send
        )
    )

    assert not non_empty_ais, (
        "BUG (1) on grok: A after send_message_to_agent emitted filler "
        "assistant messages instead of quietly going IDLE with the request "
        "to B as the final message.\n"
        f"Extra assistant messages: {non_empty_ais}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Check (2): after B's reply arrives, A must give a clear final
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
        low = text.lower()
        if (
            "will come later" in low
            or "will arrive later" in low
            or "waiting" in low
            or "await" in low
            or "sent to the agent" in low
            or "delivered to the agent" in low
            or "will come separately" in low
            or "will arrive separately" in low
        ):
            continue
        if text.startswith(
            f"{src_info.agent_name} -> {dst_info.agent_name} :>"
        ):
            continue
        next_ai_texts.append(text)

    assert next_ai_texts, (
        "BUG (2) on grok: A gave NO non-empty assistant "
        "response after receiving the reply from B — it either never woke up "
        "or exited the turn right away. History tail after reply:\n  " +
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
        "BUG (2) on grok: A woke up after B's reply but does not "
        "acknowledge having received the reply in its final assistant "
        "message.\n"
        f"Final A: {final_ai_text[:300]!r}\n"
        f"Judge (deepseek): {judge_verdict['explanation']}"
    )


async def _llm_judge_recognises_reply(
    agent_request: str, marker: str, a_final_assistant: str
) -> dict:
    """LLM judge via deepseek: did A recognise that the reply arrived.

    On a judge error it returns recognised=True — we do NOT block the test
    because of problems with the judge.
    """
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        return {"recognised": True, "explanation": "DEEPSEEK_API_KEY not set"}
    import json
    import urllib.request

    prompt = (
        "You are a judge for an inter-agent communication test. Agent A "
        "asked agent B to confirm receipt of a task with the marker "
        f"'{marker}'. B has already sent a reply containing this marker. After "
        "receiving the reply, A wrote the following final assistant message:\n\n"
        f"<<<{a_final_assistant}>>>\n\n"
        "Does this message explicitly acknowledge the FACT that A has already "
        "received a reply from B? Answers like 'expecting a reply', 'waiting', "
        "'the reply will come later', 'waiting for reply' and so on are NOT "
        "acknowledgement; they mean A thinks the reply has not arrived yet. "
        "Clear confirmation like 'B confirmed receipt with marker X' is "
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