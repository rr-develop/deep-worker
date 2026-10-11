"""A restart onto the ACP provider must not amnesia the agent (2026-08-16).

Live incident on a running agent: the agent was moved from ``deepseek`` to
``claude-code`` (ACP). Its LangGraph history held 178 messages, but the ACP
provider had no ``acp_session_state.json`` yet, so ``sent_count`` seeded to
0 and ``session_id`` to ``None`` — the "brand-new session" branch.

Step 7 correctly computes ``new_messages = state_messages[0:]`` (the WHOLE
history). Step 7b then re-slices it down to the LAST ``HumanMessage``,
because its only guard is ``sent_count < len(state_messages)`` — trivially
true on a first turn. The brand-new-session branch (a), unlike the
``session/load``-failed fallback (b), never re-expands ``user_text_parts``.

Net effect: 177 messages collapse to 1. The freshly created CLI session is
handed only the newest user line and the agent "does not remember" anything
it did before the restart. ``claude-code-sdk`` restores history explicitly
(``session_restore``/``session_synthesis``, DESIGN_HISTORY_RESTORE_ON_RESTART.md)
and ``codex_provider`` has no 7b re-slice at all, so both are unaffected.

7b itself is legitimate: mid-turn LangGraph appends (ToolMessage / echo
AIMessage) must not be re-fed to a LIVE session. It must simply not fire on
the first turn of a session that has no history yet.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.profile import AcpProfile


class _FakeAcpClient:
    """Records the prompt blocks handed to ``session/prompt``."""

    def __init__(self) -> None:
        self.prompts: list[list[dict]] = []
        self.session_new_calls = 0

    async def ensure_started(self) -> None:
        return None

    async def session_new(self, cwd: str, mcp_servers: list) -> str:  # noqa: ARG002
        self.session_new_calls += 1
        return "sess-new-1"

    async def wait_session_ready(self, session_id: str) -> None:  # noqa: ARG002
        return None

    def session_prompt(self, session_id: str, prompt: list, notification_sink: Any):  # noqa: ARG002
        self.prompts.append(prompt)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result({"stopReason": "end_turn"})
        return fut


def _history(n_pairs: int = 88) -> list:
    """A realistic restarted-agent history: system + many prior turns."""
    msgs: list = [SystemMessage(content="you are an agent")]
    for i in range(n_pairs):
        msgs.append(HumanMessage(content=f"user question {i}"))
        msgs.append(AIMessage(content=f"assistant answer {i}"))
    msgs.append(HumanMessage(content="LATEST question after the restart"))
    return msgs


def _model(client: _FakeAcpClient, client_state: AcpClientState) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._acp_client = client
    model._client_state = client_state
    model._proxy_mcp = None
    model._proxy_started = True
    model._acp_started = True
    model._tools_metadata = []
    model._working_dir = "/tmp"
    model._profile = AcpProfile(provider="claude-code", cmd=("true",))
    model._stats_callback = None
    model._usage_event_callback = None
    return model


@pytest.mark.asyncio
async def test_first_acp_turn_replays_the_whole_history() -> None:
    """On the very first ACP turn for a thread (no persisted session), the
    brand-new CLI session must receive the agent's prior user turns — not
    just the newest line."""
    client = _FakeAcpClient()
    cs = AcpClientState()
    model = _model(client, cs)

    messages = _history()
    await model.ainvoke(
        messages,
        config={"configurable": {"thread_id": "amnesia-thread"}},
    )

    assert client.session_new_calls == 1, "expected a brand-new session"
    assert client.prompts, "no session/prompt was sent"
    text = "\n".join(
        block.get("text", "")
        for block in client.prompts[0]
        if isinstance(block, dict)
    )
    assert "LATEST question after the restart" in text
    assert "user question 0" in text, (
        "the first ACP turn dropped the agent's entire prior history — "
        "only the trailing HumanMessage survived (step 7b re-slice)"
    )
    assert "user question 87" in text


@pytest.mark.asyncio
async def test_live_session_still_gets_only_the_trailing_delta() -> None:
    """The 7b re-slice must KEEP working for a session the CLI already
    carries: LangGraph appends a ToolMessage/echo AIMessage after a mid-turn
    tool return, and re-feeding those stale rows to a live session is the
    defect 7b exists to prevent. Only the first-turn case is exempted."""
    client = _FakeAcpClient()
    cs = AcpClientState()
    model = _model(client, cs)

    state = cs.get_or_create("live-thread")
    state.session_id = "sess-live"
    state.session_live_in_process = True
    state.sent_count = 3

    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        AIMessage(content="stale echo"),
        HumanMessage(content="NEW question"),
    ]
    await model.ainvoke(
        messages,
        config={"configurable": {"thread_id": "live-thread"}},
    )

    assert client.session_new_calls == 0, "must reuse the live session"
    text = "\n".join(
        block.get("text", "")
        for block in client.prompts[0]
        if isinstance(block, dict)
    )
    assert "NEW question" in text
    assert "old question" not in text, (
        "a live session was re-fed history it already has"
    )
