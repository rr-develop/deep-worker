"""Three paths that must never degrade into a full-history dump.

Reviewer blockers 3a–3c. Each of these used to end, directly or
indirectly, in "open a fresh CLI session and re-send the whole visible
history" — the failure mode that the summarizer port exists to prevent.

3a  The CLI answers a summarize prompt with tool_calls → empty content →
    "empty summary" → hot demoted to cold → cold dumps.
3b  The summarizer port is missing on a session-stateful provider → the
    identical-prefix path re-``ainvoke``s the agent thread → 4b → dump.
3c  The 4b net itself resets state and then CONTINUES the request, opening
    session/new with every HumanMessage as the first prompt.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_worker.providers.acp.chat_model import (
    AcpChatModel,
    AcpSummarizationResetRequired,
)
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import ThreadState
from deep_worker.providers.acp.profile import AcpProfile


THREAD = "dump_guard_agent"


def _model(client_state: AcpClientState) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._acp_client = MagicMock()
    model._acp_client.session_cancel = AsyncMock()
    model._acp_client.session_new = AsyncMock(return_value="sess-new")
    model._acp_client.session_prompt = MagicMock()
    model._profile = AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
    )
    model._client_state = client_state
    model._proxy_mcp = None
    model._tools_metadata = []
    model._usage_event_callback = None
    model._working_dir = "/tmp"
    return model


# ── 3a: tool_calls during summarize are retried, not demoted ──────────


def test_summarize_retries_when_cli_answers_with_tool_calls() -> None:
    """A tool_call reply must trigger a re-prompt, not "empty summary"."""
    cs = AcpClientState()
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    model = _model(cs)

    replies = [
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {}, "id": "t1"}
        ]),
        AIMessage(content="THE SUMMARY"),
    ]
    seen_prompts: list[str] = []

    async def _consume(state, thread_id):  # noqa: ARG001
        return replies.pop(0)

    def _prompt(*, session_id, prompt, notification_sink):  # noqa: ARG001
        seen_prompts.append(prompt[0]["text"])
        return asyncio.get_event_loop().create_future()

    model._consume_turn = _consume
    model._acp_client.session_prompt = _prompt

    result = asyncio.run(
        model._prompt_for_summary(st, THREAD, "summarize this", path="hot")
    )

    assert result.text == "THE SUMMARY"
    assert len(seen_prompts) == 2, "the tool_call reply was not re-prompted"
    assert "Tools are unavailable during summarization" in seen_prompts[1]
    assert "summarize this" in seen_prompts[1], "original instruction dropped"


def test_summarize_gives_up_after_bounded_retries() -> None:
    """A model that always wants tools must not spin forever."""
    cs = AcpClientState()
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    model = _model(cs)

    calls = {"n": 0}

    async def _consume(state, thread_id):  # noqa: ARG001
        calls["n"] += 1
        return AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {}, "id": f"t{calls['n']}"}
        ])

    model._consume_turn = _consume
    model._acp_client.session_prompt = MagicMock(
        side_effect=lambda **kw: asyncio.get_event_loop().create_future()
    )

    with pytest.raises(RuntimeError, match="empty summary"):
        asyncio.run(
            model._prompt_for_summary(st, THREAD, "summarize", path="hot")
        )
    assert calls["n"] == 3, f"expected 1 try + 2 retries, got {calls['n']}"


def test_summarize_empty_without_tool_calls_fails_immediately() -> None:
    """A genuinely empty reply is not a tool problem — do not burn retries."""
    cs = AcpClientState()
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    model = _model(cs)

    calls = {"n": 0}

    async def _consume(state, thread_id):  # noqa: ARG001
        calls["n"] += 1
        return AIMessage(content="")

    model._consume_turn = _consume
    model._acp_client.session_prompt = MagicMock(
        side_effect=lambda **kw: asyncio.get_event_loop().create_future()
    )

    with pytest.raises(RuntimeError, match="empty summary"):
        asyncio.run(
            model._prompt_for_summary(st, THREAD, "summarize", path="hot")
        )
    assert calls["n"] == 1


# ── 3c: the 4b net stops instead of re-prompting ──────────────────────


def test_4b_net_raises_instead_of_opening_a_fresh_session() -> None:
    """sent_count > state length must reset AND abort the request."""
    cs = AcpClientState()
    cs.set_profile("claude-code", {"model": "fable"})
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-old"
    st.sent_count = 12  # more than the post-summary state below
    model = _model(cs)
    model._adopt_pump = AsyncMock()
    model._discard_pending_prompt = MagicMock()

    messages = [SystemMessage(content="sys"), HumanMessage(content="only one")]

    with pytest.raises(AcpSummarizationResetRequired):
        asyncio.run(model.ainvoke(messages, config={
            "configurable": {"thread_id": THREAD}
        }))

    model._acp_client.session_new.assert_not_awaited()
    after = cs.get_or_create(THREAD)
    assert after.session_id is None, "reset did not drop the stale session"
    assert after.sent_count == 0


# ── 3e: SummaryResult.usage must not be silently None ─────────────────


def test_summary_usage_falls_back_to_the_live_gauge() -> None:
    """ProvenanceMiddleware only wraps graph calls; asummarize bypasses it.

    Without a fallback every SummaryResult.usage is None, so the summary
    line in the offload carries no usage at all (reviewer note 3e).
    """
    cs = AcpClientState()
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    st.usage_snapshot = {"used": 4242, "size": 200000}
    model = _model(cs)

    async def _consume(state, thread_id):  # noqa: ARG001
        return AIMessage(content="THE SUMMARY")

    model._consume_turn = _consume
    model._acp_client.session_prompt = MagicMock(
        side_effect=lambda **kw: asyncio.get_event_loop().create_future()
    )

    result = asyncio.run(
        model._prompt_for_summary(st, THREAD, "summarize", path="hot")
    )
    assert result.usage is not None, "usage was left None"
    assert result.usage["context_used"] == 4242


def test_summary_usage_prefers_provenance_when_present() -> None:
    """The fallback must not override a real middleware-written value."""
    cs = AcpClientState()
    st = cs.get_or_create(THREAD)
    st.session_id = "sess-1"
    st.usage_snapshot = {"used": 4242, "size": 200000}
    model = _model(cs)

    async def _consume(state, thread_id):  # noqa: ARG001
        return AIMessage(
            content="THE SUMMARY",
            response_metadata={"deep_worker_usage": {"input_tokens": 7}},
        )

    model._consume_turn = _consume
    model._acp_client.session_prompt = MagicMock(
        side_effect=lambda **kw: asyncio.get_event_loop().create_future()
    )

    result = asyncio.run(
        model._prompt_for_summary(st, THREAD, "summarize", path="hot")
    )
    assert result.usage == {"input_tokens": 7}
