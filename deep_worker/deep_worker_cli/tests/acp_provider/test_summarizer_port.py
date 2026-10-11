"""ACP SummarizerPort: hot in-session prompt, cold ephemeral session, no 4b dump."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.profile import AcpProfile
from deep_worker_cli.context_limit.hot_cache import HotColdDecision, decide


class _FakeAcpClient:
    def __init__(self, reply_text: str = "SESSION INTENT\nsummarized") -> None:
        self.reply_text = reply_text
        self.prompts: list[tuple[str, list]] = []
        self.session_new_calls: list[list] = []
        self.session_load_calls: list[str] = []
        self.session_cancel_calls: list[str] = []
        self._n = 0

    async def ensure_started(self) -> None:
        return None

    async def session_new(self, cwd: str, mcp_servers: list) -> str:  # noqa: ARG002
        self.session_new_calls.append(list(mcp_servers))
        self._n += 1
        return f"sess-new-{self._n}"

    async def session_load(self, session_id: str, cwd: str, mcp_servers: list | None = None):  # noqa: ARG002
        self.session_load_calls.append(session_id)
        return {"result": {"sessionId": session_id}}

    async def session_cancel(self, session_id: str | None) -> None:
        if session_id:
            self.session_cancel_calls.append(session_id)

    async def wait_session_ready(self, session_id: str) -> None:  # noqa: ARG002
        return None

    def session_prompt(self, session_id: str, prompt: list, notification_sink: Any):
        self.prompts.append((session_id, prompt))
        if notification_sink is not None:
            notification_sink.put_nowait(
                {
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": self.reply_text},
                        }
                    }
                }
            )
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result({"result": {"stopReason": "end_turn"}})
        return fut


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


def _hot_decision() -> HotColdDecision:
    now = datetime(2026, 8, 19, 15, 10, tzinfo=UTC)
    ai = AIMessage(
        content="ok",
        response_metadata={
            "deep_worker_origin": {"provider": "claude-code", "model": "fable"},
            "deep_worker_usage": {
                "received_at": (now - timedelta(seconds=30)).isoformat()
            },
        },
    )
    return decide(
        now=now,
        current_provider="claude-code",
        current_model="fable",
        messages=[ai],
        session_id="sess-live",
    )


@pytest.mark.asyncio
async def test_hot_summarize_prompts_existing_session_no_session_new() -> None:
    client = _FakeAcpClient()
    cs = AcpClientState()
    model = _model(client, cs)
    state = cs.get_or_create("rg1_worker")
    state.session_id = "sess-live"
    state.session_live_in_process = True
    state.sent_count = 10

    result = await model.asummarize(
        thread_id="rg1_worker",
        messages=[HumanMessage(content="old"), AIMessage(content="ok")],
        instruction="Summarize. No tools.",
        decision=_hot_decision(),
    )

    assert result.path == "hot"
    assert "SESSION INTENT" in result.text
    assert client.session_new_calls == []
    assert client.session_load_calls == []
    assert [sid for sid, _ in client.prompts] == ["sess-live"]
    prompt_text = client.prompts[0][1][0]["text"]
    assert "Summarize. No tools." in prompt_text
    assert "old" not in prompt_text


_PROXY_SERVER_NAME = "deep_worker_acp_proxy_test"


class _FakeProxy:
    server_name = _PROXY_SERVER_NAME


class _ToolCallingThenProseClient(_FakeAcpClient):
    """CLI answers the first summarize prompt with a tool call, then prose.

    Reproduces DESIGN §5.2a-3: the model reaches for a tool instead of
    summarizing, which yields empty content. Treating that as "empty
    summary" would demote a HOT summarize to a COLD one — a full-history
    dump into a fresh session, i.e. the incident.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        # Interleaved log of BOTH kinds of call, so a test can pin ordering
        # and not merely presence.
        self.events: list[str] = []

    async def session_cancel(self, session_id: str | None) -> None:
        self.events.append(f"cancel:{session_id}")
        await super().session_cancel(session_id)

    def session_prompt(self, session_id: str, prompt: list, notification_sink: Any):
        self.calls += 1
        self.events.append(f"prompt:{session_id}")
        self.prompts.append((session_id, prompt))
        if notification_sink is not None:
            if self.calls == 1:
                # A channel-A proxy dispatch (grok "dispatch_builtin" style):
                # rawInput.server names our proxy, so this surfaces as a real
                # tool_call rather than a built-in passthrough.
                notification_sink.put_nowait(
                    {
                        "params": {
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "call-1",
                                "title": "CallMcpTool",
                                "rawInput": {
                                    "server": _PROXY_SERVER_NAME,
                                    "toolName": "read_file",
                                    "arguments": {"file_path": "/tmp/x"},
                                },
                            }
                        }
                    }
                )
            else:
                notification_sink.put_nowait(
                    {
                        "params": {
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": self.reply_text},
                            }
                        }
                    }
                )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        if self.calls == 1:
            # Turn 1 asks for a tool and stays OPEN: a real CLI does not
            # resolve session/prompt while it waits for tool results. Leaving
            # the future pending is what makes _consume_turn take the
            # notification branch and surface the tool_call — and it is the
            # state the retry must session/cancel out of.
            return fut
        loop.call_soon(
            lambda: fut.done() or fut.set_result(
                {"result": {"stopReason": "end_turn"}}
            )
        )
        return fut


@pytest.mark.asyncio
async def test_tool_calls_during_summarize_never_dump_into_agent_session() -> None:
    """Tool calls mid-summarize must re-prompt in place, never dump.

    Two invariants, both incident-critical (reviewer B3, 2026-08-21):
      1. no session/new and no session/load — the fat history is never
         replayed into a fresh session;
      2. the still-open CLI turn is session/cancel-ed BEFORE the retry
         prompt, so the second prompt cannot collide with a turn that is
         waiting on tool results.
    """
    client = _ToolCallingThenProseClient()
    cs = AcpClientState()
    model = _model(client, cs)
    model._proxy_mcp = _FakeProxy()
    state = cs.get_or_create("rg1_worker")
    state.session_id = "sess-live"
    state.session_live_in_process = True
    state.sent_count = 10

    result = await model.asummarize(
        thread_id="rg1_worker",
        messages=[HumanMessage(content="old history"), AIMessage(content="ok")],
        instruction="Summarize. No tools.",
        decision=_hot_decision(),
    )

    assert result.path == "hot"
    assert "SESSION INTENT" in result.text
    # (1) never dumped
    assert client.session_new_calls == []
    assert client.session_load_calls == []
    assert [sid for sid, _ in client.prompts] == ["sess-live", "sess-live"]
    assert "old history" not in client.prompts[1][1][0]["text"]
    # (2) cancelled the open turn BEFORE re-prompting. Ordering, not just
    # presence: a cancel issued after the retry prompt would not prevent the
    # collision, and an assertion on session_cancel_calls alone passes that
    # mutation.
    assert client.events == [
        "prompt:sess-live",
        "cancel:sess-live",
        "prompt:sess-live",
    ]
    assert "Tools are unavailable" in client.prompts[1][1][0]["text"]


@pytest.mark.asyncio
async def test_hot_mid_turn_cancels_before_prompt() -> None:
    client = _FakeAcpClient()
    cs = AcpClientState()
    model = _model(client, cs)
    state = cs.get_or_create("rg1_worker")
    state.session_id = "sess-live"
    state.session_live_in_process = True
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    state.pending_session_prompt_future = fut

    await model.asummarize(
        thread_id="rg1_worker",
        messages=[HumanMessage(content="q")],
        instruction="Summarize.",
        decision=_hot_decision(),
    )

    assert client.session_cancel_calls[0] == "sess-live"
    assert fut.cancelled()
    assert [sid for sid, _ in client.prompts] == ["sess-live"]


@pytest.mark.asyncio
async def test_cold_uses_ephemeral_session_without_mcp_and_does_not_load_old() -> None:
    client = _FakeAcpClient()
    cs = AcpClientState()
    model = _model(client, cs)
    state = cs.get_or_create("rg1_worker")
    state.session_id = "f44ee866"
    state.session_live_in_process = False
    state.sent_count = 722

    result = await model.asummarize(
        thread_id="rg1_worker",
        messages=[
            HumanMessage(content="user question 0"),
            AIMessage(content="answer"),
        ],
        instruction="Summarize.",
        decision=HotColdDecision(False, "ttl_expired:14700", 14700.0, 240, None),
    )

    assert result.path == "cold"
    assert client.session_load_calls == []
    assert client.session_new_calls == [[]]
    assert client.prompts[0][0].startswith("sess-new-")
    dump = client.prompts[0][1][0]["text"]
    assert "user question 0" in dump
    agent = cs.get("rg1_worker")
    assert agent is not None
    assert agent.session_id == "f44ee866"
    assert agent.sent_count == 722


@pytest.mark.asyncio
async def test_areset_clears_queued_and_drops_session() -> None:
    client = _FakeAcpClient()
    cs = AcpClientState()
    model = _model(client, cs)
    state = cs.get_or_create("rg1_worker")
    state.session_id = "f44ee866"
    state.sent_count = 722
    state.queued_tool_calls.append({"id": "x", "name": "Bash", "args": {}})
    state.queued_text.append("partial")

    await model.areset_after_summary(thread_id="rg1_worker")

    after = cs.get("rg1_worker")
    assert after is not None
    assert after.session_id is None
    assert after.sent_count == 0
    assert after.queued_tool_calls == []
    assert after.queued_text == []


@pytest.mark.asyncio
async def test_identical_prefix_fails_loud_when_port_lost_on_acp() -> None:
    """Losing ``asummarize`` on a session-stateful provider must RAISE.

    Reviewer blocker 3b (2026-08-21): the identical-prefix fallback
    re-``ainvoke``s the model on the agent's own thread_id. On ACP that
    replays the whole history into a fresh CLI session via the 4b net —
    the dump the port exists to avoid. ``AcpChatModel.bind_tools`` returns
    ``self`` today, so the port survives binding; a wrapper that does not
    preserve it would regress silently. Refuse instead of dumping.
    """
    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        _has_summarizer_port,
        _is_session_stateful,
    )

    class _PortlessAcpWrapper:
        """Looks like ACP to the provenance probe, but has no asummarize."""

        _deep_worker_provider = "claude-code"

        async def ainvoke(self, *a, **kw):  # noqa: ANN002, ANN003, ARG002
            raise AssertionError(
                "ainvoke must NOT be reached — that is the dump path"
            )

    wrapper = _PortlessAcpWrapper()
    assert not _has_summarizer_port(wrapper)
    assert _is_session_stateful(wrapper), "probe failed to classify ACP"

    mw = object.__new__(SummarizationMiddleware)
    mw.model = wrapper

    with pytest.raises(RuntimeError, match="summarization port unavailable"):
        await mw._ainvoke_identical_prefix(
            [HumanMessage(content="hi")], "summarize", "rg1_worker"
        )


@pytest.mark.asyncio
async def test_identical_prefix_still_works_for_stateless_providers() -> None:
    """The guard must not block API providers, which have no session state."""
    from deepagents.middleware.summarization import SummarizationMiddleware

    class _Deepseek:
        _deep_worker_provider = "deepseek"

        async def ainvoke(self, payload, config=None):  # noqa: ARG002
            return AIMessage(content="SUMMARY OK")

    mw = object.__new__(SummarizationMiddleware)
    mw.model = _Deepseek()

    text = await mw._ainvoke_identical_prefix(
        [HumanMessage(content="hi")], "summarize", "some_thread"
    )
    assert text == "SUMMARY OK"


def test_session_stateful_probe_is_an_allow_list() -> None:
    """B2 regression: only real CLI-session provider STRINGS are stateful.

    The first version answered "is it NOT stateless?", which classified
    every unknown provider — and every MagicMock, whose attribute access
    returns a truthy mock — as session-stateful. That made
    ``_ainvoke_identical_prefix`` raise inside seven unrelated
    summarization tests.
    """
    from unittest.mock import MagicMock

    from deepagents.middleware.summarization import _is_session_stateful

    class _P:
        def __init__(self, provider):
            self._deep_worker_provider = provider

    # ``grok`` moved here on 2026-08-24: it sends only
    # ``messages[sent_count:]`` and persists its session id across restarts,
    # which is session-stateful by definition. It had been asserted stateless
    # just below, which pinned the defect rather than catching it.
    for provider in ("claude-code", "claude-code-sdk", "codex", "acp", "grok"):
        assert _is_session_stateful(_P(provider)), provider
    for provider in ("deepseek", "openai", "anthropic", "google"):
        assert not _is_session_stateful(_P(provider)), provider

    # Unknown strings are NOT assumed stateful.
    assert not _is_session_stateful(_P("some-future-api"))
    assert not _is_session_stateful(_P(""))
    assert not _is_session_stateful(_P(None))

    # A bare test double must never look like a CLI session.
    assert not _is_session_stateful(MagicMock())
    assert not _is_session_stateful(object())


def test_stateful_and_stateless_sets_do_not_overlap() -> None:
    """The two classifications must stay mutually exclusive."""
    from deep_worker_cli.context_limit.hot_cache import (
        SESSION_STATEFUL_PROVIDERS,
        STATELESS_PROVIDERS,
    )

    assert not (SESSION_STATEFUL_PROVIDERS & STATELESS_PROVIDERS)
