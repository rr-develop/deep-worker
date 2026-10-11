"""The ACP chat model stamps the CLI's live context gauge on its AIMessages.

rg1_worker 2026-08-19: ``usage_update`` said 468k, the LangChain heuristic
said ~150k, ``-l 400k`` was never honoured. The provider is the only party
that knows the real number; it must hand it to the summarizer/guard via
``response_metadata`` (persisted, restart-safe). These tests pin:

* the final AIMessage of a turn carries ``{used, size, first_id}`` taken
  from ``state.usage_snapshot`` BEFORE the turn-boundary reset;
* a mid-turn ``AIMessage(tool_calls=...)`` carries it too (the CLI reports
  usage after every model call, not only at turn end);
* no snapshot → no stamp (heuristic path stays intact);
* ``first_id`` is the first NON-System message id of the request — the
  validity key the consumer compares against.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import NOTIF_QUEUE_MAX_ITEMS, ThreadState
from deep_worker.providers.acp.profile import AcpProfile
from deep_worker_cli.context_limit.observed_context import (
    CONTEXT_OBSERVATION_KEY,
    context_observation,
    first_state_message_id,
)


def _model() -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._acp_client = MagicMock()
    model._acp_client.session_cancel = AsyncMock()
    model._profile = AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
    )
    model._client_state = AcpClientState()
    model._proxy_mcp = None
    model._tools_metadata = []
    model._usage_event_callback = None
    model._record_resolved_model = AsyncMock()
    return model


def _state(*, used: int | None, first_id: str | None = "h0") -> ThreadState:
    st = ThreadState()
    st.session_id = "s1"
    st.context_first_id = first_id
    if used is not None:
        st.usage_snapshot = {"used": used, "size": 1_000_000}
    return st


class TestFinalMessageStamp:
    def test_final_message_carries_snapshot_and_first_id(self) -> None:
        model = _model()
        state = _state(used=468_146)
        reply = {"result": {"stopReason": "end_turn"}}

        msg = asyncio.run(model._finalize_response(reply, ["hello"], state))

        assert isinstance(msg, AIMessage)
        assert msg.content == "hello"
        assert msg.response_metadata[CONTEXT_OBSERVATION_KEY] == {
            "used": 468_146,
            "size": 1_000_000,
            "first_id": "h0",
        }
        # Turn-boundary reset still happens AFTER the stamp.
        assert state.usage_snapshot is None

    def test_no_snapshot_no_stamp(self) -> None:
        model = _model()
        state = _state(used=None)
        msg = asyncio.run(
            model._finalize_response({"result": {"stopReason": "end_turn"}}, ["x"], state)
        )
        assert CONTEXT_OBSERVATION_KEY not in msg.response_metadata
        assert context_observation(msg) is None

    def test_meta_usage_path_still_stamps_snapshot(self) -> None:
        """When the reply carries per-turn ``_meta`` token counts the
        snapshot is still the context gauge — stamp it before the reset."""
        model = _model()
        state = _state(used=250_000)
        reply = {
            "result": {
                "stopReason": "end_turn",
                "_meta": {"inputTokens": 1000, "outputTokens": 50},
            }
        }
        msg = asyncio.run(model._finalize_response(reply, ["x"], state))
        assert context_observation(msg) == {
            "used": 250_000,
            "size": 1_000_000,
            "first_id": "h0",
        }
        assert state.usage_snapshot is None

    def test_zero_or_garbage_used_is_not_stamped(self) -> None:
        model = _model()
        state = _state(used=None)
        state.usage_snapshot = {"used": 0, "size": 1_000_000}
        msg = asyncio.run(
            model._finalize_response({"result": {"stopReason": "end_turn"}}, ["x"], state)
        )
        assert CONTEXT_OBSERVATION_KEY not in msg.response_metadata


class TestMidTurnStamp:
    def test_tool_call_message_carries_snapshot(self) -> None:
        """A pump-queued tool_call surfaces as AIMessage(tool_calls) while
        the response future is still in flight — it is stamped too."""

        async def _run() -> None:
            model = _model()
            state = _state(used=300_000)
            state.current_notif_queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
            state.pending_session_prompt_future = asyncio.get_running_loop().create_future()
            state.queued_tool_calls = [
                {"id": "tc1", "name": "Bash", "args": {"command": "ls"}, "type": "tool_call"}
            ]
            model._correlate_proxy_tool_calls = AsyncMock()

            msg = await model._consume_turn(state, "agent-1")
            assert msg.tool_calls and msg.tool_calls[0]["id"] == "tc1"
            assert context_observation(msg) == {
                "used": 300_000,
                "size": 1_000_000,
                "first_id": "h0",
            }
            # Mid-turn: snapshot is NOT reset (turn still running).
            assert state.usage_snapshot == {"used": 300_000, "size": 1_000_000}
            # Leave no dangling future for the loop teardown.
            state.pending_session_prompt_future.cancel()

        asyncio.run(_run())


class TestFirstIdContract:
    def test_first_id_is_first_non_system_message(self) -> None:
        """Pin the producer/consumer agreement: the factory system prompt
        and a guard hint are skipped; the first STATE message identifies
        the history."""
        msgs = [
            SystemMessage(content="factory", id="sys"),
            SystemMessage(content="[context-guard] hint", id="hint"),
            HumanMessage(content="q", id="h0"),
            AIMessage(content="a", id="a0"),
        ]
        assert first_state_message_id(msgs) == "h0"

    def test_stamp_uses_state_context_first_id(self) -> None:
        model = _model()
        state = _state(used=10_000, first_id="abc-123")
        msg = asyncio.run(
            model._finalize_response({"result": {"stopReason": "end_turn"}}, ["x"], state)
        )
        assert context_observation(msg)["first_id"] == "abc-123"


# ── End-to-end through ainvoke (fake CLI client) ─────────────────────────


class _FakeAcpClient:
    """Fake CLI: emits one ``usage_update`` then ends the turn."""

    def __init__(self, used: int, size: int) -> None:
        self._used = used
        self._size = size
        self.prompts: list[list[dict]] = []

    async def ensure_started(self) -> None:
        return None

    async def session_new(self, cwd: str, mcp_servers: list) -> str:  # noqa: ARG002
        return "sess-1"

    async def wait_session_ready(self, session_id: str) -> None:  # noqa: ARG002
        return None

    def session_prompt(self, session_id: str, prompt: list, notification_sink):  # noqa: ARG002
        self.prompts.append(prompt)
        notification_sink.put_nowait(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "usage_update",
                        "used": self._used,
                        "size": self._size,
                    },
                },
            }
        )
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result({"result": {"stopReason": "end_turn"}})
        return fut


def _e2e_model(client: _FakeAcpClient) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._acp_client = client
    model._client_state = AcpClientState()
    model._proxy_mcp = None
    model._proxy_started = True
    model._acp_started = True
    model._tools_metadata = []
    model._working_dir = "/tmp"
    model._profile = AcpProfile(provider="claude-code", cmd=("true",))
    model._stats_callback = None
    model._usage_event_callback = None
    return model


def test_ainvoke_returns_stamped_ai_message_with_real_first_id() -> None:
    """The full producer path: ``_agenerate`` captures the first state
    message id, the CLI reports ``usage_update``, the returned AIMessage
    carries ``{used, size, first_id}`` — exactly what the summarizer's
    token_counter will read back on the next turn."""

    async def _run() -> None:
        client = _FakeAcpClient(used=468_146, size=1_000_000)
        model = _e2e_model(client)
        messages = [
            SystemMessage(content="factory prompt", id="sys-0"),
            HumanMessage(content="first question", id="h-first"),
            AIMessage(content="first answer", id="a-first"),
            HumanMessage(content="latest question", id="h-latest"),
        ]
        reply = await model.ainvoke(
            messages, config={"configurable": {"thread_id": "stamp-thread"}}
        )
        assert isinstance(reply, AIMessage)
        assert context_observation(reply) == {
            "used": 468_146,
            "size": 1_000_000,
            "first_id": "h-first",
        }

    asyncio.run(_run())
