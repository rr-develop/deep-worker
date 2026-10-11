"""Unit tests for bounded/fail-loud state and the EOF recoverable-crash
transition (DESIGN §4.5 + §5.3, review items 4 & 5).

Covers:
  * a turn-level overflow propagating out of ``_consume_turn`` (no hang) —
    the notification sink itself no longer raises it (it applies
    backpressure; see ``test_notification_backpressure.py``), so this pins
    the consumer's reaction to a failed response future,
  * ``_pending_requests`` reject-new cap,
  * chat-model tool_call accumulator overflow → fail-loud via _fail_turn,
  * refine-pending reject-new cap,
  * forwarded-extra-system-keys reject-new cap,
  * ``_on_eof``: every waiter failed, all maps cleared, client_state reset
    (builtin passthrough futures + session liveness).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker.providers.acp.acp_client import (
    AcpClient,
    AcpClientError,
    AcpSinkOverflowError,
)
from deep_worker.providers.acp.chat_model import (
    _collect_runtime_system_chunks,
)
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import (
    FORWARDED_EXTRA_KEYS_MAX,
    MAX_PENDING_REQUESTS,
    NOTIF_QUEUE_MAX_ITEMS,
    QUEUED_TOOL_CALLS_MAX,
    REFINE_PENDING_MAX,
    ThreadState,
)
from deep_worker.providers.acp.profile import AcpProfile


def _client(*, client_state: AcpClientState | None = None) -> AcpClient:
    client = AcpClient(cmd=("true",), client_state=client_state)
    client._write = MagicMock()
    return client


def _model() -> "object":
    """AcpChatModel WITHOUT constructor side effects; only fields used by
    _append_tool_call_bounded / _fail_turn / _consume_turn are set."""
    from deep_worker.providers.acp.chat_model import AcpChatModel

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
    return model


def _sink_notif(session_id: str, update: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": update or {"sessionUpdate": "agent_message_chunk", "content": {"text": "x"}},
        },
    }


class TestSinkOverflowFailLoud:
    def test_overflow_does_not_hang_consume_turn(self) -> None:
        """End-to-end-ish: a turn whose response future carries
        AcpSinkOverflowError raises out of _consume_turn immediately instead
        of blocking forever on an empty notification queue."""
        async def _run() -> None:
            model = _model()
            state = ThreadState()
            state.session_id = "s1"
            state.current_notif_queue = asyncio.Queue(maxsize=NOTIF_QUEUE_MAX_ITEMS)
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            fut.set_exception(
                AcpSinkOverflowError(f"ACP notification sink overflowed for 's1'")
            )
            state.pending_session_prompt_future = fut

            with pytest.raises(AcpSinkOverflowError):
                await model._consume_turn(state, "agent-1")
            # Turn cleanup: future released, pump cancelled, CLI cancel called.
            assert state.pending_session_prompt_future is None
            model._acp_client.session_cancel.assert_awaited_once()

        asyncio.run(_run())


class TestPendingRequestsCap:
    def test_reject_new_at_capacity(self) -> None:
        async def _run() -> None:
            client = _client()
            for i in range(MAX_PENDING_REQUESTS):
                fut: asyncio.Future = asyncio.get_running_loop().create_future()
                fut.set_result({"id": i, "result": {}})
                client._pending_requests[i] = fut
            with pytest.raises(AcpClientError, match="too many in-flight"):
                await client._call("session/new", {"cwd": "."})

        asyncio.run(_run())


class TestToolCallAccumulatorOverflow:
    def test_overflow_fails_turn(self) -> None:
        async def _run() -> None:
            model = _model()
            state = ThreadState()
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            state.pending_session_prompt_future = fut
            acc: list[dict] = [{"id": f"t{i}"} for i in range(QUEUED_TOOL_CALLS_MAX)]

            model._append_tool_call_bounded(state, acc, {"id": "overflow"})

            assert len(acc) == QUEUED_TOOL_CALLS_MAX  # not appended
            assert fut.done()
            with pytest.raises(AcpSinkOverflowError):
                fut.result()

        asyncio.run(_run())


class TestRefinePendingCap:
    def test_reject_new_at_capacity(self) -> None:
        async def _run() -> None:
            model = _model()
            state = ThreadState()
            state.refine_pending_tool_calls = {
                f"t{i}" for i in range(REFINE_PENDING_MAX)
            }
            upd = {
                "sessionUpdate": "tool_call",
                "toolCallId": "overflow",
                "title": "Bash",
                "rawInput": {},
                "status": "pending",
                "_meta": {"claudeCode": {"toolName": "Bash"}},
            }
            model._apply_update(
                state, "agent-1", _sink_notif("s1", upd), [], []
            )
            # reject-new: the hold was NOT added, so nothing is held back.
            assert "overflow" not in state.refine_pending_tool_calls

        asyncio.run(_run())


class TestForwardedExtraKeysCap:
    def test_reject_new_at_capacity(self) -> None:
        from langchain_core.messages import SystemMessage

        async def _run() -> None:
            base_keys = {f"k{i}" for i in range(FORWARDED_EXTRA_KEYS_MAX)}
            new_msg = SystemMessage(content="fresh runtime hint", id="fresh-1")
            chunks, _first_chars, new_keys = _collect_runtime_system_chunks(
                [SystemMessage(content="first", id="first")],
                [new_msg],
                is_first_prompt_in_session=False,
                forwarded_first_system_chars=5,
                forwarded_extra_system_keys=base_keys,
            )
            # reject-new: key not deduplicated AND chunk not emitted again.
            assert "fresh-1" not in new_keys
            assert chunks == []

        asyncio.run(_run())


class TestEofRecoverableCrash:
    def test_eof_fails_all_waiters_and_clears_state(self) -> None:
        async def _run() -> None:
            cs = AcpClientState()
            cs.set_profile("claude-code")
            st = cs.get_or_create("agent-1")
            st.session_live_in_process = True
            st.builtin_passthrough_futures["t1"] = (
                asyncio.get_running_loop().create_future()
            )

            client = _client(client_state=cs)
            req_fut = asyncio.get_running_loop().create_future()
            client._pending_requests[1] = req_fut
            prompt_fut = asyncio.get_running_loop().create_future()
            client._active_prompts["s1"] = prompt_fut
            ready_fut = asyncio.get_running_loop().create_future()
            client._session_ready["s1"] = ready_fut
            client._sinks["s1"] = asyncio.Queue()
            proc = MagicMock()
            client._proc = proc
            client._reader_task = MagicMock()

            client._on_eof()

            # Every waiter failed with the same terminal error.
            for fut in (req_fut, prompt_fut, ready_fut):
                assert fut.done()
                with pytest.raises(AcpClientError, match="EOF"):
                    fut.result()
            # All client maps cleared.
            assert not client._pending_requests
            assert not client._active_prompts
            assert not client._session_ready
            assert not client._sinks
            assert client._proc is None
            assert client._reader_task is None
            # Thread state: passthrough waiter failed, session no longer live.
            assert st.builtin_passthrough_futures == {}
            assert not st.session_live_in_process

        asyncio.run(_run())

    def test_eof_during_initialize_fails_inflight_call(self) -> None:
        """Phase: EOF while a JSON-RPC request is in flight (initialize).
        The caller's ``_call`` raises instead of hanging on a reply that can
        never arrive."""
        async def _run() -> None:
            client = _client()
            call_task = asyncio.ensure_future(client._call("initialize"))
            # The request future is registered; let the loop spin once.
            await asyncio.sleep(0)
            assert len(client._pending_requests) == 1
            client._on_eof()
            with pytest.raises(AcpClientError, match="EOF"):
                await call_task
            assert not client._pending_requests

        asyncio.run(_run())

    def test_eof_during_readiness_wait_fails_wait_session_ready(self) -> None:
        """Phase: EOF while ``wait_session_ready`` awaits the one-shot
        readiness future. The wait raises immediately (no hang, no timeout
        fallback silently continuing)."""
        async def _run() -> None:
            client = _client()
            client._session_ready["s1"] = asyncio.get_running_loop().create_future()
            wait_task = asyncio.ensure_future(client.wait_session_ready("s1"))
            await asyncio.sleep(0)
            client._on_eof()
            with pytest.raises(AcpClientError, match="EOF"):
                await wait_task

        asyncio.run(_run())

    def test_eof_during_active_prompt_fails_response_future(self) -> None:
        """Phase: EOF while a ``session/prompt`` is outstanding. The response
        future carries the terminal error, so the turn raises instead of
        hanging."""
        async def _run() -> None:
            client = _client()
            fut = client.session_prompt(
                session_id="s1",
                prompt=[{"type": "text", "text": "hi"}],
                notification_sink=asyncio.Queue(),
            )
            assert not fut.done()
            client._on_eof()
            with pytest.raises(AcpClientError, match="EOF"):
                fut.result()
            assert not client._active_prompts
            assert not client._sinks

        asyncio.run(_run())

    def test_eof_between_turns_leaves_client_restartable(self) -> None:
        """Phase: EOF between turns (no active prompt). The client is fully
        reset: process/reader dropped, maps empty — a subsequent ``start()``
        re-spawns a fresh CLI subprocess."""
        async def _run() -> None:
            client = _client()
            # Emulate a previously running subprocess.
            client._proc = MagicMock()
            client._reader_task = MagicMock()
            client._on_eof()
            assert client._proc is None
            assert client._reader_task is None
            assert not client._pending_requests
            assert not client._sinks
            assert not client._active_prompts
            assert not client._session_ready
            # Restart path: start() is willing to create a NEW subprocess.
            assert client._closed is False

        asyncio.run(_run())
