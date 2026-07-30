"""Focused unit tests for the AgentProcess startup-query hook.

See ``deep_worker/deep_worker_cli/docs/DESIGN_STARTUP_QUERY_OPTIONS.md`` §6.3.

These call the unbound methods with a ``MagicMock`` ``fake_self`` (the same
pattern used by ``test_restart_notice_delivery.py``) so we don't stand up a
real AgentProcess. The compiled ``agent`` is mocked (``aget_state`` /
``aupdate_state``); the SummarizationMiddleware is real (backend = a mock
BackendProtocol) so the offload path exercises production code.

Contract under test:

  * ``_run_startup_or_send_query`` reads current_messages via aget_state,
    calls apply_pseudo_summary(self.name, ...), then wipes via aupdate_state
    with RemoveMessage(REMOVE_ALL_MESSAGES) BEFORE any _handle_invoke
    (Blocker-2). Offload failure skips the wipe (atomicity).
  * sum_query + query are TWO SEQUENTIAL turns (bug-fix 2026-07-04): turn 1
    executes the sum-query (empty message_list over the installed
    summary-shape state), turn 2 executes the plain query as the next user
    message. The summary-shape message carries lc_source=summarization and
    the plain query does not (Gap-5).
  * ``_auto_run_startup_query`` clears the pending fields (one-shot) and is a
    no-op when neither is set.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage

from deep_worker_cli.process.agent_process import AgentProcess
from deepagents.backends.protocol import (  # type: ignore[import-not-found]
    BackendProtocol,
    FileDownloadResponse,
    WriteResult,
)
from deepagents.middleware.summarization import (
    HistoryOffloadError,
    SummarizationMiddleware,
)


class MockBackend(BackendProtocol):
    """Minimal BackendProtocol recording writes; can simulate failure."""

    def __init__(
        self, *, should_fail: bool = False, error_message: str | None = None
    ) -> None:
        self.write_calls: list[tuple[str, str]] = []
        self.edit_calls: list[tuple[str, str, str]] = []
        self.should_fail = should_fail
        self.error_message = error_message

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        return ""

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        return ""

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [
            FileDownloadResponse(path=p, content=None, error="file_not_found")
            for p in paths
        ]

    async def adownload_files(
        self, paths: list[str]
    ) -> list[FileDownloadResponse]:
        return self.download_files(paths)

    def write(self, path: str, content: str) -> WriteResult:
        self.write_calls.append((path, content))
        if self.should_fail:
            return WriteResult(error=self.error_message or "mock failure")
        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        return self.write(path, content)

    def edit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ):
        self.edit_calls.append((path, old_string, new_string))
        if self.should_fail:
            from deepagents.backends.protocol import EditResult

            return EditResult(error=self.error_message or "mock failure")
        from deepagents.backends.protocol import EditResult

        return EditResult(path=path, occurrences=1)

    async def aedit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ):
        return self.edit(path, old_string, new_string, replace_all)


def make_conversation_messages(
    num_old: int = 4, num_recent: int = 0
) -> list[AnyMessage]:
    return [
        HumanMessage(content=f"m{i}", id=f"h{i}") for i in range(num_old + num_recent)
    ]


def make_mock_model() -> MagicMock:
    model = MagicMock()
    model.invoke.return_value = MagicMock(text="summary")
    model._llm_type = "test-model"
    model.profile = {"max_input_tokens": 100000}
    model._get_ls_params.return_value = {"ls_provider": "test"}
    return model


def _real_middleware(backend: Any) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=make_mock_model(),
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
        history_path_prefix="/conversation_history",
    )


def _make_fake_self(
    *,
    backend: Any,
    current_messages: list[AnyMessage] | None = None,
) -> MagicMock:
    """Build a fake AgentProcess with a real middleware + mocked graph."""
    import asyncio

    fake = MagicMock()
    fake.name = "ag7"
    # Real lock so the offload→wipe→invoke critical section behaves as in prod.
    fake._invoke_lock = asyncio.Lock()

    mw = _real_middleware(backend)
    fake._find_summarization_middleware = MagicMock(return_value=mw)

    # Mock compiled graph: aget_state returns a state carrying current_messages;
    # aupdate_state records the wipe payload.
    state = MagicMock()
    state.values = {"messages": current_messages or []}
    fake.agent = MagicMock()
    fake.agent.aget_state = AsyncMock(return_value=state)
    fake.agent.aupdate_state = AsyncMock(return_value=None)

    # The sequence now calls _handle_invoke_owned DIRECTLY (it already holds
    # _invoke_lock). Capture its command.
    fake._handle_invoke_owned = AsyncMock(return_value={"success": True})
    return fake


@pytest.mark.asyncio
async def test_run_startup_query_query_only_appends_human_message() -> None:
    backend = MockBackend()
    fake = _make_fake_self(backend=backend)

    await AgentProcess._run_startup_or_send_query(fake, None, "do the thing")

    # No offload / no wipe for query-only.
    fake.agent.aupdate_state.assert_not_awaited()
    assert not backend.write_calls
    # Single invoke with a message_list carrying the plain query.
    fake._handle_invoke_owned.assert_awaited_once()
    cmd = fake._handle_invoke_owned.await_args.args[0]
    msgs = cmd["message_list"]
    assert len(msgs) == 1
    assert msgs[0].content == "do the thing"
    assert msgs[0].additional_kwargs.get("lc_source") != "summarization"


@pytest.mark.asyncio
async def test_run_startup_query_sum_query_only_wipes_and_invokes() -> None:
    backend = MockBackend()
    current = make_conversation_messages(num_old=4, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    await AgentProcess._run_startup_or_send_query(fake, "fresh start", None)

    # Offload happened to <agent>.jsonl.
    assert backend.write_calls
    assert backend.write_calls[0][0] == "/conversation_history/ag7.jsonl"
    # Wipe via aupdate_state with RemoveMessage(REMOVE_ALL_MESSAGES) + summary.
    fake.agent.aupdate_state.assert_awaited_once()
    update_payload = fake.agent.aupdate_state.await_args.args[1]
    wipe_msgs = update_payload["messages"]
    assert isinstance(wipe_msgs[0], RemoveMessage)
    assert wipe_msgs[1].additional_kwargs.get("lc_source") == "summarization"
    # Invoke had an empty message_list (drive from installed state).
    cmd = fake._handle_invoke_owned.await_args.args[0]
    assert cmd["message_list"] == []


@pytest.mark.asyncio
async def test_run_startup_query_both_two_sequential_turns() -> None:
    """CONTRACT (DESIGN §4.6, user requirement): sum_query + query are TWO
    SEQUENTIAL user requests — "first comes the sum-query request, then
    immediately the query request". The agent EXECUTES the sum-query first (its own turn
    on the installed summary-shape state), then EXECUTES the plain query as
    the next user message. NOT a single turn with [summary-shape, query]
    (the old behaviour never ran the sum-query as its own request — bug
    2026-07-04)."""
    backend = MockBackend()
    current = make_conversation_messages(num_old=4, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    await AgentProcess._run_startup_or_send_query(fake, "summary text", "plain q")

    # Wipe installed exactly [summary-shape] once (before either turn).
    fake.agent.aupdate_state.assert_awaited_once()
    installed = fake.agent.aupdate_state.await_args.args[1]["messages"]
    assert isinstance(installed[0], RemoveMessage)
    assert installed[1].additional_kwargs.get("lc_source") == "summarization"

    # TWO invoke turns, in order.
    assert fake._handle_invoke_owned.await_count == 2
    turn1_msgs = fake._handle_invoke_owned.await_args_list[0].args[0]["message_list"]
    turn2_msgs = fake._handle_invoke_owned.await_args_list[1].args[0]["message_list"]

    # Turn 1 executes the sum-query: empty message_list re-drives the graph on
    # the freshly-installed [summary-shape] state.
    assert turn1_msgs == []
    # Turn 2 executes the plain query as the next user message.
    assert len(turn2_msgs) == 1
    assert turn2_msgs[0].content == "plain q"
    assert turn2_msgs[0].additional_kwargs.get("lc_source") != "summarization"


@pytest.mark.asyncio
async def test_run_startup_query_both_wipe_precedes_both_turns() -> None:
    """Ordering: the offload+wipe must happen BEFORE turn 1 (sum-query),
    which must happen BEFORE turn 2 (query). Structural, not timing-based."""
    backend = MockBackend()
    current = make_conversation_messages(num_old=4, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    events: list[str] = []

    async def _rec_update(*a: Any, **k: Any) -> None:
        events.append("wipe")

    async def _rec_owned(cmd: Any, **k: Any) -> dict:
        ml = cmd["message_list"]
        events.append("turn:sum" if ml == [] else f"turn:query:{ml[0].content}")
        return {"success": True}

    fake.agent.aupdate_state = AsyncMock(side_effect=_rec_update)
    fake._handle_invoke_owned = AsyncMock(side_effect=_rec_owned)

    await AgentProcess._run_startup_or_send_query(fake, "summary text", "plain q")

    assert events == ["wipe", "turn:sum", "turn:query:plain q"]


@pytest.mark.asyncio
async def test_first_message_is_summary_shape_second_is_plain() -> None:
    """Gap-5: across the wipe payload + invoke input, the summary-shape message
    carries lc_source=summarization and the plain query does NOT — so a later
    real summarization won't filter the query via _is_summary_message."""
    backend = MockBackend()
    current = make_conversation_messages(num_old=4, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    await AgentProcess._run_startup_or_send_query(fake, "summary text", "plain q")

    # Summary-shape is in the wipe payload (installed before turn 1).
    summary_msg = fake.agent.aupdate_state.await_args.args[1]["messages"][1]
    # Plain query is the sole message of turn 2 (the SECOND invoke).
    plain_msg = fake._handle_invoke_owned.await_args_list[1].args[0]["message_list"][0]
    assert summary_msg.additional_kwargs.get("lc_source") == "summarization"
    assert "lc_source" not in plain_msg.additional_kwargs


@pytest.mark.asyncio
async def test_offload_failure_skips_wipe() -> None:
    """Blocker-2 atomicity: HistoryOffloadError from the offload → no wipe,
    error propagates (caller/catch-all handles ERROR transition)."""
    backend = MockBackend(should_fail=True, error_message="disk full")
    current = make_conversation_messages(num_old=4, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    with pytest.raises(HistoryOffloadError):
        await AgentProcess._run_startup_or_send_query(fake, "s", "q")

    fake.agent.aupdate_state.assert_not_awaited()
    fake._handle_invoke_owned.assert_not_awaited()


@pytest.mark.asyncio
async def test_reads_current_messages_from_aget_state() -> None:
    backend = MockBackend()
    current = make_conversation_messages(num_old=6, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    await AgentProcess._run_startup_or_send_query(fake, "s", None)

    fake.agent.aget_state.assert_awaited_once()
    # thread_id in the config passed to aget_state is the agent name.
    cfg = fake.agent.aget_state.await_args.args[0]
    assert cfg["configurable"]["thread_id"] == "ag7"


@pytest.mark.asyncio
async def test_offload_wipe_invoke_serialised_under_invoke_lock() -> None:
    """BLOCKER fix: the whole offload→wipe→invoke sequence holds _invoke_lock,
    so a concurrent normal turn (which also acquires _invoke_lock) cannot
    interleave with the wipe. We prove serialisation: while
    _run_startup_or_send_query holds the lock, a competing coroutine that
    tries to acquire the SAME lock is blocked until the sequence finishes."""
    import asyncio

    backend = MockBackend()
    current = make_conversation_messages(num_old=4, num_recent=0)
    fake = _make_fake_self(backend=backend, current_messages=current)

    events: list[str] = []

    async def _slow_owned(cmd: Any, **kwargs: Any) -> dict:
        # Simulate a turn that takes a moment while holding the lock.
        events.append("startup:invoke-begin")
        await asyncio.sleep(0.05)
        events.append("startup:invoke-end")
        return {"success": True}

    fake._handle_invoke_owned = AsyncMock(side_effect=_slow_owned)

    async def _competing_turn() -> None:
        # A normal invoke serialises on the same lock. It must not run until
        # the startup sequence releases the lock.
        await asyncio.sleep(0.01)  # ensure startup grabs the lock first
        events.append("competing:want-lock")
        async with fake._invoke_lock:
            events.append("competing:got-lock")

    await asyncio.gather(
        AgentProcess._run_startup_or_send_query(fake, "s", "q"),
        _competing_turn(),
    )

    # The competing turn wanted the lock DURING the startup invoke, but only
    # got it AFTER the startup sequence finished — i.e. no interleave.
    assert "competing:want-lock" in events
    assert events.index("competing:got-lock") > events.index("startup:invoke-end")
    # aupdate_state (wipe) happened before the competing turn ran.
    fake.agent.aupdate_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_run_startup_query_clears_pending_fields() -> None:
    fake = MagicMock()
    fake.name = "ag7"
    fake._pending_startup_sum_query = "s"
    fake._pending_startup_query = "q"
    fake._run_startup_or_send_query = AsyncMock(return_value=None)

    await AgentProcess._auto_run_startup_query(fake)

    assert fake._pending_startup_sum_query is None
    assert fake._pending_startup_query is None
    fake._run_startup_or_send_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_run_startup_query_no_op_when_neither_set() -> None:
    fake = MagicMock()
    fake.name = "ag7"
    fake._pending_startup_sum_query = None
    fake._pending_startup_query = None
    fake._run_startup_or_send_query = AsyncMock(return_value=None)

    await AgentProcess._auto_run_startup_query(fake)

    fake._run_startup_or_send_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_run_startup_query_swallows_non_fatal_errors() -> None:
    """A failing startup query must not crash the process — the coroutine
    logs and returns (the agent stays alive; user can retry)."""
    fake = MagicMock()
    fake.name = "ag7"
    fake._pending_startup_sum_query = None
    fake._pending_startup_query = "q"
    fake._run_startup_or_send_query = AsyncMock(side_effect=RuntimeError("boom"))

    # Must not raise.
    await AgentProcess._auto_run_startup_query(fake)
