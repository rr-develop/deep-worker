import asyncio
import gc
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, RemoveMessage

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.rpyc_service.agent_service import AgentService


@pytest.mark.asyncio
async def test_request_summarization_replaces_checkpoint_once() -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._invoke_lock = None
    process.agent = MagicMock()
    process.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={"messages": [HumanMessage(content="old", id="old")]}
        )
    )
    process.agent.aupdate_state = AsyncMock()
    replacement = [HumanMessage(content="summary", id="summary")]
    middleware = MagicMock()
    middleware.apply_requested_summary = AsyncMock(
        return_value=SimpleNamespace(
            summarized=True,
            replacement_messages=replacement,
            history_path="/conversation_history/summary-agent.jsonl",
        )
    )
    process._find_summarization_middleware = MagicMock(return_value=middleware)

    result = await process.request_summarization("Focus")

    assert result["summarized"] is True
    middleware.apply_requested_summary.assert_awaited_once()
    process.agent.aupdate_state.assert_awaited_once()
    update = process.agent.aupdate_state.await_args.args[1]["messages"]
    assert isinstance(update[0], RemoveMessage)
    assert update[1:] == replacement


@pytest.mark.asyncio
async def test_request_summarization_no_change_does_not_write_checkpoint() -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._invoke_lock = None
    process.agent = MagicMock()
    process.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={"messages": [HumanMessage(content="only", id="only")]}
        )
    )
    process.agent.aupdate_state = AsyncMock()
    middleware = MagicMock()
    middleware.apply_requested_summary = AsyncMock(
        return_value=SimpleNamespace(
            summarized=False,
            replacement_messages=None,
            history_path=None,
        )
    )
    process._find_summarization_middleware = MagicMock(return_value=middleware)

    result = await process.request_summarization()

    assert result["summarized"] is False
    process.agent.aupdate_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_summarization_truncation_only_writes_once() -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._invoke_lock = None
    process.agent = MagicMock()
    process.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={"messages": [HumanMessage(content="old", id="old")]}
        )
    )
    process.agent.aupdate_state = AsyncMock()
    replacement = [HumanMessage(content="truncated", id="truncated")]
    middleware = MagicMock()
    middleware.apply_requested_summary = AsyncMock(
        return_value=SimpleNamespace(
            summarized=False,
            replacement_messages=replacement,
            history_path=None,
        )
    )
    process._find_summarization_middleware = MagicMock(return_value=middleware)

    result = await process.request_summarization()

    assert result["summarized"] is False
    process.agent.aupdate_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_request_summarization_serializes_on_invoke_lock() -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._invoke_lock = asyncio.Lock()
    process.agent = MagicMock()
    process.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(values={"messages": []})
    )
    process.agent.aupdate_state = AsyncMock()
    middleware = MagicMock()
    middleware.apply_requested_summary = AsyncMock(
        return_value=SimpleNamespace(
            summarized=False,
            replacement_messages=None,
            history_path=None,
        )
    )
    process._find_summarization_middleware = MagicMock(return_value=middleware)
    await process._invoke_lock.acquire()

    task = asyncio.create_task(process.request_summarization())
    await asyncio.sleep(0)
    middleware.apply_requested_summary.assert_not_awaited()
    process._invoke_lock.release()
    await task
    middleware.apply_requested_summary.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("model"), OSError("offload")])
async def test_request_summarization_failure_does_not_write_checkpoint(
    error: Exception,
) -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._invoke_lock = None
    process.agent = MagicMock()
    process.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(values={"messages": []})
    )
    process.agent.aupdate_state = AsyncMock()
    middleware = MagicMock()
    middleware.apply_requested_summary = AsyncMock(side_effect=error)
    process._find_summarization_middleware = MagicMock(return_value=middleware)

    with pytest.raises(type(error), match=str(error)):
        await process.request_summarization()
    process.agent.aupdate_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_exposed_summarize_success_and_error_contracts() -> None:
    service = AgentService.__new__(AgentService)
    process = MagicMock()
    process.request_summarization = AsyncMock(
        return_value={"summarized": True, "history_path": "/history.jsonl"}
    )
    service._agent_process = process

    success = json.loads(await service.exposed_summarize("Focus"))
    assert success == {
        "success": True,
        "summarized": True,
        "history_path": "/history.jsonl",
    }
    process.request_summarization = AsyncMock(side_effect=RuntimeError("failed"))
    failure = json.loads(await service.exposed_summarize())
    assert failure == {"success": False, "error": "failed"}


@pytest.mark.asyncio
async def test_exposed_summarize_propagates_cancellation() -> None:
    service = AgentService.__new__(AgentService)
    process = MagicMock()
    process.request_summarization = AsyncMock(side_effect=asyncio.CancelledError)
    service._agent_process = process

    with pytest.raises(asyncio.CancelledError):
        await service.exposed_summarize()


@pytest.mark.asyncio
async def test_startup_summary_is_one_shot_and_failure_is_observed() -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._pending_startup_sum_query = None
    process._pending_startup_query = None
    process._pending_summarize = True
    process._pending_summary_prompt = "Focus"
    process.request_summarization = AsyncMock(side_effect=RuntimeError("boom"))

    task = asyncio.create_task(process._auto_run_startup_query())
    retained = {task}
    task.add_done_callback(retained.discard)
    task.add_done_callback(process._log_startup_task_result)
    del task
    gc.collect()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not retained
    assert process._pending_summarize is False
    assert process._pending_summary_prompt is None
    process.request_summarization.assert_awaited_once_with("Focus")
    await process._auto_run_startup_query()
    process.request_summarization.assert_awaited_once()


def test_requested_summarization_adds_no_polling_or_thread_sites() -> None:
    source = inspect.getsource(AgentProcess.request_summarization)
    for forbidden in ("sleep(", "Thread(", "to_thread(", "while True"):
        assert forbidden not in source


def test_startup_task_uses_detached_strong_ref_pattern() -> None:
    source = inspect.getsource(AgentProcess.start)
    assert "_detached_invoke_tasks.add(startup_task)" in source
    assert "_log_startup_task_result" in source
    assert "_auto_run_startup_query" in source
    assert "create_task" in source


@pytest.mark.asyncio
async def test_second_startup_run_cannot_replay_summary() -> None:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._pending_startup_sum_query = None
    process._pending_startup_query = None
    process._pending_summarize = True
    process._pending_summary_prompt = "Focus"
    process.request_summarization = AsyncMock(
        return_value={"summarized": True, "history_path": None}
    )

    await process._auto_run_startup_query()
    await process._auto_run_startup_query()

    process.request_summarization.assert_awaited_once_with("Focus")
    assert process._pending_summarize is False


# ---------------------------------------------------------------------------
# Status during a requested summarization (owner report, 2026-08-24).
#
# The web UI showed the agent IDLE for the whole duration of a "Summarize
# context" click. Summarization is a real model turn — it can take tens of
# seconds on a cold path — so an idle badge is simply wrong: the operator
# cannot tell the work started, and nothing stops them from firing a message
# into an agent that is already busy.
#
# `request_summarization` never touched `session_manager`, unlike every other
# turn-running path (`_handle_invoke` sets RUNNING at :3915).
# ---------------------------------------------------------------------------


def _status_probe_process(*, raises: BaseException | None = None):
    """An AgentProcess wired so status transitions are observable in order.

    ``raises`` (when set) is thrown from inside the summarization call so the
    test can distinguish RUNNING-set-before-work from RUNNING-set-after.
    """
    from deep_worker_cli.types import AgentStatus

    process = AgentProcess.__new__(AgentProcess)
    process.name = "summary-agent"
    process._invoke_lock = None
    process.agent = MagicMock()
    process.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={"messages": [HumanMessage(content="old", id="old")]}
        )
    )
    process.agent.aupdate_state = AsyncMock()

    transitions: list[tuple[str, Any]] = []
    status_when_summarizing: list[AgentStatus] = []

    session_manager = MagicMock()

    async def _set_status(status, **_kwargs):
        transitions.append(("status", status))

    async def _set_error(message):
        transitions.append(("error", message))

    session_manager.set_status = AsyncMock(side_effect=_set_status)
    session_manager.set_error = AsyncMock(side_effect=_set_error)
    # A starting turn clears the previous turn's recoverable-error marker
    # (docs/DESIGN_RECOVERABLE_ERROR_VISIBILITY.md §4.4). Deliberately not
    # recorded in ``transitions``: that list pins the STATUS sequence, and
    # mixing a non-status call into it would only force the assertions to
    # be rewritten around an unrelated event.
    session_manager.clear_error = AsyncMock()
    process.session_manager = session_manager

    async def _apply(*_args, **_kwargs):
        # Sampled INSIDE the summarization call: this is what the web UI polls
        # while the work is in flight. Setting RUNNING after the fact would
        # leave this list empty and the badge idle for the whole turn.
        status_when_summarizing.extend(
            s for kind, s in transitions if kind == "status"
        )
        if raises is not None:
            raise raises
        return SimpleNamespace(
            summarized=True,
            replacement_messages=[HumanMessage(content="summary", id="summary")],
            history_path=None,
        )

    middleware = MagicMock()
    middleware.apply_requested_summary = AsyncMock(side_effect=_apply)
    process._find_summarization_middleware = MagicMock(return_value=middleware)
    return process, transitions, status_when_summarizing


@pytest.mark.asyncio
async def test_requested_summarization_marks_agent_running_while_working() -> None:
    """RED: the agent must read RUNNING (web: "working") during the turn."""
    from deep_worker_cli.types import AgentStatus

    process, transitions, during = _status_probe_process()

    await process.request_summarization("Focus")

    assert during == [AgentStatus.RUNNING], (
        "agent was not RUNNING while the summarization was in flight; the web "
        f"UI shows it idle for the whole turn (observed: {during})"
    )
    assert transitions == [
        ("status", AgentStatus.RUNNING),
        ("status", AgentStatus.IDLE),
    ], f"expected RUNNING then IDLE, got {transitions}"


@pytest.mark.asyncio
async def test_requested_summarization_fatal_failure_settles_in_error() -> None:
    """POLICIES §12: a fatal summarization failure MUST be loud, not IDLE.

    The requested path used to swallow ``SummarizationFailedError`` into a
    plain ``{success:false}`` and leave the badge at IDLE — exactly the
    silent-failure surface the automatic path was hardened against on
    2026-08-23. Route through ``set_error`` (ERROR + visible message) so the
    operator sees WHAT died, not just "the click did nothing".
    """
    from deep_worker_cli.types import AgentStatus
    from deepagents.middleware.summarization import SummarizationFailedError

    fatal = SummarizationFailedError("summary provider unavailable")
    process, transitions, _during = _status_probe_process(raises=fatal)

    with pytest.raises(SummarizationFailedError):
        await process.request_summarization()

    assert transitions[0] == ("status", AgentStatus.RUNNING)
    assert transitions[-1][0] == "error", (
        "fatal summarization failure did not land in ERROR; the agent is "
        f"back to IDLE with no visible failure signal (got {transitions})"
    )
    error_msg = transitions[-1][1]
    assert "summary provider unavailable" in error_msg, (
        f"error message must preserve the real cause, got: {error_msg!r}"
    )
    # RUNNING was cleared exactly once — the ERROR terminal transition IS
    # that clearing (set_error owns the same status slot as set_status).
    status_calls = [t for t in transitions if t[0] == "status"]
    assert status_calls == [("status", AgentStatus.RUNNING)], (
        f"IDLE must NOT be pinned for the fatal case, got {status_calls}"
    )


@pytest.mark.asyncio
async def test_requested_summarization_nonfatal_failure_returns_to_idle() -> None:
    """Non-fatal transient errors go back to IDLE (not sticky ERROR)."""
    from deep_worker_cli.types import AgentStatus

    # A bare ``RuntimeError`` is NOT in FATAL_EXCEPTION_TYPES, so the requested
    # path treats it as recoverable — the agent's turn state is intact and
    # the operator can just click Summarize again.
    process, transitions, _during = _status_probe_process(
        raises=RuntimeError("transient hiccup")
    )

    with pytest.raises(RuntimeError, match="transient hiccup"):
        await process.request_summarization()

    assert transitions == [
        ("status", AgentStatus.RUNNING),
        ("status", AgentStatus.IDLE),
    ], (
        "non-fatal failure must clear RUNNING and return to IDLE; ERROR is "
        f"reserved for fatal classes (got {transitions})"
    )


@pytest.mark.asyncio
async def test_requested_summarization_without_session_manager_still_works() -> None:
    """`session_manager` is optional on this path — must not become required."""
    process, _transitions, _during = _status_probe_process()
    process.session_manager = None

    result = await process.request_summarization()

    assert result["summarized"] is True
