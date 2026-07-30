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
