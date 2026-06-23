from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.process.agent_process import AgentProcess


def _process_with_blocking_agent() -> tuple[AgentProcess, asyncio.Event, list[str]]:
    process = AgentProcess.__new__(AgentProcess)
    process.name = "serialized-agent"
    process.config = SimpleNamespace(working_dir="/tmp")
    process.session_manager = None
    process.task_manager = None
    process.is_running = False
    process.current_worker_task = None
    process._last_broadcasted_count = {}
    process._pending_restart_notice = None

    release_first = asyncio.Event()
    entered: list[str] = []

    async def ainvoke(payload, _config):
        message = payload["messages"][0].content
        entered.append(message)
        if message == "first":
            await release_first.wait()
        return {
            "messages": [
                HumanMessage(content=message),
                AIMessage(content=f"done:{message}"),
            ]
        }

    process.agent = SimpleNamespace(ainvoke=ainvoke)
    return process, release_first, entered


@pytest.mark.asyncio
async def test_top_level_invocations_are_serialized() -> None:
    process, release_first, entered = _process_with_blocking_agent()

    first = asyncio.create_task(
        process._handle_invoke(
            {"message": "first", "stream": False},
            notify_callback=AsyncMock(),
        )
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        process._handle_invoke(
            {"message": "second", "stream": False},
            notify_callback=AsyncMock(),
        )
    )
    await asyncio.sleep(0.05)

    assert entered == ["first"]
    assert process.current_worker_task is first

    release_first.set()
    first_result = await asyncio.wait_for(first, timeout=1)
    second_result = await asyncio.wait_for(second, timeout=1)

    assert first_result["success"] is True
    assert second_result["success"] is True
    assert entered == ["first", "second"]
    assert process.current_worker_task is None
    assert process.is_running is False


@pytest.mark.asyncio
async def test_cancelling_active_owner_releases_next_invocation() -> None:
    process, _release_first, entered = _process_with_blocking_agent()

    first = asyncio.create_task(
        process._handle_invoke(
            {"message": "first", "stream": False},
            notify_callback=AsyncMock(),
        )
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        process._handle_invoke(
            {"message": "second", "stream": False},
            notify_callback=AsyncMock(),
        )
    )
    await asyncio.sleep(0.05)

    assert process.current_worker_task is first
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second_result = await asyncio.wait_for(second, timeout=1)
    assert second_result["success"] is True
    assert entered == ["first", "second"]
    assert process.current_worker_task is None


@pytest.mark.asyncio
async def test_cancelling_queued_waiter_does_not_affect_active_owner() -> None:
    process, release_first, entered = _process_with_blocking_agent()

    first = asyncio.create_task(
        process._handle_invoke(
            {"message": "first", "stream": False},
            notify_callback=AsyncMock(),
        )
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        process._handle_invoke(
            {"message": "second", "stream": False},
            notify_callback=AsyncMock(),
        )
    )
    await asyncio.sleep(0.05)

    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert entered == ["first"]
    assert process.current_worker_task is first

    release_first.set()
    first_result = await asyncio.wait_for(first, timeout=1)
    assert first_result["success"] is True
    assert entered == ["first"]
    assert process.current_worker_task is None
