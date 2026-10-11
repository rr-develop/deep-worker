from typing import AsyncIterator
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routers.agents import router
from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry


@pytest.fixture
def service() -> AgentService:
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    return AgentService(registry=registry)


@pytest.fixture
async def client(service: AgentService) -> AsyncIterator[AsyncClient]:
    await service.initialize()
    app = FastAPI()
    app.include_router(router)
    app.state.agent_service = service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as value:
        yield value
    await service.shutdown()


@pytest.mark.asyncio
async def test_summarize_returns_accepted_request_id(
    client: AsyncClient,
    service: AgentService,
) -> None:
    service.summarize_agent = AsyncMock(
        return_value={
            "success": True,
            "agent_name": "test-agent",
            "request_id": "summary-request",
        }
    )

    response = await client.post("/api/agents/test-agent/summarize")

    assert response.status_code == 202
    assert response.json()["request_id"] == "summary-request"
    service.summarize_agent.assert_awaited_once_with("test-agent")


@pytest.mark.asyncio
async def test_summarize_unknown_agent_is_404(
    client: AsyncClient,
) -> None:
    response = await client.post("/api/agents/missing/summarize")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_service_emits_exactly_one_terminal_event_with_request_id() -> None:
    service = AgentService(registry=MockProcessRegistry())
    service.ensure_connecting = AsyncMock(return_value=True)
    service._roots["test-agent"] = SimpleNamespace(
        summarize=AsyncMock(
            return_value=json.dumps(
                {
                    "success": True,
                    "summarized": False,
                    "history_path": None,
                }
            )
        )
    )
    service._broadcast_message = AsyncMock()
    tasks: list[asyncio.Task[None]] = []

    def retain_task(
        make_awaitable,
        *,
        success_callback,
        error_callback,
        **kwargs,
    ) -> None:
        # Mirrors the real helper: it takes a FACTORY (BACKLOG §12) and calls
        # it INSIDE the task, so the remote call never runs on the caller's
        # event loop.
        async def run() -> None:
            try:
                await success_callback(await make_awaitable())
            except BaseException as exc:
                await error_callback(exc)

        tasks.append(asyncio.create_task(run()))

    with patch(
        "rpyc_async.utils.helpers.fire_and_forget_call",
        side_effect=retain_task,
    ):
        accepted = await service.summarize_agent("test-agent")
        await tasks[0]

    service._broadcast_message.assert_awaited_once()
    event = service._broadcast_message.await_args.args[0]
    assert event["type"] == "agent_summarized"
    assert event["request_id"] == accepted["request_id"]
    assert event["summarized"] is False


def test_ui_consumes_summary_terminal_event_and_tracks_button() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "static" / "js" / "main.js"
    ).read_text(encoding="utf-8")
    assert "case 'agent_summarized':" in source
    assert "handleAgentSummarized(data)" in source
    assert "finalizeRequest(data.request_id)" in source
    assert "trackButton(data.request_id, btn, 'agent_summarize'" in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ["not-json", json.dumps({"success": False, "error": "denied"})],
)
async def test_service_failure_emits_one_correlated_error(response: str) -> None:
    service = AgentService(registry=MockProcessRegistry())
    service.ensure_connecting = AsyncMock(return_value=True)
    service._roots["test-agent"] = SimpleNamespace(
        summarize=AsyncMock(return_value=response)
    )
    service._broadcast_message = AsyncMock()
    tasks: list[asyncio.Task[None]] = []

    def retain_task(awaitable, *, success_callback, error_callback, **kwargs) -> None:
        async def run() -> None:
            try:
                await success_callback(await awaitable)
            except BaseException as exc:
                await error_callback(exc)

        tasks.append(asyncio.create_task(run()))

    with patch(
        "rpyc_async.utils.helpers.fire_and_forget_call",
        side_effect=retain_task,
    ):
        accepted = await service.summarize_agent("test-agent")
        await tasks[0]

    service._broadcast_message.assert_awaited_once()
    event = service._broadcast_message.await_args.args[0]
    assert event["type"] == "agent_error"
    assert event["request_id"] == accepted["request_id"]


@pytest.mark.asyncio
async def test_service_ignores_adversarial_second_terminal_callback() -> None:
    service = AgentService(registry=MockProcessRegistry())
    service.ensure_connecting = AsyncMock(return_value=True)
    service._roots["test-agent"] = SimpleNamespace(
        summarize=AsyncMock(return_value=json.dumps({"success": True, "summarized": True}))
    )
    service._broadcast_message = AsyncMock()
    callbacks = {}

    def capture(make_awaitable, *, success_callback, error_callback, **kwargs) -> None:
        # fire_and_forget_call takes a FACTORY (BACKLOG §12), so there is
        # no un-awaited coroutine to close: never calling it is enough.
        callbacks["success"] = success_callback
        callbacks["error"] = error_callback

    with patch(
        "rpyc_async.utils.helpers.fire_and_forget_call",
        side_effect=capture,
    ):
        accepted = await service.summarize_agent("test-agent")
    await callbacks["success"](
        json.dumps({"success": True, "summarized": True})
    )
    await callbacks["error"](ConnectionError("late disconnect"))

    service._broadcast_message.assert_awaited_once()
    assert (
        service._broadcast_message.await_args.args[0]["request_id"]
        == accepted["request_id"]
    )


@pytest.mark.asyncio
async def test_service_uses_real_fire_and_forget_strong_ref_owner() -> None:
    from rpyc_async.utils import helpers as faf_helpers

    service = AgentService(registry=MockProcessRegistry())
    service.ensure_connecting = AsyncMock(return_value=True)
    release = asyncio.Event()

    async def controlled_summarize() -> str:
        await release.wait()
        return json.dumps(
            {"success": True, "summarized": True, "history_path": "/h.jsonl"}
        )

    service._roots["test-agent"] = SimpleNamespace(summarize=controlled_summarize)
    done = asyncio.Event()

    async def broadcast_and_signal(message: dict) -> None:
        done.set()

    service._broadcast_message = AsyncMock(side_effect=broadcast_and_signal)

    accepted = await service.summarize_agent("test-agent")
    owned = [
        task
        for task in faf_helpers._INFLIGHT
        if getattr(task, "get_name", lambda: "")() == "summarize_agent_test-agent"
    ]
    assert owned, "real fire_and_forget_call must pin the summarize task"
    release.set()
    await asyncio.wait_for(done.wait(), timeout=2.0)
    service._broadcast_message.assert_awaited_once()
    event = service._broadcast_message.await_args.args[0]
    assert event["type"] == "agent_summarized"
    assert event["request_id"] == accepted["request_id"]
    assert event["summarized"] is True


@pytest.mark.asyncio
async def test_service_transport_exception_emits_one_error() -> None:
    service = AgentService(registry=MockProcessRegistry())
    service.ensure_connecting = AsyncMock(return_value=True)

    async def boom() -> str:
        raise ConnectionError("disconnect")

    service._roots["test-agent"] = SimpleNamespace(summarize=boom)
    done = asyncio.Event()

    async def broadcast_and_signal(message: dict) -> None:
        done.set()

    service._broadcast_message = AsyncMock(side_effect=broadcast_and_signal)

    accepted = await service.summarize_agent("test-agent")
    await asyncio.wait_for(done.wait(), timeout=2.0)

    service._broadcast_message.assert_awaited_once()
    event = service._broadcast_message.await_args.args[0]
    assert event["type"] == "agent_error"
    assert event["request_id"] == accepted["request_id"]
    assert "disconnect" in event["error"]
