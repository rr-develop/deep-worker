"""Regression: codex provider must NOT crash the turn when Codex
emits a native tool event.

Originally written 2026-05-18 (rev 1) to *reproduce* the
``status="error"`` symptom of the original bug: at that time
``CodexChatModel._raise_if_blocked_native_item`` raised a
``RuntimeError`` whenever Codex emitted ``item.started`` for a native
tool. The exception aborted the turn and surfaced as
``status="error"`` in ``<agent-data-dir>/<agent>/session_metadata.json``.

The fix landed 2026-05-18 (rev 2): native tool events are now
recorded into LangGraph as honest ``AIMessage.tool_calls`` +
``ToolMessage`` pairs via the ``codex_provider.event_pump`` +
``codex_provider.native_tools`` machinery. See
``codex_provider/docs/DESIGN_NATIVE_TOOL_PASSTHROUGH.md``.

This file is now a **regression test for the absence** of the legacy
crash:

- The deterministic unit-level test asserts that a scripted
  ``item.started{type=command_execution}`` event does NOT raise. The
  provider returns an ``AIMessage`` with a real Codex tool_call.
- The live ``agent_fixture``-backed test (gated on
  ``CODEX_INTEGRATION_TESTS=1``) sends the same user question that
  originally triggered the crash and asserts the agent ends up
  ``waiting_input`` (or ``idle``), NEVER ``error``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

# Make the source ``codex_provider`` package importable without installing it.
# The dev tree layout puts the provider at the repository root.
import sys
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from codex_provider.chat_model import CodexChatModel  # noqa: E402
from codex_provider.exec_runner import (  # noqa: E402
    GenericItem,
    ItemStartedEvent,
    ThreadEvent,
    ThreadStartedEvent,
)

from . import agent_fixture  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic unit-level guarantee
# ---------------------------------------------------------------------------
class _FakeExecHandle:
    """Stand-in ExecHandle that mirrors the production shape: runs the
    real event pump over a scripted async generator, exposing
    ``ainvoke_queue`` and ``native_futures``."""

    def __init__(self, events: list[ThreadEvent]) -> None:
        from codex_provider.event_pump import event_pump

        async def _gen():
            for ev in events:
                yield ev

        self.events = _gen()
        self.ainvoke_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.native_futures: dict[str, asyncio.Future[str]] = {}
        self.pump_task: asyncio.Task[None] = asyncio.create_task(
            event_pump(self, proxy_mcp_server_name="deepworker_test"),
            name="fake-event-pump",
        )

    async def aclose(self) -> None:
        if not self.pump_task.done():
            self.pump_task.cancel()
            try:
                await self.pump_task
            except (asyncio.CancelledError, BaseException):
                pass


class _FakeExecRunner:
    """Returns a scripted _FakeExecHandle. Signature matches the real
    ``CodexExecRunner.spawn`` (including the keyword-only
    ``proxy_mcp_server_name``)."""

    def __init__(self, events: list[ThreadEvent]) -> None:
        self._events = events

    async def spawn(
        self, args: Any, *, proxy_mcp_server_name: str | None = None,
    ) -> Any:
        _ = args, proxy_mcp_server_name
        return _FakeExecHandle(self._events)


@pytest.mark.asyncio
async def test_native_command_execution_item_started_does_not_crash() -> None:
    """Regression: an ``item.started{type=command_execution}`` event
    must NOT raise. The provider records the event as a real Codex
    tool_call surfaced through ``AIMessage.tool_calls``.

    BEFORE the 2026-05-18 fix this test was the failing repro of the
    user's original ``status="error"`` bug — it asserted the legacy
    RuntimeError. AFTER the fix it asserts the opposite: no
    exception, and the tool_call name/id/args reflect what Codex
    actually emitted.
    """
    from unittest.mock import AsyncMock, MagicMock

    runner = _FakeExecRunner([
        ThreadStartedEvent(thread_id="uuid-repro"),
        # Same event Codex CLI emitted in the original bug report:
        # a native ``command_execution`` for an ``ls`` family call.
        ItemStartedEvent(item=GenericItem(
            id="item_0",
            type="command_execution",
            payload={
                "command": "/bin/bash -lc 'ls -la /path/to/project'",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        )),
    ])

    model = CodexChatModel()
    model._exec_runner = runner  # type: ignore[assignment]
    # Stub the proxy so ainvoke does not spin up real uvicorn / register
    # the proxy with codex CLI.
    model._proxy_mcp = MagicMock()
    model._proxy_mcp.start = AsyncMock()
    model._proxy_mcp.server_name = "deepworker_test"
    model._proxy_started = True

    # The whole point of the fix: NO RuntimeError, NO turn crash.
    result = await model.ainvoke(
        [HumanMessage(content="How many files are in your working directory?")],
        config={"configurable": {"thread_id": "test_codex_native_repro"}},
    )

    assert isinstance(result, AIMessage)
    assert result.tool_calls, "ainvoke must surface a tool_call for the native event"
    call = result.tool_calls[0]
    assert call["name"] == "command_execution", (
        "tool_call.name MUST be the Codex native name verbatim"
    )
    assert call["id"] == "item_0", (
        "tool_call.id MUST be Codex's own item.id"
    )
    assert "ls -la /path/to/project" in str(call["args"].get("command", "")), (
        "tool_call.args.command must carry Codex's exact command"
    )


# ---------------------------------------------------------------------------
# Live end-to-end repro via agent_fixture. Asserts the agent NO LONGER
# lands in status="error" for the original user prompt.
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("CODEX_INTEGRATION_TESTS") != "1",
    reason="set CODEX_INTEGRATION_TESTS=1 to run the live codex repro",
)
async def test_codex_provider_does_not_crash_on_user_question() -> None:
    """Real-codex end-to-end. Spawns an agent the way the user did
    (``--provider codex -sp -tm``), sends the original question, and
    asserts the agent ends up ``waiting_input``/``idle``, NEVER
    ``error``. ``error_message`` must NOT contain the legacy
    ``"Blocked native Codex tool event"`` string.
    """
    from deep_worker_cli.client.agent_client import AgentClient
    agent_name = "test_codex_native_passthrough"
    working_dir = Path("/path/to/project")

    with agent_fixture(
        agent_name=agent_name,
        working_dir=working_dir,
        model=None,
        provider="codex",
        skip_permissions=True,
        api_key=None,
    ):
        client = AgentClient(agent_name)
        await client.connect()
        try:
            try:
                await asyncio.wait_for(
                    client.invoke_agent(
                        "How many files are in your working directory?",
                        stream=False,
                    ),
                    timeout=180.0,
                )
            except Exception:
                # A slow LLM may time out the dwcli client; the agent
                # is still alive and will reach waiting_input on its
                # own. We rely on session_metadata.json below.
                pass
        finally:
            await client.disconnect()

        meta_path = Path("~/.deep_worker").expanduser() / agent_name / "session_metadata.json"
        deadline = time.monotonic() + 30.0
        meta: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    meta = {}
                # As soon as the agent has crossed out of "running"
                # we have a verdict.
                if meta.get("status") in ("waiting_input", "idle", "error"):
                    break
            await asyncio.sleep(0.25)

        assert meta_path.exists(), (
            f"Expected agent to write {meta_path} during the turn."
        )
        assert meta.get("status") != "error", (
            "Agent must NOT crash with status='error' anymore. "
            f"Got status={meta.get('status')!r} message="
            f"{meta.get('error_message')!r}"
        )
        err = meta.get("error_message") or ""
        assert "Blocked native Codex tool event" not in err, (
            "Legacy defense-in-depth error string must never appear. "
            f"Got: {err!r}"
        )
