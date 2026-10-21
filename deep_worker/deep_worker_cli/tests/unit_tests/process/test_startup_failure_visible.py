"""A startup failure must be VISIBLE, not kill the process.

Incident (2026-08-29). The only HTTP server of an external MCP pointed at a
dead port. The mcp SDK did not surface a ``ConnectError`` but a
``CancelledError`` (the anyio TaskGroup cancels its tasks), which inherits from
``BaseException`` and therefore passed straight through the
``except Exception`` in the initialization handler. The process died BEFORE
RPyC started and before registration in the registry — the agent was no longer
visible in the web UI, and reading its history became impossible.

Owner requirement: no silent crashes. The agent must come up in the ``error``
state with readable text and stay visible — precisely when it is broken, its
history is needed most.

Three things are pinned down:
1. ``CancelledError`` from a foreign scope does NOT kill initialization;
2. the agent ends up in ``error`` with text naming the server and URL;
3. an external cancellation (we are being shut down) is still propagated.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.cancellation import is_external_cancellation, root_cause_text
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


def _make_agent() -> AgentProcess:
    """AgentProcess without a real start; data_dir is in a temporary directory
    so the test does not touch production state in ``~/.deep_worker``."""
    agent = AgentProcess.__new__(AgentProcess)
    agent.name = "startup_probe"
    agent.session_manager = SessionManager(data_dir=Path(tempfile.mkdtemp()))
    agent.task_manager = None
    return agent


@pytest.mark.asyncio
async def test_cancelled_error_from_foreign_scope_is_not_external() -> None:
    """A cancellation from inside a foreign anyio scope is not external.

    This is the very signal the fix uses to tell a "subsystem failure" from
    "we are being shut down": nobody cancelled our task → ``cancelling() == 0``.
    """
    err = asyncio.CancelledError("Cancelled by cancel scope deadbeef")

    assert is_external_cancellation(err) is False


@pytest.mark.asyncio
async def test_real_external_cancellation_is_detected() -> None:
    """After ``task.cancel()`` the same error must be considered external."""
    seen: list[bool] = []
    started = asyncio.Event()

    async def _worker() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as e:
            seen.append(is_external_cancellation(e))
            raise

    task = asyncio.create_task(_worker())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen == [True], "an outside cancellation must be recognized and propagated"


def test_keyboard_interrupt_and_system_exit_are_external() -> None:
    """A request to stop the process must not be treated as a subsystem failure."""
    assert is_external_cancellation(KeyboardInterrupt()) is True
    assert is_external_cancellation(SystemExit()) is True


def test_ordinary_exception_is_not_cancellation() -> None:
    """An ordinary error must go to the general handler, not be propagated."""
    assert is_external_cancellation(RuntimeError("boom")) is False


@pytest.mark.asyncio
async def test_dead_mcp_server_puts_agent_into_error_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dead MCP → agent in ``error``, text names the server and URL.

    Without the fix, ``CancelledError`` escaped from ``_start_external_mcp_pool``
    and took the process down; neither a status nor a message appeared.
    """
    from deep_worker_cli.external_mcp.config import (
        ExternalMCPStartupError,
        ServerSpec,
    )

    agent = _make_agent()

    async def _boom(*_a: Any, **_kw: Any) -> Any:
        raise asyncio.CancelledError("Cancelled by cancel scope deadbeef")

    monkeypatch.setattr(
        "deep_worker_cli.external_mcp.pool.ExternalMCPClientPool.create", _boom
    )
    monkeypatch.setattr(
        "deep_worker_cli.external_mcp.config.load_mcp_config",
        lambda **_kw: {
            "code_graph_cross_file_update": ServerSpec(
                name="code_graph_cross_file_update",
                type="http",
                url="http://127.0.0.1:40587/cross-file-update-mcp",
            )
        },
    )

    agent.config = type("_Cfg", (), {"working_dir": str(tmp_path)})()
    agent.mcp_config_paths = []
    agent.mcp_server_inline = []

    with pytest.raises(ExternalMCPStartupError) as excinfo:
        await agent._start_external_mcp_pool()

    text = str(excinfo.value)
    assert "code_graph_cross_file_update" in text, f"server name missing: {text}"
    assert "40587" in text, f"address missing: {text}"

    # Exactly what the initialization handler does with this error.
    await agent.session_manager.set_error(f"Agent initialization failed: {text}")

    assert agent.session_manager.status == AgentStatus.ERROR
    assert "40587" in (agent.session_manager.error_message or "")


def test_root_cause_unwraps_exception_group() -> None:
    """anyio yields an ``ExceptionGroup`` — the owner needs the fact, not the wrapper.

    A live check against a dead port produced the text "unhandled errors in a
    TaskGroup (1 sub-exception)": the real ``ConnectError`` was hiding inside.
    """
    inner = ConnectionError("All connection attempts failed")
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [inner])

    text = root_cause_text(group)

    assert "All connection attempts failed" in text
    assert "TaskGroup" not in text


def test_root_cause_follows_cause_chain() -> None:
    """``raise X from Y`` — show Y, otherwise the cause is lost."""
    try:
        try:
            raise ConnectionError("connection refused")
        except ConnectionError as cause:
            raise RuntimeError("wrapper") from cause
    except RuntimeError as e:
        assert "connection refused" in root_cause_text(e)


def test_describe_mcp_servers_uses_cli_configs(tmp_path: Path) -> None:
    """A server from ``--mcp-config`` must make it into the error text.

    The first version of the helper did not pass ``cli_configs`` through and
    printed ``<none configured>`` — the message named neither the server nor
    the address, so the visibility requirement went unmet.
    """
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        '{"mcpServers":{"probe":{"url":"http://127.0.0.1:40587/mcp"}}}'
    )

    described = AgentProcess._describe_mcp_servers(
        None, None, None, [cfg], []
    )

    assert "probe" in described
    assert "40587" in described
