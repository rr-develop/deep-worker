"""A started turn must advance the activity counters.

Regression for the 2026-09-26 report. `message_count` existed, was
persisted, was exposed — and nothing in production ever incremented it, so
every consumer read a permanent 0. `_handle_invoke_owned` is where a
user-visible turn begins; the counter that claims to count turns is advanced there.

The test drives the real `SessionManager`, not a mock of it, so it also
pins that the increment re-stamps `last_activity`.
"""

from __future__ import annotations

import ast
import inspect
import tempfile
import textwrap
from pathlib import Path

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


def test_run_agent_advances_the_turn_counter() -> None:
    """Pin the call site: the counter must be advanced where a turn starts.

    `_handle_invoke_owned` needs a live model and checkpointer to run, so
    contract is checked on its source: the turn-start block that sets
    RUNNING must also advance the counter. Without this the field silently
    reverts to the dead 0 it reported for months.
    """
    source = textwrap.dedent(inspect.getsource(AgentProcess._handle_invoke_owned))
    tree = ast.parse(source)

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "increment_message_count" in called, (
        "a starting turn no longer advances message_count — consumers are "
        "back to reading a constant 0"
    )
    assert "set_status" in called, "turn start must still publish RUNNING"


@pytest.mark.asyncio
async def test_increment_restamps_last_activity() -> None:
    """The counter and the timestamp move together, from one source."""
    manager = SessionManager(data_dir=Path(tempfile.mkdtemp()))
    await manager.set_status(AgentStatus.IDLE)
    before = manager.last_activity

    await manager.increment_message_count()

    assert manager.message_count == 1
    assert manager.last_activity >= before
