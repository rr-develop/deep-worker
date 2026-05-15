"""``SessionManager`` must push a ``status_change`` event on every state
transition so subscribers learn about it without any polling.

Contract pinned here:

* A ``SessionManager`` created without a notifier keeps its original
  behaviour (persist only). Backward-compatibility.
* A ``SessionManager`` constructed with an async ``notifier`` callable
  invokes it ONCE per ``set_status`` / ``set_error`` / ``clear_error``
  call with a payload that at minimum contains
  ``{"type": "status_change", "status": <enum value>}``.
* Persistence happens regardless of notifier success: a failing
  notifier is logged but not raised, otherwise a transient subscriber
  error would cause the agent to lose status data on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.types import AgentStatus


@pytest.mark.asyncio
async def test_no_notifier_works_like_before(tmp_path: Path) -> None:
    """Regression: no-arg SessionManager still persists without errors."""
    sm = SessionManager(tmp_path)
    await sm.set_status(AgentStatus.RUNNING)
    assert sm.status == AgentStatus.RUNNING


@pytest.mark.asyncio
async def test_set_status_calls_notifier_once_with_status_change(
    tmp_path: Path,
) -> None:
    notifier = AsyncMock()
    sm = SessionManager(tmp_path, notifier=notifier)  # type: ignore[call-arg]
    await sm.set_status(AgentStatus.RUNNING)

    assert notifier.await_count == 1
    (payload,) = notifier.await_args.args
    assert isinstance(payload, dict)
    assert payload["type"] == "status_change"
    assert payload["status"] == AgentStatus.RUNNING.value


@pytest.mark.asyncio
async def test_set_error_also_pushes(tmp_path: Path) -> None:
    notifier = AsyncMock()
    sm = SessionManager(tmp_path, notifier=notifier)  # type: ignore[call-arg]
    await sm.set_error("boom")

    assert notifier.await_count == 1
    (payload,) = notifier.await_args.args
    assert payload["type"] == "status_change"
    assert payload["status"] == AgentStatus.ERROR.value
    # Error message must be surfaced too so subscribers don't have to
    # poll a separate RPC.
    assert payload.get("error_message") == "boom"


@pytest.mark.asyncio
async def test_clear_error_pushes_with_current_status(tmp_path: Path) -> None:
    notifier = AsyncMock()
    sm = SessionManager(tmp_path, notifier=notifier)  # type: ignore[call-arg]
    await sm.set_error("bad")  # 1st push
    notifier.reset_mock()
    await sm.clear_error()  # 2nd push

    assert notifier.await_count == 1
    (payload,) = notifier.await_args.args
    assert payload["type"] == "status_change"
    # After clear, error_message must be None (honest, not omitted).
    assert payload.get("error_message") is None


@pytest.mark.asyncio
async def test_notifier_failure_does_not_break_persistence(
    tmp_path: Path,
) -> None:
    """If the subscriber hangs or throws, set_status must still
    commit to disk — otherwise a bad WS client could corrupt
    agent state."""

    async def broken(payload: dict[str, Any]) -> None:
        raise RuntimeError("subscriber exploded")

    sm = SessionManager(tmp_path, notifier=broken)  # type: ignore[call-arg]
    await sm.set_status(AgentStatus.RUNNING)
    # On-disk state must reflect the new status.
    assert sm.status == AgentStatus.RUNNING

    import json
    meta = json.loads((tmp_path / "session_metadata.json").read_text())
    assert meta["status"] == AgentStatus.RUNNING.value
