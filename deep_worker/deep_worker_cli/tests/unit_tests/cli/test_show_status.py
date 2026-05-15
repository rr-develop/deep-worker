"""``dwcli --status`` / ``deep-worker-cli --status`` must display honest data only.

Hard requirements pinned by these tests:

* The working directory shown MUST come from the agent (via
  ``get_metadata``) — no "/tmp" placeholder, no current-working-dir
  fallback.
* The TaskManager state MUST be displayed (active / disabled / N/A),
  reflecting exactly what the agent returned in ``get_status``.
* If the agent did not return a field (e.g. ``last_activity`` or
  ``tasks_completed``), the output must say "N/A"/"unknown" — never
  fabricate a value or use the current time.
* The existing ``AgentClient.get_status()`` contract is NOT changed;
  ``show_status`` is free to call additional RPCs (e.g. ``get_metadata``)
  to get honest data.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fake_metadata(**over: Any) -> dict[str, Any]:
    """Return the inner ``metadata`` dict — the shape ``get_agent_metadata``
    is contracted to return (it already unwraps ``{"success":..,"metadata":..}``
    on the AgentClient side)."""
    base = {
        "directory": "/path/to/project",
        "pid": 12345,
        "started": "2026-04-24T10:56:48.044891",
        "rpyc_port": 47555,
        "agent_type": "deep-worker",
        "model": "opus",
        "provider": "claude-code",
    }
    base.update(over)
    return base


def _fake_status_tm_active(**over: Any) -> Any:
    """``get_status`` shape when TaskManager IS configured and running."""
    from deep_worker_cli.basesession.protocol_types import SessionInfo, SessionStatus
    # Make a minimal SessionInfo; specific fields under test are set to
    # distinguishable sentinels so we can catch leaks of fabricated data.
    si = MagicMock(spec=SessionInfo)
    si.name = "demo_agent"
    si.status = SessionStatus.WORKING
    si.session_type = "langchain"
    si.working_dir = over.get("working_dir", "WRONG_SHOULD_NOT_BE_USED")
    si.created_at = over.get("created_at", None)
    si.last_activity = over.get("last_activity", None)
    si.tasks_completed = over.get("tasks_completed", 0)
    si.error_message = None
    si.data = None
    si.metadata = None
    return si


class _FakeAgentClient:
    """Stand-in for AgentClient with controllable get_status / metadata."""

    def __init__(
        self,
        *,
        metadata_payload: dict[str, Any] | None,
        status_obj: Any,
        task_manager_fields: dict[str, bool] | None = None,
    ) -> None:
        self.agent_name = "demo_agent"
        self._metadata_payload = metadata_payload
        self._status_obj = status_obj
        self._tm = task_manager_fields or {}
        self.connected = True
        self.connect = AsyncMock()
        self.disconnect = AsyncMock()

    async def get_status(self):
        return self._status_obj

    async def get_agent_metadata(self) -> dict[str, Any] | None:
        """New helper on AgentClient (this is what show_status will call).

        Must return the raw metadata dict (from ``exposed_get_metadata``)
        or None when unavailable. Not a SessionInfo — no remapping.
        """
        return self._metadata_payload

    async def get_task_manager_info(self) -> dict[str, Any] | None:
        """New helper on AgentClient for TaskManager state.

        Returns ``{"configured": bool, "active": bool | None, "disabled": bool | None}``
        where ``configured=False`` means the agent's get_status response
        did not include any ``task_manager_*`` keys.
        """
        if not self._tm:
            return {"configured": False, "active": None, "disabled": None}
        return {
            "configured": True,
            "active": self._tm.get("task_manager_active"),
            "disabled": self._tm.get("task_manager_disabled"),
        }


# --------------------------------------------------------------------------- #
# Honest working_dir
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_show_status_prints_working_dir_from_metadata(capsys) -> None:
    """Working dir shown must come from get_metadata, NOT from the
    SessionInfo placeholder on AgentClient.get_status."""
    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=_fake_metadata(directory="/path/to/project"),
        status_obj=_fake_status_tm_active(working_dir="/tmp"),  # placeholder
        task_manager_fields={"task_manager_active": True, "task_manager_disabled": False},
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    assert "/path/to/project" in out, out
    # The placeholder used historically. Must never reach the terminal.
    assert "Working Dir:     /tmp" not in out, (
        f"Placeholder '/tmp' leaked into --status output:\n{out}"
    )


@pytest.mark.asyncio
async def test_show_status_never_shows_placeholder_tmp(capsys) -> None:
    """Even when get_metadata fails, we must not substitute a made-up
    working dir. Print N/A instead — never lie."""
    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=None,  # metadata RPC failed
        status_obj=_fake_status_tm_active(working_dir="/tmp"),
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    # Placeholder must be gone.
    assert "/tmp" not in out, f"'/tmp' leaked when metadata was unavailable:\n{out}"
    # An honest "unknown" marker must appear on the Working Dir row.
    assert "Working Dir" in out
    assert "N/A" in out or "unknown" in out.lower()


# --------------------------------------------------------------------------- #
# TaskManager visibility
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_show_status_displays_task_manager_active(capsys) -> None:
    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=_fake_metadata(),
        status_obj=_fake_status_tm_active(),
        task_manager_fields={
            "task_manager_active": True,
            "task_manager_disabled": False,
        },
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    assert "TaskManager" in out, out
    assert "active" in out.lower(), out


@pytest.mark.asyncio
async def test_show_status_displays_task_manager_disabled(capsys) -> None:
    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=_fake_metadata(),
        status_obj=_fake_status_tm_active(),
        task_manager_fields={
            "task_manager_active": False,
            "task_manager_disabled": True,
        },
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    assert "TaskManager" in out
    assert "disabled" in out.lower() or "inactive" in out.lower()


@pytest.mark.asyncio
async def test_show_status_reports_tm_not_configured(capsys) -> None:
    """When the agent's get_status response had no ``task_manager_*``
    fields, the CLI must say 'not configured' / 'N/A' — never guess."""
    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=_fake_metadata(),
        status_obj=_fake_status_tm_active(),
        task_manager_fields=None,  # not configured
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    assert "TaskManager" in out
    assert (
        "not configured" in out.lower()
        or "N/A" in out
        or "unknown" in out.lower()
    )


# --------------------------------------------------------------------------- #
# No fabricated created_at / last_activity / tasks_completed
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_show_status_uses_started_at_from_metadata_not_now(capsys) -> None:
    """The 'Created' line must come from metadata.started, not from
    a datetime.now() placeholder on the SessionInfo."""
    from datetime import datetime

    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=_fake_metadata(started="2026-04-24T10:56:48.044891"),
        # SessionInfo would contain datetime.now() — we must NOT use it.
        status_obj=_fake_status_tm_active(created_at=datetime(2099, 1, 1, 0, 0, 0)),
        task_manager_fields={"task_manager_active": True, "task_manager_disabled": False},
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    # The real start-time from the agent metadata.
    assert "10:56:48" in out, out
    # The SessionInfo placeholder (year 2099) must never appear.
    assert "2099" not in out


@pytest.mark.asyncio
async def test_show_status_missing_last_activity_prints_na(capsys) -> None:
    from deep_worker_cli.cli.client_command import show_status

    client = _FakeAgentClient(
        metadata_payload=_fake_metadata(),
        status_obj=_fake_status_tm_active(last_activity=None, tasks_completed=0),
        task_manager_fields={"task_manager_active": True, "task_manager_disabled": False},
    )
    with patch("deep_worker_cli.cli.client_command.AgentClient", return_value=client):
        rc = await show_status("demo_agent")
    assert rc == 0
    out = capsys.readouterr().out
    # ``Last Activity`` row is present but must be N/A — never made up.
    if "Last Activity" in out:
        # If it's printed at all it must be N/A; mistakes would show
        # e.g. "1970-01-01" or a fresh ``datetime.now()``.
        assert "N/A" in out or "unknown" in out.lower()
