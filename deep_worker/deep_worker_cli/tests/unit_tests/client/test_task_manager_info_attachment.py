"""`get_task_manager_info` must not invent a switch for a detached agent.

The agent reports attachment in ``task_manager_attached`` and, for a
detached agent, ``task_manager_disabled: None`` — "there is no switch".
The old derivation probed for mere KEY PRESENCE:

    has_tm_fields = (
        "task_manager_active" in response
        or "task_manager_disabled" in response
    )

so an explicit ``None`` made a TaskManager-less agent look configured,
and ``bool(None)`` then reported ``disabled=False`` — asserting "the
switch exists and it is on". ``deep-worker --status`` printed
``TaskManager: inactive`` instead of ``not configured`` for the majority
of agents (``task_manager=False`` in their ``session_metadata.json``).

These tests drive the REAL method over the real payload shapes. The only
pre-existing test touching this code
(``tests/unit_tests/cli/test_show_status.py``) replaces
``get_task_manager_info`` with a stub, so the actual derivation never ran
there and a full green suite said nothing about it.
"""

from __future__ import annotations

from typing import Any

import pytest

from deep_worker_cli.client.agent_client import AgentClient


def _client(response: dict[str, Any]) -> AgentClient:
    """An AgentClient whose ``status_raw`` returns ``response`` verbatim."""
    client = AgentClient.__new__(AgentClient)

    async def _send_command(_command: dict[str, Any]) -> dict[str, Any]:
        return response

    client._send_command = _send_command  # type: ignore[method-assign]
    return client


def _status(**extra: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "success": True,
        "status": "idle",
        "error": None,
        "last_activity": "2026-10-11T09:00:00",
        "message_count": 1,
        "tasks_completed": 0,
    }
    response.update(extra)
    return response


@pytest.mark.asyncio
async def test_detached_agent_is_not_reported_configured() -> None:
    """The reported regression: ``disabled: None`` meant "no switch"."""
    client = _client(
        _status(task_manager_attached=False, task_manager_disabled=None)
    )

    info = await client.get_task_manager_info()

    assert info == {"configured": False, "active": None, "disabled": None}, (
        "a detached agent was described as having a switch that is on — "
        "this prints 'TaskManager: inactive' instead of 'not configured'"
    )


@pytest.mark.asyncio
async def test_legacy_payload_without_any_tm_keys_still_detached() -> None:
    """An agent process predating the explicit field must still work.

    It is long-running and will not be restarted for this change, so the
    legacy encoding (no ``task_manager_*`` keys at all) has to keep
    resolving to "not configured".
    """
    client = _client(_status())

    info = await client.get_task_manager_info()

    assert info == {"configured": False, "active": None, "disabled": None}


@pytest.mark.asyncio
async def test_attached_and_disabled_is_reported_as_disabled() -> None:
    """The switch must survive the trip, distinctly from ``active``."""
    client = _client(
        _status(
            task_manager_attached=True,
            task_manager_active=False,
            task_manager_disabled=True,
        )
    )

    info = await client.get_task_manager_info()

    assert info == {"configured": True, "active": False, "disabled": True}


@pytest.mark.asyncio
async def test_attached_enabled_and_idle_is_not_disabled() -> None:
    """Enabled but sleeping between tasks: ``active`` false, switch off."""
    client = _client(
        _status(
            task_manager_attached=True,
            task_manager_active=False,
            task_manager_disabled=False,
        )
    )

    info = await client.get_task_manager_info()

    assert info == {"configured": True, "active": False, "disabled": False}


@pytest.mark.asyncio
async def test_legacy_attached_payload_without_explicit_field() -> None:
    """Legacy attached shape: keys present, no ``task_manager_attached``."""
    client = _client(
        _status(task_manager_active=True, task_manager_disabled=False)
    )

    info = await client.get_task_manager_info()

    assert info == {"configured": True, "active": True, "disabled": False}
