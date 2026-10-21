"""`exposed_get_status` must ship the activity fields it alone knows.

Regression for the 2026-09-26 report. The minimal status payload carried
only `status`/`error`, so the web layer had nothing to report as
`last_activity` and substituted the process start time — making a busy
agent look twelve days idle.

The session manager stamps `last_activity` on every turn boundary. These
tests pin that the value leaves the agent, because no consumer downstream
can reconstruct it.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from deep_worker_cli.rpyc_service.agent_service import AgentService
from deep_worker_cli.session.session_manager import AgentStatus


def _service(
    session_manager: object,
    task_manager: object | None = None,
    started_at: datetime | None = None,
) -> AgentService:
    service = AgentService()
    service._agent_process = SimpleNamespace(
        session_manager=session_manager,
        task_manager=task_manager,
        started_at=started_at,
    )
    return service


def _session_manager(
    *,
    status: AgentStatus = AgentStatus.IDLE,
    last_activity: datetime | None = None,
    message_count: int = 0,
    tasks_completed: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        error_message=None,
        last_activity=last_activity or datetime(2026, 9, 26, 13, 15, 46),
        message_count=message_count,
        tasks_completed=tasks_completed,
    )


@pytest.mark.asyncio
async def test_status_reports_last_activity() -> None:
    """Without this field the consumer has to invent one — and it did."""
    stamp = datetime(2026, 9, 26, 13, 15, 46, 738659)
    service = _service(_session_manager(last_activity=stamp))

    response = json.loads(await service.exposed_get_status())

    assert response["success"] is True
    assert response["last_activity"] == stamp.isoformat()


@pytest.mark.asyncio
async def test_status_reports_uptime_and_started_at() -> None:
    """Uptime must come from the agent — nothing downstream knows it.

    Reported 2026-10-19: the the task manager status tool returned
    ``uptime_seconds: null`` for every agent and the UI showed "--",
    because this payload carried neither the start time nor a duration.
    ``last_activity`` cannot stand in: it is a turn boundary, so on a
    busy agent it is minutes old and on an idle one it is days old.
    """
    started = datetime.now()
    service = _service(_session_manager(), started_at=started)

    response = json.loads(await service.exposed_get_status())

    assert response["started_at"] == started.isoformat()
    # Both fields travel: the duration for a one-shot reader, the
    # timestamp so a UI can tick without re-asking.
    assert response["uptime_seconds"] is not None
    assert response["uptime_seconds"] >= 0.0
    # A freshly-stamped start time cannot have produced a large uptime;
    # this would catch a wrong epoch or a swapped subtraction.
    assert response["uptime_seconds"] < 60.0


@pytest.mark.asyncio
async def test_uptime_grows_with_a_known_start_time() -> None:
    """A fixed start in the past must yield that elapsed time.

    The previous test would also pass if the field were hard-coded to
    0.0, so pin the arithmetic against a start time whose distance from
    now is known.
    """
    from datetime import timedelta

    started = datetime.now() - timedelta(hours=2)
    service = _service(_session_manager(), started_at=started)

    response = json.loads(await service.exposed_get_status())

    # 2h = 7200s, with generous slack for test execution time.
    assert 7190.0 <= response["uptime_seconds"] <= 7260.0
    assert response["started_at"] == started.isoformat()


@pytest.mark.asyncio
async def test_unknown_start_time_is_null_not_zero() -> None:
    """Before ``start()`` stamps it, uptime is unknown — not zero.

    Reporting 0 would claim "just started" about an agent whose start
    time we do not have.
    """
    service = _service(_session_manager(), started_at=None)

    response = json.loads(await service.exposed_get_status())

    assert response["started_at"] is None
    assert response["uptime_seconds"] is None


@pytest.mark.asyncio
async def test_status_reports_counters() -> None:
    """Counters travel with the status that describes the same moment."""
    service = _service(_session_manager(message_count=7, tasks_completed=2))

    response = json.loads(await service.exposed_get_status())

    assert response["message_count"] == 7
    assert response["tasks_completed"] == 2


@pytest.mark.asyncio
async def test_working_status_is_reported_as_working() -> None:
    """The busy/idle answer must come from the agent, not a default."""
    service = _service(_session_manager(status=AgentStatus.RUNNING))

    response = json.loads(await service.exposed_get_status())

    assert response["status"] == "working"


@pytest.mark.asyncio
async def test_status_reports_task_manager_attachment_explicitly() -> None:
    """Attachment must be a FIELD, not the absence of other fields.

    It used to be encoded by omitting the ``task_manager_*`` keys, which
    a consumer reading the parsed payload cannot tell from "unknown" —
    and that ambiguity produced a false "TaskManager is off" report
    about a running, enabled one.
    """
    task_manager = SimpleNamespace(state="waiting", disabled=False)
    service = _service(_session_manager(), task_manager=task_manager)

    response = json.loads(await service.exposed_get_status())

    assert response["task_manager_attached"] is True
    assert response["task_manager_disabled"] is False
    assert response["task_manager_active"] is True


@pytest.mark.asyncio
async def test_status_distinguishes_disabled_from_idle() -> None:
    """The switch is its own field: ``active`` cannot answer for it.

    A disabled loop never leaves ``inactive``, so ``active`` is false
    both when the operator switched it off and when it is simply
    sleeping between tasks.
    """
    disabled = _service(
        _session_manager(),
        task_manager=SimpleNamespace(state="inactive", disabled=True),
    )
    idle = _service(
        _session_manager(),
        task_manager=SimpleNamespace(state="inactive", disabled=False),
    )

    disabled_response = json.loads(await disabled.exposed_get_status())
    idle_response = json.loads(await idle.exposed_get_status())

    assert disabled_response["task_manager_active"] is False
    assert idle_response["task_manager_active"] is False
    assert disabled_response["task_manager_disabled"] is True
    assert idle_response["task_manager_disabled"] is False
    assert disabled_response["task_manager_attached"] is True
    assert idle_response["task_manager_attached"] is True


@pytest.mark.asyncio
async def test_detached_agent_reports_false_attachment_and_null_switch() -> None:
    """No TaskManager: attachment is False, and there is no switch.

    ``disabled`` must be ``None`` rather than ``False`` — a detached
    agent has no switch to describe, and ``False`` would assert
    "enabled".
    """
    service = _service(_session_manager(), task_manager=None)

    response = json.loads(await service.exposed_get_status())

    assert response["task_manager_attached"] is False
    assert response["task_manager_disabled"] is None
    assert "task_manager_active" not in response
