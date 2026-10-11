"""The three TaskManager states must be distinguishable over the API.

``task_manager_active`` answers ONE question: is the TM loop working on a
task right now. It is not a setting. Three separate states exist:

  * attached — a TaskManager is wired to the agent at all;
  * disabled — the operator's on/off switch (``TaskManager.disabled``);
  * active   — the loop is not ``inactive`` at this moment.

Before this change the REST ``detailed_status_response`` published only
``task_manager_active``, so a consumer could not tell "switched off" from
"switched on and sleeping between tasks" from "no TaskManager at all" —
and that ambiguity already produced the false conclusion "TaskManager is
off" about a running, enabled TaskManager.

The agent side encodes attachment by KEY ABSENCE
(``deep_worker_cli/rpyc_service/agent_service.py::exposed_get_status``
omits both keys when ``task_manager is None``), which is unreadable to
any consumer that only sees the parsed payload. Both the REST path and
the WS broadcast path therefore publish ``task_manager_attached`` and
``task_manager_disabled`` explicitly.

Field names and null semantics are fixed by the WorkControl side
(``work_control_api`` commit ``6df91c6``): all three fields are
``boolean | null``, ``null`` means UNKNOWN, and ``disabled`` is ``null``
when nothing is attached because then no switch exists. ``null`` must
never be read as ``false``, and no field may be inferred from another.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.agent_service import AgentService
from tests.fixtures.mock_registry import MockProcessRegistry

#: Flat payload shape of the agent's ``exposed_get_status``. Attachment is
#: encoded by the PRESENCE of the two ``task_manager_*`` keys.
AGENT_LAST_ACTIVITY = "2026-10-11T09:15:46.738659"


@pytest.fixture
def mock_registry() -> MockProcessRegistry:
    registry = MockProcessRegistry()
    registry.add_mock_agent("test-agent", 12345, 18861, "/tmp/test")
    return registry


@pytest.fixture
def agent_service(mock_registry: MockProcessRegistry) -> AgentService:
    return AgentService(registry=mock_registry)


def _agent_response(**extra: object) -> dict[str, object]:
    """A realistic ``exposed_get_status`` payload plus the given TM keys."""
    response: dict[str, object] = {
        "success": True,
        "status": "idle",
        "error": None,
        "last_activity": AGENT_LAST_ACTIVITY,
        "message_count": 7,
        "tasks_completed": 2,
    }
    response.update(extra)
    return response


async def _rest_detailed_status(
    agent_service: AgentService, agent_name: str, agent_response: dict
) -> dict:
    """Drive the real ``get_agent_detailed_status`` success callback."""
    agent_service._connections[agent_name] = MagicMock()
    agent_service._roots[agent_name] = MagicMock()

    broadcasts: list[dict] = []

    async def capture(message: dict) -> None:
        broadcasts.append(message)

    with (
        patch("rpyc_async.utils.helpers.fire_and_forget_call") as mock_faf,
        patch.object(agent_service, "_broadcast_message", capture),
    ):
        await agent_service.get_agent_detailed_status(agent_name)
        assert mock_faf.called, "detailed status never reached the agent"
        on_success = mock_faf.call_args.kwargs["success_callback"]
        await on_success(json.dumps(agent_response))

    assert broadcasts, "no detailed_status_response was broadcast"
    message = broadcasts[-1]
    assert message["type"] == "detailed_status_response", message
    return message["status"]


async def _ws_status_broadcast(
    agent_service: AgentService, agent_name: str, agent_response: dict
) -> dict:
    """Drive the real ``_update_agent_status`` success callback."""
    mock_root = MagicMock()
    agent_service._connections[agent_name] = MagicMock()
    agent_service._roots[agent_name] = mock_root

    broadcasts: list[dict] = []

    async def capture(message: dict) -> None:
        broadcasts.append(message)

    with (
        patch("rpyc_async.utils.helpers.fire_and_forget_call") as mock_faf,
        patch.object(agent_service, "_broadcast_message", capture),
    ):
        await agent_service._update_agent_status(agent_name, mock_root)
        assert mock_faf.called, "status poll never reached the agent"
        on_success = mock_faf.call_args.kwargs["success_callback"]
        await on_success(json.dumps(agent_response))

    statuses = [m for m in broadcasts if m.get("type") == "agent_status"]
    assert statuses, f"no agent_status broadcast; got {broadcasts}"
    return statuses[-1]


class TestRestPublishesAllThreeStates:
    async def test_attached_and_disabled(self, agent_service: AgentService) -> None:
        """(а) An agent whose TM the operator switched off.

        The switch is ON-disk truth; ``active`` is false simply because a
        disabled loop never leaves ``inactive``. Reporting only ``active``
        is exactly what made "off" indistinguishable from "idle".
        """
        await agent_service.initialize()

        status = await _rest_detailed_status(
            agent_service,
            "test-agent",
            _agent_response(
                task_manager_active=False,
                task_manager_disabled=True,
            ),
        )

        assert status["task_manager_attached"] is True
        assert status["task_manager_disabled"] is True
        assert status["task_manager_active"] is False

    async def test_attached_enabled_but_idle(
        self, agent_service: AgentService
    ) -> None:
        """(б) Enabled TM sleeping between tasks — the false-positive case.

        ``active=False`` here means "no task right now", NOT "off". The
        switch must be readable as ``disabled=False``.
        """
        await agent_service.initialize()

        status = await _rest_detailed_status(
            agent_service,
            "test-agent",
            _agent_response(
                task_manager_active=False,
                task_manager_disabled=False,
            ),
        )

        assert status["task_manager_attached"] is True
        assert status["task_manager_disabled"] is False
        assert status["task_manager_active"] is False

    async def test_not_attached_reports_false_and_null(
        self, agent_service: AgentService
    ) -> None:
        """(в) No TaskManager: attachment is an explicit false, not a gap.

        A missing key is indistinguishable from "unknown" downstream, so
        attachment must be published as ``False``. With nothing attached
        there is no switch, hence ``disabled is None`` — per the
        WorkControl contract, ``None`` must not be rendered as ``False``.
        """
        await agent_service.initialize()

        status = await _rest_detailed_status(
            agent_service, "test-agent", _agent_response()
        )

        assert "task_manager_attached" in status, (
            "attachment must be an explicit field — key absence is what "
            "consumers cannot read"
        )
        assert status["task_manager_attached"] is False
        assert "task_manager_disabled" in status
        assert status["task_manager_disabled"] is None, (
            "no TaskManager means no switch; False would claim 'enabled'"
        )


class TestCachedMirrorIsNotDowngraded:
    async def test_tm_silent_status_reply_preserves_known_state(
        self, agent_service: AgentService
    ) -> None:
        """A reply that mentions no TaskManager must not erase knowledge.

        Publishing ``attached=False`` in the payload is correct — the
        caller asked and that is the answer for a detached agent. But
        the CACHED ``AgentInfo`` mirror feeds ``GET /api/agents``, and
        overwriting a previously known ``True``/``True`` with
        ``False``/``None`` because one reply happened to omit the keys
        replaces knowledge with a guess.
        """
        await agent_service.initialize()

        info = await agent_service.get_agent("test-agent")
        info.has_task_manager = True
        info.task_manager_disabled = True

        await _ws_status_broadcast(
            agent_service, "test-agent", _agent_response()
        )

        assert info.has_task_manager is True, (
            "a TM-silent reply downgraded known attachment to False"
        )
        assert info.task_manager_disabled is True, (
            "a TM-silent reply erased the known switch position"
        )

    async def test_tm_bearing_reply_does_update_the_mirror(
        self, agent_service: AgentService
    ) -> None:
        """The guard must not freeze the mirror: real news still lands."""
        await agent_service.initialize()

        info = await agent_service.get_agent("test-agent")
        info.has_task_manager = True
        info.task_manager_disabled = True

        await _ws_status_broadcast(
            agent_service,
            "test-agent",
            _agent_response(
                task_manager_active=False,
                task_manager_disabled=False,
            ),
        )

        assert info.has_task_manager is True
        assert info.task_manager_disabled is False, (
            "an operator re-enabling TM must reach the cached mirror"
        )


class TestRestAndBroadcastAgree:
    @pytest.mark.parametrize(
        "tm_keys",
        [
            {"task_manager_active": False, "task_manager_disabled": True},
            {"task_manager_active": False, "task_manager_disabled": False},
            {"task_manager_active": True, "task_manager_disabled": False},
            {},
        ],
        ids=["disabled", "enabled-idle", "enabled-active", "not-attached"],
    )
    async def test_same_agent_same_values(
        self, agent_service: AgentService, tm_keys: dict
    ) -> None:
        """(г) One agent, one truth — whichever way the caller asked.

        REST (``detailed_status_response``) and the WS ``agent_status``
        push read the same agent response, so they must publish the same
        three fields with the same values. Divergence here is how a UI
        and an MCP client end up disagreeing about the same agent.
        """
        await agent_service.initialize()
        response = _agent_response(**tm_keys)

        rest = await _rest_detailed_status(agent_service, "test-agent", response)
        broadcast = await _ws_status_broadcast(agent_service, "test-agent", response)

        fields = ("task_manager_attached", "task_manager_disabled")
        rest_view = {f: rest.get(f) for f in fields}
        ws_view = {f: broadcast.get(f) for f in fields}

        assert rest_view == ws_view, (
            f"REST and WS disagree about the same agent: "
            f"{rest_view} vs {ws_view}"
        )
        for field in fields:
            assert field in rest, f"REST omits {field}"
            assert field in broadcast, f"WS broadcast omits {field}"

        # ``task_manager_active`` obeys ONE rule on both paths: publish it
        # when the agent reported it, omit it when the agent did not. The
        # WS rule is the canonical one (tests/unit/test_status_update.py
        # ::test_broadcast_omits_task_manager_when_absent); the REST
        # snapshot used to default the key to ``False`` instead, which
        # asserts "the loop is idle" about an agent that said nothing —
        # exactly the flattening of UNKNOWN into ``false`` the other two
        # fields forbid.
        if "task_manager_active" in tm_keys:
            assert "task_manager_active" in rest, "REST omits a reported active"
            assert "task_manager_active" in broadcast, "WS omits a reported active"
            assert rest["task_manager_active"] == broadcast["task_manager_active"]
        else:
            assert "task_manager_active" not in broadcast, (
                "WS must omit task_manager_active the agent never reported"
            )
            assert "task_manager_active" not in rest, (
                "REST must omit task_manager_active the agent never reported, "
                f"not default it to {rest.get('task_manager_active')!r}"
            )

    async def test_reported_active_is_coerced_to_bool_on_both_paths(
        self, agent_service: AgentService
    ) -> None:
        """A non-bool ``active`` must not reach consumers raw.

        The WS push coerces with ``bool()``, so REST has to as well: a
        client comparing the two snapshots of one agent would otherwise
        see ``1`` from one path and ``True`` from the other, and a JSON
        consumer doing a strict ``=== true`` would read the raw ``1`` as
        "not active" while the WS client reads "active".
        """
        await agent_service.initialize()
        response = _agent_response(task_manager_active=1, task_manager_disabled=0)

        rest = await _rest_detailed_status(agent_service, "test-agent", response)
        broadcast = await _ws_status_broadcast(agent_service, "test-agent", response)

        assert rest["task_manager_active"] is True, (
            f"REST leaked a raw {rest['task_manager_active']!r}"
        )
        assert broadcast["task_manager_active"] is True, (
            f"WS leaked a raw {broadcast['task_manager_active']!r}"
        )
