"""The TaskManager off switch must stay off.

Reported 2026-10-10: ``set_task_manager(name, enabled=False)`` was accepted and
``get_agent_status`` reported ``task_manager_active=false``, yet the manager
started producing decisions again -- after a new task and after a restart. The
observations in ``rg1_worker.err.log`` pin both halves: ``[TaskManager] Disabled``
appears 9 times against a single ``[TaskManager] Enabled``, and 14
``TaskManager iteration 1`` lines follow the 09:43Z disable with no enable and no
restart in between.

Two independent defects produced that, and each gets its own test here:

1. The evaluation loop in ``_handle_invoke_owned`` gated on ``if
   self.task_manager:`` -- the mere existence of the object -- so a disabled
   manager still drove iterations. ``init_task`` and ``on_subordinate_response``
   each return early when disabled, which is why the loop span but reached
   ``break`` immediately; the user still saw ``TaskManager iteration`` and, via
   the middleware, decisions.
2. ``TaskManager.disabled`` lived only in memory. ``launch_config.task_manager``
   (whether to build the object at all) was persisted, but the operator's
   runtime off switch was not, so every restart resurrected it as ``False``.

The existing ``test_disable_enable.py`` exercises the TaskManager object alone
and passes throughout -- the gap was never in the flag's own setters, it was in
the callers and in persistence. These tests therefore drive the production call
path and the real on-disk metadata instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.task_manager.task_manager import TaskManager


def _make_task_manager(*, session_manager=None) -> TaskManager:
    """A TaskManager with mocked dependencies, mirroring test_disable_enable."""
    mock_config = Mock()
    mock_config.working_dir = Path("/tmp")
    tm = TaskManager(
        agent_name="test-agent",
        config=mock_config,
        session_manager=session_manager if session_manager is not None else Mock(),
    )
    tm._running = True
    tm.manager_agent = Mock()
    return tm


def _fake_agent_process(tm: TaskManager) -> MagicMock:
    """A stand-in self for ``_handle_invoke_owned``.

    Only the attributes that method actually reaches before and inside the
    TaskManager loop are provided, so a failure can be about the gate under
    test rather than a missing mock. ``spec=`` is deliberately NOT used: it
    would block the plain attributes this path assigns.
    """
    fake_self = MagicMock()
    fake_self.name = "test-agent"
    fake_self.task_manager = tm
    fake_self.config = SimpleNamespace(working_dir=Path("/tmp"))
    fake_self.is_running = True
    fake_self._pending_restart_notice = None
    fake_self.current_worker_task = None
    fake_self.agent = MagicMock()
    fake_self.agent.ainvoke = AsyncMock(
        return_value={"messages": [SimpleNamespace(content="done")]}
    )
    fake_self.agent.aget_state = AsyncMock(
        return_value=SimpleNamespace(values={"messages": []})
    )
    fake_self.session_manager = MagicMock()
    fake_self.session_manager.clear_error = AsyncMock()
    fake_self.session_manager.increment_message_count = AsyncMock()
    fake_self.session_manager.increment_tasks_completed = AsyncMock()
    fake_self.session_manager.set_status = AsyncMock()
    fake_self.session_manager.add_message = AsyncMock()
    fake_self._reset_turn_compaction_report = Mock()
    fake_self._is_outgoing_inter_agent_echo = Mock(return_value=False)
    fake_self._idle_metadata_for_response = Mock(return_value=None)
    fake_self._with_compaction_flag = lambda payload: payload
    # Bind the REAL accessor, not the MagicMock's auto-attribute. An
    # auto-generated Mock is never None, so the gate would always open: the
    # loop would run, init_task would be awaited, and the disabled-case
    # assertion would fail for the wrong reason — a falsely red test rather
    # than a vacuous green one. Either way it would prove nothing about the
    # gate, so the accessor under test has to be the real one.
    fake_self._active_task_manager = lambda: AgentProcess._active_task_manager(
        fake_self
    )
    return fake_self


class TestDisabledManagerDoesNotRunOnANewTask:
    """Defect 1: a new task must not drive a disabled manager's loop."""

    @pytest.mark.asyncio
    async def test_disabled_task_manager_runs_no_iteration_on_a_new_task(self):
        """Disabled -> new task -> the evaluation loop must not run.

        Fails on the pre-fix code: the loop was entered because the gate only
        asked whether ``self.task_manager`` existed, so ``init_task`` and
        ``on_subordinate_response`` were both called on a disabled manager.
        """
        tm = _make_task_manager()
        tm.disable()
        tm.init_task = AsyncMock(wraps=tm.init_task)
        tm.on_subordinate_response = AsyncMock(wraps=tm.on_subordinate_response)

        fake_self = _fake_agent_process(tm)

        await AgentProcess._handle_invoke_owned(
            fake_self,
            {"message": "a brand new task", "stream": False},
            notify_callback=None,
        )

        assert tm.init_task.await_count == 0, (
            "a disabled TaskManager was initialised for a new task; the off "
            "switch must gate the loop, not just the inner methods"
        )
        assert tm.on_subordinate_response.await_count == 0, (
            "a disabled TaskManager was asked to evaluate the response"
        )
        assert tm.disabled is True, "the off switch must survive a new task"
        assert tm.state == "inactive", (
            "a disabled TaskManager must stay inactive, so task_manager_active "
            "stays false"
        )

    @pytest.mark.asyncio
    async def test_enabled_task_manager_still_runs_its_loop(self):
        """Positive control: the gate must not suppress an ENABLED manager.

        Without this, making the previous test pass by disabling the loop
        outright would look like success.
        """
        tm = _make_task_manager()
        tm.init_task = AsyncMock()
        tm.on_subordinate_response = AsyncMock(return_value=None)

        fake_self = _fake_agent_process(tm)

        await AgentProcess._handle_invoke_owned(
            fake_self,
            {"message": "a brand new task", "stream": False},
            notify_callback=None,
        )

        assert tm.init_task.await_count == 1, (
            "an enabled TaskManager must still be initialised for a new task"
        )
        assert tm.on_subordinate_response.await_count == 1, (
            "an enabled TaskManager must still evaluate the response"
        )


class TestDisabledSurvivesRestart:
    """Defect 2: the operator's off switch must outlive the process."""

    def test_disabled_is_persisted_and_reloaded(self, tmp_path: Path):
        """Disabled -> restart -> still disabled.

        Fails on the pre-fix code, where ``disabled`` was in-memory only:
        a fresh SessionManager over the same data_dir reported nothing, so the
        rebuilt TaskManager came back enabled.
        """
        data_dir = tmp_path / "agent"
        data_dir.mkdir()

        first = SessionManager(data_dir=data_dir, launch_config={"task_manager": True})
        assert first.task_manager_disabled is False, (
            "a never-configured off switch must default to enabled"
        )
        first.set_task_manager_disabled_sync(True)

        on_disk = json.loads((data_dir / "session_metadata.json").read_text())
        assert on_disk["task_manager_disabled"] is True, (
            "the off switch must reach disk, not just memory"
        )

        # A restart replays launch_config; the off switch must NOT be reset by it.
        second = SessionManager(data_dir=data_dir, launch_config={"task_manager": True})
        assert second.task_manager_disabled is True, (
            "the off switch was lost across a restart"
        )

    def test_explicit_enable_is_also_persisted(self, tmp_path: Path):
        """The symmetric half: turning it back on must stick too.

        A fix that persisted only the disabled state would make the switch
        one-way, so assert the enable direction round-trips as well.
        """
        data_dir = tmp_path / "agent"
        data_dir.mkdir()

        first = SessionManager(data_dir=data_dir, launch_config={"task_manager": True})
        first.set_task_manager_disabled_sync(True)
        first.set_task_manager_disabled_sync(False)

        second = SessionManager(data_dir=data_dir, launch_config={"task_manager": True})
        assert second.task_manager_disabled is False

    def test_absent_key_defaults_to_enabled(self, tmp_path: Path):
        """The default applies only when the switch was never set.

        Metadata written before this field existed must not be read as
        "disabled" -- the default is the documented launch behaviour.
        """
        data_dir = tmp_path / "agent"
        data_dir.mkdir()
        (data_dir / "session_metadata.json").write_text(
            json.dumps(
                {
                    "status": "idle",
                    "tasks_completed": 0,
                    "message_count": 0,
                    "error_message": None,
                    "last_activity": "2026-10-10T20:00:00",
                }
            )
        )

        restored = SessionManager(data_dir=data_dir)
        assert restored.task_manager_disabled is False

    def test_corrupt_value_is_not_read_as_disabled(self, tmp_path: Path):
        """A non-boolean must fall back to the default, not to "off".

        Same tri-state discipline the provenance fields use: only a real
        ``true`` disables the manager.
        """
        data_dir = tmp_path / "agent"
        data_dir.mkdir()
        (data_dir / "session_metadata.json").write_text(
            json.dumps(
                {
                    "status": "idle",
                    "tasks_completed": 0,
                    "message_count": 0,
                    "error_message": None,
                    "last_activity": "2026-10-10T20:00:00",
                    "task_manager_disabled": "yes",
                }
            )
        )

        restored = SessionManager(data_dir=data_dir)
        assert restored.task_manager_disabled is False


class TestStartupHonoursThePersistedSwitch:
    """The rebuilt TaskManager must adopt the persisted state."""

    def test_task_manager_constructor_accepts_the_persisted_switch(self):
        """TaskManager must be constructible as already-disabled.

        Without a constructor seam the persisted value cannot be applied before
        the first turn, which is exactly the window the restart bug lived in.
        """
        mock_config = Mock()
        mock_config.working_dir = Path("/tmp")
        tm = TaskManager(
            agent_name="test-agent",
            config=mock_config,
            session_manager=Mock(),
            disabled=True,
        )
        assert tm.disabled is True

    def test_default_construction_is_enabled(self):
        """Positive control for the constructor seam's default."""
        tm = _make_task_manager()
        assert tm.disabled is False
