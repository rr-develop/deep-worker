"""Tests for ``_terminate_agent`` zombie-reaping fix.

``_terminate_agent`` logs ERROR when even SIGKILL fails to kill a
process.  In practice the SIGKILL *succeeds* but the process becomes
a zombie — it still shows as alive via ``os.kill(pid, 0)`` until the
parent reaps it with ``os.waitpid()``.  The wait loop never detects
the process is gone and falls through to the error path.

The fix: in the force=True (SIGKILL) path, also call
``os.waitpid(pid, os.WNOHANG)`` inside the wait loop so that a zombie
is reaped and the subsequent ``os.kill(pid, 0)`` returns ``False``.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_DW_CLI_ROOT = Path(__file__).resolve().parents[2]
if str(_DW_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_DW_CLI_ROOT))

from tests.integration_ai.agent_fixture import (  # type: ignore[import-not-found]  # noqa: E402
    _terminate_agent,
)
from tests.integration_ai.models import (  # type: ignore[import-not-found]  # noqa: E402
    AgentFixtureInfo,
    AgentStatus,
)


def _make_info(name: str = "test_zombie", pid: int = 99999) -> AgentFixtureInfo:
    return AgentFixtureInfo(
        agent_name=name,
        working_dir=Path("/tmp"),
        rpyc_port=12345,
        pid=pid,
        status=AgentStatus.RUNNING,
        created_by_fixture=True,
    )


class TestTerminateAgentZombieReaping:
    """``_terminate_agent`` must reap zombies after SIGKILL."""

    def test_sigkill_zombie_is_reaped_via_waitpid(self) -> None:
        """When SIGKILL turns the process into a zombie (visible via
        ``os.kill(pid, 0)`` but already dead), ``_terminate_agent`` must
        call ``os.waitpid()`` to reap it so the wait loop exits
        successfully instead of logging ERROR."""
        info = _make_info()
        kill_count = 0
        waitpid_reaped = False

        def _mock_kill(pid: int, sig: int) -> None:
            nonlocal kill_count
            kill_count += 1
            # First call: SIGTERM (succeeds but process becomes zombie)
            # Second call: SIGKILL (succeeds, process stays zombie until reaped)
            pass  # succeed

        alive_calls = 0

        def _mock_is_alive(pid: int) -> bool:
            nonlocal alive_calls, waitpid_reaped
            alive_calls += 1
            if waitpid_reaped:
                return False
            # First few calls: simulate zombie (process visible but dead)
            if alive_calls <= 6:  # cover the 50-loop wait + initial checks
                return True
            return False

        def _mock_waitpid(pid: int, options: int) -> tuple[int, int]:
            nonlocal waitpid_reaped
            waitpid_reaped = True
            return (pid, 0)

        with patch.object(os, "kill", _mock_kill):
            with patch(
                "tests.integration_ai.agent_fixture._is_process_alive",
                _mock_is_alive,
            ):
                with patch.object(os, "waitpid", _mock_waitpid):
                    with patch(
                        "tests.integration_ai.agent_fixture.logger"
                    ) as mock_logger:
                        _terminate_agent(info, force=True)

        # waitpid must have been called
        assert waitpid_reaped, (
            "os.waitpid() was never called — zombie was not reaped"
        )
        # No ERROR log should have been emitted
        error_calls = [
            c for c in mock_logger.mock_calls
            if c[0] == "error"
        ]
        assert not error_calls, (
            f"Expected no ERROR log, got: {error_calls}"
        )
        # INFO about successful termination should be present
        info_calls = [
            c.args[0] for c in mock_logger.method_calls
            if c[0] == "info" and "terminated successfully" in str(c.args)
        ]
        assert info_calls, "Expected 'terminated successfully' info log"

    def test_terminate_already_dead_process(self) -> None:
        """If the process is already dead, _terminate_agent should
        log info and return without attempting to kill."""
        info = _make_info()

        with patch(
            "tests.integration_ai.agent_fixture._is_process_alive",
            return_value=False,
        ):
            with patch(
                "tests.integration_ai.agent_fixture.logger"
            ) as mock_logger:
                _terminate_agent(info, force=False)

        info_calls = [
            c for c in mock_logger.method_calls
            if c[0] == "info" and "already dead" in str(c.args)
        ]
        assert info_calls, "Expected 'already dead' info log for dead process"

    def test_terminate_no_pid(self) -> None:
        """If agent_info.pid is None, log warning and return."""
        info = _make_info(pid=None)

        with patch(
            "tests.integration_ai.agent_fixture.logger"
        ) as mock_logger:
            _terminate_agent(info, force=False)

        warning_calls = [
            c for c in mock_logger.method_calls
            if c[0] == "warning" and "no PID" in str(c.args)
        ]
        assert warning_calls, "Expected 'no PID' warning for None pid"
