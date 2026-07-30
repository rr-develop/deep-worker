"""Regression: ``daemonize()`` MUST set umask to the normal user default
(``0o022`` → files 644, dirs 755), NOT to ``0`` (files 666, dirs 777).

Background: the classic Unix double-fork daemon recipe calls ``os.umask(0)``
so that a daemon writing files with *explicit* modes does not depend on the
caller's umask. Our daemon runs an LLM agent that creates
user-visible artifacts (via ``execute`` / ``write_file``) *without* explicit
mode, so the daemon must behave like a normal interactive user shell.
System default is ``UMASK 022`` (``/etc/login.defs``), giving 644/755.

The fix, guarded here: replace ``os.umask(0)`` with ``os.umask(0o022)`` in
``daemonize()`` (agent_command.py), exposed as the module-level constant
``DAEMON_UMASK``.

This test is a **behavioural** guard — it does not grep the source for
``os.umask(0)`` (that would be brittle: ``0`` / ``0o0`` / ``000`` all match).
Instead it mocks ``os.umask`` and asserts the value actually passed in the
``daemonize()`` code path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_DW_CLI_ROOT = Path(__file__).resolve().parents[2]
if str(_DW_CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(_DW_CLI_ROOT))

from deep_worker_cli.cli import agent_command  # noqa: E402


class TestDaemonUmaskConstant:
    """The daemon umask value MUST be the exported module-level constant."""

    def test_daemon_umask_constant_is_022(self) -> None:
        """``DAEMON_UMASK`` MUST equal ``0o022`` (system default → 644/755)."""
        assert hasattr(agent_command, "DAEMON_UMASK"), (
            "agent_command must expose DAEMON_UMASK constant so callers/tests "
            "and reviewers can inspect the exact value used at daemonize time."
        )
        assert agent_command.DAEMON_UMASK == 0o022, (
            f"DAEMON_UMASK must be 0o022 (files 644, dirs 755), "
            f"got {oct(agent_command.DAEMON_UMASK)}. Any other value re-opens "
            f"the '666/777 everywhere' bug."
        )


class TestDaemonizeAppliesCorrectUmask:
    """``daemonize()`` MUST call ``os.umask`` with ``0o022``, not with ``0``.

    Uses behavioural mocking: we don't inspect the source text, we record the
    value actually passed to ``os.umask`` at runtime along the daemonize
    code path. Any regression back to ``os.umask(0)`` (or ``0o0``, ``000``,
    …) fails this test.
    """

    def test_daemonize_sets_umask_to_0o022(self) -> None:
        # Fork strategy:
        #  1st fork: return 0 → we are the "child", flow continues to setsid+umask.
        #  2nd fork: return 1 → child exits via sys.exit(0), we catch SystemExit.
        fork_results = iter([0, 1])
        umask_calls: list[int] = []

        def _fake_fork() -> int:
            return next(fork_results)

        def _record_umask(value: int) -> int:
            umask_calls.append(value)
            return 0  # previous umask (irrelevant for the test)

        with (
            patch.object(agent_command.os, "fork", _fake_fork),
            patch.object(agent_command.os, "chdir"),
            patch.object(agent_command.os, "setsid"),
            patch.object(agent_command.os, "umask", _record_umask),
            patch.object(agent_command, "_reap_empty_cgroup_leaves"),
            patch.object(agent_command, "_detach_from_session_scope"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                agent_command.daemonize(
                    "test_agent_umask", log_dir=Path("/tmp/dw_umask_test_logs")
                )

        # First (grand)child exits with 0 after second fork returns pid>0.
        assert exc_info.value.code == 0

        # os.umask MUST have been called exactly once (right after setsid).
        assert len(umask_calls) == 1, (
            f"daemonize() called os.umask {len(umask_calls)} times, expected 1. "
            f"Values: {[oct(v) for v in umask_calls]}"
        )
        # And that single value MUST be 0o022 — NOT 0, NOT 0o0, NOT 000.
        assert umask_calls[0] == 0o022, (
            f"daemonize() passed umask={oct(umask_calls[0])} — expected 0o022. "
            f"Any other value re-opens the '666/777' bug (see "
            f"docs/DESIGN_FIX_DAEMONIZE_UMASK.md)."
        )
