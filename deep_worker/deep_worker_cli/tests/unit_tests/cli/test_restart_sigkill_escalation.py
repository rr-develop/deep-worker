"""`--restart` must not leave the agent DOWN when shutdown is slow.

The old behaviour: ``stop_agent`` sent SIGTERM, waited a
flat 10s, and raised ``TimeoutError``; ``restart_agent`` turned that into
``return 1``, so step 5 (the actual start) never ran. The old process died
anyway — just later than the budget — and nothing replaced it.

Observed live on a production agent: a 644 MB ``threads.db`` pushed
the checkpointer's flush/close past 10s.

These tests pin the escalation contract:

    SIGTERM -> scaled grace -> SIGKILL -> kill grace -> verify -> start

The signal assertions check the *signal number*, not merely that
``os.kill`` was called: an assertion on the call count alone passes even
if the code only ever sends SIGTERM, which is exactly the bug.
"""

import argparse
import signal
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.types import AgentProcessInfo


PID = 99999


def _signals_sent(mock_kill: MagicMock) -> list[int]:
    """Signal numbers passed to ``os.kill``, in order."""
    return [call.args[1] for call in mock_kill.call_args_list]


def _agent_info(name: str = "myagent") -> AgentProcessInfo:
    return AgentProcessInfo(
        name=name,
        pid=PID,
        pid_file=Path(f"/tmp/deep_worker_{name}.pid"),
        working_dir="/tmp/test",
        started_at=datetime.now(),
        rpyc_port=18861,
    )


@pytest.fixture
def registry() -> MagicMock:
    from deep_worker_cli.process.registry import ProcessRegistry

    reg = MagicMock(spec=ProcessRegistry)
    reg.lookup.return_value = _agent_info()
    return reg


@pytest.fixture
def idle_data_dir(tmp_path: Path) -> Path:
    """Agent data dir whose metadata says the agent is idle (stoppable)."""
    (tmp_path / "session_metadata.json").write_text('{"status": "idle"}')
    return tmp_path


# ===========================================================================
# stop_agent: escalation
# ===========================================================================

class TestStopAgentEscalation:
    """SIGTERM grace expiry must escalate to SIGKILL, not give up."""

    @pytest.mark.asyncio
    async def test_sigterm_timeout_escalates_to_sigkill(self, idle_data_dir, registry):
        """The whole point of the fix: a slow shutdown gets SIGKILLed."""
        from deep_worker_cli.process.agent_stop import stop_agent

        # Alive until SIGKILL lands, dead immediately after — keyed off the
        # signal itself rather than a call count, so the assertion does not
        # depend on how many liveness checks the backoff happens to fit
        # into the grace window.
        killed = False

        def fake_kill(_pid: int, signum: int) -> None:
            nonlocal killed
            if signum == signal.SIGKILL:
                killed = True

        with patch(
                 "deep_worker_cli.process.agent_stop.os.kill",
                 side_effect=fake_kill,
             ) as mock_kill, \
             patch(
                 "deep_worker_cli.process.agent_stop._is_process_alive",
                 side_effect=lambda _pid: not killed,
             ):
            result = await stop_agent(
                "myagent",
                registry,
                data_dir=idle_data_dir,
                timeout=0.3,
                kill_grace=0.3,
            )

        assert result is True
        sent = _signals_sent(mock_kill)
        assert signal.SIGTERM in sent, f"SIGTERM must be tried first; sent={sent}"
        assert signal.SIGKILL in sent, (
            f"SIGTERM grace expired — SIGKILL MUST follow. Signals sent: {sent}"
        )
        assert sent.index(signal.SIGTERM) < sent.index(signal.SIGKILL), (
            f"SIGTERM must precede SIGKILL; sent={sent}"
        )

    @pytest.mark.asyncio
    async def test_graceful_exit_never_sends_sigkill(self, idle_data_dir, registry):
        """A well-behaved agent must not be SIGKILLed."""
        from deep_worker_cli.process.agent_stop import stop_agent

        with patch("deep_worker_cli.process.agent_stop.os.kill") as mock_kill, \
             patch(
                 "deep_worker_cli.process.agent_stop._is_process_alive",
                 return_value=False,
             ):
            result = await stop_agent("myagent", registry, data_dir=idle_data_dir)

        assert result is True
        sent = _signals_sent(mock_kill)
        assert sent == [signal.SIGTERM], (
            f"Graceful shutdown must send SIGTERM only; sent={sent}"
        )

    @pytest.mark.asyncio
    async def test_surviving_sigkill_raises_timeout(self, idle_data_dir, registry):
        """Unkillable (D-state) process is honestly reported, not swallowed."""
        from deep_worker_cli.process.agent_stop import stop_agent

        with patch("deep_worker_cli.process.agent_stop.os.kill") as mock_kill, \
             patch(
                 "deep_worker_cli.process.agent_stop._is_process_alive",
                 return_value=True,
             ), \
             pytest.raises(TimeoutError, match="SIGKILL"):
            await stop_agent(
                "myagent",
                registry,
                data_dir=idle_data_dir,
                timeout=0.2,
                kill_grace=0.2,
            )

        assert signal.SIGKILL in _signals_sent(mock_kill)

    @pytest.mark.asyncio
    async def test_force_false_keeps_the_old_contract(self, idle_data_dir, registry):
        """``force=False`` must never escalate — it is the opt-out."""
        from deep_worker_cli.process.agent_stop import stop_agent

        with patch("deep_worker_cli.process.agent_stop.os.kill") as mock_kill, \
             patch(
                 "deep_worker_cli.process.agent_stop._is_process_alive",
                 return_value=True,
             ), \
             pytest.raises(TimeoutError):
            await stop_agent(
                "myagent",
                registry,
                data_dir=idle_data_dir,
                timeout=0.2,
                force=False,
            )

        sent = _signals_sent(mock_kill)
        assert signal.SIGKILL not in sent, (
            f"force=False must not send SIGKILL; sent={sent}"
        )

    @pytest.mark.parametrize("escalates", [False, True], ids=["sigterm", "sigkill"])
    @pytest.mark.asyncio
    async def test_registry_entry_removed_after_stop(
        self, idle_data_dir, registry, escalates
    ):
        """A stopped agent must not leave a live registry entry behind.

        Both exit paths are covered on purpose: a version that unregistered
        only after a graceful SIGTERM left a stale entry behind whenever the
        agent had to be killed — and the next start would then see itself as
        already running. Asserting on the graceful path alone does not catch
        that, so this is parametrized rather than a single scenario.
        """
        from deep_worker_cli.process.agent_stop import stop_agent

        killed = False

        def fake_kill(_pid: int, signum: int) -> None:
            nonlocal killed
            if signum == signal.SIGKILL:
                killed = True

        # sigterm path: dead on the first check. sigkill path: alive until
        # SIGKILL lands, so the SIGTERM grace has to expire first.
        def alive(_pid: int) -> bool:
            return escalates and not killed

        with patch(
                 "deep_worker_cli.process.agent_stop.os.kill",
                 side_effect=fake_kill,
             ) as mock_kill, \
             patch(
                 "deep_worker_cli.process.agent_stop._is_process_alive",
                 side_effect=alive,
             ):
            await stop_agent(
                "myagent",
                registry,
                data_dir=idle_data_dir,
                timeout=0.3,
                kill_grace=0.3,
            )

        assert (signal.SIGKILL in _signals_sent(mock_kill)) is escalates, (
            "the scenario did not exercise the path it claims to"
        )
        registry.unregister.assert_called_once_with("myagent")


class TestStopAgentAgainstARealProcess:
    """Mocks cannot expose the kill -> check -> ProcessLookupError race."""

    @pytest.mark.asyncio
    async def test_process_ignoring_sigterm_is_really_killed(self, idle_data_dir):
        """A live process that ignores SIGTERM must be dead when we return."""
        import subprocess
        import sys

        from deep_worker_cli.process.agent_stop import stop_agent
        from deep_worker_cli.process.registry import ProcessRegistry

        # The child announces readiness on stdout AFTER installing the
        # handler: without that handshake SIGTERM can land during
        # interpreter startup and kill it, so the test would "pass" having
        # never exercised the escalation at all.
        ignore_sigterm = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "while True: time.sleep(1)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", ignore_sigterm],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready", (
            "child never signalled that SIGTERM is ignored"
        )

        registry = MagicMock(spec=ProcessRegistry)
        registry.lookup.return_value = AgentProcessInfo(
            name="frozen",
            pid=proc.pid,
            pid_file=Path(f"/tmp/deep_worker_frozen_{proc.pid}.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=18862,
        )

        try:
            result = await stop_agent(
                "frozen",
                registry,
                data_dir=idle_data_dir,
                timeout=1.0,
                kill_grace=5.0,
            )

            assert result is True
            # Reap the corpse so the liveness check below is not fooled by a
            # zombie that /proc still lists.
            proc.wait(timeout=5)
            assert proc.returncode is not None
            assert proc.returncode < 0, (
                f"expected death by signal, got returncode={proc.returncode}"
            )
            assert -proc.returncode == signal.SIGKILL, (
                f"expected SIGKILL, got signal {-proc.returncode}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


# ===========================================================================
# Grace budget scaling
# ===========================================================================

class TestGraceBudget:
    """A flat 10s does not fit any database size (e.g. 644 MB)."""

    def test_small_db_gets_the_base_budget(self, tmp_path):
        from deep_worker_cli.process.agent_stop import compute_shutdown_grace

        (tmp_path / "threads.db").write_bytes(b"x" * 1024)
        assert compute_shutdown_grace(tmp_path) == 10.0

    def test_missing_db_gets_the_base_budget(self, tmp_path):
        from deep_worker_cli.process.agent_stop import compute_shutdown_grace

        assert compute_shutdown_grace(tmp_path) == 10.0

    @pytest.mark.parametrize(
        "megabytes,expected",
        [
            (50, 10.0),    # below the first threshold
            (150, 20.0),   # > 100 MB
            (600, 40.0),   # > 500 MB  (the incident: 644 MB)
            (1500, 60.0),  # > 1000 MB
        ],
    )
    def test_budget_scales_with_db_size(self, tmp_path, megabytes, expected):
        from deep_worker_cli.process.agent_stop import compute_shutdown_grace

        db = tmp_path / "threads.db"
        with db.open("wb") as fh:
            fh.truncate(megabytes * 1024 * 1024)

        assert compute_shutdown_grace(tmp_path) == expected

    def test_budget_is_capped(self, tmp_path):
        """No database size may produce an unbounded wait."""
        from deep_worker_cli.process.agent_stop import compute_shutdown_grace

        db = tmp_path / "threads.db"
        with db.open("wb") as fh:
            fh.truncate(50 * 1024 * 1024 * 1024)  # 50 GB, sparse

        assert compute_shutdown_grace(tmp_path) == 60.0


# ===========================================================================
# restart_agent: the incident itself
# ===========================================================================

def _restart_args(name: str = "myagent") -> argparse.Namespace:
    return argparse.Namespace(name=name, clean=False, restart=True)


class TestRestartContinuesAfterEscalation:
    """`--restart` must end with a *running* agent whenever it can."""

    @pytest.mark.asyncio
    async def test_slow_shutdown_still_starts_the_new_process(self):
        """THE regression: stop was slow, so the agent was never restarted."""
        from deep_worker_cli.cli import agent_command

        effective = _restart_args()

        with patch.object(agent_command, "load_launch_config", return_value={"provider": "x"}), \
             patch.object(agent_command, "merge_restart_config", return_value=effective), \
             patch.object(agent_command, "_get_agent_data_dir", return_value=Path("/tmp")), \
             patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_registry, \
             patch(
                 "deep_worker_cli.process.agent_stop.stop_agent",
                 new=AsyncMock(return_value=True),
             ) as mock_stop, \
             patch.object(agent_command, "run_agent", new=AsyncMock(return_value=0)) as mock_run:
            mock_registry.return_value.lookup.return_value = _agent_info()
            rc = await agent_command.restart_agent(_restart_args())

        assert rc == 0
        mock_run.assert_awaited_once()
        # force=True is what makes the escalation happen at all.
        assert mock_stop.await_args.kwargs.get("force") is True, (
            "restart must request forced shutdown, otherwise a slow agent "
            "aborts the restart and the agent stays DOWN"
        )

    @pytest.mark.asyncio
    async def test_unkillable_agent_aborts_instead_of_double_start(self):
        """Two processes on one data_dir is worse than DOWN — refuse to start."""
        from deep_worker_cli.cli import agent_command

        effective = _restart_args()

        with patch.object(agent_command, "load_launch_config", return_value={"provider": "x"}), \
             patch.object(agent_command, "merge_restart_config", return_value=effective), \
             patch.object(agent_command, "_get_agent_data_dir", return_value=Path("/tmp")), \
             patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_registry, \
             patch(
                 "deep_worker_cli.process.agent_stop.stop_agent",
                 new=AsyncMock(side_effect=TimeoutError("survived SIGKILL")),
             ), \
             patch.object(agent_command, "run_agent", new=AsyncMock(return_value=0)) as mock_run:
            mock_registry.return_value.lookup.return_value = _agent_info()
            rc = await agent_command.restart_agent(_restart_args())

        assert rc == 1
        mock_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_actively_running_agent_still_refuses(self):
        """RuntimeError (status=running) keeps aborting — orthogonal contract."""
        from deep_worker_cli.cli import agent_command

        effective = _restart_args()

        with patch.object(agent_command, "load_launch_config", return_value={"provider": "x"}), \
             patch.object(agent_command, "merge_restart_config", return_value=effective), \
             patch.object(agent_command, "_get_agent_data_dir", return_value=Path("/tmp")), \
             patch("deep_worker_cli.process.registry.ProcessRegistry") as mock_registry, \
             patch(
                 "deep_worker_cli.process.agent_stop.stop_agent",
                 new=AsyncMock(side_effect=RuntimeError("actively running")),
             ), \
             patch.object(agent_command, "run_agent", new=AsyncMock(return_value=0)) as mock_run:
            mock_registry.return_value.lookup.return_value = _agent_info()
            rc = await agent_command.restart_agent(_restart_args())

        assert rc == 1
        mock_run.assert_not_awaited()
