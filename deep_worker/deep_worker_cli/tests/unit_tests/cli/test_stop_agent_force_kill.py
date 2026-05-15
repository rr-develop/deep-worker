"""TDD tests for dwcli --stop force killing frozen agents.

This test suite ensures that `dwcli --stop` ALWAYS stops agents,
even if they are frozen or unresponsive to SIGTERM.

REQUIREMENTS:
1. Graceful shutdown with SIGTERM (wait up to 5 seconds)
2. Force kill with SIGKILL if SIGTERM doesn't work
3. Verify process is ACTUALLY dead before returning success
4. Report if process couldn't be killed (permission denied, etc.)
5. Clean up registry entries for dead processes
"""

import asyncio
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.cli.client_command import stop_agent
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


class TestStopAgentForceKill:
    """Test that dwcli --stop kills frozen agents."""

    @pytest.mark.asyncio
    async def test_stop_graceful_shutdown(self, tmp_path):
        """Test that stop_agent() successfully stops responsive agent with SIGTERM."""
        # Create a dummy agent that responds to SIGTERM
        # Use custom registry file to avoid conflicts
        registry_file = tmp_path / "test_graceful_registry.json"
        registry = ProcessRegistry(registry_file=registry_file)

        # Start a simple python process that will respond to SIGTERM.
        # NOTE: `ProcessRegistry._is_process_alive()` greps /proc/<pid>/cmdline
        # for 'deep-worker', 'deep_worker', 'pytest' or 'python'. A raw
        # `sleep 300` fails that check and `lookup()` reports the entry
        # as stale — so we spawn python instead.
        import subprocess
        import sys
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        pid = proc.pid

        # Register in registry
        agent_info = AgentProcessInfo(
            name="test_graceful",
            pid=pid,
            pid_file=tmp_path / f"test_graceful_{pid}.pid",
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=12345,
        )
        registry.register(agent_info)

        try:
            # Patch stop_agent to use our custom registry
            from unittest.mock import patch
            with patch('deep_worker_cli.cli.client_command.ProcessRegistry', return_value=registry):
                # Stop should succeed
                exit_code = await stop_agent("test_graceful")
                assert exit_code == 0, "stop_agent should return 0 for successful stop"

            # Process should be dead (or zombie)
            await asyncio.sleep(0.5)  # Brief wait

            # Check if process is dead or zombie
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read()
                    parts = stat.split(") ")
                    state = parts[1].split()[0]
                    assert state == "Z", f"Process should be zombie, got state '{state}'"
            except FileNotFoundError:
                # Process completely gone - even better!
                pass

        finally:
            # Cleanup: kill process if still alive
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            registry.unregister("test_graceful")

    @pytest.mark.asyncio
    async def test_stop_frozen_agent_with_sigkill(self):
        """Test that stop_agent() KILLS frozen agent that ignores SIGTERM.

        This is the CRITICAL test: frozen agents MUST be killed with SIGKILL.
        """
        registry = ProcessRegistry()

        # Create a process that ignores SIGTERM (like a frozen agent)
        # We'll use a Python subprocess that catches and ignores SIGTERM
        import subprocess
        import sys

        # Create a script that ignores SIGTERM
        ignore_sigterm_script = """
import signal
import time

# Ignore SIGTERM (like a frozen agent)
signal.signal(signal.SIGTERM, signal.SIG_IGN)

# Block forever
while True:
    time.sleep(1)
"""

        proc = subprocess.Popen([sys.executable, "-c", ignore_sigterm_script])
        pid = proc.pid

        # Register in registry
        agent_info = AgentProcessInfo(
            name="test_frozen",
            pid=pid,
            pid_file=Path(f"/tmp/test_frozen_{pid}.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=12346,
        )
        registry.register(agent_info)

        try:
            # Verify process is actually running
            os.kill(pid, 0)  # Should NOT raise (process exists)

            # Stop should succeed even though process ignores SIGTERM
            exit_code = await stop_agent("test_frozen")

            # CRITICAL: Must return 0 (success) because agent MUST be killed
            assert exit_code == 0, "stop_agent MUST succeed even for frozen agents"

            # CRITICAL: Process MUST be dead (or zombie) after stop_agent returns
            await asyncio.sleep(0.5)  # Brief wait

            # Check process state - should be zombie or gone
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read()
                    parts = stat.split(") ")
                    if len(parts) >= 2:
                        state = parts[1].split()[0]
                        assert state == "Z", \
                            f"Process {pid} should be zombie (Z) after SIGKILL, got state '{state}'"
            except FileNotFoundError:
                # Process completely gone - perfect!
                pass

        finally:
            # Cleanup: force kill if still alive
            try:
                os.kill(pid, signal.SIGKILL)
                await asyncio.sleep(0.2)
            except ProcessLookupError:
                pass
            registry.unregister("test_frozen")

    @pytest.mark.asyncio
    async def test_stop_cleans_registry_for_dead_process(self):
        """Test that stop_agent() cleans up registry for already-dead processes."""
        registry = ProcessRegistry()

        # Create a fake PID that doesn't exist
        fake_pid = 999999  # Very unlikely to exist

        # Register fake agent
        agent_info = AgentProcessInfo(
            name="test_dead",
            pid=fake_pid,
            pid_file=Path(f"/tmp/test_dead_{fake_pid}.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=12347,
        )
        registry.register(agent_info)

        # Stop should clean up registry even though process doesn't exist
        exit_code = await stop_agent("test_dead")

        # Should return error (1) because process not found
        assert exit_code == 1, "stop_agent should return 1 for already-dead process"

        # Registry should be cleaned up
        assert registry.lookup("test_dead") is None, "Registry should be cleaned up"

    @pytest.mark.asyncio
    async def test_stop_reports_progress(self, capsys):
        """Test that stop_agent() reports what it's doing."""
        registry = ProcessRegistry()

        # Spawn a python subprocess — see the note on
        # `_is_process_alive` in test_stop_graceful_shutdown above.
        import subprocess
        import sys
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        pid = proc.pid

        # Register
        agent_info = AgentProcessInfo(
            name="test_progress",
            pid=pid,
            pid_file=Path(f"/tmp/test_progress_{pid}.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=12348,
        )
        registry.register(agent_info)

        try:
            # Stop agent
            exit_code = await stop_agent("test_progress")

            # Check output
            captured = capsys.readouterr()
            assert "Stopping agent" in captured.out, "Should report stopping"
            assert str(pid) in captured.out, "Should show PID"

            # Should report success OR timeout (both acceptable for responsive process)
            assert exit_code == 0 or "force" in captured.out.lower()

        finally:
            # Cleanup
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            registry.unregister("test_progress")

    @pytest.mark.asyncio
    async def test_stop_timeout_then_kill(self):
        """Test that stop_agent() tries SIGTERM, then SIGKILL after timeout.

        REQUIREMENTS:
        1. Send SIGTERM first
        2. Wait up to 5 seconds
        3. If still alive, send SIGKILL
        4. Report what happened
        """
        registry = ProcessRegistry()

        # Create process that ignores SIGTERM
        import subprocess
        import sys

        ignore_script = """
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True: time.sleep(0.1)
"""

        proc = subprocess.Popen([sys.executable, "-c", ignore_script])
        pid = proc.pid

        agent_info = AgentProcessInfo(
            name="test_timeout",
            pid=pid,
            pid_file=Path(f"/tmp/test_timeout_{pid}.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=12349,
        )
        registry.register(agent_info)

        try:
            start = time.time()
            exit_code = await stop_agent("test_timeout")
            duration = time.time() - start

            # Should succeed
            assert exit_code == 0, "Must succeed even with force kill"

            # Should take at least a few seconds (tried SIGTERM first)
            # But NOT more than 10 seconds total
            assert 1.0 <= duration <= 10.0, \
                f"Should try SIGTERM (few seconds), then SIGKILL. Took {duration}s"

            # Process MUST be dead or zombie
            await asyncio.sleep(0.5)
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read()
                    parts = stat.split(") ")
                    state = parts[1].split()[0]
                    assert state == "Z", \
                        f"Process {pid} should be zombie after stop_agent, got '{state}'"
            except FileNotFoundError:
                # Process gone - perfect!
                pass

        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            registry.unregister("test_timeout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
