"""
TDD tests for ProcessRegistry auto-cleanup of dead processes.

NO FALLBACKS POLICY: Dead process entries MUST be automatically cleaned up
when lookup() is called, not left in the registry to cause timeouts.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


class TestProcessRegistryAutoCleanup:
    """Test automatic cleanup of dead process entries in ProcessRegistry."""

    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry files for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            registry_file = Path(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lock', delete=False) as f:
            lock_file = Path(f.name)

        yield ProcessRegistry(registry_file=registry_file, lock_file=lock_file)

        # Cleanup
        if registry_file.exists():
            registry_file.unlink()
        if lock_file.exists():
            lock_file.unlink()

    def test_lookup_returns_none_for_dead_process(self, temp_registry):
        """
        TDD TEST: lookup() should return None for dead process and remove entry.

        NO FALLBACKS POLICY: Dead entries must NOT be returned - they cause timeouts!
        """
        # Register an agent with a PID that doesn't exist
        fake_pid = 999999  # Very unlikely to exist
        agent_info = AgentProcessInfo(
            name="dead_agent",
            pid=fake_pid,
            pid_file=Path("/tmp/dead_agent.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=12345
        )
        temp_registry.register(agent_info)

        # Lookup should return None because process is dead
        result = temp_registry.lookup("dead_agent")
        assert result is None, "lookup() must return None for dead process"

        # Verify entry was removed from registry
        result_after = temp_registry.lookup("dead_agent")
        assert result_after is None, "Dead entry must be removed from registry"

    def test_lookup_returns_info_for_alive_process(self, temp_registry):
        """
        TDD TEST: lookup() should return AgentProcessInfo for alive process.

        NO FALLBACKS POLICY: Alive entries must be returned correctly.
        """
        # Register an agent with current process PID (always alive during test)
        current_pid = os.getpid()
        agent_info = AgentProcessInfo(
            name="alive_agent",
            pid=current_pid,
            pid_file=Path("/tmp/alive_agent.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=54321
        )
        temp_registry.register(agent_info)

        # Lookup should return the agent info
        result = temp_registry.lookup("alive_agent")
        assert result is not None, "lookup() must return info for alive process"
        assert result.name == "alive_agent"
        assert result.pid == current_pid
        assert result.rpyc_port == 54321

    def test_lookup_handles_mixed_alive_and_dead(self, temp_registry):
        """
        TDD TEST: lookup() should handle mix of alive and dead processes correctly.

        NO FALLBACKS POLICY: Only alive processes should be returned.
        """
        current_pid = os.getpid()
        fake_pid = 999998

        # Register alive agent
        alive_agent = AgentProcessInfo(
            name="alive",
            pid=current_pid,
            pid_file=Path("/tmp/alive.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=11111
        )
        temp_registry.register(alive_agent)

        # Register dead agent
        dead_agent = AgentProcessInfo(
            name="dead",
            pid=fake_pid,
            pid_file=Path("/tmp/dead.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=22222
        )
        temp_registry.register(dead_agent)

        # Alive should be found
        alive_result = temp_registry.lookup("alive")
        assert alive_result is not None
        assert alive_result.pid == current_pid

        # Dead should not be found and should be cleaned up
        dead_result = temp_registry.lookup("dead")
        assert dead_result is None

    def test_list_all_excludes_dead_processes(self, temp_registry):
        """
        TDD TEST: list_all() should exclude dead processes.

        NO FALLBACKS POLICY: Dead processes must NOT appear in list.
        """
        current_pid = os.getpid()
        fake_pid = 999997

        # Register alive agent
        alive_agent = AgentProcessInfo(
            name="alive_list",
            pid=current_pid,
            pid_file=Path("/tmp/alive_list.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=33333
        )
        temp_registry.register(alive_agent)

        # Register dead agent
        dead_agent = AgentProcessInfo(
            name="dead_list",
            pid=fake_pid,
            pid_file=Path("/tmp/dead_list.pid"),
            working_dir="/tmp",
            started_at=datetime.now(),
            rpyc_port=44444
        )
        temp_registry.register(dead_agent)

        # list_all() should only return alive agents
        all_agents = temp_registry.list_all()
        agent_names = [a.name for a in all_agents]

        assert "alive_list" in agent_names, "Alive agent must be in list"
        assert "dead_list" not in agent_names, "Dead agent must NOT be in list"
