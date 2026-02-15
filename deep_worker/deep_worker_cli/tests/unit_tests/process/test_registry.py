"""Tests for ProcessRegistry."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from deep_worker_cli.exceptions import AgentAlreadyExistsError
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.types import AgentProcessInfo


@pytest.fixture
def temp_registry_dir():
    """Create a temporary directory for registry files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_registry_dir):
    """Create a ProcessRegistry with temporary files."""
    registry_file = temp_registry_dir / "registry.json"
    lock_file = temp_registry_dir / "registry.lock"
    return ProcessRegistry(registry_file=registry_file, lock_file=lock_file)


@pytest.fixture
def sample_agent_info(temp_registry_dir):
    """Create sample agent process info."""
    return AgentProcessInfo(
        name="test-agent",
        pid=os.getpid(),  # Use current process PID (we know it's alive)
        pid_file=temp_registry_dir / "test.pid",
        working_dir="/tmp/test",
        started_at=datetime.now(),
        rpyc_port=18861,
    )


class TestProcessRegistryRegister:
    """Tests for ProcessRegistry.register()."""

    def test_register_new_agent(self, registry, sample_agent_info):
        """Test registering a new agent."""
        # Should not raise
        registry.register(sample_agent_info)

        # Verify registry file was created
        assert registry.registry_file.exists()

        # Verify agent is in registry
        data = json.loads(registry.registry_file.read_text())
        assert "test-agent" in data
        assert data["test-agent"]["pid"] == sample_agent_info.pid

    def test_register_duplicate_name_same_process(self, registry, sample_agent_info):
        """Test registering agent with duplicate name and alive process."""
        # Register once
        registry.register(sample_agent_info)

        # Try to register again with same name
        duplicate_info = AgentProcessInfo(
            name="test-agent",
            pid=os.getpid(),  # Same PID (alive)
            pid_file=Path("/tmp/other.pid"),
            working_dir="/tmp/other",
            started_at=datetime.now(),
            rpyc_port=18862,
        )

        with pytest.raises(AgentAlreadyExistsError, match="already running"):
            registry.register(duplicate_info)

    def test_register_duplicate_name_dead_process(self, registry, temp_registry_dir):
        """Test registering agent with duplicate name but dead process."""
        # Register with a PID that doesn't exist
        dead_info = AgentProcessInfo(
            name="test-agent",
            pid=999999,  # Unlikely to exist
            pid_file=temp_registry_dir / "dead.pid",
            working_dir="/tmp/dead",
            started_at=datetime.now(),
            rpyc_port=18863,
        )
        registry.register(dead_info)

        # Now register with same name but alive process
        alive_info = AgentProcessInfo(
            name="test-agent",
            pid=os.getpid(),  # Current process (alive)
            pid_file=temp_registry_dir / "alive.pid",
            working_dir="/tmp/alive",
            started_at=datetime.now(),
            rpyc_port=18864,
        )

        # Should succeed (dead process cleaned up)
        registry.register(alive_info)

        # Verify new registration
        data = json.loads(registry.registry_file.read_text())
        assert data["test-agent"]["pid"] == alive_info.pid


class TestProcessRegistryLookup:
    """Tests for ProcessRegistry.lookup()."""

    def test_lookup_existing_agent(self, registry, sample_agent_info):
        """Test looking up an existing agent."""
        registry.register(sample_agent_info)

        result = registry.lookup("test-agent")
        assert result is not None
        assert result.name == "test-agent"
        assert result.pid == sample_agent_info.pid

    def test_lookup_nonexistent_agent(self, registry):
        """Test looking up a non-existent agent."""
        result = registry.lookup("nonexistent")
        assert result is None

    def test_lookup_dead_process(self, registry, temp_registry_dir):
        """Test looking up agent with dead process."""
        # Register with dead PID
        dead_info = AgentProcessInfo(
            name="dead-agent",
            pid=999999,
            pid_file=temp_registry_dir / "dead.pid",
            working_dir="/tmp/dead",
            started_at=datetime.now(),
            rpyc_port=18865,
        )
        registry.register(dead_info)

        # Lookup should return None and clean up
        result = registry.lookup("dead-agent")
        assert result is None

        # Verify it was removed from registry
        data = json.loads(registry.registry_file.read_text())
        assert "dead-agent" not in data


class TestProcessRegistryUnregister:
    """Tests for ProcessRegistry.unregister()."""

    def test_unregister_existing_agent(self, registry, sample_agent_info):
        """Test unregistering an existing agent."""
        registry.register(sample_agent_info)

        # Verify it exists
        assert registry.lookup("test-agent") is not None

        # Unregister
        registry.unregister("test-agent")

        # Verify it's gone
        assert registry.lookup("test-agent") is None

    def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering a non-existent agent (should not error)."""
        # Should not raise
        registry.unregister("nonexistent")


class TestProcessRegistryListAll:
    """Tests for ProcessRegistry.list_all()."""

    def test_list_empty_registry(self, registry):
        """Test listing agents in empty registry."""
        agents = registry.list_all()
        assert len(agents) == 0

    def test_list_multiple_agents(self, registry, temp_registry_dir):
        """Test listing multiple registered agents."""
        # Register multiple agents
        for i in range(3):
            info = AgentProcessInfo(
                name=f"agent-{i}",
                pid=os.getpid(),  # All use current PID (alive)
                pid_file=temp_registry_dir / f"agent{i}.pid",
                working_dir=f"/tmp/agent{i}",
                started_at=datetime.now(),
                rpyc_port=18870 + i,
            )
            registry.register(info)

        agents = registry.list_all()
        assert len(agents) == 3
        names = {agent.name for agent in agents}
        assert names == {"agent-0", "agent-1", "agent-2"}

    def test_list_filters_dead_processes(self, registry, temp_registry_dir):
        """Test that list_all filters out dead processes."""
        # Register alive agent
        alive_info = AgentProcessInfo(
            name="alive-agent",
            pid=os.getpid(),
            pid_file=temp_registry_dir / "alive.pid",
            working_dir="/tmp/alive",
            started_at=datetime.now(),
            rpyc_port=18880,
        )
        registry.register(alive_info)

        # Register dead agent
        dead_info = AgentProcessInfo(
            name="dead-agent",
            pid=999999,
            pid_file=temp_registry_dir / "dead.pid",
            working_dir="/tmp/dead",
            started_at=datetime.now(),
            rpyc_port=18881,
        )
        registry.register(dead_info)

        # List should only return alive agent
        agents = registry.list_all()
        assert len(agents) == 1
        assert agents[0].name == "alive-agent"


class TestProcessRegistryFileLocking:
    """Tests for ProcessRegistry file locking."""

    def test_concurrent_access(self, registry, temp_registry_dir):
        """Test that concurrent access is handled safely."""
        import threading

        results = []

        def register_agent(name):
            try:
                # Extract agent number for unique port
                agent_num = int(name.split('-')[1])
                info = AgentProcessInfo(
                    name=name,
                    pid=os.getpid(),
                    pid_file=temp_registry_dir / f"{name}.pid",
                    working_dir=f"/tmp/{name}",
                    started_at=datetime.now(),
                    rpyc_port=18890 + agent_num,
                )
                registry.register(info)
                results.append((name, True))
            except Exception as e:
                results.append((name, False, str(e)))

        # Create multiple threads trying to register different agents
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_agent, args=(f"agent-{i}",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        assert len(results) == 5
        assert all(r[1] for r in results)

        # Verify all are in registry
        agents = registry.list_all()
        assert len(agents) == 5
