"""Tests for AgentProcess."""

import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.exceptions import AgentAlreadyExistsError
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        yield {
            "tmpdir": tmpdir_path,
            "registry_file": tmpdir_path / "registry.json",
            "lock_file": tmpdir_path / "registry.lock",
            "working_dir": tmpdir_path / "working",
            "data_dir": tmpdir_path / "data",
            "pid_file": tmpdir_path / "test.pid",
        }


@pytest.fixture
def mock_registry(temp_dirs):
    """Create a mock registry."""
    return ProcessRegistry(
        registry_file=temp_dirs["registry_file"], lock_file=temp_dirs["lock_file"]
    )


@pytest.fixture
def agent_config(temp_dirs):
    """Create test agent config."""
    return AgentConfig(
        model="claude-sonnet-4-5-20250929",
        working_dir=temp_dirs["working_dir"],
    )


class TestAgentProcessInitialization:
    """Tests for AgentProcess initialization."""

    def test_create_agent_process(self, agent_config, temp_dirs, mock_registry):
        """Test creating an AgentProcess."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        assert process.name == "test-agent"
        assert process.config == agent_config
        assert process.pid_file == temp_dirs["pid_file"]
        assert process.data_dir == temp_dirs["data_dir"]

    def test_data_dir_created(self, agent_config, temp_dirs, mock_registry):
        """Test that data directory is created."""
        data_dir = temp_dirs["tmpdir"] / "new_data"
        assert not data_dir.exists()

        AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=data_dir,
        )

        assert data_dir.exists()


class TestAgentProcessRegistration:
    """Tests for agent process registration."""

    @pytest.mark.asyncio
    async def test_register_in_registry(self, agent_config, temp_dirs, mock_registry):
        """Test that agent registers itself in registry."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        # Register
        await process._register()  # noqa: SLF001

        # Verify in registry
        agent_info = mock_registry.lookup("test-agent")
        assert agent_info is not None
        assert agent_info.name == "test-agent"
        assert agent_info.pid == os.getpid()

    @pytest.mark.asyncio
    async def test_register_duplicate_name(self, agent_config, temp_dirs, mock_registry):
        """Test that registering duplicate name raises error."""
        # Create and register first process
        process1 = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )
        await process1._register()  # noqa: SLF001

        # Try to register second process with same name
        process2 = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["tmpdir"] / "other.pid",
            data_dir=temp_dirs["tmpdir"] / "other_data",
        )

        with pytest.raises(AgentAlreadyExistsError, match="already running"):
            await process2._register()  # noqa: SLF001


class TestAgentProcessPIDFile:
    """Tests for PID file management."""

    @pytest.mark.asyncio
    async def test_create_pid_file(self, agent_config, temp_dirs, mock_registry):
        """Test that PID file is created."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        await process._create_pid_file()  # noqa: SLF001

        assert temp_dirs["pid_file"].exists()
        pid = int(temp_dirs["pid_file"].read_text())
        assert pid == os.getpid()

    @pytest.mark.asyncio
    async def test_remove_pid_file(self, agent_config, temp_dirs, mock_registry):
        """Test that PID file is removed on cleanup."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        await process._create_pid_file()  # noqa: SLF001
        assert temp_dirs["pid_file"].exists()

        await process._remove_pid_file()  # noqa: SLF001
        assert not temp_dirs["pid_file"].exists()


class TestAgentProcessSignalHandling:
    """Tests for signal handling."""

    def test_setup_signal_handlers(self, agent_config, temp_dirs, mock_registry):
        """Test that signal handlers are set up."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        with patch("signal.signal") as mock_signal:
            process._setup_signal_handlers()  # noqa: SLF001

            # Verify SIGTERM and SIGINT handlers set
            assert mock_signal.call_count >= 2
            calls = [call[0] for call in mock_signal.call_args_list]
            assert any(signal.SIGTERM in call for call in calls)
            assert any(signal.SIGINT in call for call in calls)

    def test_signal_handler_sets_running_false(
        self, agent_config, temp_dirs, mock_registry
    ):
        """Test that signal handler stops the agent."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        process.running = True
        process._signal_handler(signal.SIGTERM, None)  # noqa: SLF001

        assert not process.running


class TestAgentProcessLifecycle:
    """Tests for agent lifecycle management."""

    @pytest.mark.asyncio
    async def test_unregister_on_shutdown(self, agent_config, temp_dirs, mock_registry):
        """Test that agent unregisters on shutdown."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        # Register
        await process._register()  # noqa: SLF001
        assert mock_registry.lookup("test-agent") is not None

        # Unregister
        await process._unregister()  # noqa: SLF001
        assert mock_registry.lookup("test-agent") is None
