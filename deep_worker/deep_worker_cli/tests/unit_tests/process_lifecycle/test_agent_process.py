"""Tests for AgentProcess."""

import asyncio
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

    def test_setup_signal_handlers_no_loop_falls_back_to_signal(
        self, agent_config, temp_dirs, mock_registry
    ):
        """Without a running loop, setup falls back to signal.signal."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        # No running event loop in this sync test, so add_signal_handler raises
        # and we fall back to signal.signal for SIGTERM and SIGINT.
        with patch("signal.signal") as mock_signal:
            process._setup_signal_handlers()  # noqa: SLF001

            assert mock_signal.call_count >= 2
            calls = [call[0] for call in mock_signal.call_args_list]
            assert any(signal.SIGTERM in call for call in calls)
            assert any(signal.SIGINT in call for call in calls)

    @pytest.mark.asyncio
    async def test_setup_signal_handlers_uses_loop_native(
        self, agent_config, temp_dirs, mock_registry
    ):
        """With a running loop, setup uses loop.add_signal_handler."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        loop = asyncio.get_running_loop()
        with patch.object(loop, "add_signal_handler") as mock_add:
            process._setup_signal_handlers()  # noqa: SLF001

            sigs = {call.args[0] for call in mock_add.call_args_list}
            assert signal.SIGTERM in sigs
            assert signal.SIGINT in sigs

    def test_shutdown_signal_sets_event_not_running(
        self, agent_config, temp_dirs, mock_registry
    ):
        """The shutdown signal must SET shutdown_event and must NOT clear running.

        Regression (2026-05-25): the old handler only flipped ``running=False``
        and never set ``shutdown_event``, so the main loop — blocked on
        ``await self.shutdown_event.wait()`` — never woke and the process hung
        on SIGTERM (stop_agent timed out; kill -9 required). And clearing
        ``running`` here would make the awaited ``shutdown()`` early-return
        before any cleanup, so the handler must leave ``running`` alone.
        """
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )

        process.running = True
        assert not process.shutdown_event.is_set()

        process._handle_shutdown_signal(signal.SIGTERM)  # noqa: SLF001

        assert process.shutdown_event.is_set(), "SIGTERM must set shutdown_event to wake the main loop"
        assert process.running, "handler must not pre-clear running (shutdown() owns that transition)"

    @pytest.mark.asyncio
    async def test_main_loop_wakes_on_shutdown_signal(
        self, agent_config, temp_dirs, mock_registry
    ):
        """End-to-end: firing the signal callback unblocks shutdown_event.wait()."""
        process = AgentProcess(
            name="test-agent",
            config=agent_config,
            registry=mock_registry,
            pid_file=temp_dirs["pid_file"],
            data_dir=temp_dirs["data_dir"],
        )
        process._agent_event_loop = asyncio.get_running_loop()  # noqa: SLF001
        process.shutdown_event.clear()

        # Simulate SIGTERM arriving while the main loop awaits the event.
        async def fire():
            await asyncio.sleep(0.01)
            process._handle_shutdown_signal(signal.SIGTERM)  # noqa: SLF001

        asyncio.create_task(fire())
        # Would hang forever under the old handler; must return promptly now.
        await asyncio.wait_for(process.shutdown_event.wait(), timeout=2.0)


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
