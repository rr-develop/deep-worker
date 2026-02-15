"""Unit tests for agent_command working directory handling.

Tests that agent correctly uses the working directory from which it was launched,
even after daemonization.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deep_worker_cli.cli.agent_command import run_agent, main


class TestAgentCommandWorkingDirectory:
    """Tests for working directory handling in agent_command."""

    @pytest.mark.asyncio
    async def test_uses_current_directory_when_no_dir_arg(self):
        """Test that agent uses current directory when --dir is not specified."""
        # Create a temporary directory to simulate launching from
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                # Change to temp directory (simulating user running command from there)
                os.chdir(tmpdir)

                # Create args without --dir
                args = argparse.Namespace(
                    name="test_agent",
                    dir=None,  # Not specified
                    model="deepseek-chat",
                    provider="deepseek",
                    skip_permissions=True,
                    task_manager=False,
                    config=None,
                )

                # Mock AgentProcess to capture the config
                captured_config = None

                def mock_agent_init(self, name, config, skip_permissions, enable_task_manager):
                    nonlocal captured_config
                    captured_config = config
                    self.name = name
                    self.running = False

                async def mock_start(self):
                    pass

                with patch('deep_worker_cli.cli.agent_command.AgentProcess.__init__', mock_agent_init):
                    with patch('deep_worker_cli.cli.agent_command.AgentProcess.start', mock_start):
                        # Run agent
                        await run_agent(args)

                # Verify: working_dir should be the temp directory, NOT root
                assert captured_config is not None
                assert captured_config.working_dir == Path(tmpdir)
                assert captured_config.working_dir != Path("/")

            finally:
                os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_uses_env_var_when_cwd_is_root(self):
        """Test that agent uses DEEP_WORKER_ORIGINAL_CWD when cwd is / (after daemonize)."""
        # Simulate situation after daemonize: cwd is /, but env var has original dir
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                # Simulate daemonize: chdir to root
                os.chdir("/")

                # Set environment variable with original directory
                os.environ["DEEP_WORKER_ORIGINAL_CWD"] = tmpdir

                # Create args without --dir
                args = argparse.Namespace(
                    name="test_agent",
                    dir=None,
                    model="deepseek-chat",
                    provider="deepseek",
                    skip_permissions=True,
                    task_manager=False,
                    config=None,
                )

                # Mock AgentProcess to capture the config
                captured_config = None

                def mock_agent_init(self, name, config, skip_permissions, enable_task_manager):
                    nonlocal captured_config
                    captured_config = config
                    self.name = name
                    self.running = False

                async def mock_start(self):
                    pass

                with patch('deep_worker_cli.cli.agent_command.AgentProcess.__init__', mock_agent_init):
                    with patch('deep_worker_cli.cli.agent_command.AgentProcess.start', mock_start):
                        # Run agent
                        await run_agent(args)

                # Verify: working_dir should be from env var, NOT root
                assert captured_config is not None
                assert captured_config.working_dir == Path(tmpdir)
                assert captured_config.working_dir != Path("/")

            finally:
                os.chdir(original_cwd)
                os.environ.pop("DEEP_WORKER_ORIGINAL_CWD", None)

    @pytest.mark.asyncio
    async def test_dir_arg_takes_precedence(self):
        """Test that --dir argument takes precedence over env var and cwd."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                original_cwd = os.getcwd()
                try:
                    os.chdir("/")
                    os.environ["DEEP_WORKER_ORIGINAL_CWD"] = tmpdir1

                    # Create args WITH --dir
                    args = argparse.Namespace(
                        name="test_agent",
                        dir=Path(tmpdir2),  # Explicitly specified
                        model="deepseek-chat",
                        provider="deepseek",
                        skip_permissions=True,
                        task_manager=False,
                        config=None,
                    )

                    # Mock AgentProcess
                    captured_config = None

                    def mock_agent_init(self, name, config, skip_permissions, enable_task_manager):
                        nonlocal captured_config
                        captured_config = config
                        self.name = name
                        self.running = False

                    async def mock_start(self):
                        pass

                    with patch('deep_worker_cli.cli.agent_command.AgentProcess.__init__', mock_agent_init):
                        with patch('deep_worker_cli.cli.agent_command.AgentProcess.start', mock_start):
                            await run_agent(args)

                    # Verify: working_dir should be from --dir arg
                    assert captured_config is not None
                    assert captured_config.working_dir == Path(tmpdir2)
                    assert captured_config.working_dir != Path(tmpdir1)
                    assert captured_config.working_dir != Path("/")

                finally:
                    os.chdir(original_cwd)
                    os.environ.pop("DEEP_WORKER_ORIGINAL_CWD", None)


class TestMainSetsEnvironmentVariable:
    """Tests that main() sets DEEP_WORKER_ORIGINAL_CWD before daemonizing."""

    def test_main_sets_env_var_from_cwd(self):
        """Test that main() sets env var to current directory when --dir not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            original_env = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
            try:
                # Change to temp directory
                os.chdir(tmpdir)

                # Mock sys.argv to simulate command-line args
                with patch.object(sys, 'argv', ['deep-worker', 'test_agent']):
                    # Mock daemonize and asyncio.run to prevent actual execution
                    def mock_daemonize(name):
                        # Check that env var was set BEFORE daemonize was called
                        assert "DEEP_WORKER_ORIGINAL_CWD" in os.environ
                        assert os.environ["DEEP_WORKER_ORIGINAL_CWD"] == tmpdir

                    with patch('deep_worker_cli.cli.agent_command.daemonize', mock_daemonize):
                        with patch('deep_worker_cli.cli.agent_command.asyncio.run', return_value=0):
                            # Call main()
                            main()

                # Verify env var was set correctly
                assert os.environ["DEEP_WORKER_ORIGINAL_CWD"] == tmpdir

            finally:
                os.chdir(original_cwd)
                if original_env is not None:
                    os.environ["DEEP_WORKER_ORIGINAL_CWD"] = original_env
                else:
                    os.environ.pop("DEEP_WORKER_ORIGINAL_CWD", None)

    def test_main_sets_env_var_from_dir_arg(self):
        """Test that main() sets env var to --dir value when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            original_env = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
            try:
                # Start from a different directory
                os.chdir("/tmp")

                # Mock sys.argv with --dir argument
                with patch.object(sys, 'argv', ['deep-worker', 'test_agent', '--dir', tmpdir]):
                    # Mock daemonize and asyncio.run
                    def mock_daemonize(name):
                        # Check that env var was set to --dir value, NOT cwd
                        assert "DEEP_WORKER_ORIGINAL_CWD" in os.environ
                        # Resolve tmpdir to absolute path for comparison
                        assert os.environ["DEEP_WORKER_ORIGINAL_CWD"] == str(Path(tmpdir).resolve())

                    with patch('deep_worker_cli.cli.agent_command.daemonize', mock_daemonize):
                        with patch('deep_worker_cli.cli.agent_command.asyncio.run', return_value=0):
                            # Call main()
                            main()

                # Verify env var was set correctly
                assert os.environ["DEEP_WORKER_ORIGINAL_CWD"] == str(Path(tmpdir).resolve())

            finally:
                os.chdir(original_cwd)
                if original_env is not None:
                    os.environ["DEEP_WORKER_ORIGINAL_CWD"] = original_env
                else:
                    os.environ.pop("DEEP_WORKER_ORIGINAL_CWD", None)
