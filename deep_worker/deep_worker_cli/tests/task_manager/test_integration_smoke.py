"""Smoke test for TaskManager integration with AgentProcess.

This test verifies that TaskManager can be initialized and started
within AgentProcess without errors.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess


@pytest.mark.asyncio
async def test_agent_process_with_task_manager_initializes():
    """Test that AgentProcess can initialize with TaskManager enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create agent config
        config = AgentConfig(
            model="deepseek-chat",
            working_dir=tmpdir_path
        )

        # Create agent process with TaskManager enabled
        agent_process = AgentProcess(
            name="test_tm_agent",
            config=config,
            socket_path=tmpdir_path / "test_tm.sock",
            pid_file=tmpdir_path / "test_tm.pid",
            data_dir=tmpdir_path / "data",
            skip_permissions=True,
            enable_task_manager=True
        )

        # Verify enable_task_manager is set
        assert agent_process.enable_task_manager is True
        assert agent_process.task_manager is None  # Not initialized yet

        # Note: We don't start the agent in this smoke test because:
        # 1. It would require a full LangGraph agent initialization
        # 2. It would require environment variables (API keys, etc.)
        # 3. It would start a long-running process
        # This test just verifies the plumbing is in place

        print("[SMOKE TEST] AgentProcess with enable_task_manager=True created successfully")


@pytest.mark.asyncio
async def test_agent_process_without_task_manager():
    """Test that AgentProcess works without TaskManager (default behavior)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create agent config
        config = AgentConfig(
            model="deepseek-chat",
            working_dir=tmpdir_path
        )

        # Create agent process WITHOUT TaskManager (default)
        agent_process = AgentProcess(
            name="test_no_tm_agent",
            config=config,
            socket_path=tmpdir_path / "test_no_tm.sock",
            pid_file=tmpdir_path / "test_no_tm.pid",
            data_dir=tmpdir_path / "data",
            skip_permissions=True
            # enable_task_manager not specified = False (default)
        )

        # Verify TaskManager is disabled
        assert agent_process.enable_task_manager is False
        assert agent_process.task_manager is None

        print("[SMOKE TEST] AgentProcess without TaskManager created successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
