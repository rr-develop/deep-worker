"""Pytest fixture for launching deep-worker agents with clean state.

This fixture provides a way to ensure agent has clean history before test execution.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def clean_agent():
    """Fixture to ensure a clean agent with no history.

    Usage in tests:
        @pytest.mark.parametrize("clean_agent", ["test_my_agent"], indirect=True)
        def test_something(clean_agent):
            agent_name = clean_agent
            # Agent is guaranteed to have clean history
            ...

    The fixture:
    1. Kills existing agent process if running
    2. Cleans up checkpoint/data directories
    3. Starts fresh agent process
    4. Waits for agent registration
    5. Yields agent name
    6. Cleans up after test
    """
    agent_name = None
    agent_process = None

    def _setup_clean_agent(name: str, working_dir: str = "/tmp") -> subprocess.Popen:
        """Set up a clean agent with no history."""
        nonlocal agent_name, agent_process
        agent_name = name

        logger.info(f"Setting up clean agent: {name}")

        # 1. Kill existing agent process if running
        from deep_worker_cli.process.registry import ProcessRegistry
        registry = ProcessRegistry()
        existing = registry.lookup(name)
        if existing:
            logger.info(f"Killing existing agent process: PID {existing.pid}")
            try:
                import os
                import signal
                os.kill(existing.pid, signal.SIGKILL)
                time.sleep(0.5)
            except ProcessLookupError:
                pass

        # 2. Clean up data directories
        data_dir = Path(f"~/.deep_worker/{name}").expanduser()
        if data_dir.exists():
            logger.info(f"Removing data directory: {data_dir}")
            shutil.rmtree(data_dir)

        log_dir = Path("~/.deep_worker/logs").expanduser()
        for log_file in log_dir.glob(f"{name}.*"):
            logger.info(f"Removing log file: {log_file}")
            log_file.unlink()

        # 3. Start fresh agent process
        logger.info(f"Starting fresh agent: {name}")
        agent_process = subprocess.Popen(
            ["deep-worker", name, "--dir", working_dir, "--skip-permissions"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 4. Wait for agent registration
        logger.info("Waiting for agent to register...")
        start_wait = time.time()
        timeout = 30.0
        agent_registered = False

        while time.time() - start_wait < timeout:
            agent_info = registry.lookup(name)
            if agent_info:
                agent_registered = True
                elapsed = time.time() - start_wait
                logger.info(f"✓ Agent registered after {elapsed:.1f}s (PID: {agent_info.pid})")
                break
            time.sleep(0.2)

        if not agent_registered:
            raise RuntimeError(f"Agent '{name}' failed to register within {timeout}s")

        return agent_process

    # Yield factory function that can be called from test
    yield _setup_clean_agent

    # Cleanup after test
    if agent_process:
        logger.info(f"Cleaning up agent: {agent_name}")
        try:
            agent_process.terminate()
            agent_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Agent did not terminate, killing...")
            agent_process.kill()
            agent_process.wait()


@pytest.fixture(params=["clean", "preserve_history"])
def agent_state(request) -> Literal["clean", "preserve_history"]:
    """Parametrized fixture to run test with both clean and preserved history.

    Usage:
        def test_something(agent_state):
            if agent_state == "clean":
                # Set up clean agent
                ...
            else:
                # Use existing agent with history
                ...
    """
    return request.param
