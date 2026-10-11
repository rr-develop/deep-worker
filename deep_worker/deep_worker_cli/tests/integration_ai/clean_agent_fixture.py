"""Pytest fixture for launching deep-worker agents with clean state.

This fixture provides a way to ensure agent has clean history before test execution.

CRITICAL: skip_permissions=True by default
==========================================
This fixture ALWAYS uses --skip-permissions by default to prevent tests from hanging
on approval prompts. Only set skip_permissions=False for tests that specifically
test the approval workflow (HITL - human-in-the-loop).

⚠️  DO NOT CHANGE THE DEFAULT! This would break 95% of integration tests.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import pytest
from deep_worker_cli.storage_paths import deep_worker_home

logger = logging.getLogger(__name__)


@pytest.fixture
def clean_agent(request):
    """Fixture to ensure a clean agent with no history.

    Usage in tests:
        @pytest.mark.parametrize("clean_agent", ["test_my_agent"], indirect=True)
        def test_something(clean_agent):
            agent_name = clean_agent
            # Agent is guaranteed to have clean history
            ...

    To reuse existing agent (for debugging):
        @pytest.mark.reuse_agent
        def test_something(clean_agent):
            agent_process = clean_agent("test_my_agent")
            # Will reuse existing agent if running, won't kill it

    The fixture:
    1. Kills existing agent process if running (unless reuse_agent marker present)
    2. Cleans up checkpoint/data directories
    3. Starts fresh agent process
    4. Waits for agent registration
    5. Yields agent name
    6. Cleans up after test
    """
    agent_name = None
    agent_process = None

    # Check for reuse_agent marker
    reuse = request.node.get_closest_marker("reuse_agent") is not None

    def _setup_clean_agent(
        name: str,
        working_dir: str = "/tmp",
        provider: str | None = None,
        skip_permissions: bool = True
    ) -> subprocess.Popen:
        """Set up a clean agent with no history.

        Args:
            name: Agent name
            working_dir: Working directory for agent
            provider: Optional provider (e.g., "claude-code-sdk", "anthropic")
            skip_permissions: Skip permission prompts (default: True for tests)
                             ⚠️  CRITICAL: This MUST be True for all tests that invoke tools!
                             Only set to False for tests that specifically test the approval workflow.
        """
        nonlocal agent_name, agent_process
        agent_name = name

        logger.info(f"Setting up clean agent: {name}" + (f" with provider: {provider}" if provider else ""))

        # 1. Kill existing agent process if running (unless reuse=True)
        from deep_worker_cli.process.registry import ProcessRegistry
        import os
        import signal

        registry = ProcessRegistry()
        existing = registry.lookup(name)

        if existing and reuse:
            logger.info(
                f"REUSE MODE: Keeping existing agent process PID {existing.pid}. "
                f"Will NOT kill or restart. Connect with: dwcli {name}"
            )
            # Return early - reuse existing process
            return existing.process if hasattr(existing, 'process') else None

        if existing:
            logger.info(f"Killing existing agent process: PID {existing.pid}")
            try:
                os.kill(existing.pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.info(f"Process {existing.pid} already dead")

            # CRITICAL: Wait for process to actually die and disappear from registry
            # This prevents testing old agent code when bug fixes were made
            kill_timeout = 5.0
            kill_start = time.time()
            process_dead = False

            while time.time() - kill_start < kill_timeout:
                # Check if still in registry
                if not registry.lookup(name):
                    process_dead = True
                    logger.info(f"✓ Process {existing.pid} confirmed dead and removed from registry")
                    break

                # Also check if process exists in OS
                try:
                    os.kill(existing.pid, 0)  # signal 0 just checks if process exists
                    logger.debug(f"Process {existing.pid} still exists, waiting...")
                except ProcessLookupError:
                    logger.info(f"✓ Process {existing.pid} no longer exists in OS")
                    # Give registry time to update
                    time.sleep(0.2)
                    if not registry.lookup(name):
                        process_dead = True
                        logger.info(f"✓ Confirmed removal from registry")
                        break

                time.sleep(0.1)

            if not process_dead:
                raise RuntimeError(
                    f"Old agent process {existing.pid} did not die within {kill_timeout}s. "
                    f"Cannot safely start new agent - would test old code!"
                )

        # 2. Clean up data directories
        data_dir = deep_worker_home() / name
        if data_dir.exists():
            logger.info(f"Removing data directory: {data_dir}")
            shutil.rmtree(data_dir)

        log_dir = deep_worker_home() / "logs"
        for log_file in log_dir.glob(f"{name}.*"):
            logger.info(f"Removing log file: {log_file}")
            log_file.unlink()

        # 3. Start fresh agent process
        logger.info(f"Starting fresh agent: {name}")

        # ============================================================================
        # CRITICAL: skip_permissions MUST be True by default for all integration tests!
        # ============================================================================
        # WHY: Integration tests invoke tools (read, write, execute, etc.) and would
        # hang forever waiting for approval prompts that can't be provided in CI/CD.
        #
        # ONLY set skip_permissions=False for tests that specifically test the
        # approval/interrupt workflow (HITL - human-in-the-loop).
        #
        # ⚠️  DO NOT CHANGE THIS DEFAULT WITHOUT EXPLICIT APPROVAL! ⚠️
        # Changing this will break 95% of integration tests.
        # ============================================================================
        cmd = ["deep-worker", name, "--dir", working_dir]
        if skip_permissions:
            cmd.append("--skip-permissions")
        if provider:
            cmd.extend(["--provider", provider])

        agent_process = subprocess.Popen(
            cmd,
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
