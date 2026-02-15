"""Agent fixture for AI-powered integration tests.

Provides context manager for managing deep-worker agent lifecycle during tests.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.exceptions import AgentAlreadyExistsError
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry

from .models import AgentFixtureInfo, AgentStatus

logger = logging.getLogger(__name__)


def _is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is alive.

    Args:
        pid: Process ID to check

    Returns:
        True if process is alive, False otherwise
    """
    try:
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
        return True
    except (OSError, ProcessLookupError):
        return False


def _terminate_agent(agent_info: AgentFixtureInfo, force: bool = False) -> None:
    """Terminate an agent process.

    Args:
        agent_info: Agent information
        force: If True, use SIGKILL instead of SIGTERM
    """
    if agent_info.pid is None:
        logger.warning(f"Cannot terminate {agent_info.agent_name}: no PID")
        return

    if not _is_process_alive(agent_info.pid):
        logger.info(f"Agent {agent_info.agent_name} already dead (PID {agent_info.pid})")
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    signal_name = "SIGKILL" if force else "SIGTERM"

    logger.info(f"Sending {signal_name} to agent {agent_info.agent_name} (PID {agent_info.pid})")

    try:
        os.kill(agent_info.pid, sig)

        # Wait for process to die (up to 5 seconds)
        for _ in range(50):  # 50 * 0.1 = 5 seconds
            if not _is_process_alive(agent_info.pid):
                logger.info(f"Agent {agent_info.agent_name} terminated successfully")
                return
            time.sleep(0.1)

        # Process still alive after 5 seconds
        if not force:
            logger.warning(f"Agent {agent_info.agent_name} didn't respond to SIGTERM, sending SIGKILL")
            _terminate_agent(agent_info, force=True)
        else:
            logger.error(f"Failed to kill agent {agent_info.agent_name} even with SIGKILL")

    except (OSError, ProcessLookupError) as e:
        logger.error(f"Error terminating agent {agent_info.agent_name}: {e}")


def _cleanup_agent_files(agent_info: AgentFixtureInfo) -> None:
    """Clean up agent socket and PID files.

    Args:
        agent_info: Agent information
    """
    # Remove socket file
    if agent_info.socket_path.exists():
        try:
            agent_info.socket_path.unlink()
            logger.debug(f"Removed socket: {agent_info.socket_path}")
        except OSError as e:
            logger.warning(f"Failed to remove socket {agent_info.socket_path}: {e}")

    # Remove PID file if we know where it is
    pid_file = Path(f"/tmp/deep_worker_{agent_info.agent_name}.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
            logger.debug(f"Removed PID file: {pid_file}")
        except OSError as e:
            logger.warning(f"Failed to remove PID file {pid_file}: {e}")


def _get_existing_agent(agent_name: str, registry: ProcessRegistry) -> Optional[AgentFixtureInfo]:
    """Check if agent already exists in registry.

    Args:
        agent_name: Name of agent to check
        registry: ProcessRegistry instance

    Returns:
        AgentFixtureInfo if agent exists and is alive, None otherwise
    """
    agent_info = registry.lookup(agent_name)

    if not agent_info:
        return None

    # Agent exists and is alive (lookup already verified this)
    return AgentFixtureInfo(
        agent_name=agent_name,
        working_dir=Path(agent_info.working_dir),
        socket_path=agent_info.socket_path,
        pid=agent_info.pid,
        status=AgentStatus.RUNNING,
        created_by_fixture=False,  # Pre-existing agent
    )


async def _start_agent_process(
    agent_name: str,
    working_dir: Path,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    skip_permissions: bool = True,
) -> AgentFixtureInfo:
    """Start a new agent process.

    Args:
        agent_name: Name for the agent
        working_dir: Working directory
        model: AI model to use
        provider: AI provider
        skip_permissions: Whether to skip permission checks

    Returns:
        AgentFixtureInfo for the new agent

    Raises:
        AgentAlreadyExistsError: If agent already exists
        RuntimeError: If agent fails to start
    """
    logger.info(f"Starting agent {agent_name} in {working_dir}")

    # Create config
    config = AgentConfig(
        model=model,
        working_dir=working_dir,
        provider=provider,
    )

    # Create socket path
    socket_path = Path(f"/tmp/deep_worker_{agent_name}.sock")

    # Ensure socket doesn't exist
    if socket_path.exists():
        logger.warning(f"Removing existing socket: {socket_path}")
        socket_path.unlink()

    # Create agent process
    agent = AgentProcess(
        name=agent_name,
        config=config,
        socket_path=socket_path,
        skip_permissions=skip_permissions,
    )

    # Start agent in background task
    task = asyncio.create_task(agent.start())

    # Wait for agent to start (socket file appears)
    start_time = time.time()
    timeout = 60.0  # Increased from 10 to 60 seconds for better reliability
    agent_pid: Optional[int] = None
    socket_appeared = False
    connection_attempts = 0

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        if socket_path.exists():
            if not socket_appeared:
                socket_appeared = True
                logger.info(f"Agent {agent_name}: Socket appeared after {elapsed:.1f}s")

            # Give it a moment to be ready
            await asyncio.sleep(0.2)

            # Verify we can connect and get PID from registry
            try:
                connection_attempts += 1
                logger.debug(f"Agent {agent_name}: Connection attempt {connection_attempts} at {elapsed:.1f}s")

                client = AgentClient(agent_name)
                await client.connect()
                await client.disconnect()

                # Get PID from registry
                registry = ProcessRegistry()
                agent_info = registry.lookup(agent_name)
                if agent_info:
                    agent_pid = agent_info.pid
                    logger.info(f"Agent {agent_name} started successfully (PID {agent_pid}) after {elapsed:.1f}s")

                    return AgentFixtureInfo(
                        agent_name=agent_name,
                        working_dir=working_dir,
                        socket_path=socket_path,
                        pid=agent_pid,
                        status=AgentStatus.RUNNING,
                        created_by_fixture=True,
                    )

            except Exception as e:
                logger.debug(f"Agent {agent_name}: Connection failed at {elapsed:.1f}s, attempt {connection_attempts}: {type(e).__name__}: {e}")
        else:
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:  # Log every 5 seconds
                logger.info(f"Agent {agent_name}: Waiting for socket (elapsed: {elapsed:.1f}s)")

        await asyncio.sleep(0.1)

    # Timeout waiting for agent to start
    task.cancel()
    elapsed = time.time() - start_time
    logger.error(f"Agent {agent_name} initialization timeline:")
    logger.error(f"  - Total elapsed: {elapsed:.1f}s")
    logger.error(f"  - Socket appeared: {socket_appeared} (after socket creation)")
    logger.error(f"  - Connection attempts: {connection_attempts}")
    logger.error(f"  - Socket exists at timeout: {socket_path.exists()}")

    raise RuntimeError(
        f"Agent {agent_name} failed to start within {timeout}s. "
        f"Socket: {socket_path}, exists: {socket_path.exists()}, "
        f"connection_attempts: {connection_attempts}"
    )


@contextmanager
def agent_fixture(
    agent_name: str,
    working_dir: Optional[Path] = None,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    skip_permissions: bool = True,
) -> Generator[AgentFixtureInfo, None, None]:
    """Context manager for test agent lifecycle management.

    This fixture implements the critical testing principles:

    1. Session Isolation: Enforces 'test_' prefix
    2. Session Reuse: Reuses pre-existing agents
    3. Smart Cleanup:
       - Test passes + fixture created agent → terminate
       - Test fails + fixture created agent → preserve for debugging
       - Pre-existing agent → never terminate

    Args:
        agent_name: Name of agent (must start with 'test_')
        working_dir: Working directory (defaults to /tmp)
        model: AI model to use (default: deepseek-chat)
        provider: AI provider (default: deepseek)
        skip_permissions: Skip permission prompts (default: True for tests)

    Yields:
        AgentFixtureInfo with agent details

    Raises:
        ValueError: If agent_name doesn't start with 'test_'
        RuntimeError: If agent fails to start

    Example:
        >>> with agent_fixture('test_my_agent') as agent:
        ...     client = AgentClient(agent.agent_name)
        ...     # ... test code ...
        # Agent cleaned up automatically if test passed
    """
    # Validate agent name
    if not agent_name.startswith("test_"):
        raise ValueError(
            f"Agent name must start with 'test_' for safety. Got: {agent_name}. "
            "This prevents accidentally affecting production agents."
        )

    if working_dir is None:
        working_dir = Path("/tmp")

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    registry = ProcessRegistry()
    agent_info: Optional[AgentFixtureInfo] = None
    test_passed = False

    try:
        # Check if agent already exists
        existing = _get_existing_agent(agent_name, registry)

        if existing:
            logger.info(
                f"Reusing existing agent {agent_name} (PID {existing.pid}). "
                "Will NOT terminate after test."
            )
            agent_info = existing

        else:
            # Start new agent
            logger.info(f"Creating new test agent {agent_name}")

            # Need to run async code
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            agent_info = loop.run_until_complete(
                _start_agent_process(
                    agent_name=agent_name,
                    working_dir=working_dir,
                    model=model,
                    provider=provider,
                    skip_permissions=skip_permissions,
                )
            )

        # Yield to test
        yield agent_info

        # Test completed successfully
        test_passed = True

    except Exception as e:
        # Test failed
        logger.error(f"Test failed with exception: {e}")
        test_passed = False
        raise

    finally:
        # Cleanup logic
        if agent_info:
            if agent_info.created_by_fixture:
                if test_passed:
                    logger.info(
                        f"Test passed. Terminating agent {agent_name} (created by fixture)."
                    )
                    _terminate_agent(agent_info)
                    _cleanup_agent_files(agent_info)
                    registry.unregister(agent_name)
                else:
                    logger.warning(
                        f"Test FAILED. Preserving agent {agent_name} for debugging. "
                        f"Connect with: dwcli {agent_name}"
                    )
            else:
                logger.info(
                    f"Agent {agent_name} was pre-existing. Not terminating."
                )


def verify_agent_running(agent_name: str, timeout: float = 5.0) -> bool:
    """Verify an agent is running and responsive.

    Args:
        agent_name: Name of agent to check
        timeout: Maximum time to wait (seconds)

    Returns:
        True if agent is running and responsive, False otherwise

    Example:
        >>> with agent_fixture('test_agent') as agent:
        ...     assert verify_agent_running(agent.agent_name)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _check() -> bool:
        try:
            client = AgentClient(agent_name)
            await asyncio.wait_for(client.connect(), timeout=timeout)
            status = await client.get_status()
            await client.disconnect()
            # SessionInfo is returned, check if status indicates active agent
            return status.status in ("active", "idle", "working")
        except Exception as e:
            logger.debug(f"Agent {agent_name} not responsive: {e}")
            return False

    return loop.run_until_complete(_check())
