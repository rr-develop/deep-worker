"""Agent fixture for AI-powered integration tests.

Provides context manager for managing deep-worker agent lifecycle during tests.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
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
    """Clean up agent PID files.

    Args:
        agent_info: Agent information
    """
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
        rpyc_port=agent_info.rpyc_port,
        pid=agent_info.pid,
        status=AgentStatus.RUNNING,
        created_by_fixture=False,  # Pre-existing agent
    )


def _agent_process_entry_point(
    agent_name: str,
    working_dir: Path,
    model: str,
    provider: str,
    skip_permissions: bool,
    api_key: Optional[str],
) -> None:
    """Entry point for agent subprocess - runs agent.start() in its own event loop."""
    import asyncio
    import os
    import sys

    # ============================================================================
    # CRITICAL: Ensure local deepagents code is used instead of system package
    # ============================================================================
    # Tests should ALWAYS use the code from this repository
    repo_base = Path(__file__).resolve().parents[3]
    cli_package_path = str(repo_base / "deep_worker_cli")
    deepagents_path = str(repo_base / "deepagents")

    # Insert at beginning to override system packages
    if cli_package_path not in sys.path:
        sys.path.insert(0, cli_package_path)
    if deepagents_path not in sys.path:
        sys.path.insert(0, deepagents_path)

    # Set API key in environment if provided
    if api_key:
        if provider == "deepseek":
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
            os.environ["OPENAI_MODEL"] = model
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_MODEL"] = model
        elif provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GOOGLE_MODEL"] = model

    # Create config
    config = AgentConfig(
        model=model,
        working_dir=working_dir,
        provider=provider,
        api_key=api_key,
    )

    # Create agent process
    agent = AgentProcess(
        name=agent_name,
        config=config,
        skip_permissions=skip_permissions,
        rpyc_port=True,  # Auto-assign port
    )

    # Run agent.start() in a new event loop (this is the correct way!)
    asyncio.run(agent.start())


def _start_agent_process(
    agent_name: str,
    working_dir: Path,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    skip_permissions: bool = True,
    api_key: Optional[str] = None,
) -> AgentFixtureInfo:
    """Start a new agent process using multiprocessing.

    CRITICAL REQUIREMENT:
    ====================
    This function MUST return a FULLY INITIALIZED agent that is ready to accept
    RPyC connections. The returned AgentFixtureInfo MUST represent an agent that:
    - Has RPyC server running and accepting connections
    - Is registered in ProcessRegistry with correct port
    - Can successfully handle client.connect() calls immediately

    If the agent cannot be fully initialized, this function MUST raise RuntimeError
    with a clear explanation. Returning a partially initialized agent is FORBIDDEN.

    Args:
        agent_name: Name for the agent (must be unique)
        working_dir: Working directory
        model: AI model to use
        provider: AI provider
        skip_permissions: Whether to skip permission checks
        api_key: API key for the provider

    Returns:
        AgentFixtureInfo for the FULLY INITIALIZED agent (ready to accept connections)

    Raises:
        AgentAlreadyExistsError: If agent already exists
        RuntimeError: If agent fails to start or cannot be fully initialized
    """
    logger.info(f"Starting agent {agent_name} in {working_dir}")

    # Start agent in a separate process using multiprocessing
    # This allows agent.start() to run asyncio.run() in its own event loop
    process = multiprocessing.Process(
        target=_agent_process_entry_point,
        args=(agent_name, working_dir, model, provider, skip_permissions, api_key),
        name=f"agent-{agent_name}",
        daemon=True,
    )
    process.start()

    # Wait for agent to start (RPyC becomes available)
    start_time = time.time()
    timeout = 60.0  # Increased from 10 to 60 seconds for better reliability
    agent_pid: Optional[int] = None
    agent_rpyc_port: Optional[int] = None
    registry_appeared = False
    connection_attempts = 0

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        # Check registry for agent registration
        registry = ProcessRegistry()
        agent_info_reg = registry.lookup(agent_name)

        if agent_info_reg:
            if not registry_appeared:
                registry_appeared = True
                logger.info(f"Agent {agent_name}: Registered in registry after {elapsed:.1f}s")

            # Give it a moment to be ready
            time.sleep(0.2)

            # Verify we can connect via RPyC
            try:
                connection_attempts += 1
                logger.info(f"Agent {agent_name}: Connection attempt {connection_attempts} at {elapsed:.1f}s, RPyC port={agent_info_reg.rpyc_port}")
                print(f"[FIXTURE] Connection attempt {connection_attempts} to {agent_name} on port {agent_info_reg.rpyc_port}", flush=True)

                # Create a temporary event loop for the async client operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    client = AgentClient(agent_name)
                    logger.info(f"Agent {agent_name}: Created AgentClient, calling connect()...")
                    print(f"[FIXTURE] Created AgentClient, calling connect()...", flush=True)

                    # Add timeout to prevent hanging forever
                    try:
                        loop.run_until_complete(asyncio.wait_for(client.connect(), timeout=5.0))
                    except asyncio.TimeoutError:
                        print(f"[FIXTURE] Connection timeout after 5s!", flush=True)
                        raise

                    logger.info(f"Agent {agent_name}: Connected successfully, calling disconnect()...")
                    print(f"[FIXTURE] Connected successfully!", flush=True)
                    loop.run_until_complete(client.disconnect())
                    logger.info(f"Agent {agent_name}: Disconnected successfully")
                    print(f"[FIXTURE] Disconnected successfully", flush=True)
                finally:
                    loop.close()

                # Success - agent is ready
                print(f"[FIXTURE] Getting PID and port from registry info...", flush=True)
                agent_pid = agent_info_reg.pid
                agent_rpyc_port = agent_info_reg.rpyc_port
                print(f"[FIXTURE] Got PID={agent_pid}, port={agent_rpyc_port}", flush=True)
                logger.info(f"Agent {agent_name} started successfully (PID {agent_pid}, RPyC port {agent_rpyc_port}) after {elapsed:.1f}s")
                print(f"[FIXTURE] About to return AgentFixtureInfo...", flush=True)

                return AgentFixtureInfo(
                    agent_name=agent_name,
                    working_dir=working_dir,
                    rpyc_port=agent_rpyc_port,
                    pid=agent_pid,
                    status=AgentStatus.RUNNING,
                    created_by_fixture=True,
                )

            except Exception as e:
                logger.debug(f"Agent {agent_name}: Connection failed at {elapsed:.1f}s, attempt {connection_attempts}: {type(e).__name__}: {e}")
        else:
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:  # Log every 5 seconds
                logger.info(f"Agent {agent_name}: Waiting for registry (elapsed: {elapsed:.1f}s)")

        time.sleep(0.1)

    # Timeout waiting for agent to start
    # Note: agent is running in a background thread, so no task to cancel
    elapsed = time.time() - start_time
    logger.error(f"Agent {agent_name} initialization timeline:")
    logger.error(f"  - Total elapsed: {elapsed:.1f}s")
    logger.error(f"  - Registry appeared: {registry_appeared}")
    logger.error(f"  - Connection attempts: {connection_attempts}")
    logger.error(f"  - RPyC port: {agent_rpyc_port}")

    raise RuntimeError(
        f"Agent {agent_name} failed to start within {timeout}s. "
        f"Registry appeared: {registry_appeared}, "
        f"connection_attempts: {connection_attempts}"
    )


@contextmanager
def agent_fixture(
    agent_name: str,
    working_dir: Optional[Path] = None,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    skip_permissions: bool = True,
    api_key: Optional[str] = None,
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

            # Call sync version (_start_agent_process is now sync and uses multiprocessing)
            agent_info = _start_agent_process(
                agent_name=agent_name,
                working_dir=working_dir,
                model=model,
                provider=provider,
                skip_permissions=skip_permissions,
                api_key=api_key,
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
