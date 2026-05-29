"""Agent fixture for AI-powered integration tests.

Provides context manager for managing deep-worker agent lifecycle during tests.

CRITICAL: skip_permissions=True by default
==========================================
This fixture ALWAYS uses skip_permissions=True by default to prevent tests from hanging
on approval prompts. Only set skip_permissions=False for tests that specifically
test the approval workflow (HITL - human-in-the-loop).

⚠️  DO NOT CHANGE THE DEFAULT! This would break 95% of integration tests.
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
from typing import Any, Generator, Optional

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
    """Remove on-disk artefacts a passed-test agent left behind.

    Two locations:

    1. ``/tmp/deep_worker_<agent_name>.pid`` — the PID lockfile.
    2. ``<agent-data-dir>/<agent_name>/`` — the ``SessionManager.data_dir``
       default (see ``AgentProcess.__init__``:
       ``self.data_dir = data_dir or (Path("~/.deep_worker").expanduser() / name)``).
       Holds ``session_metadata.json`` and conversation summaries.
       Used to be left on disk by every passed test, accumulating
       hundreds of stale directories per machine over time.

    Safety: ``AgentFixtureInfo.agent_name`` is pydantic-validated
    against ``^test_[a-z0-9_]+$`` at construction (see ``models.py``),
    so by the time this helper runs the name is guaranteed to be a
    test name. There is no path through which a production agent
    name could reach this code.

    Args:
        agent_info: Agent information (pydantic-validated test name).
    """
    import shutil  # local import — only used on teardown

    # 1. Remove PID file if we know where it is
    pid_file = Path(f"/tmp/deep_worker_{agent_info.agent_name}.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
            logger.debug(f"Removed PID file: {pid_file}")
        except OSError as e:
            logger.warning(f"Failed to remove PID file {pid_file}: {e}")

    # 2. Remove the agent's persistent data directory. The agent
    #    process created this in AgentProcess.__init__ for its
    #    SessionManager. Best-effort: a missing dir is fine (agent
    #    may not have reached the SessionManager init phase).
    data_dir = Path.home() / Path("~/.deep_worker").name / agent_info.agent_name
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir, ignore_errors=False)
            logger.debug(f"Removed agent data dir: {data_dir}")
        except OSError as e:
            logger.warning(f"Failed to remove data dir {data_dir}: {e}")


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
    model: Optional[str],
    provider: str,
    skip_permissions: bool,
    api_key: Optional[str],
    enable_task_manager: bool = False,
) -> None:
    """Entry point for agent subprocess - runs agent.start() in its own event loop."""
    import asyncio
    import os
    import sys

    # ============================================================================
    # CRITICAL: Ensure local deepagents code is used instead of system package
    # ============================================================================
    # This host should ALWAYS use code from the LOCAL repository (not
    # system-installed packages). We detect the repository root from the
    # agent_fixture.py path: going up 4 dirs from
    #   .../<tree>/deep_worker/deep_worker_cli/tests/integration_ai/
    # gives us the tree root (the production tree or the developer tree).
    # ============================================================================
    _fixture_path = Path(__file__).resolve()
    repo_base = _fixture_path.parents[3] / "deep_worker"
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
            if model:
                os.environ["OPENAI_MODEL"] = model
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            if model:
                os.environ["OPENAI_MODEL"] = model
        elif provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = api_key
            if model:
                os.environ["GOOGLE_MODEL"] = model

    # claude-code doesn't require API key (uses local Claude Code installation)
    # No environment setup needed

    # Create config
    config = AgentConfig(
        model=model,
        working_dir=working_dir,
        provider=provider,
        api_key=api_key,
    )

    # Create agent process
    # ⚠️ CRITICAL: ALWAYS use rpyc_port=True for auto port assignment! ⚠️
    # ============================================================================
    # NEVER use fixed ports like rpyc_port=18861 in tests!
    # Fixed ports cause race conditions when tests run concurrently:
    # - Port conflicts between parallel tests
    # - Port not released from previous test
    # - Flaky tests that fail randomly
    #
    # ALWAYS use rpyc_port=True which delegates to AgentProcess to find
    # a free port dynamically using OS port selection (port=0).
    # ============================================================================
    agent = AgentProcess(
        name=agent_name,
        config=config,
        skip_permissions=skip_permissions,
        enable_task_manager=enable_task_manager,
        rpyc_port=True,  # ⚠️ MUST be True! Never use fixed port numbers!
    )

    # Run agent.start() in a new event loop (this is the correct way!)
    asyncio.run(agent.start())


def _start_agent_process(
    agent_name: str,
    working_dir: Path,
    model: Optional[str] = None,
    provider: str = "deepseek",
    skip_permissions: bool = True,  # ⚠️ CRITICAL: Must be True by default! See docstring.
    api_key: Optional[str] = None,
    enable_task_manager: bool = False,
) -> AgentFixtureInfo:
    """Start a new agent process using multiprocessing.

    ⚠️ CRITICAL: Uses 'spawn' start method to avoid fork() deadlocks ⚠️
    ==================================================================
    Python's multiprocessing fork() method can cause deadlocks when:
    - Parent process is multi-threaded (e.g., pytest with plugins)
    - Child process uses asyncio (event loops don't survive fork)
    - Libraries like lancedb are not fork-safe

    We MUST use 'spawn' start method which starts a fresh Python interpreter.
    ==================================================================

    ⚠️ CRITICAL PORT USAGE REQUIREMENT ⚠️
    =====================================
    NEVER PASS FIXED PORTS TO AgentProcess!

    The agent creation inside _agent_process_entry_point MUST use:
        rpyc_port=True  # ✅ Correct: Auto-assign port

    NEVER use:
        rpyc_port=18861  # ❌ WRONG: Fixed port causes race conditions!

    Fixed ports cause flaky tests when:
    - Multiple tests run concurrently
    - Previous test didn't release port
    - System service uses the port

    This function creates agents with dynamic port assignment, ensuring
    each agent gets a unique port from the OS (using port=0).
    =====================================

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
        skip_permissions: Whether to skip permission checks (default: True)
                         ⚠️  CRITICAL: This MUST be True for all tests that invoke tools!
                         Integration tests invoke tools (read, write, execute, etc.) and would
                         hang forever waiting for approval prompts that can't be provided in CI/CD.
                         Only set to False for tests that specifically test the approval workflow.
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
    # CRITICAL: Use 'spawn' start method to avoid fork() deadlocks
    # This prevents issues with:
    # - Multi-threaded parent process (pytest)
    # - asyncio event loops not surviving fork()
    # - Libraries that are not fork-safe (lancedb, etc.)
    ctx = multiprocessing.get_context('spawn')
    process = ctx.Process(
        target=_agent_process_entry_point,
        args=(agent_name, working_dir, model, provider, skip_permissions, api_key, enable_task_manager),
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

            # Give AsyncioServer a moment to be ready for connections
            # CRITICAL: We do NOT test connection here because AsyncioServer
            # in multiprocessing has a bug where second sequential connection hangs!
            # The fixture connection would consume the "first working connection",
            # leaving the actual test with the broken second connection.
            # Instead, we just verify the server is registered and give it time to initialize.
            time.sleep(0.5)

            # Success - agent is ready (registered and server started)
            agent_pid = agent_info_reg.pid
            agent_rpyc_port = agent_info_reg.rpyc_port
            logger.info(f"Agent {agent_name} started successfully (PID {agent_pid}, RPyC port {agent_rpyc_port}) after {elapsed:.1f}s")
            print(f"[FIXTURE] Agent ready: PID={agent_pid}, port={agent_rpyc_port}", flush=True)

            return AgentFixtureInfo(
                    agent_name=agent_name,
                    working_dir=working_dir,
                    rpyc_port=agent_rpyc_port,
                    pid=agent_pid,
                    status=AgentStatus.RUNNING,
                    created_by_fixture=True,
                )
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
    model: Optional[str] = None,
    provider: str = "deepseek",
    skip_permissions: bool = True,
    api_key: Optional[str] = None,
    enable_task_manager: bool = False,
    pytest_item: Any = None,
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
        model: AI model to use (default: None → system default per provider)
        provider: AI provider (default: deepseek)
        skip_permissions: Skip permission prompts (default: True for tests)
                         ⚠️  CRITICAL: This MUST be True for all tests that invoke tools!
                         Only set to False for tests that specifically test the approval workflow.

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
                enable_task_manager=enable_task_manager,
            )

        # Hand the freshly-spawned (or reused) agent off to the
        # session-finish sweeper so that — even if pytest itself is
        # killed before our ``finally`` runs — leaked passed-test
        # agents still get cleaned up. The sweeper checks
        # ``created_by_fixture`` and the test outcome before doing
        # anything, so it is safe to register pre-existing agents
        # too.
        if pytest_item is not None:
            try:
                from .conftest import register_test_agent  # noqa: PLC0415
                register_test_agent(
                    agent_name,
                    pytest_item,
                    created_by_fixture=agent_info.created_by_fixture,
                )
            except Exception as exc:  # noqa: BLE001
                # Don't let bookkeeping break the test.
                logger.debug(f"register_test_agent failed: {exc}")

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
