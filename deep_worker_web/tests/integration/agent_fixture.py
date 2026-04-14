"""Integration test fixture for creating real deep-worker agents.

Based on deep_worker_cli.tests.integration_ai.agent_fixture.
"""

import asyncio
import multiprocessing
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from deep_worker_cli.process.registry import ProcessRegistry  # type: ignore[import-not-found]


@dataclass
class AgentFixtureInfo:
    """Information about a test agent."""

    name: str
    pid: int
    rpyc_port: int
    data_dir: str
    process: multiprocessing.Process


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

    # Ensure local deep_worker code is used
    repo_base = Path(__file__).resolve().parents[3] / "deep_worker"
    cli_package_path = str(repo_base / "deep_worker_cli")
    deepagents_path = str(repo_base / "deepagents")

    # Insert at beginning to override system packages
    if cli_package_path not in sys.path:
        sys.path.insert(0, cli_package_path)
    if deepagents_path not in sys.path:
        sys.path.insert(0, deepagents_path)

    # Import after path is set up
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig  # type: ignore[import-not-found]
    from deep_worker_cli.process.agent_process import AgentProcess  # type: ignore[import-not-found]

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

    # Create agent process with auto-assigned port
    agent = AgentProcess(
        name=agent_name,
        config=config,
        skip_permissions=skip_permissions,
        rpyc_port=True,  # Auto-assign port
    )

    # Run agent.start() in a new event loop
    asyncio.run(agent.start())


@contextmanager
def agent_fixture(
    agent_name: str | None = None,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    skip_permissions: bool = True,
    api_key: Optional[str] = None,
    cleanup_on_success: bool = True,
    timeout: float = 60.0,
) -> Iterator[AgentFixtureInfo]:
    """Create a real deep-worker agent for integration testing.

    Args:
        agent_name: Name for the agent. If None, generates unique name.
        model: AI model to use (default: deepseek-chat).
        provider: AI provider (default: deepseek).
        skip_permissions: If True, skips permission checks (needed for tests).
        api_key: API key for the provider (optional).
        cleanup_on_success: If True, cleans up agent on successful test.
                            If False or test fails, preserves for debugging.
        timeout: Maximum time to wait for agent startup (seconds).

    Yields:
        AgentFixtureInfo with agent details.

    Raises:
        TimeoutError: If agent doesn't start within timeout period.
        RuntimeError: If agent startup fails.

    Example:
        >>> with agent_fixture() as agent_info:
        ...     # Use agent_info.name, agent_info.rpyc_port, etc.
        ...     response = await send_message_to_agent(agent_info.name, "Hello")
    """
    # Generate unique agent name if not provided
    if agent_name is None:
        agent_name = f"test_agent_{os.getpid()}_{int(time.time())}"

    # Create temporary data directory
    temp_dir = tempfile.mkdtemp(prefix=f"deep_worker_test_{agent_name}_")
    working_dir = Path(temp_dir)

    # Use 'spawn' start method to avoid fork issues with asyncio
    ctx = multiprocessing.get_context('spawn')

    # Create and start agent process
    process = ctx.Process(
        target=_agent_process_entry_point,
        args=(agent_name, working_dir, model, provider, skip_permissions, api_key),
        name=f"agent-{agent_name}",
        daemon=True,
    )

    process.start()

    # Wait for agent to register in ProcessRegistry
    registry = ProcessRegistry()
    start_time = time.time()
    agent_pid = None
    agent_port = None

    while time.time() - start_time < timeout:
        agent_info_reg = registry.lookup(agent_name)

        if agent_info_reg:
            agent_pid = agent_info_reg.pid
            agent_port = agent_info_reg.rpyc_port

            # Give AsyncioServer a moment to be ready
            time.sleep(0.5)
            break

        time.sleep(0.1)

    if agent_pid is None or agent_port is None:
        # Startup failed
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

        # Cleanup temp dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        raise TimeoutError(
            f"Agent {agent_name} failed to start within {timeout} seconds"
        )

    # Create fixture info
    fixture_info = AgentFixtureInfo(
        name=agent_name,
        pid=agent_pid,
        rpyc_port=agent_port,
        data_dir=temp_dir,
        process=process,
    )

    test_failed = False

    try:
        yield fixture_info
    except Exception:
        # Test failed - preserve agent for debugging
        test_failed = True
        raise
    finally:
        # Cleanup
        should_cleanup = cleanup_on_success and not test_failed

        # Stop the agent process
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

            if process.is_alive():
                process.kill()
                process.join(timeout=2)

        # Clean up temp directory if requested
        if should_cleanup:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nPreserving failed test agent data: {temp_dir}")


@contextmanager
def multiple_agents(
    count: int = 2,
    **kwargs: Any,
) -> Iterator[list[AgentFixtureInfo]]:
    """Create multiple test agents.

    Args:
        count: Number of agents to create.
        **kwargs: Arguments passed to agent_fixture().

    Yields:
        List of AgentFixtureInfo for all created agents.

    Example:
        >>> with multiple_agents(count=3) as agents:
        ...     for agent in agents:
        ...         print(f"Agent {agent.name} on port {agent.rpyc_port}")
    """
    agents: list[AgentFixtureInfo] = []
    fixtures: list[Any] = []

    try:
        # Create all agents
        for i in range(count):
            fixture = agent_fixture(
                agent_name=f"test_agent_{i}_{os.getpid()}",
                **kwargs,
            )
            fixtures.append(fixture)
            agent_info = fixture.__enter__()
            agents.append(agent_info)

        yield agents

    finally:
        # Clean up all agents in reverse order
        for fixture in reversed(fixtures):
            try:
                fixture.__exit__(None, None, None)
            except Exception as e:
                print(f"Error cleaning up agent: {e}")
