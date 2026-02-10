"""Process management for deep_worker agents.

This module provides process lifecycle management and registry for deep-worker agents.

Key Components:
    - AgentProcess: Main agent process with lifecycle management, IPC server,
      and signal handling
    - ProcessRegistry: JSON-based registry for agent discovery and tracking
      with file locking and process liveness verification

Architecture:
    Each agent runs as a separate OS process with:
    - Unique name and PID file
    - Unix domain socket for IPC
    - Registration in shared ProcessRegistry
    - SessionManager for state tracking

Example:
    >>> from deep_worker_cli.process import AgentProcess
    >>> from deep_worker_cli.agent_config_pkg import AgentConfig
    >>>
    >>> config = AgentConfig(model="claude-sonnet-4-5-20250929", working_dir=Path.cwd())
    >>> agent = AgentProcess(name="my-agent", config=config)
    >>> await agent.start()
"""

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry

__all__ = ["AgentProcess", "ProcessRegistry"]
