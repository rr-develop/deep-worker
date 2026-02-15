#!/usr/bin/env python3
"""
Agent launcher script for starting AgentProcess as subprocess.

This script is called by DwSession.create() to start a deep-worker agent
as an independent subprocess.
"""

import asyncio
import sys
from pathlib import Path


def main():
    """Main entry point for agent launcher."""
    if len(sys.argv) < 3:
        print("Usage: agent_launcher.py <session_name> <working_dir> [model] [skip_permissions] [enable_task_manager]", file=sys.stderr)
        sys.exit(1)

    session_name = sys.argv[1]
    working_dir = Path(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else "deepseek-chat"
    skip_permissions = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
    enable_task_manager = sys.argv[5].lower() == "true" if len(sys.argv) > 5 else False

    # Import deep-worker components
    from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
    from deep_worker_cli.process.agent_process import AgentProcess

    # Create agent config
    config = AgentConfig(
        model=model,
        working_dir=working_dir
    )

    # Create agent process
    agent_process = AgentProcess(
        name=session_name,
        config=config,
        data_dir=working_dir,  # Use working_dir as data_dir for session files
        skip_permissions=skip_permissions,
        enable_task_manager=enable_task_manager
    )

    # Start agent (blocking)
    try:
        asyncio.run(agent_process.start())
        print(f"Agent {session_name} exiting normally", file=sys.stderr)
        sys.exit(0)
    except KeyboardInterrupt:
        print(f"Agent {session_name} interrupted", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Agent {session_name} error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
