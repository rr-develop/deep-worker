"""CLI command to connect to a deep-worker agent."""

import argparse
import asyncio
import sys

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for deep-worker-cli command.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="deep-worker-cli",
        description="Connect to a deep-worker agent",
    )

    parser.add_argument(
        "agent_name",
        nargs="?",
        help="Name of the agent to connect to",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all running agents",
    )

    parser.add_argument(
        "--status",
        metavar="AGENT",
        help="Show status of specified agent",
    )

    parser.add_argument(
        "--stop",
        metavar="AGENT",
        help="Stop specified agent gracefully",
    )

    return parser


async def list_agents() -> int:
    """List all running agents with enhanced formatting.

    Returns:
        Exit code
    """
    registry = ProcessRegistry()
    agents = registry.list_all()

    if not agents:
        print("No running agents found")
        print()
        print("Start an agent with: deep-worker <agent-name>")
        return 0

    print(f"Running agents ({len(agents)}):")
    print("=" * 60)
    for agent in agents:
        print(f"\n  Agent: {agent.name}")
        print(f"  PID:   {agent.pid}")
        print(f"  Started: {agent.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Working dir: {agent.working_dir}")
        print(f"  Socket: {agent.socket_path}")

    print("\n" + "=" * 60)
    print(f"\nUse 'deep-worker-cli --status <agent>' for detailed status")
    print(f"Use 'deep-worker-cli --stop <agent>' to stop an agent")

    return 0


async def show_status(agent_name: str) -> int:
    """Show detailed status of an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Exit code
    """
    try:
        client = AgentClient(agent_name)
        await client.connect()

        # Get full SessionInfo via new API
        session_info = await client.get_status()

        print(f"\nAgent Status: {agent_name}")
        print("=" * 60)
        print(f"  Status:          {session_info.status.value}")
        print(f"  Session Type:    {session_info.session_type}")
        print(f"  Working Dir:     {session_info.working_dir}")
        print(f"  Created:         {session_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if session_info.last_activity:
            print(f"  Last Activity:   {session_info.last_activity.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"  Tasks Completed: {session_info.tasks_completed}")

        if session_info.data:
            print(f"  Message Count:   {session_info.data.message_count}")
            print(f"  Process ID:      {session_info.data.process_id}")
            print(f"  Socket Path:     {session_info.data.socket_path}")

        if session_info.metadata:
            print(f"  Model:           {session_info.metadata.properties.get('model', 'N/A')}")
            print(f"  Clients:         {session_info.metadata.properties.get('clients_count', 0)}")

        if session_info.error_message:
            print(f"\n  ⚠ Error:         {session_info.error_message}")

        print("=" * 60)

        await client.disconnect()
        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print(f"\nAgent '{agent_name}' may not be running.")
        print(f"Use 'deep-worker-cli --list' to see running agents")
        return 1


async def stop_agent(agent_name: str) -> int:
    """Stop an agent gracefully.

    Args:
        agent_name: Name of the agent to stop

    Returns:
        Exit code
    """
    try:
        registry = ProcessRegistry()
        agent_info = registry.lookup(agent_name)

        if not agent_info:
            print(f"Error: Agent '{agent_name}' not found", file=sys.stderr)
            print(f"Start it with: deep-worker {agent_name}")
            return 1

        print(f"Stopping agent '{agent_name}' (PID {agent_info.pid})...")

        # Send SIGTERM for graceful shutdown
        import os
        import signal

        try:
            os.kill(agent_info.pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"Warning: Process {agent_info.pid} not found", file=sys.stderr)
            # Clean up stale registry entry
            registry.unregister(agent_name)
            return 1
        except PermissionError:
            print(f"Error: Permission denied to stop process {agent_info.pid}", file=sys.stderr)
            return 1

        # Wait briefly for shutdown
        import time

        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            if registry.lookup(agent_name) is None:
                break

        # Check if agent stopped
        if registry.lookup(agent_name) is None:
            print(f"Agent '{agent_name}' stopped successfully")
            return 0
        else:
            print(f"Warning: Agent may still be running. Check with --status {agent_name}")
            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def connect_interactive(agent_name: str) -> int:
    """Connect to agent in interactive mode.

    Args:
        agent_name: Name of the agent

    Returns:
        Exit code
    """
    # TODO: Launch Textual UI with AgentClient
    # For now, just show a message
    print(f"Connecting to agent '{agent_name}'...")
    print("Interactive mode not yet implemented")
    print("Use --status to check agent status")
    return 1


def main() -> int:
    """Main entry point for deep-worker-cli command.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle commands
    if args.list:
        return asyncio.run(list_agents())

    if args.status:
        return asyncio.run(show_status(args.status))

    if args.stop:
        return asyncio.run(stop_agent(args.stop))

    if not args.agent_name:
        parser.print_help()
        return 1

    # Connect interactively
    return asyncio.run(connect_interactive(args.agent_name))


if __name__ == "__main__":
    sys.exit(main())
