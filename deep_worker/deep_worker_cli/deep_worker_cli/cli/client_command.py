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

    parser.add_argument(
        "-p",
        "--prompt",
        metavar="TEXT",
        help="Send a single prompt in headless mode (non-interactive)",
    )

    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for headless mode (default: text)",
    )

    parser.add_argument(
        "--last-response",
        metavar="AGENT",
        help="Show the final assistant response from agent's conversation history",
    )

    parser.add_argument(
        "--history",
        metavar="AGENT",
        help="Show conversation history for specified agent",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of messages to show in history (default: 20)",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of recent messages to skip for pagination (default: 0)",
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
        print(f"  RPyC port: {agent.rpyc_port}")

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


async def send_prompt(agent_name: str, prompt: str, output_format: str = "text") -> int:
    """Send a single prompt to agent in headless mode.

    Args:
        agent_name: Name of the agent
        prompt: Prompt text to send
        output_format: Output format ("text" or "json")

    Returns:
        Exit code
    """
    try:
        client = AgentClient(agent_name)
        await client.connect()

        # Send prompt to agent
        response = await client.invoke_agent(prompt, stream=False)

        if not response.get("success"):
            error = response.get("error", "Unknown error")
            if output_format == "json":
                import json
                print(json.dumps({
                    "success": False,
                    "error": error
                }))
            else:
                print(f"Error: {error}", file=sys.stderr)
            await client.disconnect()
            return 1

        # Extract response content
        response_data = response.get("response", {})
        content = response_data.get("content", "")

        # Output based on format
        if output_format == "json":
            import json
            print(json.dumps({
                "success": True,
                "response": {
                    "role": response_data.get("role", "assistant"),
                    "content": content
                },
                "agent_name": agent_name
            }, ensure_ascii=False, indent=2))
        else:
            # Text format: just print the content
            print(content)

        await client.disconnect()
        return 0

    except Exception as e:
        if output_format == "json":
            import json
            print(json.dumps({
                "success": False,
                "error": str(e)
            }))
        else:
            print(f"\nError: {e}", file=sys.stderr)
            print(f"Agent '{agent_name}' may not be running.")
            print(f"Use 'deep-worker-cli --list' to see running agents")
        return 1


async def show_last_response(agent_name: str) -> int:
    """Show the final assistant response from agent's conversation history.

    Args:
        agent_name: Name of the agent

    Returns:
        Exit code
    """
    try:
        client = AgentClient(agent_name)
        await client.connect()

        # Get history with limit=50 to ensure we find last AI message
        response = await client.get_history(limit=50, offset=0)

        if not response.get("success"):
            print(f"Error: {response.get('error', 'Failed to get history')}", file=sys.stderr)
            return 1

        history = response.get("history", [])

        if not history:
            print(f"\nNo conversation history found for agent '{agent_name}'")
            return 0

        # Find last AIMessage in history
        last_ai_message = None
        for message in reversed(history):
            if message.get("type") == "AIMessage":
                last_ai_message = message
                break

        if not last_ai_message:
            print(f"\nNo assistant responses found in agent '{agent_name}' history")
            return 0

        # Display the last response
        print(f"\nLast Response from '{agent_name}':")
        print("=" * 60)
        print(last_ai_message.get("content", "(empty)"))
        print("=" * 60)

        await client.disconnect()
        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print(f"Agent '{agent_name}' may not be running.")
        print(f"Use 'deep-worker-cli --list' to see running agents")
        return 1


async def show_history(agent_name: str, limit: int = 20, offset: int = 0) -> int:
    """Show conversation history for an agent.

    Args:
        agent_name: Name of the agent
        limit: Maximum number of messages to retrieve
        offset: Number of recent messages to skip for pagination

    Returns:
        Exit code
    """
    try:
        client = AgentClient(agent_name)
        await client.connect()

        response = await client.get_history(limit=limit, offset=offset)

        if not response.get("success"):
            print(f"Error: {response.get('error', 'Failed to get history')}", file=sys.stderr)
            return 1

        history = response.get("history", [])
        total = response.get("total", 0)
        has_more = response.get("has_more", False)

        if not history:
            print(f"\nNo conversation history found for agent '{agent_name}'")
            return 0

        # Display history
        print(f"\nConversation History for '{agent_name}':")
        print(f"Showing {len(history)} of {total} messages (offset: {offset})")
        print("=" * 60)

        for i, message in enumerate(history, start=1):
            msg_type = message.get("type", "Unknown")
            content = message.get("content", "(empty)")

            # Format message type for display
            if msg_type == "HumanMessage":
                type_label = "USER"
                separator = "-" * 60
            elif msg_type == "AIMessage":
                type_label = "ASSISTANT"
                separator = "=" * 60
            elif msg_type == "ToolMessage":
                type_label = "TOOL"
                separator = "." * 60
            else:
                type_label = msg_type.upper()
                separator = "-" * 60

            print(f"\n[{type_label}]")
            print(content)

            if i < len(history):  # Don't print separator after last message
                print(separator)

        print("\n" + "=" * 60)

        if has_more:
            print(f"\nMore messages available. Use --offset {offset + limit} to see next page")

        if offset > 0:
            prev_offset = max(0, offset - limit)
            print(f"Use --offset {prev_offset} to see previous page")

        await client.disconnect()
        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print(f"Agent '{agent_name}' may not be running.")
        print(f"Use 'deep-worker-cli --list' to see running agents")
        return 1


async def connect_interactive(agent_name: str) -> int:
    """Connect to agent in interactive mode.

    Args:
        agent_name: Name of the agent

    Returns:
        Exit code
    """
    # Check if agent exists
    registry = ProcessRegistry()
    agent_info = registry.lookup(agent_name)

    if not agent_info:
        print(f"\nError: Agent '{agent_name}' not found", file=sys.stderr)
        print(f"\nAvailable agents:")
        agents = registry.list_all()
        if agents:
            for agent in agents:
                print(f"  - {agent.name}")
        else:
            print("  (none)")
        print(f"\nStart an agent with: deep-worker {agent_name}")
        return 1

    # Launch Textual UI using async mode (matches deepagents implementation)
    # This is critical for proper PTY input handling with stdin redirection
    from deep_worker_cli.cli.interactive import InteractiveApp

    try:
        app = InteractiveApp(agent_name=agent_name)
        await app.run_async()
        return 0
    except Exception as e:
        print(f"\nError launching interactive mode: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
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

    if args.last_response:
        return asyncio.run(show_last_response(args.last_response))

    if args.history:
        return asyncio.run(show_history(args.history, args.limit, args.offset))

    # Handle headless mode (-p/--prompt)
    if args.prompt:
        if not args.agent_name:
            print("Error: agent name required when using -p/--prompt", file=sys.stderr)
            print("Usage: dwcli -p 'prompt text' <agent_name>", file=sys.stderr)
            return 1
        return asyncio.run(send_prompt(args.agent_name, args.prompt, args.output_format))

    if not args.agent_name:
        parser.print_help()
        return 1

    # Connect interactively using async mode (matches deepagents)
    return asyncio.run(connect_interactive(args.agent_name))


if __name__ == "__main__":
    sys.exit(main())
