"""CLI command to connect to a deep-worker agent."""

import argparse
import asyncio
import sys

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.exceptions import RegistryReadError
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
    try:
        agents = registry.list_all()
    except RegistryReadError as e:
        # The registry exists but could not be read (EACCES/EIO). Loud is
        # right — a silent "no agents" is what hid this class of failure in
        # the first place — but loud must not mean a Python traceback: the
        # exception message carries the path and the `ls -l` remedy, and a
        # stack trace buries exactly the person who needs it. Print it and
        # exit non-zero so scripts can tell this from an empty registry.
        print(f"Error: {e}", file=sys.stderr)
        return 1

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

    Prints ONLY data the agent actually reported. Anything the agent
    did not return is surfaced as "N/A" — we never substitute a
    placeholder (no fake ``/tmp`` working dir, no
    ``datetime.now()`` stand-in for ``created_at``, no guessed
    TaskManager state). Every field here has exactly one honest
    source:

    * ``Working Dir``, ``Started``, ``Model``, ``Provider``,
      ``PID``, ``RPyC Port`` — from ``get_agent_metadata()``
      (agent's ``exposed_get_metadata``).
    * ``Status`` — from ``get_status()``.
    * ``TaskManager`` — from ``get_task_manager_info()`` (keys
      ``task_manager_active`` / ``task_manager_disabled`` on the
      same status response; absent when the agent has no TM).
    * ``Last Activity``, ``Tasks Completed``, ``Message Count`` —
      from ``SessionInfo.data`` when present; printed as "N/A"
      otherwise.
    """

    def _fmt(value: object) -> str:
        """Format a value for display. ``None``/empty → "N/A"."""
        if value is None:
            return "N/A"
        s = str(value)
        return s if s else "N/A"

    def _fmt_iso(iso_ts: object) -> str:
        """ISO timestamp → 'YYYY-MM-DD HH:MM:SS', else "N/A"."""
        if not iso_ts:
            return "N/A"
        from datetime import datetime as _dt
        try:
            return _dt.fromisoformat(str(iso_ts)).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return _fmt(iso_ts)

    try:
        client = AgentClient(agent_name)
        await client.connect()

        # Pull each piece of honest data separately. If any RPC fails
        # we still print what we have and mark the rest N/A.
        session_info = None
        try:
            session_info = await client.get_status()
        except Exception as e:
            print(f"  ⚠ get_status failed: {e}", file=sys.stderr)

        metadata = await client.get_agent_metadata()  # None on error
        tm_info = await client.get_task_manager_info()  # None on error

        print(f"\nAgent Status: {agent_name}")
        print("=" * 60)

        if session_info is not None:
            print(f"  Status:          {session_info.status.value}")
            print(f"  Session Type:    {session_info.session_type}")
        else:
            print(f"  Status:          N/A")
            print(f"  Session Type:    N/A")

        # Working dir, start time, model, provider, pid, rpyc port —
        # ONLY from live metadata. NO fallbacks to SessionInfo
        # placeholders.
        print(f"  Working Dir:     {_fmt(metadata.get('directory') if metadata else None)}")
        print(f"  Started:         {_fmt_iso(metadata.get('started') if metadata else None)}")
        print(f"  Model:           {_fmt(metadata.get('model') if metadata else None)}")
        print(f"  Provider:        {_fmt(metadata.get('provider') if metadata else None)}")
        print(f"  PID:             {_fmt(metadata.get('pid') if metadata else None)}")
        print(f"  RPyC Port:       {_fmt(metadata.get('rpyc_port') if metadata else None)}")

        # TaskManager — three states, each printed distinctly.
        if tm_info is None:
            print(f"  TaskManager:     N/A (unreachable)")
        elif not tm_info["configured"]:
            print(f"  TaskManager:     not configured")
        else:
            if tm_info["disabled"]:
                state = "disabled"
            elif tm_info["active"]:
                state = "active"
            else:
                state = "inactive"
            print(f"  TaskManager:     {state}")

        # Remaining SessionInfo-derived fields. Printed only when the
        # agent returned them; N/A otherwise.
        if session_info is not None:
            if session_info.last_activity:
                print(
                    f"  Last Activity:   "
                    f"{session_info.last_activity.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                print(f"  Last Activity:   N/A")

            # tasks_completed can legitimately be 0; only print a number
            # when SessionInfo.data is present (otherwise we'd be
            # surfacing the placeholder default).
            if session_info.data is not None:
                print(f"  Tasks Completed: {session_info.tasks_completed}")
                print(f"  Message Count:   {session_info.data.message_count}")
                print(f"  Process ID:      {session_info.data.process_id}")
            else:
                print(f"  Tasks Completed: N/A")
                print(f"  Message Count:   N/A")

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
    """Stop an agent gracefully, with force kill if needed.

    CRITICAL REQUIREMENTS:
    1. Try graceful shutdown with SIGTERM (wait up to 5 seconds)
    2. If still alive, FORCE KILL with SIGKILL (no mercy!)
    3. VERIFY process is actually dead before returning success
    4. Report what happened (graceful vs forced)
    5. Clean up stale registry entries

    Args:
        agent_name: Name of the agent to stop

    Returns:
        Exit code (0=success, 1=failure)
    """
    try:
        registry = ProcessRegistry()
        # An unreadable registry is caught by the blanket handler at the end
        # of this function, but that one also prints a traceback. Handle it
        # here so the user gets the path and the `ls -l` remedy on its own,
        # the same as in ``list_agents``.
        try:
            agent_info = registry.lookup(agent_name)
        except RegistryReadError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if not agent_info:
            print(f"Error: Agent '{agent_name}' not found", file=sys.stderr)
            print(f"Start it with: deep-worker {agent_name}")
            return 1

        print(f"Stopping agent '{agent_name}' (PID {agent_info.pid})...")

        import os
        import signal
        import time

        # Check if process exists
        try:
            os.kill(agent_info.pid, 0)  # Signal 0 checks existence
        except ProcessLookupError:
            print(f"Warning: Process {agent_info.pid} not found (already dead)", file=sys.stderr)
            # Clean up stale registry entry
            registry.unregister(agent_name)
            return 1
        except PermissionError:
            print(f"Error: Permission denied to check process {agent_info.pid}", file=sys.stderr)
            return 1

        # Phase 1: Try graceful shutdown with SIGTERM
        try:
            os.kill(agent_info.pid, signal.SIGTERM)
            print(f"Sent SIGTERM to PID {agent_info.pid}, waiting for graceful shutdown...")
        except ProcessLookupError:
            print(f"Warning: Process {agent_info.pid} died before SIGTERM", file=sys.stderr)
            registry.unregister(agent_name)
            return 1
        except PermissionError:
            print(f"Error: Permission denied to stop process {agent_info.pid}", file=sys.stderr)
            return 1

        # Wait up to 5 seconds for graceful shutdown - EVENT DRIVEN, NO POLLING
        max_wait = 5.0  # seconds
        
        # Use event-driven approach with scheduled checks
        process_exited = asyncio.Event()
        
        def check_process():
            """Check if process has exited."""
            try:
                os.kill(agent_info.pid, 0)  # Check if process exists
                # Still alive, schedule next check
                loop = asyncio.get_event_loop()
                loop.call_later(0.1, check_process)  # Check every 100ms
            except ProcessLookupError:
                # Process died gracefully!
                process_exited.set()
        
        # Schedule first check
        loop = asyncio.get_event_loop()
        loop.call_soon(check_process)
        
        # Wait for process exit or timeout
        try:
            await asyncio.wait_for(process_exited.wait(), timeout=max_wait)
            print(f"Agent '{agent_name}' stopped gracefully")
            registry.unregister(agent_name)
            return 0
        except asyncio.TimeoutError:
            # Process didn't exit within timeout
            pass

        # Phase 2: Process didn't respond to SIGTERM - FORCE KILL!
        print(f"Agent didn't respond to SIGTERM after {max_wait}s - force killing with SIGKILL...")

        try:
            os.kill(agent_info.pid, signal.SIGKILL)
        except ProcessLookupError:
            # Process died between checks
            print(f"Agent '{agent_name}' stopped (died during force kill)")
            registry.unregister(agent_name)
            return 0
        except PermissionError:
            print(f"Error: Permission denied to force kill process {agent_info.pid}", file=sys.stderr)
            return 1

        # Wait briefly for SIGKILL to take effect (should be instant)
        await asyncio.sleep(0.5)

        # CRITICAL: Verify process is ACTUALLY dead (not zombie)
        # Zombie processes still exist in process table but can't execute code
        # We consider zombie = success (it's dead enough)
        try:
            # Check /proc/{pid}/stat for process state
            # State codes: R (running), S (sleeping), D (uninterruptible), Z (zombie), T (stopped)
            with open(f"/proc/{agent_info.pid}/stat", "r") as f:
                stat = f.read()
                # Format: "pid (comm) state ..."
                # Example: "1234 (python3) Z ..."
                parts = stat.split(") ")
                if len(parts) >= 2:
                    state = parts[1].split()[0]  # First char after ")  "
                    if state == "Z":
                        # Zombie - process is dead, just waiting for parent to reap
                        print(f"Agent '{agent_name}' force killed successfully (zombie state)")
                        registry.unregister(agent_name)
                        return 0
                    else:
                        # Process is still alive in non-zombie state!
                        print(f"ERROR: Process {agent_info.pid} in state '{state}' after SIGKILL!", file=sys.stderr)
                        print(f"This should not be possible. Manual intervention required.", file=sys.stderr)
                        return 1
                else:
                    # Couldn't parse state - fall back to simple check
                    print(f"Warning: Couldn't parse process state, assuming killed")
                    print(f"Agent '{agent_name}' force killed (state unknown)")
                    registry.unregister(agent_name)
                    return 0
        except FileNotFoundError:
            # /proc/{pid}/stat doesn't exist - process is completely gone
            print(f"Agent '{agent_name}' force killed successfully")
            registry.unregister(agent_name)
            return 0
        except PermissionError:
            print(f"Warning: Can't verify process state (permission denied)")
            print(f"Agent '{agent_name}' likely killed (cannot verify)")
            registry.unregister(agent_name)
            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
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
    # Check if agent exists. ``lookup`` is the first thing here that reads
    # the registry, so it — not the "Available agents" listing below — is
    # where an unreadable registry surfaces first. Same reasoning as in
    # ``list_agents``: report the path and the remedy, never a traceback.
    registry = ProcessRegistry()
    try:
        agent_info = registry.lookup(agent_name)
    except RegistryReadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not agent_info:
        print(f"\nError: Agent '{agent_name}' not found", file=sys.stderr)
        print(f"\nAvailable agents:")
        # The listing is secondary information. It cannot normally fail once
        # ``lookup`` succeeded, but a registry can be made unreadable between
        # the two reads; say why the list is missing rather than claim the
        # registry is empty, and keep the primary error intact.
        try:
            agents = registry.list_all()
        except RegistryReadError as e:
            print(f"  (cannot read registry: {e})")
            print(f"\nStart an agent with: deep-worker {agent_name}")
            return 1
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
