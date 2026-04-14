"""CLI command to start a deep-worker agent process."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess


# ---------------------------------------------------------------------------
# Restart helpers
# ---------------------------------------------------------------------------

# Argparse defaults for detecting explicit CLI overrides.
# Must match the defaults in create_parser().
_PARSER_DEFAULTS = {
    "provider": "deepseek",
    "model": None,
    "dir": None,
    "skip_permissions": False,
    "task_manager": False,
    "clean": False,
    "memory_only": False,
    "use_sqlite": False,
}


def _get_agent_data_dir(name: str) -> Path:
    """Return the data directory for an agent."""
    return Path("~/.deep_worker").expanduser() / name


def load_launch_config(name: str) -> dict | None:
    """Load launch_config from session_metadata.json.

    Returns:
        The launch_config dict, or None if not available.
    """
    data_dir = _get_agent_data_dir(name)
    metadata_file = data_dir / "session_metadata.json"

    if not metadata_file.exists():
        return None

    try:
        data = json.loads(metadata_file.read_text())
        return data.get("launch_config")
    except (json.JSONDecodeError, OSError):
        return None


def merge_restart_config(
    saved: dict,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Merge saved launch_config with CLI args.

    Saved values are used as defaults.  Any CLI flag that differs from
    the argparse parser default is treated as an explicit override.

    ``--clean`` is NEVER inherited from saved config.

    Args:
        saved: The launch_config dict from session_metadata.json.
        args: Parsed CLI arguments (may contain overrides).

    Returns:
        New argparse.Namespace with effective values.
    """
    effective = argparse.Namespace(**vars(args))

    # Apply saved values as defaults
    effective.provider = saved.get("provider", _PARSER_DEFAULTS["provider"])
    effective.model = saved.get("model", _PARSER_DEFAULTS["model"])
    effective.dir = Path(saved["working_dir"]) if saved.get("working_dir") else _PARSER_DEFAULTS["dir"]
    effective.skip_permissions = saved.get("skip_permissions", False)
    effective.task_manager = saved.get("task_manager", False)
    effective.memory_only = saved.get("memory_only", False)
    effective.use_sqlite = saved.get("use_sqlite", False)
    effective.clean = False  # NEVER inherit --clean

    # Override with explicitly provided CLI args
    if args.provider != _PARSER_DEFAULTS["provider"]:
        effective.provider = args.provider
    if args.model is not _PARSER_DEFAULTS["model"]:
        effective.model = args.model
    if args.dir is not _PARSER_DEFAULTS["dir"]:
        effective.dir = args.dir
    if args.skip_permissions != _PARSER_DEFAULTS["skip_permissions"]:
        effective.skip_permissions = args.skip_permissions
    if args.task_manager != _PARSER_DEFAULTS["task_manager"]:
        effective.task_manager = args.task_manager
    if args.clean != _PARSER_DEFAULTS["clean"]:
        effective.clean = args.clean
    if args.memory_only != _PARSER_DEFAULTS["memory_only"]:
        effective.memory_only = args.memory_only
    if args.use_sqlite != _PARSER_DEFAULTS["use_sqlite"]:
        effective.use_sqlite = args.use_sqlite

    return effective


def build_launch_config(args: argparse.Namespace) -> dict:
    """Build a launch_config dict from effective argparse namespace.

    Args:
        args: Namespace with effective (merged) values.

    Returns:
        Dict suitable for storage in session_metadata.json.
    """
    from deep_worker_cli.config import get_default_model_name

    model = args.model
    if model is None:
        model = get_default_model_name(args.provider) or "unknown"

    working_dir = args.dir
    if working_dir is None:
        original_cwd = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
        working_dir = Path(original_cwd) if original_cwd else Path.cwd()

    return {
        "provider": args.provider,
        "model": model,
        "working_dir": str(working_dir),
        "skip_permissions": args.skip_permissions,
        "task_manager": args.task_manager,
        "memory_only": args.memory_only,
        "use_sqlite": args.use_sqlite,
        "clean": args.clean,
        "started_at": datetime.now().isoformat(),
        "pid": os.getpid(),
        "cli_args": " ".join(sys.argv[1:]),
    }


async def restart_agent(args: argparse.Namespace) -> int:
    """Handle --restart command.

    Loads saved launch_config, merges with CLI overrides,
    stops running agent if needed, then launches.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    from deep_worker_cli.process.agent_stop import stop_agent
    from deep_worker_cli.process.registry import ProcessRegistry

    name = args.name
    if not name:
        print("Error: Agent name is required with --restart", file=sys.stderr)
        return 1

    # 1. Load saved config
    saved = load_launch_config(name)
    if saved is None:
        print(
            f"Error: Cannot restart agent '{name}' — no previous launch configuration found.\n"
            f"Start the agent normally first:  deep-worker {name} --provider ... --model ...",
            file=sys.stderr,
        )
        return 1

    # 2. Merge with CLI overrides
    effective = merge_restart_config(saved, args)

    # 3. Stop running agent if needed
    registry = ProcessRegistry()
    existing = registry.lookup(name)
    if existing:
        try:
            data_dir = _get_agent_data_dir(name)
            await stop_agent(name, registry, data_dir=data_dir)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except TimeoutError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # 4. Handle --clean flag (if explicitly passed)
    if effective.clean:
        from deep_worker_cli.sessions import clean_agent_history
        try:
            await clean_agent_history(name)
        except Exception as e:
            print(f"Error cleaning agent history: {e}", file=sys.stderr)
            return 1

    # 5. Apply effective args and delegate to run_agent
    # Copy effective values back into args for run_agent()
    args.provider = effective.provider
    args.model = effective.model
    args.dir = effective.dir
    args.skip_permissions = effective.skip_permissions
    args.task_manager = effective.task_manager
    args.memory_only = effective.memory_only
    args.use_sqlite = effective.use_sqlite
    args.clean = False  # Already handled above if needed

    return await run_agent(args)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for deep-worker command.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="deep-worker",
        description="Start a deep-worker agent process",
    )

    parser.add_argument(
        "name",
        nargs="?",
        help="Unique name for this agent (not required with --list)",
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Working directory for the agent (default: current directory)",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: provider-specific default)",
    )

    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "gemini", "openai", "claude-code"],
        help="AI provider to use (default: deepseek)",
    )

    parser.add_argument(
        "-sp",
        "--skip-permissions",
        action="store_true",
        dest="skip_permissions",
        help="Skip permissions (auto-approve dangerous operations)",
    )

    parser.add_argument(
        "-tm",
        "--task-manager",
        action="store_true",
        dest="task_manager",
        help="Enable TaskManager for AI-powered task evaluation and correction",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean all history and state before starting (agent must not be running)",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart agent with parameters from previous launch. "
             "Additional options override saved values. "
             "--clean is NOT inherited; pass it explicitly if needed.",
    )

    parser.add_argument(
        "--list", "-ls",
        action="store_true",
        dest="list_threads",
        help="List all conversation threads in database",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for --list (default: text)",
    )

    parser.add_argument(
        "--with-counts",
        action="store_true",
        dest="with_counts",
        help="Include message counts (slow, for --list only)",
    )

    parser.add_argument(
        "--agent",
        metavar="NAME",
        help="Filter by agent name (for --list only)",
    )

    parser.add_argument(
        "--sort",
        choices=["thread_id", "agent", "updated"],
        default="thread_id",
        help="Sort field for --list (default: thread_id)",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order for --list",
    )

    parser.add_argument(
        "-rm",
        "--remove-thread",
        metavar="THREAD_ID",
        dest="remove_thread",
        help="Remove a specific thread from the database",
    )

    parser.add_argument(
        "--history",
        metavar="THREAD_ID",
        dest="show_history",
        help="Show message history for a specific thread (works offline)",
    )

    parser.add_argument(
        "--remove-before",
        metavar="DATE",
        dest="remove_before",
        help="Remove all threads older than DATE (ISO 8601 or relative like 30d, 1y)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would be deleted without actually deleting (for -rm and --remove-before)",
    )

    # Mutually exclusive group for storage mode
    storage_group = parser.add_mutually_exclusive_group()

    storage_group.add_argument(
        "--memory-only",
        action="store_true",
        dest="memory_only",
        help="Use in-memory storage only (no persistence to database)",
    )

    storage_group.add_argument(
        "--use-sqlite",
        action="store_true",
        dest="use_sqlite",
        help="Force SQLite persistence (even in test environment)",
    )

    return parser


async def run_agent(args: argparse.Namespace) -> int:
    """Run the agent process.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Determine working directory
        # Priority: --dir arg > DEEP_WORKER_ORIGINAL_CWD env > current dir
        working_dir = args.dir
        if working_dir is None:
            # Get from environment variable set by bash wrapper
            original_cwd = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
            if original_cwd:
                working_dir = Path(original_cwd)
            else:
                working_dir = Path.cwd()

        # Load or create config
        if args.config:
            config = AgentConfig.from_file(args.config)
        else:
            config = AgentConfig(
                model=args.model,
                working_dir=working_dir,
                provider=args.provider,
            )

        # Build launch_config for persistent storage
        # This dict is saved to ~/.deep_worker/{name}/session_metadata.json
        # so that launch parameters survive process termination.
        launch_config = {
            "provider": config.provider,
            "model": config.resolved_model,
            "working_dir": str(config.working_dir),
            "skip_permissions": args.skip_permissions,
            "task_manager": args.task_manager,
            "memory_only": args.memory_only,
            "use_sqlite": args.use_sqlite,
            "clean": getattr(args, "clean", False),
            "started_at": datetime.now().isoformat(),
            "pid": os.getpid(),
            "cli_args": " ".join(sys.argv[1:]),
        }

        # Create and start agent
        agent = AgentProcess(
            name=args.name,
            config=config,
            skip_permissions=args.skip_permissions,
            enable_task_manager=args.task_manager,
            memory_only=args.memory_only,
            use_sqlite=args.use_sqlite,
            launch_config=launch_config,
        )

        try:
            await agent.start()

            # Keep running until shutdown signal
            while agent.running:
                await asyncio.sleep(1)

            return 0

        finally:
            # Always cleanup checkpointer on exit
            await agent.cleanup()

    except KeyboardInterrupt:
        print("\nShutdown requested")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def daemonize(name: str, log_dir: Path | None = None) -> None:
    """Daemonize the current process (Unix double-fork).

    Args:
        name: Agent name for log files
        log_dir: Directory for log files (default: ~/.deep_worker/logs)
    """
    # Set up log directory
    if log_dir is None:
        log_dir = Path("~/.deep_worker").expanduser() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file paths
    stdout_log = log_dir / f"{name}.out.log"
    stderr_log = log_dir / f"{name}.err.log"

    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - print info and exit
            print(f"Agent '{name}' starting in background (PID will be written to /tmp/deep_worker_{name}.pid)")
            print(f"Logs: {stdout_log}")
            sys.exit(0)
    except OSError as e:
        print(f"Fork #1 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # First child exits
            sys.exit(0)
    except OSError as e:
        print(f"Fork #2 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Open log files
    stdout_fd = os.open(str(stdout_log), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    stderr_fd = os.open(str(stderr_log), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)

    # Redirect stdin to /dev/null
    stdin_fd = os.open(os.devnull, os.O_RDONLY)

    # Duplicate file descriptors
    os.dup2(stdin_fd, sys.stdin.fileno())
    os.dup2(stdout_fd, sys.stdout.fileno())
    os.dup2(stderr_fd, sys.stderr.fileno())

    # Close original file descriptors
    os.close(stdin_fd)
    os.close(stdout_fd)
    os.close(stderr_fd)


async def remove_thread_command(thread_id: str, dry_run: bool) -> int:
    """Handle -rm/--remove-thread command.

    Args:
        thread_id: Thread ID to remove
        dry_run: If True, show what would be deleted without deleting

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli.sessions import remove_thread
    from deep_worker_cli.config import console

    try:
        result = await remove_thread(thread_id, dry_run=dry_run)

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        if dry_run:
            console.print(f"[yellow]Dry run - would delete thread '{thread_id}':[/yellow]")
            console.print(f"  - Checkpoints: {result['checkpoints_found']}")
            if result.get("writes_found", 0) > 0:
                console.print(f"  - Writes: {result['writes_found']}")
            console.print("\n[dim]Run without --dry-run to actually delete[/dim]")
        else:
            console.print(f"[green]✓ Removed thread '{thread_id}':[/green]")
            console.print(f"  - Checkpoints deleted: {result['checkpoints_deleted']}")
            if result.get("writes_deleted", 0) > 0:
                console.print(f"  - Writes deleted: {result['writes_deleted']}")

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


async def show_history_command(thread_id: str, format: str) -> int:
    """Handle --history command.

    Args:
        thread_id: Thread ID to show history for
        format: Output format ("text" or "json")

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli.sessions import get_thread_history
    from deep_worker_cli.config import console

    try:
        result = await get_thread_history(thread_id, format=format)

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        if format == "json":
            print(result["json_output"])
        else:
            # Display header similar to dwcli --history
            console.print(f"\nConversation History for '{thread_id}':")
            console.print(f"Showing {len(result['messages'])} messages ({len(result['checkpoints'])} checkpoints)")
            console.print("=" * 60)

            # Display messages with sender labels, timestamps, and separators
            for i, msg in enumerate(result['messages'], 1):
                # Handle both dict and LangChain Message objects
                if isinstance(msg, dict):
                    msg_type = msg.get('type', 'unknown')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp')
                elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                    # LangChain Message object
                    msg_type = msg.type if hasattr(msg, 'type') else type(msg).__name__
                    content = msg.content
                    timestamp = getattr(msg, 'timestamp', None)
                else:
                    msg_type = type(msg).__name__
                    content = str(msg)
                    timestamp = None

                # Map message types to display labels (matching dwcli --history format)
                if msg_type in ("human", "HumanMessage"):
                    type_label = "USER"
                    separator = "-" * 60
                elif msg_type in ("ai", "AIMessage"):
                    type_label = "ASSISTANT"
                    separator = "=" * 60
                elif msg_type in ("tool", "ToolMessage"):
                    type_label = "TOOL"
                    separator = "." * 60
                elif msg_type in ("system", "SystemMessage"):
                    type_label = "SYSTEM"
                    separator = "-" * 60
                else:
                    type_label = msg_type.upper()
                    separator = "-" * 60

                # Format timestamp for display with millisecond precision
                # This precision is critical for distinguishing message order when
                # comparing events across different agent threads. Database stores
                # timestamps with microsecond precision (e.g., "2026-03-29T22:40:17.822305+00:00")
                # and we preserve milliseconds in display for accurate event sequencing.
                timestamp_str = ""
                if timestamp:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp)
                        # Format: "Mar 29, 10:40:17.822pm" (includes seconds + milliseconds)
                        ms = int(dt.microsecond / 1000)
                        timestamp_str = f" - {dt.strftime('%b %d, %-I:%M:%S')}.{ms:03d}{dt.strftime('%p').lower()}"
                    except (ValueError, AttributeError):
                        pass

                console.print(f"\n[{type_label}]{timestamp_str}")
                console.print(content)

                # Don't print separator after last message
                if i < len(result['messages']):
                    console.print(separator)

            console.print("\n" + "=" * 60)

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


async def remove_before_command(date_str: str, dry_run: bool) -> int:
    """Handle --remove-before command.

    Args:
        date_str: Date string (ISO 8601 or relative)
        dry_run: If True, show what would be deleted without deleting

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli.sessions import remove_threads_before
    from deep_worker_cli.config import console

    try:
        result = await remove_threads_before(date_str, dry_run=dry_run)

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        cutoff_date = result["cutoff_date"]

        if dry_run:
            console.print(f"[yellow]Dry run - would delete threads before {cutoff_date}:[/yellow]")
            console.print(f"  - Threads: {result['threads_found']}")
            console.print(f"  - Checkpoints: {result['checkpoints_found']}")
            if result.get("writes_found", 0) > 0:
                console.print(f"  - Writes: {result['writes_found']}")
            console.print("\n[dim]Run without --dry-run to actually delete[/dim]")
        else:
            console.print(f"[green]✓ Removed threads before {cutoff_date}:[/green]")
            console.print(f"  - Threads deleted: {result['threads_deleted']}")
            console.print(f"  - Checkpoints deleted: {result['checkpoints_deleted']}")
            if result.get("writes_deleted", 0) > 0:
                console.print(f"  - Writes deleted: {result['writes_deleted']}")

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


def main() -> int:
    """Main entry point for deep-worker command.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --list option (mutually exclusive with running agent)
    if args.list_threads:
        if args.name:
            print("Error: Cannot specify agent name with --list option", file=sys.stderr)
            return 2
        return asyncio.run(list_threads_command(args))

    # Handle -rm/--remove-thread option
    if args.remove_thread:
        return asyncio.run(remove_thread_command(args.remove_thread, args.dry_run))

    # Handle --history option
    if args.show_history:
        return asyncio.run(show_history_command(args.show_history, args.format))

    # Handle --remove-before option
    if args.remove_before:
        return asyncio.run(remove_before_command(args.remove_before, args.dry_run))

    # Require agent name for all other operations
    if not args.name:
        print("Error: Agent name is required (or use --list)", file=sys.stderr)
        parser.print_help()
        return 2

    # Validate agent name: prevent ':tm' suffix (reserved for TaskManager)
    if ":tm" in args.name:
        print(
            f"Error: Agent name '{args.name}' cannot contain ':tm' suffix.\n"
            f"This suffix is reserved for TaskManager threads.\n"
            f"Please choose a different agent name.",
            file=sys.stderr
        )
        return 1

    # Handle --restart option
    if getattr(args, "restart", False):
        # Pre-flight checks BEFORE daemonize (so errors are visible in terminal)
        saved = load_launch_config(args.name)
        if saved is None:
            print(
                f"Error: Cannot restart agent '{args.name}' — no previous launch configuration found.\n"
                f"Start the agent normally first:  deep-worker {args.name} --provider ... --model ...",
                file=sys.stderr,
            )
            return 1

        # Check if running agent can be stopped
        from deep_worker_cli.process.registry import ProcessRegistry
        from deep_worker_cli.process.agent_stop import stop_agent as _stop_agent, _read_agent_status

        registry = ProcessRegistry()
        existing = registry.lookup(args.name)
        if existing:
            data_dir = _get_agent_data_dir(args.name)
            status = _read_agent_status(data_dir)
            if status == "running":
                print(
                    f"Error: Cannot restart agent '{args.name}' — it is actively running (status: running).\n"
                    f"Wait for it to finish or interrupt it first.",
                    file=sys.stderr,
                )
                return 1

        # Validate registry writable
        validation = registry.validate_writable()
        if not validation["writable"]:
            print(f"ERROR: Cannot start agent - registry file is not writable", file=sys.stderr)
            print(f"  {validation['error']}", file=sys.stderr)
            return 1
        if validation.get("fixed"):
            print(f"✓ Fixed registry file ownership")

        # Save CWD before daemonize changes it
        effective = merge_restart_config(saved, args)
        if effective.dir:
            os.environ["DEEP_WORKER_ORIGINAL_CWD"] = str(Path(effective.dir).resolve())
        elif "DEEP_WORKER_ORIGINAL_CWD" not in os.environ:
            os.environ["DEEP_WORKER_ORIGINAL_CWD"] = os.getcwd()

        daemonize(args.name)

        # After daemonize: stop running agent (if any), then launch
        return asyncio.run(restart_agent(args))

    # Handle --clean flag BEFORE daemonization
    if args.clean:
        from deep_worker_cli.process.registry import ProcessRegistry
        from deep_worker_cli.sessions import clean_agent_history

        # Check if agent is already running
        registry = ProcessRegistry()
        existing = registry.lookup(args.name)
        if existing:
            print(
                f"Error: Agent '{args.name}' is already running (PID {existing.pid})",
                file=sys.stderr,
            )
            print(
                f"Stop it first with: pkill -f 'deep-worker {args.name}'",
                file=sys.stderr,
            )
            return 1

        # Clean agent history
        try:
            asyncio.run(clean_agent_history(args.name))
        except Exception as e:
            print(f"Error cleaning agent history: {e}", file=sys.stderr)
            return 1

    # Save current working directory BEFORE daemonize changes it to /
    # This allows run_agent() to use the original directory after daemonization
    # Priority: --dir arg > DEEP_WORKER_ORIGINAL_CWD env var (from bash) > current directory
    if args.dir:
        os.environ["DEEP_WORKER_ORIGINAL_CWD"] = str(args.dir.resolve())
    elif "DEEP_WORKER_ORIGINAL_CWD" not in os.environ:
        # Only set if not already set by bash wrapper
        os.environ["DEEP_WORKER_ORIGINAL_CWD"] = os.getcwd()

    # CRITICAL: Validate registry file is writable BEFORE daemonizing
    # If we daemonize first and then fail, user won't see the error message!
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()
    validation = registry.validate_writable()

    if not validation["writable"]:
        # Registry is not writable - show error and exit
        print(f"ERROR: Cannot start agent - registry file is not writable", file=sys.stderr)
        print(f"  {validation['error']}", file=sys.stderr)
        return 1

    if validation.get("fixed"):
        # Ownership was fixed automatically
        print(f"✓ Fixed registry file ownership")

    # Daemonize before starting agent
    daemonize(args.name)

    # Run agent
    return asyncio.run(run_agent(args))


async def list_threads_command(args: argparse.Namespace) -> int:
    """Handle --list command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli import sessions

    try:
        # Get all threads
        threads = await sessions.list_all_threads(
            agent_name=args.agent,
            sort_by=args.sort,
            reverse=args.reverse,
            include_message_count=args.with_counts,
        )

        # Format output
        if args.format == "json":
            output = format_threads_json(threads, args.sort, args.agent)
        else:  # text
            output = format_threads_text(threads, args.with_counts)

        print(output)
        return 0

    except Exception as e:
        print(f"Error listing threads: {e}", file=sys.stderr)
        return 1


def format_threads_text(threads: list[dict], with_counts: bool) -> str:
    """Format threads as aligned text table.

    Args:
        threads: List of thread dicts with thread_id, agent_name, updated_at, optional message_count
        with_counts: Include message count column

    Returns:
        Formatted text string
    """
    if not threads:
        return "\nNo threads found in database.\n"

    # Column widths
    thread_id_width = 24
    agent_name_width = 20
    messages_width = 10
    updated_width = 20

    # Build header
    if with_counts:
        header = f"{'Thread ID':<{thread_id_width}} {'Agent Name':<{agent_name_width}} {'Messages':>{messages_width}} {'Last Updated':<{updated_width}}"
        separator = "=" * (thread_id_width + agent_name_width + messages_width + updated_width + 3)
    else:
        header = f"{'Thread ID':<{thread_id_width}} {'Agent Name':<{agent_name_width}} {'Last Updated':<{updated_width}}"
        separator = "=" * (thread_id_width + agent_name_width + updated_width + 2)

    lines = ["\n" + header, separator]

    # Format each thread
    for thread in threads:
        thread_id = thread["thread_id"][:thread_id_width]  # Truncate if too long
        agent_name = thread["agent_name"] if thread["agent_name"] else "(self)"
        agent_name = agent_name[:agent_name_width]  # Truncate if too long

        # Format timestamp
        updated_at = thread.get("updated_at", "")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at).astimezone()
                updated_str = dt.strftime("%b %d, %-I:%M%p").lower()
                # Capitalize month
                parts = updated_str.split(" ")
                if parts:
                    parts[0] = parts[0].capitalize()
                    updated_str = " ".join(parts)
            except (ValueError, TypeError):
                updated_str = updated_at[:updated_width]
        else:
            updated_str = ""

        if with_counts:
            message_count = thread.get("message_count", 0)
            line = f"{thread_id:<{thread_id_width}} {agent_name:<{agent_name_width}} {message_count:>{messages_width}} {updated_str:<{updated_width}}"
        else:
            line = f"{thread_id:<{thread_id_width}} {agent_name:<{agent_name_width}} {updated_str:<{updated_width}}"

        lines.append(line)

    lines.append("")  # Empty line at end
    return "\n".join(lines)


def format_threads_json(threads: list[dict], sort_by: str, filter_agent: str | None) -> str:
    """Format threads as JSON.

    Args:
        threads: List of thread dicts
        sort_by: Sort field used
        filter_agent: Agent filter applied (or None)

    Returns:
        JSON string with pretty printing
    """
    output = {
        "threads": threads,
        "total": len(threads),
        "sort_by": sort_by,
        "filter_agent": filter_agent,
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
