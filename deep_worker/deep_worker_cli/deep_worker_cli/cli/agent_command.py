"""CLI command to start a deep-worker agent process."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess


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
        help="Unique name for this agent",
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Working directory for the agent (default: current directory)",
    )

    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model to use (default: deepseek-chat)",
    )

    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "gemini", "openai"],
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

        # Create and start agent
        agent = AgentProcess(
            name=args.name,
            config=config,
            skip_permissions=args.skip_permissions,
            enable_task_manager=args.task_manager,
        )

        await agent.start()

        # Keep running until shutdown signal
        while agent.running:
            await asyncio.sleep(1)

        return 0

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
        log_dir = Path("~/.deep_worker/logs").expanduser()
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


def main() -> int:
    """Main entry point for deep-worker command.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    # Save current working directory BEFORE daemonize changes it to /
    # This allows run_agent() to use the original directory after daemonization
    # Priority: --dir arg > DEEP_WORKER_ORIGINAL_CWD env var (from bash) > current directory
    if args.dir:
        os.environ["DEEP_WORKER_ORIGINAL_CWD"] = str(args.dir.resolve())
    elif "DEEP_WORKER_ORIGINAL_CWD" not in os.environ:
        # Only set if not already set by bash wrapper
        os.environ["DEEP_WORKER_ORIGINAL_CWD"] = os.getcwd()

    # Daemonize before starting agent
    daemonize(args.name)

    # Run agent
    return asyncio.run(run_agent(args))


if __name__ == "__main__":
    sys.exit(main())
