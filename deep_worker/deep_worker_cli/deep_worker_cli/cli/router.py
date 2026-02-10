"""CLI command router for deep_worker.

Routes between agent process launcher and client connector based on context.
"""

import sys


def deep_worker_main() -> int:
    """Entry point for 'deep-worker' command (agent process).

    Returns:
        Exit code
    """
    from deep_worker_cli.cli.agent_command import main as agent_main

    return agent_main()


def deep_worker_cli_main() -> int:
    """Entry point for 'deep-worker-cli' command (client connector).

    Returns:
        Exit code
    """
    from deep_worker_cli.cli.client_command import main as client_main

    return client_main()


if __name__ == "__main__":
    # Determine which command based on sys.argv[0]
    if "deep-worker-cli" in sys.argv[0]:
        sys.exit(deep_worker_cli_main())
    else:
        sys.exit(deep_worker_main())
