"""``python -m deep_worker_cli.semantic_memory.service`` — daemon target.

The supervisor spawns the service through ``-m`` rather than the
``deep-worker`` console script, so the detached daemon does not depend on
whatever ``PATH`` the spawning agent happened to have.

Importing this module is what pulls in ``grpc_server`` and, on the first
request, the heavy stack. Nothing above it in the tree imports this
module — that asymmetry is the whole point of the service, and
``test_package_boundary`` pins it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def build_serve_argv(socket_path: Path) -> list[str]:
    """The argv the supervisor should spawn for *socket_path*.

    Lives here, next to the parser that consumes it, so the flag names
    cannot drift apart. The supervisor takes this as an injected callable
    and never imports this module itself.
    """
    return [
        sys.executable,
        "-m",
        "deep_worker_cli.semantic_memory.service",
        "serve",
        "--socket",
        str(socket_path),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m deep_worker_cli.semantic_memory.service"
    )
    sub = parser.add_subparsers(dest="command")
    serve_parser = sub.add_parser("serve", help="run the service in the foreground")
    serve_parser.add_argument("--socket", required=True, type=Path)
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    if args.command != "serve":
        parser.print_usage()
        return 2

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Imported here, not at module scope: ``build_serve_argv`` must stay
    # callable without loading grpc.
    from deep_worker_cli.semantic_memory.service.grpc_server import serve

    try:
        asyncio.run(serve(args.socket))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
