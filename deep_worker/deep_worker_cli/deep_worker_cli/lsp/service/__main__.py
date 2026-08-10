"""``python -m deep_worker_cli.lsp.service`` — daemon self-exec target.

The supervisor double-forks into this module (see
``service_cli.build_serve_argv``). Going through ``-m`` rather than the
``deep-worker`` console script keeps the detached daemon independent of
whatever ``PATH`` the spawning agent happened to have.

Accepts the same subcommand names as ``deep-worker lsp`` so the two
paths stay interchangeable for debugging.
"""
from __future__ import annotations

import argparse
import sys

from deep_worker_cli.lsp.service.service_cli import (
    dispatch_service_command,
    register_service_subcommands,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deep_worker_cli.lsp.service")
    sub = parser.add_subparsers(dest="lsp_command")
    register_service_subcommands(sub)
    args = parser.parse_args(argv)
    code = dispatch_service_command(args)
    if code is None:
        parser.print_usage()
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
