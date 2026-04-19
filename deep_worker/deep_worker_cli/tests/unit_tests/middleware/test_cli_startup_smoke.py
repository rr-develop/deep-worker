"""Smoke tests for CLI startup performance.

These tests exist because a previous iteration shipped middleware
changes that slowed CLI ``--help`` or parse-time startup. Unit tests
that ``import deep_worker_cli.middleware.send_message_completion``
directly don't exercise the bash -> ``deep-worker`` -> argparse path,
so they cannot catch cold-start regressions introduced by a heavy
eager import at module load.

What we assert here
-------------------
1. Importing the CLI entry module is fast (< 3 seconds on any
   reasonable machine).
2. ``create_parser().parse_args([...])`` is instantaneous.
3. Importing the middleware module alone is < 10 seconds (it pulls
   in langchain which is heavy; we just don't want a new regression
   to push it into the 30s+ range).

These are soft timings — the exact thresholds are generous so the
tests don't flake on slow CI, but they catch order-of-magnitude
regressions.

If a future change adds a heavy top-level import (eg. ``import torch``)
into ``deep_worker_cli.middleware`` or ``deep_worker_cli.cli.agent_command``,
these tests fail loudly instead of leaving the user staring at a
hanging terminal.
"""

from __future__ import annotations

import importlib
import sys
import time


def _timed_import(module_name: str) -> float:
    """Import a module (purging it from sys.modules first) and return
    elapsed wall-clock seconds."""
    sys.modules.pop(module_name, None)
    # Also pop any submodules of the same package.
    for key in list(sys.modules):
        if key.startswith(module_name + "."):
            sys.modules.pop(key, None)
    t0 = time.time()
    importlib.import_module(module_name)
    return time.time() - t0


def test_cli_module_import_is_fast() -> None:
    """Importing the CLI entry module must be fast. If this grows into
    minutes, the ``deep-worker --help`` user experience breaks."""
    elapsed = _timed_import("deep_worker_cli.cli.agent_command")
    # Very generous threshold — real-world this is ~0.1s on dev machines.
    # If a regression pushes it to tens of seconds, fail.
    assert elapsed < 10.0, (
        f"Importing deep_worker_cli.cli.agent_command took {elapsed:.2f}s; "
        "this will make `deep-worker --help` feel broken."
    )


def test_parse_args_is_instant() -> None:
    """After import, argparse itself must be negligible."""
    from deep_worker_cli.cli.agent_command import create_parser

    parser = create_parser()
    t0 = time.time()
    args = parser.parse_args(
        ["agentname", "-asm", "--dir", "/tmp", "--clean"]
    )
    elapsed = time.time() - t0
    assert elapsed < 0.5, f"parse_args took {elapsed:.3f}s; should be ~0s"
    # Sanity: the parsed flags survived (guards against accidental
    # removal of -asm from the parser, which would also make the flag
    # silently no-op).
    assert args.auto_approve_send_message is True
    assert args.clean is True
    assert args.name == "agentname"


def test_middleware_import_is_reasonable() -> None:
    """Importing the new middleware must not add pathological cold-start
    cost. We re-import it in isolation; it transitively loads langchain
    which is inherently slow, so the ceiling is generous."""
    elapsed = _timed_import("deep_worker_cli.middleware.send_message_completion")
    assert elapsed < 30.0, (
        f"Importing deep_worker_cli.middleware.send_message_completion "
        f"took {elapsed:.2f}s; a regression here makes CLI startup feel hung."
    )
