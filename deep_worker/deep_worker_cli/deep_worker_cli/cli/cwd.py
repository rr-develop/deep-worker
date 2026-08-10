"""Invocation directory of the CLI command — the single source of truth.

Any command that resolves paths RELATIVE to the user's directory
(``--scope project``, repository root discovery, etc.) must take its
starting point from here, NOT from ``Path.cwd()``.

Why ``Path.cwd()`` is not enough: the bash launcher ``deep-worker`` does
``cd "$SCRIPT_DIR"`` (the directory of the launcher's physical file)
before starting Python, so ``Path.cwd()`` returns the INSTALL directory.
The launcher stores the user's real directory in
``DEEP_WORKER_ORIGINAL_CWD`` — the same standard mechanism used by
``cli/agent_command.py`` when computing the agent's ``working_dir``.

The bug that made this module appear (reproduced on the real launcher
twice, for two different commands):

* ``deep-worker lsp enable`` from a dev worktree wrote the config into
  the install directory;
* ``deep-worker mcp list --scope project`` from an arbitrary directory
  showed — and ``mcp add`` would have written — another repository's
  config.

Without the launcher (direct module call, tests) the variable is absent
— fall back to ``Path.cwd()``.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["invocation_cwd"]


def invocation_cwd() -> Path:
    """The directory the user LAUNCHED the command from.

    Returns:
        ``DEEP_WORKER_ORIGINAL_CWD`` if set by the launcher, otherwise
        ``Path.cwd()``. The path is deliberately not resolved here: the
        caller decides whether it needs ``.resolve()``
        (``resolve_workspace_root`` does that).
    """
    original = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
    return Path(original) if original else Path.cwd()
