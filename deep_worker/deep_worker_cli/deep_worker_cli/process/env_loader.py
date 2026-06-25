"""Auto-load ``~/.deepagents/env.d/*.env`` drop-ins at agent start.

Motivation (see DESIGN_EXTERNAL_MCP_TOOLS.md §7):

External MCP servers reference secrets via ``${env:VAR}`` in ``mcp.json``.
For consistency with how OS components ship secrets (one file per concern,
mode ``0o600``, symlink-friendly — same shape as systemd's
``EnvironmentFile=``) we look for *drop-in* env files under
``~/.deepagents/env.d/``. Any ``*.env`` there is loaded into ``os.environ``
with two guardrails:

1. ``override=False`` — variables already set in the environment (shell
   inheritance or the main ``~/.deepagents/.env`` loaded a moment earlier)
   are kept. This makes the main ``.env`` the single source of truth when
   it disagrees with a drop-in.
2. Files (and symlink targets) with permissions wider than ``0o600`` are
   SKIPPED with a warning. ``mcp.json`` already follows this policy; we
   mirror it for env drop-ins.

The directory is *optional*: if it does not exist the function is a
no-op, so agents without drop-ins start exactly as before.

Public API:

* :func:`load_env_d(env_d: Path) -> dict[str, str]` — returns a snapshot
  of the keys that were actually loaded (post-``override=False`` filter),
  primarily for tests / diagnostics.
"""
from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import dotenv

_logger = logging.getLogger(__name__)

# Permissions stricter than this would still pass, but anything looser
# (group- or world-readable) is rejected. Matches mcp.json policy.
_MAX_PERMS = 0o600


def _is_secure_mode(mode: int) -> bool:
    """``True`` iff the file is at most owner-rw (no group/world bits)."""
    return (mode & 0o077) == 0


def load_env_d(env_d: Path) -> dict[str, str]:
    """Load ``*.env`` drop-ins from ``env_d`` into ``os.environ``.

    Args:
        env_d: Directory to scan (typically ``~/.deepagents/env.d``).

    Returns:
        A mapping of ``{var: value}`` for keys actually applied to
        ``os.environ`` by this call. Variables that were already set
        (and therefore preserved) are NOT included.

    Notes:
        * Files are processed in ``sorted()`` order so the precedence is
          deterministic ("first filename wins" for a duplicate key).
        * Non-``.env`` files are ignored.
        * Symlinks are followed; permission checks apply to the resolved
          target (``Path.stat`` follows symlinks by default).
        * Errors reading an individual file are logged and the file is
          skipped — they never crash agent startup.
    """
    applied: dict[str, str] = {}

    if not env_d.is_dir():
        return applied

    for candidate in sorted(env_d.iterdir()):
        if candidate.suffix != ".env":
            continue
        try:
            st = candidate.stat()
        except OSError as exc:
            _logger.warning("env.d: cannot stat %s: %s", candidate, exc)
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        mode = stat.S_IMODE(st.st_mode)
        if not _is_secure_mode(mode):
            _logger.warning(
                "env.d: skipping %s — permissions %#o are looser than 0600; "
                "tighten via `chmod 0600 <target>`",
                candidate.name, mode,
            )
            continue

        # Parse without applying so we can honour override=False ourselves
        # (python-dotenv's load_dotenv(override=False) checks os.environ
        # too, but we also want to expose the applied set for tests).
        try:
            parsed = dotenv.dotenv_values(candidate)
        except Exception as exc:  # noqa: BLE001 — never block startup
            _logger.warning("env.d: failed to parse %s: %s", candidate.name, exc)
            continue

        for key, value in parsed.items():
            if value is None:
                continue
            # Preserve precedence: anything already in os.environ wins.
            # That includes earlier drop-ins (sorted order) and the main
            # ~/.deepagents/.env loaded by the caller before us.
            if key in os.environ:
                continue
            os.environ[key] = value
            applied[key] = value

    return applied
