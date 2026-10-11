"""Auto-load the agent's environment sources (``.env`` + ``env.d`` drop-ins).

Motivation (see DESIGN_EXTERNAL_MCP_TOOLS.md §7):

External MCP servers reference secrets via ``${env:VAR}`` in ``mcp.json``.
For consistency with how OS components ship secrets (one file per concern,
mode ``0o600``, symlink-friendly — same shape as systemd's
``EnvironmentFile=``) we support *drop-in* env files in an ``env.d``
directory alongside a conventional ``.env``.

Source chain (owner directive 2026-08-15: deep-worker keeps its settings
under ``~/.deep_worker``, not under the upstream ancestor's
``~/.deep_worker``). :func:`load_env_sources` applies, in order:

1. ``~/.deep_worker/.env``       — PRIMARY
2. ``~/.deep_worker/env.d/*.env`` — PRIMARY drop-ins

Existing installs keep working with no operator action; when a variable is
defined in both trees the ``~/.deep_worker`` value WINS, because the whole
chain obeys one rule:

1. ``override=False`` — "first writer wins". Variables already set in the
   environment (shell inheritance, or an earlier source in the chain) are
   kept. Loading the new tree first is therefore what makes it
   authoritative.
2. Files (and symlink targets) with permissions wider than ``0o600`` are
   SKIPPED with a warning. ``mcp.json`` already follows this policy; we
   mirror it for env drop-ins, so a permissions slip fails closed.

Every source is *optional*: missing files/directories are a no-op, so
agents without drop-ins start exactly as before.

Public API:

* :func:`load_env_sources(home=None) -> dict[str, str]` — apply the whole
  chain; returns a snapshot of the keys actually loaded.
* :func:`env_source_paths(home=None) -> list[Path]` — the chain itself, in
  precedence order, for diagnostics.
* :func:`load_env_d(env_d: Path) -> dict[str, str]` — apply a single
  ``env.d`` directory (used by the chain, and directly by tests).
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
        env_d: Directory to scan (typically ``~/.deep_worker/env.d``).

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
            # ~/.deep_worker/.env loaded by the caller before us.
            if key in os.environ:
                continue
            os.environ[key] = value
            applied[key] = value

    return applied


def _load_dot_env(path: Path) -> dict[str, str]:
    """Apply a single ``.env`` file with the same first-writer-wins rule.

    Mirrors :func:`load_env_d`'s precedence (never override an already-set
    variable) so the two source kinds compose into one ordered chain.
    """
    applied: dict[str, str] = {}
    if not path.is_file():
        return applied
    try:
        parsed = dotenv.dotenv_values(path)
    except Exception as exc:  # noqa: BLE001 — never block startup
        _logger.warning("env: failed to parse %s: %s", path, exc)
        return applied
    for key, value in parsed.items():
        if value is None or key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def env_source_paths(home: Path | None = None) -> tuple[Path, ...]:
    """Ordered env sources, highest precedence FIRST.

    deep-worker keeps its settings under ``~/.deep_worker``. The ancestor's
    ``~/.deepagents`` is NOT consulted: POLICIES §16 bans it for reads as
    well as writes, and loading secrets from a foreign tree is exactly the
    silent-authority failure that ban exists to prevent.

    Order within a tree is ``.env`` then ``env.d`` — the main file stays
    authoritative and drop-ins only fill blanks (DESIGN_EXTERNAL_MCP_TOOLS
    §7). Combined with the first-writer-wins rule used by every loader
    below, position in this tuple IS the precedence.
    """
    base = home or Path.home()
    return (
        base / ".deep_worker" / ".env",
        base / ".deep_worker" / "env.d",
    )


def load_env_sources(home: Path | None = None) -> dict[str, str]:
    """Load every env source in precedence order into ``os.environ``.

    Returns the mapping actually applied by this call (variables already
    present in the environment are preserved and therefore absent).
    Directories are handled by :func:`load_env_d` (which enforces the
    ``0600`` permission gate); plain files by :func:`_load_dot_env`.
    """
    applied: dict[str, str] = {}
    for source in env_source_paths(home):
        if source.is_dir():
            applied.update(load_env_d(source))
        else:
            applied.update(_load_dot_env(source))
    return applied
