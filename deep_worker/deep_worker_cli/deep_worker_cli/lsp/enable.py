"""Enabling/disabling the LSP option — writes the ``mcp.json`` layer.

See ``lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md`` §4.2.

Guarantees:
* merge, not overwrite: other MCP servers are not lost;
* idempotent: a repeated ``enable`` does not touch an existing entry
  (the user may have tweaked the args manually) — to overwrite, use
  ``force=True``;
* permissions: file ``0o600``, directory ``0o700`` — the config may contain
  ``${env:...}`` references to secrets. No 777/666 (project policy);
* atomic write (tmp file in the same directory + ``os.replace``), leaves no
  garbage behind.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from deep_worker_cli.lsp.serena_config import (
    SERENA_SERVER_NAME,
    build_serena_mcp_entry,
    resolve_workspace_root,
)

__all__ = [
    "SERENA_SERVER_NAME",
    "disable_lsp_at_path",
    "enable_lsp_at_path",
    "is_lsp_enabled_at_path",
    "resolve_workspace_root",
]

_CONFIG_MODE = 0o600
_DIR_MODE = 0o700


def _read_config(path: Path) -> dict[str, Any]:
    """Read the config, normalizing to the ``{"mcpServers": {...}}`` form.

    A flat cursor-style root (which the loader understands) is also
    supported and wrapped, so that entries are not lost during the merge.
    """
    if not path.exists():
        return {"mcpServers": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON ({e})") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    if "mcpServers" in raw:
        servers = raw.get("mcpServers") or {}
        if not isinstance(servers, dict):
            raise ValueError(f"{path}: 'mcpServers' must be an object")
        return {**raw, "mcpServers": dict(servers)}
    # Flat root → wrap it, losing nothing.
    return {"mcpServers": dict(raw)}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON with mode 0600 in a 0700 directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    # mkdir(mode=...) does not apply to an already-existing directory and
    # is cut by umask — set it explicitly.
    path.parent.chmod(_DIR_MODE)

    tmp = path.with_name(path.name + ".tmp")
    try:
        # Create with the required mode right away, so secrets never sit on
        # disk with broader permissions (not even for a moment).
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _CONFIG_MODE)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
        finally:
            # os.fdopen took ownership of the fd; no separate close needed.
            pass
        tmp.chmod(_CONFIG_MODE)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def is_lsp_enabled_at_path(path: Path) -> bool:
    """Whether the config contains a Serena entry (and it is not marked ``disabled``)."""
    try:
        entry = _read_config(path)["mcpServers"].get(SERENA_SERVER_NAME)
    except (OSError, ValueError):
        return False
    if not isinstance(entry, dict):
        return False
    return entry.get("disabled") is not True


def enable_lsp_at_path(
    path: Path,
    *,
    workspace_root: Path | str,
    force: bool = False,
    auto_approve: bool = True,
) -> bool:
    """Enable LSP by appending the Serena entry to ``path``.

    Returns:
        ``True`` if the file was modified, ``False`` if the entry already
        existed (idempotent no-op).
    """
    config = _read_config(path)
    servers = config["mcpServers"]

    if SERENA_SERVER_NAME in servers and not force:
        return False

    servers[SERENA_SERVER_NAME] = build_serena_mcp_entry(
        workspace_root=resolve_workspace_root(workspace_root),
        auto_approve=auto_approve,
    )
    _atomic_write(path, config)
    return True


def disable_lsp_at_path(path: Path) -> bool:
    """Remove the Serena entry from the config, keeping the other servers.

    Returns:
        ``True`` if the entry was removed, ``False`` if it was not there.
    """
    if not path.exists():
        return False
    config = _read_config(path)
    if SERENA_SERVER_NAME not in config["mcpServers"]:
        return False
    del config["mcpServers"][SERENA_SERVER_NAME]
    _atomic_write(path, config)
    return True
