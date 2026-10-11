"""Shared read/write layer for ``mcp.json`` files.

This module is the SINGLE writer of MCP config for both the CLI
(``deep-worker mcp add/remove``) and ``deep_worker_web``. Two independent
implementations writing the same file is a guaranteed source of silent
corruption, so scope resolution, the ``{"mcpServers": {...}}`` wrapper and
the atomic 0600 write live here once.

It complements — and does not replace — ``external_mcp.config``:

* ``config.load_mcp_config`` is the STRICT, agent-facing loader. It resolves
  ``${env:VAR}`` eagerly and raises ``UnresolvedEnvError`` for a missing
  variable, because starting a server with a blank token is worse than
  failing loudly.
* the raw readers here are the TOLERANT, UI-facing ones. A read endpoint
  must never 500 because one entry in one layer references an unset
  variable — the user would not even be able to open the UI to fix that
  entry. Values come back verbatim (``${env:VAR}`` included, which also
  keeps secrets out of the response) with an advisory ``env_status``.

See ``deep_worker_web/docs/DESIGN_MCP_TAB.md`` §5.1.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from deep_worker_cli.external_mcp.config import _ENV_RE
from deep_worker_cli.storage_paths import (
    agent_config_dir,
    config_home,
    ensure_private_dir,
    project_config_dir,
)

EnvStatus = Literal["ok", "unresolved", "unknown"]

#: Config-file name used by every layer.
MCP_FILENAME = "mcp.json"


class UnknownScopeError(ValueError):
    """Raised for an unparseable ``--scope`` / scope string."""


@dataclass(frozen=True)
class RawEntry:
    """One server entry as it appears on disk, WITHOUT env resolution."""

    name: str
    spec: dict[str, Any]
    layer: str
    disabled: bool = False
    env_status: EnvStatus = "ok"
    unresolved_vars: tuple[str, ...] = ()
    #: Lower-priority layers whose same-named entry this one replaces.
    overrides: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LayerError:
    layer: str
    path: str
    message: str


@dataclass(frozen=True)
class LayerRead:
    """Result of reading ONE layer. A broken layer is reported, not raised."""

    entries: dict[str, RawEntry]
    error: LayerError | None = None


@dataclass(frozen=True)
class MergedRead:
    """Result of merging several layers, lowest priority first."""

    entries: dict[str, RawEntry]
    layer_errors: list[LayerError] = field(default_factory=list)


# ── scope → path ─────────────────────────────────────────────────────


def scope_to_path(
    scope: str,
    *,
    home: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve a scope string to its ``mcp.json`` path.

    ``home`` and ``project_root`` are injected rather than discovered so
    this stays pure and testable, and so no host-specific path is ever
    baked in (DESIGN_MCP_TAB.md §3).
    """
    if scope == "user":
        return config_home(home) / MCP_FILENAME

    if scope == "project":
        if project_root is None:
            raise UnknownScopeError("scope 'project' requires a project_root")
        return project_config_dir(project_root) / MCP_FILENAME

    if scope.startswith("agent:"):
        name = scope.split(":", 1)[1].strip()
        if not name:
            raise UnknownScopeError("scope 'agent:<name>' requires a non-empty name")
        # An agent name is ONE path segment. Without this check
        # ``agent:../../x`` would write outside the home tree.
        if name != Path(name).name or name in {".", ".."}:
            raise UnknownScopeError(f"invalid agent name in scope: {name!r}")
        return agent_config_dir(name, home=home) / MCP_FILENAME

    raise UnknownScopeError(
        f"unknown scope {scope!r}; expected user | project | agent:<name>"
    )


# ── read ─────────────────────────────────────────────────────────────


def _unwrap(data: Any) -> dict[str, Any]:
    """Accept both the wrapped and the flat (cursor-style) document."""
    if not isinstance(data, dict):
        raise ValueError("mcp config root must be a JSON object")
    inner = data.get("mcpServers")
    if isinstance(inner, dict):
        return inner
    if any(not isinstance(v, dict) for v in data.values()):
        raise ValueError(
            "mcp config: expected {'mcpServers': {...}} or a flat name→spec map"
        )
    return data


def _env_status(spec: Any) -> tuple[EnvStatus, tuple[str, ...]]:
    """Advisory check of ``${env:VAR}`` references found anywhere in ``spec``.

    Advisory by nature: variables resolve in the AGENT's environment, not
    the web server's, so an ``unresolved`` verdict is a warning to show —
    never an error and never a reason to hide the entry.
    """
    missing: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for match in _ENV_RE.finditer(node):
                var, fallback = match.group(1), match.group(2)
                if var not in os.environ and fallback is None:
                    missing.append(var)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    if missing:
        # dict.fromkeys preserves first-seen order while de-duplicating.
        return "unresolved", tuple(dict.fromkeys(missing))
    return "ok", ()


def read_layer(path: Path, layer: str = "") -> LayerRead:
    """Read ONE layer verbatim. Absent → empty; malformed → reported."""
    if not path.exists():
        return LayerRead(entries={})

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        servers = _unwrap(raw)
    except (ValueError, OSError) as exc:
        return LayerRead(
            entries={},
            error=LayerError(layer=layer, path=str(path), message=str(exc)),
        )

    entries: dict[str, RawEntry] = {}
    for name, spec in servers.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            continue
        status, missing = _env_status(spec)
        entries[name] = RawEntry(
            name=name,
            spec=dict(spec),
            layer=layer,
            disabled=spec.get("disabled") is True,
            env_status=status,
            unresolved_vars=missing,
        )
    return LayerRead(entries=entries)


def read_merged_raw(layers: Iterable[tuple[str, Path]]) -> MergedRead:
    """Merge layers given lowest-priority-first.

    A higher layer fully replaces a same-named entry from a lower one —
    the same rule ``load_mcp_config`` applies — and the replaced layers are
    recorded in ``overrides`` so the UI can explain why an entry "won".
    """
    merged: dict[str, RawEntry] = {}
    errors: list[LayerError] = []

    for layer_name, path in layers:
        read = read_layer(path, layer_name)
        if read.error is not None:
            errors.append(read.error)
            continue
        for name, entry in read.entries.items():
            previous = merged.get(name)
            overrides = (
                [*previous.overrides, previous.layer] if previous is not None else []
            )
            merged[name] = RawEntry(
                name=entry.name,
                spec=entry.spec,
                layer=entry.layer,
                disabled=entry.disabled,
                env_status=entry.env_status,
                unresolved_vars=entry.unresolved_vars,
                overrides=overrides,
            )

    return MergedRead(entries=merged, layer_errors=errors)


# ── write ────────────────────────────────────────────────────────────


def _read_for_write(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"mcpServers": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"mcpServers": _unwrap(data)}


def _atomic_write_private(path: Path, data: dict[str, Any]) -> None:
    """Write 0600 atomically, creating the parent 0700.

    The temp file is created with the final mode BEFORE any content is
    written, so the secrets a config may carry are never briefly visible
    through a world-readable file.
    """
    ensure_private_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def upsert_server(path: Path, name: str, spec: dict[str, Any]) -> None:
    """Add or replace one server entry, preserving every other entry."""
    if not name or not name.strip():
        raise ValueError("server name must be non-empty")
    data = _read_for_write(path)
    data["mcpServers"][name] = dict(spec)
    _atomic_write_private(path, data)


def remove_server(path: Path, name: str) -> bool:
    """Remove one entry. Returns False if the file or entry is absent."""
    if not path.exists():
        return False
    data = _read_for_write(path)
    if name not in data["mcpServers"]:
        return False
    del data["mcpServers"][name]
    _atomic_write_private(path, data)
    return True


def set_disabled(path: Path, name: str, disabled: bool) -> bool:
    """Flip ``disabled`` on one entry. Returns False if it is absent."""
    if not path.exists():
        return False
    data = _read_for_write(path)
    entry = data["mcpServers"].get(name)
    if not isinstance(entry, dict):
        return False
    if disabled:
        entry["disabled"] = True
    else:
        entry.pop("disabled", None)
    _atomic_write_private(path, data)
    return True
