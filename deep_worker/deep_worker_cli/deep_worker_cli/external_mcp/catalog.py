"""Catalog of MCP servers available on this host.

The catalog is a convenience layer: a list of servers we know about, so
the UI can offer them for attaching without the user retyping a command
line. It is host state, so it lives in the product runtime home and NEVER
in the repository.

Design contract (deep_worker_web/docs/DESIGN_MCP_TAB.md §4):

* **Identity is the pair ``(origin, id)``.** The same server name
  legitimately appears in several config files (the same tool configured
  per project), so matching on ``id`` alone would let one scanned file
  silently overwrite another's entry, with the winner depending on
  filesystem iteration order.
* **A rescan never touches ``pinned`` entries.** Those are user-authored;
  losing them to a scan would defeat the whole point of being able to add
  entries by hand.
* **Entries not seen by a scan become ``stale``, they are not deleted.**
  A service being down or a directory temporarily unmounted must not drop
  the entry.
* **Nothing here executes a candidate server.** Discovery reads config
  files only; tool lists are obtained solely by an explicit user-triggered
  Test.
* **No host-specific literals.** Roots are injected by the caller, so the
  same code works on any host.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from deep_worker_cli.storage_paths import deep_worker_home, ensure_private_dir

Weight = Literal["light", "heavy", "service-backed", "unknown"]

CATALOG_VERSION = 1

#: Binaries whose cold start / RSS / index size make them noticeably
#: expensive. This drives an ADVISORY label only — never a refusal. The
#: user is free to attach whatever they consider necessary; see
#: DESIGN_MCP_TAB.md §1.1.
_HEAVY_BINARIES: frozenset[str] = frozenset(
    {
        "serena",
        "serena-agent",
        "kotlin-language-server",
        "kotlin-lsp",
        "pyright-langserver",
        "typescript-language-server",
        "tsserver",
        "rust-analyzer",
        "clangd",
        "gopls",
        "jdtls",
        "metals",
        "gradle",
        "mvn",
        "java",
    }
)

#: Name prefixes of our OWN in-agent proxy MCP server. It binds an
#: ephemeral port inside a specific agent process, so offering it for
#: attaching would be a footgun; it is filtered out of every scan.
_INTERNAL_PREFIXES: tuple[str, ...] = (
    "deepworker",
    "deep_worker_proxy",
    "deep_worker_acp_proxy",
)


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    origin: str
    spec: dict[str, Any]
    label: str = ""
    source: str = "scan:mcp_config"
    weight: Weight = "unknown"
    weight_reason: str = ""
    tools_hint: list[str] | None = None
    pinned: bool = False
    stale: bool = False
    discovered_at: str = ""

    @property
    def key(self) -> tuple[str, str]:
        """Identity: origin first, so ordering groups by file."""
        return (self.origin, self.id)


def catalog_path(home: Path | None = None) -> Path:
    """Location of the catalog file. Never a literal at the call site."""
    return deep_worker_home(home) / "mcp" / "catalog.json"


# ── weight classification (advisory only) ────────────────────────────


def classify_weight(spec: dict[str, Any]) -> tuple[Weight, str]:
    """Best-effort weight hint for a server spec.

    Returns ``(weight, reason)``. Unknown stays ``"unknown"`` rather than
    guessing "light": an advisory label that over-claims is worse than one
    that admits ignorance.
    """
    declared = spec.get("type")
    if declared in {"http", "sse", "streamable-http"} or (
        declared is None and spec.get("url")
    ):
        return "service-backed", "runs as its own service (HTTP transport)"

    command = spec.get("command")
    if isinstance(command, str) and command:
        binary = Path(command).name
        if binary in _HEAVY_BINARIES:
            return "heavy", f"{binary} is slow to start and holds a large index/runtime"
        # A launcher (npx/uvx/python -m ...) may still start something
        # heavy; look through the arguments before giving up.
        args = spec.get("args")
        if isinstance(args, list):
            for arg in args:
                if isinstance(arg, str) and Path(arg).name in _HEAVY_BINARIES:
                    return "heavy", f"launches {Path(arg).name}"
    return "unknown", ""


def _is_internal(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _INTERNAL_PREFIXES)


# ── display label ────────────────────────────────────────────────────


def display_label(entry: CatalogEntry, all_entries: Sequence[CatalogEntry]) -> str:
    """Human label, disambiguated only when ``id`` is not unique.

    ``id`` alone is not unique by design (§4.2), so the UI needs a stable
    way to tell two same-named entries apart without showing a full path
    on every row.
    """
    same_id = [e for e in all_entries if e.id == entry.id]
    if len(same_id) <= 1:
        return entry.id

    origin = Path(entry.origin)
    # Walk up past the config file and its ``.deep_worker`` wrapper to the
    # directory a human would recognise as "the project".
    parts = [p for p in origin.parts if p not in (os.sep, "/")]
    meaningful = [p for p in parts if p not in {".deep_worker", "mcp.json", ".mcp.json"}]
    hint = meaningful[-1] if meaningful else entry.origin
    return f"{entry.id} — {hint}"


# ── persistence ──────────────────────────────────────────────────────


def _entry_from_dict(raw: Any) -> CatalogEntry | None:
    if not isinstance(raw, dict):
        return None
    entry_id = raw.get("id")
    origin = raw.get("origin")
    spec = raw.get("spec")
    if not isinstance(entry_id, str) or not entry_id:
        return None
    if not isinstance(origin, str):
        return None
    if not isinstance(spec, dict):
        return None
    weight = raw.get("weight")
    tools_hint = raw.get("tools_hint")
    return CatalogEntry(
        id=entry_id,
        origin=origin,
        spec=dict(spec),
        label=raw.get("label") or entry_id,
        source=raw.get("source") or "scan:mcp_config",
        weight=weight if weight in {"light", "heavy", "service-backed", "unknown"}
        else "unknown",
        weight_reason=raw.get("weight_reason") or "",
        tools_hint=list(tools_hint) if isinstance(tools_hint, list) else None,
        pinned=bool(raw.get("pinned")),
        stale=bool(raw.get("stale")),
        discovered_at=raw.get("discovered_at") or "",
    )


def load_catalog(path: Path) -> list[CatalogEntry]:
    """Read the catalog. Absent or malformed → empty list, never raises.

    A broken catalog must not make the MCP tab unopenable — that is
    precisely the screen a user would go to in order to fix it.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    servers = data.get("servers")
    if not isinstance(servers, list):
        return []
    out: list[CatalogEntry] = []
    for raw in servers:
        entry = _entry_from_dict(raw)
        if entry is not None:
            out.append(entry)
    return out


def save_catalog(path: Path, entries: Iterable[CatalogEntry], *, now: str = "") -> None:
    """Write the catalog atomically as 0600 under a 0700 directory."""
    payload = {
        "version": CATALOG_VERSION,
        "updated_at": now,
        "servers": [
            {
                "id": e.id,
                "label": e.label or e.id,
                "source": e.source,
                "origin": e.origin,
                "spec": e.spec,
                "weight": e.weight,
                "weight_reason": e.weight_reason,
                "tools_hint": e.tools_hint,
                "pinned": e.pinned,
                "stale": e.stale,
                "discovered_at": e.discovered_at,
            }
            for e in entries
        ],
    }
    ensure_private_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ── scanning ─────────────────────────────────────────────────────────


def scan_config_files(
    paths: Iterable[Path], *, now: str = ""
) -> list[CatalogEntry]:
    """Read MCP client-config files and return catalog entries.

    Read-only and tolerant: a missing or malformed file is skipped rather
    than aborting the scan. **No candidate server is ever executed**, and
    ``${env:...}`` placeholders are kept verbatim so no secret is resolved
    into the catalog.
    """
    out: list[CatalogEntry] = []
    for path in paths:
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue

        inner = data.get("mcpServers")
        servers = inner if isinstance(inner, dict) else data
        if not isinstance(servers, dict):
            continue

        for name, spec in servers.items():
            if not isinstance(name, str) or not name or not isinstance(spec, dict):
                continue
            if _is_internal(name):
                continue
            weight, reason = classify_weight(spec)
            out.append(
                CatalogEntry(
                    id=name,
                    origin=str(path),
                    spec=dict(spec),
                    label=name,
                    source="scan:mcp_config",
                    weight=weight,
                    weight_reason=reason,
                    pinned=False,
                    stale=False,
                    discovered_at=now,
                )
            )
    return out


def default_scan_paths(
    *,
    home: Path | None = None,
    search_roots: Sequence[Path] = (),
) -> list[Path]:
    """Candidate config locations, derived at runtime — never hardcoded.

    ``search_roots`` is supplied by the caller (config or request), so the
    feature is not tied to any particular host layout.
    """
    # Product tree only — ``~/.deepagents`` is banned (POLICIES §16).
    product = deep_worker_home(home)
    paths: list[Path] = [product / "mcp.json"]

    # Per-agent config lives under ``agents/<name>/``.
    agent_parent = product / "agents"
    if agent_parent.is_dir():
        for child in sorted(agent_parent.iterdir()):
            if child.is_dir():
                candidate = child / "mcp.json"
                if candidate.exists():
                    paths.append(candidate)

    for root in search_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for rel in (Path(".deep_worker") / "mcp.json", Path(".mcp.json")):
                candidate = child / rel
                if candidate.exists():
                    paths.append(candidate)

    # De-duplicate while keeping order stable across runs.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


# ── merge ────────────────────────────────────────────────────────────


def merge_scan(
    existing: Sequence[CatalogEntry], scanned: Sequence[CatalogEntry]
) -> list[CatalogEntry]:
    """Merge scan results into the catalog by ``(origin, id)``.

    * pinned entries: untouched, and never marked stale;
    * same key: updated in place (keeping ``tools_hint``, which a scan
      cannot know);
    * new key: appended;
    * previously-known key absent from this scan: marked ``stale``.
    """
    scanned_by_key = {e.key: e for e in scanned}
    out: list[CatalogEntry] = []
    consumed: set[tuple[str, str]] = set()

    for entry in existing:
        found = scanned_by_key.get(entry.key)
        if entry.pinned:
            # User-authored: a scan may neither rewrite nor invalidate it.
            out.append(entry)
            if found is not None:
                consumed.add(entry.key)
            continue
        if found is None:
            out.append(replace(entry, stale=True))
            continue
        consumed.add(entry.key)
        out.append(
            replace(
                found,
                pinned=False,
                stale=False,
                # tools_hint is only ever filled by an explicit Test, so a
                # scan must not wipe a previously learned list.
                tools_hint=entry.tools_hint,
                discovered_at=entry.discovered_at or found.discovered_at,
            )
        )

    for entry in scanned:
        if entry.key not in consumed and not any(e.key == entry.key for e in out):
            out.append(entry)

    return out
