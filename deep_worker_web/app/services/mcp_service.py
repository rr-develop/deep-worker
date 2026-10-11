"""Service layer for the MCP tab.

All the file work is delegated to ``deep_worker_cli.external_mcp`` so the
CLI and the web write the same files through the same code — two writers
of one config file is a guaranteed source of silent corruption.

Blocking file I/O is pushed off the event loop with ``asyncio.to_thread``
(the stdlib's own executor — this introduces no thread of ours, per the
project's no-new-threads rule), and every agent-side call goes through the
non-blocking RPyC discipline documented in ``agent_service.py``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from deep_worker_cli.external_mcp import catalog as catalog_mod
from deep_worker_cli.external_mcp.config_store import (
    MergedRead,
    UnknownScopeError,
    read_merged_raw,
    remove_server,
    scope_to_path,
    set_disabled,
    upsert_server,
)
from deep_worker_cli.storage_paths import validate_agent_name

logger = logging.getLogger(__name__)

#: Layers, lowest priority first. Mirrors ``load_mcp_config`` (config.py
#: §3.2) so what the UI shows matches what the agent will actually load.
_LAYER_ORDER = ("user", "project", "agent")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class McpService:
    """Stateless helper around the MCP config files and the host catalog."""

    def __init__(self, home: Path | None = None) -> None:
        self._home = home

    # ── paths ────────────────────────────────────────────────────────

    def _home_dir(self) -> Path | None:
        return self._home

    def agent_layer_path(self, agent_name: str) -> Path:
        validate_agent_name(agent_name)
        return scope_to_path(f"agent:{agent_name}", home=self._home_dir())

    def _layer_paths(
        self, agent_name: str, project_root: Path | None
    ) -> list[tuple[str, Path]]:
        layers: list[tuple[str, Path]] = [
            ("user", scope_to_path("user", home=self._home_dir()))
        ]
        if project_root is not None:
            try:
                layers.append(
                    (
                        "project",
                        scope_to_path(
                            "project", home=self._home_dir(), project_root=project_root
                        ),
                    )
                )
            except UnknownScopeError:  # pragma: no cover - defensive
                pass
        layers.append(("agent", self.agent_layer_path(agent_name)))
        return layers

    # ── read ─────────────────────────────────────────────────────────

    async def get_effective_config(
        self, agent_name: str, project_root: Path | None = None
    ) -> MergedRead:
        """Raw merged view. Never resolves env, never raises on a bad layer."""
        layers = self._layer_paths(agent_name, project_root)
        return await asyncio.to_thread(read_merged_raw, layers)

    # ── validation (advisory vs fatal) ───────────────────────────────

    @staticmethod
    def validate_spec(spec: dict[str, Any]) -> tuple[bool, str, list[str]]:
        """Return ``(is_valid, error, warnings)``.

        Only a genuinely unusable spec is rejected (neither ``command`` nor
        ``url``). Everything else — heavy binaries, unresolved env — is a
        WARNING: the user chooses their tools, and this API does not
        second-guess them (DESIGN_MCP_TAB.md §1.1).
        """
        warnings: list[str] = []

        if not isinstance(spec, dict):
            return False, "spec must be an object", warnings

        declared = spec.get("type")
        has_url = isinstance(spec.get("url"), str) and spec.get("url")
        has_command = isinstance(spec.get("command"), str) and spec.get("command")

        if declared in {"http", "sse", "streamable-http"} or (
            declared is None and has_url
        ):
            if not has_url:
                return False, "http transport requires a non-empty 'url'", warnings
        elif not has_command:
            return (
                False,
                "stdio transport requires a non-empty 'command' "
                "(or set type='http' with a 'url')",
                warnings,
            )

        args = spec.get("args")
        if args is not None and (
            not isinstance(args, list) or not all(isinstance(a, str) for a in args)
        ):
            return False, "'args' must be a list of strings", warnings

        for field in ("env", "headers"):
            value = spec.get(field)
            if value is not None and not isinstance(value, dict):
                return False, f"'{field}' must be an object", warnings

        # Advisory: unresolved ${env:VAR} would make the AGENT fail to
        # start, so it is worth shouting about — but the variable may well
        # exist in the agent's environment and not in the web's.
        from deep_worker_cli.external_mcp.config_store import _env_status

        status, missing = _env_status(spec)
        if status == "unresolved":
            warnings.append(
                "references environment variable(s) not set in this process: "
                + ", ".join(missing)
                + ". If they are also unset for the agent, it will fail to start."
            )

        weight, reason = catalog_mod.classify_weight(spec)
        if weight == "heavy":
            warnings.append(f"heavy server: {reason}. It may be slow to start.")

        if spec.get("auto_approve") is True:
            warnings.append(
                "auto_approve grants this server's tools an exemption from "
                "approval prompts, even without --skip-permissions."
            )

        return True, "", warnings

    # ── write (agent layer only) ─────────────────────────────────────

    async def upsert(
        self, agent_name: str, server: str, spec: dict[str, Any]
    ) -> list[str]:
        valid, error, warnings = self.validate_spec(spec)
        if not valid:
            raise ValueError(error)
        path = self.agent_layer_path(agent_name)
        await asyncio.to_thread(upsert_server, path, server, spec)
        return warnings

    async def remove(self, agent_name: str, server: str) -> bool:
        path = self.agent_layer_path(agent_name)
        return await asyncio.to_thread(remove_server, path, server)

    async def set_enabled(self, agent_name: str, server: str, enabled: bool) -> bool:
        path = self.agent_layer_path(agent_name)
        return await asyncio.to_thread(set_disabled, path, server, not enabled)

    # ── catalog ──────────────────────────────────────────────────────

    def catalog_file(self) -> Path:
        return catalog_mod.catalog_path(self._home_dir())

    async def load_catalog(self) -> list[catalog_mod.CatalogEntry]:
        return await asyncio.to_thread(catalog_mod.load_catalog, self.catalog_file())

    async def save_catalog(self, entries: Sequence[catalog_mod.CatalogEntry]) -> None:
        await asyncio.to_thread(
            catalog_mod.save_catalog, self.catalog_file(), list(entries), now=_now_iso()
        )

    async def catalog_upsert(
        self, entry_id: str, spec: dict[str, Any], label: str = ""
    ) -> list[str]:
        """Add/replace a MANUAL catalog entry (always ``pinned``)."""
        valid, error, warnings = self.validate_spec(spec)
        if not valid:
            raise ValueError(error)

        weight, reason = catalog_mod.classify_weight(spec)
        entries = [e for e in await self.load_catalog()
                   if not (e.id == entry_id and e.source == "manual")]
        entries.append(
            catalog_mod.CatalogEntry(
                id=entry_id,
                origin="manual",
                spec=dict(spec),
                label=label or entry_id,
                source="manual",
                weight=weight,
                weight_reason=reason,
                pinned=True,
                stale=False,
                discovered_at=_now_iso(),
            )
        )
        await self.save_catalog(entries)
        return warnings

    async def catalog_remove(self, entry_id: str) -> bool:
        entries = await self.load_catalog()
        kept = [e for e in entries if e.id != entry_id]
        if len(kept) == len(entries):
            return False
        await self.save_catalog(kept)
        return True

    async def scan(
        self, search_roots: Sequence[Path] = ()
    ) -> tuple[list[catalog_mod.CatalogEntry], int]:
        """Scan config files and merge into the catalog.

        Read-only: no candidate server is executed, so this is safe to run
        at any time. Returns ``(entries, scanned_file_count)``.
        """

        def _work() -> tuple[list[catalog_mod.CatalogEntry], int]:
            paths = catalog_mod.default_scan_paths(
                home=self._home_dir(), search_roots=list(search_roots)
            )
            found = catalog_mod.scan_config_files(paths, now=_now_iso())
            existing = catalog_mod.load_catalog(self.catalog_file())
            merged = catalog_mod.merge_scan(existing, found)
            catalog_mod.save_catalog(self.catalog_file(), merged, now=_now_iso())
            return merged, len(paths)

        return await asyncio.to_thread(_work)

    # ── live tools (via the agent, non-blocking) ─────────────────────

    async def get_live_tools(self, agent_service: Any, agent_name: str) -> dict[str, Any]:
        """Tools the agent actually holds right now.

        Uses the EXISTING ``exposed_get_tools`` RPyC method, which returns
        a JSON string (netref avoidance). Follows the mandatory web RPyC
        discipline: never block a handler on connect, never evaluate a
        blocking netref as a call argument.
        """
        if not await agent_service.ensure_connecting(agent_name):
            return {"success": False, "connected": False,
                    "error": "Agent not connected (connecting…)"}

        root = agent_service._roots.get(agent_name)
        if root is None:
            return {"success": False, "connected": False,
                    "error": "Agent not connected"}

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        async def _ok(result: Any) -> None:
            if not future.done():
                future.set_result(result)

        async def _err(exc: Any) -> None:
            if not future.done():
                future.set_exception(
                    exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
                )

        from rpyc_async.core.netref import async_call
        from rpyc_async.utils.helpers import fire_and_forget_call

        fire_and_forget_call(
            lambda: async_call(root, "get_tools"),
            timeout=10.0,
            success_callback=_ok,
            error_callback=_err,
            name=f"mcp_get_tools_{agent_name}",
        )

        try:
            raw = await asyncio.wait_for(future, timeout=12.0)
        except asyncio.TimeoutError:
            return {"success": False, "connected": True, "error": "timed out"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "connected": True, "error": str(exc)}

        return self._parse_live_tools(raw)

    @staticmethod
    def _parse_live_tools(raw: Any) -> dict[str, Any]:
        """Parse ``exposed_get_tools``' JSON string into name buckets.

        External MCP tools are mangled as ``server__tool``; splitting on
        the double underscore recovers which server contributed what.
        """
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError) as exc:
            return {"success": False, "connected": True,
                    "error": f"unparseable tool list: {exc}"}

        if not isinstance(payload, dict) or not payload.get("success", True):
            return {"success": False, "connected": True,
                    "error": str((payload or {}).get("error", "unknown error"))}

        tools = payload.get("tools", payload.get("tool_names", []))
        names: list[str] = []
        for tool in tools or []:
            if isinstance(tool, str):
                names.append(tool)
            elif isinstance(tool, dict) and isinstance(tool.get("name"), str):
                names.append(tool["name"])

        by_server: dict[str, list[str]] = {}
        for name in names:
            if "__" in name:
                server, _, bare = name.partition("__")
                if server and bare:
                    by_server.setdefault(server, []).append(bare)

        return {"success": True, "connected": True,
                "tool_names": names, "by_server": by_server}
