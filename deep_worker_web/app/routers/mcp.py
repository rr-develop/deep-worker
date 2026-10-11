"""API router for the per-agent MCP tab.

Endpoints (deep_worker_web/docs/DESIGN_MCP_TAB.md §5):

    GET    /api/agents/{name}/mcp                 effective raw config
    PUT    /api/agents/{name}/mcp/{server}        add/update (agent layer)
    DELETE /api/agents/{name}/mcp/{server}        remove (agent layer)
    POST   /api/agents/{name}/mcp/{server}/enable toggle `disabled`
    POST   /api/agents/{name}/mcp/test            probe a spec (explicit)
    GET    /api/agents/{name}/mcp/live            what is attached NOW

    GET    /api/mcp/catalog                       host catalog
    POST   /api/mcp/catalog/scan                  on-demand scan + merge
    PUT    /api/mcp/catalog/{id}                  manual entry (pinned)
    DELETE /api/mcp/catalog/{id}                  remove entry

Two rules run through all of it:

* **Nothing is refused because a server looks expensive.** Weight is an
  advisory label; the user picks the tools they consider necessary.
* **Nothing is probed implicitly.** Connecting to an MCP server happens
  only in ``/mcp/test``, on an explicit click.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.mcp_schemas import (
    CatalogEntryModel,
    CatalogResponse,
    CatalogWrite,
    LayerErrorInfo,
    McpConfigResponse,
    McpEnableRequest,
    McpLiveResponse,
    McpServerEntry,
    McpServerWrite,
    McpTestRequest,
    McpTestResponse,
    McpWriteResponse,
    ScanRequest,
    ScanResponse,
)
from app.services.agent_service import AgentService
from app.services.mcp_service import McpService
from deep_worker_cli.external_mcp import catalog as catalog_mod

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])


def get_agent_service(request: Request) -> AgentService:
    if not hasattr(request.app.state, "agent_service"):
        raise HTTPException(status_code=500, detail="AgentService not initialized")
    return request.app.state.agent_service


def get_mcp_service() -> McpService:
    # Home is resolved per call (not captured at import) so tests and
    # alternative hosts can redirect it.
    return McpService()


def _validated_agent(agent_name: str) -> str:
    from deep_worker_cli.storage_paths import validate_agent_name

    try:
        validate_agent_name(agent_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return agent_name


def _to_entry_model(entry: catalog_mod.CatalogEntry, label: str) -> CatalogEntryModel:
    return CatalogEntryModel(
        id=entry.id,
        label=label,
        origin=entry.origin,
        source=entry.source,
        spec=entry.spec,
        weight=entry.weight,
        weight_reason=entry.weight_reason,
        tools_hint=entry.tools_hint,
        pinned=entry.pinned,
        stale=entry.stale,
        discovered_at=entry.discovered_at,
    )


# ── per-agent config ─────────────────────────────────────────────────


@router.get("/api/agents/{agent_name}/mcp", response_model=McpConfigResponse)
async def get_agent_mcp(
    agent_name: str,
    service: McpService = Depends(get_mcp_service),
) -> McpConfigResponse:
    """Effective merged config for one agent.

    Deliberately tolerant (§5.1): ``${env:VAR}`` is NOT resolved and a
    malformed layer is reported rather than raised, because this is the
    screen a user opens to fix exactly those problems.
    """
    _validated_agent(agent_name)
    merged = await service.get_effective_config(agent_name)

    servers: list[McpServerEntry] = []
    for entry in merged.entries.values():
        weight, reason = catalog_mod.classify_weight(entry.spec)
        servers.append(
            McpServerEntry(
                name=entry.name,
                spec=entry.spec,
                layer=entry.layer,
                disabled=entry.disabled,
                env_status=entry.env_status,
                unresolved_vars=list(entry.unresolved_vars),
                overrides=list(entry.overrides),
                editable=entry.layer == "agent",
                weight=weight,
                weight_reason=reason,
            )
        )
    servers.sort(key=lambda s: s.name)

    return McpConfigResponse(
        agent_name=agent_name,
        servers=servers,
        layer_errors=[
            LayerErrorInfo(layer=e.layer, path=e.path, message=e.message)
            for e in merged.layer_errors
        ],
    )


@router.put("/api/agents/{agent_name}/mcp/{server}", response_model=McpWriteResponse)
async def put_agent_mcp(
    agent_name: str,
    server: str,
    body: McpServerWrite,
    service: McpService = Depends(get_mcp_service),
) -> McpWriteResponse:
    """Add or update an entry in the agent layer.

    Only a structurally unusable spec is rejected. Weight and unresolved
    env come back as warnings — the write still happens.
    """
    _validated_agent(agent_name)
    if not server or not server.strip():
        raise HTTPException(status_code=400, detail="server name must be non-empty")

    try:
        warnings = await service.upsert(agent_name, server, body.spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"write failed: {exc}") from None

    return McpWriteResponse(server=server, warnings=warnings)


@router.delete("/api/agents/{agent_name}/mcp/{server}", response_model=McpWriteResponse)
async def delete_agent_mcp(
    agent_name: str,
    server: str,
    service: McpService = Depends(get_mcp_service),
) -> McpWriteResponse:
    _validated_agent(agent_name)
    removed = await service.remove(agent_name, server)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"server {server!r} is not present in the agent layer",
        )
    return McpWriteResponse(server=server)


@router.post(
    "/api/agents/{agent_name}/mcp/{server}/enable", response_model=McpWriteResponse
)
async def enable_agent_mcp(
    agent_name: str,
    server: str,
    body: McpEnableRequest,
    service: McpService = Depends(get_mcp_service),
) -> McpWriteResponse:
    _validated_agent(agent_name)
    ok = await service.set_enabled(agent_name, server, body.enabled)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"server {server!r} is not present in the agent layer",
        )
    return McpWriteResponse(server=server)


@router.post("/api/agents/{agent_name}/mcp/test", response_model=McpTestResponse)
async def test_agent_mcp(
    agent_name: str,
    body: McpTestRequest,
    service: McpService = Depends(get_mcp_service),
) -> McpTestResponse:
    """Connect to a server, list its tools, disconnect.

    Runs in the WEB process, bounded, so a hanging or hostile server can
    never park the agent. Explicit user action only — never automatic.
    Returns names without descriptions, to keep the rendered DOM small.
    """
    _validated_agent(agent_name)

    valid, error, _ = service.validate_spec(body.spec)
    if not valid:
        raise HTTPException(status_code=400, detail=error)

    from app.services.mcp_probe import probe_server

    try:
        names = await probe_server(body.spec, timeout=body.timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        return McpTestResponse(success=False, error=str(exc))

    return McpTestResponse(success=True, tool_count=len(names), tool_names=names)


@router.get("/api/agents/{agent_name}/mcp/live", response_model=McpLiveResponse)
async def live_agent_mcp(
    agent_name: str,
    service: McpService = Depends(get_mcp_service),
    agents: AgentService = Depends(get_agent_service),
) -> McpLiveResponse:
    """Tools actually attached to the running agent (configured-vs-live)."""
    _validated_agent(agent_name)
    result = await service.get_live_tools(agents, agent_name)
    return McpLiveResponse(
        success=bool(result.get("success")),
        connected=bool(result.get("connected")),
        tool_names=list(result.get("tool_names", [])),
        by_server=dict(result.get("by_server", {})),
        error=str(result.get("error", "")),
    )


# ── host catalog ─────────────────────────────────────────────────────


@router.get("/api/mcp/catalog", response_model=CatalogResponse)
async def get_catalog(
    service: McpService = Depends(get_mcp_service),
) -> CatalogResponse:
    """Read the catalog. Never scans, never probes.

    An absent or malformed catalog reads as empty — a fresh host is a
    normal state, not an error.
    """
    entries = await service.load_catalog()
    return CatalogResponse(
        servers=[
            _to_entry_model(e, catalog_mod.display_label(e, entries)) for e in entries
        ]
    )


@router.post("/api/mcp/catalog/scan", response_model=ScanResponse)
async def scan_catalog(
    body: ScanRequest | None = None,
    service: McpService = Depends(get_mcp_service),
) -> ScanResponse:
    """Scan known config locations and merge into the catalog.

    Reads files only — no candidate server is ever executed. Pinned
    entries survive untouched; entries that disappeared are marked stale
    rather than deleted.
    """
    roots = [Path(r) for r in (body.search_roots if body else [])]
    try:
        entries, scanned = await service.scan(roots)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"scan failed: {exc}") from None

    return ScanResponse(
        scanned_files=scanned,
        found=len(entries),
        servers=[
            _to_entry_model(e, catalog_mod.display_label(e, entries)) for e in entries
        ],
    )


@router.put("/api/mcp/catalog/{entry_id}", response_model=McpWriteResponse)
async def put_catalog_entry(
    entry_id: str,
    body: CatalogWrite,
    service: McpService = Depends(get_mcp_service),
) -> McpWriteResponse:
    """Add a manual catalog entry. Always ``pinned`` — a later scan must
    never overwrite or remove what the user typed in by hand."""
    if not entry_id or not entry_id.strip():
        raise HTTPException(status_code=400, detail="entry id must be non-empty")
    try:
        warnings = await service.catalog_upsert(entry_id, body.spec, body.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return McpWriteResponse(server=entry_id, warnings=warnings, restart_required=False)


@router.delete("/api/mcp/catalog/{entry_id}", response_model=McpWriteResponse)
async def delete_catalog_entry(
    entry_id: str,
    service: McpService = Depends(get_mcp_service),
) -> McpWriteResponse:
    removed = await service.catalog_remove(entry_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no catalog entry {entry_id!r}")
    return McpWriteResponse(server=entry_id, restart_required=False)
