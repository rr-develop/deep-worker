"""Pydantic schemas for the MCP tab API.

See deep_worker_web/docs/DESIGN_MCP_TAB.md §5.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EnvStatus = Literal["ok", "unresolved", "unknown"]
Weight = Literal["light", "heavy", "service-backed", "unknown"]


class LayerErrorInfo(BaseModel):
    """One config layer that could not be parsed.

    Reported alongside the entries that DID parse, so a single broken file
    never takes the whole tab down.
    """

    layer: str
    path: str
    message: str


class McpServerEntry(BaseModel):
    """One effective server entry, as stored (env NOT resolved)."""

    name: str
    spec: dict[str, Any]
    layer: str = Field(description="user | project | agent | cli")
    disabled: bool = False
    env_status: EnvStatus = "ok"
    unresolved_vars: list[str] = Field(default_factory=list)
    overrides: list[str] = Field(
        default_factory=list,
        description="Lower-priority layers this entry replaces.",
    )
    editable: bool = Field(
        default=False,
        description="True when the entry lives in the agent layer, which is "
        "the only layer this API writes to.",
    )
    weight: Weight = "unknown"
    weight_reason: str = ""


class McpConfigResponse(BaseModel):
    agent_name: str
    servers: list[McpServerEntry] = Field(default_factory=list)
    layer_errors: list[LayerErrorInfo] = Field(default_factory=list)


class McpServerWrite(BaseModel):
    """Body for PUT — the raw ``mcpServers`` entry shape."""

    spec: dict[str, Any]


class McpWriteResponse(BaseModel):
    success: bool = True
    server: str
    warnings: list[str] = Field(
        default_factory=list,
        description="Advisory only. A warning never blocks the write; the "
        "user decides which tools they need.",
    )
    restart_required: bool = Field(
        default=True,
        description="External MCP servers bind at agent start, so a change "
        "takes effect only after an explicit restart.",
    )


class McpEnableRequest(BaseModel):
    enabled: bool


class CatalogEntryModel(BaseModel):
    id: str
    label: str = ""
    origin: str = ""
    source: str = "scan:mcp_config"
    spec: dict[str, Any] = Field(default_factory=dict)
    weight: Weight = "unknown"
    weight_reason: str = ""
    tools_hint: list[str] | None = None
    pinned: bool = False
    stale: bool = False
    discovered_at: str = ""


class CatalogResponse(BaseModel):
    servers: list[CatalogEntryModel] = Field(default_factory=list)


class CatalogWrite(BaseModel):
    spec: dict[str, Any]
    label: str = ""


class ScanRequest(BaseModel):
    """Search roots are supplied by the caller so the feature is not tied
    to any particular host layout (§3)."""

    search_roots: list[str] = Field(default_factory=list)


class ScanResponse(BaseModel):
    success: bool = True
    scanned_files: int = 0
    found: int = 0
    servers: list[CatalogEntryModel] = Field(default_factory=list)


class McpTestRequest(BaseModel):
    spec: dict[str, Any]
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)


class McpTestResponse(BaseModel):
    success: bool
    tool_count: int = 0
    tool_names: list[str] = Field(
        default_factory=list,
        description="Names only — descriptions are deliberately not "
        "returned, to keep the rendered DOM small.",
    )
    error: str = ""


class McpLiveResponse(BaseModel):
    """Tools the agent actually has right now, for configured-vs-live drift."""

    success: bool
    connected: bool = False
    tool_names: list[str] = Field(default_factory=list)
    by_server: dict[str, list[str]] = Field(default_factory=dict)
    error: str = ""
