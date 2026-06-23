"""External MCP config loader.

Layered JSON config for external MCP servers: user → project →
per-agent → CLI ad-hoc. A higher layer fully replaces an entry with
the same name from a lower layer. See DESIGN_EXTERNAL_MCP_TOOLS.md §3.

Deliberately NO global state and no side-effects on
import — pure load/merge/validation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class UnresolvedEnvError(ValueError):
    """Raised when ``${env:VAR}`` references a variable that is not set
    (and no ``:-fallback`` is provided). Loud-by-design — starting a
    server with an empty token is worse than failing fast."""


@dataclass(frozen=True)
class ServerSpec:
    """One MCP server entry, parsed and validated.

    Frozen so the same instance can be safely shared between layers
    (lower layers are dropped wholesale on name collision, not mutated).
    """

    name: str
    type: str  # "stdio" | "http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    auto_approve: bool = False


# ``${env:VAR}`` or ``${env:VAR:-fallback}``. Fallback may itself contain
# nothing fancier than a literal string (no recursion needed).
_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _resolve_env_in_value(raw: str) -> str:
    """Resolve every ``${env:VAR}``/``${env:VAR:-fallback}`` in ``raw``.

    Raises ``UnresolvedEnvError`` if a referenced variable is not set
    and no fallback is provided. Variables ARE allowed to expand to
    empty strings as long as they ARE set — that matches shell ``${VAR}``.
    """

    def _sub(m: re.Match[str]) -> str:
        var, fallback = m.group(1), m.group(2)
        if var in os.environ:
            return os.environ[var]
        if fallback is not None:
            return fallback
        raise UnresolvedEnvError(
            f"environment variable {var!r} referenced by mcp.json is not set"
        )

    return _ENV_RE.sub(_sub, raw)


def _resolve_env_in_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    """Apply env-substitution to every string value of a flat mapping."""
    out: dict[str, str] = {}
    for k, v in mapping.items():
        if not isinstance(v, str):
            # JSON allows numbers/booleans here; coerce to str for env.
            v = json.dumps(v)
        out[str(k)] = _resolve_env_in_value(v)
    return out


def _parse_one(name: str, raw: dict[str, Any]) -> ServerSpec | None:
    """Parse one server entry. Returns None if ``disabled: true``.

    Loud failure for unknown shapes (e.g. neither command nor url) —
    a config typo is easier to catch at start than in production.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"mcp server {name!r} must be a JSON object")
    if raw.get("disabled") is True:
        return None

    declared_type = raw.get("type")
    if declared_type == "http" or (declared_type is None and "url" in raw):
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"mcp server {name!r}: http requires non-empty 'url'")
        headers_in = raw.get("headers") or {}
        if not isinstance(headers_in, dict):
            raise ValueError(f"mcp server {name!r}: headers must be an object")
        return ServerSpec(
            name=name,
            type="http",
            url=_resolve_env_in_value(url),
            headers=_resolve_env_in_mapping(headers_in),
            auto_approve=bool(raw.get("auto_approve", False)),
        )

    # stdio default
    command = raw.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError(
            f"mcp server {name!r}: stdio requires non-empty 'command' "
            f"(or set 'type':'http' and 'url')"
        )
    args_in = raw.get("args") or []
    if not isinstance(args_in, list) or not all(isinstance(a, str) for a in args_in):
        raise ValueError(f"mcp server {name!r}: 'args' must be a list of strings")
    env_in = raw.get("env") or {}
    if not isinstance(env_in, dict):
        raise ValueError(f"mcp server {name!r}: env must be an object")
    cwd = raw.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError(f"mcp server {name!r}: 'cwd' must be a string")
    return ServerSpec(
        name=name,
        type="stdio",
        command=_resolve_env_in_value(command),
        args=[_resolve_env_in_value(a) for a in args_in],
        env=_resolve_env_in_mapping(env_in),
        cwd=_resolve_env_in_value(cwd) if cwd else None,
        auto_approve=bool(raw.get("auto_approve", False)),
    )


def _parse_layer(data: Any) -> dict[str, ServerSpec]:
    """Parse one config document (root or flat). Empty / None → {}.

    Supports both forms:
      * ``{"mcpServers": {"name": {...}}}``   (de-facto MCP standard)
      * ``{"name": {...}, "other": {...}}``   (cursor-style flat)
    """
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("mcp config root must be a JSON object")
    if "mcpServers" in data and isinstance(data["mcpServers"], dict):
        servers = data["mcpServers"]
    else:
        # Reject mixed-shape: ``mcpServers`` + extra top-level keys is a typo.
        # Pure flat form: every value must be a dict.
        if any(not isinstance(v, dict) for v in data.values()):
            raise ValueError(
                "mcp config: top-level entries must be either {'mcpServers': "
                "{...}} or a flat name→spec map of objects"
            )
        servers = data

    out: dict[str, ServerSpec] = {}
    for name, raw in servers.items():
        if not isinstance(name, str) or not name:
            raise ValueError("mcp server name must be a non-empty string")
        spec = _parse_one(name, raw)
        if spec is not None:
            out[name] = spec
    return out


def _read_json_file(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        # Wrap as ValueError so callers can catch one type.
        raise ValueError(f"mcp config {path}: {e}") from None


def load_mcp_config(
    *,
    user_path: Path | None,
    project_path: Path | None,
    agent_path: Path | None,
    cli_configs: Iterable[Path] = (),
    cli_servers: Iterable[str] = (),
) -> dict[str, ServerSpec]:
    """Merge layered MCP config and return the final ``{name: ServerSpec}``.

    Layers (lowest → highest priority):
        1. user_path     — ~/.deepagents/mcp.json
        2. project_path  — <project>/.deepagents/mcp.json
        3. agent_path    — ~/.deepagents/<agent>/mcp.json
        4. cli_configs   — every ``--mcp-config <path>``, in order
        5. cli_servers   — every ``--mcp-server '<name>={...}'``, in order

    A higher layer fully replaces a same-named entry from a lower one.

    All env-substitution happens here; if any ``${env:VAR}`` reference
    is unresolved (and has no ``:-fallback``) ``UnresolvedEnvError`` is
    raised — failing loud is safer than starting a server with a blank
    token.
    """
    merged: dict[str, ServerSpec] = {}

    for layer_path in (user_path, project_path, agent_path):
        layer = _parse_layer(_read_json_file(layer_path))
        merged.update(layer)

    for cfg in cli_configs:
        merged.update(_parse_layer(_read_json_file(cfg)))

    for inline in cli_servers:
        # Format: '<name>={...json...}'  (single-arg per --mcp-server)
        if "=" not in inline:
            raise ValueError(
                "--mcp-server requires '<name>={...json...}' form, got: "
                f"{inline!r}"
            )
        name, _, body = inline.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--mcp-server: empty name in {inline!r}")
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--mcp-server {name!r}: JSON parse error: {e}"
            ) from None
        spec = _parse_one(name, raw if isinstance(raw, dict) else {})
        if spec is not None:
            merged[name] = spec

    return merged
