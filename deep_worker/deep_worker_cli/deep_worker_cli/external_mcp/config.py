"""External MCP config loader.

Layered JSON config for external MCP servers: user → project →
per-agent → CLI ad-hoc. A higher layer fully replaces a same-named
entry from a lower one. See DESIGN_EXTERNAL_MCP_TOOLS.md §3.

Deliberately NO global state, no side effects on import — pure
loading/merging/validation.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class UnresolvedEnvError(ValueError):
    """Raised when ``${env:VAR}`` references a variable that is not set
    (and no ``:-fallback`` is provided). Loud-by-design — starting a
    server with an empty token is worse than failing fast.
    """


class UnresolvedAgentPlaceholderError(ValueError):
    """Raised when ``${agent:...}`` cannot be resolved by this caller.

    Same loud-by-design contract as :class:`UnresolvedEnvError`: a
    user-scope bridge entry whose root silently expanded to an empty
    string would index the wrong tree, which is precisely the defect
    the placeholder exists to prevent (see
    ``docs/design/lsp_user_scope_follows_agent_cwd.md`` §3).
    """


class ExternalMCPStartupError(RuntimeError):
    """An external MCP failed to start, and the agent must SURFACE that.

    Same loud-by-design contract as ``UnresolvedEnvError``: the failure
    reaches the agent's initialization handler, which moves the agent to
    the ``error`` state with this text. The agent still continues startup,
    brings up RPyC and registers — otherwise it would be missing from the
    web UI and its history could not be read (see a related internal
    incident analysis, not included here).
    """


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

#: ``${agent:working_dir}`` — the agent's own workspace root, substituted
#: at agent start. Deliberately a SEPARATE namespace from ``env:`` so the
#: two can never collide: this value is passed in explicitly by the
#: caller, never read from ambient process state.
_AGENT_RE = re.compile(r"\$\{agent:([A-Za-z_][A-Za-z0-9_]*)\}")

#: The only recognised ``${agent:...}`` key. Kept as a named constant so
#: the writer in ``lsp/enable.py`` and the reader here cannot drift.
AGENT_WORKING_DIR_PLACEHOLDER = "${agent:working_dir}"


def _resolve_agent_in_value(raw: str, working_dir: str | None) -> str:
    """Resolve ``${agent:working_dir}`` in ``raw``.

    ``working_dir`` is the agent's workspace root, passed in explicitly
    by the caller — this module keeps no global state, so a loader that
    was given no working directory (CLI inspection, web UI) must not
    guess one. In that case the placeholder is UNRESOLVED and raises,
    rather than expanding to an empty root.
    """

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key != "working_dir":
            raise UnresolvedAgentPlaceholderError(
                f"unknown placeholder ${{agent:{key}}} in mcp.json; "
                f"the only supported key is 'working_dir'"
            )
        if not working_dir:
            raise UnresolvedAgentPlaceholderError(
                "${agent:working_dir} in mcp.json cannot be resolved here: "
                "no agent working directory was supplied to the loader"
            )
        return working_dir

    return _AGENT_RE.sub(_sub, raw)


def _resolve_env_in_value(raw: str, working_dir: str | None = None) -> str:
    """Resolve every ``${env:VAR}``/``${env:VAR:-fallback}`` in ``raw``.

    Raises ``UnresolvedEnvError`` if a referenced variable is not set
    and no fallback is provided. Variables ARE allowed to expand to
    empty strings as long as they ARE set — that matches shell ``${VAR}``.

    ``${agent:working_dir}`` is resolved in the same pass so that every
    place a value can appear (``command``/``args``/``env``/``cwd``)
    supports both namespaces without duplicating call sites.
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

    return _AGENT_RE.sub(
        lambda m: _resolve_agent_in_value(m.group(0), working_dir),
        _ENV_RE.sub(_sub, raw),
    )


def _resolve_env_in_mapping(
    mapping: dict[str, Any], working_dir: str | None = None
) -> dict[str, str]:
    """Apply env-substitution to every string value of a flat mapping."""
    out: dict[str, str] = {}
    for k, v in mapping.items():
        if not isinstance(v, str):
            # JSON allows numbers/booleans here; coerce to str for env.
            v = json.dumps(v)
        out[str(k)] = _resolve_env_in_value(v, working_dir)
    return out


def _parse_one(
    name: str, raw: dict[str, Any], working_dir: str | None = None
) -> ServerSpec | None:
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
            url=_resolve_env_in_value(url, working_dir),
            headers=_resolve_env_in_mapping(headers_in, working_dir),
            auto_approve=bool(raw.get("auto_approve")),
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
        command=_resolve_env_in_value(command, working_dir),
        args=[_resolve_env_in_value(a, working_dir) for a in args_in],
        env=_resolve_env_in_mapping(env_in, working_dir),
        cwd=_resolve_env_in_value(cwd, working_dir) if cwd else None,
        auto_approve=bool(raw.get("auto_approve")),
    )


def _parse_layer(
    data: Any, working_dir: str | None = None
) -> dict[str, ServerSpec]:
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
        spec = _parse_one(name, raw, working_dir)
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
    working_dir: Path | str | None = None,
) -> dict[str, ServerSpec]:
    """Merge layered MCP config and return the final ``{name: ServerSpec}``.

    Layers (lowest → highest priority):
        1. user_path     — ~/.deep_worker/mcp.json
        2. project_path  — <project>/.deep_worker/mcp.json
        3. agent_path    — ~/.deep_worker/agents/<agent>/mcp.json
        4. cli_configs   — every ``--mcp-config <path>``, in order
        5. cli_servers   — every ``--mcp-server '<name>={...}'``, in order

    A higher layer fully replaces a same-named entry from a lower one.

    All env-substitution happens here; if any ``${env:VAR}`` reference
    is unresolved (and has no ``:-fallback``) ``UnresolvedEnvError`` is
    raised — failing loud is safer than starting a server with a blank
    token.

    ``working_dir`` is the agent's own directory. It resolves
    ``${agent:working_dir}``, which is how a USER-scope bridge entry
    expresses "navigate my own project" without baking one project's
    path into a config shared by every agent. The root is normalised
    through ``resolve_workspace_root`` (git root, else the directory
    itself), so a user-scope and a project-scope entry written in the
    same tree denote the same root. Callers that have no agent context
    (CLI inspection, web UI) pass ``None``; a config that then uses the
    placeholder raises ``UnresolvedAgentPlaceholderError`` instead of
    silently resolving to an empty path.
    """
    resolved_working_dir: str | None = None
    if working_dir:
        from deep_worker_cli.lsp.serena_config import resolve_workspace_root

        resolved_working_dir = str(resolve_workspace_root(working_dir))

    merged: dict[str, ServerSpec] = {}

    for layer_path in (user_path, project_path, agent_path):
        layer = _parse_layer(_read_json_file(layer_path), resolved_working_dir)
        merged.update(layer)

    for cfg in cli_configs:
        merged.update(
            _parse_layer(_read_json_file(cfg), resolved_working_dir)
        )

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
        spec = _parse_one(
            name, raw if isinstance(raw, dict) else {}, resolved_working_dir
        )
        if spec is not None:
            merged[name] = spec

    return merged
