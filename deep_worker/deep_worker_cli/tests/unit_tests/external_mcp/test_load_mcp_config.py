"""Unit tests for load_mcp_config.

Contract (see DESIGN_EXTERNAL_MCP_TOOLS.md §3.2):

* Layers: user-global → project → per-agent → CLI ad-hoc. A higher
  layer fully replaces an entry with the same name from a lower one.
* Parsing: both roots (``mcpServers`` and a flat dict).
* ``${env:VAR}`` substitution (fail loud when the variable is missing;
  ``${env:VAR:-fallback}`` is supported).
* An empty/missing file is silently ignored.
* ``disabled: true`` skips the entry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deep_worker_cli.external_mcp.config import (
    ServerSpec,
    UnresolvedEnvError,
    load_mcp_config,
)


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_empty_layers_returns_empty(tmp_path: Path) -> None:
    out = load_mcp_config(
        user_path=tmp_path / "user.json",
        project_path=tmp_path / "project.json",
        agent_path=tmp_path / "agent.json",
        cli_configs=[],
        cli_servers=[],
    )
    assert out == {}


def test_parses_mcp_servers_root(tmp_path: Path) -> None:
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {"echo": {"command": "echo", "args": ["hi"]}}})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert "echo" in out
    assert isinstance(out["echo"], ServerSpec)
    assert out["echo"].type == "stdio"
    assert out["echo"].command == "echo"
    assert out["echo"].args == ["hi"]


def test_parses_flat_root_for_cursor_compat(tmp_path: Path) -> None:
    f = tmp_path / "u.json"
    # cursor-style: no mcpServers wrapper, just names → spec.
    _write(f, {"echo": {"command": "echo"}})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert "echo" in out
    assert out["echo"].command == "echo"


def test_http_type_inferred_from_url(tmp_path: Path) -> None:
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {"intra": {"url": "http://127.0.0.1:9/mcp"}}})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert out["intra"].type == "http"
    assert out["intra"].url == "http://127.0.0.1:9/mcp"


def test_disabled_entry_is_skipped(tmp_path: Path) -> None:
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {
        "on": {"command": "echo"},
        "off": {"command": "echo", "disabled": True},
    }})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert set(out.keys()) == {"on"}


def test_project_overrides_user_by_name(tmp_path: Path) -> None:
    user_f = tmp_path / "u.json"
    proj_f = tmp_path / "p.json"
    _write(user_f, {"mcpServers": {"github": {"command": "old-binary"}}})
    _write(proj_f, {"mcpServers": {"github": {"command": "new-binary"}}})
    out = load_mcp_config(user_path=user_f, project_path=proj_f, agent_path=None)
    assert out["github"].command == "new-binary"


def test_agent_overrides_project(tmp_path: Path) -> None:
    u = tmp_path / "u.json"
    p = tmp_path / "p.json"
    a = tmp_path / "a.json"
    _write(u, {"mcpServers": {"x": {"command": "u"}}})
    _write(p, {"mcpServers": {"x": {"command": "p"}}})
    _write(a, {"mcpServers": {"x": {"command": "a"}}})
    out = load_mcp_config(user_path=u, project_path=p, agent_path=a)
    assert out["x"].command == "a"


def test_cli_config_overrides_agent_layer(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    c = tmp_path / "c.json"
    _write(a, {"mcpServers": {"x": {"command": "a"}}})
    _write(c, {"mcpServers": {"x": {"command": "c"}}})
    out = load_mcp_config(
        user_path=None, project_path=None, agent_path=a, cli_configs=[c]
    )
    assert out["x"].command == "c"


def test_cli_server_inline_overrides_all_layers(tmp_path: Path) -> None:
    c = tmp_path / "c.json"
    _write(c, {"mcpServers": {"x": {"command": "from-cli-cfg"}}})
    out = load_mcp_config(
        user_path=None, project_path=None, agent_path=None,
        cli_configs=[c],
        cli_servers=['x={"command":"from-inline"}'],
    )
    assert out["x"].command == "from-inline"


def test_env_substitution_happy_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "ghs_real")
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {"gh": {
        "command": "gh-mcp",
        "env": {"TOKEN": "${env:GH_TOKEN}"},
    }}})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert out["gh"].env == {"TOKEN": "ghs_real"}


def test_env_substitution_missing_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ", raising=False)
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {"gh": {
        "command": "gh-mcp",
        "env": {"TOKEN": "${env:DEFINITELY_NOT_SET_XYZ}"},
    }}})
    with pytest.raises(UnresolvedEnvError):
        load_mcp_config(user_path=f, project_path=None, agent_path=None)


def test_env_substitution_with_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAYBE_UNSET", raising=False)
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {"x": {
        "command": "x",
        "env": {"K": "${env:MAYBE_UNSET:-fallback}"},
    }}})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert out["x"].env == {"K": "fallback"}


def test_missing_files_are_silently_ignored(tmp_path: Path) -> None:
    out = load_mcp_config(
        user_path=tmp_path / "nope1.json",
        project_path=tmp_path / "nope2.json",
        agent_path=tmp_path / "nope3.json",
    )
    assert out == {}


def test_corrupt_json_is_loud_error(tmp_path: Path) -> None:
    f = tmp_path / "u.json"
    f.write_text("{ not json")
    with pytest.raises(ValueError):
        load_mcp_config(user_path=f, project_path=None, agent_path=None)


def test_auto_approve_per_server_is_parsed(tmp_path: Path) -> None:
    f = tmp_path / "u.json"
    _write(f, {"mcpServers": {
        "trusted": {"command": "x", "auto_approve": True},
        "default": {"command": "y"},
    }})
    out = load_mcp_config(user_path=f, project_path=None, agent_path=None)
    assert out["trusted"].auto_approve is True
    assert out["default"].auto_approve is False


def test_http_headers_support_env_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_TOKEN", "secret")
    config = tmp_path / "mcp.json"
    _write(
        config,
        {
            "mcpServers": {
                "remote": {
                    "type": "http",
                    "url": "https://example.invalid/mcp",
                    "headers": {"Authorization": "Bearer ${env:MCP_TOKEN}"},
                }
            }
        },
    )

    out = load_mcp_config(
        user_path=config, project_path=None, agent_path=None
    )

    assert out["remote"].headers == {"Authorization": "Bearer secret"}
