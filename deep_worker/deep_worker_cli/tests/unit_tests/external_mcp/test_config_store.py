"""Tests for the shared MCP config store (``external_mcp.config_store``).

These helpers are the SINGLE writer of ``mcp.json`` for both the CLI
(``deep-worker mcp add/remove``) and deep_worker_web. Divergence between
the two would silently corrupt user config, so the contract is pinned
here: scope→path resolution, the ``{"mcpServers": {...}}`` wrapper,
atomic 0600 writes, and the raw (non-env-resolving) read used by the web
read endpoint.

See deep_worker_web/docs/DESIGN_MCP_TAB.md §5.1.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from deep_worker_cli.external_mcp.config_store import (
    LayerRead,
    RawEntry,
    UnknownScopeError,
    read_layer,
    read_merged_raw,
    remove_server,
    scope_to_path,
    upsert_server,
)


# ── scope → path ─────────────────────────────────────────────────────


def test_scope_user_path(tmp_path: Path) -> None:
    assert scope_to_path("user", home=tmp_path) == tmp_path / ".deep_worker" / "mcp.json"


def test_scope_agent_path(tmp_path: Path) -> None:
    assert (
        scope_to_path("agent:bob", home=tmp_path)
        == tmp_path / ".deep_worker" / "agents" / "bob" / "mcp.json"
    )


def test_scope_agent_requires_name(tmp_path: Path) -> None:
    with pytest.raises(UnknownScopeError):
        scope_to_path("agent:", home=tmp_path)


def test_scope_agent_rejects_path_traversal(tmp_path: Path) -> None:
    """An agent name is a single path segment; ``..`` must not escape."""
    with pytest.raises(UnknownScopeError):
        scope_to_path("agent:../../etc", home=tmp_path)


def test_unknown_scope_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownScopeError):
        scope_to_path("nope", home=tmp_path)


def test_scope_project_uses_given_root(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    assert (
        scope_to_path("project", home=tmp_path, project_root=root)
        == root / ".deep_worker" / "mcp.json"
    )


# ── write: atomicity, permissions, wrapper ───────────────────────────


def test_upsert_creates_file_0600_and_dir_0700(tmp_path: Path) -> None:
    path = tmp_path / ".deep_worker" / "agents" / "bob" / "mcp.json"
    upsert_server(path, "srv", {"type": "http", "url": "http://x/mcp"})

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_upsert_wraps_in_mcp_servers(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    upsert_server(path, "srv", {"type": "http", "url": "http://x/mcp"})
    assert json.loads(path.read_text())["mcpServers"]["srv"]["url"] == "http://x/mcp"


def test_upsert_migrates_flat_cursor_style_file(tmp_path: Path) -> None:
    """A cursor-style flat file is migrated to the wrapped form, keeping
    the pre-existing entries."""
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"old": {"command": "x"}}))

    upsert_server(path, "srv", {"type": "http", "url": "http://x/mcp"})

    data = json.loads(path.read_text())
    assert set(data["mcpServers"]) == {"old", "srv"}


def test_upsert_preserves_other_entries(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    upsert_server(path, "a", {"command": "a"})
    upsert_server(path, "b", {"command": "b"})
    assert set(json.loads(path.read_text())["mcpServers"]) == {"a", "b"}


def test_upsert_overwrites_same_name(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    upsert_server(path, "a", {"command": "old"})
    upsert_server(path, "a", {"command": "new"})
    assert json.loads(path.read_text())["mcpServers"]["a"]["command"] == "new"


def test_upsert_leaves_no_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    upsert_server(path, "a", {"command": "a"})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "mcp.json"]
    assert leftovers == []


def test_remove_server(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    upsert_server(path, "a", {"command": "a"})
    upsert_server(path, "b", {"command": "b"})

    assert remove_server(path, "a") is True
    assert set(json.loads(path.read_text())["mcpServers"]) == {"b"}


def test_remove_missing_server_returns_false(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    upsert_server(path, "a", {"command": "a"})
    assert remove_server(path, "ghost") is False


def test_remove_from_absent_file_returns_false(tmp_path: Path) -> None:
    assert remove_server(tmp_path / "nope.json", "a") is False


# ── raw read: never resolves env, never explodes ─────────────────────


def test_read_layer_absent_file_is_empty_not_error(tmp_path: Path) -> None:
    got = read_layer(tmp_path / "nope.json")
    assert got.entries == {}
    assert got.error is None


def test_read_layer_keeps_env_placeholder_verbatim(tmp_path: Path) -> None:
    """The raw read must NOT resolve ``${env:...}`` — that both avoids
    a 500 on a missing variable and keeps secrets out of the response."""
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {"mcpServers": {"s": {"type": "http", "url": "http://x",
                                  "headers": {"Authorization": "${env:TOK}"}}}}
        )
    )
    entry = read_layer(path).entries["s"]
    assert entry.spec["headers"]["Authorization"] == "${env:TOK}"


def test_read_layer_unresolved_env_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEFINITELY_UNSET_VAR_XYZ", raising=False)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"s": {"command": "${env:DEFINITELY_UNSET_VAR_XYZ}"}}})
    )

    got = read_layer(path)

    assert got.error is None
    assert got.entries["s"].env_status == "unresolved"
    assert "DEFINITELY_UNSET_VAR_XYZ" in got.entries["s"].unresolved_vars


def test_read_layer_resolvable_env_is_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SET_VAR_XYZ", "v")
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"s": {"command": "${env:SET_VAR_XYZ}"}}}))
    assert read_layer(path).entries["s"].env_status == "ok"


def test_read_layer_env_with_fallback_is_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UNSET_WITH_FALLBACK_XYZ", raising=False)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {"mcpServers": {"s": {"command": "${env:UNSET_WITH_FALLBACK_XYZ:-def}"}}}
        )
    )
    assert read_layer(path).entries["s"].env_status == "ok"


def test_read_layer_no_env_reference_is_ok(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"s": {"command": "plain"}}}))
    assert read_layer(path).entries["s"].env_status == "ok"


def test_read_layer_malformed_json_reports_error_not_raises(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{not json")

    got = read_layer(path)

    assert got.error is not None
    assert got.entries == {}


def test_read_layer_disabled_entry_is_kept(tmp_path: Path) -> None:
    """Unlike the loader, the raw read keeps disabled entries — the UI
    must be able to show and re-enable them."""
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"s": {"command": "x", "disabled": True}}})
    )
    entry = read_layer(path).entries["s"]
    assert entry.disabled is True


# ── merge across layers ──────────────────────────────────────────────


def _write(path: Path, name: str, spec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {name: spec}}))


def test_merge_reports_source_layer(tmp_path: Path) -> None:
    user = tmp_path / "user.json"
    agent = tmp_path / "agent.json"
    _write(user, "u", {"command": "u"})
    _write(agent, "a", {"command": "a"})

    merged = read_merged_raw([("user", user), ("agent", agent)])

    assert merged.entries["u"].layer == "user"
    assert merged.entries["a"].layer == "agent"


def test_merge_higher_layer_wins_and_records_override(tmp_path: Path) -> None:
    user = tmp_path / "user.json"
    agent = tmp_path / "agent.json"
    _write(user, "s", {"command": "from-user"})
    _write(agent, "s", {"command": "from-agent"})

    merged = read_merged_raw([("user", user), ("agent", agent)])

    assert merged.entries["s"].spec["command"] == "from-agent"
    assert merged.entries["s"].layer == "agent"
    assert merged.entries["s"].overrides == ["user"]


def test_merge_one_broken_layer_does_not_sink_the_rest(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    _write(good, "g", {"command": "g"})
    bad.write_text("{broken")

    merged = read_merged_raw([("user", good), ("project", bad)])

    assert "g" in merged.entries
    assert any(e.layer == "project" for e in merged.layer_errors)


def test_merge_absent_layers_are_silent(tmp_path: Path) -> None:
    merged = read_merged_raw(
        [("user", tmp_path / "a.json"), ("agent", tmp_path / "b.json")]
    )
    assert merged.entries == {}
    assert merged.layer_errors == []
