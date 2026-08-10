"""Unit tests for the client-side bridge entry (DESIGN §15 commit 4).

The bridge is the per-agent stdio MCP process. Its ONLY job is:

1. ensure the shared per-project service is running (auto-start),
2. proxy MCP tool calls to it over gRPC.

Invariants pinned here:

* the bridge NEVER spawns Serena itself (policy §8: only
  ``deep_worker_cli.lsp.service*`` may) — enforced by an AST scan of
  the bridge module;
* stdio-fallback is FORBIDDEN: if the service cannot be reached the
  bridge fails loudly rather than degrading to an in-process backend;
* the mcp.json entry generated for the bridge points at the bridge,
  not at the ``serena`` binary.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from deep_worker_cli.lsp import bridge as bridge_mod
from deep_worker_cli.lsp import bridge_loop as bridge_loop_mod
from deep_worker_cli.lsp.bridge import (
    BridgeUnavailable,
    build_bridge_mcp_entry,
    connect_or_fail,
)


@pytest.mark.parametrize("mod", [bridge_mod, bridge_loop_mod])
def test_bridge_module_never_spawns_serena(mod):
    """AST invariant: no subprocess/exec/fork of a serena binary.

    Applies to EVERY client-side bridge module: only the allowlisted
    package ``deep_worker_cli.lsp.service*`` may spawn the backend.
    """
    src = Path(inspect.getfile(mod)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned_attrs = {"Popen", "run", "call", "check_output", "check_call"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in banned_attrs:
                owner = getattr(fn.value, "id", "")
                assert owner != "subprocess", ast.dump(node)
            if isinstance(fn, ast.Attribute) and fn.attr.startswith("exec"):
                assert getattr(fn.value, "id", "") != "os", ast.dump(node)
            if isinstance(fn, ast.Attribute) and fn.attr == "fork":
                assert getattr(fn.value, "id", "") != "os", ast.dump(node)
    # It may name the MCP server "serena" (tool names stay stable),
    # but it must never import the module that knows how to exec the
    # backend binary, nor reference that binary's CLI verb.
    imported = {
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
    } | {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    }
    assert "subprocess" not in imported, imported
    assert not any(m.endswith("serena_config") for m in imported), imported
    assert "start-mcp-server" not in src


@pytest.mark.parametrize("mod", [bridge_mod, bridge_loop_mod])
def test_bridge_has_no_stdio_fallback(mod):
    """The word 'fallback' must not gate a degraded in-process path."""
    src = Path(inspect.getfile(mod)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any("fallback" in n.lower() for n in names), names


def test_connect_or_fail_raises_when_service_absent(tmp_path: Path, monkeypatch):
    from deep_worker_cli.lsp.service.supervisor import ServiceStartTimeout

    class _Sup:
        def __init__(self, *a, **kw) -> None:
            pass

        def ensure_started(self, root: Path):
            raise ServiceStartTimeout("boom")

    monkeypatch.setattr("deep_worker_cli.lsp.bridge.ServiceSupervisor", _Sup)
    with pytest.raises(BridgeUnavailable) as ei:
        connect_or_fail(tmp_path, registry_dir=tmp_path / "reg")
    # Must be actionable, and must NOT suggest a stdio fallback.
    msg = str(ei.value).lower()
    assert "fallback" not in msg
    assert "lsp" in msg


def test_bridge_mcp_entry_points_at_bridge_not_serena(tmp_path: Path):
    entry = build_bridge_mcp_entry(workspace_root=tmp_path)
    assert entry["type"] == "stdio"
    joined = " ".join([entry["command"], *entry["args"]])
    assert "bridge" in joined
    assert "start-mcp-server" not in joined  # that's the serena CLI shape
    assert str(tmp_path.resolve()) in joined
