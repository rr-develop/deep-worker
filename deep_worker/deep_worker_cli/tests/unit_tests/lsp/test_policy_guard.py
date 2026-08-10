"""Tests for the §14 / policy-§8 pre-commit regression lock.

The policy forbids three things. The guard must detect each, and — just
as importantly — must NOT fire on the legitimate code we already ship,
or it will be disabled within a week.

Allowlist note: the design draft named ``lsp.bridge`` as the module
permitted to spawn the backend, but the policy (§8, authoritative) says
**only** ``deep_worker_cli.lsp.service*``. The implementation followed
the policy — the bridge deliberately cannot spawn anything — so the
guard enforces the policy's allowlist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.lsp.policy_guard import (
    Violation,
    scan_mcp_config,
    scan_python_source,
    scan_paths,
)


# ── 1. mcp.json entries that exec a heavy binary over stdio ──────────


def test_heavy_binary_stdio_entry_is_rejected(tmp_path: Path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "serena": {
                        "type": "stdio",
                        "command": "serena",
                        "args": ["start-mcp-server"],
                    }
                }
            }
        )
    )
    violations = scan_mcp_config(cfg)
    assert violations, "heavy stdio entry was not flagged"
    assert any("serena" in v.message for v in violations)


@pytest.mark.parametrize(
    "binary",
    [
        "kotlin-language-server",
        "pyright-langserver",
        "typescript-language-server",
        "rust-analyzer",
        "clangd",
        "gopls",
        "jdtls",
        "metals",
        "gradle",
        "mvn",
    ],
)
def test_every_blocklisted_binary_is_caught(tmp_path: Path, binary: str):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"x": {"command": binary, "args": []}}})
    )
    assert scan_mcp_config(cfg), f"{binary} not flagged"


def test_default_transport_counts_as_stdio(tmp_path: Path):
    """§8: stdio may be implicit — omitting ``type`` must not evade."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"x": {"command": "serena", "args": []}}})
    )
    assert scan_mcp_config(cfg), "implicit stdio transport evaded the guard"


def test_heavy_binary_hidden_in_args_is_caught(tmp_path: Path):
    """``node .../tsserver.js`` hides the heavy binary in ``args``.

    Checking only ``command`` would let the most common wrapper form
    through untouched.
    """
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ts": {
                        "type": "stdio",
                        "command": "node",
                        "args": ["/usr/lib/node_modules/tsserver"],
                    }
                }
            }
        )
    )
    violations = scan_mcp_config(cfg)
    assert violations, "heavy binary hidden in args evaded the guard"
    assert any("tsserver" in v.message for v in violations)


def test_bridge_entry_is_accepted(tmp_path: Path):
    """The shape we actually ship must not trip the guard."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "serena": {
                        "type": "stdio",
                        "command": "deep-worker",
                        "args": ["lsp", "bridge", "--project-root", "/x"],
                    }
                }
            }
        )
    )
    assert scan_mcp_config(cfg) == []


def test_remote_transport_with_heavy_binary_is_allowed(tmp_path: Path):
    """The ban is on in-process stdio, not on the binary existing."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "x": {"type": "sse", "url": "http://localhost:1/sse"}
                }
            }
        )
    )
    assert scan_mcp_config(cfg) == []


# ── 2. Python spawning a heavy binary outside the allowlist ──────────


_SPAWNS = [
    "import subprocess\nsubprocess.Popen(['serena', 'start-mcp-server'])\n",
    "import subprocess\nsubprocess.run(['rust-analyzer'])\n",
    "import asyncio\n"
    "asyncio.create_subprocess_exec('serena', 'start-mcp-server')\n",
]


@pytest.mark.parametrize("src", _SPAWNS)
def test_spawn_outside_allowlist_is_rejected(tmp_path: Path, src: str):
    mod = tmp_path / "deep_worker_cli" / "lsp" / "bridge.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(src)
    violations = scan_python_source(mod, repo_root=tmp_path)
    assert violations, f"spawn not flagged: {src!r}"


@pytest.mark.parametrize("src", _SPAWNS)
def test_spawn_inside_service_package_is_allowed(tmp_path: Path, src: str):
    """``deep_worker_cli.lsp.service*`` is the ONLY allowed spawner."""
    mod = tmp_path / "deep_worker_cli" / "lsp" / "service" / "backend.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(src)
    assert scan_python_source(mod, repo_root=tmp_path) == []


def test_bridge_may_not_spawn_even_via_stdio_client(tmp_path: Path):
    mod = tmp_path / "deep_worker_cli" / "lsp" / "bridge_loop.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(
        "from mcp.client.stdio import stdio_client\n"
        "stdio_client(StdioServerParameters(command='serena'))\n"
    )
    assert scan_python_source(mod, repo_root=tmp_path)


def test_harmless_subprocess_use_is_not_flagged(tmp_path: Path):
    """Guards that cry wolf get disabled. Ordinary spawns are fine."""
    mod = tmp_path / "deep_worker_cli" / "tools.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(
        "import subprocess\n"
        "subprocess.run(['git', 'status'])\n"
        "subprocess.Popen(['ls', '-la'])\n"
    )
    assert scan_python_source(mod, repo_root=tmp_path) == []


# ── 3. exemption list may not change without a design doc ────────────


def test_exemption_change_without_design_doc_is_rejected(tmp_path: Path):
    policy = tmp_path / "docs" / "POLICY_NO_INPROCESS_HEAVY_MCP.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("...exemptions...\n")

    violations = scan_paths(
        [policy], repo_root=tmp_path, changed_paths=[policy]
    )
    assert violations, "exemption edit without a DESIGN doc was allowed"


def test_exemption_change_with_design_doc_is_allowed(tmp_path: Path):
    policy = tmp_path / "docs" / "POLICY_NO_INPROCESS_HEAVY_MCP.md"
    design = tmp_path / "docs" / "DESIGN_SOMETHING.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("...exemptions...\n")
    design.write_text("# design\n")

    violations = scan_paths(
        [policy], repo_root=tmp_path, changed_paths=[policy, design]
    )
    assert violations == []


# ── the real repository must pass its own guard ──────────────────────


def _find_repo_root() -> Path:
    """Walk up to the real repository root.

    Deliberately NOT ``parents[N]``: a magic index silently points at
    the wrong tree when a file moves, and that is exactly what happened
    here — an earlier version used ``parents[4]``
    (``deep_worker/``), under which ``rglob("mcp.json")`` matches
    nothing, so the repo scan passed while the repo's own
    ``.deepagents/mcp.json`` was violating the policy.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists() and (
            candidate / "docs" / "POLICY_NO_INPROCESS_HEAVY_MCP.md"
        ).exists():
            return candidate
    raise AssertionError(f"repository root not found above {here}")


def _repo_targets(repo_root: Path) -> tuple[list[Path], list[Path]]:
    py = [
        p
        for p in repo_root.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
    ]
    configs = [
        p
        for p in repo_root.rglob("mcp.json")
        if ".venv" not in p.parts and ".git" not in p.parts
    ]
    return py, configs


def test_repo_scan_actually_reaches_the_config_files():
    """Guard against a silently empty scan.

    The repo-wide test below is only meaningful if it actually looks at
    ``mcp.json`` files and Python sources. Without this assertion a
    mis-resolved root makes the scan vacuous and green.
    """
    repo_root = _find_repo_root()
    py, configs = _repo_targets(repo_root)

    assert (repo_root / "docs" / "POLICY_NO_INPROCESS_HEAVY_MCP.md").exists()
    assert len(py) > 100, f"only {len(py)} python files under {repo_root}"
    assert configs, (
        f"no mcp.json found under {repo_root} — the repo scan would be "
        f"vacuous and pass without checking anything"
    )


def test_this_repository_passes_the_guard():
    """The strongest test: run the guard over our own shipped code.

    Scans from the real repository root. The first run of this guard
    found a live violation in the repo's own ``.deepagents/mcp.json``
    (a legacy direct-spawn entry).
    """
    repo_root = _find_repo_root()
    py, configs = _repo_targets(repo_root)
    violations = scan_paths(
        [*py, *configs], repo_root=repo_root, changed_paths=[]
    )
    assert violations == [], "\n".join(str(v) for v in violations[:10])


def test_legacy_entry_at_the_repo_config_path_would_fail_the_scan(
    tmp_path: Path,
):
    """The precise regression that slipped through once already.

    A ``.deepagents/mcp.json`` holding the legacy direct-spawn entry —
    at the same relative location as the real one — must be flagged.
    """
    cfg = tmp_path / ".deepagents" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "serena": {
                        "type": "stdio",
                        "command": "serena",
                        "args": ["start-mcp-server", "--project", str(tmp_path)],
                    }
                }
            }
        )
    )
    violations = scan_paths([cfg], repo_root=tmp_path, changed_paths=[])
    assert violations, "the exact historical violation was not detected"
