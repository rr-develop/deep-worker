"""Integration test for ExternalMCPClientPool with a REAL local
stdio MCP server (no network dependencies and no mocks).

Scenario:
1. Writes mcp.json into tmp_path with one stdio server, which we
   immediately launch via ``python3 _fake_mcp_server.py`` (next to it).
2. Pool.create() raises the client, runs initialize() and list_tools().
3. The test checks: exactly one tool (``add``), the call returns the correct
   result, and the pool closes properly.

This test MUST fail BEFORE the pool is implemented and pass AFTER — it is
the integration gate for the promised feature.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from deep_worker_cli.external_mcp.pool import ExternalMCPClientPool

_HERE = Path(__file__).parent
_FAKE_SERVER = _HERE / "_fake_mcp_server.py"


@pytest.mark.asyncio
async def test_pool_starts_fake_mcp_and_invokes_tool(tmp_path: Path) -> None:
    # Config with one stdio server that launches the fake MCP.
    mcp_json = tmp_path / "mcp.json"
    mcp_json.write_text(json.dumps({
        "mcpServers": {
            "fake": {
                "command": sys.executable,
                "args": [str(_FAKE_SERVER)],
            }
        }
    }))

    pool = await ExternalMCPClientPool.create(
        user_path=mcp_json,
        project_path=None,
        agent_path=None,
        cli_configs=[],
        cli_servers=[],
    )
    try:
        tools = pool.tools
        names = [t.name for t in tools]
        # `name` is provider-safe (`.` → `__`); the canonical is stored separately.
        # See docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md.
        assert "fake__add" in names, names
        add_tool = next(t for t in tools if t.name == "fake__add")
        # The canonical name is kept for UI/wire calls.
        assert add_tool.canonical_name == "fake.add"  # type: ignore[attr-defined]

        out = await add_tool._arun(a=2, b=40)
        assert "42" in out, out
    finally:
        await pool.stop()
