"""Integration test for ExternalMCPClientPool with a REAL local
stdio MCP server (no network dependencies and no mocks).

Scenario:
1. Writes mcp.json into tmp_path containing one stdio server, which
   we immediately launch via ``python3 _fake_mcp_server.py`` (next to
   this file).
2. Pool.create() brings up the client, runs initialize() and list_tools().
3. The test verifies: exactly one tool (``add``), the call returns the
   correct result, and the pool shuts down cleanly.

This test MUST fail BEFORE the pool is implemented and pass AFTER —
it is the integration gate for the promised feature.
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
        # Must be exactly one tool — our add — with the MCP prefix.
        assert "mcp__fake__add" in names, names
        add_tool = next(t for t in tools if t.name == "mcp__fake__add")

        out = await add_tool._arun(a=2, b=40)
        assert "42" in out, out
    finally:
        await pool.stop()
