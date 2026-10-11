"""Integration tests for the MCP tab — NO MOCKS.

A real MCP server (built with the pinned ``mcp`` SDK) is started as a real
child process and probed through the real API surface over a real ASGI
transport. The only thing stubbed out is the process registry, which is
about agent discovery rather than anything under test here.

What this proves that unit tests cannot:

* ``POST /mcp/test`` really speaks MCP to a real stdio server and returns
  the tool list it actually advertises;
* a server that hangs on ``initialize`` is bounded and reported, instead
  of parking the request forever;
* the scan really reads config files off disk and never executes them;
* a written entry is really loadable by the agent-side loader — i.e. the
  web and the agent agree on the file format.

Budget: the whole module is bounded well under 60 s, and every spawned
process is killed in a ``finally``.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# A minimal but real MCP server over stdio, using the same SDK the product
# pins. Written to a temp file and executed as a child process.
_REAL_SERVER = textwrap.dedent(
    '''
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types

    app = Server("integration-test-server")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name="echo", description="Echo back",
                       inputSchema={"type": "object",
                                    "properties": {"text": {"type": "string"}}}),
            types.Tool(name="ping", description="Ping",
                       inputSchema={"type": "object", "properties": {}}),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        return [types.TextContent(type="text", text=f"{name}:ok")]

    async def main() -> None:
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())

    anyio.run(main)
    '''
)

# A server that completes the handshake at the transport level and then
# never answers — the realistic hang, which a connect-only timeout misses.
_HANGING_SERVER = textwrap.dedent(
    """
    import sys, time
    sys.stdin.readline()
    time.sleep(3600)
    """
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    path = tmp_path / "real_mcp_server.py"
    path.write_text(_REAL_SERVER)
    return path


def _sdk_path() -> str:
    """Directory holding the installed ``mcp`` SDK.

    The child inherits a ``HOME`` pointing at the test's fake home, so the
    usual ``~/.local/lib`` entry drops out of its ``sys.path``. Passing the
    real location explicitly keeps the child runnable without weakening the
    home isolation the rest of the module depends on.
    """
    import mcp as _mcp

    return str(Path(_mcp.__file__).resolve().parent.parent)


def _stdio_spec(script: Path) -> dict:
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(script)],
        "env": {"PYTHONPATH": _sdk_path()},
    }


@pytest.fixture
def hanging_script(tmp_path: Path) -> Path:
    path = tmp_path / "hanging_server.py"
    path.write_text(_HANGING_SERVER)
    return path


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture
async def client(fake_home: Path):
    from fastapi import FastAPI

    from app.routers.mcp import router

    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── a real server, over a real transport ─────────────────────────────


async def test_test_endpoint_lists_tools_of_a_real_server(
    client: AsyncClient, server_script: Path
) -> None:
    resp = await asyncio.wait_for(
        client.post(
            "/api/agents/itest/mcp/test",
            json={
                "spec": _stdio_spec(server_script),
                "timeout_seconds": 25,
            },
        ),
        timeout=40,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True, body
    assert body["tool_count"] == 2
    assert set(body["tool_names"]) == {"echo", "ping"}


async def test_test_endpoint_returns_names_without_descriptions(
    client: AsyncClient, server_script: Path
) -> None:
    """DOM-size discipline: descriptions must not travel to the browser."""
    resp = await asyncio.wait_for(
        client.post(
            "/api/agents/itest/mcp/test",
            json={
                "spec": _stdio_spec(server_script),
                "timeout_seconds": 25,
            },
        ),
        timeout=40,
    )

    assert "Echo back" not in resp.text


async def test_hanging_server_is_bounded_not_parked(
    client: AsyncClient, hanging_script: Path
) -> None:
    """The whole call must come back, not just the socket connect."""
    resp = await asyncio.wait_for(
        client.post(
            "/api/agents/itest/mcp/test",
            json={
                "spec": {"type": "stdio", "command": sys.executable,
                         "args": [str(hanging_script)]},
                "timeout_seconds": 3,
            },
        ),
        timeout=25,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"]


async def test_unstartable_server_reports_error(client: AsyncClient) -> None:
    resp = await asyncio.wait_for(
        client.post(
            "/api/agents/itest/mcp/test",
            json={
                "spec": {"type": "stdio",
                         "command": "/nonexistent/definitely-not-here"},
                "timeout_seconds": 5,
            },
        ),
        timeout=25,
    )

    assert resp.json()["success"] is False


# ── write → the agent-side loader really accepts it ──────────────────


async def test_written_entry_is_loadable_by_the_agent_loader(
    client: AsyncClient, fake_home: Path, server_script: Path
) -> None:
    """The web and the agent must agree on the file format. This asserts
    it against the REAL loader, not a copy of its rules."""
    resp = await client.put(
        "/api/agents/itest/mcp/real",
        json={"spec": _stdio_spec(server_script)},
    )
    assert resp.status_code == 200, resp.text

    from deep_worker_cli.external_mcp.config import load_mcp_config

    loaded = load_mcp_config(
        user_path=None,
        project_path=None,
        agent_path=fake_home / ".deep_worker" / "agents" / "itest" / "mcp.json",
    )

    assert "real" in loaded
    assert loaded["real"].command == sys.executable


async def test_written_file_is_private(
    client: AsyncClient, fake_home: Path
) -> None:
    await client.put(
        "/api/agents/itest/mcp/srv", json={"spec": {"command": "x"}}
    )
    path = fake_home / ".deep_worker" / "agents" / "itest" / "mcp.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_pool_actually_starts_a_written_server(
    client: AsyncClient, fake_home: Path, server_script: Path
) -> None:
    """End-to-end: what the UI writes is what the pool can start, and the
    adapter tools carry the mangled ``server__tool`` names the agent sees."""
    await client.put(
        "/api/agents/itest/mcp/real",
        json={"spec": _stdio_spec(server_script)},
    )

    from deep_worker_cli.external_mcp.pool import ExternalMCPClientPool

    pool = await asyncio.wait_for(
        ExternalMCPClientPool.create(
            user_path=None,
            project_path=None,
            agent_path=fake_home / ".deep_worker" / "agents" / "itest" / "mcp.json",
        ),
        timeout=30,
    )
    try:
        names = {t.name for t in pool.tools}
        assert "real__echo" in names, names
        assert "real__ping" in names, names
    finally:
        await asyncio.wait_for(pool.stop(), timeout=15)


# ── catalog & scan against a real filesystem ─────────────────────────


async def test_scan_finds_real_config_and_never_executes_it(
    client: AsyncClient, fake_home: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "was_executed.marker"
    cfg = fake_home / ".deep_worker" / "agents" / "otheragent" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "discovered": {
                        "command": "/bin/sh",
                        "args": ["-c", f"touch {marker}"],
                    }
                }
            }
        )
    )

    resp = await asyncio.wait_for(
        client.post("/api/mcp/catalog/scan", json={}), timeout=25
    )

    assert resp.status_code == 200
    assert "discovered" in [s["id"] for s in resp.json()["servers"]]
    assert not marker.exists(), "scan must never execute a candidate server"


async def test_manual_entry_survives_a_rescan(
    client: AsyncClient, fake_home: Path
) -> None:
    """The durability guarantee for hand-added entries, end to end."""
    await client.put(
        "/api/mcp/catalog/handmade",
        json={"spec": {"type": "http", "url": "http://127.0.0.1:9/mcp"}},
    )

    await asyncio.wait_for(
        client.post("/api/mcp/catalog/scan", json={}), timeout=25
    )

    body = (await client.get("/api/mcp/catalog")).json()
    entry = next(s for s in body["servers"] if s["id"] == "handmade")
    assert entry["pinned"] is True
    assert entry["spec"]["url"] == "http://127.0.0.1:9/mcp"


async def test_catalog_file_is_private_and_outside_the_repo(
    client: AsyncClient, fake_home: Path
) -> None:
    await client.put("/api/mcp/catalog/x", json={"spec": {"command": "x"}})

    path = fake_home / ".deep_worker" / "mcp" / "catalog.json"
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


# ── the tolerant read, for real ──────────────────────────────────────


async def test_get_survives_a_real_broken_file(
    client: AsyncClient, fake_home: Path
) -> None:
    path = fake_home / ".deep_worker" / "agents" / "itest" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ definitely not json")

    resp = await client.get("/api/agents/itest/mcp")

    assert resp.status_code == 200
    assert resp.json()["layer_errors"]


async def test_get_survives_a_real_unset_env_reference(
    client: AsyncClient, fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REALLY_UNSET_VAR_ABC", raising=False)
    path = fake_home / ".deep_worker" / "agents" / "itest" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"mcpServers": {"s": {"type": "http", "url": "http://x",
                                  "headers": {"A": "${env:REALLY_UNSET_VAR_ABC}"}}}}
        )
    )

    resp = await client.get("/api/agents/itest/mcp")

    assert resp.status_code == 200
    entry = resp.json()["servers"][0]
    assert entry["env_status"] == "unresolved"
    # And the secret placeholder is returned verbatim, never resolved.
    assert entry["spec"]["headers"]["A"] == "${env:REALLY_UNSET_VAR_ABC}"
