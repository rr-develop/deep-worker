"""Unit tests for the MCP tab API (``app/routers/mcp.py``).

Contract from deep_worker_web/docs/DESIGN_MCP_TAB.md:

* ``GET /api/agents/{name}/mcp`` is a RAW merge — it must never 500 on an
  unresolved ``${env:VAR}`` or on one malformed layer (§5.1), because that
  is the very screen a user would open to fix the broken entry.
* Writes go to the AGENT layer only, atomically, 0600 (§5, §9).
* The catalog is host state under the product home, tolerant to being
  absent or malformed (§4).
* Weight is advisory metadata; nothing is ever refused because of it
  (§1.1) — the user picks the tools they consider necessary.
* Rendering never probes a server; tool counts come from an explicit Test
  or from ``/mcp/live`` (§7.2).
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def mcp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both config roots at a temp home for the whole app."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture()
def client(mcp_home: Path) -> TestClient:
    from app.main import app

    return TestClient(app)


def _agent_layer(home: Path, agent: str) -> Path:
    return home / ".deep_worker" / "agents" / agent / "mcp.json"


def _write_layer(path: Path, servers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}))


# ── GET /mcp — tolerant raw merge ────────────────────────────────────


def test_get_mcp_empty_when_nothing_configured(client: TestClient) -> None:
    resp = client.get("/api/agents/someagent/mcp")
    assert resp.status_code == 200
    assert resp.json()["servers"] == []


def test_get_mcp_lists_agent_layer_entry(client: TestClient, mcp_home: Path) -> None:
    _write_layer(_agent_layer(mcp_home, "a1"), {"srv": {"command": "x"}})

    body = client.get("/api/agents/a1/mcp").json()

    assert [s["name"] for s in body["servers"]] == ["srv"]
    assert body["servers"][0]["layer"] == "agent"


def test_get_mcp_does_not_resolve_env(client: TestClient, mcp_home: Path) -> None:
    """Secrets must not be resolved into the response."""
    _write_layer(
        _agent_layer(mcp_home, "a1"),
        {"s": {"type": "http", "url": "http://x",
               "headers": {"Authorization": "${env:SOME_TOKEN}"}}},
    )

    body = client.get("/api/agents/a1/mcp").json()

    assert body["servers"][0]["spec"]["headers"]["Authorization"] == "${env:SOME_TOKEN}"


def test_get_mcp_unresolved_env_is_200_with_warning(
    client: TestClient, mcp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this endpoint exists to avoid: a missing variable
    must not take the whole tab down with a 500."""
    monkeypatch.delenv("UNSET_TOKEN_XYZ", raising=False)
    _write_layer(
        _agent_layer(mcp_home, "a1"), {"s": {"command": "${env:UNSET_TOKEN_XYZ}"}}
    )

    resp = client.get("/api/agents/a1/mcp")

    assert resp.status_code == 200
    entry = resp.json()["servers"][0]
    assert entry["env_status"] == "unresolved"
    assert "UNSET_TOKEN_XYZ" in entry["unresolved_vars"]


def test_get_mcp_malformed_layer_is_reported_not_fatal(
    client: TestClient, mcp_home: Path
) -> None:
    path = _agent_layer(mcp_home, "a1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken json")

    resp = client.get("/api/agents/a1/mcp")

    assert resp.status_code == 200
    assert resp.json()["layer_errors"]


def test_get_mcp_reports_disabled_entries(client: TestClient, mcp_home: Path) -> None:
    _write_layer(
        _agent_layer(mcp_home, "a1"), {"s": {"command": "x", "disabled": True}}
    )
    assert client.get("/api/agents/a1/mcp").json()["servers"][0]["disabled"] is True


# ── PUT / DELETE / enable — agent layer only ─────────────────────────


def test_put_creates_entry_in_agent_layer(client: TestClient, mcp_home: Path) -> None:
    resp = client.put(
        "/api/agents/a1/mcp/srv",
        json={"spec": {"type": "http", "url": "http://x/mcp"}},
    )

    assert resp.status_code == 200
    saved = json.loads(_agent_layer(mcp_home, "a1").read_text())
    assert saved["mcpServers"]["srv"]["url"] == "http://x/mcp"


def test_put_writes_0600(client: TestClient, mcp_home: Path) -> None:
    client.put("/api/agents/a1/mcp/srv", json={"spec": {"command": "x"}})
    path = _agent_layer(mcp_home, "a1")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_put_does_not_touch_user_layer(client: TestClient, mcp_home: Path) -> None:
    user = mcp_home / ".deep_worker" / "mcp.json"
    _write_layer(user, {"other": {"command": "keep"}})

    client.put("/api/agents/a1/mcp/srv", json={"spec": {"command": "x"}})

    assert json.loads(user.read_text())["mcpServers"] == {"other": {"command": "keep"}}


def test_put_heavy_server_is_allowed(client: TestClient, mcp_home: Path) -> None:
    """Weight is advisory: the user decides. This must NOT be refused."""
    resp = client.put(
        "/api/agents/a1/mcp/lsp",
        json={"spec": {"type": "stdio", "command": "serena"}},
    )

    assert resp.status_code == 200
    assert resp.json()["warnings"]  # advisory hint is present…
    saved = json.loads(_agent_layer(mcp_home, "a1").read_text())
    assert saved["mcpServers"]["lsp"]["command"] == "serena"  # …and it was written


def test_put_unresolved_env_warns_but_saves(
    client: TestClient, mcp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_TOKEN_XYZ", raising=False)

    resp = client.put(
        "/api/agents/a1/mcp/srv",
        json={"spec": {"type": "http", "url": "http://x",
                       "headers": {"A": "${env:MISSING_TOKEN_XYZ}"}}},
    )

    assert resp.status_code == 200
    assert any("MISSING_TOKEN_XYZ" in w for w in resp.json()["warnings"])
    assert _agent_layer(mcp_home, "a1").exists()


def test_put_rejects_invalid_spec(client: TestClient) -> None:
    """Neither command nor url — genuinely unusable, so this IS refused."""
    resp = client.put("/api/agents/a1/mcp/srv", json={"spec": {"type": "stdio"}})
    assert resp.status_code == 400


def test_put_rejects_bad_agent_name(client: TestClient) -> None:
    resp = client.put("/api/agents/..%2F..%2Fetc/mcp/srv", json={"spec": {"command": "x"}})
    assert resp.status_code in (400, 404)


def test_delete_removes_entry(client: TestClient, mcp_home: Path) -> None:
    _write_layer(_agent_layer(mcp_home, "a1"), {"a": {"command": "x"}, "b": {"command": "y"}})

    resp = client.delete("/api/agents/a1/mcp/a")

    assert resp.status_code == 200
    assert set(json.loads(_agent_layer(mcp_home, "a1").read_text())["mcpServers"]) == {"b"}


def test_delete_missing_entry_is_404(client: TestClient, mcp_home: Path) -> None:
    _write_layer(_agent_layer(mcp_home, "a1"), {"a": {"command": "x"}})
    assert client.delete("/api/agents/a1/mcp/ghost").status_code == 404


def test_enable_toggles_disabled_flag(client: TestClient, mcp_home: Path) -> None:
    _write_layer(_agent_layer(mcp_home, "a1"), {"a": {"command": "x", "disabled": True}})

    resp = client.post("/api/agents/a1/mcp/a/enable", json={"enabled": True})

    assert resp.status_code == 200
    assert "disabled" not in json.loads(_agent_layer(mcp_home, "a1").read_text())["mcpServers"]["a"]


def test_disable_sets_flag(client: TestClient, mcp_home: Path) -> None:
    _write_layer(_agent_layer(mcp_home, "a1"), {"a": {"command": "x"}})

    client.post("/api/agents/a1/mcp/a/enable", json={"enabled": False})

    saved = json.loads(_agent_layer(mcp_home, "a1").read_text())
    assert saved["mcpServers"]["a"]["disabled"] is True


# ── catalog ──────────────────────────────────────────────────────────


def test_catalog_empty_on_fresh_host(client: TestClient) -> None:
    """A fresh host is a normal state, not an error (§3)."""
    resp = client.get("/api/mcp/catalog")
    assert resp.status_code == 200
    assert resp.json()["servers"] == []


def test_catalog_malformed_reads_as_empty(client: TestClient, mcp_home: Path) -> None:
    path = mcp_home / ".deep_worker" / "mcp" / "catalog.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")

    resp = client.get("/api/mcp/catalog")

    assert resp.status_code == 200
    assert resp.json()["servers"] == []


def test_catalog_put_adds_pinned_entry(client: TestClient, mcp_home: Path) -> None:
    resp = client.put(
        "/api/mcp/catalog/mine",
        json={"spec": {"type": "http", "url": "http://x/mcp"}},
    )

    assert resp.status_code == 200
    body = client.get("/api/mcp/catalog").json()
    assert body["servers"][0]["pinned"] is True


def test_catalog_scan_preserves_pinned_entry(
    client: TestClient, mcp_home: Path
) -> None:
    """The durability guarantee for user-added entries (§4.2)."""
    client.put("/api/mcp/catalog/mine", json={"spec": {"command": "mine"}})

    client.post("/api/mcp/catalog/scan", json={})

    ids = [s["id"] for s in client.get("/api/mcp/catalog").json()["servers"]]
    assert "mine" in ids


def test_catalog_scan_finds_configured_server(
    client: TestClient, mcp_home: Path
) -> None:
    _write_layer(mcp_home / ".deep_worker" / "agents" / "other" / "mcp.json", {"found": {"command": "x"}})

    client.post("/api/mcp/catalog/scan", json={})

    ids = [s["id"] for s in client.get("/api/mcp/catalog").json()["servers"]]
    assert "found" in ids


def test_catalog_scan_does_not_execute_servers(
    client: TestClient, mcp_home: Path
) -> None:
    """A scan must never run a candidate: point it at a command that would
    fail loudly if executed, and require a clean 200."""
    marker = mcp_home / "executed.marker"
    _write_layer(
        mcp_home / ".deep_worker" / "agents" / "other" / "mcp.json",
        {"danger": {"command": "/bin/sh", "args": ["-c", f"touch {marker}"]}},
    )

    resp = client.post("/api/mcp/catalog/scan", json={})

    assert resp.status_code == 200
    assert not marker.exists()


def test_catalog_scan_reports_no_tool_counts(client: TestClient, mcp_home: Path) -> None:
    """§7.2: no hidden probing, so a scanned entry has no tools_hint."""
    _write_layer(mcp_home / ".deep_worker" / "agents" / "other" / "mcp.json", {"s": {"command": "x"}})

    client.post("/api/mcp/catalog/scan", json={})

    entry = next(s for s in client.get("/api/mcp/catalog").json()["servers"] if s["id"] == "s")
    assert entry["tools_hint"] is None


def test_catalog_delete(client: TestClient, mcp_home: Path) -> None:
    client.put("/api/mcp/catalog/mine", json={"spec": {"command": "x"}})

    assert client.delete("/api/mcp/catalog/mine").status_code == 200
    assert client.get("/api/mcp/catalog").json()["servers"] == []


def test_catalog_weight_is_advisory_label(client: TestClient) -> None:
    client.put("/api/mcp/catalog/heavy1", json={"spec": {"command": "serena"}})

    entry = client.get("/api/mcp/catalog").json()["servers"][0]

    assert entry["weight"] == "heavy"
    assert entry["weight_reason"]


# ── /mcp/live — tools actually attached to the running agent ──────────


class _FakeRoot:
    """Minimal stand-in for the agent's RPyC root: exposes get_tools as an
    async callable returning the same JSON-string shape the real
    ``exposed_get_tools`` produces (netref-free by design)."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def get_tools(self) -> str:
        return self._payload


class _FakeAgentService:
    """Stand-in for AgentService: the two members get_live_tools touches."""

    def __init__(self, root: _FakeRoot) -> None:
        self._roots = {"live-agent": root}

    async def ensure_connecting(self, agent_name: str) -> bool:
        return agent_name in self._roots


@pytest.fixture()
def live_client(mcp_home: Path) -> TestClient:
    """TestClient whose app.state.agent_service is a fake rooted agent."""
    from fastapi import FastAPI

    from app.routers.mcp import router as mcp_router

    app = FastAPI()
    app.include_router(mcp_router)
    app.state.agent_service = _FakeAgentService(
        _FakeRoot(json.dumps({"success": True, "tools": [
            {"name": "real__echo"},
            {"name": "real__ping"},
        ]}))
    )
    return TestClient(app)


def test_live_returns_parsed_tools_by_server(live_client: TestClient) -> None:
    resp = live_client.get("/api/agents/live-agent/mcp/live")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["connected"] is True
    assert sorted(body["tool_names"]) == ["real__echo", "real__ping"]
    assert body["by_server"]["real"] == ["echo", "ping"]


def test_live_disconnected_agent_reports_not_connected(live_client: TestClient) -> None:
    resp = live_client.get("/api/agents/nobody/mcp/live")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["connected"] is False


def test_live_unparseable_tools_does_not_500(live_client: TestClient) -> None:
    """A malformed payload from the agent must surface as an error field,
    not as a server crash."""
    from fastapi import FastAPI

    from app.routers.mcp import router as mcp_router

    app = FastAPI()
    app.include_router(mcp_router)
    app.state.agent_service = _FakeAgentService(_FakeRoot("not json at all"))
    bad = TestClient(app)

    resp = bad.get("/api/agents/live-agent/mcp/live")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["connected"] is True
    assert "unparseable" in body["error"]
