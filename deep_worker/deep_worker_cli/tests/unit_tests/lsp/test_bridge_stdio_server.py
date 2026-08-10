"""Tests for the bridge's MCP stdio server (DESIGN §15 commit 5c).

5b delivered ``BridgeSession`` (the gRPC proxy). 5c is the transport:
``deep-worker lsp bridge`` must actually speak MCP over stdio, so an
agent can list and call the tools.

Pinned here:

* the advertised tool list matches the service surface exactly, and each
  tool carries an input schema (an agent cannot call what it cannot see);
* a successful call returns the backend payload as MCP content;
* a FAILING call is reported as an MCP **error**, not as a successful
  empty result — the recurring invariant of this design;
* the schema declares the arguments the model must pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_worker_cli.lsp.bridge_stdio import (
    build_mcp_server,
    tool_definitions,
)
from deep_worker_cli.lsp.bridge_loop import ToolCallFailed


class _FakeSession:
    """Stands in for a connected BridgeSession."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if self._error is not None:
            raise self._error
        return self._result


# ── advertised surface ───────────────────────────────────────────────


def test_tool_definitions_cover_the_whole_service_surface():
    from deep_worker_cli.lsp.bridge_loop import tool_names

    defs = tool_definitions()
    assert {d.name for d in defs} == set(tool_names())


def test_every_tool_has_a_usable_input_schema():
    """A tool with no schema is uncallable by the model."""
    for d in tool_definitions():
        assert d.description, f"{d.name} has no description"
        schema = d.inputSchema
        assert schema.get("type") == "object", d.name
        assert "properties" in schema, d.name


def test_symbol_tools_declare_their_required_argument():
    """``find_symbol`` without ``name_path`` is meaningless.

    Being present in ``properties`` is not enough — it must also be in
    ``required``, otherwise the model may omit it and get a nonsense
    whole-project query back.
    """
    by_name = {d.name: d for d in tool_definitions()}
    schema = by_name["find_symbol"].inputSchema
    assert "name_path" in schema["properties"], schema
    assert "name_path" in schema.get("required", []), schema


def test_every_tool_marks_its_essential_arguments_required():
    """No tool may advertise an empty ``required`` list.

    Each of these tools acts on a specific symbol or file; a call with
    no arguments is never meaningful, so an empty ``required`` would
    invite the model to make one.
    """
    for d in tool_definitions():
        assert d.inputSchema.get("required"), (
            f"{d.name} declares no required arguments"
        )


# ── dispatch ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_successful_call_returns_payload_as_content():
    session = _FakeSession(result={"symbols": ["Foo"]})
    server = build_mcp_server(session)

    handler = server.request_handlers
    result = await _call(server, "find_symbol", {"name_path": "Foo"})

    assert session.calls == [("find_symbol", {"name_path": "Foo"})]
    text = "".join(c.text for c in result.content)
    assert json.loads(text) == {"symbols": ["Foo"]}
    assert result.isError is not True


@pytest.mark.asyncio
async def test_failed_call_is_reported_as_mcp_error_not_empty_result():
    """The invariant: a failure must never look like 'no matches'."""
    session = _FakeSession(error=ToolCallFailed("service is down"))
    server = build_mcp_server(session)

    result = await _call(server, "find_symbol", {"name_path": "Foo"})

    assert result.isError is True, "failure was reported as a success"
    text = "".join(c.text for c in result.content)
    assert "service is down" in text
    # And it must not look like an empty result set.
    assert text.strip() not in ("", "[]", "{}", "null")


@pytest.mark.asyncio
async def test_arguments_reach_the_session_verbatim():
    session = _FakeSession(result={})
    server = build_mcp_server(session)

    args = {"name_path": "A/b", "depth": 3, "include_body": True}
    await _call(server, "find_symbol", args)

    assert session.calls == [("find_symbol", args)]


# ── helper ───────────────────────────────────────────────────────────


async def _call(server, name: str, arguments: dict):
    """Invoke the server's registered call_tool handler."""
    import mcp.types as types

    handler = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    server_result = await handler(req)
    return server_result.root


# ── enable now writes the bridge entry (§14 policy invariant) ────────


def test_enable_writes_bridge_entry_not_a_direct_backend_spawn(tmp_path: Path):
    """§14: no agent-local mcp.json may exec the backend directly.

    This is the regression invariant for the whole policy: once the
    stdio loop exists, ``lsp enable`` must produce a bridge entry so N
    agents share one service instead of spawning N backends.
    """
    from deep_worker_cli.lsp.enable import enable_lsp_at_path

    cfg = tmp_path / ".deepagents" / "mcp.json"
    enable_lsp_at_path(cfg, workspace_root=tmp_path)

    entry = json.loads(cfg.read_text())["mcpServers"]["serena"]
    argv = " ".join([entry["command"], *entry["args"]])

    assert "start-mcp-server" not in argv, (
        "enable still spawns the backend directly — policy violation"
    )
    assert "bridge" in argv, argv
    assert str(tmp_path.resolve()) in argv
