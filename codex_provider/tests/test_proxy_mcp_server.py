"""Tests for ``codex_provider.proxy_mcp_server``.

The proxy MCP server is the in-agent-process bridge between Codex CLI
and LangGraph's ToolNode. Behaviours under test:

* Construction: takes a list of ``ToolMetadata`` and exposes each tool
  via MCP's ``list_tools`` + ``call_tool`` contract.
* ``call_tool(name, args)`` creates an awaitable Future and parks it
  in ``pending_calls`` (it does NOT execute the tool itself — that's
  the architectural invariant).
* ``resolve_call(call_id, result)`` sets the corresponding Future's
  result; the awaiting MCP handler returns the value to Codex.
* ``has_pending_calls`` / ``get_pending_calls`` expose what's parked
  so ``CodexChatModel.ainvoke`` can return ``tool_calls`` to LangGraph.
* Timeout: if no ``resolve_call`` arrives within ``timeout`` seconds,
  the handler returns an ``is_error`` MCP content block.
* Lifecycle: ``start()`` binds an HTTP listener on a free localhost
  port; ``stop()`` shuts it down; pending Futures auto-expire.

The HTTP transport itself is exercised by the live integration test —
here we test the in-process logic by calling handlers directly.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from codex_provider.custom_types import ToolMetadata  # noqa: E402
from codex_provider.proxy_mcp_server import ProxyMCPServer  # noqa: E402


@pytest.fixture
def metadata() -> list[ToolMetadata]:
    return [
        {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    ]


class TestConstruction:
    def test_can_construct_with_metadata(self, metadata: list[ToolMetadata]) -> None:
        srv = ProxyMCPServer(metadata)
        assert srv.tool_names == ["read_file", "write_file"]

    def test_construct_with_no_tools(self) -> None:
        srv = ProxyMCPServer([])
        assert srv.tool_names == []


@pytest.mark.asyncio
class TestCallToolParkingFuture:
    async def test_call_tool_parks_future_and_blocks(
        self, metadata: list[ToolMetadata]
    ) -> None:
        srv = ProxyMCPServer(metadata, timeout=5.0)

        # Drive the handler in the background — it should not return
        # until we resolve the Future.
        task = asyncio.create_task(srv._call_tool_for_test("read_file", {"path": "/tmp/x"}))
        # Yield control so the task gets to schedule its Future and park
        await asyncio.sleep(0)
        assert srv.has_pending_calls()
        pending = srv.get_pending_calls()
        assert len(pending) == 1
        assert pending[0]["name"] == "read_file"
        assert pending[0]["args"] == {"path": "/tmp/x"}

        call_id = pending[0]["id"]
        srv.resolve_call(call_id, {"content": [{"type": "text", "text": "result-text"}]})

        result = await task
        # result is a list of MCP content blocks (the MCP server contract)
        # — what came out of our handler.
        assert result and isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "result-text"
        assert not srv.has_pending_calls()  # finally clause cleared it

    async def test_timeout_returns_is_error(
        self, metadata: list[ToolMetadata]
    ) -> None:
        srv = ProxyMCPServer(metadata, timeout=0.1)  # 100 ms
        result = await srv._call_tool_for_test("read_file", {"path": "/tmp/x"})
        # On timeout the handler returns an is_error block, NOT raises.
        # This way Codex sees a tool-error and continues.
        assert any(block.get("type") == "text" and "timeout" in block.get("text", "").lower()
                   for block in result)

    async def test_resolve_unknown_id_is_noop(
        self, metadata: list[ToolMetadata]
    ) -> None:
        srv = ProxyMCPServer(metadata)
        # Should not raise; no Future exists.
        srv.resolve_call("does_not_exist", {"content": []})


@pytest.mark.asyncio
class TestServerLifecycle:
    """``start()`` binds an HTTP listener on a free localhost port;
    ``stop()`` shuts it down. The port is exposed via ``.port``."""

    async def test_start_binds_a_port_and_stop_releases_it(
        self, metadata: list[ToolMetadata]
    ) -> None:
        srv = ProxyMCPServer(metadata)
        await srv.start()
        try:
            assert isinstance(srv.port, int)
            assert srv.port > 0
            # Server name reflects PID so concurrent agents don't clash
            assert srv.server_name.startswith("deepworker_")
        finally:
            await srv.stop()

    async def test_double_start_is_idempotent(
        self, metadata: list[ToolMetadata]
    ) -> None:
        srv = ProxyMCPServer(metadata)
        await srv.start()
        port_a = srv.port
        await srv.start()  # second call should not rebind
        port_b = srv.port
        await srv.stop()
        assert port_a == port_b

    async def test_double_stop_is_idempotent(
        self, metadata: list[ToolMetadata]
    ) -> None:
        srv = ProxyMCPServer(metadata)
        await srv.start()
        await srv.stop()
        await srv.stop()  # must not raise
