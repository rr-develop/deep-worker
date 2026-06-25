"""MCPToolAdapter — a wrapper over a single MCP tool record.

Contract (see DESIGN_EXTERNAL_MCP_TOOLS.md §3.3 and
DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md):

* ``name`` = **provider-safe** name (`.` and `/` replaced with `__`),
  the final name by which ToolNode and all providers route.
* ``canonical_name`` = the original MCP name ``<server>.<tool>`` (with dots/
  slashes) — the source of truth for the UI and the MCP-wire call.
* ``description`` is taken from MCP ``tool.description``.
* ``_arun`` proxies into ``session.call_tool(tool_name, kwargs)`` —
  with ``_bare_tool_name`` (the original bare name on the MCP-server side, without
  the server prefix and without any mangling).
* Result: if MCP returned a structure — it is serialized into a JSON string,
  if plain text — returned as-is.

This unit test does NOT raise a real MCP — it only mocks ``ClientSession``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.middleware.bounded_tool_call import BoundedToolCallMiddleware
from langchain.tools.tool_node import ToolCallRequest, ToolRuntime
from langchain_core.messages import ToolMessage

from deep_worker_cli.external_mcp.adapter import MCPToolAdapter


class _FakeSession:
    """Imitates exactly the slice of mcp.ClientSession the adapter uses:
    one ``call_tool(name, args)`` method returning an object with
    ``content`` and ``isError`` attributes."""

    def __init__(self, payload: Any, is_error: bool = False) -> None:
        self.payload = payload
        self.is_error = is_error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> Any:
        self.calls.append((name, args))
        # The MCP SDK returns CallToolResult with .content: list[TextContent|...]
        # and .isError. We imitate the minimal shape.
        text_block = SimpleNamespace(type="text", text=self.payload)
        return SimpleNamespace(content=[text_block], isError=self.is_error)


@pytest.mark.asyncio
async def test_adapter_name_is_mangled_canonical_preserved() -> None:
    """``name`` is provider-safe (mangled), ``canonical_name`` is the original."""
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="github",
        tool_name="create_issue",
        description="Create a GitHub issue",
        input_schema={"type": "object", "properties": {}},
    )
    # `.` in `<server>.<tool>` → `__` in the provider-safe name.
    assert a.name == "github__create_issue"
    assert a.canonical_name == "github.create_issue"


@pytest.mark.asyncio
async def test_adapter_name_mangled_for_dotted_tool_name() -> None:
    """Real scenario from a chrome bridge: a tool with a dot in its name
    (`browser.list_tabs`) → mangled `chrome__browser__list_tabs`,
    canonical — `chrome.browser.list_tabs`."""
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="chrome",
        tool_name="browser.list_tabs",
        description="",
        input_schema={"type": "object", "properties": {}},
    )
    assert a.name == "chrome__browser__list_tabs"
    assert a.canonical_name == "chrome.browser.list_tabs"


@pytest.mark.asyncio
async def test_adapter_description_propagated() -> None:
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="x",
        tool_name="y",
        description="Hello from MCP",
        input_schema={"type": "object", "properties": {}},
    )
    assert a.description == "Hello from MCP"


@pytest.mark.asyncio
async def test_adapter_uses_provided_mangler_for_name_only() -> None:
    """When the caller passes a ``mangler``, the adapter registers its
    canonical name in it and takes the name from there. The wire call always goes by the
    bare tool_name (without the server prefix, without mangling)."""
    from deep_worker_cli.external_mcp.mangler import MCPToolNameMangler

    mangler = MCPToolNameMangler()
    sess = _FakeSession("ok")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="chrome",
        tool_name="browser.status",
        description="",
        input_schema={"type": "object", "properties": {}},
        mangler=mangler,
    )
    assert a.name == "chrome__browser__status"
    assert a.canonical_name == "chrome.browser.status"
    # The mangler really received the canonical.
    assert mangler.to_canonical("chrome__browser__status") == "chrome.browser.status"
    # The wire call goes by the bare tool_name, without the server prefix and without mangling.
    await a._arun()
    assert sess.calls == [("browser.status", {})]


@pytest.mark.asyncio
async def test_adapter_arun_forwards_kwargs_to_call_tool() -> None:
    sess = _FakeSession("done")
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="github",
        tool_name="create_issue",
        description="",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title"],
        },
    )
    out = await a._arun(title="hi", body="there")
    assert "done" in out
    # The adapter must proxy EXACTLY under the bare tool_name, without the prefix.
    assert sess.calls == [("create_issue", {"title": "hi", "body": "there"})]


@pytest.mark.asyncio
async def test_adapter_serialises_structured_payload() -> None:
    sess = _FakeSession({"id": 42, "url": "https://x/42"})
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="x", tool_name="y",
        description="", input_schema={"type": "object", "properties": {}},
    )
    out = await a._arun()
    # A structured payload must come back as a JSON string.
    assert json.loads(out) == {"id": 42, "url": "https://x/42"}


@pytest.mark.asyncio
async def test_adapter_is_error_surfaced_as_exception() -> None:
    sess = _FakeSession("boom", is_error=True)
    a = MCPToolAdapter(
        session=sess,  # type: ignore[arg-type]
        server_name="x", tool_name="y",
        description="", input_schema={"type": "object", "properties": {}},
    )
    # SafeToolCallMiddleware will turn RuntimeError into ToolMessage(status="error").
    with pytest.raises(RuntimeError):
        await a._arun()


@pytest.mark.asyncio
async def test_adapter_slow_call_uses_existing_bounded_middleware() -> None:
    started = asyncio.Event()

    class _SlowSession:
        async def call_tool(self, name: str, args: dict) -> Any:
            started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

    adapter = MCPToolAdapter(
        session=_SlowSession(),
        server_name="slow",
        tool_name="wait",
        description="wait forever",
        input_schema={"type": "object", "properties": {}},
    )
    state = {"messages": []}
    runtime = ToolRuntime(
        state=state,
        context=None,
        tool_call_id="toolu_MCP",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )
    request = ToolCallRequest(
        runtime=runtime,
        tool_call={"id": "toolu_MCP", "name": adapter.name, "args": {}},
        state=state,
        tool=adapter,
    )

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        result = await adapter._arun()
        return ToolMessage(content=result, tool_call_id="toolu_MCP")

    middleware = BoundedToolCallMiddleware()
    loop = asyncio.get_running_loop()
    real_time = loop.time
    skew = {"seconds": 0.0}
    loop.time = lambda: real_time() + skew["seconds"]  # type: ignore[method-assign]
    try:
        task = asyncio.create_task(
            middleware.awrap_tool_call(request, handler)
        )
        await started.wait()
        skew["seconds"] = middleware.wait_seconds + 1
        await asyncio.sleep(0.02)
        result = await task
        assert isinstance(result, ToolMessage)
        assert result.status != "error"
        assert "in_progress" in str(result.content)
        assert middleware.get_job_registry()
    finally:
        loop.time = real_time  # type: ignore[method-assign]
        for job in middleware.get_job_registry().values():
            job.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job.task
