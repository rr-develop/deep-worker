"""ExternalMCPClientPool — name bijection and all-or-nothing collisions.

Contract (see docs/DESIGN_EXTERNAL_MCP_TOOL_NAME_MANGLING.md §3.2 and §6):

* The pool owns ONE ``MCPToolNameMangler`` for the whole agent lifecycle.
* It catches mangled-name collisions between DIFFERENT servers.
* A collision in the N-th tool of a server → the server is skipped ENTIRELY, and
  the agent's main mangler does not receive partial registrations
  (all-or-nothing per server).
* Correctly raised servers get into ``pool.tools`` with the mangled `name`
  and keep the canonical in ``adapter.canonical_name``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from deep_worker_cli.external_mcp.config import ServerSpec
from deep_worker_cli.external_mcp.pool import ExternalMCPClientPool


class _StubSession:
    """A minimal double of ``mcp.ClientSession``: returns a given
    list of tools from ``list_tools`` and accepts any ``call_tool``."""

    def __init__(self, tool_names: list[str]) -> None:
        self._tool_names = tool_names
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> Any:
        tools = [
            SimpleNamespace(
                name=n,
                description=f"stub {n}",
                inputSchema={"type": "object", "properties": {}},
            )
            for n in self._tool_names
        ]
        return SimpleNamespace(tools=tools)

    async def call_tool(self, name: str, args: dict) -> Any:
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"ok({name})")],
            isError=False,
        )


async def _start_pool_with_servers(
    monkeypatch: pytest.MonkeyPatch,
    specs: dict[str, tuple[str, list[str]]],
) -> ExternalMCPClientPool:
    """Raises a pool with pre-defined servers `{server_name:
    (transport_type, [tool_names])}`. The ``mcp`` SDK is NOT used —
    we replace the extension points with stub adapters."""

    # Each server gets its own _StubSession.
    sessions = {name: _StubSession(tools) for name, (_, tools) in specs.items()}

    class _FakeStack:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_: Any) -> bool:
            return False

    # We replace _start_all so that it does not touch the real mcp.
    pool = ExternalMCPClientPool()
    pool._started = True  # type: ignore[attr-defined]

    from deep_worker_cli.external_mcp.adapter import MCPToolAdapter
    from deep_worker_cli.external_mcp.mangler import MCPToolNameMangler

    for name, (_ttype, tool_names) in specs.items():
        sess = sessions[name]
        await sess.initialize()
        listed = await sess.list_tools()
        try:
            tentative = MCPToolNameMangler()
            for prior in pool.mangler.all_canonical_names():
                tentative.add(prior)
            adapters = [
                MCPToolAdapter(
                    session=sess,
                    server_name=name,
                    tool_name=t.name,
                    description=t.description,
                    input_schema=t.inputSchema,
                    mangler=tentative,
                )
                for t in listed.tools
            ]
            # All-or-nothing commit into the main mangler.
            for a in adapters:
                pool.mangler.add(a.canonical_name)
            pool._tools.extend(adapters)  # type: ignore[attr-defined]
        except Exception:
            # "server failed to start" — skip it, the rest keep working.
            pass

    return pool


@pytest.mark.asyncio
async def test_pool_uses_single_mangler_across_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two different servers, both without collisions — both get into pool.tools."""
    pool = await _start_pool_with_servers(
        monkeypatch,
        {
            "github": ("stdio", ["create_issue", "list_repos"]),
            "chrome": ("http", ["browser.list_tabs", "browser.status"]),
        },
    )
    names = sorted(t.name for t in pool.tools)
    assert names == sorted([
        "github__create_issue",
        "github__list_repos",
        "chrome__browser__list_tabs",
        "chrome__browser__status",
    ])
    # The bijection in pool.mangler: chrome tools are there (mangled),
    # github tools are not (names without dots/slashes).
    assert pool.mangler.to_canonical("chrome__browser__status") == "chrome.browser.status"
    assert "github.create_issue" in pool.mangler.forward


@pytest.mark.asyncio
async def test_collision_between_servers_fails_only_offender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collision between two servers: server `ns` with the tool `a.b` gives
    canonical `ns.a.b` → mangled `ns__a__b`; server `ns/a` with the tool `b`
    gives canonical `ns/a.b` → mangled `ns__a__b`. The same mangled for
    different canonicals → collision. The offending server is skipped
    ENTIRELY (all-or-nothing), the first one remains."""
    pool = await _start_pool_with_servers(
        monkeypatch,
        {
            "ns": ("stdio", ["a.b"]),
            # 'ns/a' has two tools; `b` collides with `ns.a.b` → all-or-
            # nothing: both are skipped.
            "ns/a": ("stdio", ["b", "harmless"]),
        },
    )
    names = sorted(t.name for t in pool.tools)
    # Only the first server's tool remains.
    assert names == ["ns__a__b"]
    # The main mangler is NOT polluted with partial additions from the second:
    # 'ns__a__harmless' is not there (because the whole server was rolled back).
    assert "ns__a__harmless" not in pool.mangler.all_mangled_names()


@pytest.mark.asyncio
async def test_collision_within_one_server_fails_only_that_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within one server: both canonicals `x.y` and `x/y` mangle to
    `x__y`. This server is skipped entirely."""
    pool = await _start_pool_with_servers(
        monkeypatch,
        {
            "good": ("stdio", ["alpha", "beta"]),
            "bad": ("stdio", ["x.y", "x/y"]),
        },
    )
    names = sorted(t.name for t in pool.tools)
    # 'bad' is skipped entirely, 'good' — remains.
    assert names == ["good__alpha", "good__beta"]
    assert all("x__y" != t.name for t in pool.tools)
