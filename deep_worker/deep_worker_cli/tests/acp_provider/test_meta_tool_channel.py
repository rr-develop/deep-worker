"""``session/new`` must carry the tool-configuration channel the adapter reads.

The warning deep-worker printed for ``claude-code`` ("the ACP adapter exposes
no tool-configuration channel — session/new carries only {cwd, mcpServers}")
described OUR payload, not the adapter's capability. Verified against the
installed adapter @agentclientprotocol/claude-agent-acp 0.68.0 and the
Claude Agent SDK 0.3.232 it depends on:

* ``dist/acp-agent.js:4628`` reads ``_meta.claudeCode.options`` into
  ``userProvidedOptions``;
* ``:4650`` resolves the built-in set from ``userProvidedOptions?.tools``;
* ``:4730`` concatenates ``userProvidedOptions?.disallowedTools`` onto its own;
* SDK ``sdk.mjs:151`` turns those into ``--tools`` / ``--disallowedTools``
  (``--tools ""`` for the empty array), and passes MCP servers through the
  SEPARATE ``--mcp-config`` flag;
* SDK ``sdk.d.ts:1457`` states it outright: "Specify the base set of available
  built-in tools. ``[]`` (empty array) - Disable all built-in tools".

So ``tools: []`` removes Claude Code's OWN built-ins (Bash, Read, Write, …)
while the proxy-MCP tools from the allowlist keep working — which is exactly
what a hermetic restricted run needs.

The byte-for-byte assertion matters as much as the positive one: an agent
with no restriction must produce the IDENTICAL ``session/new`` params it
produced before this channel existed, so the default path cannot regress.
"""

from __future__ import annotations

import pytest

from deep_worker.providers.acp.acp_client import AcpClient
from deep_worker_cli.tool_selection import (
    parse_tool_selection,
    native_tool_caveat,
)


def _client() -> AcpClient:
    """An ``AcpClient`` whose transport is captured instead of spawned."""
    client = object.__new__(AcpClient)
    client._session_ready = {}
    client._env = {}
    client._session_new_validator = None
    client._proc = None
    return client


async def _capture(client: AcpClient, **kwargs) -> dict:
    """Run ``session_new`` against a stubbed ``_call`` and return its params."""
    seen: dict = {}

    async def fake_call(method: str, params: dict) -> dict:
        seen["method"] = method
        seen["params"] = params
        return {"result": {"sessionId": "sess-1"}}

    client._call = fake_call  # type: ignore[method-assign]
    await client.session_new(cwd="/w", mcp_servers=[], **kwargs)
    return seen["params"]


@pytest.mark.asyncio
async def test_no_meta_keeps_session_new_payload_byte_identical() -> None:
    """The unrestricted path must not gain a single key.

    This is the regression guard for every existing agent: the channel is
    opt-in, so an agent without a selection has to keep producing exactly
    ``{cwd, mcpServers}``.
    """
    params = await _capture(_client())
    assert params == {"cwd": "/w", "mcpServers": []}
    assert "_meta" not in params


@pytest.mark.asyncio
async def test_empty_selection_sends_no_meta() -> None:
    """An empty selection is the default and must be indistinguishable from it."""
    selection = parse_tool_selection([])
    assert not selection.restricts
    params = await _capture(_client(), meta=selection.acp_meta())
    assert params == {"cwd": "/w", "mcpServers": []}


@pytest.mark.asyncio
async def test_restricted_selection_disables_builtins_via_meta() -> None:
    """A deny-all allowlist must reach the adapter as ``tools: []``.

    ``tools: []`` is the SDK's documented "disable all built-in tools"; the
    allowlisted deep-worker tool is an MCP tool and travels via
    ``mcpServers``/``--mcp-config``, so it is deliberately NOT listed here.
    """
    selection = parse_tool_selection(["repo_graph_link__submit_links"])
    assert selection.restricts
    params = await _capture(_client(), meta=selection.acp_meta())
    options = params["_meta"]["claudeCode"]["options"]
    assert options["tools"] == []


@pytest.mark.asyncio
async def test_meta_shape_matches_adapter_contract() -> None:
    """The nesting must be exactly what ``acp-agent.js:4628`` dereferences."""
    selection = parse_tool_selection(["read_file"])
    meta = selection.acp_meta()
    assert set(meta) == {"claudeCode"}
    assert set(meta["claudeCode"]) == {"options"}
    assert isinstance(meta["claudeCode"]["options"]["tools"], list)


@pytest.mark.asyncio
async def test_session_load_carries_meta_too() -> None:
    """A resume must re-assert the restriction.

    ``session/load`` re-creates the query in a NEW subprocess (that is why
    the model-pin validator runs on both entry points — ROUND 8 defect #5).
    Omitting ``_meta`` here would silently restore the full built-in set for
    every agent that survived a restart.
    """
    client = _client()
    seen: dict = {}

    async def fake_call(_method: str, params: dict) -> dict:
        seen["params"] = params
        return {"result": {}}

    client._call = fake_call  # type: ignore[method-assign]
    selection = parse_tool_selection(["read_file"])
    await client.session_load(
        "sess-1", cwd="/w", mcp_servers=[], meta=selection.acp_meta()
    )
    assert seen["params"]["_meta"]["claudeCode"]["options"]["tools"] == []


@pytest.mark.asyncio
async def test_session_load_without_meta_unchanged() -> None:
    """Resume on an unrestricted agent keeps its original three keys."""
    client = _client()
    seen: dict = {}

    async def fake_call(_method: str, params: dict) -> dict:
        seen["params"] = params
        return {"result": {}}

    client._call = fake_call  # type: ignore[method-assign]
    await client.session_load("sess-1", cwd="/w", mcp_servers=[])
    assert seen["params"] == {"sessionId": "sess-1", "cwd": "/w", "mcpServers": []}


def test_claude_code_no_longer_warns_about_unsuppressable_builtins() -> None:
    """claude-code has a working channel, so the warning must be gone.

    Leaving it in place told operators to switch provider to get a hermetic
    run — advice that is now wrong, and the reason rg_linker was launched
    with built-ins live.
    """
    selection = parse_tool_selection(["repo_graph_link__submit_links"])
    assert native_tool_caveat("claude-code", selection) is None


def test_other_providers_still_warn() -> None:
    """codex and grok genuinely cannot switch their natives off."""
    selection = parse_tool_selection(["repo_graph_link__submit_links"])
    for provider in ("codex", "grok"):
        warning = native_tool_caveat(provider, selection)
        assert warning is not None
        assert provider in warning


def test_unrestricted_selection_never_warns_for_any_provider() -> None:
    """No restriction, nothing to warn about."""
    selection = parse_tool_selection([])
    for provider in ("claude-code", "codex", "grok"):
        assert native_tool_caveat(provider, selection) is None


# --------------------------------------------------------------------------
# Wiring guards.
#
# The helpers above prove ``acp_meta()`` builds the right payload and that
# the client forwards it. Neither proves the PRODUCTION path passes it: a
# test that calls the helper itself cannot see whether ``ainvoke`` does.
# These read the source so a silently-dropped kwarg fails here.
# --------------------------------------------------------------------------


def _session_calls() -> list[tuple[str, int, bool]]:
    """Every ``session_new``/``session_load`` call in chat_model + has-meta."""
    import ast
    from pathlib import Path

    import deep_worker.providers.acp.chat_model as cm

    tree = ast.parse(Path(cm.__file__).read_text())
    found: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name in ("session_new", "session_load"):
            found.append(
                (name, node.lineno, any(k.arg == "meta" for k in node.keywords))
            )
    return found


def test_every_agent_session_call_forwards_meta() -> None:
    """All agent-session opens must carry ``meta``.

    The cold-summarizer's ephemeral session is the one exception: it exists
    only to hold a history dump, runs with no MCP servers, and is closed
    immediately, so it has no tools to restrict.
    """
    calls = _session_calls()
    assert calls, "AST scan found no session calls — the scan itself is broken"
    without = [(n, ln) for n, ln, has in calls if not has]
    assert len(without) == 1, (
        f"expected exactly one meta-less call (the cold summarizer), got {without}"
    )


def test_agent_process_sets_session_meta_from_the_selection() -> None:
    """The provider attribute must actually be fed from ``_tool_selection``.

    Without this the channel is dead code: ``_session_meta`` would stay None
    for every agent and the built-ins would remain callable exactly as
    before.
    """
    import ast
    from pathlib import Path

    import deep_worker_cli.process.agent_process as ap

    source = Path(ap.__file__).read_text()
    assert "_session_meta" in source, (
        "agent_process never mentions _session_meta — the selection is not "
        "reaching the provider"
    )
    assert "acp_meta()" in source, (
        "agent_process never calls acp_meta() — nothing builds the payload"
    )
    # The attribute write and the selection read must be in one function.
    tree = ast.parse(source)
    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.dump(node)
        if "_session_meta" in body and "acp_meta" in body:
            wired = True
            break
    assert wired, "_session_meta and acp_meta() are not wired in one function"
