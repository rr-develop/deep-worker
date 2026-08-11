"""Regression: ``ClientManager.get_or_create_client`` cache-hit MUST
detect a tool_set mismatch and rotate the session — never serve a
client whose ``ProxyMCPServer`` advertises a stale tool inventory.

Background — incident 2026-06-13 (#2)
=====================================
``get_or_create_client`` historically did:

    if thread_id in self.clients:
        return self.clients[thread_id]   # ← ignores tools_metadata!

That is the **third** failure mode in the MCP-lost cluster: the very
first turn after agent start could cache a ``ProxyMCPServer`` built
from a TOO-SMALL tool list (e.g. 7 ``web_tools`` instead of the final
23-with-middleware). Every subsequent turn — even one that arrived
through a properly bound model with the full 23 — silently inherited
the stale 4/7-tool proxy because of the unconditional cache return.

After the prewarm fix (incident 2026-06-13 #2, step B) the FIRST
``get_or_create_client`` happens with the FINAL tool inventory and the
cache is correct from the start. But that fix alone leaves a foot-gun:
any future code path that calls ``get_or_create_client`` with a
DIFFERENT ``tools_metadata`` will silently get the stale proxy.

Contract enforced here
======================
1. **Cache HIT with MATCHING tool_set** → return cached pair (status-quo
   for normal turns). Connect MUST NOT be called again.
2. **Cache HIT with MISMATCHING tool_set** → rotate the session: tear
   down the stale client, build a NEW ``ProxyMCPServer`` with the
   incoming ``tools_metadata``, store it in the cache, return the new
   pair. The next-turn-after-prewarm regression is impossible after
   this.
3. **First call (miss)** → create+cache as today.

⚠ DO NOT replace the mismatch path with a silent "use cached anyway":
that is the EXACT pre-fix behaviour the incident-2026-06-13 (#2)
post-mortem cites as the cluster's defence-in-depth gap.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deep_worker.providers.claude_code_provider.client_manager import ClientManager
from deep_worker.providers.claude_code_provider.custom_types import ToolMetadata


def _md(name: str) -> ToolMetadata:
    """Compact ToolMetadata factory for assertion clarity."""
    return {
        "name": name,
        "description": f"desc:{name}",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


@pytest.mark.asyncio
async def test_cache_hit_with_matching_toolset_is_a_real_cache_hit() -> None:
    """No mismatch → no rotation. Backwards-compatible with the
    pre-incident cache behaviour for the normal turn-after-turn case.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    client_mock = Mock()
    client_mock.connect = AsyncMock()
    client_mock.disconnect = AsyncMock()

    tools = [_md("alpha"), _md("beta")]
    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        return_value=client_mock,
    ):
        c1, p1 = await manager.get_or_create_client(
            thread_id="t1", tools_metadata=tools
        )
        c2, p2 = await manager.get_or_create_client(
            thread_id="t1", tools_metadata=tools
        )

    assert c1 is c2 and p1 is p2, "Identical tool_set MUST hit the cache"
    assert client_mock.connect.call_count == 1, (
        "Identical tool_set MUST NOT cause a reconnect"
    )


@pytest.mark.asyncio
async def test_cache_hit_with_mismatching_toolset_rotates() -> None:
    """The CORE invariant — incident 2026-06-13 (#2) step C.

    Second call arrives with a SUPERSET of tools (this is the EXACT
    production-restart shape: prewarm-7 then turn-23). The manager MUST
    rotate: old client disconnect, new client with the new tool_set,
    cache slot replaced.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    # Two distinct mocks so we can assert WHICH one is returned each
    # time. The factory returns first then second on subsequent
    # constructions.
    client_old = Mock(); client_old.connect = AsyncMock(); client_old.disconnect = AsyncMock()
    client_new = Mock(); client_new.connect = AsyncMock(); client_new.disconnect = AsyncMock()
    seq = [client_old, client_new]
    def factory(*_a: Any, **_kw: Any) -> Mock:
        return seq.pop(0)

    tools_small = [_md("alpha"), _md("beta")]
    tools_full = [_md("alpha"), _md("beta"), _md("gamma"), _md("delta")]

    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        side_effect=factory,
    ):
        c1, p1 = await manager.get_or_create_client(
            thread_id="t1", tools_metadata=tools_small
        )
        c2, p2 = await manager.get_or_create_client(
            thread_id="t1", tools_metadata=tools_full
        )

    assert c1 is client_old
    assert c2 is client_new, (
        "Mismatching tool_set MUST cause rotation — incident-06-13 (#2) "
        "step C contract. Got the cached old client instead."
    )
    assert c1 is not c2
    assert p1 is not p2
    # Old client must be cleanly disconnected on rotation.
    client_old.disconnect.assert_awaited()
    # New proxy advertises the FULL tool set.
    assert sorted(p2.tool_names) == sorted(t["name"] for t in tools_full)
    # Cache slot now points at the NEW pair.
    assert manager.clients["t1"] == (c2, p2)


@pytest.mark.asyncio
async def test_cache_hit_mismatch_when_tool_subset_also_rotates() -> None:
    """Symmetric case: incoming set is SMALLER. Still a mismatch, still
    rotates. We do not assume "shrinkage is OK" — the contract is
    "incoming tool_set IS the new truth".
    """
    from claude_agent_sdk import ClaudeAgentOptions

    manager = ClientManager(
        base_options=ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    )

    client_old = Mock(); client_old.connect = AsyncMock(); client_old.disconnect = AsyncMock()
    client_new = Mock(); client_new.connect = AsyncMock(); client_new.disconnect = AsyncMock()
    seq = [client_old, client_new]
    def factory(*_a: Any, **_kw: Any) -> Mock:
        return seq.pop(0)

    tools_full = [_md("alpha"), _md("beta"), _md("gamma")]
    tools_shrunk = [_md("alpha")]

    with patch(
        "claude_code_provider.client_manager.ClaudeSDKClient",
        side_effect=factory,
    ):
        c1, _ = await manager.get_or_create_client(
            thread_id="t1", tools_metadata=tools_full
        )
        c2, p2 = await manager.get_or_create_client(
            thread_id="t1", tools_metadata=tools_shrunk
        )

    assert c1 is not c2, "Shrinkage MUST also be honoured — no silent reuse"
    assert sorted(p2.tool_names) == ["alpha"]
