"""Proxy-MCP correlation must not burn 10s per tool call on CLIs that never
execute our tools over MCP.

Measured 2026-08-24 on grok CLI 1.0.5 (live, two agents): every deep-worker
tool call cost a flat extra ~10s. Our proxy served ZERO ``CallToolRequest``
for the whole session — grok LISTS our tools over MCP but DISPATCHES them on
the ACP ``tool_call`` channel (``use_tool``) and takes the result back the
same way. ``_correlate_proxy_tool_calls`` nevertheless waited the full
10s for a parked MCP handler that, by construction, was never going to
appear; the tool then ran through LangGraph's ToolNode as usual, so the only
effect was dead latency on every single call (send turn: 30s wall, 20s of it
model time).

The correlation exists only to map an ACP toolCallId onto a PARKED MCP
handler id so ``resolve_pending_tool`` can unpark it. If the CLI has never
parked anything, there is nothing to map. So the wait is adaptive: the full
budget once this process has actually served an MCP ``tools/call``, a short
probe before that. A CLI that does use the MCP execution path parks the
handler essentially simultaneously with the ACP notification, so the probe
is ample; the full budget returns as soon as the first call is served.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from deep_worker.providers.acp.chat_model import (
    PROXY_CORRELATION_PROBE_SEC,
    PROXY_CORRELATION_TIMEOUT_SEC,
    AcpChatModel,
)
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import ThreadState
from deep_worker.providers.acp.profile import AcpProfile

PROXY = "deep_worker_acp_proxy_555"


class _RecordingProxy:
    """Proxy stub that records the timeout the correlation asked for."""

    def __init__(self, *, served: bool, pending: list | None = None) -> None:
        self.server_name = PROXY
        self.served_tool_call = served
        self._pending = pending or []
        self.timeouts: list[float] = []

    async def wait_for_pending_calls(self, *, tool_name, arguments, timeout):
        self.timeouts.append(timeout)
        if self._pending:
            return self._pending
        await asyncio.sleep(0)
        raise asyncio.TimeoutError


def _profile() -> AcpProfile:
    return AcpProfile(
        provider="grok",
        cmd=("grok", "agent", "stdio"),
        mcp_dispatch_style="dispatch_builtin",
        tool_name_from_update="from_title",
    )


def _model(proxy: _RecordingProxy) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._profile = _profile()
    model._proxy_mcp = proxy
    model._client_state = AcpClientState()
    model._tools_metadata = [
        {"name": "send_message_to_agent", "description": "s", "input_schema": {}}
    ]
    return model


_TC = [{"id": "acp-1", "name": "send_message_to_agent", "args": {"a": 1}}]


@pytest.mark.asyncio
async def test_probe_is_short_when_cli_never_executed_over_mcp() -> None:
    proxy = _RecordingProxy(served=False)
    model = _model(proxy)
    await model._correlate_proxy_tool_calls(ThreadState(), list(_TC))
    assert proxy.timeouts == [PROXY_CORRELATION_PROBE_SEC]
    assert PROXY_CORRELATION_PROBE_SEC <= 1.0


@pytest.mark.asyncio
async def test_full_budget_once_an_mcp_call_has_been_served() -> None:
    proxy = _RecordingProxy(served=True)
    model = _model(proxy)
    await model._correlate_proxy_tool_calls(ThreadState(), list(_TC))
    assert proxy.timeouts == [PROXY_CORRELATION_TIMEOUT_SEC]
    assert PROXY_CORRELATION_TIMEOUT_SEC >= 10.0


@pytest.mark.asyncio
async def test_probe_miss_is_not_warned_but_full_budget_miss_is(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A probe miss is EXPECTED traffic on an ACP-executing CLI — warning on
    it produced 69 warnings in three turns and buried real problems."""
    model = _model(_RecordingProxy(served=False))
    with caplog.at_level(logging.WARNING):
        await model._correlate_proxy_tool_calls(ThreadState(), list(_TC))
    assert "no pending MCP handler" not in caplog.text

    caplog.clear()
    model = _model(_RecordingProxy(served=True))
    with caplog.at_level(logging.WARNING):
        await model._correlate_proxy_tool_calls(ThreadState(), list(_TC))
    assert "no pending MCP handler" in caplog.text


@pytest.mark.asyncio
async def test_correlation_still_maps_a_parked_handler() -> None:
    """The mapping itself is untouched: a handler that IS parked must still
    be correlated, or resolve_pending_tool cannot unpark it."""
    pending = [{"id": "handler-9", "name": "send_message_to_agent", "args": {"a": 1}}]
    proxy = _RecordingProxy(served=True, pending=pending)
    model = _model(proxy)
    state = ThreadState()
    await model._correlate_proxy_tool_calls(state, list(_TC))
    assert state.proxy_call_id_by_acp_id == {"acp-1": "handler-9"}


def test_proxy_exposes_the_served_flag_and_starts_false() -> None:
    from deep_worker.providers.acp.proxy_mcp_server import ProxyMCPServer

    proxy = object.__new__(ProxyMCPServer)
    assert ProxyMCPServer.served_tool_call is False
    assert proxy.served_tool_call is False


@pytest.mark.asyncio
async def test_serving_a_tool_call_flips_the_flag() -> None:
    """The flag must be set by the REAL handler path, not only by name."""
    from deep_worker.providers.acp.proxy_mcp_server import ProxyMCPServer

    proxy = ProxyMCPServer(tools_metadata=[], long_running_wait_seconds=0.05)
    assert proxy.served_tool_call is False
    await proxy._handle_tool_call("send_message_to_agent", {})
    assert proxy.served_tool_call is True
