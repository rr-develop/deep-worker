"""Abandoning an ACP session must CLOSE it, not just cancel it.

``session/cancel`` interrupts the in-flight turn but deliberately leaves the
CLI subprocess alive — the adapter's own ``closeQueryStream`` doc is explicit
that only ``query.close()`` terminates it, and only ``session/close`` reaches
that. So every path that walks away from a session id without closing it
leaks a whole SDK process, plus the stdio MCP servers that process spawned.

Observed 2026-08-29: one ACP bridge held nine orphaned ``claude`` processes
(~4.3 GB RSS) whose sessions had ended hours earlier, each still holding its
own ``repo_graph.mcp`` server.

These tests pin the three abandonment paths and the capability gate. They
assert on the WIRE call (``session/close`` for the right id), because that is
the only thing that actually reaps the process — releasing local bookkeeping
looks identical from the outside while the leak continues.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_worker.providers.acp.acp_client import AcpClient
from deep_worker.providers.acp.chat_model import AcpChatModel
from deep_worker.providers.acp.client_state import AcpClientState
from deep_worker.providers.acp.custom_types import ThreadState
from deep_worker.providers.acp.profile import AcpProfile


THREAD = "leak_guard_agent"


def _model(client_state: AcpClientState) -> AcpChatModel:
    model = object.__new__(AcpChatModel)
    model._acp_client = MagicMock()
    model._acp_client.session_cancel = AsyncMock()
    model._acp_client.session_close = AsyncMock(return_value=True)
    model._acp_client.session_new = AsyncMock(return_value="sess-new")
    model._profile = AcpProfile(
        provider="claude-code",
        cmd=("node", "/nonexistent/index.js"),
        mcp_dispatch_style="mcp_name_prefix",
        known_builtins=frozenset({"Bash", "Read"}),
        tool_name_from_update="from_claude_code_meta",
    )
    model._client_state = client_state
    model._proxy_mcp = None
    model._tools_metadata = []
    model._usage_event_callback = None
    model._working_dir = "/tmp"
    return model


# ── the client-side gate ───────────────────────────────────────────────


def _client() -> AcpClient:
    client = object.__new__(AcpClient)
    client._session_close_supported = True
    client._call = AsyncMock(return_value={"result": {}})
    client._active_prompts = {}
    client._sinks = {}
    client._session_ready = {}
    return client


@pytest.mark.asyncio
async def test_session_close_sends_the_wire_call_and_releases_state() -> None:
    client = _client()
    client._sinks["sess-1"] = object()

    assert await client.session_close("sess-1") is True

    client._call.assert_awaited_once_with("session/close", {"sessionId": "sess-1"})
    # Local routing state goes too — a closed session must stop receiving
    # updates, or a late notification resolves against a dead subprocess.
    assert "sess-1" not in client._sinks


@pytest.mark.asyncio
async def test_close_is_skipped_when_the_adapter_never_advertised_it() -> None:
    """Capability comes from the live handshake, never a hardcoded provider
    list. An adapter without ``sessionCapabilities.close`` must not be sent a
    method it will reject — but local state is still released."""
    client = _client()
    client._session_close_supported = False
    client._sinks["sess-1"] = object()

    assert await client.session_close("sess-1") is False

    client._call.assert_not_awaited()
    assert "sess-1" not in client._sinks


@pytest.mark.asyncio
async def test_a_refusing_cli_never_breaks_the_turn() -> None:
    """Close runs on abandonment paths: the session is already logically
    gone, so a CLI that errors must be survivable, not fatal."""
    client = _client()
    client._call = AsyncMock(side_effect=RuntimeError("Session not found"))

    assert await client.session_close("sess-1") is False  # no raise
    assert "sess-1" not in client._sinks


@pytest.mark.asyncio
async def test_close_of_none_is_a_noop() -> None:
    client = _client()
    assert await client.session_close(None) is False
    client._call.assert_not_awaited()


# ── the three abandonment paths ────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_summary_reset_closes_the_abandoned_session() -> None:
    """``record_summarization`` nulls ``session_id`` and the next turn opens
    a fresh one. The pre-summary session is never used again, so it must be
    closed — this fires on every summarization, making it the highest-volume
    leak of the three."""
    state = AcpClientState()
    state._states[THREAD] = ThreadState(
        session_id="sess-doomed", session_live_in_process=True
    )
    model = _model(state)
    model._adopt_pump = AsyncMock()
    model._discard_pending_prompt = MagicMock()

    await model.areset_after_summary(thread_id=THREAD)

    model._acp_client.session_close.assert_awaited_once_with("sess-doomed")
    # And the reset itself still happened.
    assert state._states[THREAD].session_id is None


@pytest.mark.asyncio
async def test_cold_summary_closes_its_ephemeral_session() -> None:
    """The cold dump session exists only for that one prompt."""
    state = AcpClientState()
    model = _model(state)
    model._acp_client.session_new = AsyncMock(return_value="sess-cold")
    model._prompt_for_summary = AsyncMock(return_value="a summary")
    model._acp_client.wait_session_ready = AsyncMock()

    await model._summarize_cold(THREAD, [], "summarize this")

    model._acp_client.session_close.assert_awaited_once_with("sess-cold")


@pytest.mark.asyncio
async def test_cold_summary_closes_even_when_the_prompt_fails() -> None:
    """The close lives in a ``finally``: a summarization that raises must not
    keep the ephemeral subprocess alive."""
    state = AcpClientState()
    model = _model(state)
    model._acp_client.session_new = AsyncMock(return_value="sess-cold")
    model._prompt_for_summary = AsyncMock(side_effect=RuntimeError("boom"))
    model._acp_client.wait_session_ready = AsyncMock()

    with pytest.raises(RuntimeError):
        await model._summarize_cold(THREAD, [], "summarize this")

    model._acp_client.session_close.assert_awaited_once_with("sess-cold")
