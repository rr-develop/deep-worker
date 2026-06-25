"""Integration tests for A2A wiring inside ``AgentProcess`` (stage 4, RC-6/RC-7).

These exercise the real ``_maybe_start_a2a_server`` and ``_register`` methods
WITHOUT mocks of the A2A layer: a real uvicorn server is started and a real
``ProcessRegistry`` round-trips the ``a2a_port``/``a2a_url`` fields.

To avoid the heavy ``AgentProcess.__init__`` (which builds the whole agent),
we construct a bare instance via ``object.__new__`` and set only the few
attributes the two methods under test read — they are otherwise self-contained.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.registry import ProcessRegistry


def _bare_agent(name, registry, tmp_path):
    ap = object.__new__(AgentProcess)
    ap.name = name
    ap.registry = registry
    ap._a2a_handle = None
    ap.started_at = None
    ap.rpyc_port = 12345
    ap.pid_file = tmp_path / f"{name}.pid"

    class _Cfg:
        working_dir = str(tmp_path)

    ap.config = _Cfg()

    # Minimal turn engine so the inbound A2A executor has something to call.
    async def _handle_invoke(command, connection=None, notify_callback=None,
                             approval_callback=None, thread_id_override=None):
        return {"success": True,
                "response": {"role": "assistant", "content": "ok"}}

    ap._handle_invoke = _handle_invoke
    ap.cancel_current_worker_task = lambda: False
    return ap


@pytest.mark.asyncio
async def test_a2a_disabled_by_default_no_handle_no_registry_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEP_WORKER_A2A_ENABLED", raising=False)
    reg = ProcessRegistry(
        registry_file=tmp_path / "registry.json",
        lock_file=tmp_path / "registry.lock",
    )
    ap = _bare_agent("agentDisabled", reg, tmp_path)

    await ap._maybe_start_a2a_server()
    assert ap._a2a_handle is None

    await ap._register()
    info = reg.lookup("agentDisabled")
    assert info is not None
    assert getattr(info, "a2a_port", None) is None
    assert getattr(info, "a2a_url", None) is None


@pytest.mark.asyncio
async def test_a2a_enabled_starts_server_and_persists_registry_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEP_WORKER_A2A_ENABLED", "1")
    monkeypatch.setenv("DEEP_WORKER_A2A_HOST", "127.0.0.1")
    monkeypatch.setenv("DEEP_WORKER_A2A_PORT", "0")
    reg = ProcessRegistry(
        registry_file=tmp_path / "registry.json",
        lock_file=tmp_path / "registry.lock",
    )
    ap = _bare_agent("agentEnabled", reg, tmp_path)

    await ap._maybe_start_a2a_server()
    try:
        assert ap._a2a_handle is not None
        assert ap._a2a_handle.port > 0
        assert ap._a2a_handle.url.startswith("http://127.0.0.1:")

        await ap._register()
        info = reg.lookup("agentEnabled")
        assert info is not None
        assert info.a2a_port == ap._a2a_handle.port
        assert info.a2a_url == ap._a2a_handle.url

        # The advertised URL resolves to a live AgentCard over real HTTP.
        import httpx
        from a2a.client import A2ACardResolver

        async with httpx.AsyncClient(timeout=10) as hc:
            card = await A2ACardResolver(hc, info.a2a_url).get_agent_card()
        assert card.name == "agentEnabled"
    finally:
        if ap._a2a_handle is not None:
            await ap._a2a_handle.shutdown()
