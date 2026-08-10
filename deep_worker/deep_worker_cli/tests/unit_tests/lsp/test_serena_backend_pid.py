"""``SerenaBackend.pid`` — commit 10 unit coverage.

We do NOT spawn Serena here. The prod-path transport
(:class:`_OwnedStdioSession`) is exercised directly against a tiny
echo child that speaks the MCP JSON-RPC init handshake, which is
enough to prove that:

* ``backend.pid`` reports the real child's OS pid after ``start()``.
* ``backend.pid`` returns ``None`` before start and after stop.
* When the fake-session test seam (``_open_mcp_session`` monkey-patch)
  short-circuits the real transport, ``backend.pid`` stays ``None``.
"""
from __future__ import annotations

import os
import sys

import pytest

from deep_worker_cli.lsp.service.serena_backend import (
    SerenaBackend,
    _OwnedStdioSession,
)


_JSONRPC_ECHO = r"""
import json, sys

def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("method") == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0"},
                },
            }
            sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()
        elif msg.get("method") == "notifications/initialized":
            continue
        elif msg.get("method") == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": {"tools": []},
            }
            sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()

main()
"""


@pytest.mark.asyncio
async def test_owned_session_reports_child_pid() -> None:
    argv = [sys.executable, "-u", "-c", _JSONRPC_ECHO]
    sess = _OwnedStdioSession(argv)
    assert sess.pid is None
    async with sess:
        pid = sess.pid
        assert isinstance(pid, int)
        assert os.path.isdir(f"/proc/{pid}")


@pytest.mark.asyncio
async def test_backend_pid_reflects_real_child(tmp_path) -> None:
    argv = [sys.executable, "-u", "-c", _JSONRPC_ECHO]
    from deep_worker_cli.lsp.service import serena_backend as sb

    orig = sb.build_serena_spawn_argv

    def _fake_argv(_root):
        return argv

    sb.build_serena_spawn_argv = _fake_argv  # ty: ignore[invalid-assignment]
    try:
        b = SerenaBackend(tmp_path)
        try:
            await b.start()
            pid = b.pid
            assert isinstance(pid, int)
            assert os.path.isdir(f"/proc/{pid}")
        finally:
            await b.stop()
        assert b.pid is None
    finally:
        sb.build_serena_spawn_argv = orig


@pytest.mark.asyncio
async def test_backend_pid_is_none_when_fake_session_bypasses_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from deep_worker_cli.lsp.service import serena_backend as sb

    class _FakeSession:
        async def initialize(self) -> None: ...
        async def list_tools(self):
            class _L:
                tools: list = []

            return _L()

    class _CM:
        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, *_exc):
            return None

    async def _fake_open(_argv):
        return _CM()

    monkeypatch.setattr(sb, "_open_mcp_session", _fake_open)

    b = SerenaBackend(tmp_path)
    await b.start()
    try:
        assert b.pid is None
    finally:
        await b.stop()
