"""``SerenaBackend.serena_version`` — commit 12 (observability).

Design amendment A1: the version comes from the SINGLE ``initialize()``
handshake the transport already performs. We must never re-issue
``initialize()`` just to read a version — a second handshake against a
live MCP session is a protocol error.

Pinned invariants:

* ``_OwnedStdioSession.server_version`` carries ``serverInfo.version``
  from the real handshake, and ``SerenaBackend.serena_version``
  mirrors it after ``start()``.
* The version is cleared on ``stop()``: reporting the version of a
  backend that is no longer running is worse than reporting nothing.
* A peer that reports an empty ``version`` must not break the attach —
  the field degrades to ``""`` (fail-quiet, cosmetic field). Note that
  ``serverInfo`` itself is mandatory in the MCP schema, so the only
  reachable degradation is an empty/blank version string.
"""
from __future__ import annotations

import json
import sys

import pytest

from deep_worker_cli.lsp.service.serena_backend import (
    SerenaBackend,
    _OwnedStdioSession,
)


def _echo_child(server_info: dict) -> str:
    """Return source for a child speaking a minimal MCP handshake."""
    result: dict = {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": server_info,
    }
    return f"""
import json, sys

RESULT = {json.dumps(result)}

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        msg = json.loads(raw)
    except Exception:
        continue
    method = msg.get("method")
    if method == "initialize":
        out = {{"jsonrpc": "2.0", "id": msg["id"], "result": RESULT}}
    elif method == "tools/list":
        out = {{"jsonrpc": "2.0", "id": msg["id"],
                "result": {{"tools": []}}}}
    else:
        continue
    sys.stdout.write(json.dumps(out) + "\\n")
    sys.stdout.flush()
"""


@pytest.mark.asyncio
async def test_owned_session_captures_server_version() -> None:
    argv = [
        sys.executable,
        "-u",
        "-c",
        _echo_child({"name": "serena", "version": "1.6.1"}),
    ]
    sess = _OwnedStdioSession(argv)
    assert sess.server_version == ""
    async with sess:
        assert sess.server_version == "1.6.1"


@pytest.mark.asyncio
async def test_backend_exposes_and_clears_version(tmp_path) -> None:
    argv = [
        sys.executable,
        "-u",
        "-c",
        _echo_child({"name": "serena", "version": "9.9.9"}),
    ]
    from deep_worker_cli.lsp.service import serena_backend as sb

    orig = sb.build_serena_spawn_argv
    sb.build_serena_spawn_argv = lambda _root: argv  # ty: ignore[invalid-assignment]
    try:
        b = SerenaBackend(tmp_path)
        assert b.serena_version == ""
        try:
            await b.start()
            assert b.serena_version == "9.9.9"
        finally:
            await b.stop()
        assert b.serena_version == "", (
            "a stopped backend must not keep advertising its version"
        )
    finally:
        sb.build_serena_spawn_argv = orig


@pytest.mark.asyncio
async def test_blank_version_degrades_to_empty(tmp_path) -> None:
    argv = [sys.executable, "-u", "-c", _echo_child({"name": "x", "version": ""})]
    from deep_worker_cli.lsp.service import serena_backend as sb

    orig = sb.build_serena_spawn_argv
    sb.build_serena_spawn_argv = lambda _root: argv  # ty: ignore[invalid-assignment]
    try:
        b = SerenaBackend(tmp_path)
        try:
            await b.start()
            # Attach must have SUCCEEDED — a missing cosmetic field is
            # not a reason to refuse the backend.
            assert b.running is True
            assert b.serena_version == ""
        finally:
            await b.stop()
    finally:
        sb.build_serena_spawn_argv = orig
