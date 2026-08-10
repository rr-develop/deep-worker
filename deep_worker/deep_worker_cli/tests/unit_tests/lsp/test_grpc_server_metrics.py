"""Commit 12 (observability) — pin the new Health/Status metric fields.

Pinned invariants (design §3, §4):

* ``backend_uptime_seconds`` starts at 0 (backend not attached),
  becomes > 0 after a successful ``attach_backend``, and returns to
  0 on ``mark_backend_not_ready``.
* ``restarts_total`` bumps on ``note_restart(success=True)`` only,
  never on the first ``attach_backend`` (see MUT-X + first-attach
  regression in test_serena_supervisor).
* ``tool_calls_total`` / ``last_tool_call_unix`` advance ONLY when
  the handler returns ``is_error=False``. Errored calls must NOT
  count — MUT-Y.
* ``StatusReply`` mirrors all of the above plus ``serena_version``
  cached from the backend.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.lsp.service.grpc_server import (
    LSPServiceServer,
    ServerConfig,
)
from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb


def _server(tmp_path: Path) -> LSPServiceServer:
    slug = tmp_path / "slug"
    slug.mkdir(mode=0o700, exist_ok=True)
    return LSPServiceServer(
        ServerConfig(
            socket_path=slug / "grpc.sock",
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            slug=slug.name,
        )
    )


class _FakeBackend:
    """Minimal backend surface a real ``attach_backend`` will touch."""

    def __init__(self, version: str = "fake-1.0") -> None:
        self._version = version

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def set_on_lost(self, _hook) -> None:
        pass

    async def call_tool(self, *_a, **_k) -> pb.ToolResult:
        return pb.ToolResult(content_json="{}")

    async def refresh(self, _reason: str) -> None:
        pass

    @property
    def serena_version(self) -> str:
        return self._version


@pytest.mark.asyncio
async def test_health_backend_uptime_zero_when_not_ready(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    reply = await srv._servicer.Health(pb.HealthRequest(), None)
    assert reply.alive is False
    assert reply.backend_uptime_seconds == 0
    assert reply.restarts_total == 0
    assert reply.serena_version == ""


@pytest.mark.asyncio
async def test_health_includes_backend_uptime_after_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    srv = _server(tmp_path)
    ok = await srv.attach_backend(_FakeBackend(version="1.6.1"))
    assert ok is True
    # Nudge the internal timestamp into the past so we don't have to
    # sleep for the assert.
    assert srv._backend_started_ns is not None
    srv._backend_started_ns -= 5 * 1_000_000_000
    reply = await srv._servicer.Health(pb.HealthRequest(), None)
    assert reply.alive is True
    assert reply.backend_uptime_seconds >= 5
    assert reply.serena_version == "1.6.1"


@pytest.mark.asyncio
async def test_health_backend_uptime_resets_on_not_ready(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    await srv.attach_backend(_FakeBackend(version="1.6.1"))
    assert srv._backend_started_ns is not None
    # Simulate "backend died / rate limit exhausted".
    srv.mark_backend_not_ready()
    reply = await srv._servicer.Health(pb.HealthRequest(), None)
    assert reply.alive is False
    assert reply.backend_uptime_seconds == 0
    assert reply.serena_version == ""


@pytest.mark.asyncio
async def test_restarts_total_increments_on_note_restart(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    srv.note_restart(success=False)  # a failed attempt must NOT count
    srv.note_restart(success=True)
    srv.note_restart(success=True)
    reply = await srv._servicer.Health(pb.HealthRequest(), None)
    assert reply.restarts_total == 2


@pytest.mark.asyncio
async def test_tool_call_metrics_advance_on_success(tmp_path: Path) -> None:
    srv = _server(tmp_path)

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        return pb.ToolResult(content_json='{"ok":true}')

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()
    before = int(time.time())
    r1 = await srv._servicer.FindSymbol(
        pb.FindSymbolRequest(args_json="{}"), None
    )
    r2 = await srv._servicer.FindSymbol(
        pb.FindSymbolRequest(args_json="{}"), None
    )
    assert r1.result.content_json == '{"ok":true}'
    assert r2.result.content_json == '{"ok":true}'

    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.tool_calls_total == 2
    assert status.last_tool_call_unix >= before


@pytest.mark.asyncio
async def test_tool_call_metrics_ignore_errors(tmp_path: Path) -> None:
    srv = _server(tmp_path)

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        return pb.ToolResult(is_error=True, error_message="nope")

    srv.set_tool_handler(handler)
    srv.mark_backend_ready()
    _ = await srv._servicer.FindSymbol(
        pb.FindSymbolRequest(args_json="{}"), None
    )
    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.tool_calls_total == 0, (
        "errored tool call must NOT bump the success counter (MUT-Y)"
    )
    assert status.last_tool_call_unix == 0


@pytest.mark.asyncio
async def test_status_reply_carries_new_metrics(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    await srv.attach_backend(_FakeBackend(version="2.0-test"))
    srv.note_restart(success=True)

    async def handler(_rpc: str, _args: str) -> pb.ToolResult:
        return pb.ToolResult(content_json="{}")

    srv.set_tool_handler(handler)
    _ = await srv._servicer.FindSymbol(
        pb.FindSymbolRequest(args_json="{}"), None
    )

    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.serena_version == "2.0-test"
    assert status.restarts_total == 1
    assert status.tool_calls_total == 1
    assert status.backend_uptime_seconds >= 0
    assert status.last_tool_call_unix > 0


@pytest.mark.asyncio
async def test_transient_retries_total_in_status(tmp_path: Path) -> None:
    """Transient LSP cancellations we rode out are counted (§4.5).

    Unlike restarts, EVERY retry counts: the number is what turns
    "the language server probably reindexes a lot" from a hypothesis
    into something measurable.
    """
    srv = _server(tmp_path)
    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.transient_retries_total == 0

    srv.note_transient_retry()
    srv.note_transient_retry()
    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.transient_retries_total == 2


@pytest.mark.asyncio
async def test_ls_terminated_retries_total_in_status(tmp_path: Path) -> None:
    """Language-server deaths we replayed past are counted (§4.6).

    This is the number that tells "the LS keeps crashing but agents
    never noticed" apart from "everything is fine".
    """
    srv = _server(tmp_path)
    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.ls_terminated_retries_total == 0

    srv.note_ls_terminated_retry()
    srv.note_ls_terminated_retry()
    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.ls_terminated_retries_total == 2


@pytest.mark.asyncio
async def test_attach_backend_wires_transient_retry_hook(tmp_path: Path) -> None:
    """The counter is useless unless attach_backend actually wires it."""
    srv = _server(tmp_path)

    class _HookRecordingBackend(_FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.transient_hook = None

        def set_on_transient_retry(self, hook) -> None:
            self.transient_hook = hook

    backend = _HookRecordingBackend()
    assert await srv.attach_backend(backend) is True
    assert backend.transient_hook is not None, (
        "attach_backend must install the transient-retry hook"
    )
    backend.transient_hook()
    status = await srv._servicer.ProjectStatus(pb.StatusRequest(), None)
    assert status.transient_retries_total == 1
