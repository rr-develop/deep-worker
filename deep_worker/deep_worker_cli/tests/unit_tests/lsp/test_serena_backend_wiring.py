"""Units for wiring SerenaBackend into LSPServiceServer / cmd_serve_impl.

See ``docs/DESIGN_LSP_SERENA_BACKEND_ATTACH.md`` A1–A3, A6.

These tests do not start a real serena — they pin:

* A1: `cmd_serve_impl` no longer calls `mark_backend_ready()`
  unconditionally (otherwise Health.alive=True without a backend is a lie);
* A2: a `SerenaBackend.start()` failure does not kill the service; the
  administrative surface keeps running with alive=False;
* A3: when the backend is lost, `mark_backend_not_ready()` returns
  Health.alive to False.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


async def test_server_without_backend_is_not_ready(tmp_path: Path) -> None:
    """A1: without a backend Health.alive must be False.

    Before this commit ``cmd_serve_impl`` called ``mark_backend_ready()``
    right after ``start()``, which was honest on commit 4 (when there was
    no backend at all) and became a lie as soon as the backend appeared.
    The test catches the regression "unconditional mark_backend_ready
    came back".
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb

    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            watch_enabled=False,
        )
    )
    await server.start()
    try:
        reply = await server._servicer.Health(pb.HealthRequest(), None)
        assert reply.alive is False, (
            "server without backend reports alive=True — "
            "regression of A1 (unconditional mark_backend_ready)"
        )
    finally:
        await server.stop(grace=0.1)


async def test_mark_backend_not_ready_flips_health_alive_off(
    tmp_path: Path,
) -> None:
    """A3: after losing the backend Health.alive must be False."""
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb

    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            watch_enabled=False,
        )
    )
    await server.start()
    try:
        server.mark_backend_ready()
        alive_before = (await server._servicer.Health(pb.HealthRequest(), None)).alive
        assert alive_before is True

        server.mark_backend_not_ready()
        alive_after = (await server._servicer.Health(pb.HealthRequest(), None)).alive
        assert alive_after is False, (
            "mark_backend_not_ready did not flip Health.alive back off"
        )
    finally:
        await server.stop(grace=0.1)


def test_cmd_serve_impl_does_not_mark_ready_unconditionally() -> None:
    """A1 + MUT-J (commit 9) still holds under commit 10 wiring.

    Historic bug: ``cmd_serve_impl`` used to flip ``mark_backend_ready()``
    unconditionally, so ``Health.alive`` reported True over a dead
    backend. Commit 9 killed that; commit 10 wires the backend through
    ``SerenaSupervisor``, whose ``start()`` calls ``attach_backend``
    (which internally is the ONLY place ``mark_backend_ready`` may
    fire). This test pins both invariants:

    * ``service_cli.py`` never calls ``mark_backend_ready(...)``.
    * The backend is bootstrapped through the supervisor's ``start()``.
    """
    from deep_worker_cli.lsp.service import service_cli

    src = Path(service_cli.__file__).read_text()
    unconditional = [
        l for l in src.splitlines()
        if "mark_backend_ready(" in l and not l.lstrip().startswith("#")
    ]
    assert not unconditional, (
        f"service_cli.py has an unconditional mark_backend_ready call — "
        f"A1 regression: {unconditional}"
    )
    assert "SerenaSupervisor(" in src, (
        "cmd_serve_impl must own the backend through SerenaSupervisor"
    )
    assert "set_backend_owner(supervisor)" in src, (
        "supervisor must be handed to the server so shutdown delegates "
        "backend teardown to it (design commit 10 §6)"
    )
    assert "await supervisor.start()" in src, (
        "supervisor's start() is the ONLY place that runs attach_backend "
        "in the prod path"
    )


async def test_transport_error_flips_health_alive_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A3 (real): Serena's pipe died → the next call_tool call gets
    is_error, AND Health.alive must become False.

    Without this, A3 remains dead code: mark_backend_not_ready() exists,
    but nobody calls it in prod, and the client would see a knowingly
    false alive=True on a dead backend.

    We check on a live server plus a backend with a fake MCP session that
    fails session.call_tool. No server mocks — the full gRPC surface
    works.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb
    from deep_worker_cli.lsp.service.serena_backend import SerenaBackend

    # A backend with a fake session that fails on call_tool.
    class _DyingSession:
        async def initialize(self) -> None: return None
        async def list_tools(self):  # type: ignore[no-untyped-def]
            from types import SimpleNamespace
            return SimpleNamespace(tools=[SimpleNamespace(name="find_symbol")])
        async def call_tool(self, name, arguments):  # type: ignore[no-untyped-def]
            raise BrokenPipeError("pipe closed — Serena died")

    from deep_worker_cli.lsp.service import serena_backend as sb

    class _CM:
        async def __aenter__(self): return _DyingSession()
        async def __aexit__(self, *exc): return None

    async def _fake_open(_argv): return _CM()

    monkeypatch.setattr(sb, "_open_mcp_session", _fake_open, raising=True)

    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            watch_enabled=False,
        )
    )
    await server.start()
    try:
        backend = SerenaBackend(tmp_path)
        ok = await server.attach_backend(backend)
        assert ok is True, "backend should have started (initialize/list_tools succeeded)"

        alive_before = (
            await server._servicer.Health(pb.HealthRequest(), None)
        ).alive
        assert alive_before is True, "attach_backend did not flip alive on"

        # The first tool call fails with BrokenPipeError on the session side.
        result = await backend.call_tool("FindSymbol", "{}")
        assert result.is_error is True
        assert result.error_message, "fail-loud requires non-empty error_message"
        assert backend.running is False, (
            "backend still reports running after transport death — "
            "A3 not wired"
        )

        alive_after = (
            await server._servicer.Health(pb.HealthRequest(), None)
        ).alive
        assert alive_after is False, (
            "Health.alive stayed True after backend transport death — "
            "A3 hook (mark_backend_not_ready) not called from the "
            "production error path"
        )
    finally:
        await server.stop(grace=0.1)


async def test_serve_impl_survives_backend_start_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A2: if SerenaBackend.start() fails, cmd_serve_impl does not
    crash; the service keeps listening; Health.alive=False; tool_handler=None.

    Otherwise the registry would point at the pid of a "live" service
    that does not exist — a typical trap hiding startup errors.
    """
    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )
    from deep_worker_cli.lsp.service.proto import lsp_service_pb2 as pb
    from deep_worker_cli.lsp.service import serena_backend as sb

    async def _boom(_argv):
        raise RuntimeError("serena not installed")

    monkeypatch.setattr(sb, "_open_mcp_session", _boom, raising=True)

    sock = tmp_path / "grpc.sock"
    server = LSPServiceServer(
        ServerConfig(
            socket_path=sock,
            project_root=tmp_path,
            idle_shutdown_s=60.0,
            watch_enabled=False,
        )
    )
    await server.start()
    try:
        backend = sb.SerenaBackend(tmp_path)
        # attach_backend swallows start failure by design (A2):
        outcome = await server.attach_backend(backend)
        assert outcome is False, "attach_backend should report failure via False"

        # Administrative RPCs stay alive:
        reply = await server._servicer.Health(pb.HealthRequest(), None)
        assert reply.alive is False, (
            "server marked itself ready despite backend start failure"
        )
        # tool RPCs return UNIMPLEMENTED (fail-loud), not an empty ToolResult
        # — the §5 regression point.
        assert server._tool_handler is None
    finally:
        await server.stop(grace=0.1)
