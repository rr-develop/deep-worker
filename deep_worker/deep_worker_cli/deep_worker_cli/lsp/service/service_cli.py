"""CLI wiring for the shared per-project LSP service.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §15 commit 4.

Public surface (registered on ``deep-worker lsp`` by
``cli/lsp_command.py``)::

    deep-worker lsp serve          [--project-root DIR]
    deep-worker lsp stop           [--project-root DIR]
    deep-worker lsp service-status [--project-root DIR]
    deep-worker lsp __serve_impl__ --project-root DIR --socket PATH \
                                   [--idle-shutdown SECONDS]

``__serve_impl__`` is private: it is the process the supervisor
launches, via the throwaway launcher in ``supervisor._spawn``. Users
never type it. It is the ONLY entrypoint that runs the gRPC server
in-process, and it lives inside the allowlisted package
``deep_worker_cli.lsp.service*`` (policy §8) — no other module is
permitted to spawn the backend.

``serve``/``stop``/``service-status`` are thin clients: they touch only
the registry + supervisor, never import grpc, and therefore stay cheap
enough to run on every agent start (auto-start path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from deep_worker_cli.lsp.service.registry import ServiceRegistry
from deep_worker_cli.lsp.service.supervisor import (
    ServiceStartTimeout,
    ServiceSupervisor,
)

_SOCK_FILENAME: Final = "grpc.sock"
_SERVE_MODULE: Final = "deep_worker_cli.lsp.service"
_DEFAULT_IDLE_SHUTDOWN_S: Final = 1800.0  # 30 min, design §16.7


# ── argv construction (production spawn_argv) ────────────────────────


def build_serve_argv(project_root: Path, slug_dir: Path) -> list[str]:
    """Return the argv the supervisor should exec for a new service.

    Uses ``sys.executable -m deep_worker_cli.lsp.service`` rather than
    the ``deep-worker`` console script: a detached daemon must not
    depend on a shim being on ``PATH`` at spawn time (the agent may
    have been launched from a venv whose ``bin`` is not exported).
    """
    return [
        sys.executable,
        "-m",
        _SERVE_MODULE,
        "__serve_impl__",
        "--project-root",
        str(project_root.resolve()),
        "--socket",
        str(slug_dir / _SOCK_FILENAME),
    ]


# ── helpers shared by the three thin commands ────────────────────────


def _resolve_root(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "project_root", None)
    if explicit:
        return Path(explicit).resolve()
    from deep_worker_cli.cli.cwd import invocation_cwd
    from deep_worker_cli.lsp.serena_config import resolve_workspace_root

    return resolve_workspace_root(invocation_cwd()).resolve()


def _registry(args: argparse.Namespace) -> ServiceRegistry:
    base = getattr(args, "registry_dir", None)
    return ServiceRegistry(base_dir=Path(base) if base else None)


# ── thin client commands ─────────────────────────────────────────────


def cmd_serve(args: argparse.Namespace) -> int:
    root = _resolve_root(args)
    registry = _registry(args)
    sup = ServiceSupervisor(registry, lambda d: build_serve_argv(root, d))
    try:
        entry = sup.ensure_started(root)
    except ServiceStartTimeout as e:
        print(f"error: LSP service for {root} did not become ready: {e}")
        return 1
    print(f"LSP service ready: pid={entry.pid} socket={entry.socket_path}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    root = _resolve_root(args)
    registry = _registry(args)
    sup = ServiceSupervisor(registry, lambda d: build_serve_argv(root, d))
    sup.stop(root)
    print(f"LSP service for {root}: stopped")
    return 0


def cmd_service_status(args: argparse.Namespace) -> int:
    root = _resolve_root(args)
    registry = _registry(args)
    entry = registry.read(root)
    if entry is None:
        print(f"LSP service for {root}: not running")
        return 1
    if registry.is_stale(entry):
        print(f"LSP service for {root}: stale entry (pid {entry.pid} is gone)")
        return 1
    if not Path(entry.socket_path).exists():
        print(f"LSP service for {root}: registered but socket missing")
        return 1
    print(
        f"LSP service for {root}: running\n"
        f"  pid      : {entry.pid}\n"
        f"  slug     : {registry.slug_dir(root).name}\n"
        f"  transport: {entry.transport}\n"
        f"  socket   : {entry.socket_path}"
    )
    # A6: the registry block above is printed unconditionally — it is
    # the only part that works without grpc. The live metrics come
    # from the service itself and may legitimately fail (socket left
    # behind by a wedged process), which is itself a status worth
    # reporting, hence exit 1 rather than a silent success.
    try:
        metrics = _query_metrics(entry.socket_path)
    except Exception as e:  # noqa: BLE001 — reachability IS the result
        print(f"  unreachable: {e}")
        return 1
    print(
        f"  backend  : alive={metrics['alive']} "
        f"serena={metrics['serena_version'] or '?'} "
        f"uptime={metrics['backend_uptime_seconds']}s "
        f"restarts={metrics['restarts_total']}\n"
        f"  metrics  : clients={metrics['attached_clients']} "
        f"tool_calls={metrics['tool_calls_total']} "
        f"transient_retries={metrics['transient_retries_total']} "
        f"ls_terminated_retries={metrics['ls_terminated_retries_total']} "
        f"last_tool_call_unix={metrics['last_tool_call_unix']} "
        f"last_refresh_unix={metrics['last_refresh_unix']}"
    )
    return 0


def _query_metrics(socket_path: str) -> dict[str, object]:
    """Fetch Health + ProjectStatus over the service's UDS.

    Imports grpc lazily: ``service-status`` must stay importable (and
    cheap) on the auto-start path, which never needs grpc.
    """
    import asyncio  # noqa: PLC0415

    from grpc import aio as grpc_aio  # noqa: PLC0415

    from deep_worker_cli.lsp.service.proto import (  # noqa: PLC0415
        lsp_service_pb2 as pb,
        lsp_service_pb2_grpc as pbg,
    )

    async def _run() -> dict[str, object]:
        channel = grpc_aio.insecure_channel(f"unix:{socket_path}")
        try:
            stub = pbg.LspServiceStub(channel)
            health = await stub.Health(pb.HealthRequest(), timeout=5.0)
            status = await stub.ProjectStatus(pb.StatusRequest(), timeout=5.0)
        finally:
            await channel.close()
        return {
            "alive": health.alive,
            "serena_version": status.serena_version or health.serena_version,
            "backend_uptime_seconds": status.backend_uptime_seconds,
            "restarts_total": status.restarts_total,
            "attached_clients": status.attached_clients,
            "tool_calls_total": status.tool_calls_total,
            "last_tool_call_unix": status.last_tool_call_unix,
            "last_refresh_unix": status.last_refresh_unix,
            "transient_retries_total": status.transient_retries_total,
            "ls_terminated_retries_total": (
                status.ls_terminated_retries_total
            ),
        }

    return asyncio.run(_run())


def cmd_bridge(args: argparse.Namespace) -> int:
    """Ensure the shared service is up, then serve MCP over stdio.

    Auto-start lives here: an agent launching the bridge is exactly the
    moment we know the project is in use. There is no stdio fallback —
    a failure to reach the service is terminal and loud (policy §8).
    """
    from deep_worker_cli.lsp.bridge import BridgeUnavailable, connect_or_fail

    root = _resolve_root(args)
    reg_dir = getattr(args, "registry_dir", None)
    try:
        entry = connect_or_fail(
            root, registry_dir=Path(reg_dir) if reg_dir else None
        )
    except BridgeUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    import asyncio

    from deep_worker_cli.lsp.bridge_loop import BridgeSession
    from deep_worker_cli.lsp.bridge_stdio import serve_stdio

    async def _run() -> None:
        session = BridgeSession(entry.socket_path)
        await session.connect()
        try:
            await serve_stdio(session)
        finally:
            await session.close()

    try:
        asyncio.run(_run())
    except Exception as e:  # noqa: BLE001 — must surface, not degrade
        # No stdio fallback: if we cannot serve the shared service we
        # exit non-zero so the agent sees a tool-less session as a
        # failure rather than silently losing navigation.
        print(f"error: bridge session failed: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Rewrite a legacy per-agent serena entry to launch the bridge."""
    from deep_worker_cli.cli.lsp_command import _config_path
    from deep_worker_cli.lsp.migrate import migrate_mcp_config

    path = _config_path(args)
    # Same rule as ``lsp enable``: the USER layer is shared by every
    # agent, so it gets the "own working directory" form rather than
    # whatever literal the legacy entry happened to name.
    follow_agent_cwd = getattr(args, "scope", "project") == "user"
    if migrate_mcp_config(path, follow_agent_cwd=follow_agent_cwd):
        root_note = (
            "each agent's own working directory"
            if follow_agent_cwd
            else "the entry's project root"
        )
        print(
            f"migrated {path} → bridge ({root_note}; backup at {path}.bak)"
        )
        return 0
    print(f"{path}: nothing to migrate")
    return 0


# ── private daemon entrypoint ────────────────────────────────────────


def cmd_serve_impl(args: argparse.Namespace) -> int:
    """Run the gRPC server in THIS process until idle-shutdown/SIGTERM.

    Imported lazily so the thin client path never pays the grpc import
    cost (it runs on every agent start).
    """
    import asyncio

    from deep_worker_cli.lsp.service.grpc_server import (
        LSPServiceServer,
        ServerConfig,
    )

    socket_path = Path(args.socket)
    project_root = Path(args.project_root).resolve()
    idle = float(getattr(args, "idle_shutdown", None) or _DEFAULT_IDLE_SHUTDOWN_S)

    server = LSPServiceServer(
        ServerConfig(
            socket_path=socket_path,
            project_root=project_root,
            idle_shutdown_s=idle,
            slug=socket_path.parent.name,
        )
    )

    async def _run() -> None:
        import signal

        from deep_worker_cli.lsp.service.serena_backend import SerenaBackend
        from deep_worker_cli.lsp.service.serena_supervisor import (
            SerenaSupervisor,
        )

        await server.start()
        # Commit 10: readiness AND auto-restart of Serena are driven
        # by SerenaSupervisor. It does its own attach_backend on start,
        # arms a pidfd monitor on the child pid, and re-attaches a
        # freshly built SerenaBackend on death (≤3 attempts / 60s).
        # A1 invariant preserved: if the first attach fails, readiness
        # stays False and the service keeps serving admin RPCs.
        supervisor = SerenaSupervisor(
            server=server,
            backend_factory=lambda: SerenaBackend(project_root),
        )
        server.set_backend_owner(supervisor)
        await supervisor.start()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(server.stop(grace=1.0))
            )
        await server.wait_closed()

    asyncio.run(_run())
    return 0


# ── parser registration (called from cli/lsp_command.py) ─────────────


def register_service_subcommands(lsp_sub) -> None:
    """Add serve/stop/service-status/__serve_impl__ to ``lsp``."""
    for name, help_text in (
        ("serve", "Start (or reuse) the shared LSP service for a project"),
        ("stop", "Stop the shared LSP service for a project"),
        ("service-status", "Show shared LSP service state for a project"),
    ):
        p = lsp_sub.add_parser(name, help=help_text)
        p.add_argument(
            "--project-root",
            default=None,
            help="workspace root (default: git root of cwd)",
        )
        p.add_argument(
            "--registry-dir",
            default=None,
            help=argparse.SUPPRESS,  # test hook
        )

    bridge = lsp_sub.add_parser(
        "bridge", help="Run the stdio MCP bridge to the shared LSP service"
    )
    bridge.add_argument(
        "--project-root",
        default=None,
        help="workspace root (default: git root of cwd)",
    )
    bridge.add_argument("--registry-dir", default=None, help=argparse.SUPPRESS)

    mig = lsp_sub.add_parser(
        "migrate", help="Rewrite a legacy per-agent serena entry to the bridge"
    )
    mig.add_argument(
        "--scope", default="project", help="project (default) | user | agent:<name>"
    )
    mig.add_argument("--config", default=None, help="explicit path to mcp.json")

    impl = lsp_sub.add_parser("__serve_impl__", help=argparse.SUPPRESS)
    impl.add_argument("--project-root", required=True)
    impl.add_argument("--socket", required=True)
    impl.add_argument("--idle-shutdown", default=None, type=float)


def dispatch_service_command(args: argparse.Namespace) -> int | None:
    """Return an exit code if ``args`` targets a service subcommand."""
    command = getattr(args, "lsp_command", None)
    if command == "serve":
        return cmd_serve(args)
    if command == "stop":
        return cmd_stop(args)
    if command == "service-status":
        return cmd_service_status(args)
    if command == "__serve_impl__":
        return cmd_serve_impl(args)
    if command == "bridge":
        return cmd_bridge(args)
    if command == "migrate":
        return cmd_migrate(args)
    return None
