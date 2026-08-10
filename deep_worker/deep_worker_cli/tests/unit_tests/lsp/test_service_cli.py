"""Unit tests for ``deep-worker lsp serve|stop|service-status`` wiring.

Covers DESIGN_LSP_SHARED_SERVICE.md §15 commit 4:

* production ``spawn_argv`` builds a detached self-exec argv that runs
  the private ``__serve_impl__`` entrypoint (never Serena directly, and
  never from outside ``deep_worker_cli.lsp.service*``);
* the argv is absolute and carries the resolved project root + socket;
* ``lsp serve`` is idempotent (delegates to ``ensure_started``);
* ``lsp stop`` is idempotent even with no service running;
* ``lsp service-status`` reports both the running and the absent case
  without raising and with distinct exit codes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.registry import ServiceRegistry
from deep_worker_cli.lsp.service.service_cli import (
    build_serve_argv,
    cmd_service_status,
    cmd_serve,
    cmd_stop,
)


@pytest.fixture()
def registry(tmp_path: Path) -> ServiceRegistry:
    return ServiceRegistry(base_dir=tmp_path / "registry")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    return root


def _ns(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ── spawn_argv ───────────────────────────────────────────────────────


def test_serve_argv_uses_private_impl_entrypoint(project: Path, tmp_path: Path):
    slug_dir = tmp_path / "slug"
    argv = build_serve_argv(project, slug_dir)
    assert "__serve_impl__" in argv, argv
    # Must go through the deep-worker CLI, not a bare serena exec.
    assert "serena" not in " ".join(argv).lower()


def test_serve_argv_is_absolute_and_carries_root_and_socket(
    project: Path, tmp_path: Path
):
    slug_dir = tmp_path / "slug"
    argv = build_serve_argv(project, slug_dir)
    assert Path(argv[0]).is_absolute(), argv[0]
    joined = " ".join(argv)
    assert str(project.resolve()) in joined
    assert str(slug_dir / "grpc.sock") in joined


def test_serve_argv_prefers_current_interpreter_module_form(
    project: Path, tmp_path: Path
):
    """Guard against PATH-dependent resolution: a detached daemon must
    not rely on a ``deep-worker`` shim being on PATH at spawn time."""
    argv = build_serve_argv(project, tmp_path / "slug")
    assert argv[0] == sys.executable
    assert argv[1] == "-m"


# ── serve ────────────────────────────────────────────────────────────


def test_cmd_serve_starts_and_is_idempotent(
    project: Path, registry: ServiceRegistry, monkeypatch
):
    calls: list[Path] = []

    class _FakeSup:
        def __init__(self, *a, **kw) -> None:
            pass

        def ensure_started(self, root: Path):
            calls.append(root)
            from deep_worker_cli.lsp.service.registry import RegistryEntry

            return RegistryEntry(
                project_root=root,
                pid=4242,
                transport="uds",
                socket_path=str(project / "grpc.sock"),
                started_at_ns=1,
            )

    monkeypatch.setattr(
        "deep_worker_cli.lsp.service.service_cli.ServiceSupervisor", _FakeSup
    )
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_serve(args) == 0
    assert cmd_serve(args) == 0
    assert len(calls) == 2  # idempotency is the supervisor's contract


def test_cmd_serve_reports_start_timeout_as_nonzero(
    project: Path, registry: ServiceRegistry, monkeypatch
):
    from deep_worker_cli.lsp.service.supervisor import ServiceStartTimeout

    class _Boom:
        def __init__(self, *a, **kw) -> None:
            pass

        def ensure_started(self, root: Path):
            raise ServiceStartTimeout("nope")

    monkeypatch.setattr(
        "deep_worker_cli.lsp.service.service_cli.ServiceSupervisor", _Boom
    )
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_serve(args) != 0


# ── stop ─────────────────────────────────────────────────────────────


def test_cmd_stop_is_idempotent_with_nothing_running(
    project: Path, registry: ServiceRegistry
):
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_stop(args) == 0
    assert cmd_stop(args) == 0


# ── status ───────────────────────────────────────────────────────────


def test_cmd_status_absent_returns_nonzero(project: Path, registry: ServiceRegistry):
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_service_status(args) != 0


def test_cmd_status_registered_but_socket_missing_returns_nonzero(
    project: Path, registry: ServiceRegistry, capsys
):
    """A live pid with no socket means the service crashed mid-init.

    Reporting that as 'running' would make auto-start skip a respawn
    and leave the agent with no tools, so it must be non-zero.
    """
    from deep_worker_cli.lsp.service.registry import RegistryEntry
    import os
    import time

    registry.slug_dir(project)  # create the dir, but no grpc.sock
    registry.write(
        RegistryEntry(
            project_root=project.resolve(),
            pid=os.getpid(),  # live pid → not stale
            transport="uds",
            socket_path=str(registry.slug_dir(project) / "grpc.sock"),
            started_at_ns=time.time_ns(),
        )
    )
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_service_status(args) != 0
    assert "socket missing" in capsys.readouterr().out


def test_cmd_status_stale_pid_returns_nonzero(
    project: Path, registry: ServiceRegistry, capsys
):
    """A registry entry pointing at a dead pid must not read as running."""
    from deep_worker_cli.lsp.service.registry import RegistryEntry
    import time

    sock = registry.slug_dir(project) / "grpc.sock"
    sock.touch()
    registry.write(
        RegistryEntry(
            project_root=project.resolve(),
            pid=2**22,  # above pid_max on Linux → guaranteed dead
            transport="uds",
            socket_path=str(sock),
            started_at_ns=time.time_ns(),
        )
    )
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_service_status(args) != 0
    assert "stale" in capsys.readouterr().out


def test_cmd_status_registered_but_unreachable_reports_and_fails(
    project: Path, registry: ServiceRegistry, capsys
):
    """Registry says "running", but nothing answers on the socket.

    Commit 12 (design §5, amendment A6) tightened this contract: the
    registry block is still printed — it is the only thing available
    without grpc — but ``service-status`` now actually talks to the
    service, so a socket file that nothing is listening on is reported
    as ``unreachable:`` and exits non-zero. Previously this exact state
    exited 0, which made a wedged service look healthy.
    """
    from deep_worker_cli.lsp.service.registry import RegistryEntry
    import os
    import time

    slug_dir = registry.slug_dir(project)
    sock = slug_dir / "grpc.sock"
    sock.touch()
    registry.write(
        RegistryEntry(
            project_root=project.resolve(),
            pid=os.getpid(),  # live pid → not stale
            transport="uds",
            socket_path=str(sock),
            started_at_ns=time.time_ns(),
        )
    )
    args = _ns(project_root=str(project), registry_dir=str(registry.base_dir))
    assert cmd_service_status(args) == 1
    out = capsys.readouterr().out
    assert str(os.getpid()) in out  # registry block still printed
    assert "unreachable:" in out
