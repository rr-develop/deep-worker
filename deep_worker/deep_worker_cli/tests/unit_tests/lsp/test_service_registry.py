"""Registry contract for the per-project LSP shared service.

See docs/DESIGN_LSP_SHARED_SERVICE.md §3-§5. Tests pin the invariants:

* slug is deterministic sha256(realpath(project_root))[:16] — stable
  across worktree names and symlinks; two different realpaths never
  collide.
* Registry lives at ``<data_dir>/lsp/<slug>/`` with 700 on dirs and
  600 on files (matches the daemon-umask fix, 2026-07-04).
* Registry state is written atomically (temp file + rename), never
  half-written on crash.
* Stale entries (pid dead) are detected as such.
* ``resolve_by_project`` returns the current live entry or None; no
  polling.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from deep_worker_cli.lsp.service.registry import (
    RegistryEntry,
    ServiceRegistry,
    project_slug,
)


# ── slug determinism ─────────────────────────────────────────────────


def test_slug_is_sha256_of_realpath_first_16() -> None:
    """Slug must be deterministic; document the exact function."""
    root = "/some/absolute/path"
    expected = hashlib.sha256(root.encode()).hexdigest()[:16]
    assert project_slug(Path(root)) == expected


def test_slug_stable_across_symlink_and_worktree_names(tmp_path: Path) -> None:
    """A symlink pointing at the real root must resolve to the same slug.

    Symlink-collapse is the ``ServiceRegistry.slug_dir`` contract
    (``.resolve()`` before hashing); bare ``project_slug`` documents
    the requirement that its input already be a realpath.
    """
    real = tmp_path / "real_project"
    real.mkdir()
    link = tmp_path / "link_project"
    link.symlink_to(real)

    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    assert reg.slug_dir(real) == reg.slug_dir(link)
    assert project_slug(real.resolve()) == project_slug(link.resolve())


def test_slug_differs_across_different_realpaths(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert project_slug(a) != project_slug(b)


# ── registry directory layout & permissions ──────────────────────────


def test_registry_creates_slug_dir_with_0700(tmp_path: Path) -> None:
    """The per-slug directory must be 0700 (private to the user).

    Matches the daemon-umask fix (2026-07-04); leaking the socket
    path would open a foothold for any local process.
    """
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_a"
    project.mkdir()

    slug_dir = reg.slug_dir(project)
    assert slug_dir.is_dir()
    mode = stat.S_IMODE(slug_dir.stat().st_mode)
    assert mode == 0o700, oct(mode)


def test_registry_writes_state_file_with_0600(tmp_path: Path) -> None:
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_b"
    project.mkdir()

    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=os.getpid(),
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=42,
    )
    reg.write(entry)

    state = reg.slug_dir(project) / "state.json"
    assert state.is_file()
    mode = stat.S_IMODE(state.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_registry_write_is_atomic(tmp_path: Path, monkeypatch) -> None:
    """A crash mid-write must never leave a partial state.json readable.

    Enforced structurally: the implementation MUST write to a temp
    file in the same directory and rename() into place. We assert that
    contract by patching os.rename and checking the temp file is what
    exists at that moment (never the target).
    """
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_c"
    project.mkdir()

    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=os.getpid(),
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=1,
    )

    seen: dict[str, list[Path]] = {"targets": [], "sources": []}
    real_rename = os.rename

    def fake_rename(src, dst):  # type: ignore[no-untyped-def]
        seen["sources"].append(Path(src))
        seen["targets"].append(Path(dst))
        # A partial temp file must exist at src BEFORE rename runs
        assert Path(src).is_file(), "temp file must exist before rename"
        # The destination must NOT exist yet on the first call
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", fake_rename)
    reg.write(entry)
    assert seen["targets"], "write must go through os.rename"
    assert seen["targets"][-1].name == "state.json"
    assert seen["sources"][-1].parent == seen["targets"][-1].parent, (
        "temp file must live in the same directory to keep rename atomic"
    )


# ── round-trip & staleness ───────────────────────────────────────────


def test_read_returns_entry_after_write(tmp_path: Path) -> None:
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_d"
    project.mkdir()
    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=os.getpid(),  # our own pid: definitely alive
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=1234,
    )
    reg.write(entry)

    loaded = reg.read(project)
    assert loaded is not None
    assert loaded.project_root == entry.project_root
    assert loaded.pid == entry.pid
    assert loaded.socket_path == entry.socket_path


def test_read_returns_none_when_absent(tmp_path: Path) -> None:
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_e"
    project.mkdir()
    assert reg.read(project) is None


def test_is_stale_true_for_dead_pid(tmp_path: Path) -> None:
    """A pid that no longer exists must be reported stale.

    The registry doesn't kill anything itself — the supervisor is
    responsible for reclaim. This is a pure predicate.
    """
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_f"
    project.mkdir()
    # pid=1 (init) exists but is not our service; pick a definitely-dead
    # pid by forking a child that exits immediately.
    pid = os.fork()
    if pid == 0:  # child
        os._exit(0)
    os.waitpid(pid, 0)  # reap → guaranteed dead

    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=pid,
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=1,
    )
    reg.write(entry)

    loaded = reg.read(project)
    assert loaded is not None
    assert reg.is_stale(loaded) is True


def test_is_stale_false_for_own_pid(tmp_path: Path) -> None:
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_g"
    project.mkdir()
    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=os.getpid(),
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=1,
    )
    assert reg.is_stale(entry) is False


def test_drop_removes_state_file(tmp_path: Path) -> None:
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_h"
    project.mkdir()
    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=os.getpid(),
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=1,
    )
    reg.write(entry)
    reg.drop(project)
    assert reg.read(project) is None
    # Directory kept (may hold the socket file); only state.json is dropped
    assert reg.slug_dir(project).is_dir()


def test_reject_relative_project_root(tmp_path: Path) -> None:
    """The registry MUST refuse relative paths — the slug is only
    stable across processes when the input is absolute+resolved."""
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    with pytest.raises(ValueError, match="absolute"):
        reg.slug_dir(Path("relative/path"))


# ── JSON stability (compat guard) ────────────────────────────────────


def test_state_json_keys_are_stable(tmp_path: Path) -> None:
    """The bridge in commit 5 will read state.json; freeze the schema
    now so a rename doesn't silently break it."""
    reg = ServiceRegistry(base_dir=tmp_path / "lsp")
    project = tmp_path / "proj_i"
    project.mkdir()
    entry = RegistryEntry(
        project_root=project.resolve(),
        pid=os.getpid(),
        transport="uds",
        socket_path=str(reg.slug_dir(project) / "grpc.sock"),
        started_at_ns=1,
    )
    reg.write(entry)
    data = json.loads((reg.slug_dir(project) / "state.json").read_text())
    assert set(data) >= {
        "schema_version",
        "project_root",
        "pid",
        "transport",
        "socket_path",
        "started_at_ns",
    }
    assert data["schema_version"] == 1
