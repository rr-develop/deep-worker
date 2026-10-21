"""Registry of the running semantic-memory service.

Design: docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md §6.

Unlike the LSP registry there is exactly ONE service per OS user, so the
state lives in a single directory rather than under a per-project slug.
What this suite pins:

* atomic write — never a half-written state.json;
* permissions 0700 on the dir, 0600 on the file;
* a corrupt file reads as absent, so the supervisor can rewrite it;
* liveness uses BOTH pid and ``start_ticks`` — the PID-reuse guard the
  LSP registry lacks.
"""
from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path

import pytest

from deep_worker_cli.semantic_memory.registry import (
    SCHEMA_VERSION,
    ServiceRegistry,
    ServiceState,
    read_start_ticks,
    tenant_slug,
)


@pytest.fixture
def registry(tmp_path: Path) -> ServiceRegistry:
    """Registry rooted in tmp_path — never the developer's real home."""
    return ServiceRegistry(base_dir=tmp_path / "semantic_memory")


def _state(**overrides: object) -> ServiceState:
    defaults: dict[str, object] = {
        "pid": os.getpid(),
        "socket_path": "/tmp/does-not-matter/grpc.sock",
        "ready_at_ns": 123456789,
        "protocol_version": 1,
        "start_ticks": read_start_ticks(os.getpid()),
    }
    defaults.update(overrides)
    return ServiceState(**defaults)  # type: ignore[arg-type]


class TestTenantSlug:
    def test_is_deterministic_and_16_hex_chars(self) -> None:
        slug = tenant_slug(Path("/projects/example"))
        assert slug == tenant_slug(Path("/projects/example"))
        assert len(slug) == 16
        assert all(c in "0123456789abcdef" for c in slug)

    def test_differs_between_paths(self) -> None:
        assert tenant_slug(Path("/projects/a")) != tenant_slug(Path("/projects/b"))

    def test_rejects_relative_path(self) -> None:
        # A relative path would let two agents in different cwds land on
        # different slugs for the same tenant.
        with pytest.raises(ValueError, match="absolute"):
            tenant_slug(Path("projects/a"))


class TestWriteRead:
    def test_roundtrip(self, registry: ServiceRegistry) -> None:
        state = _state()
        registry.write(state)
        assert registry.read() == state

    def test_missing_file_reads_as_none(self, registry: ServiceRegistry) -> None:
        assert registry.read() is None

    def test_corrupt_file_reads_as_none(self, registry: ServiceRegistry) -> None:
        registry.write(_state())
        registry.state_path.write_text("{not json", encoding="utf-8")
        assert registry.read() is None

    def test_wrong_types_read_as_none(self, registry: ServiceRegistry) -> None:
        registry.write(_state())
        registry.state_path.write_text(
            json.dumps({"pid": "not-an-int", "socket_path": "/x",
                        "ready_at_ns": 1, "protocol_version": 1,
                        "start_ticks": 1}),
            encoding="utf-8",
        )
        assert registry.read() is None

    def test_schema_version_is_persisted(self, registry: ServiceRegistry) -> None:
        registry.write(_state())
        payload = json.loads(registry.state_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION

    def test_write_leaves_no_temp_files(self, registry: ServiceRegistry) -> None:
        registry.write(_state())
        leftovers = [p.name for p in registry.base_dir.iterdir()
                     if p.name != "state.json"]
        assert leftovers == []

    def test_rewrite_replaces_previous_state(self, registry: ServiceRegistry) -> None:
        registry.write(_state(pid=111))
        registry.write(_state(pid=222))
        entry = registry.read()
        assert entry is not None and entry.pid == 222


class TestPermissions:
    def test_dir_is_private(self, registry: ServiceRegistry) -> None:
        registry.write(_state())
        mode = stat.S_IMODE(registry.base_dir.stat().st_mode)
        assert mode == 0o700, f"expected 0700, got 0o{mode:o}"

    def test_state_file_is_private(self, registry: ServiceRegistry) -> None:
        registry.write(_state())
        mode = stat.S_IMODE(registry.state_path.stat().st_mode)
        assert mode == 0o600, f"expected 0600, got 0o{mode:o}"

    def test_existing_wide_open_dir_is_tightened(
        self, tmp_path: Path
    ) -> None:
        # mkdir(mode=...) is a no-op on an existing directory and is
        # masked by the ambient umask — this is exactly how the code-graph server
        # ended up with a 0755 svc/ dir. An explicit chmod must fix it.
        base = tmp_path / "semantic_memory"
        base.mkdir(mode=0o755)
        os.chmod(base, 0o755)
        reg = ServiceRegistry(base_dir=base)
        reg.write(_state())
        assert stat.S_IMODE(base.stat().st_mode) == 0o700


class TestLiveness:
    def test_own_process_is_not_stale(self, registry: ServiceRegistry) -> None:
        assert ServiceRegistry.is_stale(_state()) is False

    def test_pid_zero_is_stale(self, registry: ServiceRegistry) -> None:
        assert ServiceRegistry.is_stale(_state(pid=0)) is True

    def test_dead_pid_is_stale(self, registry: ServiceRegistry) -> None:
        # PID 2^22 is above the default pid_max on Linux, so it cannot exist.
        assert ServiceRegistry.is_stale(_state(pid=4_194_304)) is True

    def test_mismatched_start_ticks_is_stale(self) -> None:
        # THE point of start_ticks: our own pid is alive, but the recorded
        # start time belongs to a different (recycled) process.
        recycled = _state(start_ticks=read_start_ticks(os.getpid()) + 1)
        assert ServiceRegistry.is_stale(recycled) is True

    def test_zero_start_ticks_skips_the_check(self) -> None:
        # 0 means "not recorded" (older schema); fall back to pid liveness
        # rather than declaring every such entry stale.
        assert ServiceRegistry.is_stale(_state(start_ticks=0)) is False

    def test_foreign_owned_pid_is_stale(self, monkeypatch) -> None:
        """EPERM means the pid belongs to someone else — i.e. NOT our service.

        Treating it as alive is a real hole, not a theoretical one: the
        service dies leaving its socket file behind, the pid is recycled by
        a process of another uid, and ``ensure_started`` then hands back a
        socket nobody is listening on. The client's single retry re-runs
        ``ensure_started``, gets the same dead socket, and the agent is
        stuck in a loop. Since liveness deliberately does not issue a
        Health RPC (design §7), this check is the only thing standing
        between a recycled pid and that loop.
        """

        def _eperm(_pid: int, _sig: int) -> None:
            raise OSError(errno.EPERM, "Operation not permitted")

        monkeypatch.setattr(os, "kill", _eperm)
        assert ServiceRegistry.is_stale(_state()) is True


class TestReadStartTicks:
    def test_returns_positive_for_live_process(self) -> None:
        assert read_start_ticks(os.getpid()) > 0

    def test_returns_zero_for_dead_process(self) -> None:
        assert read_start_ticks(4_194_304) == 0
