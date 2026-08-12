"""On-disk registry of running per-project LSP services.

See ``docs/DESIGN_LSP_SHARED_SERVICE.md`` §3 (identity/slug) and §5
(directory layout). Contract:

* Slug = ``sha256(realpath(project_root))[:16]``.
* Layout::

      <base_dir>/<slug>/
          state.json    # 0600
          grpc.sock     # 0600, created by the service itself
          service.log   # 0600, appended by the service

  ``base_dir`` defaults to ``~/.deep_worker/lsp``; tests override it.
* ``state.json`` is written **atomically** — temp file in the same
  directory + ``os.rename``. Never half-written on crash.
* Permissions: 0700 on the slug directory, 0600 on ``state.json``.
  Matches the daemon-umask fix (2026-07-04); leaking the socket path
  would open a foothold for any local process.
* This module is a pure predicate + I/O layer: it does NOT start /
  stop / kill anything. The supervisor owns lifecycle.

Public surface used by the supervisor, the bridge (commit 5), and the
CLI:

    project_slug(project_root)  -> str
    ServiceRegistry(base_dir).slug_dir(project_root)  -> Path
    ServiceRegistry.write(entry) / read(project_root) / drop(project_root)
    ServiceRegistry.is_stale(entry)  -> bool
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal
from deep_worker_cli.storage_paths import deep_worker_home

SCHEMA_VERSION: Final = 1
_DIR_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_STATE_FILENAME: Final = "state.json"


TransportKind = Literal["uds", "tcp"]


@dataclass(frozen=True)
class RegistryEntry:
    """One running service, as persisted in ``state.json``.

    ``project_root`` is stored as an absolute realpath — the same input
    that ``project_slug`` receives — so the file is self-describing
    (readers can verify it matches its slug directory).

    ``socket_path`` is the UDS path for ``transport="uds"``, or the
    ``host:port`` string for ``transport="tcp"``. Kept as a string in
    the schema for forward compatibility.
    """

    project_root: Path
    pid: int
    transport: TransportKind
    socket_path: str
    started_at_ns: int
    token: str | None = None  # PSK for TCP transport; None for UDS

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["project_root"] = str(self.project_root)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RegistryEntry":
        # schema_version is informational; ignored on read for now
        # (v1 is the only schema). Callers can inspect it if needed.
        raw_root = data["project_root"]
        raw_transport = data["transport"]
        raw_socket = data["socket_path"]
        if not isinstance(raw_root, str):
            raise ValueError("project_root must be a string")
        if raw_transport not in ("uds", "tcp"):
            raise ValueError(f"unknown transport {raw_transport!r}")
        if not isinstance(raw_socket, str):
            raise ValueError("socket_path must be a string")
        raw_pid = data["pid"]
        raw_started = data["started_at_ns"]
        if not isinstance(raw_pid, int):
            raise ValueError("pid must be an int")
        if not isinstance(raw_started, int):
            raise ValueError("started_at_ns must be an int")
        raw_token = data.get("token")
        if raw_token is not None and not isinstance(raw_token, str):
            raise ValueError("token must be a string or null")
        return cls(
            project_root=Path(raw_root),
            pid=raw_pid,
            transport=raw_transport,  # type: ignore[arg-type]
            socket_path=raw_socket,
            started_at_ns=raw_started,
            token=raw_token,
        )


def _default_base_dir() -> Path:
    """``~/.deep_worker/lsp`` — process-agnostic default.

    Never created here; the registry creates the tree on first
    ``slug_dir`` call so tests using tmp_path see no side effects on
    the developer's home.
    """
    return deep_worker_home() / "lsp"


def project_slug(project_root: Path) -> str:
    """Deterministic 16-hex-char slug for a project root.

    The input MUST be an absolute path already resolved through
    ``realpath`` — the same rule is enforced by ``slug_dir``. Otherwise
    two agents in different worktree names of the same repo would land
    on different slugs and never share the service.

    Note: we do NOT re-resolve here. ``ServiceRegistry.slug_dir`` calls
    ``.resolve()`` explicitly to make the boundary obvious; callers of
    the bare ``project_slug`` are trusted to pass the resolved path.
    """
    if not project_root.is_absolute():
        raise ValueError(
            f"project_root must be absolute, got {project_root!r}"
        )
    return hashlib.sha256(str(project_root).encode()).hexdigest()[:16]


class ServiceRegistry:
    """Reader/writer for ``<base_dir>/<slug>/state.json``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir if base_dir is not None else _default_base_dir()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def slug_dir(self, project_root: Path) -> Path:
        """Directory that holds the per-project registry + socket.

        Resolves the input to an absolute realpath (that is the value
        the slug is computed from). Refuses relative paths so a bug
        in the caller cannot land under an unexpected slug.
        """
        if not project_root.is_absolute():
            raise ValueError(
                f"project_root must be absolute, got {project_root!r}"
            )
        resolved = project_root.resolve()
        slug = project_slug(resolved)
        path = self._base_dir / slug
        # mkdir with the correct mode from the start; if the tree
        # already exists we defensively re-chmod (only the last
        # component — parents may legitimately be shared).
        path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        try:
            path.chmod(_DIR_MODE)
        except OSError:  # pragma: no cover — permission denied on foreign dir
            pass
        return path

    def state_path(self, project_root: Path) -> Path:
        return self.slug_dir(project_root) / _STATE_FILENAME

    def write(self, entry: RegistryEntry) -> None:
        """Persist ``entry`` atomically as ``state.json`` (0600).

        Temp file in the SAME directory as the target so ``os.rename``
        is atomic on POSIX (same filesystem guaranteed). Also fsyncs
        the temp file before rename — power-loss-safe.
        """
        target = self.state_path(entry.project_root)
        parent = target.parent
        payload = json.dumps(entry.to_dict(), sort_keys=True, indent=2).encode(
            "utf-8"
        )

        # NamedTemporaryFile so the temp path is safe against races.
        # delete=False because we hand it off to rename ourselves.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".state.", suffix=".tmp", dir=str(parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, _FILE_MODE)
            os.rename(str(tmp_path), str(target))
        except Exception:
            # Never leave garbage behind on failure.
            tmp_path.unlink(missing_ok=True)
            raise

    def read(self, project_root: Path) -> RegistryEntry | None:
        """Return the current entry for ``project_root``, or None.

        Returns None both when the file is absent and when it is
        malformed — a corrupt state.json is treated as absent so the
        supervisor can rewrite it. Corruption is logged by the caller
        (registry itself has no logger dependency).
        """
        path = self.state_path(project_root)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
            return RegistryEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def drop(self, project_root: Path) -> None:
        """Remove ``state.json`` (keep the directory + socket).

        Idempotent: no-op if the file is already gone.
        """
        path = self.state_path(project_root)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def is_stale(entry: RegistryEntry) -> bool:
        """True iff the recorded pid is no longer a running process.

        Uses ``os.kill(pid, 0)`` — the standard existence probe. Does
        not verify that the process is *our* service; that check is
        the supervisor's job (via the gRPC Health RPC).
        """
        if entry.pid <= 0:
            return True
        try:
            os.kill(entry.pid, 0)
        except OSError as e:
            # ESRCH: no such process → stale. EPERM: exists but we
            # can't signal it → treat as alive (some other user).
            return e.errno == errno.ESRCH
        return False
