"""On-disk state of the running semantic-memory service.

See ``docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md`` §6. Contract:

* There is exactly **one** service per OS user (unit of sharing is the
  host, not the project — design §2), so there is no slug directory:
  the state lives directly in ``<base_dir>``::

      <base_dir>/                 # 0700, default ~/.deep_worker/semantic_memory
          state.json              # 0600
          grpc.sock               # 0600, created by the service itself
          spawn.lock              # 0600, flock'd by the supervisor
          service.log             # 0600, appended by the service

* ``state.json`` is written **atomically** — temp file in the same
  directory, ``fsync``, then ``os.rename``, then ``fsync`` on the parent
  directory so the rename survives power loss.
* Permissions: 0700 on the directory, 0600 on the files. ``mkdir(mode=)``
  is a no-op on an existing directory and is masked by the ambient umask,
  so the mode is re-asserted with an explicit ``chmod`` — this is exactly
  how ``the code-graph server`` ended up with a world-readable ``svc/``.
* Liveness checks BOTH the pid and its start time in kernel ticks. The pid
  alone is not enough: pids get recycled, and a recycled pid would make a
  dead service look alive forever.

This module is a pure predicate + I/O layer: it does NOT start, stop, or
kill anything. The supervisor owns lifecycle.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from deep_worker_cli.storage_paths import deep_worker_home

SCHEMA_VERSION: Final = 1
PROTOCOL_VERSION: Final = 1

_DIR_MODE: Final = 0o700
_FILE_MODE: Final = 0o600
_STATE_FILENAME: Final = "state.json"

#: Field 22 of ``/proc/<pid>/stat`` (0-indexed 21) is the process start
#: time in kernel ticks since boot. Verified on this host.
_PROC_STAT_STARTTIME_INDEX: Final = 21


@dataclass(frozen=True)
class ServiceState:
    """One running service, as persisted in ``state.json``."""

    pid: int
    socket_path: str
    ready_at_ns: int
    protocol_version: int = PROTOCOL_VERSION
    start_ticks: int = 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ServiceState":
        def _as_int(name: str, value: object) -> int:
            # bool is a subclass of int; reject it explicitly so a JSON
            # `true` cannot masquerade as a pid.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int")
            return value

        socket_path = data["socket_path"]
        if not isinstance(socket_path, str):
            raise ValueError("socket_path must be a string")
        return cls(
            pid=_as_int("pid", data["pid"]),
            socket_path=socket_path,
            ready_at_ns=_as_int("ready_at_ns", data["ready_at_ns"]),
            protocol_version=_as_int(
                "protocol_version",
                data.get("protocol_version", PROTOCOL_VERSION),
            ),
            start_ticks=_as_int("start_ticks", data.get("start_ticks", 0)),
        )


def read_start_ticks(pid: int) -> int:
    """Process start time in kernel ticks, or 0 if unavailable.

    Reading ``/proc/<pid>/stat`` is the cheapest exact guard against pid
    reuse: a recycled pid necessarily has a different start time. The
    comm field (2) may contain spaces and parentheses, so the tail is
    taken after the final ``)``.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return 0
    try:
        tail = raw[raw.rindex(")") + 2 :].split()
        # After comm, the first field is `state`, which is index 2 of the
        # full record — so the full-record index shifts by 2 here.
        return int(tail[_PROC_STAT_STARTTIME_INDEX - 2])
    except (ValueError, IndexError):
        return 0


def tenant_slug(path: Path) -> str:
    """Deterministic 16-hex-char id for a repository or summaries dir.

    Used for cache keys and log lines, not for directory layout (there is
    only one service directory). The input MUST be absolute so two agents
    with different cwds cannot address the same tenant differently.
    """
    if not path.is_absolute():
        raise ValueError(f"path must be absolute, got {path!r}")
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


def _default_base_dir() -> Path:
    """``~/.deep_worker/semantic_memory`` — one per OS user.

    Never created here; the registry creates the tree on demand so that
    merely importing this module has no side effects on the real home.
    """
    return deep_worker_home() / "semantic_memory"


class ServiceRegistry:
    """Reader/writer for ``<base_dir>/state.json``."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir if base_dir is not None else _default_base_dir()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def state_path(self) -> Path:
        return self._base_dir / _STATE_FILENAME

    @property
    def socket_path(self) -> Path:
        return self._base_dir / "grpc.sock"

    @property
    def lock_path(self) -> Path:
        return self._base_dir / "spawn.lock"

    @property
    def log_path(self) -> Path:
        return self._base_dir / "service.log"

    def ensure_dir(self) -> Path:
        """Create ``base_dir`` 0700 and re-assert the mode if it exists."""
        self._base_dir.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        try:
            self._base_dir.chmod(_DIR_MODE)
        except OSError:  # pragma: no cover - foreign-owned dir
            pass
        return self._base_dir

    def write(self, state: ServiceState) -> None:
        """Persist ``state`` atomically as ``state.json`` (0600)."""
        parent = self.ensure_dir()
        target = self.state_path
        payload = json.dumps(
            state.to_dict(), sort_keys=True, indent=2
        ).encode("utf-8")

        fd, tmp_name = tempfile.mkstemp(
            prefix=".state.", suffix=".tmp", dir=str(parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_path, _FILE_MODE)
            os.rename(str(tmp_path), str(target))
            self._fsync_dir(parent)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """Make the rename itself durable, not just the file contents."""
        try:
            dir_fd = os.open(str(path), os.O_RDONLY)
        except OSError:  # pragma: no cover
            return
        try:
            os.fsync(dir_fd)
        except OSError:  # pragma: no cover - not all filesystems support it
            pass
        finally:
            os.close(dir_fd)

    def read(self) -> ServiceState | None:
        """Current state, or None when absent **or malformed**.

        A corrupt state.json is treated as absent so the supervisor can
        simply rewrite it instead of needing a repair path.
        """
        try:
            raw = self.state_path.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                return None
            return ServiceState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def drop(self) -> None:
        """Remove ``state.json``; keep the directory and socket."""
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def is_stale(state: ServiceState) -> bool:
        """True iff the recorded process is gone or was replaced.

        Two conditions, both necessary:

        1. the pid is not running at all;
        2. the pid IS running but its start time differs from the one we
           recorded — i.e. the pid was recycled by an unrelated process.

        ``start_ticks == 0`` means "not recorded"; fall back to (1) alone
        rather than declaring every such entry stale.
        """
        if state.pid <= 0:
            return True
        try:
            os.kill(state.pid, 0)
        except OSError as exc:
            # ESRCH: the process is gone.
            # EPERM: a process exists but belongs to another uid — which
            # means it is NOT our service. Our service always runs as the
            # user that started it, so a foreign owner can only mean the
            # pid was recycled. Treating that as "alive" would hand the
            # caller a socket nobody is listening on, and since liveness
            # deliberately issues no Health RPC (design §7), nothing
            # downstream would catch it: the client's single retry would
            # re-derive the same dead socket forever.
            return exc.errno in (errno.ESRCH, errno.EPERM)
        if state.start_ticks:
            current = read_start_ticks(state.pid)
            if current and current != state.start_ticks:
                return True
        return False
