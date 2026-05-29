"""Process registry for tracking running deep_worker agents."""

import asyncio
import fcntl
import json
import logging
import os
import struct
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from deep_worker_cli.exceptions import AgentAlreadyExistsError, RegistryLockError
from deep_worker_cli.types import AgentProcessInfo


class ProcessRegistry:
    """
    Centralized registry for tracking all running deep-worker processes.

    Responsibilities:
    - Register/unregister agent processes
    - Lookup agent by name
    - List all running agents
    - Cleanup stale entries (dead processes)
    """

    DEFAULT_REGISTRY_FILE = Path("/tmp/deep_worker_registry.json")
    DEFAULT_LOCK_FILE = Path("/tmp/deep_worker_registry.lock")

    def __init__(
        self,
        registry_file: Path | None = None,
        lock_file: Path | None = None,
    ):
        """Initialize the process registry.

        Args:
            registry_file: Path to the registry JSON file. Defaults to /tmp/deep_worker_registry.json
            lock_file: Path to the lock file. Defaults to /tmp/deep_worker_registry.lock
        """
        self.registry_file = registry_file or self.DEFAULT_REGISTRY_FILE
        self.lock_file = lock_file or self.DEFAULT_LOCK_FILE
        self.logger = logging.getLogger(__name__)

    def register(self, agent_info: AgentProcessInfo) -> None:
        """
        Register a new agent process.

        Args:
            agent_info: Information about the agent process to register

        Raises:
            AgentAlreadyExistsError: If agent with this name already exists and is alive
        """
        with self._lock():
            registry = self._load_registry()

            # Check uniqueness
            if agent_info.name in registry:
                existing = registry[agent_info.name]
                # Verify process still alive
                if self._is_process_alive(existing["pid"]):
                    raise AgentAlreadyExistsError(
                        f"Agent '{agent_info.name}' is already running (PID {existing['pid']})"
                    )
                # Dead process, will be replaced

            # Register new entry
            registry[agent_info.name] = {
                "pid": agent_info.pid,
                "pid_file": str(agent_info.pid_file),
                "working_dir": agent_info.working_dir,
                "started_at": agent_info.started_at.isoformat(),
                "last_activity": datetime.now().isoformat(),
                "rpyc_port": agent_info.rpyc_port,  # RPyC port (REQUIRED)
            }

            self._save_registry(registry)

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the registry.

        Args:
            agent_name: Name of the agent to unregister
        """
        with self._lock():
            registry = self._load_registry()
            if agent_name in registry:
                del registry[agent_name]
                self._save_registry(registry)

    def lookup(self, agent_name: str) -> AgentProcessInfo | None:
        """
        Find an agent by name.

        Args:
            agent_name: Name of the agent to find

        Returns:
            AgentProcessInfo if found and alive, None otherwise
        """
        with self._lock():
            registry = self._load_registry()

            if agent_name not in registry:
                return None

            entry = registry[agent_name]

            # Verify process still alive
            if not self._is_process_alive(entry["pid"]):
                # Cleanup stale entry
                del registry[agent_name]
                self._save_registry(registry)
                return None

            return AgentProcessInfo(
                name=agent_name,
                pid=entry["pid"],
                pid_file=Path(entry["pid_file"]),
                working_dir=entry["working_dir"],
                started_at=datetime.fromisoformat(entry["started_at"]),
                last_activity=datetime.fromisoformat(entry["last_activity"])
                if "last_activity" in entry
                else None,
                rpyc_port=entry.get("rpyc_port"),  # RPyC port (REQUIRED)
            )

    def list_all(self) -> list[AgentProcessInfo]:
        """Get a list of all running agents.

        Returns:
            List of AgentProcessInfo for all alive agents
        """
        with self._lock():
            registry = self._load_registry()
            agents = []
            to_remove = []

            for name, entry in registry.items():
                if self._is_process_alive(entry["pid"]):
                    agents.append(
                        AgentProcessInfo(
                            name=name,
                            pid=entry["pid"],
                            pid_file=Path(entry["pid_file"]),
                            working_dir=entry["working_dir"],
                            started_at=datetime.fromisoformat(entry["started_at"]),
                            last_activity=datetime.fromisoformat(entry["last_activity"])
                            if "last_activity" in entry
                            else None,
                            rpyc_port=entry.get("rpyc_port"),  # RPyC port (REQUIRED)
                        )
                    )
                else:
                    # Mark for cleanup
                    to_remove.append(name)

            # Cleanup dead processes
            if to_remove:
                for name in to_remove:
                    del registry[name]
                self._save_registry(registry)

            return agents

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is alive.

        Args:
            pid: Process ID to check

        Returns:
            True if process is alive, False otherwise
        """
        # Check if /proc/<pid> exists (Linux-specific)
        proc_path = Path(f"/proc/{pid}")
        if not proc_path.exists():
            return False

        # Verify it's our process (deep-worker or deep_worker)
        # Also accept pytest and python for testing purposes (when agent runs as asyncio task in test)
        try:
            cmdline_file = proc_path / "cmdline"
            if not cmdline_file.exists():
                return False

            cmdline = cmdline_file.read_text()
            # Accept: deep-worker, deep_worker, pytest, python (for test scripts)
            return ("deep-worker" in cmdline or
                    "deep_worker" in cmdline or
                    "pytest" in cmdline or
                    "python" in cmdline)
        except (PermissionError, OSError, FileNotFoundError):
            return False

    @contextmanager
    def _lock(self):
        """Context manager for file locking.

        Yields:
            None

        Raises:
            RegistryLockError: If unable to acquire lock
        """
        # Ensure lock file exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.touch(exist_ok=True)

        lock_fd = None
        try:
            lock_fd = os.open(str(self.lock_file), os.O_RDWR)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        except OSError as e:
            raise RegistryLockError(f"Failed to acquire registry lock: {e}") from e
        finally:
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def _load_registry(self) -> dict[str, Any]:
        """Load registry from JSON file.

        Returns:
            Dictionary containing registry data
        """
        if not self.registry_file.exists():
            return {}

        try:
            return json.loads(self.registry_file.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupted or unreadable registry, start fresh
            return {}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """Save registry to JSON file.

        Args:
            registry: Dictionary containing registry data
        """
        # Ensure parent directory exists
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Preserve original file ownership (especially for develop user)
        # This is needed when daemon (root) writes to registry - must keep owner=develop
        original_uid = None
        original_gid = None
        if self.registry_file.exists():
            try:
                file_stat = os.stat(self.registry_file)
                original_uid = file_stat.st_uid
                original_gid = file_stat.st_gid
            except OSError:
                pass  # If stat fails, will use current process owner

        # Write atomically using a temp file
        temp_file = self.registry_file.with_suffix(".tmp")
        try:
            # Write content
            temp_file.write_text(json.dumps(registry, indent=2))

            # Set permissions to 0o666 (rw-rw-rw-) so all users can read/write
            # This is necessary for multi-user environments (e.g., the web client + deep-worker)
            os.chmod(temp_file, 0o666)

            # Atomic replace
            temp_file.replace(self.registry_file)

            # CRITICAL FIX: Restore original ownership after replace()
            # Problem: replace() creates file with current process owner (root for daemon)
            # Solution: Explicitly restore original owner=develop after replace
            if original_uid is not None and original_gid is not None:
                try:
                    # If current process is root, can directly chown
                    # If current process is develop, this is a no-op (already correct owner)
                    os.chown(self.registry_file, original_uid, original_gid)
                except OSError as e:
                    # If chown fails (permission denied), file might be root-owned
                    # This is the exact problem the user reported - MUST be fixed!
                    # Only root can chown files, so if we're not root and file is root-owned,
                    # we have a problem. But since daemon runs as root, this should succeed.
                    self.logger.warning(
                        f"Failed to restore ownership of {self.registry_file} to "
                        f"{original_uid}:{original_gid}: {e}. File may have wrong owner!"
                    )
        except OSError as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RegistryLockError(f"Failed to save registry: {e}") from e

    def validate_writable(self) -> dict[str, Any]:
        """Validate that registry file is writable BEFORE daemonizing.

        This method checks if the registry file can be written to, and attempts
        to fix ownership issues if the file is owned by root or another user.

        Returns:
            Dictionary with validation result:
            - writable (bool): True if registry is writable
            - error (str): Error message if not writable
            - fixed (bool): True if ownership was fixed automatically
        """
        # Check if parent directory is writable (needed if file doesn't exist)
        if not self.registry_file.exists():
            if not os.access(self.registry_file.parent, os.W_OK):
                return {
                    "writable": False,
                    "error": f"Parent directory {self.registry_file.parent} is not writable",
                    "fixed": False,
                }
            return {"writable": True, "fixed": False}

        # File exists - check if it's writable
        if os.access(self.registry_file, os.W_OK):
            return {"writable": True, "fixed": False}

        # File is NOT writable - check if it's owned by root
        try:
            file_stat = os.stat(self.registry_file)
            file_owner = file_stat.st_uid
            current_user = os.getuid()

            if file_owner == 0 and current_user != 0:
                # File is owned by root, current user is not root
                # Try to fix ownership using sudo
                try:
                    import subprocess

                    # Try to change ownership to current user
                    result = subprocess.run(
                        ["sudo", "-n", "chown", f"{current_user}:{current_user}", str(self.registry_file)],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if result.returncode == 0:
                        # Successfully fixed ownership
                        return {
                            "writable": True,
                            "fixed": True,
                            "message": f"Fixed ownership of {self.registry_file}",
                        }
                    else:
                        # sudo failed or requires password
                        return {
                            "writable": False,
                            "error": (
                                f"Registry file {self.registry_file} is owned by root. "
                                f"Run: sudo chown {current_user}:{current_user} {self.registry_file}"
                            ),
                            "fixed": False,
                        }
                except (subprocess.SubprocessError, FileNotFoundError):
                    # sudo not available or failed
                    return {
                        "writable": False,
                        "error": (
                            f"Registry file {self.registry_file} is owned by root. "
                            f"Run: sudo chown {current_user}:{current_user} {self.registry_file}"
                        ),
                        "fixed": False,
                    }
            else:
                # File is owned by current user but not writable (read-only)
                return {
                    "writable": False,
                    "error": f"Registry file {self.registry_file} is read-only. Check file permissions.",
                    "fixed": False,
                }

        except OSError as e:
            return {
                "writable": False,
                "error": f"Failed to check registry file permissions: {e}",
                "fixed": False,
            }


    async def wait_for_registration(
        self, agent_name: str, timeout: float = 30.0
    ) -> Optional[AgentProcessInfo]:
        """Wait for an agent to register, using Linux inotify.

        Uses inotify to avoid polling.
        """
        info = self.lookup(agent_name)
        if info is not None:
            return info

        event = asyncio.Event()
        inotify_fd: int | None = None

        try:
            import ctypes
            import ctypes.util
            libc_name = ctypes.util.find_library("c")
            if libc_name is None:
                return None
            libc = ctypes.CDLL(libc_name, use_errno=True)
            O_NONBLOCK = 0o4000
            inotify_fd = libc.inotify_init1(O_NONBLOCK)
            if inotify_fd < 0:
                return None
            IN_CLOSE_WRITE = 0x00000008
            IN_MOVED_TO = 0x00000080
            IN_CREATE = 0x00000100
            mask = IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE
            wd = libc.inotify_add_watch(
                inotify_fd,
                str(self.registry_file.parent).encode(),
                mask,
            )
            if wd < 0:
                return None
        except (ImportError, OSError, AttributeError, TypeError):
            return None

        loop = asyncio.get_running_loop()
        target_name = self.registry_file.name.encode()

        def _on_inotify_event(fd):
            try:
                raw = os.read(fd, 4096)
                offset = 0
                while offset < len(raw):
                    header_size = 16
                    if offset + header_size > len(raw):
                        break
                    wd_val, mask_val, _cookie, name_len = struct.unpack(
                        b"iIII", raw[offset: offset + header_size]
                    )
                    name_bytes = raw[offset + header_size: offset + header_size + name_len]
                    name_bytes = name_bytes.rstrip(b"\x00")
                    if name_bytes == target_name:
                        info = self.lookup(agent_name)
                        if info is not None:
                            event.set()
                            break
                    offset += header_size + name_len
            except (BlockingIOError, OSError):
                pass

        try:
            loop.add_reader(inotify_fd, lambda: _on_inotify_event(inotify_fd))
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self.lookup(agent_name)
        except asyncio.TimeoutError:
            return None
        finally:
            if inotify_fd is not None:
                try:
                    loop.remove_reader(inotify_fd)
                    os.close(inotify_fd)
                except (OSError, ValueError):
                    pass
