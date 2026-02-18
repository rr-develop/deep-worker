"""Process registry for tracking running deep_worker agents."""

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

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

        # Write atomically using a temp file
        temp_file = self.registry_file.with_suffix(".tmp")
        try:
            # Write content
            temp_file.write_text(json.dumps(registry, indent=2))

            # Set permissions to 0o666 (rw-rw-rw-) so all users can read/write
            # This is necessary for multi-user environments (e.g., a web service running
            # alongside deep-worker under a different user)
            os.chmod(temp_file, 0o666)

            # Atomic replace
            temp_file.replace(self.registry_file)
        except OSError as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RegistryLockError(f"Failed to save registry: {e}") from e
