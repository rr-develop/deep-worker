"""Session manager for deep_worker agents."""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from deep_worker_cli.storage_paths import (
    CURRENT_STORAGE_VERSION,
    STORAGE_VERSION_DEEPAGENTS_PER_AGENT,
    ensure_private_dir,
    ensure_private_file,
)
from deep_worker_cli.types import AgentStatus

logger = logging.getLogger(__name__)

StatusNotifier = Callable[[dict[str, Any]], Awaitable[None]]


class SessionManager:
    """Manages session state and metadata for a deep_worker agent.

    Responsibilities:
    - Track agent status (idle, running, waiting_input, interrupted,
      error, shutdown) — see ``deep_worker_cli.types.AgentStatus``
    - Count completed tasks
    - Count messages
    - Persist metadata to disk
    - Track last activity timestamp
    - Store launch_config (CLI args, provider, model, etc.) for post-mortem inspection
    """

    status_metadata: dict[str, Any] | None

    def __init__(
        self,
        data_dir: Path,
        launch_config: dict | None = None,
        *,
        storage_version: int | None = None,
        notifier: StatusNotifier | None = None,
    ):
        """Initialize the session manager.

        Args:
            data_dir: Directory for storing session data
            launch_config: Optional dict with CLI launch parameters (provider, model,
                working_dir, skip_permissions, task_manager, etc.). When provided,
                it is persisted to session_metadata.json so that launch parameters
                survive process termination and can be inspected later.
            storage_version: Optional storage layout version for newly-created
                metadata. Existing metadata remains authoritative when present.
            notifier: Optional async callable that receives a dict
                describing every state transition (``set_status``,
                ``set_error``, ``clear_error``). The envelope is
                ``{"type": "status_change", "status": <enum_value>,
                "error_message": <str|None>}``. ``AgentProcess`` wires
                this so subscribers learn about state changes event-
                driven, with no polling. If None, SessionManager
                persists as before — no push.
        """
        self._notifier = notifier
        self.data_dir = data_dir
        self.metadata_file = data_dir / "session_metadata.json"
        launch_storage_version = (
            launch_config.get("storage_version") if isinstance(launch_config, dict) else None
        )
        self.storage_version = (
            storage_version
            or (launch_storage_version if isinstance(launch_storage_version, int) else None)
            or CURRENT_STORAGE_VERSION
        )
        self.status_metadata: dict[str, Any] | None = None

        ensure_private_dir(self.data_dir)

        # Load or initialize metadata
        if self.metadata_file.exists():
            self._load_metadata()
        else:
            self.status = AgentStatus.IDLE
            self.tasks_completed = 0
            self.message_count = 0
            self.error_message: str | None = None
            self.status_metadata: dict[str, Any] | None = None
            self.last_activity = datetime.now()
            self.launch_config: dict | None = None

        # If a new launch_config is provided, this is a fresh agent start/restart.
        # ⚠️ CRITICAL: MUST reset status to IDLE and clear error_message!
        # Without this, a previous crash (status=error) persists across restarts
        # and the agent appears stuck in error state even though it's functional.
        # DO NOT remove this reset — it prevents the "restart doesn't clear error" bug.
        if launch_config is not None:
            self.launch_config = launch_config
            self.status = AgentStatus.IDLE
            self.error_message = None
            self.status_metadata = None

        # Persist (writes launch_config + reset status if present)
        if not self.metadata_file.exists() or launch_config is not None:
            self._persist_sync()

    async def _notify_status(self) -> None:
        """Push a ``status_change`` event to the subscriber channel.

        Defensive: a broken / slow subscriber MUST NOT corrupt the
        persisted state. We log and swallow any exception so the
        caller can continue.
        """
        if self._notifier is None:
            return
        payload: dict[str, Any] = {
            "type": "status_change",
            "status": self.status.value,
            "error_message": self.error_message,
        }
        if self.status_metadata is not None:
            payload["status_metadata"] = self.status_metadata
        try:
            await self._notifier(payload)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning("status notifier raised: %s", exc)

    async def set_status(
        self,
        status: AgentStatus,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set the agent status.

        Args:
            status: New status for the agent
            metadata: Optional status metadata for the new state.
        """
        self.status = status
        self.status_metadata = dict(metadata) if isinstance(metadata, dict) else None
        self.last_activity = datetime.now()
        await self._persist()
        await self._notify_status()

    async def set_error(self, error_message: str) -> None:
        """Set error status with message.

        Args:
            error_message: Error message to record
        """
        self.status = AgentStatus.ERROR
        self.error_message = error_message
        self.status_metadata = None
        self.last_activity = datetime.now()
        await self._persist()
        await self._notify_status()

    async def clear_error(self) -> None:
        """Clear error message."""
        self.error_message = None
        await self._persist()
        await self._notify_status()

    async def update_launch_config_model(self, model: str | None) -> None:
        if not (
            isinstance(model, str)
            and model.strip()
            and model.strip().lower() != "unknown"
        ):
            return
        if self.launch_config is None:
            return
        resolved = model.strip()
        if self.launch_config.get("model") == resolved:
            return
        self.launch_config["model"] = resolved
        await self._persist()
        if self._notifier is None:
            return
        payload: dict[str, Any] = {
            "type": "status_change",
            "status": self.status.value,
            "error_message": self.error_message,
            "provider": self.launch_config.get("provider"),
            "model": resolved,
        }
        if self.status_metadata is not None:
            payload["status_metadata"] = self.status_metadata
        try:
            await self._notifier(payload)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning("status notifier raised: %s", exc)

    async def increment_tasks_completed(self) -> None:
        """Increment the tasks completed counter."""
        self.tasks_completed += 1
        self.last_activity = datetime.now()
        await self._persist()

    async def increment_message_count(self) -> None:
        """Increment the message count."""
        self.message_count += 1
        self.last_activity = datetime.now()
        await self._persist()

    def _build_metadata_dict(self) -> dict:
        """Build the metadata dict for persistence.

        Returns:
            Dict with all session metadata including launch_config if present.
        """
        metadata = {
            "status": self.status.value,
            "tasks_completed": self.tasks_completed,
            "message_count": self.message_count,
            "error_message": self.error_message,
            "last_activity": self.last_activity.isoformat(),
            "storage_version": self.storage_version,
        }
        if self.status_metadata is not None:
            metadata["status_metadata"] = self.status_metadata
        if self.launch_config is not None:
            metadata["launch_config"] = self.launch_config
        return metadata

    def _unique_temp_path(self) -> Path:
        """A per-call unique temp path for atomic metadata writes.

        ⚠️ MUST be unique per write. Previously every persist used the SAME
        ``session_metadata.tmp`` path. ``set_status`` is called from
        on_tool_start, on_tool_end AND the main turn, and ``_persist`` runs
        each in a thread pool — so multiple persists ran CONCURRENTLY against
        the one shared ``.tmp``. One writer's ``replace()``/``unlink()``
        removed the temp file under another → ``FileNotFoundError(2)`` (incident
        observed 2026-05-22, a deepseek agent — many fast tool turns made it loud). A
        unique name (pid + uuid) means concurrent writers never touch each
        other's temp file; the last ``replace`` onto the real path wins, and
        ``replace`` is atomic so the metadata file is never half-written.
        """
        return self.metadata_file.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")

    def _persist_sync(self) -> None:
        """Persist metadata to disk synchronously (for initialization only)."""
        metadata = self._build_metadata_dict()

        # Write atomically using a UNIQUE temp file (see _unique_temp_path).
        temp_file = self._unique_temp_path()
        try:
            temp_file.write_text(json.dumps(metadata, indent=2))
            ensure_private_file(temp_file)
            temp_file.replace(self.metadata_file)
            ensure_private_file(self.metadata_file)
        except OSError:
            if temp_file.exists():
                temp_file.unlink()
            raise

    async def _persist(self) -> None:
        """Persist metadata to disk asynchronously to avoid blocking event loop."""
        metadata = self._build_metadata_dict()

        # Write atomically using a UNIQUE temp file (run in thread pool to
        # avoid blocking). The unique name is what makes concurrent persists
        # safe — see _unique_temp_path for the incident observed 2026-05-22 race.
        temp_file = self._unique_temp_path()

        def _write_to_disk():
            try:
                temp_file.write_text(json.dumps(metadata, indent=2))
                ensure_private_file(temp_file)
                temp_file.replace(self.metadata_file)
                ensure_private_file(self.metadata_file)
            except OSError:
                if temp_file.exists():
                    temp_file.unlink()
                raise

        # Run blocking I/O in thread pool
        await asyncio.to_thread(_write_to_disk)

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        try:
            data = json.loads(self.metadata_file.read_text())

            self.status = AgentStatus(data.get("status", "idle"))
            self.tasks_completed = data.get("tasks_completed", 0)
            self.message_count = data.get("message_count", 0)
            self.error_message = data.get("error_message")
            raw_status_metadata = data.get("status_metadata")
            self.status_metadata = (
                raw_status_metadata if isinstance(raw_status_metadata, dict) else None
            )
            self.last_activity = datetime.fromisoformat(
                data.get("last_activity", datetime.now().isoformat())
            )
            version = data.get("storage_version")
            self.storage_version = (
                version if isinstance(version, int) else STORAGE_VERSION_DEEPAGENTS_PER_AGENT
            )
            # Load launch_config if present (backward compatible — old files won't have it)
            self.launch_config = data.get("launch_config")
        except (json.JSONDecodeError, OSError, ValueError):
            # If metadata is corrupted, start fresh
            self.status = AgentStatus.IDLE
            self.tasks_completed = 0
            self.message_count = 0
            self.error_message = None
            self.status_metadata = None
            self.last_activity = datetime.now()
            self.launch_config = None
            self.storage_version = STORAGE_VERSION_DEEPAGENTS_PER_AGENT
            self._persist_sync()
