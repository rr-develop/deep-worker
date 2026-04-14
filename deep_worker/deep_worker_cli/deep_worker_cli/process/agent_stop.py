"""Utility to gracefully stop a running deep_worker agent."""

import asyncio
import json
import logging
import os
import signal
from pathlib import Path

from deep_worker_cli.process.registry import ProcessRegistry

logger = logging.getLogger(__name__)


def _is_process_alive(pid: int) -> bool:
    """Check if process with given PID is alive."""
    try:
        return Path(f"/proc/{pid}").exists()
    except OSError:
        return False


def _read_agent_status(data_dir: Path) -> str:
    """Read agent status from session_metadata.json.

    Returns:
        Status string ("idle", "running", "waiting_input", "error", "shutdown").
        Defaults to "idle" if file missing or unreadable.
    """
    metadata_file = data_dir / "session_metadata.json"
    try:
        data = json.loads(metadata_file.read_text())
        return data.get("status", "idle")
    except (OSError, json.JSONDecodeError, ValueError):
        return "idle"


async def stop_agent(
    name: str,
    registry: ProcessRegistry,
    *,
    data_dir: Path | None = None,
    timeout: float = 10.0,
) -> bool:
    """Gracefully stop a running agent.

    Args:
        name: Agent name.
        registry: ProcessRegistry to look up agent.
        data_dir: Agent data directory (for reading status).
                  Defaults to ~/.deep_worker/{name}.
        timeout: Seconds to wait for process to exit after SIGTERM.

    Returns:
        True if agent was stopped (or was not running).

    Raises:
        RuntimeError: If agent is actively running (status="running").
        TimeoutError: If process did not exit within timeout.
    """
    if data_dir is None:
        data_dir = Path("~/.deep_worker").expanduser() / name

    # Check registry
    agent_info = registry.lookup(name)
    if agent_info is None:
        logger.info(f"Agent '{name}' is not running — nothing to stop")
        return True

    pid = agent_info.pid

    # Check if actively working
    status = _read_agent_status(data_dir)
    if status == "running":
        raise RuntimeError(
            f"Cannot stop agent '{name}' — it is actively running (status: running). "
            f"Wait for it to finish or interrupt it first."
        )

    # Send SIGTERM
    logger.info(f"Stopping agent '{name}' (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # Already dead
        logger.info(f"Agent '{name}' (PID {pid}) already exited")
        return True

    # Wait for exit
    elapsed = 0.0
    poll_interval = 0.3
    while elapsed < timeout:
        if not _is_process_alive(pid):
            logger.info(f"Agent '{name}' (PID {pid}) stopped")
            return True
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"Agent '{name}' (PID {pid}) did not stop within {timeout:.0f} seconds. "
        f"Check the process manually: kill -9 {pid}"
    )
