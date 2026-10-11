"""Utility to gracefully stop a running deep_worker agent."""

import asyncio
import json
import logging
import os
import signal
from pathlib import Path

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.storage_paths import agent_root, deep_worker_home

logger = logging.getLogger(__name__)

#: Graceful-shutdown budget for a small agent, in seconds.
BASE_SHUTDOWN_GRACE = 10.0

#: Seconds to wait for the process to die after SIGKILL. The kernel does
#: not negotiate, so this only covers scheduling and unlink of the address
#: space; a process still alive afterwards is stuck in uninterruptible I/O.
KILL_GRACE = 5.0

#: ``(threads.db size in MB, grace in seconds)`` — first match wins, so the
#: thresholds are listed largest-first. The last entry is the hard ceiling.
SHUTDOWN_GRACE_TABLE: tuple[tuple[float, float], ...] = (
    (1000.0, 60.0),
    (500.0, 40.0),
    (100.0, 20.0),
)


def _is_process_alive(pid: int) -> bool:
    """Check if the process is alive, treating a zombie as dead.

    A killed child stays in /proc as a zombie until its parent reaps it, so
    a bare ``/proc/{pid}`` existence check reports it as alive forever when
    the parent is busy — which made SIGKILL look like it had failed. A
    zombie holds no memory and no file handles and cannot touch the
    checkpoint database, so for shutdown purposes it is dead.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False

    # Field 3 is the state character, after the (possibly space- and
    # paren-containing) comm field in parentheses — hence rsplit on ") ".
    try:
        state = stat.rsplit(") ", 1)[1].split(" ", 1)[0]
    except IndexError:
        return True

    return state != "Z"


def compute_shutdown_grace(data_dir: Path) -> float:
    """Seconds to wait for a graceful exit, scaled by checkpoint DB size.

    A flat budget does not fit every agent: shutdown flushes and closes the
    checkpointer, and that cost grows with ``threads.db``. In one observed
    case a 644 MB database pushed the close past the old flat 10s, so
    ``--restart`` aborted and left the agent DOWN.

    Deliberately a discrete table rather than a throughput constant: a
    MB/s figure depends on WAL mode, fsync behaviour and the disk, so it
    would need re-measuring (and would go stale silently). The table is
    deterministic and each step is pinned by a test.

    The result is always bounded — an unbounded wait is exactly the
    anti-pattern removed in 29aef55.
    """
    try:
        size_mb = (data_dir / "threads.db").stat().st_size / (1024 * 1024)
    except OSError:
        # No database yet, or unreadable — the base budget is plenty.
        return BASE_SHUTDOWN_GRACE

    for threshold_mb, grace in SHUTDOWN_GRACE_TABLE:
        if size_mb > threshold_mb:
            return grace
    return BASE_SHUTDOWN_GRACE


def _read_agent_status(data_dir: Path) -> str:
    """Read agent status from session_metadata.json.

    Returns:
        Status string — one of the ``AgentStatus`` enum values:
        "idle", "running", "waiting_input", "interrupted", "error",
        "shutdown". Defaults to "idle" if file missing or unreadable.
    """
    metadata_file = data_dir / "session_metadata.json"
    try:
        data = json.loads(metadata_file.read_text())
        return data.get("status", "idle")
    except (OSError, json.JSONDecodeError, ValueError):
        return "idle"


async def _wait_for_exit(pid: int, timeout: float) -> bool:
    """Wait up to ``timeout`` for ``pid`` to disappear.

    Event driven, no polling loop: each check reschedules itself with
    exponential backoff, and the whole thing is bounded by ``wait_for``.

    Returns:
        True if the process exited, False if the budget ran out.
    """
    process_exited = asyncio.Event()
    check_interval = 0.5
    max_interval = 5.0

    def check_process() -> None:
        if not _is_process_alive(pid):
            process_exited.set()
            return

        nonlocal check_interval
        # get_running_loop, never get_event_loop: under 3.12 the latter
        # raises once a thread has no current loop (cf. the grok fix).
        loop = asyncio.get_running_loop()
        loop.call_later(check_interval, check_process)
        check_interval = min(check_interval * 1.5, max_interval)

    loop = asyncio.get_running_loop()
    loop.call_soon(check_process)

    try:
        await asyncio.wait_for(process_exited.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def stop_agent(
    name: str,
    registry: ProcessRegistry,
    *,
    data_dir: Path | None = None,
    timeout: float | None = None,
    kill_grace: float = KILL_GRACE,
    force: bool = True,
    self_requested: bool = False,
    self_requested_token: str | None = None,
) -> bool:
    """Stop a running agent: SIGTERM, then SIGKILL if it will not go.

    Escalation order::

        SIGTERM -> grace (scaled by threads.db size) -> SIGKILL
                -> kill_grace -> verify dead -> unregister

    ``force=True`` is the default because the alternative silently loses
    agents: the caller that matters (``--restart``) used to turn a slow
    shutdown into an abort, so the old process died anyway — just after the
    budget — and no new one was started. ``--stop``
    (``cli/client_command.py``) has escalated to SIGKILL since it was
    written; this brings the two paths in line. Pass ``force=False`` to opt
    out and keep the SIGTERM-only contract.

    Args:
        name: Agent name.
        registry: ProcessRegistry to look up agent.
        data_dir: Agent data directory (for reading status).
                  Defaults to ~/.deep_worker/{name}.
        timeout: Seconds to wait after SIGTERM. Defaults to a budget scaled
                 by checkpoint DB size (see ``compute_shutdown_grace``).
        kill_grace: Seconds to wait for death after SIGKILL.
        force: Escalate to SIGKILL when the SIGTERM grace expires.
        self_requested: The agent itself asked to be stopped (``restart_self``).
            Waives ONLY the "actively running" guard — that guard exists to
            stop a third party from interrupting a turn it does not know
            about, and the agent is never a third party to its own turn. An
            agent calling ``restart_self`` is always mid-turn, so without
            this the driver would refuse every self-restart. Escalation,
            grace budgets and registry bookkeeping are unchanged. Never set
            this for an external stop.

            NOT sufficient on its own: the flag is a claim, and a claim is
            free to make from any shell. It must be backed by
            ``self_requested_token``.
        self_requested_token: One-time capability minted by ``restart_self``
            INSIDE the target process and written to that agent's private
            data dir (mode 0600). Verified and consumed here. Without a
            matching unconsumed token the waiver does not apply and a busy
            agent is refused exactly as if ``self_requested`` were False —
            so ``deep-worker --restart <busy> --self-requested`` from an
            external shell cannot bypass the active-turn guard.
            See ``process/self_restart_token.py``.

    Returns:
        True if agent was stopped (or was not running).

    Raises:
        RuntimeError: If agent is actively running (status="running") and
            ``self_requested`` is False.
        TimeoutError: If the process outlived SIGTERM (``force=False``) or
            outlived SIGKILL (``force=True``) — the latter means it is stuck
            in uninterruptible I/O and only a reboot or the I/O completing
            will clear it.
    """
    if data_dir is None:
        data_dir = agent_root(name)

    if timeout is None:
        timeout = compute_shutdown_grace(data_dir)

    # Check registry
    agent_info = registry.lookup(name)
    if agent_info is None:
        logger.info(f"Agent '{name}' is not running — nothing to stop")
        return True

    pid = agent_info.pid

    # Check if actively working.
    #
    # The "actively running" guard protects a THIRD PARTY from yanking an
    # agent out from under a turn it knows nothing about. It must not fire
    # when the agent asks to be stopped itself: an agent calling
    # ``restart_self`` is BY CONSTRUCTION mid-turn (status="running"), so
    # applying the guard to it would refuse every self-restart. That is the
    # reason ``restart_self`` used to bypass the driver with ``os._exit``,
    # which killed the in-process MCP server before the tool result could be
    # delivered — the CLI then recorded the dangling call as
    # ``toolDenialKind: "user-rejected"`` and fed the model a fabricated
    # "The user doesn't want to proceed with this tool use". The user had
    # denied nothing. See DESIGN_RESTART_SELF_FALSE_DENIAL.md.
    #
    # Self-stop is narrow ON PURPOSE: the caller must opt in explicitly via
    # ``self_requested=True``. Only the ``restart_self`` path sets it (it is
    # carried from the agent through ``--restart --self-requested``), so an
    # external ``--stop`` / ``--restart`` of a busy agent is refused exactly
    # as before. Note the driver runs in a SEPARATE process, so this cannot
    # be inferred from ``os.getpid()`` — the sanction has to be passed down.
    #
    # ...and because it is passed down, it must be UNFORGEABLE. A bare
    # ``--self-requested`` on the command line is a claim anyone can make;
    # accepting it would hand every external caller a bypass of the
    # active-turn guard. The flag is therefore only honoured when it is
    # backed by a one-time token that ``restart_self`` minted inside the
    # target process's own private data dir. Consumed here (delete-then-
    # compare), so it is valid exactly once and cannot be replayed.
    sanctioned_self_stop = False
    if self_requested:
        from deep_worker_cli.process.self_restart_token import (  # noqa: PLC0415
            consume_self_restart_token,
        )

        sanctioned_self_stop = consume_self_restart_token(
            data_dir, self_requested_token
        )
        if not sanctioned_self_stop:
            logger.warning(
                f"Refusing --self-requested stop of '{name}': no valid "
                f"self-restart token. The active-turn guard stays in force."
            )

    status = _read_agent_status(data_dir)
    if status == "running" and not sanctioned_self_stop:
        raise RuntimeError(
            f"Cannot stop agent '{name}' — it is actively running (status: running). "
            f"Wait for it to finish or interrupt it first."
        )

    # Send SIGTERM
    logger.info(
        f"Stopping agent '{name}' (PID {pid}), grace {timeout:.0f}s..."
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # Already dead
        logger.info(f"Agent '{name}' (PID {pid}) already exited")
        registry.unregister(name)
        return True

    if await _wait_for_exit(pid, timeout):
        logger.info(f"Agent '{name}' (PID {pid}) stopped")
        registry.unregister(name)
        return True

    if not force:
        raise TimeoutError(
            f"Agent '{name}' (PID {pid}) did not stop within {timeout:.0f} seconds. "
            f"Check the process manually: kill -9 {pid}"
        )

    # SIGTERM grace expired — escalate. The old process WILL die eventually,
    # so refusing to continue here is what left agents DOWN.
    logger.warning(
        f"Agent '{name}' (PID {pid}) ignored SIGTERM for {timeout:.0f}s — "
        f"escalating to SIGKILL"
    )
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        # Died between the last check and the signal.
        logger.info(f"Agent '{name}' (PID {pid}) exited before SIGKILL landed")
        registry.unregister(name)
        return True

    if not await _wait_for_exit(pid, kill_grace):
        raise TimeoutError(
            f"Agent '{name}' (PID {pid}) survived SIGKILL after "
            f"{kill_grace:.0f} seconds — it is stuck in uninterruptible I/O "
            f"(state D). Starting a second process on the same data "
            f"directory would corrupt it; investigate the PID manually."
        )

    logger.info(f"Agent '{name}' (PID {pid}) killed")
    registry.unregister(name)
    return True
