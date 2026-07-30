"""Server management endpoints."""

import os
import sys
import subprocess
import threading
import time
import logging
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _server_run_py() -> Path:
    """Return the run.py for the currently running checkout."""
    argv0 = Path(sys.argv[0]) if sys.argv else Path("run.py")
    if argv0.name == "run.py" and argv0.is_absolute():
        return argv0
    return Path(__file__).resolve().parents[2] / "run.py"


def _restart_helper_code() -> str:
    """Python code for the detached one-shot restart helper."""
    return r'''
import os
import subprocess
import sys
import time
from pathlib import Path

old_pid = int(sys.argv[1])
run_py = Path(sys.argv[2])
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    try:
        os.kill(old_pid, 0)
    except OSError:
        break
    time.sleep(0.05)

# The replacement server bootstraps its own sys.path (run.py
# _bootstrap_sys_path), so it does not need a PYTHONPATH handed down. Keep
# the inherited environment anyway: dropping it is what made a stale
# PYTHONPATH-dependent startup fail invisibly.
#
# Startup output goes to a file, NOT DEVNULL. A restart that dies during
# import (ModuleNotFoundError, bad port, ...) must leave a trace instead of
# vanishing after the old process has already exited.
startup_log = Path("/tmp/deep_worker_web/restart_startup.log")
try:
    startup_log.parent.mkdir(parents=True, exist_ok=True)
    sink = open(startup_log, "ab", buffering=0)
except OSError:
    sink = None

proc = subprocess.Popen(
    [sys.executable, str(run_py)],
    cwd=str(run_py.parent),
    start_new_session=True,
    stdin=subprocess.DEVNULL,
    stdout=sink if sink is not None else subprocess.DEVNULL,
    stderr=subprocess.STDOUT if sink is not None else subprocess.DEVNULL,
)

# If the replacement dies immediately, record it. Without this the restart
# silently leaves no server at all.
time.sleep(2.0)
rc = proc.poll()
if rc is not None and sink is not None:
    sink.write(
        f"FATAL: replacement server exited with code {rc} "
        f"within 2s of restart\\n".encode()
    )
'''.strip()


def do_restart() -> None:
    """Perform server restart from a background thread.

    The endpoint returns the HTTP response before this function exits the
    process. Keep the short delay so the response can flush, then start a
    detached helper. The helper waits for this process to disappear and only
    then starts the replacement, avoiding the bind race on 127.0.0.1:7887/7888.
    """
    logger.info("=" * 60)
    logger.info("RESTART INITIATED")
    current_pid = os.getpid()
    run_py = _server_run_py()
    logger.info(f"Current PID: {current_pid}")
    logger.info(f"Parent PID: {os.getppid()} (left untouched)")
    logger.info(f"Restart run.py: {run_py}")

    # This function runs in the endpoint's background thread. Preserve the
    # response-send gap before os._exit(0); otherwise the client can see a
    # dropped request even though the restart was scheduled.
    logger.debug("Sleeping 0.1s to ensure HTTP response is sent...")
    time.sleep(0.1)
    logger.debug("Sleep completed")

    try:
        helper = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _restart_helper_code(),
                str(current_pid),
                str(run_py),
            ],
            cwd=str(run_py.parent),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Restart helper started with PID: {helper.pid}")
    except Exception as e:
        logger.error(f"ERROR starting restart helper: {e}", exc_info=True)
        # Also record it next to the replacement server's startup output, so a
        # failed restart is diagnosable from one place. This path does NOT
        # reach os._exit(0) below, so the current server keeps serving.
        try:
            startup_log = Path("/tmp/deep_worker_web/restart_startup.log")
            startup_log.parent.mkdir(parents=True, exist_ok=True)
            with open(startup_log, "a", encoding="utf-8") as fh:
                fh.write(
                    f"FATAL: restart helper failed to spawn: {e!r}; "
                    f"current server (pid {current_pid}) left running\n"
                )
        except OSError:
            pass
        raise

    logger.info("Exiting current server process for restart")
    os._exit(0)


@router.post("/api/server/restart")
async def restart_server() -> dict[str, Any]:
    """
    Restart the deep-worker-web server in a new process.

    Returns response immediately, then restarts in background.

    Returns:
        JSON response indicating restart initiated
    """
    try:
        logger.info("Server restart requested via API")

        # Schedule restart in background thread
        restart_thread = threading.Thread(target=do_restart, daemon=True)
        restart_thread.start()

        # Return immediately so client gets the response
        return {
            "success": True,
            "message": "Server restart initiated"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart server: {str(e)}"
        )


@router.get("/api/server/uptime")
async def get_uptime() -> dict[str, Any]:
    """
    Get server uptime information.

    Returns:
        JSON with uptime_seconds and uptime_formatted
    """
    from app.utils.uptime import get_uptime_data
    return get_uptime_data()
