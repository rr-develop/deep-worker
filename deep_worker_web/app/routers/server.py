"""Server management endpoints."""

import os
import sys
import subprocess
import threading
import time
import signal
import logging
from typing import Any
from fastapi import APIRouter, HTTPException
import psutil

logger = logging.getLogger(__name__)
router = APIRouter()


def do_restart() -> None:
    """
    Perform server restart.

    Waits briefly to ensure HTTP response is sent, then starts new process
    and immediately terminates current one.
    """
    logger.info("=" * 60)
    logger.info("RESTART INITIATED")
    logger.info(f"Current PID: {os.getpid()}")
    logger.info(f"Parent PID: {os.getppid()}")

    # Brief wait to ensure HTTP response is sent (0.1s is sufficient)
    logger.debug("Sleeping 0.1s to ensure HTTP response is sent...")
    time.sleep(0.1)
    logger.debug("Sleep completed")

    # Get the Python executable
    python_exe = sys.executable

    # Determine how the server was started
    script_args = sys.argv.copy()

    # Check if we were launched via run.py
    if len(script_args) > 0:
        first_arg = script_args[0]

        # If run.py is in the path, use it
        if 'run.py' in first_arg:
            script_args = [python_exe, 'run.py']
        # If running via uvicorn directly
        elif first_arg.endswith('uvicorn') or 'uvicorn' in first_arg:
            # Keep the original command
            script_args = [python_exe] + script_args
        else:
            # Default to run.py
            script_args = [python_exe, 'run.py']
    else:
        # Default to run.py
        script_args = [python_exe, 'run.py']

    # Get current working directory
    cwd = os.getcwd()

    logger.info(f"Starting new process: {' '.join(script_args)}")
    logger.info(f"Working directory: {cwd}")

    # Start new process
    try:
        new_proc = subprocess.Popen(
            script_args,
            cwd=cwd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"New process started with PID: {new_proc.pid}")
    except Exception as e:
        logger.error(f"ERROR starting new process: {e}", exc_info=True)
        raise

    logger.info("Preparing to terminate process tree...")

    # Kill the PARENT process to free the port
    try:
        current_pid = os.getpid()
        parent_pid = os.getppid()

        logger.info(f"Current PID: {current_pid}, Parent PID: {parent_pid}")

        # Check if we have a parent process
        if parent_pid > 1:
            logger.info(f"Killing parent process PID {parent_pid}")

            # Try graceful shutdown first (SIGTERM)
            try:
                os.kill(parent_pid, signal.SIGTERM)
                logger.info("Sent SIGTERM to parent, waiting 0.5s...")
                time.sleep(0.5)
            except OSError:
                pass  # Process already dead

            # Check if parent is still alive
            try:
                # os.kill(pid, 0) doesn't kill, just checks if process exists
                os.kill(parent_pid, 0)
                # Still alive, force kill with SIGKILL
                logger.warning("Parent still alive after SIGTERM, sending SIGKILL")
                os.kill(parent_pid, signal.SIGKILL)
                time.sleep(0.2)
            except OSError:
                # Process is dead (as expected)
                logger.info("Parent process terminated")

            # If we're still alive, force exit
            logger.info("Forcing current process exit")
            os._exit(0)
        else:
            # No parent (shouldn't happen, but handle anyway)
            logger.warning("No parent process found, killing self")
            os.kill(current_pid, signal.SIGKILL)
            os._exit(0)

    except Exception as e:
        logger.error(f"EXCEPTION during shutdown: {e}", exc_info=True)
        # Exit anyway to complete restart (new process already started)
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
