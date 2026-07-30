"""Live integration test for the web server Restart Server endpoint.

This covers the regression where POST /api/server/restart returned success but
left the web server port down. It starts the real dev checkout server on its
configured port (127.0.0.1:7888), triggers the restart endpoint, and verifies
that a replacement process comes back on the same port with a different PID.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
import pytest

WEB_APP_DIR = Path(__file__).resolve().parents[2]
WEB_APP_RUN_PY = WEB_APP_DIR / "run.py"
DEV_URL = "http://127.0.0.1:7888"
DEV_PORT = 7888


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _health() -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(f"{DEV_URL}/health", timeout=0.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _wait_for_health(timeout: float = 10.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = _health()
        if data and data.get("status") == "healthy":
            return data
        time.sleep(0.1)
    raise AssertionError(f"{DEV_URL}/health did not become healthy within {timeout}s")


def _pid_listening_on_port(port: int) -> int | None:
    for conn in psutil.net_connections(kind="tcp"):
        if not conn.laddr:
            continue
        if conn.status != psutil.CONN_LISTEN:
            continue
        if conn.laddr.ip == "127.0.0.1" and conn.laddr.port == port:
            return conn.pid
    return None


def _wait_for_listening_pid(port: int, *, not_pid: int, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = _pid_listening_on_port(port)
        if pid is not None and pid != not_pid:
            return pid
        time.sleep(0.1)
    raise AssertionError(f"port {port} did not get a replacement listener")


def _terminate_pid(pid: int | None) -> None:
    if pid is None:
        return
    with contextlib.suppress(psutil.NoSuchProcess):
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.integration
def test_restart_endpoint_replaces_real_dev_server_process() -> None:
    if _port_is_open(DEV_PORT):
        pytest.skip(f"127.0.0.1:{DEV_PORT} already in use; refusing to kill an unknown dev server")

    proc = subprocess.Popen(
        [sys.executable, str(WEB_APP_RUN_PY)],
        cwd=str(WEB_APP_DIR),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    replacement_pid: int | None = None
    try:
        _wait_for_health()
        old_pid = proc.pid
        assert _pid_listening_on_port(DEV_PORT) == old_pid

        request = urllib.request.Request(f"{DEV_URL}/api/server/restart", method="POST")
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["success"] is True

        replacement_pid = _wait_for_listening_pid(DEV_PORT, not_pid=old_pid)
        _wait_for_health()

        assert replacement_pid != old_pid
        assert proc.poll() is not None, "old server process should have exited"
        assert _pid_listening_on_port(DEV_PORT) == replacement_pid
    finally:
        if proc.poll() is None:
            _terminate_pid(proc.pid)
        _terminate_pid(replacement_pid)
        # Give the OS a moment to release 7888 before the next test/session.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and _port_is_open(DEV_PORT):
            time.sleep(0.1)
