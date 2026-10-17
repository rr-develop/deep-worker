from __future__ import annotations

import json
import os
import pty
import re
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

CLI_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = CLI_ROOT / "deep-worker"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(CLI_ROOT), str(CLI_ROOT.parent / "deepagents"))
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("DEEP_WORKER_WEB_RELOAD", None)
    env.pop("DEEP_WORKER_TREE_ROOT", None)
    env.pop("DEEP_WORKER_DEV_TREE", None)
    return env


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _stop_pid(pid: int) -> None:
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        if not select.select([fd], [], [], 30)[0]:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            assert select.select([fd], [], [], 30)[0]
    finally:
        os.close(fd)


def _start(entry: list[str], port: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*entry, "web", "--port", str(port)], capture_output=True,
        text=True, env=_env(), cwd="/tmp", timeout=120,
    )


@pytest.mark.parametrize("entry", [[str(WRAPPER)], [sys.executable, "-m", "deep_worker_cli.cli.router"]])
def test_background_server_survives_terminal_and_shell_death(entry):
    port = _port()
    shell, terminal = pty.fork()
    if shell == 0:
        os.chdir("/tmp")
        os.execvpe(
            "/bin/bash",
            ["bash", "--noprofile", "--norc", "-c",
             '"$@"; code=$?; printf "\\n__EXIT__:%s\\n" "$code"; read -r unused',
             "web-test", *entry, "web", "--port", str(port)],
            _env(),
        )
    server = None
    output = b""
    try:
        deadline = time.monotonic() + 120
        while b"__EXIT__:" not in output and b"Uvicorn running on" not in output:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and select.select([terminal], [], [], remaining)[0], output
            output += os.read(terminal, 65536)
        output = re.sub(rb"\x1b\[[0-9;]*m", b"", output)
        returned = b"__EXIT__:0" in output
        match = re.search(rb"PID: (\d+)", output) or re.search(rb"Started server process \[(\d+)\]", output)
        assert match, output
        server = int(match.group(1))
        sid = os.getsid(server)
        pgid = os.getpgid(server)
        shell_sid = os.getsid(shell)
        shell_pgid = os.getpgid(shell)
        server_handle = os.pidfd_open(server)
        print(subprocess.check_output(["ps", "-p", str(server), "-o", "pid,ppid,pgid,sid,tty,cmd"], text=True))
        descriptors = []
        for path in Path(f"/proc/{server}/fd").iterdir():
            try:
                descriptors.append(os.readlink(path))
            except FileNotFoundError:
                pass
        stdin = os.readlink(f"/proc/{server}/fd/0")
        fd = os.pidfd_open(shell)
        os.killpg(shell, signal.SIGHUP)
        os.close(terminal)
        terminal = -1
        assert select.select([fd], [], [], 30)[0]
        os.close(fd)
        os.waitpid(shell, 0)
        try:
            assert not select.select([server_handle], [], [], 0)[0], "server died with the terminal"
            signal.pidfd_send_signal(server_handle, signal.SIGHUP)
        finally:
            os.close(server_handle)
        for path in ("/static/index.html", "/health", "/api/agents"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as response:
                assert response.status == 200
                if path == "/health":
                    assert json.load(response)["port"] == port
        assert returned, "the launch command never returned after successful startup"
        assert sid != shell_sid and pgid != shell_pgid
        assert not any(name.startswith("/dev/pts/") or name == "/dev/tty" for name in descriptors)
        assert stdin == "/dev/null"
        log = Path(re.search(rb"Log: ([^\r\n]+)", output).group(1).decode())
        assert log.is_file()
        errors = Path(re.search(rb"Errors: ([^\r\n]+)", output).group(1).decode())
        result = subprocess.run(
            [*entry, "web", "--stop", "--port", str(port)],
            env=_env(), cwd="/tmp", capture_output=True, text=True, timeout=45,
        )
        assert result.returncode == 0, result.stderr
        assert "GET /health" in log.read_text() + errors.read_text()
        server = None
    finally:
        if terminal >= 0:
            os.close(terminal)
        if server is not None:
            _stop_pid(server)
        try:
            os.kill(shell, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(shell, 0)
        except ChildProcessError:
            pass


def test_occupied_port_fails_without_starting_a_second_server():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        result = _start([str(WRAPPER)], occupied.getsockname()[1])
    assert result.returncode == 1
    assert "cannot bind" in result.stderr
    assert "PID:" not in result.stdout


def test_stop_refuses_a_pid_file_for_an_unrelated_process():
    from deep_worker_cli.process.web_daemon import _identity, state_path

    port = _port()
    path = state_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "start_ticks": _identity(os.getpid())}))
    result = subprocess.run(
        [str(WRAPPER), "web", "--stop", "--port", str(port)],
        env=_env(), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "refusing to stop a different process" in result.stderr


def test_managed_server_can_restart_on_the_same_port():
    from deep_worker_cli.process.web_daemon import state_path

    port = _port()
    pid = None
    try:
        for _ in range(2):
            result = _start([str(WRAPPER)], port)
            assert result.returncode == 0, result.stderr
            pid = int(re.search(r"PID: (\d+)", result.stdout).group(1))
            state = json.loads(state_path(port).read_text())
            assert state["pid"] == pid
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
                assert response.status == 200
            duplicate = _start([str(WRAPPER)], port)
            assert duplicate.returncode == 1
            assert "cannot bind" in duplicate.stderr
            stopped = subprocess.run(
                [str(WRAPPER), "web", "--stop", "--port", str(port)],
                capture_output=True, text=True, env=_env(), timeout=45,
            )
            assert stopped.returncode == 0, stopped.stderr
            pid = None
            assert not state_path(port).exists()
    finally:
        if pid is not None:
            _stop_pid(pid)


def test_failed_startup_returns_failure_without_leaving_a_daemon(tmp_path):
    from deep_worker_cli.process.web_daemon import state_path

    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class Block:\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname == 'app.main': raise ImportError('startup control')\n"
        "sys.meta_path.insert(0, Block())\n"
    )
    env = _env()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env["PYTHONPATH"]
    port = _port()
    result = subprocess.run(
        [str(WRAPPER), "web", "--port", str(port)],
        env=env, cwd="/tmp", capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1
    assert "see" in result.stderr
    assert "PID:" not in result.stdout
    assert not state_path(port).exists()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
