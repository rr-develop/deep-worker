from __future__ import annotations

import errno
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
        [*entry, "--web", "--port", str(port)], capture_output=True,
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
             "web-test", *entry, "--web", "--port", str(port)],
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
            [*entry, "--web", "--stop", "--port", str(port)],
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
        [str(WRAPPER), "--web", "--stop", "--port", str(port)],
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
                [str(WRAPPER), "--web", "--stop", "--port", str(port)],
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
        [str(WRAPPER), "--web", "--port", str(port)],
        env=env, cwd="/tmp", capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1
    assert "see" in result.stderr
    assert "PID:" not in result.stdout
    assert not state_path(port).exists()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))


# ---------------------------------------------------------------------------
# Kernels without the pidfd-era syscalls.
#
# The supported floor is Linux 4.19 (docs/REQUIREMENTS.md): WSL2 ships
# kernels that old, and uv-installed CPython is built without
# ``<sys/pidfd.h>`` on any kernel. Both answer ENOSYS. The start path
# used to call ``pidfd_open`` unconditionally, so ``--web`` died with a
# bare "[Errno 38] Function not implemented" on a server that had
# started perfectly well.
#
# Every post-4.19 call the web path can reach is gagged below, not just
# the one the user happened to report:
#
#   os.pidfd_open             Linux 5.3
#   signal.pidfd_send_signal  Linux 5.1  — a SEPARATE floor, so it can
#                                          be missing where pidfd_open
#                                          worked; that is the branch
#                                          ``_signal`` falls back from.
#   os.waitid(P_PIDFD)        Linux 5.4  — unused by the web path; gagged
#                                          so a future use cannot pass
#                                          untested.
# ---------------------------------------------------------------------------

_ENOSYS_STUB = (
    "import errno, os, signal\n"
    "def _gone(*a, **k):\n"
    " raise OSError(errno.ENOSYS, 'Function not implemented')\n"
    "os.pidfd_open = _gone\n"
    "signal.pidfd_send_signal = _gone\n"
    "_real_waitid = getattr(os, 'waitid', None)\n"
    "if _real_waitid is not None:\n"
    " _P_PIDFD = getattr(os, 'P_PIDFD', None)\n"
    " def _waitid(idtype, *a, **k):\n"
    "  if _P_PIDFD is not None and idtype == _P_PIDFD: _gone()\n"
    "  return _real_waitid(idtype, *a, **k)\n"
    " os.waitid = _waitid\n"
)

def _isolated_env(home):
    """``_env()`` pointed at a throwaway HOME.

    These tests must not write into the real ``~/.deep_worker``: they
    start servers and leave state files behind.
    """
    env = _env()
    env["HOME"] = str(home)
    return env


def test_web_starts_when_pidfd_open_is_unavailable(tmp_path):
    """ENOSYS from ``pidfd_open`` must not fail a healthy start."""
    (tmp_path / "sitecustomize.py").write_text(_ENOSYS_STUB)
    home = tmp_path / "home"
    home.mkdir()
    env = _isolated_env(home)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env["PYTHONPATH"]
    port = _port()
    result = subprocess.run(
        [str(WRAPPER), "--web", "--port", str(port)],
        env=env, cwd="/tmp", capture_output=True, text=True, timeout=120,
    )
    pid = None
    try:
        assert result.returncode == 0, f"stderr={result.stderr}"
        assert "Function not implemented" not in result.stderr
        match = re.search(r"^PID: (\d+)$", result.stdout, re.M)
        assert match, f"no PID reported; stdout={result.stdout}"
        pid = int(match.group(1))
        # Ready means serving, not merely spawned.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as response:
            assert response.status == 200
        # The server is discoverable despite the missing syscall, so it
        # can never become a silent orphan.
        status = subprocess.run(
            [str(WRAPPER), "--web", "--status", "--port", str(port)],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert status.returncode == 0
        assert str(pid) in status.stdout
    finally:
        if pid is not None:
            _stop_pid(pid)


def test_startup_failure_is_reported_on_channel_eof_without_pidfd(tmp_path):
    """A child dying before ready is an EOF, not a pidfd event."""
    (tmp_path / "sitecustomize.py").write_text(
        _ENOSYS_STUB
        + "import sys\n"
        "class Block:\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname == 'app.main': raise ImportError('startup control')\n"
        "sys.meta_path.insert(0, Block())\n"
    )
    home = tmp_path / "home"
    home.mkdir()
    env = _isolated_env(home)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env["PYTHONPATH"]
    port = _port()
    result = subprocess.run(
        [str(WRAPPER), "--web", "--port", str(port)],
        env=env, cwd="/tmp", capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1
    assert "PID:" not in result.stdout
    # The verdict comes from the channel closing, and says so.
    assert "server exited before becoming ready" in result.stderr
    assert "Function not implemented" not in result.stderr
    assert "see" in result.stderr
    # No daemon and no state left behind, even with no pidfd to wait on.
    from deep_worker_cli.process.web_daemon import state_path

    assert not (home / ".deep_worker" / "services_shared" / "web" / f"{port}.json").exists()
    assert not state_path(port).exists()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))



def _assert_dead(pid: int) -> None:
    """Block until ``pid`` is gone, event-driven, or fail.

    ``start`` already waited for the child itself, so by here the death
    has normally happened. The pidfd makes the confirmation a single
    kernel wakeup instead of a poll; if the syscall is unavailable the
    check degrades to one authoritative ``kill(pid, 0)``.
    """
    try:
        handle = os.pidfd_open(pid)
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        raise AssertionError(f"child {pid} survived a failed start")
    try:
        assert select.select([handle], [], [], 30)[0], (
            f"child {pid} survived a failed start"
        )
    finally:
        os.close(handle)


#: A child that announces its pid, declines to become ready, and then
#: stays alive until signalled. A child that dies on its own (an import
#: error, say) proves nothing about signal delivery: the port frees
#: itself either way. Only a live child makes the cleanup's SIGTERM the
#: single thing standing between a failed start and a stray daemon.
_LIVE_CHILD_THAT_NEVER_READIES = '''
import argparse, json, os, signal, socket, sys

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--listen-fd", type=int, required=True)
parser.add_argument("--ready-fd", type=int, required=True)
args = parser.parse_args()
channel = socket.socket(fileno=args.ready_fd)
channel.sendall(json.dumps({"pid": os.getpid()}).encode() + b"\\n")
channel.sendall(json.dumps({"error": "declining to become ready"}).encode() + b"\\n")
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
while True:
    signal.pause()
'''


def test_failed_start_reaps_a_live_child_when_pidfd_signalling_is_missing(
    tmp_path, monkeypatch
):
    """``pidfd_send_signal``'s floor (5.1) is not ``pidfd_open``'s (5.3).

    With ``pidfd_open`` alive a handle IS obtained, so cleanup takes the
    pidfd branch and nothing else can deliver the signal. If that branch
    swallows ENOSYS without falling back to ``kill``, a failed start
    leaves the daemon running — which is what the first version of this
    fix did.
    """
    from deep_worker_cli.process import web_daemon

    child_script = tmp_path / "child.py"
    child_script.write_text(_LIVE_CHILD_THAT_NEVER_READIES)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    # Stand in for the daemon module with a child that stays alive.
    real_popen = subprocess.Popen

    def fake_popen(argv, **kwargs):
        rewritten = [argv[0], str(child_script), *argv[3:]]
        return real_popen(rewritten, **kwargs)

    monkeypatch.setattr(web_daemon.subprocess, "Popen", fake_popen)
    # The kernel has pidfd_open but not pidfd_send_signal.
    monkeypatch.setattr(
        web_daemon.signal, "pidfd_send_signal",
        lambda *a, **k: (_ for _ in ()).throw(
            OSError(errno.ENOSYS, "Function not implemented")
        ),
    )

    port = _port()
    pid_seen = {}
    real_open_pidfd = web_daemon._open_pidfd

    def recording_open_pidfd(pid):
        pid_seen["pid"] = pid
        return real_open_pidfd(pid)

    monkeypatch.setattr(web_daemon, "_open_pidfd", recording_open_pidfd)

    code = web_daemon.start("127.0.0.1", port, False)
    assert code == 1
    pid = pid_seen.get("pid")
    assert pid, "the child never announced a pid, so nothing was tested"
    # A pidfd WAS available — otherwise this test would be exercising
    # the no-pidfd path and prove nothing about signal delivery.
    assert real_open_pidfd(os.getpid()) is not None

    # The oracle: the child is dead. With the fallback removed it stays
    # alive, because pidfd_send_signal raised and nothing else fired.
    _assert_dead(pid)


def test_failed_start_reaps_a_live_child_with_no_pidfd_at_all(tmp_path, monkeypatch):
    """Same oracle on the no-pidfd path: ``kill`` is the only option."""
    from deep_worker_cli.process import web_daemon

    child_script = tmp_path / "child.py"
    child_script.write_text(_LIVE_CHILD_THAT_NEVER_READIES)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    real_popen = subprocess.Popen

    def fake_popen(argv, **kwargs):
        rewritten = [argv[0], str(child_script), *argv[3:]]
        return real_popen(rewritten, **kwargs)

    monkeypatch.setattr(web_daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        web_daemon.os, "pidfd_open",
        lambda *a, **k: (_ for _ in ()).throw(
            OSError(errno.ENOSYS, "Function not implemented")
        ),
    )

    port = _port()
    pid_seen = {}
    real_open_pidfd = web_daemon._open_pidfd

    def recording_open_pidfd(pid):
        pid_seen["pid"] = pid
        handle = real_open_pidfd(pid)
        assert handle is None, "this test must exercise the no-pidfd path"
        return handle

    monkeypatch.setattr(web_daemon, "_open_pidfd", recording_open_pidfd)

    code = web_daemon.start("127.0.0.1", port, False)
    assert code == 1
    pid = pid_seen.get("pid")
    assert pid, "the child never announced a pid, so nothing was tested"
    _assert_dead(pid)


def test_web_path_uses_no_syscall_newer_than_the_supported_floor():
    """Guard the floor itself, not just the calls we thought to gag.

    ``_ENOSYS_STUB`` can only emulate calls someone listed. This asserts
    the inverse: the web path names no post-4.19 syscall beyond the two
    that have a documented fallback, so introducing one fails here
    rather than only on a user's old kernel.
    """
    from deep_worker_cli.process import web_daemon

    source = Path(web_daemon.__file__).read_text()
    # Post-4.19 syscalls reachable from CPython. ``pidfd_open`` and
    # ``pidfd_send_signal`` are deliberately absent from this list:
    # both are guarded (``_open_pidfd`` / ``_signal``) and gagged by
    # ``_ENOSYS_STUB``.
    forbidden = {
        "P_PIDFD": "waitid(P_PIDFD), Linux 5.4",
        "clone3": "Linux 5.3",
        "close_range": "Linux 5.9",
        "openat2": "Linux 5.6",
        "faccessat2": "Linux 5.8",
        "pidfd_getfd": "Linux 5.6",
        "memfd_secret": "Linux 5.14",
    }
    found = {name: why for name, why in forbidden.items() if name in source}
    assert not found, (
        f"{found} postdate the Linux 4.19 floor in docs/REQUIREMENTS.md; "
        "either guard it with a fallback and extend _ENOSYS_STUB, or "
        "raise the documented floor"
    )


def test_server_survives_a_start_command_that_dies_before_ready(tmp_path):
    """A working server outlives the command that could not report it.

    The parent closing the readiness channel early (its own ENOSYS, a
    Ctrl-C, a closed terminal) used to kill the server with BrokenPipe
    on ``sendall``. Reporting is not the service.
    """
    from deep_worker_cli.process import web_daemon

    home = tmp_path / "home"
    home.mkdir()
    port = _port()
    listener = socket.socket()
    listener.bind(("127.0.0.1", port))
    listener.listen(16)
    reader, writer = socket.socketpair()
    reader.close()  # the start command is gone before "ready" is sent

    # ``serve`` removes the state file on the way out, so accountability
    # is asserted at the moment readiness is reported — from inside the
    # server, which is where a real ``--web --status`` would see it.
    observed = {}

    class FakeConfig:
        def __init__(self, target, **kwargs):
            self.target = target

    class FakeServer:
        """Stands in for uvicorn: runs ``startup``, then returns."""

        def __init__(self, config):
            self.config = config
            self.started = False

        async def startup(self, sockets=None):
            self.started = True

        async def serve(self, sockets=None):
            # ``ReadyServer.startup`` is the code under test: it writes
            # the state file and reports readiness into a dead channel.
            await self.startup(sockets=sockets)
            state = web_daemon.state_path(port)
            observed["state_while_serving"] = state.exists()
            observed["recorded_pid"] = json.loads(state.read_text())["pid"]

    fake_uvicorn = type("uvicorn", (), {"Server": FakeServer, "Config": FakeConfig})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(home))
        patch.setitem(sys.modules, "uvicorn", fake_uvicorn)
        try:
            # Must not raise: the regression propagated BrokenPipeError
            # out of ``sendall`` and tore the whole server down.
            code = web_daemon.serve(
                "127.0.0.1", port, False, listener=listener, ready=writer
            )
            assert code == 0, "a reporting failure must not fail the server"
            # The server kept serving and left the record that makes it
            # discoverable rather than a silent orphan. (``status`` is
            # not called here: its identity check reads
            # ``/proc/<pid>/cmdline``, which is pytest in-process, not a
            # web_daemon — that path is covered by the subprocess test
            # above.)
            assert observed["state_while_serving"] is True
            assert observed["recorded_pid"] == os.getpid()
            assert web_daemon.state_path(port).is_relative_to(home / ".deep_worker")
        finally:
            web_daemon._remove_state(port, os.getpid())
            writer.close()
            listener.close()
