from __future__ import annotations

import argparse
import asyncio
import json
import os
import select
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from deep_worker_cli.storage_paths import deep_worker_home, ensure_private_dir

_START_TIMEOUT = 90.0


def state_path(port: int) -> Path:
    return deep_worker_home() / "services_shared" / "web" / f"{port}.json"


def _identity(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _write_state(port: int) -> dict[str, Any]:
    path = state_path(port)
    ensure_private_dir(path.parent)
    state = {"pid": os.getpid(), "start_ticks": _identity(os.getpid()), "port": port}
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(state, stream)
    os.replace(temporary, path)
    return state


def _remove_state(port: int, pid: int) -> None:
    path = state_path(port)
    try:
        if json.loads(path.read_text())["pid"] == pid:
            path.unlink()
    except (OSError, ValueError, KeyError):
        pass


def stop(port: int) -> int:
    path = state_path(port)
    try:
        state = json.loads(path.read_text())
        pid = int(state["pid"])
        handle = os.pidfd_open(pid)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"deep-worker web: no managed server on port {port}: {exc}", file=sys.stderr)
        return 1
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if (
            _identity(pid) != state["start_ticks"]
            or b"deep_worker_cli.process.web_daemon" not in command
            or state.get("port") != port
            or command[command.index(b"--port") + 1] != str(port).encode()
        ):
            print("deep-worker web: refusing to stop a different process", file=sys.stderr)
            return 1
        signal.pidfd_send_signal(handle, signal.SIGTERM)
        if not select.select([handle], [], [], 30.0)[0]:
            print(f"deep-worker web: PID {pid} did not stop within 30 seconds", file=sys.stderr)
            return 1
        _remove_state(port, pid)
        print(f"Stopped web server on port {port} (PID {pid})")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"deep-worker web: cannot stop server: {exc}", file=sys.stderr)
        return 1
    finally:
        os.close(handle)


def start(host: str, port: int, reload: bool) -> int:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(2048)
    except OSError as exc:
        listener.close()
        print(f"deep-worker web: cannot bind {host}:{port}: {exc}", file=sys.stderr)
        return 1
    port = listener.getsockname()[1]
    reader, writer = socket.socketpair()
    log_dir = deep_worker_home() / "logs"
    name = f"web-{port}"
    ensure_private_dir(log_dir)
    argv = [
        sys.executable, "-m", "deep_worker_cli.process.web_daemon",
        "--port", str(port), "--listen-fd", str(listener.fileno()),
        "--ready-fd", str(writer.fileno()),
    ]
    if reload:
        listener.close()
        reader.close()
        writer.close()
        print("deep-worker web: --reload requires --foreground", file=sys.stderr)
        return 1
    child = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, close_fds=True,
        pass_fds=(listener.fileno(), writer.fileno()),
        start_new_session=True,
    )
    writer.close()
    listener.close()
    pid = None
    handle = None
    success = False
    buffer = b""
    import time

    deadline = time.monotonic() + _START_TIMEOUT
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([reader], [], [], remaining)[0]:
                raise TimeoutError("server startup timed out")
            data = reader.recv(4096)
            if not data:
                raise RuntimeError("server exited before becoming ready")
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                event = json.loads(line)
                if "pid" in event and pid is None:
                    pid = int(event["pid"])
                    handle = os.pidfd_open(pid)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("ready"):
                    success = True
                    print(f"Web UI: http://{host}:{port}")
                    print(f"PID: {pid}")
                    print(f"Log: {log_dir / (name + '.out.log')}")
                    print(f"Errors: {log_dir / (name + '.err.log')}")
                    return 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(
            f"deep-worker web: {exc}; see {log_dir / (name + '.err.log')}",
            file=sys.stderr,
        )
        return 1
    finally:
        reader.close()
        if handle is not None:
            if not success:
                try:
                    signal.pidfd_send_signal(handle, signal.SIGTERM)
                    if not select.select([handle], [], [], 30.0)[0]:
                        signal.pidfd_send_signal(handle, signal.SIGKILL)
                        select.select([handle], [], [], 30.0)
                except ProcessLookupError:
                    pass
                if pid is not None:
                    _remove_state(port, pid)
            os.close(handle)
        child.wait(timeout=30)


def serve(host: str, port: int, debug: bool, listener: socket.socket, ready: socket.socket) -> int:
    import uvicorn

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets=sockets)
            if self.started:
                _write_state(port)
                ready.sendall(json.dumps({"ready": True}).encode() + b"\n")
                ready.close()

    server = ReadyServer(uvicorn.Config(
        "app.main:app", host=host, port=port,
        log_level="debug" if debug else "info",
    ))
    try:
        asyncio.run(server.serve(sockets=[listener]))
        return 0 if server.started else 1
    finally:
        ready.close()
        listener.close()
        _remove_state(port, os.getpid())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--listen-fd", type=int, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args()
    channel = socket.socket(fileno=args.ready_fd)
    listener = socket.socket(fileno=args.listen_fd)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    from deep_worker_cli.process.daemon import daemonize

    daemonize(f"web-{args.port}")
    try:
        channel.sendall(json.dumps({"pid": os.getpid()}).encode() + b"\n")
        from deep_worker_cli.cli.router import _dispatch_web

        return _dispatch_web(
            ["web", "--foreground", "--port", str(args.port)],
            listener=listener, ready=channel,
        ) or 0
    except BaseException as exc:
        try:
            channel.sendall(json.dumps({"error": str(exc)}).encode() + b"\n")
        except OSError:
            pass
        raise
    finally:
        channel.close()
        listener.close()


if __name__ == "__main__":
    sys.exit(main())
