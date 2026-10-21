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

#: Attribute ``_syscall`` stamps on an ``OSError`` to name the call.
_CALL_ATTR = "deep_worker_call"


def state_path(port: int) -> Path:
    return deep_worker_home() / "services_shared" / "web" / f"{port}.json"


def _syscall(call: str, fn: Any, *args: Any) -> Any:
    """Run ``fn``, tagging any ``OSError`` with the call's name.

    ``errno`` alone is not a diagnosis: "[Errno 38] Function not
    implemented" is the same text whichever call is missing, and the
    user has no way to tell. The tag turns it into a name they can
    report and we can look up.
    """
    try:
        return fn(*args)
    except OSError as exc:
        # ``setattr`` rather than an attribute write: ``OSError`` has no
        # such slot declared, and the type checker is right to say so.
        setattr(exc, _CALL_ATTR, call)
        raise


def _open_pidfd(pid: int) -> int | None:
    """A pidfd for ``pid``, or ``None`` where the syscall is missing.

    ``os.pidfd_open`` needs the syscall (Linux >= 5.3) AND
    ``<sys/pidfd.h>`` at build time. Both are commonly missing: WSL2
    ships kernels as old as 4.19, and uv-installed CPython is built
    without the header. Either way the answer is ENOSYS.

    A pidfd is a convenience here and never a requirement — the
    readiness channel already reports both outcomes (bytes = ready,
    EOF = dead) — so an absent syscall must degrade, not abort the
    start. See ``docs/REQUIREMENTS.md`` for the supported kernel floor.
    """
    try:
        return os.pidfd_open(pid)
    except (OSError, AttributeError):
        return None


def _signal(pid: int, handle: int | None, number: int) -> None:
    """Best-effort signal, via pidfd when that works.

    ``pidfd_send_signal`` is Linux >= 5.1 — a lower floor than
    ``pidfd_open``'s 5.3, but a separate one, so it can fail even where
    a pidfd was obtained. A failure falls back to ``kill`` on the pid
    rather than leaving the child unsignalled. Racing a recycled pid is
    not a concern: every caller has already confirmed identity.
    """
    if handle is not None:
        try:
            signal.pidfd_send_signal(handle, number)
            return
        except (OSError, AttributeError):
            pass
    try:
        os.kill(pid, number)
    except OSError:
        pass


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


def _is_our_server(pid: int, state: dict[str, Any], port: int) -> bool:
    """Is ``pid`` still the web server this state file describes?

    A PID alone proves nothing: it can be recycled onto an unrelated
    process. ``start_ticks`` pins the exact process incarnation and the
    cmdline check pins what it is running, so ``status`` and ``stop``
    cannot disagree about identity.
    """
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    return not (
        _identity(pid) != state["start_ticks"]
        or b"deep_worker_cli.process.web_daemon" not in command
        or state.get("port") != port
        or command[command.index(b"--port") + 1] != str(port).encode()
    )


def status(port: int) -> int:
    """Report whether a managed server owns ``port``. 0 = running."""
    path = state_path(port)
    try:
        state = json.loads(path.read_text())
        pid = int(state["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        print(f"deep-worker --web: no managed server on port {port}")
        return 1
    try:
        alive = _is_our_server(pid, state, port)
    except (OSError, ValueError, KeyError, TypeError):
        alive = False
    if not alive:
        print(
            f"deep-worker --web: no managed server on port {port} "
            f"(stale state file for PID {pid})"
        )
        return 1
    print(f"deep-worker --web: running on http://127.0.0.1:{port} (PID {pid})")
    return 0


def stop(port: int) -> int:
    path = state_path(port)
    try:
        state = json.loads(path.read_text())
        pid = int(state["pid"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"deep-worker --web: no managed server on port {port}: {exc}", file=sys.stderr)
        return 1
    # Optional, as in ``start``: opening it eagerly used to make a
    # missing syscall indistinguishable from "no managed server".
    handle = _open_pidfd(pid)
    try:
        if not _is_our_server(pid, state, port):
            print("deep-worker --web: refusing to stop a different process", file=sys.stderr)
            return 1
        _signal(pid, handle, signal.SIGTERM)
        if handle is None:
            # Without a pidfd there is no event to wait on: the server
            # is init's child, not ours, so ``waitpid`` cannot see it
            # and anything else here would be a poll. Report the signal
            # as sent and let ``--web --status`` be the check.
            _remove_state(port, pid)
            print(f"Asked web server on port {port} to stop (PID {pid})")
            return 0
        if not select.select([handle], [], [], 30.0)[0]:
            print(f"deep-worker --web: PID {pid} did not stop within 30 seconds", file=sys.stderr)
            return 1
        _remove_state(port, pid)
        print(f"Stopped web server on port {port} (PID {pid})")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"deep-worker --web: cannot stop server: {exc}", file=sys.stderr)
        return 1
    finally:
        if handle is not None:
            os.close(handle)


def start(host: str, port: int, reload: bool) -> int:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(2048)
    except OSError as exc:
        listener.close()
        print(f"deep-worker --web: cannot bind {host}:{port}: {exc}", file=sys.stderr)
        return 1
    port = listener.getsockname()[1]
    try:
        reader, writer = socket.socketpair()
    except OSError as exc:
        listener.close()
        print(f"deep-worker --web: socketpair failed: {exc}", file=sys.stderr)
        return 1
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
        print("deep-worker --web: --reload requires --foreground", file=sys.stderr)
        return 1
    try:
        # ``start_new_session`` is setsid(2) in the child and
        # ``pass_fds`` needs fd inheritance, so this can fail on a
        # restricted kernel; the message below names it.
        child = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True,
            pass_fds=(listener.fileno(), writer.fileno()),
            start_new_session=True,
        )
    except OSError as exc:
        reader.close()
        writer.close()
        listener.close()
        print(
            f"deep-worker --web: spawn (setsid) failed: {exc}; "
            f"see {deep_worker_home() / 'logs' / (name + '.err.log')}",
            file=sys.stderr,
        )
        return 1
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
            if remaining <= 0:
                raise TimeoutError("server startup timed out")
            if not _syscall("select", select.select, [reader], [], [], remaining)[0]:
                raise TimeoutError("server startup timed out")
            data = _syscall("recv", reader.recv, 4096)
            if not data:
                raise RuntimeError("server exited before becoming ready")
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                event = json.loads(line)
                if "pid" in event and pid is None:
                    pid = int(event["pid"])
                    # Optional: see ``_open_pidfd``. Readiness is decided
                    # by this channel, so a missing syscall must not fail
                    # a server that is starting up perfectly well.
                    handle = _open_pidfd(pid)
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
        # Name the failing call. A bare "[Errno 38] Function not
        # implemented" told the user nothing about WHAT was not
        # implemented, which is the whole difficulty of diagnosing this
        # class of failure on kernels and CPython builds that lack a
        # call we assumed. ``_syscall`` tags the OSErrors it wraps.
        detail = f"{getattr(exc, _CALL_ATTR, None) or 'startup'} failed: {exc}" \
            if isinstance(exc, OSError) else str(exc)
        print(
            f"deep-worker --web: {detail}; see {log_dir / (name + '.err.log')}",
            file=sys.stderr,
        )
        return 1
    finally:
        if not success and pid is not None:
            # Reap a child that announced its pid but never became
            # ready. Keyed on ``pid``, not on ``handle``: without a
            # pidfd the cleanup still has to happen, or a failed start
            # would leave the daemon behind.
            _signal(pid, handle, signal.SIGTERM)
            if handle is not None:
                if not select.select([handle], [], [], 30.0)[0]:
                    _signal(pid, handle, signal.SIGKILL)
                    select.select([handle], [], [], 30.0)
            else:
                # No pidfd to wait on. The readiness socket is still the
                # death signal: the child holds the only other end, so
                # the kernel reports EOF when it exits. Event-driven, no
                # poll — and bounded, because SIGTERM was already sent.
                select.select([reader], [], [], 30.0)
            _remove_state(port, pid)
        if handle is not None:
            os.close(handle)
        reader.close()
        child.wait(timeout=30)


def serve(host: str, port: int, debug: bool, listener: socket.socket, ready: socket.socket) -> int:
    import uvicorn

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets=sockets)
            if self.started:
                # State first, report second: the state file is what
                # makes the server discoverable by ``--web --status``
                # and killable by ``--web --stop``, so it must exist
                # before anything can go wrong with the report.
                _write_state(port)
                try:
                    ready.sendall(json.dumps({"ready": True}).encode() + b"\n")
                except OSError:
                    # The start command died before hearing back (its
                    # own ENOSYS, a Ctrl-C, a closed terminal). It can
                    # no longer tell the user anything, but this server
                    # is up and serving — killing it would throw away
                    # working state for a reporting failure. It stays,
                    # and the state file above keeps it accountable
                    # rather than a silent orphan.
                    pass
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

        # Must be the flag spelling: ``_dispatch_web`` declines anything
        # else, and a positional "web" is an agent NAME. Getting this
        # wrong makes the daemon child fall through to the agent
        # launcher and the server never becomes ready.
        return _dispatch_web(
            ["--web", "--foreground", "--port", str(args.port)],
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
