"""Cross-process server harness for the integration tests in this dir.

Why this exists
---------------
``rpyc_async`` MUST NOT drive an ``AsyncioServer`` and its client in
the same process / same event loop — first-time ``HANDLE_INSPECT``
deadlocks the reader-in-callback (see
``docs/RPYC_NETREF_DEADLOCK_ANALYSIS.md`` and upstream commit
``18d40df test: forbid same-process server+client topology``).

So tests in this directory launch the server in a child Python
process via ``asyncio.create_subprocess_exec``, read the announced
port from the child's stdout with ``await stdout.readline()``
(kernel-level event-driven, no polling), then connect to it from
the test's own asyncio loop via ``async_connect``.

Shutdown is a ``proc.terminate()`` (SIGTERM); the child's
``add_signal_handler`` flips its asyncio.Event and closes the
server gracefully.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

_SERVER_SCRIPT = Path(__file__).with_name("_mock_agent_service_server.py")


@asynccontextmanager
async def mock_agent_service_subprocess(
    name: str = "test-agent",
    working_dir: str = "/tmp",
    *,
    ready_timeout: float = 15.0,
) -> AsyncIterator[tuple[asyncio.subprocess.Process, int]]:
    """Run ``_mock_agent_service_server.py`` as a child process.

    Yields ``(proc, port)``. The child writes ``PORT <int>\\n`` to its
    stdout the moment its ``AsyncioServer`` is listening; we await
    that line — no polling.

    Cleanup: SIGTERM the child, ``await proc.wait()``. The child's
    ``shutdown_event`` resolves through ``add_signal_handler`` and
    closes the server cleanly. If the child does not exit within
    5 s of SIGTERM we follow up with SIGKILL.

    Memory-safety:
    * The child process is the natural scope of every rpyc object
      it creates — they die with it. No long-lived references on
      the parent side.
    * On the parent we only hold ``proc`` (an ``asyncio.subprocess.Process``)
      until the context exits, then release it.
    """
    # Run python directly so the child does not depend on package
    # entry points; the script is self-contained.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_SERVER_SCRIPT),
        "--name", name,
        "--working-dir", working_dir,
        "--port", "0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Detach from any TTY job-control so SIGINT to the test
        # runner does not also kill the child unexpectedly.
        start_new_session=True,
    )

    try:
        # Wait for the readiness line. ``readline`` is event-driven
        # on a non-blocking pipe — the loop only resumes us when
        # the kernel signals readable data. No timer poll.
        line = await asyncio.wait_for(
            proc.stdout.readline(),  # type: ignore[union-attr]
            timeout=ready_timeout,
        )
        if not line:
            # Child died before announcing. Drain stderr for a useful
            # error.
            stderr = b""
            try:
                stderr = await asyncio.wait_for(
                    proc.stderr.read(),  # type: ignore[union-attr]
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                pass
            raise RuntimeError(
                "mock agent-service subprocess exited before announcing port. "
                f"stderr: {stderr.decode('utf-8', 'replace')!r}"
            )

        token, _, port_str = line.decode("ascii", "replace").strip().partition(" ")
        if token != "PORT":
            raise RuntimeError(
                f"unexpected first line from subprocess: {line!r}"
            )
        port = int(port_str)
        if port <= 0:
            raise RuntimeError(f"subprocess returned non-positive port: {port}")

        yield proc, port

    finally:
        # SIGTERM → child's signal handler sets shutdown_event,
        # AsyncioServer closes cleanly. No polling.
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                with _suppress_async_exc():
                    await proc.wait()
        # Drain stdout/stderr pipes so file descriptors are closed
        # and no Task/Future is left lingering.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.feed_eof()  # type: ignore[attr-defined]
                except Exception:
                    pass


from contextlib import contextmanager


@contextmanager
def _suppress_async_exc():
    try:
        yield
    except (asyncio.CancelledError, ProcessLookupError):
        pass
    except Exception:  # noqa: BLE001 — best-effort drain
        pass
