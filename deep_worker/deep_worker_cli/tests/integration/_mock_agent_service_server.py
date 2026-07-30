"""Standalone AsyncioServer hosting a mock-bound AgentService.

Used by the cross-process integration tests in this directory to honour
the project-wide rule that **rpyc_async MUST NEVER drive server+client
in the same process** (see the agent instructions and
``docs/RPYC_NETREF_DEADLOCK_ANALYSIS.md`` for the deadlock that
otherwise triggers on the first ``HANDLE_INSPECT``).

Lifecycle
---------
* Reads ``--name``, ``--working-dir``, ``--port`` (default ``0`` =
  OS-assigned), ``--ready-fd`` (default ``1`` = stdout).
* Brings up a single ``AsyncioServer`` bound to a tiny
  ``BoundAgentService`` (mocked ``_agent_process``).
* As soon as the listener has a real port, writes a single line
  ``PORT <int>\\n`` to ``--ready-fd`` (stdout by default). Tests
  consume that line with ``await proc.stdout.readline()`` — fully
  event-driven, no polling.
* Then waits on a ``shutdown_event`` that is set by SIGTERM/SIGINT.
  Closes the server cleanly on shutdown.

No claude-code provider, no checkpointer, no agent graph — this is a
*protocol-only* harness for AsyncioServer + AgentService round-trips.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

# The local rpyc_async checkout must be on sys.path BEFORE importing rpyc.
_RPYC_ASYNC = Path("/path/to/rpyc_async")
if str(_RPYC_ASYNC) not in sys.path:
    sys.path.insert(0, str(_RPYC_ASYNC))


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--working-dir", default="/tmp")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--ready-fd",
        type=int,
        default=1,
        help="File descriptor to write 'PORT <n>\\n' on once the server "
             "is listening. Default: stdout (1).",
    )
    args = parser.parse_args()

    # Import lazily so the import path is short and any failure shows
    # up in the parent's captured stderr.
    from rpyc_async.utils.async_server import AsyncioServer  # noqa: PLC0415
    from deep_worker_cli.rpyc_service.agent_service import (  # noqa: PLC0415
        AgentService,
    )

    # Minimal mock_agent — the exposed methods we exercise only touch
    # ``name`` / ``config.working_dir`` / ``session_manager`` / a few
    # optional attrs. Anything missing returns ``getattr(..., None)``.
    class _MockConfig:
        def __init__(self, working_dir: str) -> None:
            self.working_dir = working_dir
            self.resolved_model = None
            self.provider = None
            self.model = None

    class _MockAgent:
        def __init__(self, name: str, working_dir: str) -> None:
            self.name = name
            self.config = _MockConfig(working_dir)
            self.session_manager = None
            self.task_manager = None
            self.started_at = None
            self.rpyc_port: int | None = None  # filled in below
            self.config = _MockConfig(working_dir)

    mock_agent = _MockAgent(args.name, args.working_dir)

    class BoundAgentService(AgentService):
        def on_connect(self, conn) -> None:  # noqa: ANN001
            # ``AgentService.on_connect`` resets ``self._agent_process``
            # to None on every new connection — same shape as the
            # production ``BoundAgentService`` in
            # ``deep_worker_cli/process/agent_process.py``. We re-bind
            # the mock here so each connection sees it.
            super().on_connect(conn)
            self.set_agent_process(mock_agent)

    server = AsyncioServer(BoundAgentService, hostname="127.0.0.1", port=args.port)
    await server.start()
    port = server.listener.getsockname()[1]
    mock_agent.rpyc_port = port

    # Announce readiness — single line, atomic write, no polling.
    msg = f"PORT {port}\n".encode()
    try:
        os.write(args.ready_fd, msg)
    except OSError:
        # If the parent closed the readiness pipe early, fall back to
        # stdout so the test still has *some* signal.
        sys.stdout.buffer.write(msg)
        sys.stdout.buffer.flush()

    # Wait for SIGTERM / SIGINT — event-driven, no polling.
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            # On platforms without signal handlers (e.g. Windows),
            # ``shutdown.set()`` will simply never fire and the
            # parent kills the process. Acceptable for tests.
            pass

    try:
        await shutdown.wait()
    finally:
        await server.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
