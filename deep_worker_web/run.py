"""Main entry point for the web application.

CRITICAL SECURITY NOTICE:
========================
This application MUST ONLY run on 127.0.0.1 on one of two approved
ports — and no other:

- 7887 for the production / hot-fix checkout
- 7888 for the developer checkout

``app.config`` derives the correct port from the checkout location
at import time and refuses to start anywhere else.
"""

import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from types import FrameType

import uvicorn

from app.config import settings

# Rotating log: ≤2 GB total. Path is hard-coded so the file can't be
# silently redirected via shell `> /tmp/X.log` again (that leaked 64 GB
# once stdout/stderr were captured by uvicorn into an unrotated sink).
# _setup_logging() also takes over sys.stdout/sys.stderr, so even
# nohup-style redirects route through the RotatingFileHandler.
_LOG_DIR = Path("/tmp/web_app")
_LOG_FILE = _LOG_DIR / "web_app.log"
_LOG_MAX_BYTES = 512 * 1024 * 1024
_LOG_BACKUP_COUNT = 3


class _StreamToLogger:
    """File-like wrapper that forwards writes to a logger.

    Used to capture stray ``print()`` calls and uncaught-exception tracebacks
    that Python writes directly to ``sys.stderr``.
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._logger.log(self._level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            self._logger.log(self._level, self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        return False


def _setup_logging() -> None:
    """Install rotating file logging and redirect stdout/stderr to it.

    The server MUST NOT start without log rotation.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    # Replace existing handlers so app/main.py's basicConfig() doesn't
    # also keep a stdout StreamHandler alive (which would be captured
    # by any outer shell redirect).
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)


def _enforce_rotating_logging() -> None:
    """Refuse to start if rotating logging is not active.

    This is a hard guard against future regressions: even if _setup_logging
    is removed, replaced, or fails silently, the server will not start.
    """
    root = logging.getLogger()
    has_rotating = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    if not has_rotating:
        # Write directly to fd 2 — sys.stderr may not be usable yet.
        os.write(2, b"FATAL: rotating log handler not installed; refusing to start.\n")
        sys.exit(1)


def signal_handler(sig: int, frame: FrameType | None) -> None:
    """Handle interrupt signals gracefully."""
    print(f"\nReceived interrupt signal ({sig}), shutting down gracefully...")
    sys.exit(0)


def main() -> None:
    """Start the web app server."""
    # Logging setup MUST come before anything else writes to stdout/stderr.
    # _enforce_rotating_logging() then aborts startup if rotation isn't active.
    _setup_logging()
    _enforce_rotating_logging()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)

    # Security validation (config.py already validates on import, but be explicit).
    # `settings.validate_security()` enforces the (checkout -> port) pairing —
    # 7887 for the production tree, 7888 for the developer tree — so a
    # second sanity check here just re-runs it with a friendly error message
    # in case settings got mutated between import and main().
    print(f"🔒 SECURITY: Server will bind to {settings.host}:{settings.port} ONLY")
    print(f"📂 Active checkout: {settings.checkout_root}")
    print("⚠️  This application MUST NOT be exposed to external networks!")
    print()

    try:
        settings.validate_security()
    except ValueError as exc:
        print(f"❌ CRITICAL SECURITY ERROR: {exc}")
        sys.exit(1)

    print(f"✅ Security check passed: {settings.host}:{settings.port}")
    print(f"🚀 Starting the web app server...")
    print(f"📝 Open http://{settings.host}:{settings.port} in your browser")
    print()

    # Start server. ``reload`` is intentionally orthogonal to ``debug``
    # — see docs/DESIGN_AGENT_SERVICE_MEMORY_FIX.md. Opt in via
    # ``DEEP_WORKER_WEB_RELOAD=1`` only when actively iterating on code.
    #
    # ``log_level`` follows ``settings.debug``. ``settings.debug``
    # defaults to False — DO NOT flip that default without reading
    # a related internal incident analysis (not included here). DEBUG turns
    # uvicorn into a record-per-event firehose that fragments the
    # glibc heap into multi-GB RSS within hours on a long-lived
    # process. The condition stays here so a developer can still opt
    # in via ``DEEP_WORKER_WEB_DEBUG=1`` for a short session, but the
    # default path must always go to "info".
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
