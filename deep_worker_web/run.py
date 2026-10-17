"""Main entry point for deep-worker-web application.

CRITICAL SECURITY NOTICE:
========================
This application MUST ONLY run on 127.0.0.1. That is enforced and
cannot be switched off.

``app.config`` derives the DEFAULT port from the checkout location at
import time:

- 7887 for the production / hot-fix checkout
- 7888 for the developer checkout
- 7887 anywhere else, including an installed wheel

It no longer refuses to start outside those trees — that refusal is
precisely what kept the web UI unusable from a pip install.
"""

import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from types import FrameType


def _bootstrap_sys_path() -> list[str]:
    """Make this checkout's sibling packages importable before any app import.

    The server MUST NOT depend on an ambient ``PYTHONPATH`` to find its own
    tree. It used to: ``app.routers.agents`` imports
    ``deep_worker_cli.model_predicate`` at module scope, so launching
    ``run.py`` without a hand-crafted ``PYTHONPATH`` died with
    ``ModuleNotFoundError: No module named 'deep_worker_cli'``.

    That turned every env-dropping launcher into an outage. The in-process
    restart helper (``app/routers/server.py``) is exactly such a launcher: it
    spawns ``run.py`` with no ``env=`` and with stdout/stderr at DEVNULL, so
    the traceback was swallowed and the web server simply never came back.

    Paths are resolved relative to THIS file, so a dev checkout always loads
    its own packages and never the prod tree's. Kept in sync with the agent
    spawn path list in ``app/routers/agents.py``.
    """
    tree_root = Path(__file__).resolve().parent.parent
    candidates = [
        tree_root / "deep_worker" / "deep_worker_cli",
        tree_root / "deep_worker" / "deepagents",
        Path("/path/to/rpyc_async"),
    ]

    # This file's own directory must be importable for ``import app.*`` to
    # work when run.py is invoked by absolute path from another cwd.
    web_root = Path(__file__).resolve().parent
    added: list[str] = []
    for path in [web_root, *candidates]:
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
    return added


# MUST run before ``app.*`` is imported — app.routers.agents needs
# deep_worker_cli at import time.
_BOOTSTRAPPED_PATHS = _bootstrap_sys_path()

import uvicorn

from app.config import settings

# Rotating log: ≤2 GB total. Path is hard-coded so the file can't be
# silently redirected via shell `> /tmp/X.log` again (that leaked 64 GB
# once stdout/stderr were captured by uvicorn into an unrotated sink).
# _setup_logging() also takes over sys.stdout/sys.stderr, so even
# nohup-style redirects route through the RotatingFileHandler.
_LOG_DIR = Path("/tmp/deep_worker_web")
_LOG_FILE = _LOG_DIR / "deep_worker_web.log"
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


def _enforce_importable_tree() -> None:
    """Refuse to start if the tree's own packages are not importable.

    Hard guard against a repeat of the ``ModuleNotFoundError:
    deep_worker_cli`` startup crash. Failing here — with an explanatory
    message on fd 2 — beats importing a half-wired app that only explodes
    later on the first request to /api/agents.
    """
    import importlib.util

    # Only the CLI package is checked here. ``app.config`` is already imported
    # at run.py module load, so re-checking it in main() would be redundant.
    # The residual value of this guard is an early, explicit failure for the
    # sibling package that uvicorn would otherwise only miss later — it loads
    # ``app.routers`` lazily from the "app.main:app" string, so a broken path
    # would surface inside the worker on the first /api/agents request.
    required = ("deep_worker_cli.model_predicate",)
    missing = []
    for name in required:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)

    if missing:
        os.write(
            2,
            (
                "FATAL: required modules not importable: "
                f"{', '.join(missing)}\n"
                f"sys.path bootstrap added: {_BOOTSTRAPPED_PATHS}\n"
                "The server resolves these from its own checkout; a missing "
                "path means the tree layout changed. Refusing to start.\n"
            ).encode(),
        )
        sys.exit(1)


def signal_handler(sig: int, frame: FrameType | None) -> None:
    """Handle interrupt signals gracefully."""
    print(f"\nReceived interrupt signal ({sig}), shutting down gracefully...")
    sys.exit(0)


def main() -> None:
    """Start the deep-worker-web server."""
    # Logging setup MUST come before anything else writes to stdout/stderr.
    # _enforce_rotating_logging() then aborts startup if rotation isn't active.
    _setup_logging()
    _enforce_rotating_logging()
    _enforce_importable_tree()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)

    # Security validation (explicit check before server start).
    # ``settings.validate_security()`` enforces the one hard invariant —
    # loopback only. Re-running it here catches a host mutated between
    # import and main() and reports it with a friendly message.
    #
    # It deliberately does NOT police the (checkout -> port) pairing any
    # more: that whitelist refused to start from a pip install, and its
    # port branch accepted an occupied port while rejecting a free one.
    # A busy port is reported by bind() with the OS's own message.
    print(f"🔒 SECURITY: Server will bind to {settings.host}:{settings.port} ONLY")
    print(f"📂 Active checkout: {settings.checkout_root or 'installed (no checkout)'}")
    print("⚠️  This application MUST NOT be exposed to external networks!")
    print()

    try:
        settings.validate_security()
    except ValueError as exc:
        print(f"❌ CRITICAL SECURITY ERROR: {exc}")
        sys.exit(1)

    print(f"✅ Security check passed: {settings.host}:{settings.port}")
    print(f"🚀 Starting deep-worker-web server...")
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
