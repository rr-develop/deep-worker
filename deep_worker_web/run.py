"""Main entry point for deep-worker-web application.

CRITICAL SECURITY NOTICE:
========================
This application MUST ONLY run on 127.0.0.1 on one of two approved
ports — and no other:

- 7887 for the production / hot-fix checkout
- 7888 for the developer checkout

``app.config`` derives the correct port from the checkout location
at import time and refuses to start anywhere else.
"""

import signal
import sys
from types import FrameType

import uvicorn

from app.config import settings


def signal_handler(sig: int, frame: FrameType | None) -> None:
    """Handle interrupt signals gracefully."""
    print(f"\nReceived interrupt signal ({sig}), shutting down gracefully...")
    sys.exit(0)


def main() -> None:
    """Start the deep-worker-web server."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)

    # Security validation (config.py already validates on import, but be explicit).
    # `settings.validate_security()` enforces the (checkout -> port) pairing —
    # 7887 for the production checkout, 7888 for the development checkout — so a
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
    print(f"🚀 Starting deep-worker-web server...")
    print(f"📝 Open http://{settings.host}:{settings.port} in your browser")
    print()

    # Start server. ``reload`` is intentionally orthogonal to ``debug``
    # — see docs/DESIGN_AGENT_SERVICE_MEMORY_FIX.md. Opt in via
    # ``DEEP_WORKER_WEB_RELOAD=1`` only when actively iterating on code.
    #
    # ``log_level`` follows ``settings.debug``. ``settings.debug``
    # defaults to False — DO NOT flip that default without reading the
    # analysis of the earlier debug-log-storm incident. DEBUG turns
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
