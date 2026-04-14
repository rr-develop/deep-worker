"""Main entry point for deep-worker-web application.

CRITICAL SECURITY NOTICE:
========================
This application MUST ONLY run on 127.0.0.1:7887.
The config module enforces this requirement.
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

    # Security validation (config.py already validates on import, but be explicit)
    print(f"🔒 SECURITY: Server will bind to {settings.host}:{settings.port} ONLY")
    print("⚠️  This application MUST NOT be exposed to external networks!")
    print()

    # Double-check security
    if settings.host != "127.0.0.1":
        print(f"❌ CRITICAL SECURITY ERROR: Host must be 127.0.0.1! Current: {settings.host}")
        sys.exit(1)

    if settings.port != 7887:
        print(f"❌ CRITICAL SECURITY ERROR: Port must be 7887! Current: {settings.port}")
        sys.exit(1)

    print(f"✅ Security check passed: {settings.host}:{settings.port}")
    print(f"🚀 Starting deep-worker-web server...")
    print(f"📝 Open http://{settings.host}:{settings.port} in your browser")
    print()

    # Start server
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
