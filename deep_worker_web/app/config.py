"""Configuration for deep-worker-web application.

CRITICAL SECURITY REQUIREMENT:
==============================
This application MUST ONLY run on 127.0.0.1. The host is always
127.0.0.1 — NO OTHER addresses are allowed for security reasons.
This application is designed for LOCAL TRUSTED ENVIRONMENT ONLY:
- No authentication/authorization
- Direct access to deep-worker agents
- Full access to agent capabilities
- Must NEVER be exposed to external networks

Violating these requirements creates serious security risks.

Why the port is pinned per checkout
-----------------------------------
Several clones of this project can co-exist on one host: a canonical
"running" copy serving the agents you rely on, plus scratch copies
where risky refactors happen so they cannot break the running agents
or the running web server.

Running them all on the same port would crash one of the servers and
would make it impossible to tell one UI from another in the browser.
So each checkout binds to exactly one port. The port defaults to 7887
and is configurable via ``DEEP_WORKER_WEB_PORT``; give every extra
checkout its own value (7888, 7889, …) to keep them operationally
isolated.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Port selection
# ---------------------------------------------------------------------------

#: Port used when ``DEEP_WORKER_WEB_PORT`` is not set.
#: Keep in sync with deep_worker_web/run.py and the project README.
_DEFAULT_PORT = 7887


def _detect_checkout_root() -> Path:
    """Return the absolute path of the checkout this file lives in.

    This module sits at ``deep_worker_web/app/config.py``; walking
    two parents up yields ``<repo>``. We resolve symlinks so the value is
    stable regardless of how the directory was referenced when the
    process started.
    """
    # config.py -> app/ -> deep_worker_web/ -> <repo root>
    return Path(__file__).resolve().parents[2]


_CHECKOUT_ROOT = _detect_checkout_root()


class Settings(BaseSettings):
    """Application settings with strict security validation."""

    host: str = "127.0.0.1"
    port: int = _DEFAULT_PORT
    # ``debug`` flips uvicorn's log level to DEBUG and turns on a
    # record-per-event firehose: every ASGI lifespan event, every WS
    # keepalive ping, every inotify wakeup gets a ``logger.debug``
    # call. On a long-lived process with even a few stuck CLOSE-WAIT
    # sockets that produces ~2700 ``logging.Formatter.formatTime``
    # calls/sec. Each call allocates two short strings; Python frees
    # them, but glibc never returns the freed arenas to the OS, so
    # RSS grows into the multi-GB range purely from heap
    # fragmentation — no Python object is leaking. One such incident
    # took the web server to 9.2 GB this way.
    # KEEP THIS FALSE BY DEFAULT. Opt in via DEEP_WORKER_WEB_DEBUG=1
    # only for short-lived debugging sessions.
    debug: bool = False
    # Uvicorn's reload feature forces a WatchFiles + multiprocessing.spawn
    # subprocess layout that we don't want paying for in production (and
    # which silently bloats memory over long-lived runs, see
    # docs/DESIGN_AGENT_SERVICE_MEMORY_FIX.md). Default OFF; opt in
    # explicitly via DEEP_WORKER_WEB_RELOAD=1 when doing dev work.
    reload: bool = False

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_prefix = "DEEP_WORKER_WEB_"

    @property
    def checkout_root(self) -> Path:
        """The repo this server is running from (resolved absolute path)."""
        return _CHECKOUT_ROOT

    def validate_security(self) -> bool:
        """Validate the host/port this server is about to bind to.

        Returns:
            True if security check passes.

        Raises:
            ValueError: If host is not 127.0.0.1, or if the port lies
                outside the unprivileged range.
        """
        if self.host != "127.0.0.1":
            raise ValueError(
                f"SECURITY ERROR: Host must be 127.0.0.1 only! Current: {self.host}\n"
                "This application must NEVER be exposed to external networks.\n"
                "Running on 0.0.0.0 or any other address is STRICTLY PROHIBITED."
            )

        if not 1024 <= self.port <= 65535:
            raise ValueError(
                f"SECURITY ERROR: port {self.port} is outside the "
                "unprivileged range 1024-65535.\n"
                "Set DEEP_WORKER_WEB_PORT to a free port in that range."
            )

        return True


# Initialize settings and validate security immediately
settings = Settings()
settings.validate_security()
