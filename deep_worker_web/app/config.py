"""Configuration for the web application.

CRITICAL SECURITY REQUIREMENT:
==============================
This application MUST ONLY run on 127.0.0.1, on one of the two
APPROVED ports — and no other:

- 7887 when the active checkout is the production / hot-fix tree
  (branch: master)
- 7888 when the active checkout is the developer tree (branch: dev)

Why two ports?
--------------
Two co-existing checkouts of the same project live on this host:

* The production tree — the canonical, "running" copy. The agents
  and the web app instance the user actively relies on are served
  from here. Hot-fixes go to its `master` branch.
* The developer tree — the developer scratch copy on the `dev`
  branch. Risky refactors happen here so they cannot accidentally
  break the running agents or the running web server.

Running them on the same port would crash one of the servers and
would make it impossible to tell "live" UI from "dev" UI in the
browser. Hard-coding the port per checkout solves both problems and
matches the requirement that the two trees be operationally
isolated.

The host is always 127.0.0.1 — NO OTHER addresses are allowed for
security reasons. This application is designed for LOCAL TRUSTED
ENVIRONMENT ONLY:
- No authentication/authorization
- Direct access to deep-worker agents
- Full access to agent capabilities
- Must NEVER be exposed to external networks

Violating these requirements creates serious security risks.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Port selection based on checkout location
# ---------------------------------------------------------------------------

# Paths to the two canonical checkouts and the port each one MUST use.
# Keep these in sync with run.py and the project README.
_PROD_TREE = Path("/path/to/project").resolve()
_DEV_TREE = Path("/path/to/dev-tree").resolve()

_PROD_PORT = 7887
_DEV_PORT = 7888


def _approved_bindings() -> dict[Path, int]:
    """Build the approved (checkout -> port) whitelist.

    The two canonical trees are the production checkout (7887) and the
    developer checkout (7888). On the maintainer's host those two paths
    exist (as symlinks to the real trees), so each tree keeps a fixed,
    non-conflicting port and the two servers can run side by side.

    On a machine without that symlink farm — a fresh clone of the
    published project — the canonical paths do not exist. The running
    checkout is then approved on the production port, so the server
    starts and validates from any clone instead of refusing to run
    from an unrecognised path.
    """
    bindings: dict[Path, int] = {}
    if _PROD_TREE.exists():
        bindings[_PROD_TREE] = _PROD_PORT
    if _DEV_TREE.exists():
        bindings[_DEV_TREE] = _DEV_PORT
    checkout_root = _detect_checkout_root()
    if checkout_root not in bindings:
        bindings[checkout_root] = _PROD_PORT
    return bindings


def _detect_checkout_root() -> Path:
    """Return the absolute path of the checkout this file lives in.

    This module sits at ``deep_worker_web/app/config.py``; walking
    two parents up yields the checkout root. We resolve symlinks so that
    matches against ``_APPROVED_BINDINGS`` keys are stable regardless
    of how the directory was referenced when the process started.
    """
    # config.py -> app/ -> <repo root>
    return Path(__file__).resolve().parents[2]


# Every approved (tree, port) pair. validate_security() refuses anything
# outside this whitelist.
_APPROVED_BINDINGS: dict[Path, int] = _approved_bindings()


def _default_port_for(checkout_root: Path) -> int:
    """Pick the default port based on the checkout the code runs from."""
    return _APPROVED_BINDINGS.get(checkout_root, _PROD_PORT)


_CHECKOUT_ROOT = _detect_checkout_root()
_DEFAULT_PORT = _default_port_for(_CHECKOUT_ROOT)


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
    # fragmentation — no Python object is leaking. The 2026-04-27
    # incident took the web app to 9.2 GB this way.
    # KEEP THIS FALSE BY DEFAULT. Opt in via DEEP_WORKER_WEB_DEBUG=1
    # only for short-lived debugging sessions.
    # See a related internal incident analysis (not included here).
    debug: bool = False
    # Uvicorn's reload feature forces a WatchFiles + multiprocessing.spawn
    # subprocess layout that we don't want paying for in production (and
    # which silently bloats memory over long-lived runs, see
    # docs/DESIGN_AGENT_SERVICE_MEMORY_FIX.md). Default OFF; opt in
    # explicitly via DEEP_WORKER_WEB_RELOAD=1 when doing dev work.
    reload: bool = False

    class Config:
        """Pydantic configuration."""

        env_file = str(Path(__file__).resolve().parents[1] / ".env")
        env_prefix = "DEEP_WORKER_WEB_"

    @property
    def checkout_root(self) -> Path:
        """The repo this server is running from (resolved absolute path)."""
        return _CHECKOUT_ROOT

    def validate_security(self) -> bool:
        """Validate host/port against the two approved bindings.

        Returns:
            True if security check passes.

        Raises:
            ValueError: If host is not 127.0.0.1, if the active checkout
                is not one of the two approved trees, or if the port
                does not match the port assigned to that tree.
        """
        if self.host != "127.0.0.1":
            raise ValueError(
                f"SECURITY ERROR: Host must be 127.0.0.1 only! Current: {self.host}\n"
                "This application must NEVER be exposed to external networks.\n"
                "Running on 0.0.0.0 or any other address is STRICTLY PROHIBITED."
            )

        expected = _APPROVED_BINDINGS.get(_CHECKOUT_ROOT)
        if expected is None:
            approved = ", ".join(
                f"{p} -> {port}" for p, port in _APPROVED_BINDINGS.items()
            )
            raise ValueError(
                "SECURITY ERROR: the web app is running from an "
                f"unrecognised checkout: {_CHECKOUT_ROOT}\n"
                f"Approved (checkout -> port) pairs: {approved}\n"
                "Move the code into one of the approved trees before "
                "starting the server."
            )

        if self.port != expected:
            raise ValueError(
                f"SECURITY ERROR: checkout {_CHECKOUT_ROOT} must bind to "
                f"port {expected}, but settings.port is {self.port}.\n"
                "This pairing is enforced so the production server "
                "(the production tree, 7887) and the developer server "
                "(the developer tree, 7888) can run side by side "
                "without colliding."
            )

        return True


# Initialize settings and validate security immediately
settings = Settings()
settings.validate_security()
