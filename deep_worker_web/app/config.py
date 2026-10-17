"""Configuration for deep-worker-web application.

CRITICAL SECURITY REQUIREMENT:
==============================
This application MUST ONLY run on 127.0.0.1. That is the one hard
invariant, enforced by ``Settings.validate_security()``, and there is
no env var, flag or config key that relaxes it.

The PORT is a default, not a requirement. Each known checkout has one
so the prod and dev servers can run side by side unattended:

- 7887 when the active checkout is the production / hot-fix tree
  located at /path/to/production  (branch: master)
- 7888 when the active checkout is the developer tree located at
  /path/to/dev  (branch: dev)
- 7887 as the fallback anywhere else, including an installed wheel,
  which has no checkout at all.

An explicitly chosen port is honoured. ``validate_security()`` used to
veto one that disagreed with the table, which was both unshippable
(a pip install matches no tree) and backwards (it compared against the
table rather than reality, accepting an occupied port and rejecting a
free one). A port already in use is reported by ``bind()``.

Why two ports?
--------------
Two co-existing clones of the same project live on this host:

* /path/to/production — the canonical, "running" copy. The agents
  and the deep_worker_web instance the user actively relies on are
  served from here. Hot-fixes go to its `master` branch.
* /path/to/dev — the developer scratch copy on the `dev`
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

import os
import subprocess
from pathlib import Path

from pydantic_settings import BaseSettings

from . import paths


# ---------------------------------------------------------------------------
# Port selection based on checkout location
# ---------------------------------------------------------------------------

# Paths to the known checkouts and the port each one MUST use.
# Keep these in sync with deep_worker_web/run.py and the project README.
_PROD_TREE = Path("/path/to/production").resolve()
_DEV_TREE = Path("/path/to/dev").resolve()
_FIXES_TREE = Path("/path/to/fixes").resolve()
_TASKS_TREE = Path("/path/to/tasks").resolve()

_PROD_PORT = 7887
_DEV_PORT = 7888
_FIXES_PORT = 7889
_TASKS_PORT = 7890

# The port each known checkout DEFAULTS to, so the prod and dev servers
# can run side by side without being told which port to take. A tree that
# is not listed — including an installed wheel, which has no tree at all —
# falls back to _PROD_PORT. This table picks a default; it is not a
# whitelist, and validate_security() does not consult it.
_APPROVED_BINDINGS: dict[Path, int] = {
    _PROD_TREE: _PROD_PORT,
    _DEV_TREE: _DEV_PORT,
    _FIXES_TREE: _FIXES_PORT,
    _TASKS_TREE: _TASKS_PORT,
}

# Name of the active checkout (for logs / status broadcasts)
_CHECKOUT_NAME: dict[Path, str] = {
    _PROD_TREE: "production",
    _DEV_TREE: "dev",
    _FIXES_TREE: "fixes",
    _TASKS_TREE: "tasks",
}


def _detect_checkout_root() -> Path | None:
    """Return the absolute path of the checkout this file lives in.

    ``None`` when running from the installed wheel: there is no checkout,
    hence no ``(tree, port)`` pair to look up. Walking parents ourselves
    would land on ``<site-packages>/deep_worker_cli`` — a directory that
    exists, so the mistake would never surface. See ``app/paths.py``.

    Symlinks are resolved so that matches against ``_APPROVED_BINDINGS``
    keys are stable regardless of how the directory was referenced when
    the process started.
    """
    root = paths.source_tree_root()
    return None if root is None else root.resolve()


def _default_port_for(checkout_root: Path | None) -> int:
    """Pick the default port based on the checkout the code runs from.

    ``None`` means "installed, no checkout" and takes the same branch as
    any unrecognised tree: the fallback port. Behaviour for the known
    checkouts is unchanged (dev 7888, prod 7887).
    """
    if checkout_root is None:
        return _PROD_PORT
    return _APPROVED_BINDINGS.get(checkout_root, _PROD_PORT)


def get_git_commit_info(tree_root: str | None) -> tuple[str, str]:
    """Return (commit_sha, describe) for the git tree at *tree_root*.

    Both values default to ``"unknown"`` on any error so the caller
    never crashes.
    """
    if not tree_root:
        return "unknown", "unknown"
    try:
        sha = subprocess.run(
            ["git", "-C", tree_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not sha:
            sha = "unknown"
    except Exception:
        sha = "unknown"
    try:
        describe = subprocess.run(
            ["git", "-C", tree_root, "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not describe:
            describe = sha
    except Exception:
        describe = sha
    return sha, describe


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
    # incident took the web server to 9.2 GB this way.
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
    def checkout_root(self) -> Path | None:
        """The repo this server is running from (resolved absolute path).

        ``None`` when running from the installed wheel — there is no
        checkout to report.
        """
        return _CHECKOUT_ROOT

    def validate_security(self) -> bool:
        """Refuse to serve anything but loopback.

        This is the ONE hard invariant of this configuration, and it is
        deliberately the only check left here.

        What used to be here and why it is gone
        ---------------------------------------
        Two further branches policed WHERE the code lives: the active tree
        had to be one of four hard-coded checkouts, and the
        port had to equal the one assigned to that tree. Neither can ship.

        * The checkout whitelist refuses a pip install unconditionally —
          an installed distribution has no checkout to whitelist. That
          single branch is what kept the web UI source-only.
        * The port branch had its intent inverted. It compared the port
          against a TABLE, not against reality, so it accepted a port that
          was already in use and rejected a free one (measured: 7888
          occupied → True, free 39947 → ValueError). Port collisions are
          detected by ``bind()``, which reports them with the OS's own
          message; this method is not a port allocator and never was one.

        The port a checkout DEFAULTS to still comes from
        ``_APPROVED_BINDINGS`` via ``_default_port_for`` — picking a
        default and vetoing an explicit choice are different jobs.

        Returns:
            True if the binding is loopback.

        Raises:
            ValueError: If host is not exactly ``127.0.0.1``.
        """
        # Runs on the EFFECTIVE value, after env/.env/constructor have all
        # been applied. Should ``host`` ever gain another configuration
        # mechanism, this check must stay downstream of it: validating a
        # class default while the server binds elsewhere would be worse
        # than no check at all. There is deliberately no env var, flag or
        # config key that turns this off.
        if self.host != "127.0.0.1":
            raise ValueError(
                f"SECURITY ERROR: Host must be 127.0.0.1 only! Current: {self.host}\n"
                "This application must NEVER be exposed to external networks.\n"
                "Running on 0.0.0.0 or any other address is STRICTLY PROHIBITED."
            )

        return True


# Initialize settings (security is validated explicitly in run.py)
settings = Settings()
