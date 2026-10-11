"""Make a local ``rpyc_async`` checkout win over an installed distribution.

Local sources, when present, take priority; without them the library
installed via ``pip install`` is used. A checkout is opted into by pointing
the ``RPYC_ASYNC_LOCAL_CHECKOUT`` environment variable at it.

So the resolution order is, unconditionally:

1. the checkout named by ``RPYC_ASYNC_LOCAL_CHECKOUT``, if it holds one,
   regardless of what is installed and regardless of version numbers, then
2. whatever ``pip install`` put in ``site-packages``.

Import this module BEFORE the first ``import rpyc_async`` in any entry
point that needs the AsyncioServer protocol.

Why the sources have to win
---------------------------
A checkout can be ahead of the latest published release: for example
``rpyc_async.core.netref.async_call`` is a hard requirement of the web UI's
MCP service, and a host resolving an older PyPI release cannot send
inter-agent messages from the web UI. A dependency pin cannot express
"newer than anything published", so the checkout is the source of truth
whenever it is configured and present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable naming a local checkout of the fork. Unset or empty
#: means "use the installed distribution".
LOCAL_CHECKOUT_ENV = "RPYC_ASYNC_LOCAL_CHECKOUT"


def _configured_path() -> Path | None:
    """The checkout path from ``RPYC_ASYNC_LOCAL_CHECKOUT``, or None."""
    value = os.environ.get(LOCAL_CHECKOUT_ENV, "").strip()
    return Path(value) if value else None


#: Where a developer checkout of the fork lives, if one is configured.
LOCAL_RPYC_ASYNC_PATH: Path | None = _configured_path()


def _is_checkout(path: Path | None) -> bool:
    """Whether *path* actually contains an importable ``rpyc_async`` package.

    Checked rather than assumed so that a stray empty directory cannot shadow
    a working installed distribution — that would turn a cosmetic filesystem
    leftover into an ``ImportError`` at startup.
    """
    if path is None:
        return False
    return (path / "rpyc_async" / "__init__.py").is_file()


def prefer_local_rpyc_async() -> Path | None:
    """Put a local checkout at the FRONT of ``sys.path``; return it, or None.

    Front, not back, because an installed distribution would otherwise win —
    which is exactly the inversion this function exists to prevent.

    Idempotent, and safe to call after ``rpyc_async`` is already imported
    (it just will not retroactively change that import), so every entry point
    can call it unconditionally at module scope.
    """
    path = LOCAL_RPYC_ASYNC_PATH
    if path is None or not _is_checkout(path):
        return None

    entry = str(path)
    if sys.path and sys.path[0] == entry:
        return path
    # Drop any later copy before inserting, so repeated calls cannot grow
    # sys.path without bound.
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    return path


#: Resolved on first import so the side effect happens exactly once per
#: process, before any module-level ``import rpyc_async`` that follows.
ACTIVE_LOCAL_CHECKOUT: Path | None = prefer_local_rpyc_async()
