"""Single source of truth for making the ``semantic_search`` package importable.

The ``semantic_search`` package is stored "wrapped":

    <tree>/semantic_search/                 <- wrapper dir, no __init__.py
    <tree>/semantic_search/semantic_search/ <- the real package

For ``import semantic_search.history_indexer`` to resolve, the WRAPPER dir
``<tree>/semantic_search`` must be on ``sys.path``. Several runtime modules
(``middleware/history_memory.py``, ``history_memory_tool.py``,
``deepagents/tools/project_memory.py``) each inject it with their own
``parent``-count arithmetic. Offline CLI code paths that never load those
modules (e.g. ``deep-worker --reindex-history <agent>`` for a stopped agent)
must inject the path themselves — hence this helper.

Keep this module stdlib-only (``pathlib`` / ``sys``) so it is cheap to import
from anywhere, including hot startup paths.
"""

from __future__ import annotations

import sys
from pathlib import Path


def semantic_search_wrapper_dir() -> Path:
    """Return the expected ``<tree>/semantic_search`` wrapper directory.

    This file lives at
    ``<tree>/deep_worker/deep_worker_cli/deep_worker_cli/semantic_search_path.py``.
    Four ``parent`` hops reach ``<tree>``:

        semantic_search_path.py
          .parent   -> deep_worker_cli (package)
          .parent   -> deep_worker_cli (project)
          .parent   -> deep_worker
          .parent   -> <tree>

    The nested real package then lives at ``<returned>/semantic_search``.
    """
    return Path(__file__).resolve().parent.parent.parent.parent / "semantic_search"


def ensure_semantic_search_importable() -> bool:
    """Idempotently put the ``semantic_search`` wrapper dir on ``sys.path``.

    Returns ``True`` when the wrapper directory exists (so
    ``import semantic_search`` will resolve), ``False`` otherwise. Safe to
    call repeatedly — a no-op once the path is already present.
    """
    wrapper = semantic_search_wrapper_dir()
    if not wrapper.exists():
        return False
    entry = str(wrapper)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return True
