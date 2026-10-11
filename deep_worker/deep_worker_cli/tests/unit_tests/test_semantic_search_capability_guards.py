"""B4: capability guards for the semantic_search provider (DESIGN §3.6).

The semantic-search extra is optional. Its absence must never surface a raw
``ModuleNotFoundError``:

* Automatic startup catch-up (``_count_pending_history_lines``,
  ``_history_index_exists``): log-once + ``(available=False)`` semantics —
  startup continues with history memory disabled, NO raise.
* Explicit requests (``enable_history_memory=True`` in
  ``create_cli_agent``, ``history_memory`` tool): hard-fail with an
  actionable ``CapabilityError`` / ``ImportError`` carrying the remedy
  ``pip install deep-worker[semantic-search]``.

Slice 6 note: the reindex paths no longer import the provider in-process —
they go through the shared semantic-memory service over gRPC. The former
``_history_indexer_funcs`` helpers were removed from both ``agent_command``
and ``agent_process``; ``test_no_in_process_history_indexer_path`` pins that
they stay gone, so the heavy import cannot come back as a foot-gun.

All provider resolution goes through
``deep_worker_cli.provider_paths.ensure_provider_importable`` — no sys.path
hacks (the old ``Path(__file__).parent / "semantic_search"`` hack was
removed).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from deep_worker_cli.exceptions import CapabilityError
from deep_worker_cli.provider_paths import ProviderOrigin, ProviderResolution

_REPO_ROOT = Path(__file__).resolve().parents[4]

_SEMANTIC_SEARCH_PKG = "deep_worker.providers.semantic_search"


def _unavailable_resolution(pkg: str = _SEMANTIC_SEARCH_PKG) -> ProviderResolution:
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="not found",
        unavailable="provider package not installed",
        remedy=None,
    )


# ---------------------------------------------------------------------------
# Reindex now goes through the service — the in-process heavy path is gone
# ---------------------------------------------------------------------------


def test_no_in_process_history_indexer_path() -> None:
    """Neither reindex entry point may resolve the provider in-process.

    Slice 6 moved indexing behind the shared service; ``_history_indexer_funcs``
    was the last reachable helper that imported ``history_indexer`` into the
    agent/CLI process. It must stay deleted, otherwise the heavy stack (torch)
    can be pulled back in by a single call site.
    """
    from deep_worker_cli.cli import agent_command
    from deep_worker_cli.process import agent_process

    assert not hasattr(agent_command, "_history_indexer_funcs")
    assert not hasattr(agent_command, "_run_reindex_with_progress")
    assert not hasattr(agent_process, "_history_indexer_funcs")
    assert not hasattr(agent_process, "_clear_then_index_history")
