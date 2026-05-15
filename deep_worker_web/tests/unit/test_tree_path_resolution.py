"""Regression tests for tree-local path resolution in deep_worker_web.

Background
==========
deep_worker_web can be checked out at TWO locations on the same host:

  * deep_worker_web — prod checkout  (port 7887)
  * deep_worker_web — dev checkout   (port 7888)

For --dev to work end-to-end, the dev web server MUST spawn dev-tree
agents (so they share the same deep_worker_cli / deepagents codebase).
Previously the code hard-coded an absolute prod-checkout path in the
spawn helper and the registry-loader, which meant the dev web server
silently spawned prod-tree agents — defeating the whole point of
operational isolation.

These tests are textual: they assert the files do not contain the
literal prod path. The fix is to derive paths from ``__file__``.

Design doc: docs/DESIGN_DEV_FLAG.md
"""

from __future__ import annotations

from pathlib import Path


# ``deep_worker_web`` root = parent x3 of this file.
_THIS_FILE = Path(__file__).resolve()
_WEB_ROOT = _THIS_FILE.parent.parent.parent  # → deep_worker_web/

AGENTS_ROUTER = _WEB_ROOT / "app" / "routers" / "agents.py"
AGENT_SERVICE = _WEB_ROOT / "app" / "services" / "agent_service.py"


_DW = "deep_worker"
_BAD_PROD = "/path/to/prod-tree/deep_worker/deep_worker_cli"
_BAD_PROD_DEEPAGENTS = f"/path/to/prod-tree/{_DW}/deepagents"
_BAD_DEV = "/path/to/dev-tree/deep_worker/deep_worker_cli"
_BAD_DEV_DEEPAGENTS = f"/path/to/dev-tree/{_DW}/deepagents"


def test_agents_router_exists() -> None:
    assert AGENTS_ROUTER.exists(), f"not found: {AGENTS_ROUTER}"


def test_agent_service_exists() -> None:
    assert AGENT_SERVICE.exists(), f"not found: {AGENT_SERVICE}"


def test_agents_router_has_no_hardcoded_tree_paths() -> None:
    """``app/routers/agents.py`` builds PYTHONPATH for spawned agents.
    It must NOT contain a hard-coded prod or dev path — it must derive
    the path from its own location so that the dev web server spawns
    dev-tree agents and the prod web server spawns prod-tree agents.
    """
    text = AGENTS_ROUTER.read_text()
    for bad in (_BAD_PROD, _BAD_PROD_DEEPAGENTS, _BAD_DEV, _BAD_DEV_DEEPAGENTS):
        assert bad not in text, (
            f"{AGENTS_ROUTER.name} still hard-codes {bad!r}. "
            f"This breaks operational isolation: the dev web server "
            f"would silently spawn agents on the prod codebase (or "
            f"vice versa). Derive the path from __file__ instead. "
            f"See docs/DESIGN_DEV_FLAG.md."
        )


def test_agent_service_has_no_hardcoded_tree_paths() -> None:
    """``app/services/agent_service.py`` adds deep_worker_cli to sys.path
    so it can import ProcessRegistry. Same rule: derive from __file__,
    don't hard-code a specific tree."""
    text = AGENT_SERVICE.read_text()
    for bad in (_BAD_PROD, _BAD_PROD_DEEPAGENTS, _BAD_DEV, _BAD_DEV_DEEPAGENTS):
        assert bad not in text, (
            f"{AGENT_SERVICE.name} still hard-codes {bad!r}. "
            f"Derive from __file__ so the import follows the tree the "
            f"web server lives in. See docs/DESIGN_DEV_FLAG.md."
        )


def test_agents_router_derives_paths_from_file() -> None:
    """Sanity check: the file should reference ``__file__`` (which is
    the fix). Loose, but catches a wholesale revert."""
    text = AGENTS_ROUTER.read_text()
    assert "__file__" in text, (
        f"{AGENTS_ROUTER.name} no longer mentions __file__ — was the "
        f"tree-local path fix reverted?"
    )


def test_agent_service_derives_paths_from_file() -> None:
    text = AGENT_SERVICE.read_text()
    assert "__file__" in text, (
        f"{AGENT_SERVICE.name} no longer mentions __file__ — was the "
        f"tree-local path fix reverted?"
    )
