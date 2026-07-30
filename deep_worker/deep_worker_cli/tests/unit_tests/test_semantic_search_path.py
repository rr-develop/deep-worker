from __future__ import annotations

import sys

from deep_worker_cli.semantic_search_path import (
    ensure_semantic_search_importable,
    semantic_search_wrapper_dir,
)


def test_wrapper_dir_exists_and_contains_nested_history_indexer() -> None:
    """Parent-count regression guard.

    The computed wrapper dir must exist and contain the nested real package
    with ``history_indexer.py``. If the ``parent`` arithmetic in the helper
    ever drifts, this fails loudly instead of surfacing as a runtime
    ModuleNotFoundError on the offline reindex path.
    """
    wrapper = semantic_search_wrapper_dir()
    assert wrapper.exists(), f"wrapper dir missing: {wrapper}"
    assert wrapper.name == "semantic_search"
    nested = wrapper / "semantic_search" / "history_indexer.py"
    assert nested.exists(), f"nested history_indexer.py missing: {nested}"


def test_ensure_importable_returns_true_and_adds_path() -> None:
    entry = str(semantic_search_wrapper_dir())
    saved = list(sys.path)
    try:
        while entry in sys.path:
            sys.path.remove(entry)
        assert ensure_semantic_search_importable() is True
        assert sys.path[0] == entry
    finally:
        sys.path[:] = saved


def test_ensure_importable_is_idempotent() -> None:
    entry = str(semantic_search_wrapper_dir())
    saved = list(sys.path)
    try:
        while entry in sys.path:
            sys.path.remove(entry)
        assert ensure_semantic_search_importable() is True
        assert ensure_semantic_search_importable() is True
        assert sys.path.count(entry) == 1
    finally:
        sys.path[:] = saved


def test_ensure_importable_actually_imports_history_indexer() -> None:
    """After ensure(), the nested submodule resolves (both funcs present)."""
    ensure_semantic_search_importable()
    from semantic_search.history_indexer import (  # noqa: F401
        clear_history_index,
        index_all_history_files,
    )

    assert callable(clear_history_index)
    assert callable(index_all_history_files)
