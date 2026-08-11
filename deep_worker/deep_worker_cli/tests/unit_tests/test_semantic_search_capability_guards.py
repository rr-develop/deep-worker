"""B4: capability guards for the semantic_search provider (DESIGN §3.6).

The semantic-search extra is optional. Its absence must never surface a raw
``ModuleNotFoundError``:

* Automatic startup catch-up (``_count_pending_history_lines``,
  ``_history_index_exists``): log-once + ``(available=False)`` semantics —
  startup continues with history memory disabled, NO raise.
* Explicit requests (``_history_indexer_funcs`` for reindex /
  ``--reindex-history``, ``enable_history_memory=True`` in
  ``create_cli_agent``, ``history_memory`` tool): hard-fail with an
  actionable ``CapabilityError`` / ``ImportError`` carrying the remedy
  ``pip install deep-worker[semantic-search]``.

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
# _history_indexer_funcs → CapabilityError (explicit reindex path)
# ---------------------------------------------------------------------------


def test_agent_process_history_indexer_funcs_raises_capability_error_when_unavailable() -> None:
    """Reindex path in ``AgentProcess`` must raise CapabilityError with
    capability='semantic-search' when the provider is unavailable — and must
    NOT reach the history_indexer import."""
    from deep_worker_cli.process import agent_process as mod

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == _SEMANTIC_SEARCH_PKG or name.startswith(_SEMANTIC_SEARCH_PKG + "."):
            raise AssertionError(f"history_indexer import reached: {name}")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_unavailable_resolution(),
        ),
    ):
        with pytest.raises(CapabilityError) as exc_info:
            mod._history_indexer_funcs()

    assert exc_info.value.capability == "semantic-search"
    assert exc_info.value.remedy == "pip install deep-worker[semantic-search]"


def test_agent_command_history_indexer_funcs_raises_capability_error_when_unavailable() -> None:
    """Offline CLI reindex path must raise CapabilityError (not
    ModuleNotFoundError) when the provider is unavailable."""
    from deep_worker_cli.cli import agent_command as mod

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == _SEMANTIC_SEARCH_PKG or name.startswith(_SEMANTIC_SEARCH_PKG + "."):
            raise AssertionError(f"history_indexer import reached: {name}")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_unavailable_resolution(),
        ),
    ):
        with pytest.raises(CapabilityError) as exc_info:
            mod._history_indexer_funcs()

    assert exc_info.value.capability == "semantic-search"
    assert exc_info.value.remedy == "pip install deep-worker[semantic-search]"


# ---------------------------------------------------------------------------
# _count_pending_history_lines — auto catch-up, log-once, no raise
# ---------------------------------------------------------------------------


def _make_jsonl(tmp_path: Path) -> Path:
    j = tmp_path / "conv.jsonl"
    j.write_text("line1\nline2\n", encoding="utf-8")
    return j


def test_count_pending_history_lines_returns_zero_when_provider_unavailable(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Auto catch-up: unavailable provider → return 0 (nothing pending), NO
    raise; a single warning is logged."""
    import logging

    from deep_worker_cli import agent as agent_mod

    monkeypatch.setattr(agent_mod, "_semantic_search_warned_once", False)
    jsonl = _make_jsonl(tmp_path)

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=_unavailable_resolution(),
    ):
        with caplog.at_level(logging.WARNING, logger="deep_worker_cli.agent"):
            pending = agent_mod._count_pending_history_lines(tmp_path, [jsonl])

    assert pending == 0
    assert any(
        "semantic-search unavailable" in r.message for r in caplog.records
    )


def test_count_pending_history_lines_warns_only_once(tmp_path: Path, monkeypatch, caplog) -> None:
    """The auto-path warning must fire exactly once across repeated calls."""
    import logging

    from deep_worker_cli import agent as agent_mod

    monkeypatch.setattr(agent_mod, "_semantic_search_warned_once", False)
    jsonl = _make_jsonl(tmp_path)

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=_unavailable_resolution(),
    ):
        with caplog.at_level(logging.WARNING, logger="deep_worker_cli.agent"):
            agent_mod._count_pending_history_lines(tmp_path, [jsonl])
            agent_mod._count_pending_history_lines(tmp_path, [jsonl])
            agent_mod._count_pending_history_lines(tmp_path, [jsonl])

    semantic_warnings = [
        r for r in caplog.records if "semantic-search unavailable" in r.message
    ]
    assert len(semantic_warnings) == 1


# ---------------------------------------------------------------------------
# create_cli_agent — enable_history_memory guard (textual)
# ---------------------------------------------------------------------------

_AGENT_FILE = (
    _REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli" / "agent.py"
)


def test_agent_enable_history_memory_guard_resolves_before_import() -> None:
    """``if enable_history_memory:`` block MUST resolve the provider via
    ``ensure_provider_importable`` and raise ``CapabilityError`` on
    ``resolution.unavailable`` BEFORE importing ``index_all_history_files``."""
    text = _AGENT_FILE.read_text(encoding="utf-8")
    block_start = text.index("if enable_history_memory:")
    block = text[block_start:]
    resolve_idx = block.index('ensure_provider_importable(\n            "deep_worker.providers.semantic_search"')
    import_idx = block.index(
        "from deep_worker.providers.semantic_search.history_indexer import index_all_history_files"
    )
    assert resolve_idx < import_idx, (
        "enable_history_memory block must resolve the provider BEFORE importing "
        "index_all_history_files"
    )
    between = block[resolve_idx:import_idx]
    assert "resolution.unavailable" in between
    assert "raise CapabilityError" in between
    assert "capability=\"semantic-search\"" in between
    assert "deep-worker[semantic-search]" in between


# ---------------------------------------------------------------------------
# history_memory middleware / tool — auto path and remedy
# ---------------------------------------------------------------------------


def test_history_index_exists_returns_false_when_provider_unavailable() -> None:
    """Middleware probe: unavailable provider → False (no raise), so the
    middleware simply stays disabled."""
    from deep_worker_cli.middleware.history_memory import _history_index_exists

    with patch(
        "deep_worker_cli.provider_paths.ensure_provider_importable",
        return_value=_unavailable_resolution(),
    ):
        assert _history_index_exists(Path("/nonexistent")) is False


def test_history_memory_tool_reports_install_remedy_when_unavailable() -> None:
    """The ``history_memory`` tool body must turn the missing extra into a
    user-facing error naming the extra install command."""
    from deep_worker_cli import history_memory_tool as mod

    with (
        patch.object(mod, "_history_search", None),
        patch(
            "deep_worker_cli.provider_paths.ensure_provider_importable",
            return_value=_unavailable_resolution(),
        ),
    ):
        result = mod._run_history_search("query", "/nonexistent", 5)

    assert "deep-worker[semantic-search]" in result
    assert result.startswith("Error: semantic search dependencies not available")


def test_history_memory_tool_has_no_sys_path_hack() -> None:
    """B4 removed the ``sys.path.insert(Path(__file__).parent/.../semantic_search)``
    hack — the tool must resolve through provider_paths only."""
    tool_file = (
        _REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli" / "history_memory_tool.py"
    )
    text = tool_file.read_text(encoding="utf-8")
    assert "sys.path.insert" not in text
    assert "sys.path.append" not in text
    # The remedy must reference the extra (not a bare "semantic-search" path).
    assert "deep-worker[semantic-search]" in text


def test_middleware_history_memory_has_no_module_level_resolve() -> None:
    """Importing the middleware package must NOT eagerly resolve the provider
    (module-level resolve was removed — resolution is lazy inside
    ``_history_index_exists``)."""
    mw_file = (
        _REPO_ROOT / "deep_worker" / "deep_worker_cli" / "deep_worker_cli" / "middleware" / "history_memory.py"
    )
    text = mw_file.read_text(encoding="utf-8")
    module_header = text[: text.index("def _history_index_exists")]
    assert "ensure_provider_importable" not in module_header
