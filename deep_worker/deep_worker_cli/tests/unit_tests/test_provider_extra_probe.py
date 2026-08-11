"""Extra-stack capability probe for vendored providers (DESIGN §3.6/§4).

In the wheel all 5 vendor packages are always present, so the real gate for
an optional capability is its heavy extra stack (llama-index for
semantic-search). ``provider_paths.ensure_provider_importable`` must report
``unavailable`` + remedy when the package resolves but the extra is missing —
otherwise explicit call sites (``_history_indexer_funcs``,
``enable_history_memory``, reindex) raise a raw ``ModuleNotFoundError``
instead of ``CapabilityError``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deep_worker_cli.provider_paths import (
    ProviderOrigin,
    ProviderResolution,
    _apply_extra_probe,
    ensure_provider_importable,
)

_SEMANTIC_SEARCH_PKG = "deep_worker.providers.semantic_search"


def _installed_resolution(pkg: str = _SEMANTIC_SEARCH_PKG) -> ProviderResolution:
    return ProviderResolution(
        pkg=pkg,
        mode="auto",
        origin=ProviderOrigin.INSTALLED,
        roots=(),
        reason="installed",
        unavailable=None,
        remedy=None,
    )


# ---------------------------------------------------------------------------
# _apply_extra_probe
# ---------------------------------------------------------------------------


def test_apply_extra_probe_reports_unavailable_when_extra_missing() -> None:
    """Package resolves, but ``llama_index`` is not importable → unavailable
    with the ``deep-worker[semantic-search]`` remedy."""
    with patch("importlib.util.find_spec", return_value=None):
        result = _apply_extra_probe(_installed_resolution())

    assert result.unavailable is not None
    assert "llama_index" in result.unavailable
    assert result.remedy == "pip install deep-worker[semantic-search]"


def test_apply_extra_probe_keeps_resolution_when_extra_present() -> None:
    """Package resolves and ``llama_index`` is importable → unchanged."""
    with patch("importlib.util.find_spec", return_value=object()):
        result = _apply_extra_probe(_installed_resolution())

    assert result.unavailable is None
    assert result.remedy is None


def test_apply_extra_probe_classifies_exact_probe_module_missing() -> None:
    """find_spec raising ModuleNotFoundError naming EXACTLY the probe module
    → missing-extra case (unavailable + remedy), never a raw raise."""
    import importlib.util

    def _boom(_name: str):
        raise ModuleNotFoundError("No module named 'llama_index'", name="llama_index")

    with patch("importlib.util.find_spec", side_effect=_boom):
        result = _apply_extra_probe(_installed_resolution())

    assert result.unavailable is not None
    assert result.remedy == "pip install deep-worker[semantic-search]"


def test_apply_extra_probe_reraise_unrelated_module_missing() -> None:
    """find_spec raising ModuleNotFoundError for an UNRELATED module (broken
    parent / import machinery defect) must re-raise, not mask as extra-missing."""

    def _boom(_name: str):
        raise ModuleNotFoundError("No module named 'some_broken_dep'", name="some_broken_dep")

    with patch("importlib.util.find_spec", side_effect=_boom):
        with pytest.raises(ModuleNotFoundError, match="some_broken_dep"):
            _apply_extra_probe(_installed_resolution())


def test_apply_extra_probe_reraise_unclassified_import_error() -> None:
    """A bare ModuleNotFoundError without a name (or any non-MNF ImportError)
    is an internal defect → re-raise, never 'extra not installed'."""

    def _boom(_name: str):
        raise ModuleNotFoundError("parent missing")

    with patch("importlib.util.find_spec", side_effect=_boom):
        with pytest.raises(ModuleNotFoundError, match="parent missing"):
            _apply_extra_probe(_installed_resolution())


def test_apply_extra_probe_reraise_value_error() -> None:
    """Broken import machinery (ValueError) must re-raise, never mask."""

    def _boom(_name: str):
        raise ValueError("broken meta path finder")

    with patch("importlib.util.find_spec", side_effect=_boom):
        with pytest.raises(ValueError, match="broken meta path finder"):
            _apply_extra_probe(_installed_resolution())


def test_apply_extra_probe_does_not_overwrite_unavailable_resolution() -> None:
    """An already-unavailable resolution (provider itself missing) is returned
    unchanged — the extra probe never overwrites a more fundamental failure."""
    already = _installed_resolution()
    already = ProviderResolution(
        pkg=already.pkg,
        mode=already.mode,
        origin=already.origin,
        roots=already.roots,
        reason="provider not found under active site roots",
        unavailable="provider not installed",
        remedy="install deep-worker (wheel)",
    )
    with patch("importlib.util.find_spec", return_value=None):
        result = _apply_extra_probe(already)

    assert result is already
    assert result.unavailable == "provider not installed"
    assert result.remedy == "install deep-worker (wheel)"


def test_apply_extra_probe_noop_for_providers_without_probe() -> None:
    """Providers without an extra probe (e.g. claude_code_provider) are
    untouched."""
    pkg = "deep_worker.providers.claude_code_provider"
    resolution = _installed_resolution(pkg)

    with patch("importlib.util.find_spec", return_value=None):
        result = _apply_extra_probe(resolution)

    assert result is resolution
    assert result.unavailable is None


# ---------------------------------------------------------------------------
# ensure_provider_importable end-to-end (real resolve + probe)
# ---------------------------------------------------------------------------


def test_ensure_provider_importable_unavailable_when_extra_missing() -> None:
    """End-to-end: with llama_index absent, semantic_search reports
    unavailable with the remedy (regardless of the package origin)."""
    import importlib.util as _util

    real_find_spec = _util.find_spec

    def _fake_find_spec(name):
        if name == "llama_index":
            return None
        return real_find_spec(name)

    with patch("importlib.util.find_spec", side_effect=_fake_find_spec):
        result = ensure_provider_importable(_SEMANTIC_SEARCH_PKG)

    assert result.unavailable is not None
    assert result.remedy == "pip install deep-worker[semantic-search]"
