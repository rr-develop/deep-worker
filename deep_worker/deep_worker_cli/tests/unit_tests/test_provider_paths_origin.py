"""Installed-origin ALL-checks + env source-root trust contract.

Covers ``provider_paths.py``:

- ``DEEP_WORKER_SOURCE_ROOT`` is a TRUSTED declaration: when set but invalid
  (relative / missing marker / not a repo layout) resolution must report an
  actionable ``unavailable`` — NEVER silently fall back to the nearby
  checkout or the installed wheel;
- ``_installed_spec_location`` (pre-import) requires EVERY applicable
  spec origin / ``submodule_search_locations`` entry to lie under the active
  purelib/platlib roots — a single outside entry (source+installed mix) is
  rejected;
- ``_verify_installed`` (post-import) applies the same ALL-check to the real
  module ``__file__`` / ``__path__``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_worker_cli import provider_paths as pp

_PKG = "deep_worker.providers.semantic_search"


@pytest.fixture(autouse=True)
def _clean_resolution_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with an empty memo cache and no env overrides."""
    pp._resolved.clear()
    monkeypatch.delenv(pp._ENV_SOURCE_ROOT, raising=False)
    monkeypatch.delenv(pp._ENV_PROVIDER_SOURCE, raising=False)
    for name in list(sys.modules):
        if name.startswith("deep_worker.providers."):
            del sys.modules[name]


def _fake_spec(origin: str | None, locations: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        origin=origin,
        submodule_search_locations=locations,
    )


# ---------------------------------------------------------------------------
# DEEP_WORKER_SOURCE_ROOT trust: invalid → error, never fallback
# ---------------------------------------------------------------------------


def test_source_root_from_env_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(pp._ENV_SOURCE_ROOT, raising=False)
    assert pp._source_root_from_env() is None


def test_source_root_from_env_relative_raises(monkeypatch) -> None:
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, "relative/path")
    with pytest.raises(pp.SourceRootError, match="not an absolute path"):
        pp._source_root_from_env()


def test_source_root_from_env_missing_marker_raises(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "not_a_repo"
    root.mkdir()
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(root))
    with pytest.raises(pp.SourceRootError, match="not a trusted repository root"):
        pp._source_root_from_env()


def test_source_root_from_env_nonexistent_dir_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(tmp_path / "does_not_exist"))
    with pytest.raises(pp.SourceRootError):
        pp._source_root_from_env()


def test_find_source_root_propagates_invalid_env_not_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    """Explicit-but-invalid env must NOT silently fall back to nearby root."""
    bad = tmp_path / "bad"
    bad.mkdir()
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(bad))
    with pytest.raises(pp.SourceRootError):
        pp._find_source_root()


def test_resolve_source_invalid_env_is_actionable_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(bad))
    res = pp.resolve_provider(_PKG, mode="source")
    assert res.unavailable is not None
    assert pp._ENV_SOURCE_ROOT in res.unavailable
    assert res.remedy is not None
    assert res.origin is pp.ProviderOrigin.SOURCE


def test_resolve_auto_invalid_env_does_not_fall_back_to_installed(
    monkeypatch, tmp_path: Path
) -> None:
    """auto + invalid explicit root → unavailable, NOT installed origin."""
    bad = tmp_path / "bad"
    bad.mkdir()
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(bad))
    res = pp.resolve_provider(_PKG, mode="auto")
    assert res.unavailable is not None
    assert res.origin is pp.ProviderOrigin.SOURCE  # intent was source
    assert pp._ENV_SOURCE_ROOT in res.reason


def test_resolve_auto_valid_env_uses_source_root(
    monkeypatch, tmp_path: Path
) -> None:
    """Valid env root → SOURCE origin with that root on sys.path."""
    root = tmp_path / "repo"
    (root / "deep_worker" / "deep_worker_cli").mkdir(parents=True)
    (root / "deep_worker" / "deep_worker_cli" / "pyproject.toml").write_text("")
    for sub in (
        "claude_code_provider",
        "codex_provider",
        "grok_provider",
        "semantic_search",
        "deepworker_a2a",
    ):
        (root / "deep_worker" / "providers" / sub).mkdir(parents=True)
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(root))
    res = pp.resolve_provider(_PKG, mode="auto")
    assert res.unavailable is None
    assert res.origin is pp.ProviderOrigin.SOURCE
    assert str(root) in res.roots


def test_ensure_source_root_on_path_invalid_env_raises(
    monkeypatch, tmp_path: Path
) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    monkeypatch.setenv(pp._ENV_SOURCE_ROOT, str(bad))
    with pytest.raises(pp.SourceRootError):
        pp.ensure_source_root_on_path()


# ---------------------------------------------------------------------------
# _installed_spec_location — pre-import ALL-check
# ---------------------------------------------------------------------------


def _set_site_roots(monkeypatch, tmp_path: Path) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    site_root = site.resolve()
    monkeypatch.setattr(pp, "_site_roots", lambda: [site_root])
    return site_root


def test_installed_spec_location_all_under_site_roots(
    monkeypatch, tmp_path: Path
) -> None:
    site_root = _set_site_roots(monkeypatch, tmp_path)
    origin = str(site_root / "deep_worker" / "providers" / "semantic_search" / "__init__.py")
    loc = str(site_root / "deep_worker" / "providers" / "semantic_search")
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec(origin, [loc])
    )
    got = pp._installed_spec_location(_PKG)
    assert got is not None
    assert got == Path(origin).resolve()


def test_installed_spec_location_mixed_origin_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    """One entry outside the site roots → whole origin rejected (no mix)."""
    site_root = _set_site_roots(monkeypatch, tmp_path)
    outside = (tmp_path / "outside").resolve()
    origin = str(site_root / "deep_worker" / "providers" / "semantic_search" / "__init__.py")
    mixed = [str(site_root / "deep_worker" / "providers" / "semantic_search"), str(outside)]
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec(origin, mixed)
    )
    assert pp._installed_spec_location(_PKG) is None


def test_installed_spec_location_all_outside_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    _set_site_roots(monkeypatch, tmp_path)
    outside = str((tmp_path / "outside").resolve())
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec(outside, [outside])
    )
    assert pp._installed_spec_location(_PKG) is None


def test_installed_spec_location_empty_candidates_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    _set_site_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: _fake_spec(None, [])
    )
    assert pp._installed_spec_location(_PKG) is None


def test_installed_spec_location_missing_spec_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    _set_site_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert pp._installed_spec_location(_PKG) is None


# ---------------------------------------------------------------------------
# _verify_installed — post-import ALL-check
# ---------------------------------------------------------------------------


def _put_module(monkeypatch, name: str, file_path: str | None, path_attr) -> None:
    monkeypatch.setitem(
        sys.modules,
        name,
        SimpleNamespace(__file__=file_path, __path__=path_attr),
    )


def test_verify_installed_file_and_path_all_under_site(
    monkeypatch, tmp_path: Path
) -> None:
    site_root = _set_site_roots(monkeypatch, tmp_path)
    f = str(site_root / "deep_worker" / "providers" / "semantic_search" / "__init__.py")
    p = [str(site_root / "deep_worker" / "providers" / "semantic_search")]
    _put_module(monkeypatch, _PKG, f, p)
    assert pp._verify_installed(_PKG) is True


def test_verify_installed_path_has_outside_entry_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    site_root = _set_site_roots(monkeypatch, tmp_path)
    f = str(site_root / "deep_worker" / "providers" / "semantic_search" / "__init__.py")
    p = [
        str(site_root / "deep_worker" / "providers" / "semantic_search"),
        str((tmp_path / "outside").resolve()),
    ]
    _put_module(monkeypatch, _PKG, f, p)
    assert pp._verify_installed(_PKG) is False


def test_verify_installed_file_outside_rejected(monkeypatch, tmp_path: Path) -> None:
    _set_site_roots(monkeypatch, tmp_path)
    f = str((tmp_path / "outside").resolve() / "__init__.py")
    _put_module(monkeypatch, _PKG, f, None)
    assert pp._verify_installed(_PKG) is False


def test_verify_installed_no_module_false(monkeypatch, tmp_path: Path) -> None:
    _set_site_roots(monkeypatch, tmp_path)
    assert pp._verify_installed(_PKG) is False


def test_post_import_checked_unavailable_on_mixed_origin(
    monkeypatch, tmp_path: Path
) -> None:
    """resolve_provider on an already-imported module whose real ``__path__``
    mixes site-packages with an outside entry reports unavailable — the
    post-import ALL-check must reject it even when the pre-import spec looked
    fully installed."""
    site_root = _set_site_roots(monkeypatch, tmp_path)
    f = str(site_root / "deep_worker" / "providers" / "semantic_search" / "__init__.py")
    p = [
        str(site_root / "deep_worker" / "providers" / "semantic_search"),
        str((tmp_path / "outside").resolve()),
    ]
    _put_module(monkeypatch, _PKG, f, p)
    # Pre-import spec is clean (all under site roots) → resolution is INSTALLED.
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: _fake_spec(
            f, [str(site_root / "deep_worker" / "providers" / "semantic_search")]
        ),
    )
    res = pp.resolve_provider(_PKG, mode="installed")
    assert res.unavailable is not None
    assert "post-import" in res.reason
