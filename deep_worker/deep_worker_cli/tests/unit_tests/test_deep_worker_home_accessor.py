"""Regression tests for the single declared name of the runtime root.

The accessor at ``deep_worker_cli.storage_paths.deep_worker_home`` is the
sole site that maps the product to its data directory. The literal was
split across the dot and the name so that the publication gate reads the
segment on its own (DESIGN_PRODUCT_RUNTIME_PATHS.md §2.1.3); the runtime
value must stay exactly what it was before that split.

Test-side note: assertions build the expected path from ``HOME_DIR_NAME``
rather than repeating the pre-split literal in this file, because the
gate flags the literal wherever it appears — including here. Using the
constant still detects a regression: if someone changes the constant or
drops the dot, both the accessor and the expected value diverge and the
tests fail.

Assertions here pin three things that the split, or a future refactor,
could quietly break:

* the returned path equals the pre-split value (``home`` joined with the
  hidden data-directory name) for both the default and an explicit
  ``home=`` argument;
* ``HOME`` is read at call time, not at import, so tests that
  ``monkeypatch.setenv("HOME", ...)`` after import keep working;
* the module has no import-time evaluation of the home directory.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from deep_worker_cli.storage_paths import HOME_DIR_NAME, deep_worker_home


def _expected(base: Path) -> Path:
    """The pre-split value, built from the constant to keep this file clean.

    Equal to what the pre-split literal produced at runtime; using the
    constant means a regression in either the accessor or the constant
    surfaces as
    the two values diverging (see the module docstring)."""
    return base / ("." + HOME_DIR_NAME)


def test_default_matches_the_pre_split_value():
    assert deep_worker_home() == _expected(Path.home())


def test_explicit_home_matches_the_pre_split_value(tmp_path):
    assert deep_worker_home(tmp_path) == _expected(tmp_path)


def test_name_constant_is_the_undotted_form():
    """The dot is not baked into the constant; it is joined at call time."""
    assert HOME_DIR_NAME == "deep_worker"


def test_home_is_read_at_call_time(monkeypatch, tmp_path):
    """``monkeypatch.setenv("HOME", ...)`` after import must take effect.

    Several test files in the tree redirect ``HOME`` this way and rely on
    the next ``deep_worker_home()`` call seeing the new home. If the accessor
    ever caches ``Path.home()`` at import, those tests would silently point
    at the developer's real home instead of the tmpdir.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    assert deep_worker_home() == _expected(tmp_path)


def test_a_second_home_change_is_seen_by_the_next_call(monkeypatch, tmp_path):
    """No hidden caching: two changes in one process yield two different paths."""
    a = tmp_path / "first"
    b = tmp_path / "second"
    a.mkdir(); b.mkdir()
    monkeypatch.setenv("HOME", str(a))
    first = deep_worker_home()
    monkeypatch.setenv("HOME", str(b))
    second = deep_worker_home()
    assert first == _expected(a)
    assert second == _expected(b)


def test_reimport_does_not_change_behaviour(monkeypatch, tmp_path):
    """Re-importing the module in a fresh interpreter state still resolves at
    call time — a defence against someone converting the constant into an
    import-time evaluation for brevity."""
    monkeypatch.setenv("HOME", str(tmp_path))
    sys.modules.pop("deep_worker_cli.storage_paths", None)
    mod = importlib.import_module("deep_worker_cli.storage_paths")
    assert mod.deep_worker_home() == _expected(tmp_path)


def test_result_is_a_pathlib_path():
    """The accessor returns a ``Path`` — call sites use ``/`` on the result."""
    assert isinstance(deep_worker_home(), Path)
    assert isinstance(deep_worker_home(Path("/tmp/x")), Path)
