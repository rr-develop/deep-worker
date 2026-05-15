"""Config: ``debug`` must be orthogonal to ``reload``.

``reload=True`` forces uvicorn into a WatchFiles + multiprocessing.spawn
layout that pays overhead we only want in dev. Production / CI must
start with ``reload=False`` by default, regardless of ``debug``.
"""
from __future__ import annotations

import os
from importlib import reload as importlib_reload
from typing import Iterator

import pytest


@pytest.fixture
def clean_reload_env() -> Iterator[None]:
    """Make sure DEEP_WORKER_WEB_RELOAD doesn't leak between tests."""
    prev = os.environ.pop("DEEP_WORKER_WEB_RELOAD", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["DEEP_WORKER_WEB_RELOAD"] = prev
        else:
            os.environ.pop("DEEP_WORKER_WEB_RELOAD", None)


def _fresh_settings() -> object:
    """Re-import app.config so env-var changes take effect."""
    from app import config as cfg_mod

    importlib_reload(cfg_mod)
    return cfg_mod.Settings()


def test_reload_defaults_to_false(clean_reload_env: None) -> None:
    s = _fresh_settings()
    assert s.reload is False  # type: ignore[attr-defined]


def test_debug_true_does_not_imply_reload(clean_reload_env: None) -> None:
    """``debug`` and ``reload`` are orthogonal.

    ``debug`` defaults to False (see test_config.py for rationale —
    DEBUG logging fragments the glibc heap on long-lived processes).
    But even when an operator explicitly opts in via
    ``DEEP_WORKER_WEB_DEBUG=1``, ``reload`` must stay False — it has
    its own opt-in env var.
    """
    os.environ["DEEP_WORKER_WEB_DEBUG"] = "1"
    try:
        s = _fresh_settings()
        assert s.debug is True  # type: ignore[attr-defined]
        assert s.reload is False  # type: ignore[attr-defined]
    finally:
        os.environ.pop("DEEP_WORKER_WEB_DEBUG", None)


def test_env_var_opt_in_enables_reload(clean_reload_env: None) -> None:
    os.environ["DEEP_WORKER_WEB_RELOAD"] = "1"
    s = _fresh_settings()
    assert s.reload is True  # type: ignore[attr-defined]
