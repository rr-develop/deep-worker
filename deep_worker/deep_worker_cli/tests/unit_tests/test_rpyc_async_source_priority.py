"""A local ``rpyc_async`` checkout must outrank an installed distribution.

A checkout named by ``RPYC_ASYNC_LOCAL_CHECKOUT`` wins; without one we fall
back to ``pip install rpyc-async``. A published release can lag the checkout
(e.g. it may lack ``rpyc_async.core.netref.async_call``, which the web UI
needs to send inter-agent messages), so the resolution order matters.

Why behaviour and not just text: a guard that only greps for marker strings
can pass while the actual resolution order is wrong. Each test below
exercises ``prefer_local_rpyc_async`` against a real directory layout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from deep_worker_cli import rpyc_async_source

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "deep_worker_cli"

#: Every module that imports ``rpyc_async`` at module scope. Each one is an
#: entry point in its own right (agent process, CLI client, TUI, RPyC
#: service), so each must establish the priority itself — importing one does
#: not help the others.
RPYC_IMPORTING_MODULES = (
    PACKAGE_ROOT / "process" / "agent_process.py",
    PACKAGE_ROOT / "client" / "agent_client.py",
    PACKAGE_ROOT / "cli" / "interactive.py",
    PACKAGE_ROOT / "rpyc_service" / "agent_service.py",
)


def _make_checkout(root: Path) -> Path:
    """Create a minimal but genuine ``rpyc_async`` package under *root*."""
    pkg = root / "rpyc_async"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _restore_sys_path():
    """``prefer_local_rpyc_async`` mutates ``sys.path``; undo it after."""
    original = list(sys.path)
    yield
    sys.path[:] = original


class TestResolutionOrder:
    def test_checkout_goes_to_the_front_of_sys_path(self, tmp_path, monkeypatch):
        """Front, not back — behind an installed dist it would never win."""
        checkout = _make_checkout(tmp_path / "checkout" / "rpyc_async")
        monkeypatch.setattr(rpyc_async_source, "LOCAL_RPYC_ASYNC_PATH", checkout)
        monkeypatch.setattr(sys, "path", ["/somewhere/site-packages", "/other"])

        returned = rpyc_async_source.prefer_local_rpyc_async()

        assert returned == checkout
        assert sys.path[0] == str(checkout), (
            "a local checkout must precede site-packages, otherwise the "
            "installed distribution still wins and the priority is inverted"
        )

    def test_missing_path_falls_back_to_the_installed_package(
        self, tmp_path, monkeypatch
    ):
        """No checkout -> sys.path untouched, so the pip package is used."""
        absent = tmp_path / "checkout" / "rpyc_async"
        monkeypatch.setattr(rpyc_async_source, "LOCAL_RPYC_ASYNC_PATH", absent)
        before = ["/somewhere/site-packages"]
        monkeypatch.setattr(sys, "path", list(before))

        assert rpyc_async_source.prefer_local_rpyc_async() is None
        assert sys.path == before

    def test_empty_directory_is_not_treated_as_a_checkout(
        self, tmp_path, monkeypatch
    ):
        """A bare directory must not shadow a working installed package.

        Existence alone is not enough: a leftover empty directory would then
        put an unimportable path first and turn a cosmetic filesystem
        artefact into a startup ImportError.
        """
        empty = tmp_path / "checkout" / "rpyc_async"
        empty.mkdir(parents=True)
        monkeypatch.setattr(rpyc_async_source, "LOCAL_RPYC_ASYNC_PATH", empty)
        before = ["/somewhere/site-packages"]
        monkeypatch.setattr(sys, "path", list(before))

        assert rpyc_async_source.prefer_local_rpyc_async() is None
        assert sys.path == before

    def test_repeated_calls_do_not_grow_sys_path(self, tmp_path, monkeypatch):
        """Each entry point calls this; the effect must be idempotent."""
        checkout = _make_checkout(tmp_path / "checkout" / "rpyc_async")
        monkeypatch.setattr(rpyc_async_source, "LOCAL_RPYC_ASYNC_PATH", checkout)
        monkeypatch.setattr(sys, "path", ["/somewhere/site-packages"])

        for _ in range(3):
            rpyc_async_source.prefer_local_rpyc_async()

        assert sys.path.count(str(checkout)) == 1

    def test_unconfigured_means_installed_package(self, monkeypatch):
        """No RPYC_ASYNC_LOCAL_CHECKOUT -> no checkout, never a guessed path."""
        monkeypatch.delenv(rpyc_async_source.LOCAL_CHECKOUT_ENV, raising=False)
        assert rpyc_async_source._configured_path() is None

    def test_configured_path_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(rpyc_async_source.LOCAL_CHECKOUT_ENV, str(tmp_path))
        assert rpyc_async_source._configured_path() == tmp_path

    def test_none_path_is_not_a_checkout(self, monkeypatch):
        monkeypatch.setattr(rpyc_async_source, "LOCAL_RPYC_ASYNC_PATH", None)
        before = list(sys.path)
        assert rpyc_async_source.prefer_local_rpyc_async() is None
        assert sys.path == before


class TestEntryPointsEstablishPriority:
    """Each importer must pull in the resolver BEFORE importing rpyc_async.

    Order is the whole point: a side-effect import placed after
    ``import rpyc_async`` would be dead weight, since the module would
    already be bound from whichever path won first.
    """

    @pytest.mark.parametrize(
        "module_path", RPYC_IMPORTING_MODULES, ids=lambda p: p.name
    )
    def test_resolver_is_imported_before_rpyc_async(self, module_path: Path):
        src = module_path.read_text(encoding="utf-8")

        resolver = re.search(
            r"^import deep_worker_cli\.rpyc_async_source", src, re.MULTILINE
        )
        assert resolver, (
            f"{module_path.name} imports rpyc_async at module scope but never "
            "imports deep_worker_cli.rpyc_async_source, so on a host with a "
            "checkout it may bind the installed distribution instead"
        )

        first_rpyc = re.search(
            r"^(?:import rpyc_async|from rpyc_async)", src, re.MULTILINE
        )
        assert first_rpyc, f"{module_path.name} no longer imports rpyc_async"
        assert resolver.start() < first_rpyc.start(), (
            f"{module_path.name} imports rpyc_async before the resolver; the "
            "sys.path change must happen first or it has no effect"
        )


class TestNoResurrectedProhibition:
    """The retracted "PyPI only" claim must not creep back into the headers."""

    @pytest.mark.parametrize(
        "module_path", RPYC_IMPORTING_MODULES, ids=lambda p: p.name
    )
    def test_headers_do_not_claim_pypi_only(self, module_path: Path):
        src = module_path.read_text(encoding="utf-8")
        assert "IMPORTED FROM THE INSTALLED PACKAGE" not in src, (
            f"{module_path.name} reasserts the retracted claim that only the "
            "installed package is legitimate; a checkout takes priority"
        )
