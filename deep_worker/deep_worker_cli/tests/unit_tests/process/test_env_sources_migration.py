"""Env sources come from ``~/.deep_worker`` ONLY (POLICIES §16).

Owner directive (2026-08-31), superseding the 2026-08-15 migration
contract: ``~/.deepagents`` is banned for READS as well as writes. The
earlier version of this file pinned a legacy fallback; that fallback is
now itself the defect, because loading secrets out of the ancestor's
tree is exactly the silent-authority failure the ban exists to prevent.

Contract:

* ``~/.deep_worker/.env`` and ``~/.deep_worker/env.d`` are the ONLY sources.
* ``~/.deepagents`` is never consulted, even when it exists on disk.
* Ordering within the tree is unchanged: ``.env`` before ``env.d``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from deep_worker_cli.process.env_loader import env_source_paths, load_env_sources


def _write(path: Path, body: str, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(mode)
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MM_TEST_KEY", "MM_ONLY_LEGACY", "MM_BOTH"):
        monkeypatch.delenv(var, raising=False)


class TestEnvSourcePaths:
    """Only the product tree may appear among the env sources."""

    def test_legacy_tree_is_never_a_source(self, tmp_path: Path) -> None:
        as_str = [str(p) for p in env_source_paths(home=tmp_path)]
        assert not any(".deepagents" in p for p in as_str), (
            "~/.deepagents must never be consulted — POLICIES §16 bans it "
            "for reads as well as writes"
        )

    def test_dot_env_precedes_env_d(self, tmp_path: Path) -> None:
        as_str = [str(p) for p in env_source_paths(home=tmp_path)]
        dot_env = next(i for i, p in enumerate(as_str) if p.endswith(".env"))
        env_d = next(i for i, p in enumerate(as_str) if p.endswith("env.d"))
        assert dot_env < env_d, ".env must load before env.d"

    def test_product_tree_is_represented(self, tmp_path: Path) -> None:
        as_str = [str(p) for p in env_source_paths(home=tmp_path)]
        assert any(".deep_worker" in p for p in as_str)


class TestLoadEnvSources:
    def test_loads_from_new_tree(self, tmp_path: Path) -> None:
        _write(tmp_path / ".deep_worker" / "env.d" / "minimax.env",
               "MM_TEST_KEY=from-new\n")
        applied = load_env_sources(home=tmp_path)
        assert applied.get("MM_TEST_KEY") == "from-new"
        assert os.environ["MM_TEST_KEY"] == "from-new"

    def test_legacy_tree_is_ignored_even_when_present(self, tmp_path: Path) -> None:
        """A value that exists ONLY in ~/.deepagents must not be loaded."""
        _write(tmp_path / ".deepagents" / "env.d" / "old.env",
               "MM_ONLY_LEGACY=from-legacy\n")
        applied = load_env_sources(home=tmp_path)
        assert "MM_ONLY_LEGACY" not in applied
        assert "MM_ONLY_LEGACY" not in os.environ

    def test_product_tree_value_is_used_and_legacy_cannot_shadow_it(
        self, tmp_path: Path
    ) -> None:
        """A stale legacy copy must not influence the resolved value."""
        _write(tmp_path / ".deep_worker" / "env.d" / "a.env", "MM_BOTH=new\n")
        _write(tmp_path / ".deepagents" / "env.d" / "a.env", "MM_BOTH=legacy\n")
        load_env_sources(home=tmp_path)
        assert os.environ["MM_BOTH"] == "new"

    def test_dot_env_beats_env_d_in_same_tree(self, tmp_path: Path) -> None:
        _write(tmp_path / ".deep_worker" / ".env", "MM_BOTH=dotenv\n")
        _write(tmp_path / ".deep_worker" / "env.d" / "z.env", "MM_BOTH=dropin\n")
        load_env_sources(home=tmp_path)
        assert os.environ["MM_BOTH"] == "dotenv"

    def test_preexisting_environ_is_never_overridden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MM_TEST_KEY", "from-shell")
        _write(tmp_path / ".deep_worker" / "env.d" / "a.env", "MM_TEST_KEY=from-file\n")
        load_env_sources(home=tmp_path)
        assert os.environ["MM_TEST_KEY"] == "from-shell"

    def test_missing_trees_are_a_noop(self, tmp_path: Path) -> None:
        assert load_env_sources(home=tmp_path) == {}

    def test_loose_permissions_still_rejected_in_new_tree(
        self, tmp_path: Path
    ) -> None:
        """The 0600 gate must not be lost in the migration."""
        _write(tmp_path / ".deep_worker" / "env.d" / "bad.env",
               "MM_TEST_KEY=leaky\n", mode=0o644)
        load_env_sources(home=tmp_path)
        assert "MM_TEST_KEY" not in os.environ
