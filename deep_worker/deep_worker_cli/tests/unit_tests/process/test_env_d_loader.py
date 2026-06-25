"""TDD: auto-load of ``~/.deepagents/env.d/*.env`` at agent start.

Design (see DESIGN_EXTERNAL_MCP_TOOLS.md §7):

* For each ``*.env`` file in ``~/.deepagents/env.d/`` (sorted), agent start
  loads variables into ``os.environ``.
* Already-set variables (incl. those from ``~/.deepagents/.env`` loaded
  *before* this step, and anything inherited from the parent shell) are
  NOT overridden. The main ``.env`` keeps priority.
* Files with permissions wider than ``0o600`` are SKIPPED with a warning
  in the agent log — this matches the existing ``mcp.json`` policy
  (secrets must not be world/group readable).
* Missing directory ⇒ silent no-op (so agents without env.d work as today).
* Non-``.env`` files inside ``env.d`` are ignored.
* Files are loaded in deterministic, sorted order so that an earlier
  ``A.env`` keeps priority over a later ``B.env`` for the same key
  (consistent with the "first set wins" rule above).

Tests target the pure helper ``load_env_d`` exposed by
``deep_worker_cli.process.env_loader`` so we can exercise the behaviour
without spawning a full agent.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from deep_worker_cli.process.env_loader import load_env_d


def _write(p: Path, body: str, mode: int = 0o600) -> None:
    p.write_text(body)
    p.chmod(mode)


def test_loads_single_env_file_into_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    _write(env_d / "my-chrome-mcp.env", "MCP_ADMIN_TOKEN=token-xyz\n")
    monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)

    loaded = load_env_d(env_d)

    assert "MCP_ADMIN_TOKEN" in loaded
    assert os.environ["MCP_ADMIN_TOKEN"] == "token-xyz"


def test_missing_directory_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_d = tmp_path / "env.d"  # NOT created
    monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)

    loaded = load_env_d(env_d)

    assert loaded == {}
    assert "MCP_ADMIN_TOKEN" not in os.environ


def test_existing_variable_is_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main ``.env`` (and shell-inherited vars) MUST keep priority.

    This is the safety net that lets a user override a stale env.d-supplied
    value just by editing ``~/.deepagents/.env`` without removing the
    drop-in.
    """
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    _write(env_d / "x.env", "MY_VAR=from-envd\n")
    monkeypatch.setenv("MY_VAR", "from-main-env")

    load_env_d(env_d)

    assert os.environ["MY_VAR"] == "from-main-env"


def test_file_with_loose_permissions_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    loose = env_d / "loose.env"
    _write(loose, "LOOSE_TOKEN=should-not-load\n", mode=0o644)
    tight = env_d / "tight.env"
    _write(tight, "TIGHT_TOKEN=should-load\n", mode=0o600)
    monkeypatch.delenv("LOOSE_TOKEN", raising=False)
    monkeypatch.delenv("TIGHT_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger="deep_worker_cli.process.env_loader"):
        load_env_d(env_d)

    assert "LOOSE_TOKEN" not in os.environ, (
        "0644-mode env file MUST be ignored — it could leak secrets"
    )
    assert os.environ.get("TIGHT_TOKEN") == "should-load"
    assert any(
        "loose.env" in rec.getMessage() and "0600" in rec.getMessage()
        for rec in caplog.records
    ), f"expected a warning naming loose.env and 0600; got: {caplog.records!r}"


def test_only_dotenv_files_are_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    _write(env_d / "a.env", "A_VAR=loaded\n")
    _write(env_d / "ignored.txt", "B_VAR=should-not-load\n")
    _write(env_d / "README", "README_TOKEN=should-not-load\n")
    monkeypatch.delenv("A_VAR", raising=False)
    monkeypatch.delenv("B_VAR", raising=False)
    monkeypatch.delenv("README_TOKEN", raising=False)

    load_env_d(env_d)

    assert os.environ.get("A_VAR") == "loaded"
    assert "B_VAR" not in os.environ
    assert "README_TOKEN" not in os.environ


def test_loads_in_sorted_order_and_first_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files are loaded sorted; earlier filename wins for the same key.

    Combined with override=False this gives a fully deterministic, no-flap
    resolution order.
    """
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    _write(env_d / "a.env", "SHARED=from-a\n")
    _write(env_d / "b.env", "SHARED=from-b\n")
    monkeypatch.delenv("SHARED", raising=False)

    load_env_d(env_d)

    assert os.environ["SHARED"] == "from-a"


def test_symlink_to_canonical_file_is_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical case: ``env.d/my-chrome-mcp.env`` is a symlink to
    ``~/.config/my-chrome-mcp/admin.env`` (mode 0600). The loader MUST
    follow the symlink and check the *target's* permissions.
    """
    canonical = tmp_path / "admin.env"
    _write(canonical, "MCP_ADMIN_TOKEN=via-symlink\n", mode=0o600)
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    (env_d / "my-chrome-mcp.env").symlink_to(canonical)
    monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)

    load_env_d(env_d)

    assert os.environ.get("MCP_ADMIN_TOKEN") == "via-symlink"


def test_symlink_to_loose_permission_file_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canonical = tmp_path / "loose.env"
    _write(canonical, "LEAKY=nope\n", mode=0o644)
    env_d = tmp_path / "env.d"
    env_d.mkdir()
    (env_d / "leaky.env").symlink_to(canonical)
    monkeypatch.delenv("LEAKY", raising=False)

    with caplog.at_level(logging.WARNING, logger="deep_worker_cli.process.env_loader"):
        load_env_d(env_d)

    assert "LEAKY" not in os.environ
