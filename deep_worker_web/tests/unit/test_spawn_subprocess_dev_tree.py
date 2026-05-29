"""Regression: web-UI restart MUST honor the agent's persisted dev_tree.

Bug (2026-05-23): ``_spawn_subprocess`` built the relaunch PYTHONPATH from
the WEB SERVER's OWN checkout (``_tree_root`` derived from ``__file__``),
ignoring the agent's saved ``launch_config['dev_tree']``. So an agent
originally launched with ``--dev`` (dev codebase, /path/to/dev-tree),
when restarted from the PROD web UI (:7887), was respawned on the PROD
codebase — silently losing ``--dev``. The CLI ``deep-worker --restart``
launcher already auto-detects this; the web path bypassed it.

Contract under test:
  * config['dev_tree'] truthy  -> spawn from the DEV tree: PYTHONPATH points
    at /path/to/dev-tree/... AND env DEEP_WORKER_DEV_TREE=1 (so the
    relaunched agent re-persists dev_tree=True).
  * config['dev_tree'] falsy/absent -> unchanged: spawn from the web server's
    own tree (prod when run from the production tree).
  * DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE is honored (so tests don't depend on a
    real dev-tree checkout).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.routers.agents import _spawn_subprocess


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the Popen cmd + env that _spawn_subprocess builds."""
    calls: list[dict[str, Any]] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        calls.append({"cmd": cmd, "env": kwargs.get("env", {})})
        return MagicMock()

    monkeypatch.setattr("app.routers.agents.subprocess.Popen", fake_popen)
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    return calls


BASE = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "working_dir": "/path/to/dev-tree",
    "skip_permissions": True,
}


def test_dev_tree_true_spawns_from_dev_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": True}

    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)

    env = calls[0]["env"]
    pythonpath = env.get("PYTHONPATH", "")
    assert "/tmp/fake_dev_tree/deep_worker/deep_worker_cli" in pythonpath, (
        f"a dev_tree agent must be relaunched on the DEV tree, not the web "
        f"server's own tree. PYTHONPATH was {pythonpath!r}"
    )
    # The relaunched agent must re-persist dev_tree=True via this env marker.
    assert env.get("DEEP_WORKER_DEV_TREE") == "1"


def test_dev_tree_false_spawns_from_web_server_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": False}

    _spawn_subprocess("prod_agent", cfg, model=None, provider=None, clean=False)

    env = calls[0]["env"]
    pythonpath = env.get("PYTHONPATH", "")
    assert "/tmp/fake_dev_tree" not in pythonpath, (
        f"a non-dev agent must NOT be forced onto the dev tree. PYTHONPATH "
        f"was {pythonpath!r}"
    )


def test_dev_tree_absent_keeps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No dev_tree key at all (legacy config) -> default (web server tree)."""
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = dict(BASE)  # no dev_tree key

    _spawn_subprocess("legacy_agent", cfg, model=None, provider=None, clean=False)

    env = calls[0]["env"]
    assert "/tmp/fake_dev_tree" not in env.get("PYTHONPATH", "")
    # No spurious dev marker for a non-dev agent.
    assert env.get("DEEP_WORKER_DEV_TREE") != "1"


def test_dev_tree_true_without_override_uses_canonical_dev_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the test override, dev_tree=True must target the canonical
    /path/to/dev-tree."""
    monkeypatch.delenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", raising=False)
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": True}

    _spawn_subprocess("agent_a", cfg, model=None, provider=None, clean=False)

    pythonpath = calls[0]["env"].get("PYTHONPATH", "")
    assert "/path/to/dev-tree/deep_worker/deep_worker_cli" in pythonpath


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
