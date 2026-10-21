"""Web-restart MUST preserve every persisted boolean option.

Bug (2026-06-05): ``_spawn_subprocess`` hand-built the relaunch CLI as a
sequence of opt-in conditionals (``if config.get("skip_permissions")`` etc.).
Any persisted boolean NOT in that list — notably ``history_memory`` (``-m``),
``auto_approve_send_message`` (``-asm``), ``dev_tree`` (``--dev``) — was
silently dropped on restart. The user disabled `-m` only by NOT using it;
the bug is that web restart did the same. Tests below pin the contract.

Contract under test:
  * Every dest listed in ``_BOOLEAN_FLAGS`` is emitted as its CLI flag when
    persisted with a truthy value.
  * No flag is emitted when the persisted value is falsy / missing.
  * ``_NON_PRESERVED_BOOLEAN_OPTIONS`` (``clean`` etc.) are NEVER emitted
    from saved config — ``clean`` is controlled exclusively by the explicit
    ``clean=`` argument to ``_spawn_subprocess``.

See docs/DESIGN_RESTART_OPTION_PRESERVATION.md.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.routers.agents import _spawn_subprocess
from deep_worker_cli.cli.agent_command import (
    _BOOLEAN_FLAGS,
    _NON_PRESERVED_BOOLEAN_OPTIONS,
)


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the Popen cmd that _spawn_subprocess builds.

    Each captured entry has:
      * ``cmd`` — the full argv (incl. ``python3 -m deep_worker_cli...``)
      * ``env`` — the env mapping
      * ``agent_args`` — argv slice starting AFTER the agent name (the
        actual flags the agent process will see). Using ``agent_args`` for
        flag-presence checks avoids false positives from ``python3 -m`` —
        e.g. searching for ``-m`` (history_memory) in the full cmd would
        always match the Python interpreter's own ``-m`` switch.
    """
    calls: list[dict[str, Any]] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        # The CLI layout is:
        #   python3 -m deep_worker_cli.cli.agent_command <AGENT_NAME> <flags...>
        # so agent_args = everything after index 4.
        agent_args = cmd[5:] if len(cmd) > 5 else []
        calls.append({
            "cmd": cmd, "env": kwargs.get("env", {}), "agent_args": agent_args,
        })
        return MagicMock()

    monkeypatch.setattr("app.routers.agents.subprocess.Popen", fake_popen)
    monkeypatch.setattr("builtins.open", lambda *a, **k: MagicMock())
    return calls


BASE = {
    "provider": "claude-code-sdk",
    "model": "opus",
    "working_dir": "/path/to/production",
}


# ---------------------------------------------------------------------------
# Specific regressions for the 2026-06-05 bug
# ---------------------------------------------------------------------------


def test_spawn_preserves_history_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for the actual 2026-06-05 incident: -m dropped on web restart."""
    calls = _capture(monkeypatch)
    cfg = {**BASE, "history_memory": True}
    _spawn_subprocess("agent_1", cfg, model=None, provider=None, clean=False)
    assert "-m" in calls[0]["agent_args"], (
        f"history_memory=True was persisted but '-m' was NOT emitted to the "
        f"spawn cmd. This is the 2026-06-05 regression. cmd={calls[0]["agent_args"]}"
    )


def test_spawn_preserves_auto_approve_send_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture(monkeypatch)
    cfg = {**BASE, "auto_approve_send_message": True}
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    assert "-asm" in calls[0]["agent_args"], (
        f"auto_approve_send_message=True must emit -asm. cmd={calls[0]["agent_args"]}"
    )


def test_spawn_preserves_dev_tree_via_cli_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dev_tree has BOTH an env-var path (already in place) AND must now be
    emitted on the CLI for self-describing logs / cli_args."""
    monkeypatch.setenv("DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree")
    calls = _capture(monkeypatch)
    cfg = {**BASE, "dev_tree": True}
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    assert "--dev" in calls[0]["agent_args"], (
        f"dev_tree=True must emit --dev on the CLI line. cmd={calls[0]["agent_args"]}"
    )


# ---------------------------------------------------------------------------
# Negative-classification guarantees (clean / no_update_preferred)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excluded_dest", sorted(_NON_PRESERVED_BOOLEAN_OPTIONS))
def test_spawn_does_not_emit_excluded_options_from_saved_config(
    excluded_dest: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clean and no_update_preferred must NEVER be emitted from saved config.

    They are per-launch / one-shot intents. The registry skips them; only
    explicit per-call arguments may emit them (clean= kwarg of
    _spawn_subprocess; no_update_preferred has no spawn-time path).
    """
    calls = _capture(monkeypatch)
    cfg = {**BASE, excluded_dest: True}
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    # We don't know the dest's CLI form here (it's NOT in _BOOLEAN_FLAGS by
    # definition). For clean it's '--clean'; for no_update_preferred it's
    # '--no-update-preferred'. The key invariant is that NEITHER appears.
    cmd_str = " ".join(calls[0]["cmd"])
    assert "--clean" not in calls[0]["agent_args"], (
        f"persisted {excluded_dest}=True must not emit --clean. cmd={cmd_str}"
    )
    assert "--no-update-preferred" not in calls[0]["agent_args"], (
        f"persisted {excluded_dest}=True must not emit --no-update-preferred. "
        f"cmd={cmd_str}"
    )


def test_spawn_preserves_external_mcp_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture(monkeypatch)
    cfg = {
        **BASE,
        "mcp_config_paths": ["/tmp/one.json", "/tmp/two.json"],
        "mcp_server_inline": ['demo={"command":"demo"}'],
    }

    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)

    args = calls[0]["agent_args"]
    assert args.count("--mcp-config") == 2
    assert "/tmp/one.json" in args
    assert "/tmp/two.json" in args
    assert args.count("--mcp-server") == 1
    assert 'demo={"command":"demo"}' in args


def test_spawn_emits_clean_only_when_explicit_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--clean is governed exclusively by the clean= kwarg, never by config."""
    calls = _capture(monkeypatch)
    cfg = dict(BASE)
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=True)
    assert "--clean" in calls[0]["agent_args"], "explicit clean=True must emit --clean"

    calls2 = _capture(monkeypatch)
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    assert "--clean" not in calls2[0]["agent_args"], (
        "clean=False must NOT emit --clean"
    )


# ---------------------------------------------------------------------------
# Falsy / missing behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dest, true_flag, _false_flag", _BOOLEAN_FLAGS)
def test_spawn_omits_flag_when_value_is_false(
    dest: str, true_flag: str, _false_flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture(monkeypatch)
    cfg = {**BASE, dest: False}
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    assert true_flag not in calls[0]["agent_args"], (
        f"{dest}=False must NOT emit {true_flag!r}. cmd={calls[0]["agent_args"]}"
    )


@pytest.mark.parametrize("dest, true_flag, _false_flag", _BOOLEAN_FLAGS)
def test_spawn_omits_flag_when_key_missing(
    dest: str, true_flag: str, _false_flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy config that predates this option (key absent) must behave
    identically to value=False."""
    calls = _capture(monkeypatch)
    cfg = dict(BASE)  # NO booleans set at all
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    assert true_flag not in calls[0]["agent_args"], (
        f"missing {dest!r} must NOT emit {true_flag!r}. cmd={calls[0]["agent_args"]}"
    )


# ---------------------------------------------------------------------------
# THE registry-driven sweep — adding a new bool to _BOOLEAN_FLAGS auto-extends
# this coverage. No per-flag maintenance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dest, true_flag, _false_flag", _BOOLEAN_FLAGS)
def test_spawn_emits_flag_for_every_registry_entry(
    dest: str, true_flag: str, _false_flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For EVERY persisted boolean, value=True must result in true_flag in cmd.

    This is the self-extending part: adding a new bool to _BOOLEAN_FLAGS
    automatically grows this test. No silent regression possible.
    """
    # dev_tree needs an override env var to avoid touching a real /repos path
    if dest == "dev_tree":
        monkeypatch.setenv(
            "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/tmp/fake_dev_tree"
        )
    calls = _capture(monkeypatch)
    cfg = {**BASE, dest: True}
    _spawn_subprocess("ag", cfg, model=None, provider=None, clean=False)
    assert true_flag in calls[0]["agent_args"], (
        f"{dest}=True must emit {true_flag!r}. cmd={calls[0]["agent_args"]}"
    )
