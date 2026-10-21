"""``deep-worker --restart <name>`` must preserve ``-l`` (context_limit).

Reported by ``agent_a`` 2026-08-18: agent launched with
``deep-worker -r agent_a -pp claude-code --model fable -l 400k`` consumed
over 600k tokens despite the explicit 400k ceiling. Root cause: the
explicit ``-l`` did NOT survive provider-switch restarts (it was reset
to the parser default on each restart, while the model kept running
with whatever default the provider picked).

See ``docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md`` §3.1.

Four scenarios are pinned here:

1. **Preserve on bare restart**: ``-l 400k`` → ``--restart`` (no flags) → 400k.
2. **Preserve across provider change**: ``-l 400k -pp claude-code`` →
   ``--restart -pp grok`` → 400k. This is the agent_a scenario.
3. **Explicit override**: ``-l 400k`` → ``--restart -l 800k`` → 800k.
4. **Default clears**: ``-l 400k`` → ``--restart -l default`` → ``None``.

Each test observes the field at the ``run_agent`` boundary (the
single place where the CLI dataflow is observable), following the
pattern of the existing ``test_restart_preserves_launch_config.py``
suite (see ``DESIGN_RESTART_PRESERVES_LAUNCH_CONFIG.md`` §8 for why
"read the persisted file" is insufficient as a test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deep_worker_cli.cli import agent_command
from deep_worker_cli.cli.agent_command import _CONTEXT_LIMIT_CLEAR_SENTINEL


# Match the parser defaults from ``agent_command._PARSER_DEFAULTS``.
_PARSER_DEFAULTS = {
    "preferred_provider": None,
}


def _write_saved_config(
    tmp_path: Path,
    name: str,
    extra: dict,
) -> Path:
    """Seed a saved ``session_metadata.json`` carrying the given extras.

    Mirrors the pattern from ``test_restart_preserves_launch_config.py``:
    the file lives at ``<deep_worker_home>/agents/<name>/session_metadata.json``
    (DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §4.5).
    """
    data_dir = tmp_path / "agents" / name
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir()
    launch_config = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "working_dir": str(tmp_path),
        "skip_permissions": False,
        "task_manager": False,
        "memory_only": False,
        "history_memory": False,
        "use_sqlite": False,
        "clean": False,
        "auto_approve_send_message": False,
        "dev_tree": False,
        "web_search_policy": "auto",
        "preferred_provider": "deepseek",
        "preferred_model": None,
        "mcp_config_paths": [],
        "mcp_server_inline": [],
        "turn_ending_tools": [],
        "storage_version": 5,
        "tree_root": "/tmp",
        "started_at": "2026-08-18T00:00:00",
        "pid": 1234,
        "cli_args": "",
    }
    launch_config.update(extra)
    metadata = {
        "status": "idle",
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": "2026-08-18T00:00:00",
        "storage_version": 5,
        "launch_config": launch_config,
    }
    (data_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2))
    return data_dir


def _restart_args(name: str, **cli_overrides) -> argparse.Namespace:
    """Build the argparse namespace ``deep-worker --restart <name>`` produces.

    All fields the CLI parser sets on a bare ``--restart`` invocation are
    listed here so tests can override individual ones. ``context_limit``
    defaults to ``None`` to match the bare-restart behaviour.
    """
    ns = argparse.Namespace(
        name=name,
        restart=True,
        provider="deepseek",
        model=None,
        dir=None,
        skip_permissions=False,
        task_manager=False,
        memory_only=False,
        history_memory=False,
        use_sqlite=False,
        clean=False,
        auto_approve_send_message=False,
        dev_tree=False,
        web_search_policy="auto",
        preferred_provider=_PARSER_DEFAULTS["preferred_provider"],
        preferred_model=None,
        no_update_preferred=False,
        mcp_config_paths=[],
        mcp_server_inline=[],
        turn_ending_tools=[],
        config=None,
        summarize=False,
        fork=None,
        fork_from=None,
        context_limit=None,
    )
    for k, v in cli_overrides.items():
        setattr(ns, k, v)
    return ns


class _EventLike:
    async def wait(self):
        return None


def _no_registered_agent(monkeypatch):
    """Make ``ProcessRegistry.lookup`` return None so restart skips the stop step."""

    class _EmptyRegistry:
        def __init__(self, *a, **k):
            pass

        def lookup(self, name):
            return None

    monkeypatch.setattr(
        "deep_worker_cli.process.registry.ProcessRegistry",
        _EmptyRegistry,
    )


@pytest.fixture
def _sandbox(tmp_path, monkeypatch):
    """Point deep_worker_home at a per-test temp dir.

    Both call sites are patched because ``agent_root`` (in
    ``deep_worker_cli.storage_paths``) uses its own module-level binding
    to ``deep_worker_home``, while ``agent_command.py`` keeps a direct
    import alias. DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §4.5.
    """
    monkeypatch.setattr(
        "deep_worker_cli.cli.agent_command.deep_worker_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "deep_worker_cli.storage_paths.deep_worker_home",
        lambda h=None: tmp_path,
    )
    return tmp_path


# ─── The four required scenarios ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_preserves_context_limit(_sandbox, monkeypatch):
    """Bare ``--restart`` after ``-l 400k`` MUST keep 400k.

    Operator expectation: the explicit limit survives unless the
    operator changes it. This is the baseline scenario.
    """
    _write_saved_config(
        _sandbox, "a", {"context_limit": 400_000}
    )
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["context_limit"] = getattr(effective_ns, "context_limit", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    rc = await agent_command.restart_agent(_restart_args("a"))

    assert rc == 0
    assert captured.get("context_limit") == 400_000, (
        f"context_limit was dropped on bare --restart: "
        f"got {captured.get('context_limit')!r}, expected 400_000. "
        f"This is the agent_a bug (2026-08-18): the limit is "
        f"silently reset to None on every restart, while the model "
        f"keeps running with whatever default the provider picks."
    )


@pytest.mark.asyncio
async def test_restart_with_different_provider_preserves_context_limit(
    _sandbox, monkeypatch
):
    """``-l 400k -pp claude-code`` → ``--restart -pp grok`` → 400k.

    THE agent_a scenario: the operator set 400k on claude-code, the
    agent got hot-failed-over to grok, and the limit was reset. The
    fix: the limit is independent of (provider, model) — only an
    explicit ``-l`` change moves it.
    """
    _write_saved_config(
        _sandbox, "a",
        {"context_limit": 400_000, "preferred_provider": "claude-code"},
    )
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["context_limit"] = getattr(effective_ns, "context_limit", None)
        captured["preferred_provider"] = getattr(
            effective_ns, "preferred_provider", None
        )
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    rc = await agent_command.restart_agent(
        _restart_args("a", preferred_provider="grok")
    )

    assert rc == 0
    assert captured.get("context_limit") == 400_000, (
        f"context_limit must survive provider change: "
        f"got {captured.get('context_limit')!r}, expected 400_000. "
        f"This is the EXACT agent_a scenario (2026-08-18)."
    )
    assert captured.get("preferred_provider") == "grok", (
        "the provider change itself should still take effect"
    )


@pytest.mark.asyncio
async def test_restart_explicit_l_flag_overrides_saved_context_limit(
    _sandbox, monkeypatch
):
    """``-l 400k`` → ``--restart -l 800k`` → 800k (operator wins).

    An explicit ``-l`` on the restart command line MUST override the
    saved value — that's the whole point of running the restart with
    a different ``-l``. The "preserve" rule only applies to bare
    restarts.
    """
    _write_saved_config(
        _sandbox, "a", {"context_limit": 400_000}
    )
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["context_limit"] = getattr(effective_ns, "context_limit", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    rc = await agent_command.restart_agent(
        _restart_args("a", context_limit=800_000)
    )

    assert rc == 0
    assert captured.get("context_limit") == 800_000, (
        f"explicit -l on restart must override saved value: "
        f"got {captured.get('context_limit')!r}, expected 800_000"
    )


@pytest.mark.asyncio
async def test_restart_with_different_provider_preserves_context_limit(
    _sandbox, monkeypatch
):
    """``-l 400k -pp claude-code`` → ``--restart -pp grok`` → 400k.

    THE agent_a scenario: the operator set 400k on claude-code, the
    agent got hot-failed-over to grok, and the limit was reset. The
    fix: the limit is independent of (provider, model) — only an
    explicit ``-l`` change moves it.
    """
    _write_saved_config(
        _sandbox, "a",
        {"context_limit": 400_000, "preferred_provider": "claude-code"},
    )
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["context_limit"] = getattr(effective_ns, "context_limit", None)
        captured["preferred_provider"] = getattr(
            effective_ns, "preferred_provider", None
        )
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    rc = await agent_command.restart_agent(
        _restart_args("a", preferred_provider="grok")
    )

    assert rc == 0
    assert captured.get("context_limit") == 400_000, (
        f"context_limit must survive provider change: "
        f"got {captured.get('context_limit')!r}, expected 400_000. "
        f"This is the EXACT agent_a scenario (2026-08-18)."
    )
    assert captured.get("preferred_provider") == "grok", (
        "the provider change itself should still take effect"
    )


@pytest.mark.asyncio
async def test_restart_explicit_l_flag_overrides_saved_context_limit(
    _sandbox, monkeypatch
):
    """``-l 400k`` → ``--restart -l 800k`` → 800k (operator wins).

    An explicit ``-l`` on the restart command line MUST override the
    saved value — that's the whole point of running the restart with
    a different ``-l``. The "preserve" rule only applies to bare
    restarts.
    """
    _write_saved_config(
        _sandbox, "a", {"context_limit": 400_000}
    )
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["context_limit"] = getattr(effective_ns, "context_limit", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    rc = await agent_command.restart_agent(
        _restart_args("a", context_limit=800_000)
    )

    assert rc == 0
    assert captured.get("context_limit") == 800_000, (
        f"explicit -l on restart must override saved value: "
        f"got {captured.get('context_limit')!r}, expected 800_000"
    )


@pytest.mark.asyncio
async def test_restart_l_default_clears_saved_context_limit(
    _sandbox, monkeypatch
):
    """``-l 400k`` → ``--restart -l default`` → ``None`` (back to default).

    ``-l default`` is the operator's way of saying "I want the default
    behaviour back". It must clear the saved value so the agent uses
    the provider's default window on the next run.
    """
    _write_saved_config(
        _sandbox, "a", {"context_limit": 400_000}
    )
    _no_registered_agent(monkeypatch)

    captured: dict = {}

    async def _capture_run_agent(effective_ns):
        captured["context_limit"] = getattr(effective_ns, "context_limit", None)
        return 0

    monkeypatch.setattr(agent_command, "run_agent", _capture_run_agent)

    # argparse's ``type=parse_context_limit`` converts the CLI token
    # ``"default"`` to the sentinel; we replicate that here.
    rc = await agent_command.restart_agent(
        _restart_args("a", context_limit=_CONTEXT_LIMIT_CLEAR_SENTINEL)
    )

    assert rc == 0
    assert captured.get("context_limit") is None, (
        f"-l default on restart must clear saved value: "
        f"got {captured.get('context_limit')!r}, expected None. "
        f"The operator explicitly asked for the default behaviour; "
        f"the saved 400k must NOT be restored."
    )