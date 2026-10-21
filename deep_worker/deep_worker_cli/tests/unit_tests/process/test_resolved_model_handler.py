"""Regression: the resolver/display handler must not re-pin a foreign model.

Incident agent_e (2026-06-29): an agent launched as ``codex`` with
``config.model = None`` (unpinned) had its ``launch_config["model"]``
re-pinned to a stale ``grok-composer-2.5-fast`` on every restart because the
startup display seed read codex's session jsonl (which had recorded the
foreign string when the original ``-pp`` bug fed codex that model) and then
persisted it via ``SessionManager.update_launch_config_model``.

Fix 2 (DESIGN_FIX_CROSS_PROVIDER_MODEL_POISON.md): the handler now distinguishes
display from persistence:

  - Startup session-log seed → ``persist=False`` → display only.
  - Live provider callback → ``persist=True`` BUT only persists when the agent
    was launched with an explicit model (``config.model is not None``).
    An unpinned agent NEVER has its execution config silently pinned from a
    resolved value (display path still updates fine).

These tests pin the contract directly on the extracted factory
``_make_resolved_model_handler``.
"""

from __future__ import annotations

import pytest

from deep_worker_cli.process.agent_process import _make_resolved_model_handler


class _RecordingConfig:
    """Stand-in for ``AgentConfig`` capturing display-model updates."""

    def __init__(self, model: str | None) -> None:
        self.model = model
        self.display_calls: list[str] = []

    def set_dynamic_display_model(self, name: str) -> None:
        self.display_calls.append(name)


class _RecordingSessionManager:
    """Stand-in for ``SessionManager.update_launch_config_model``."""

    def __init__(self) -> None:
        self.persist_calls: list[str] = []

    async def update_launch_config_model(self, name: str) -> None:
        self.persist_calls.append(name)


@pytest.mark.asyncio
async def test_seed_persist_false_unpinned_updates_display_only() -> None:
    """Startup seed for an unpinned agent: display yes, persist no."""
    cfg = _RecordingConfig(model=None)
    sm = _RecordingSessionManager()
    handler = _make_resolved_model_handler(config=cfg, session_manager=sm)

    await handler("grok-composer-2.5-fast", persist=False)

    assert cfg.display_calls == ["grok-composer-2.5-fast"]
    assert sm.persist_calls == [], (
        "startup seed must NEVER persist a resolved model; "
        f"got {sm.persist_calls!r}"
    )


@pytest.mark.asyncio
async def test_live_callback_unpinned_does_not_persist() -> None:
    """Even a live runtime callback (default ``persist=True``) must NOT pin a
    resolved model on an unpinned agent. This is the durable guarantee:
    ``launch_config["model"]`` stays ``None`` for an agent launched without
    --model, no matter how the resolved value arrives.
    """
    cfg = _RecordingConfig(model=None)
    sm = _RecordingSessionManager()
    handler = _make_resolved_model_handler(config=cfg, session_manager=sm)

    await handler("some-default-model")  # persist=True (default)

    assert cfg.display_calls == ["some-default-model"]
    assert sm.persist_calls == [], (
        "unpinned agent must never have execution config re-pinned from a "
        f"resolved value; got {sm.persist_calls!r}"
    )


@pytest.mark.asyncio
async def test_live_callback_pinned_persists() -> None:
    """When the agent was launched WITH an explicit model, the live callback
    keeps the existing behavior — persists the resolved model so display
    continuity is preserved across restarts for genuinely pinned agents."""
    cfg = _RecordingConfig(model="opus")
    sm = _RecordingSessionManager()
    handler = _make_resolved_model_handler(config=cfg, session_manager=sm)

    await handler("opus")  # persist=True

    assert cfg.display_calls == ["opus"]
    assert sm.persist_calls == ["opus"]


@pytest.mark.asyncio
async def test_seed_persist_false_pinned_still_display_only() -> None:
    """The ``persist=False`` flag is authoritative — even on a pinned agent,
    a seed call with persist=False updates display only. Symmetric contract:
    the *callers* decide; the handler obeys."""
    cfg = _RecordingConfig(model="opus")
    sm = _RecordingSessionManager()
    handler = _make_resolved_model_handler(config=cfg, session_manager=sm)

    await handler("opus", persist=False)

    assert cfg.display_calls == ["opus"]
    assert sm.persist_calls == []


@pytest.mark.asyncio
async def test_no_session_manager_is_safe() -> None:
    """Defensive: handler must not crash when session_manager is None."""
    cfg = _RecordingConfig(model="opus")
    handler = _make_resolved_model_handler(config=cfg, session_manager=None)

    await handler("opus")  # persist=True path with no SM
    assert cfg.display_calls == ["opus"]


@pytest.mark.asyncio
async def test_config_without_set_dynamic_display_model_is_safe() -> None:
    """Defensive: an ``AgentConfig`` without ``set_dynamic_display_model``
    (the attribute is referenced via getattr on the live config) must not
    crash the handler. Display update is silently skipped."""
    class _MinimalConfig:
        def __init__(self) -> None:
            self.model = None

    cfg = _MinimalConfig()
    sm = _RecordingSessionManager()
    handler = _make_resolved_model_handler(config=cfg, session_manager=sm)

    await handler("foo", persist=False)
    await handler("foo")  # persist=True but model=None → still no persist
    assert sm.persist_calls == []
