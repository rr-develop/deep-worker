"""Every session-stateful provider implements the SummarizerPort. Enforced, not assumed.

This file is the mechanism behind the invariant documented on
``SESSION_STATEFUL_PROVIDERS``. Before it existed the pairing was folk
knowledge, and drifted three separate ways:

  * ``codex`` was classified stateful with no port -> every summarization was a
    deterministic ``SummarizationFailedError`` (2026-08-23).
  * ``grok`` was classified STATELESS while sending only ``messages[sent_count:]``
    and persisting its session id -> it skipped the guard and silently dumped
    the full history into a fresh session.
  * ``claude-code-sdk`` implemented ``asummarize`` but never grew
    ``areset_after_summary``, and spelled its halves ``_asummarize_*`` against
    ACP's ``_summarize_*``. Nothing noticed, because both call sites duck-type
    the port with ``getattr``.

A misspelled or missing method must fail HERE, not in production at 3am on the
one turn that crosses the context limit.
"""

from __future__ import annotations

import asyncio

import pytest

from deep_worker_cli.context_limit.hot_cache import (
    SESSION_STATEFUL_PROVIDERS,
    STATELESS_PROVIDERS,
)
from deep_worker_cli.context_limit.summarizer_port import (
    SummarizerPort,
    implements_summarizer_port,
)

# provider id -> (module path, class name) of the model implementing the port.
PROVIDER_MODELS: dict[str, tuple[str, str]] = {
    "acp": ("providers.acp.chat_model", "AcpChatModel"),
    "claude-code": ("providers.acp.chat_model", "AcpChatModel"),
    "codex": ("providers.codex_provider.chat_model", "CodexChatModel"),
    "grok": ("providers.grok_provider.chat_model", "GrokChatModel"),
}

# Stateful providers whose model is not part of this distribution: the
# claude-code-sdk provider needs a fork of the Claude Agent SDK and is not
# shipped. Every other stateful provider must be mapped and checked.
_NOT_SHIPPED = frozenset({"claude-code-sdk"})
_SHIPPED_STATEFUL = sorted(SESSION_STATEFUL_PROVIDERS - _NOT_SHIPPED)


def _load(provider: str) -> type:
    import importlib

    module_path, class_name = PROVIDER_MODELS[provider]
    return getattr(importlib.import_module(module_path), class_name)


def test_every_stateful_provider_is_mapped_to_a_model() -> None:
    """A new stateful provider cannot be added without declaring its model."""
    assert set(_SHIPPED_STATEFUL) <= set(PROVIDER_MODELS), (
        "provider(s) in SESSION_STATEFUL_PROVIDERS with no entry in "
        f"PROVIDER_MODELS: {set(_SHIPPED_STATEFUL) - set(PROVIDER_MODELS)}"
    )


@pytest.mark.parametrize("provider", _SHIPPED_STATEFUL)
def test_stateful_provider_implements_summarizer_port(provider: str) -> None:
    """The load-bearing assertion: stateful <=> both port halves present."""
    model_cls = _load(provider)
    for name in ("asummarize", "areset_after_summary"):
        method = getattr(model_cls, name, None)
        assert method is not None, (
            f"{model_cls.__name__} ({provider}) is session-stateful but does "
            f"not implement {name}. Without it the summarization middleware "
            f"either crashes ({provider}=codex, 2026-08-23) or silently dumps "
            f"the entire history into a fresh provider session."
        )
        assert asyncio.iscoroutinefunction(method), (
            f"{model_cls.__name__}.{name} must be `async def` — the call sites "
            f"await it and gate on iscoroutinefunction()."
        )


@pytest.mark.parametrize("provider", _SHIPPED_STATEFUL)
def test_stateful_provider_satisfies_the_protocol(provider: str) -> None:
    """Structural conformance, so the Protocol is a real check and not a doc."""
    assert issubclass(_load(provider), SummarizerPort)


def test_stateless_and_stateful_sets_are_disjoint() -> None:
    """A provider is one or the other. ``grok`` was briefly neither-correct."""
    overlap = SESSION_STATEFUL_PROVIDERS & STATELESS_PROVIDERS
    assert not overlap, f"provider(s) classified both ways: {overlap}"


def test_grok_is_session_stateful() -> None:
    """Pins the 2026-08-24 correction.

    ``GrokChatModel`` sends ``state_messages[state.sent_count:]`` (the delta
    only) and persists ``grok_session_id`` + ``sent_count`` across restarts.
    That is session-stateful by definition. The previous test asserted the
    opposite and thereby pinned the defect in place.
    """
    assert "grok" in SESSION_STATEFUL_PROVIDERS
    assert "grok" not in STATELESS_PROVIDERS


def test_implements_summarizer_port_rejects_a_mock() -> None:
    """MagicMock must not read as a working port.

    An instance-level ``getattr`` probe returns a truthy child for any name, so
    every test double would look session-stateful and drag unrelated tests onto
    the stateful path (reviewer round B2, 2026-08-21).
    """
    from unittest.mock import MagicMock

    assert not implements_summarizer_port(MagicMock())


def test_implements_summarizer_port_rejects_half_a_port() -> None:
    """``asummarize`` alone is not the port — that is the claude-code-sdk gap."""

    class HalfPort:
        async def asummarize(self, **_kw: object) -> None: ...

    class FullPort(HalfPort):
        async def areset_after_summary(self, **_kw: object) -> None: ...

    assert not implements_summarizer_port(HalfPort())
    assert implements_summarizer_port(FullPort())


def test_implements_summarizer_port_rejects_sync_methods() -> None:
    """Both call sites gate on ``iscoroutinefunction``; a sync def is invisible."""

    class SyncPort:
        def asummarize(self, **_kw: object) -> None: ...
        def areset_after_summary(self, **_kw: object) -> None: ...

    assert not implements_summarizer_port(SyncPort())
