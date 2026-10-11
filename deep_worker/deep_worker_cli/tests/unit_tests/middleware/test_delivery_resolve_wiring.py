"""The delivery-resolve middleware is wired only for models that expose
``resolve_pending_tool`` (proxy-MCP models; DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md).

The shared SendMessageCompletionMiddleware stays provider-agnostic; models
without proxy-MCP must not get the early-resolve middleware. We unit-test the
small wiring helper in isolation so we don't have to build a whole agent graph.
"""

from __future__ import annotations

from typing import Any

from deep_worker_cli.agent import _maybe_add_delivery_resolve_middleware


class _FakeNonClaudeModel:
    """A stand-in for codex/deepseek models (no proxy-MCP)."""


def test_not_added_for_non_claude_model() -> None:
    """Codex/deepseek (and a plain string model) must NOT get it."""
    stack: list[Any] = []
    _maybe_add_delivery_resolve_middleware(stack, _FakeNonClaudeModel())
    assert stack == []

    stack2: list[Any] = []
    _maybe_add_delivery_resolve_middleware(stack2, "anthropic:claude-sonnet-4-5")
    assert stack2 == []


