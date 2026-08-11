"""TDD: the claude-code delivery-resolve middleware is wired ONLY for
claude-code models (Bug 2, DESIGN_SUBSCRIPTION_AND_DELIVERY_FIX.md).

The shared SendMessageCompletionMiddleware stays provider-agnostic; the
proxy-MCP early-resolve is added to the stack only when the agent's model is a
ClaudeCodeChatModel. We unit-test the small wiring helper in isolation so we
don't have to build a whole agent graph.
"""

from __future__ import annotations

from typing import Any

from deep_worker_cli.agent import _maybe_add_delivery_resolve_middleware


class _FakeNonClaudeModel:
    """A stand-in for codex/deepseek models (no proxy-MCP)."""


def _mw_type_names(stack: list[Any]) -> list[str]:
    return [type(m).__name__ for m in stack]


def test_added_for_claude_code_model() -> None:
    """A ClaudeCodeChatModel instance gets the delivery-resolve middleware."""
    from deep_worker.providers.claude_code_provider import ClaudeCodeChatModel

    model = ClaudeCodeChatModel(base_options=None)
    stack: list[Any] = []
    _maybe_add_delivery_resolve_middleware(stack, model)

    assert "InterAgentDeliveryResolveMiddleware" in _mw_type_names(stack)


def test_not_added_for_non_claude_model() -> None:
    """Codex/deepseek (and a plain string model) must NOT get it."""
    stack: list[Any] = []
    _maybe_add_delivery_resolve_middleware(stack, _FakeNonClaudeModel())
    assert stack == []

    stack2: list[Any] = []
    _maybe_add_delivery_resolve_middleware(stack2, "anthropic:claude-sonnet-4-5")
    assert stack2 == []


def test_middleware_bound_to_the_model() -> None:
    """The middleware must hold the exact model instance so resolve_pending_tool
    targets the right ClientManager."""
    from deep_worker.providers.claude_code_provider import ClaudeCodeChatModel

    model = ClaudeCodeChatModel(base_options=None)
    stack: list[Any] = []
    _maybe_add_delivery_resolve_middleware(stack, model)

    mw = stack[0]
    assert getattr(mw, "_model", None) is model
