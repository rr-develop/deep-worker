# pyright: reportPrivateUsage=false, reportUnusedVariable=false
"""Usage-event callback must survive LangChain bind_tools (new-instance).

DESIGN_AGENT_TOKEN_STATS.md §5.2: custom providers receive the token-stats
state-machine callback via ``set_usage_event_callback``. ``bind_tools``
returns a NEW ``ClaudeCodeChatModel`` (LangChain immutability contract) and
PrivateAttrs are not copied automatically — the instance LangGraph actually
uses (turn-node bind + ``_bind_tools_for_middleware``) is the bind_tools
result, so the callback must be propagated or token stats silently go
missing (live integration test: empty ``token_stats.jsonl``).

Contract enforced here:
1. ``bind_tools([t])`` copy inherits ``_usage_event_callback``.
2. ``bind_tools([])`` is a no-op returning ``self`` — callback trivially kept.
3. A callback set on the ORIGINAL model after a bind still lands on the
   original only (no aliasing surprise) — the copy is a snapshot at bind time.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.tools import tool as langchain_tool

from deep_worker.providers.claude_code_provider.chat_model import ClaudeCodeChatModel


@langchain_tool
def _t_alpha(x: int) -> int:
    """Alpha tool."""
    return x


def _collector() -> tuple[list[tuple[str, object]], Callable[..., Coroutine[Any, Any, None]]]:
    events: list[tuple[str, object]] = []

    async def cb(**kwargs: object) -> None:
        events.append((str(kwargs.get("event", "")), kwargs.get("usage")))

    return events, cb


class TestUsageEventCallbackSurvivesBindTools:
    def test_bind_tools_copy_inherits_callback(self) -> None:
        events, cb = _collector()
        model = ClaudeCodeChatModel(base_options=None)
        model.set_usage_event_callback(cb)

        bound = model.bind_tools([_t_alpha])

        assert bound._usage_event_callback is cb

        # End-to-end: the copy actually emits through the inherited callback.
        async def emit() -> None:
            await bound._emit_usage_event(
                type("E", (), {"usage": {"input_tokens": 5}, "is_error": False})()
            )

        asyncio.run(emit())
        assert events == [("terminal", {"input_tokens": 5})]

    def test_empty_bind_tools_returns_self_keeps_callback(self) -> None:
        events, cb = _collector()
        model = ClaudeCodeChatModel(base_options=None)
        model.set_usage_event_callback(cb)

        bound = model.bind_tools([])

        assert bound is model
        assert bound._usage_event_callback is cb

    def test_copy_is_snapshot_not_alias(self) -> None:
        events, cb = _collector()
        model = ClaudeCodeChatModel(base_options=None)
        model.set_usage_event_callback(cb)

        bound = model.bind_tools([_t_alpha])
        assert bound._usage_event_callback is cb

        # Setting a NEW callback on the original after bind must NOT leak
        # into the already-created copy (bind-time snapshot semantics).
        events2, cb2 = _collector()
        model.set_usage_event_callback(cb2)
        assert bound._usage_event_callback is cb

    def test_fresh_model_bind_has_no_callback_by_default(self) -> None:
        bound = ClaudeCodeChatModel(base_options=None).bind_tools([_t_alpha])
        assert bound._usage_event_callback is None
