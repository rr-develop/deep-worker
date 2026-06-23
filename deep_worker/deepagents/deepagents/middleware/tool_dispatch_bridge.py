"""Provider-agnostic bridge so the unified tool-call bound also covers a
proxy provider's "stage A" — the wait for the CLI to actually dispatch the
announced tool call over its control channel and yield the AUTHORITATIVE
arguments.

WHY THIS EXISTS (POLICIES.md §7.1, owner 2026-06-17): CLI providers
(claude-code / codex) announce a tool call in the model's message stream and
deliver the real ``tools/call`` (with the authoritative args — the
announcement args can differ, observed live, 2026-05-26) a moment later over a
separate control channel. The old code waited for that delivery INSIDE
``ainvoke`` with NO timeout (``wait_for_pending_call_by_id``) — a wait OUTSIDE
the single bounded point, i.e. a hole that hung live agents.

This bridge moves that wait INTO ``BoundedToolCallMiddleware.awrap_tool_call``
so it is covered by the SAME 120s timer as execution. ``deepagents`` does NOT
import any provider: a provider REGISTERS a dispatch-waiter for its thread_id;
the middleware looks it up by thread_id and awaits it (bounded). Normal
(non-proxy) providers register nothing → no-op (their args are already
authoritative because the LLM returns them directly).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

# A waiter takes the tool_call_id and returns the authoritative args dict once
# the provider has dispatched the call (or ``None`` if it cannot — e.g. an
# orphaned/unknown id; the middleware then proceeds with the announced args).
DispatchWaiter = Callable[[str], Awaitable[dict[str, Any] | None]]

_WAITERS: dict[str, DispatchWaiter] = {}


def register_dispatch_waiter(thread_id: str, waiter: DispatchWaiter) -> None:
    """A proxy provider registers how to await dispatch of a tool call for
    ``thread_id``. Overwrites any prior waiter for the same thread."""
    _WAITERS[thread_id] = waiter


def unregister_dispatch_waiter(thread_id: str) -> None:
    """Drop the waiter for ``thread_id`` (provider cleanup). Idempotent."""
    _WAITERS.pop(thread_id, None)


def get_dispatch_waiter(thread_id: str | None) -> DispatchWaiter | None:
    """The dispatch-waiter for ``thread_id``, or ``None`` for non-proxy
    providers (then dispatch waiting is a no-op)."""
    if thread_id is None:
        return None
    return _WAITERS.get(thread_id)
