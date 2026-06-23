"""Per-agent cache of JSON-ready dicts built from LangChain messages.

See ``deep_worker/docs/DESIGN_CONTEXT_DELTA_CACHE.md`` §4.7. Background:

Every call to ``_handle_get_history`` used to rebuild a dict for every
message in the returned slice, every time — O(messages) allocation and
attribute access on each RPyC hit. Since a message is immutable once it
lands in history (its ``id`` never changes, its content never changes,
it is either kept or dropped by summarization — never mutated), we can
memoize the per-message dict keyed by id.

This module provides:

    * ``serialize_message(msg)`` — pure function, no caching; used as
      the miss path of the cache and as a primitive for tests.
    * ``MessageDictCache`` — thread-unaware in-process cache. The
      agent owns one instance for its lifetime.

Invalidation:

    * Never on "this one message changed" — that does not happen.
    * Only on summarization, when the owner calls ``prune_to(keep_ids)``
      with the set of ids that remain in the live history. Entries whose
      id is not in ``keep_ids`` are evicted.

Safety:

    * Messages without an ``id`` attribute (streaming chunks, partial
      constructs) are NOT cached — each call re-serializes. Caching
      them could pin an obsolete dict once the real id arrives.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from deepagents.middleware.message_timestamps import CREATED_AT_KEY, TOOL_CALL_CREATED_AT_KEY


def serialize_message(msg: Any) -> dict[str, Any]:
    """Convert a LangChain message into a JSON-ready dict.

    Shape matches the contract of ``_handle_get_history`` pre-cache —
    callers expect the same ``type`` / ``id`` / ``content`` /
    ``tool_calls`` / ``tool_call_id`` / ``name`` / ``status`` keys.

    The function is pure: same input → same output; no I/O; no global
    state. The cache wraps this.
    """
    msg_dict: dict[str, Any] = {
        "type": msg.__class__.__name__,
        "id": getattr(msg, "id", None),
    }
    additional_kwargs = getattr(msg, "additional_kwargs", {})
    if isinstance(additional_kwargs, dict):
        created_at = additional_kwargs.get(CREATED_AT_KEY)
        if isinstance(created_at, str):
            msg_dict[CREATED_AT_KEY] = created_at

    if isinstance(msg, HumanMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        msg_dict["content"] = content

    elif isinstance(msg, SystemMessage):
        # ⚠️ POLICY: SystemMessages MUST be serialized WITH their content.
        # They carry the agent's base system prompt AND runtime middleware
        # hints (e.g. the inter-agent reply hint from
        # InterAgentReplyMiddleware). Dropping them silently from history
        # produces a misleading "full" rendering — explicitly forbidden.
        # See the full-dialog rendering design notes
        # §"System messages are first-class in history rendering".
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        msg_dict["content"] = content

    elif isinstance(msg, AIMessage):
        content = msg.content
        if isinstance(content, str):
            msg_dict["content"] = content
        elif isinstance(content, list):
            # Concatenate text blocks (the only shape we currently emit
            # in streaming). Non-text blocks fall through silently — a
            # plain-text history view is what the web UI needs.
            text_content = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content += block.get("text", "")
                elif isinstance(block, str):
                    text_content += block
            msg_dict["content"] = text_content
        else:
            msg_dict["content"] = str(content)

        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            tool_call_created_at = {}
            if isinstance(additional_kwargs, dict):
                raw_map = additional_kwargs.get(TOOL_CALL_CREATED_AT_KEY)
                if isinstance(raw_map, dict):
                    tool_call_created_at = raw_map
            serialized_tool_calls = []
            for tc in tool_calls:
                call_id = tc.get("id")
                call_dict = {
                    "id": call_id,
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                }
                if isinstance(call_id, str):
                    call_created_at = tool_call_created_at.get(call_id)
                    if isinstance(call_created_at, str):
                        call_dict[CREATED_AT_KEY] = call_created_at
                serialized_tool_calls.append(call_dict)
            msg_dict["tool_calls"] = serialized_tool_calls

    elif isinstance(msg, ToolMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        msg_dict["content"] = content
        msg_dict["tool_call_id"] = getattr(msg, "tool_call_id", None)
        msg_dict["name"] = getattr(msg, "name", "unknown")
        msg_dict["status"] = getattr(msg, "status", "success")

    return msg_dict


class MessageDictCache:
    """In-memory cache of per-message serialised dicts, keyed by message id.

    Owned by an AgentProcess; lives for the agent's lifetime. Not
    thread-safe — the agent runs in a single asyncio event loop, so
    concurrent access is serialised by the loop.
    """

    __slots__ = ("_by_id", "_hits", "_misses")

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def serialize(self, msg: Any) -> dict[str, Any]:
        """Return the serialised dict for ``msg``, caching the result
        iff the message has an ``id``."""
        msg_id = getattr(msg, "id", None)
        if not isinstance(msg_id, str) or not msg_id:
            # Uncacheable — serialise fresh every time.
            return serialize_message(msg)
        cached = self._by_id.get(msg_id)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        built = serialize_message(msg)
        self._by_id[msg_id] = built
        return built

    def prune_to(self, keep_ids: set[str]) -> int:
        """Evict every entry whose id is not in ``keep_ids``.

        Intended for post-summarization cleanup: pass the set of ids
        from the live message list and the cache drops the rest.
        Returns the number of entries evicted.
        """
        to_drop = [k for k in self._by_id if k not in keep_ids]
        for k in to_drop:
            del self._by_id[k]
        return len(to_drop)

    def clear(self) -> None:
        self._by_id.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._by_id),
            "hits": self._hits,
            "misses": self._misses,
        }
