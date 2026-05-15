"""Tests for the message-dict cache (DESIGN_CONTEXT_DELTA_CACHE.md, Phase 3).

The serialiser converts a LangChain message object into a JSON-ready dict.
Once a message's ``id`` is known, the dict is immutable — messages are
append-only in a conversation (summarization drops messages, but never
mutates the ones that survive). So we can memoize the per-message dict
and reuse it on every ``get_history`` call.

These tests exercise the pure serialiser + its cache without touching
SQLite, agents, or RPyC.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_worker_cli.process.message_dict_cache import (
    MessageDictCache,
    serialize_message,
)


class TestSerializerPure:
    def test_human_message_shape(self) -> None:
        msg = HumanMessage(content="hello", id="h1")
        out = serialize_message(msg)
        assert out == {"type": "HumanMessage", "id": "h1", "content": "hello"}

    def test_ai_message_string_content(self) -> None:
        msg = AIMessage(content="hi there", id="a1")
        out = serialize_message(msg)
        assert out["type"] == "AIMessage"
        assert out["id"] == "a1"
        assert out["content"] == "hi there"
        assert "tool_calls" not in out

    def test_ai_message_with_tool_calls(self) -> None:
        tc = {"id": "call_1", "name": "grep", "args": {"q": "x"}}
        msg = AIMessage(content="running grep", id="a2", tool_calls=[tc])
        out = serialize_message(msg)
        assert out["tool_calls"] == [
            {"id": "call_1", "name": "grep", "args": {"q": "x"}}
        ]

    def test_ai_message_block_list_content(self) -> None:
        msg = AIMessage(
            content=[{"type": "text", "text": "part A"}, {"type": "text", "text": " part B"}],
            id="a3",
        )
        out = serialize_message(msg)
        assert out["content"] == "part A part B"

    def test_tool_message(self) -> None:
        msg = ToolMessage(
            content="ok", tool_call_id="call_1", name="grep", id="t1",
        )
        out = serialize_message(msg)
        assert out["type"] == "ToolMessage"
        assert out["tool_call_id"] == "call_1"
        assert out["name"] == "grep"
        assert out["content"] == "ok"


class TestMessageDictCache:
    def test_first_call_populates_cache(self) -> None:
        cache = MessageDictCache()
        msg = HumanMessage(content="x", id="h-new")
        out1 = cache.serialize(msg)
        out2 = cache.serialize(msg)
        assert out1 == out2
        # Cache hit must return the SAME dict object — not a rebuild.
        assert out1 is out2

    def test_different_messages_cached_independently(self) -> None:
        cache = MessageDictCache()
        m1 = HumanMessage(content="a", id="h1")
        m2 = AIMessage(content="b", id="a1")
        d1 = cache.serialize(m1)
        d2 = cache.serialize(m2)
        assert d1["type"] == "HumanMessage"
        assert d2["type"] == "AIMessage"
        # Both cached.
        assert cache.serialize(m1) is d1
        assert cache.serialize(m2) is d2

    def test_message_without_id_is_not_cached(self) -> None:
        """Safety: an id-less message may be transient / partial chunk.
        Serialize each time without caching to avoid poisoning the cache
        with dicts that may change later."""
        cache = MessageDictCache()
        m = HumanMessage(content="x")  # no id argument
        d1 = cache.serialize(m)
        d2 = cache.serialize(m)
        # Content matches but the dicts are independent objects.
        assert d1 == d2
        assert d1 is not d2

    def test_invalidate_ids(self) -> None:
        """Summarization may drop message ids from history. The cache
        owner invalidates by passing a set of ids that remain."""
        cache = MessageDictCache()
        m1 = HumanMessage(content="a", id="h1")
        m2 = AIMessage(content="b", id="a1")
        d1 = cache.serialize(m1)
        d2 = cache.serialize(m2)

        # Keep only h1.
        cache.prune_to({"h1"})

        # h1 is still hit.
        assert cache.serialize(m1) is d1
        # a1 was evicted → new object.
        assert cache.serialize(m2) is not d2

    def test_stats(self) -> None:
        cache = MessageDictCache()
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        m = HumanMessage(content="x", id="h")
        cache.serialize(m)
        cache.serialize(m)
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["misses"] == 1
        assert stats["hits"] == 1
