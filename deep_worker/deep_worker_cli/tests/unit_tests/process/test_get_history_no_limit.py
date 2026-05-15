"""Tests for the no-limit contract on AgentProcess._handle_get_history.

Per design (DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md §6/§3.6 RPyC),
the agent-side history retrieval must support "no limit" so the
Full Dialog endpoint can return the entire conversation. Three
sentinels mean "everything":

- ``limit`` key absent
- ``limit=None``
- ``limit=0`` or ``limit<0``

Positive integer limits keep working (legacy contract for dwcli).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _make_agent_process_with_messages(messages: list[Any]):
    """Build a minimal AgentProcess stub whose agent.aget_state returns the
    given message list. No RPyC, no real LangGraph compile.

    NOTE: this factory intentionally bypasses ``AgentProcess.__init__`` via
    ``__new__`` — calling the real ``__init__`` would drag in RPyC, the
    task manager, the registry, etc. The downside is that every attribute
    the code-under-test reads has to be mirrored here BY HAND. If you
    extend ``AgentProcess.__init__`` with another attribute that
    ``_handle_get_history`` (or any method exercised here) touches, mirror
    it below — and pin the expectation with a dedicated contract test so
    the drift is caught immediately next time.

    Tracked attributes (production parity with ``AgentProcess.__init__``):

    * ``name`` — identifier used in error messages.
    * ``agent`` — LangGraph compiled graph, mocked to return a static
      message list.
    * ``_message_dict_cache`` — per-message serialisation memoiser
      (``DESIGN_CONTEXT_DELTA_CACHE.md §4.7``). Production wires this in
      ``AgentProcess.__init__``; the new delta-cache code path
      (commit ``8c2e28d``) made it mandatory. A single shared instance
      per stub preserves the incremental contract across multiple
      ``_handle_get_history`` calls on the same stub, which is exactly
      what ``TestIncrementalDictCache`` pins below.
    """
    from deep_worker_cli.process.agent_process import AgentProcess
    from deep_worker_cli.process.message_dict_cache import MessageDictCache

    proc = AgentProcess.__new__(AgentProcess)
    # only the bits _handle_get_history reads
    proc.name = "test-agent"

    fake_state = MagicMock()
    fake_state.values = {"messages": messages}

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=fake_state)
    proc.agent = fake_agent  # type: ignore[assignment]

    # Mirror the delta-cache attribute from production __init__.
    proc._message_dict_cache = MessageDictCache()

    return proc


def _msgs(n: int) -> list[Any]:
    """Build ``n`` alternating Human/AI messages, each with a STABLE id.

    A stable id is mandatory for the MessageDictCache contract — without
    an id the message is intentionally not cached (see
    ``message_dict_cache.serialize``). Tests that pin the incremental
    behaviour rely on these ids.
    """
    out: list[Any] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(HumanMessage(content=f"q{i}", id=f"h-{i}"))
        else:
            out.append(AIMessage(content=f"a{i}", id=f"a-{i}"))
    return out


class TestNoLimitContract:
    @pytest.mark.asyncio
    async def test_no_limit_key_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"command": "get_history"})
        assert resp["success"] is True
        assert resp["total"] == 20
        assert len(resp["history"]) == 20, (
            "absent limit MUST return every message, no implicit cap"
        )
        assert resp["has_more"] is False

    @pytest.mark.asyncio
    async def test_explicit_none_limit_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": None})
        assert resp["total"] == 20
        assert len(resp["history"]) == 20

    @pytest.mark.asyncio
    async def test_zero_limit_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": 0})
        assert len(resp["history"]) == 20

    @pytest.mark.asyncio
    async def test_negative_limit_returns_all(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": -1})
        assert len(resp["history"]) == 20

    @pytest.mark.asyncio
    async def test_positive_limit_still_works_legacy(self) -> None:
        proc = _make_agent_process_with_messages(_msgs(20))
        resp = await proc._handle_get_history({"limit": 5})
        # legacy: last N
        assert len(resp["history"]) == 5
        # last message is q18 or a19 — we just check tail
        last_content = resp["history"][-1]["content"]
        assert last_content == "a19", f"expected last message 'a19', got {last_content!r}"
        assert resp["has_more"] is True


# ============================================================================
# Incremental MessageDictCache contract.
#
# The delta-based context cache (DESIGN_CONTEXT_DELTA_CACHE.md §4.7) is a
# load-bearing optimisation: every _handle_get_history call MUST NOT
# re-serialise messages it has already seen, and adding a NEW message
# MUST only add one miss, not rebuild the whole cache.
#
# These tests are the executable spec for that contract. If any future
# refactor breaks incrementality, these go RED immediately.
# ============================================================================


class TestIncrementalDictCache:
    @pytest.mark.asyncio
    async def test_second_call_on_unchanged_history_is_all_hits(self) -> None:
        """Twice in a row on the same history: second call MUST be all hits.

        Confirms memoisation by id — the cache never re-serialises an
        already-known message.
        """
        msgs = _msgs(10)
        proc = _make_agent_process_with_messages(msgs)

        r1 = await proc._handle_get_history({"limit": None})
        assert r1["success"] is True
        stats1 = proc._message_dict_cache.stats()
        assert stats1["misses"] == 10, (
            f"first call must populate cache with 10 fresh entries, "
            f"got stats={stats1!r}"
        )
        # hits on first call only come from messages whose id collides
        # with a cached one (none here) — so hits == 0.
        assert stats1["hits"] == 0
        assert stats1["size"] == 10

        r2 = await proc._handle_get_history({"limit": None})
        assert r2["success"] is True
        stats2 = proc._message_dict_cache.stats()
        # no new misses — every message was already cached.
        assert stats2["misses"] == 10, (
            f"second call must NOT re-serialise; misses must stay at 10, "
            f"got stats={stats2!r} (incremental-cache contract violated)"
        )
        assert stats2["hits"] == 10, (
            f"second call must register 10 hits (one per message); "
            f"got stats={stats2!r}"
        )

    @pytest.mark.asyncio
    async def test_appending_a_new_message_serialises_only_that_message(
        self,
    ) -> None:
        """Grow history by 1 → cache misses grow by EXACTLY 1.

        Rebuilding the whole cache would cause N+1 misses instead of 1.
        This is the core incrementality guarantee.
        """
        msgs = _msgs(5)
        proc = _make_agent_process_with_messages(msgs)

        await proc._handle_get_history({"limit": None})
        misses_before = proc._message_dict_cache.stats()["misses"]
        assert misses_before == 5

        # Append ONE new message with a fresh id (simulates one agent step).
        msgs.append(AIMessage(content="a5", id="a-5"))
        # state.values["messages"] is the same list object — it already
        # reflects the append (factory shares the reference).

        await proc._handle_get_history({"limit": None})
        misses_after = proc._message_dict_cache.stats()["misses"]
        assert misses_after == misses_before + 1, (
            f"appending one message must produce exactly one new miss; "
            f"misses_before={misses_before}, misses_after={misses_after} "
            f"(incremental-cache contract violated)"
        )

    @pytest.mark.asyncio
    async def test_prune_to_is_the_only_eviction_path(self) -> None:
        """prune_to evicts exactly the ids not in keep_ids; nothing else does.

        Mimics the post-summarization invalidation path described in
        DESIGN_CONTEXT_DELTA_CACHE.md §4.7.
        """
        msgs = _msgs(6)  # ids: h-0, a-1, h-2, a-3, h-4, a-5
        proc = _make_agent_process_with_messages(msgs)

        await proc._handle_get_history({"limit": None})
        assert proc._message_dict_cache.stats()["size"] == 6

        # Simulate summarization: only the last two messages survive.
        keep = {"h-4", "a-5"}
        evicted = proc._message_dict_cache.prune_to(keep)
        assert evicted == 4
        assert proc._message_dict_cache.stats()["size"] == 2

        # A further call on a history restricted to the surviving ids
        # should register pure hits — no re-serialisation.
        survivors = [m for m in msgs if getattr(m, "id", None) in keep]
        proc2 = _make_agent_process_with_messages(survivors)
        # seed proc2's cache with the state prune_to left us in
        proc2._message_dict_cache = proc._message_dict_cache
        hits_before = proc2._message_dict_cache.stats()["hits"]

        await proc2._handle_get_history({"limit": None})
        hits_after = proc2._message_dict_cache.stats()["hits"]
        assert hits_after - hits_before == 2, (
            f"post-prune call on survivors must be all hits; delta="
            f"{hits_after - hits_before} (expected 2)"
        )
