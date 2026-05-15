"""Tests for DeltaSqliteCheckpointer cache (DESIGN_CONTEXT_DELTA_CACHE.md, Phase 1).

Covers:
    - "nothing changed" → zero SQLite queries on repeated aget_tuple
    - aput populates the cache from the checkpoint it already has in hand
    - aput bumps version; listeners are notified
    - historical browse (specific checkpoint_id != latest) does NOT evict
      the cached latest
    - summarization replaces cache entry instead of patching
    - feature flag disables caching cleanly
    - agent restart = cold cache (one chain walk, then hits)

These tests intentionally count conn.execute calls by wrapping the SQLite
connection, so "zero SQLite queries" is an observable assertion, not a
guess.
"""
from __future__ import annotations

import uuid
from typing import Any

import aiosqlite
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer


def _make_checkpoint(
    messages: list[Any],
    updated_channels: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    cp_id = uuid.uuid4().hex[:16]
    return {
        "v": 1,
        "id": cp_id,
        "ts": "2026-04-22T12:00:00",
        "channel_values": {"messages": list(messages), "working_dir": "/tmp"},
        "channel_versions": {"messages": f"{len(messages):032}.0"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": updated_channels or ["messages"],
    }, cp_id


class _CountingConn:
    """Wraps aiosqlite.Connection and counts execute/executescript calls.

    Everything else is forwarded verbatim. Used by tests to assert that
    a cache hit does zero SQLite I/O.
    """

    def __init__(self, inner: aiosqlite.Connection) -> None:
        self._inner = inner
        self.execute_count = 0
        self.executescript_count = 0

    def reset(self) -> None:
        self.execute_count = 0
        self.executescript_count = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.execute_count += 1
        return self._inner.execute(*args, **kwargs)

    def executescript(self, *args: Any, **kwargs: Any) -> Any:
        self.executescript_count += 1
        return self._inner.executescript(*args, **kwargs)

    async def commit(self) -> None:
        await self._inner.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.fixture
async def checkpointer(tmp_path):  # type: ignore[no-untyped-def]
    db_path = str(tmp_path / "cache_test.db")
    async with aiosqlite.connect(db_path) as raw_conn:
        counting = _CountingConn(raw_conn)
        cp = DeltaSqliteCheckpointer(counting, snapshot_interval=0)  # type: ignore[arg-type]
        yield cp, counting


@pytest.fixture
async def uncached_checkpointer(tmp_path):  # type: ignore[no-untyped-def]
    """Feature flag OFF → old behaviour, full chain walk every time."""
    db_path = str(tmp_path / "nocache_test.db")
    async with aiosqlite.connect(db_path) as raw_conn:
        counting = _CountingConn(raw_conn)
        cp = DeltaSqliteCheckpointer(
            counting,  # type: ignore[arg-type]
            snapshot_interval=0,
            cache_enabled=False,
        )
        yield cp, counting


class TestCachePhase1:
    """Cache hit on read path after aput."""

    @pytest.mark.asyncio
    async def test_aput_populates_cache_then_aget_tuple_is_zero_sql(
        self, checkpointer
    ) -> None:
        cp, counting = checkpointer
        # Prime — this write builds everything in RAM already.
        msg = [HumanMessage(content="hello")]
        checkpoint, cp_id = _make_checkpoint(msg)
        config_in = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        cfg = await cp.aput(config_in, checkpoint, {}, {})  # type: ignore[arg-type]

        # Now read back — must be served from cache with ZERO execute().
        counting.reset()
        loaded = await cp.aget_tuple(cfg)  # type: ignore[arg-type]

        assert loaded is not None
        assert loaded.checkpoint["id"] == cp_id
        assert counting.execute_count == 0, (
            "aget_tuple after aput must be a cache hit (zero SQLite queries); "
            f"got {counting.execute_count}"
        )

    @pytest.mark.asyncio
    async def test_repeated_aget_tuple_is_zero_sql(self, checkpointer) -> None:
        cp, counting = checkpointer
        msg = [HumanMessage(content="hi")]
        checkpoint, _ = _make_checkpoint(msg)
        cfg = await cp.aput(
            {"configurable": {"thread_id": "t2", "checkpoint_ns": ""}},
            checkpoint, {}, {},
        )  # type: ignore[arg-type]

        # Warm the cache — first read may do work (it won't, because aput
        # populates it, but be permissive).
        await cp.aget_tuple(cfg)  # type: ignore[arg-type]

        counting.reset()
        for _ in range(10):
            await cp.aget_tuple(cfg)  # type: ignore[arg-type]

        assert counting.execute_count == 0, (
            f"10 repeat reads of unchanged state: expected 0 SQL, got "
            f"{counting.execute_count}"
        )

    @pytest.mark.asyncio
    async def test_latest_lookup_without_checkpoint_id_is_cache_hit(
        self, checkpointer
    ) -> None:
        """Config without checkpoint_id means "give me the latest" — must
        still hit the cache."""
        cp, counting = checkpointer
        msg = [HumanMessage(content="x")]
        checkpoint, _ = _make_checkpoint(msg)
        await cp.aput(
            {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}},
            checkpoint, {}, {},
        )  # type: ignore[arg-type]

        counting.reset()
        loaded = await cp.aget_tuple(
            {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}}
        )  # type: ignore[arg-type]

        assert loaded is not None
        assert counting.execute_count == 0

    @pytest.mark.asyncio
    async def test_aput_after_aput_does_not_walk_chain(self, checkpointer) -> None:
        """One delta on top of cached state: aput must read prev from
        cache, not rebuild via chain walk."""
        cp, counting = checkpointer
        thread = {"configurable": {"thread_id": "t4", "checkpoint_ns": ""}}

        msg1 = [HumanMessage(content="a")]
        cp1, cp1_id = _make_checkpoint(msg1)
        cfg1 = await cp.aput(thread, cp1, {}, {})  # type: ignore[arg-type]

        counting.reset()
        # Second write: aput needs prev state to compute delta. It must
        # come from the cache, not from a SELECT.
        msg2 = msg1 + [AIMessage(content="b")]
        cp2, _ = _make_checkpoint(msg2)
        await cp.aput(cfg1, cp2, {}, {})  # type: ignore[arg-type]

        # The only SQL aput does on the cache-hit path is the INSERT itself
        # (into checkpoints + thread_metadata = 2 executes, plus one commit).
        # It must NOT do a "SELECT type, checkpoint FROM checkpoints" chain walk.
        # To assert this robustly, check that no SELECT ran.
        # Counting wrapper sees every execute — we inspect through a sentinel.
        # Simpler proxy: cache hit path issues <= 2 executes (the two INSERTs).
        assert counting.execute_count <= 2, (
            f"aput with cached prev state: expected ≤2 SQL (2 INSERTs), got "
            f"{counting.execute_count}"
        )


class TestCacheInvalidation:
    """Write path bumps cache version and replaces content."""

    @pytest.mark.asyncio
    async def test_aput_bumps_version_and_notifies_listeners(
        self, checkpointer
    ) -> None:
        cp, _ = checkpointer
        seen: list[tuple[str, str, int]] = []

        def listener(thread_id: str, ns: str, version: int) -> None:
            seen.append((thread_id, ns, version))

        cp.add_cache_listener(listener)

        thread = {"configurable": {"thread_id": "tL", "checkpoint_ns": ""}}
        cp1, _ = _make_checkpoint([HumanMessage(content="1")])
        cfg1 = await cp.aput(thread, cp1, {}, {})  # type: ignore[arg-type]
        cp2, _ = _make_checkpoint([HumanMessage(content="1"), AIMessage(content="2")])
        await cp.aput(cfg1, cp2, {}, {})  # type: ignore[arg-type]

        assert len(seen) == 2
        assert seen[0][0] == "tL"
        assert seen[0][2] < seen[1][2], "version must increase monotonically"

    @pytest.mark.asyncio
    async def test_cache_size_stays_bounded_per_thread(self, checkpointer) -> None:
        """One entry per (thread_id, checkpoint_ns). Many writes → one entry."""
        cp, _ = checkpointer
        thread = {"configurable": {"thread_id": "tB", "checkpoint_ns": ""}}
        cfg = thread
        messages: list[Any] = []
        for i in range(20):
            messages = messages + [HumanMessage(content=f"m{i}")]
            checkpoint, _ = _make_checkpoint(messages)
            cfg = await cp.aput(cfg, checkpoint, {}, {})  # type: ignore[arg-type]

        stats = cp.get_cache_stats()
        assert stats["cache_size_entries"] == 1


class TestHistoricalBrowse:
    """Fetching a non-latest checkpoint_id must not evict the cached latest."""

    @pytest.mark.asyncio
    async def test_historical_fetch_preserves_latest_cache(
        self, checkpointer
    ) -> None:
        cp, counting = checkpointer
        thread = {"configurable": {"thread_id": "tH", "checkpoint_ns": ""}}
        cp1, cp1_id = _make_checkpoint([HumanMessage(content="one")])
        cfg1 = await cp.aput(thread, cp1, {}, {})  # type: ignore[arg-type]
        cp2, cp2_id = _make_checkpoint(
            [HumanMessage(content="one"), AIMessage(content="two")]
        )
        cfg2 = await cp.aput(cfg1, cp2, {}, {})  # type: ignore[arg-type]

        # Request an older checkpoint explicitly.
        historical_cfg = {
            "configurable": {
                "thread_id": "tH",
                "checkpoint_ns": "",
                "checkpoint_id": cp1_id,
            }
        }
        older = await cp.aget_tuple(historical_cfg)  # type: ignore[arg-type]
        assert older is not None
        assert older.checkpoint["id"] == cp1_id

        # Latest must still be a cache hit.
        counting.reset()
        latest = await cp.aget_tuple(cfg2)  # type: ignore[arg-type]
        assert latest is not None
        assert latest.checkpoint["id"] == cp2_id
        assert counting.execute_count == 0


class TestSummarization:
    """Summarization = full snapshot = cache replacement, not append-delta."""

    @pytest.mark.asyncio
    async def test_summarization_replaces_cache(self, checkpointer) -> None:
        cp, counting = checkpointer
        thread = {"configurable": {"thread_id": "tS", "checkpoint_ns": ""}}

        cp1, _ = _make_checkpoint([HumanMessage(content="long-1")])
        cfg1 = await cp.aput(thread, cp1, {}, {})  # type: ignore[arg-type]
        cp2, _ = _make_checkpoint(
            [HumanMessage(content="long-1"), AIMessage(content="long-2")]
        )
        cfg2 = await cp.aput(cfg1, cp2, {}, {})  # type: ignore[arg-type]

        # Summarization: first message replaced with a summary. Shrinks
        # and/or reshapes history. Checkpointer already detects this and
        # stores a full snapshot (see _is_summarization).
        summary_msgs = [AIMessage(content="SUMMARY")]
        cp3, cp3_id = _make_checkpoint(summary_msgs)
        cfg3 = await cp.aput(cfg2, cp3, {}, {})  # type: ignore[arg-type]

        counting.reset()
        loaded = await cp.aget_tuple(cfg3)  # type: ignore[arg-type]
        assert loaded is not None
        assert loaded.checkpoint["id"] == cp3_id
        assert len(loaded.checkpoint["channel_values"]["messages"]) == 1
        assert counting.execute_count == 0, (
            "post-summarization cache must serve the new state directly"
        )


class TestFeatureFlag:
    """cache_enabled=False restores old walk-every-time behaviour."""

    @pytest.mark.asyncio
    async def test_disabled_cache_still_walks_chain(
        self, uncached_checkpointer
    ) -> None:
        cp, counting = uncached_checkpointer
        thread = {"configurable": {"thread_id": "tF", "checkpoint_ns": ""}}
        cp1, _ = _make_checkpoint([HumanMessage(content="x")])
        cfg1 = await cp.aput(thread, cp1, {}, {})  # type: ignore[arg-type]

        counting.reset()
        loaded = await cp.aget_tuple(cfg1)  # type: ignore[arg-type]
        assert loaded is not None
        # At least one SELECT must have happened — cache is off.
        assert counting.execute_count >= 1


class TestColdStart:
    """Agent restart simulation: fresh checkpointer, DB has deltas, first
    read walks the chain once and caches the result; subsequent reads hit
    cache."""

    @pytest.mark.asyncio
    async def test_cold_start_walks_once_then_caches(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        db_path = str(tmp_path / "restart.db")

        # Phase 1: populate DB with a chain of deltas.
        async with aiosqlite.connect(db_path) as raw1:
            cp1 = DeltaSqliteCheckpointer(raw1, snapshot_interval=0)
            thread = {"configurable": {"thread_id": "tR", "checkpoint_ns": ""}}
            messages: list[Any] = []
            cfg = thread
            for i in range(5):
                messages = messages + [HumanMessage(content=f"m{i}")]
                ck, _ = _make_checkpoint(messages)
                cfg = await cp1.aput(cfg, ck, {}, {})  # type: ignore[arg-type]
            final_cfg = cfg

        # Phase 2: fresh checkpointer, cache is empty.
        async with aiosqlite.connect(db_path) as raw2:
            counting = _CountingConn(raw2)
            cp2 = DeltaSqliteCheckpointer(counting, snapshot_interval=0)  # type: ignore[arg-type]

            # First read: cold → must hit SQLite.
            counting.reset()
            loaded = await cp2.aget_tuple(final_cfg)  # type: ignore[arg-type]
            assert loaded is not None
            assert len(loaded.checkpoint["channel_values"]["messages"]) == 5
            cold_count = counting.execute_count
            assert cold_count > 0, "cold read must query SQLite"

            # Second read: must be a cache hit.
            counting.reset()
            loaded2 = await cp2.aget_tuple(final_cfg)  # type: ignore[arg-type]
            assert loaded2 is not None
            assert counting.execute_count == 0


class TestObservability:
    """get_cache_stats() exposes counters for debugging."""

    @pytest.mark.asyncio
    async def test_stats_report_hits_and_misses(self, checkpointer) -> None:
        cp, _ = checkpointer
        thread = {"configurable": {"thread_id": "tO", "checkpoint_ns": ""}}
        ck, _ = _make_checkpoint([HumanMessage(content="x")])
        cfg = await cp.aput(thread, ck, {}, {})  # type: ignore[arg-type]

        before = cp.get_cache_stats()
        await cp.aget_tuple(cfg)  # type: ignore[arg-type]
        after = cp.get_cache_stats()

        assert after["cache_hits_total"] == before["cache_hits_total"] + 1
        assert after["chain_walks_total"] == before["chain_walks_total"]
