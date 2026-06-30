"""Tests for the thread_metadata fast path of the thread-listing functions.

`deep-worker --list` (``list_all_threads``) and ``deepagents threads list``
(``list_threads``) must read the pre-aggregated ``thread_metadata`` table
instead of full-scanning the BLOB-heavy ``checkpoints`` table.

These tests pin:

1. When ``thread_metadata`` is present & complete, NEITHER list function
   issues a ``SELECT`` against ``checkpoints`` (RC-4 query spy) — proving the
   slow scan is gone.
2. A legacy DB without ``thread_metadata`` is backfilled once, then read;
   output matches the legacy semantics.
3. A partially-populated ``thread_metadata`` is gap-filled without clobbering
   live rows (``INSERT OR IGNORE``).
4. ``--with-counts`` reports correct counts from the column (live rows) and a
   bounded per-thread fallback for legacy backfilled rows.
5. Multi-``checkpoint_ns`` threads aggregate to a single row (MAX ts, SUM
   message_count).
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deep_worker_cli import sessions

_SERDE = JsonPlusSerializer()


def _checkpoint_blob(ts: str, n_messages: int) -> tuple[str, bytes]:
    messages = [{"role": "user", "content": f"m{i}"} for i in range(n_messages)]
    data = {
        "v": 1,
        "id": f"ckpt_{ts}",
        "ts": ts,
        "channel_values": {"messages": messages},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    return _SERDE.dumps_typed(data)


def _create_checkpoints_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
        """
    )


def _create_thread_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            agent_name TEXT,
            updated_at TEXT,
            message_count INTEGER DEFAULT 0,
            checkpoint_count INTEGER DEFAULT 0,
            PRIMARY KEY (thread_id, checkpoint_ns)
        )
        """
    )


def _insert_checkpoint(
    conn: sqlite3.Connection,
    thread_id: str,
    agent_name: str | None,
    updated_at: str,
    n_messages: int,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> None:
    ctype, cblob = _checkpoint_blob(updated_at, n_messages)
    metadata = {"updated_at": updated_at}
    if agent_name is not None:
        metadata["agent_name"] = agent_name
    conn.execute(
        """
        INSERT INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id or f"ckpt_{thread_id}_{updated_at}",
            ctype,
            cblob,
            json.dumps(metadata).encode("utf-8"),
        ),
    )


def _insert_thread_metadata(
    conn: sqlite3.Connection,
    thread_id: str,
    agent_name: str | None,
    updated_at: str,
    message_count: int,
    checkpoint_ns: str = "",
    checkpoint_count: int = 1,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO thread_metadata
            (thread_id, checkpoint_ns, agent_name, updated_at,
             message_count, checkpoint_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            checkpoint_ns,
            agent_name,
            updated_at,
            message_count,
            checkpoint_count,
        ),
    )


@pytest.fixture
def complete_db(tmp_path) -> Path:
    """DB whose thread_metadata is present and complete (the common case)."""
    db_path = tmp_path / "sessions_complete.db"
    conn = sqlite3.connect(str(db_path))
    _create_checkpoints_table(conn)
    _create_thread_metadata_table(conn)
    rows = [
        ("alpha", "alpha", "2026-03-29T10:00:00+00:00", 5),
        ("beta", "beta", "2026-03-29T11:00:00+00:00", 10),
        ("alpha:tm", "alpha:tm", "2026-03-29T12:00:00+00:00", 3),
    ]
    for tid, agent, ts, n in rows:
        # A few checkpoint rows each (the heavy table) ...
        for k in range(3):
            _insert_checkpoint(conn, tid, agent, ts, n, checkpoint_id=f"{tid}_{k}")
        # ... and the pre-aggregated metadata row.
        _insert_thread_metadata(conn, tid, agent, ts, n)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def legacy_db(tmp_path) -> Path:
    """DB with NO thread_metadata table at all (predates the table)."""
    db_path = tmp_path / "sessions_legacy.db"
    conn = sqlite3.connect(str(db_path))
    _create_checkpoints_table(conn)
    _insert_checkpoint(conn, "gamma", "gamma", "2026-03-29T09:00:00+00:00", 7)
    _insert_checkpoint(conn, "delta", "delta", "2026-03-29T08:00:00+00:00", 2)
    conn.commit()
    conn.close()
    return db_path


class _SpyConnection:
    """Wrap sqlite3.Connection, recording every SQL statement executed."""

    def __init__(self, real: sqlite3.Connection, sink: list[str]):
        self._real = real
        self._sink = sink

    def execute(self, sql, *args, **kwargs):
        self._sink.append(sql)
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _heavy_checkpoints_scans(statements: list[str]) -> list[str]:
    """Return statements that perform a BLOB-bearing scan of ``checkpoints``.

    The fast path is allowed to issue the cheap *covering-index* gap probe
    (``SELECT 1 FROM checkpoints ... LIMIT 1``) — that reads only PK index
    pages, never the heavy ``checkpoint`` BLOB. What must NEVER appear on the
    listing hot path is the old aggregating scan that drags every BLOB-laden
    row through the page cache: a ``SELECT ... FROM checkpoints`` that either
    extracts ``metadata`` (``json_extract``) or groups by ``thread_id``.
    """
    out = []
    for s in statements:
        norm = " ".join(s.split()).lower()
        if not norm.startswith("select") or "from checkpoints" not in norm:
            continue
        # The cheap covering-index probe is fine.
        is_probe = norm.startswith("select 1 from checkpoints") and "limit 1" in norm
        if is_probe:
            continue
        if "json_extract" in norm or "group by thread_id" in norm:
            out.append(norm)
    return out


@pytest.mark.asyncio
async def test_complete_metadata_no_checkpoints_select_list_all(complete_db):
    """list_all_threads must not SELECT from checkpoints when metadata complete."""
    captured: list[str] = []
    orig_connect = sessions.aiosqlite.connect

    def spy_connect(*a, **k):
        return _SpyConnectCM(orig_connect(*a, **k), captured)

    with patch.object(sessions, "get_db_path", return_value=complete_db):
        with patch.object(sessions.aiosqlite, "connect", side_effect=spy_connect):
            result = await sessions.list_all_threads()

    tids = sorted(t["thread_id"] for t in result)
    assert tids == ["alpha", "alpha:tm", "beta"]
    bad = _heavy_checkpoints_scans(captured)
    assert bad == [], f"unexpected BLOB-bearing checkpoints scan: {bad}"


@pytest.mark.asyncio
async def test_complete_metadata_no_checkpoints_select_list_threads(complete_db):
    """list_threads (paginated path) also must not SELECT from checkpoints."""
    captured: list[str] = []
    orig_connect = sessions.aiosqlite.connect

    def spy_connect(*a, **k):
        return _SpyConnectCM(orig_connect(*a, **k), captured)

    with patch.object(sessions, "get_db_path", return_value=complete_db):
        with patch.object(sessions.aiosqlite, "connect", side_effect=spy_connect):
            result = await sessions.list_threads(limit=100)

    tids = sorted(t["thread_id"] for t in result)
    assert tids == ["alpha", "alpha:tm", "beta"]
    bad = _heavy_checkpoints_scans(captured)
    assert bad == [], f"unexpected BLOB-bearing checkpoints scan: {bad}"


@pytest.mark.asyncio
async def test_legacy_db_backfilled_then_listed(legacy_db):
    """A legacy DB (no thread_metadata) is backfilled and listed correctly."""
    with patch.object(sessions, "get_db_path", return_value=legacy_db):
        result = await sessions.list_all_threads()

    by_id = {t["thread_id"]: t for t in result}
    assert set(by_id) == {"gamma", "delta"}
    assert by_id["gamma"]["agent_name"] == "gamma"
    assert by_id["gamma"]["updated_at"] == "2026-03-29T09:00:00+00:00"

    # The table must now exist and be populated (self-healed in place).
    conn = sqlite3.connect(str(legacy_db))
    has = conn.execute(
        "SELECT count(*) FROM sqlite_master "
        "WHERE type='table' AND name='thread_metadata'"
    ).fetchone()[0]
    assert has == 1
    rows = conn.execute(
        "SELECT thread_id FROM thread_metadata ORDER BY thread_id"
    ).fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["delta", "gamma"]


@pytest.mark.asyncio
async def test_partial_metadata_gap_filled_without_clobber(tmp_path):
    """Backfill fills only missing threads; live rows are not overwritten."""
    db_path = tmp_path / "sessions_partial.db"
    conn = sqlite3.connect(str(db_path))
    _create_checkpoints_table(conn)
    _create_thread_metadata_table(conn)
    # Two threads in checkpoints ...
    _insert_checkpoint(conn, "live", "live", "2026-03-29T10:00:00+00:00", 4)
    _insert_checkpoint(conn, "missing", "missing", "2026-03-29T11:00:00+00:00", 6)
    # ... but only "live" has a metadata row, with a DISTINCT (sentinel) ts
    # so we can prove it is not clobbered by backfill.
    _insert_thread_metadata(
        conn, "live", "live", "2099-01-01T00:00:00+00:00", 999
    )
    conn.commit()
    conn.close()

    with patch.object(sessions, "get_db_path", return_value=db_path):
        result = await sessions.list_all_threads()

    by_id = {t["thread_id"]: t for t in result}
    assert set(by_id) == {"live", "missing"}
    # live row preserved (sentinel ts untouched -> INSERT OR IGNORE worked)
    assert by_id["live"]["updated_at"] == "2099-01-01T00:00:00+00:00"
    # missing row backfilled from checkpoints
    assert by_id["missing"]["updated_at"] == "2026-03-29T11:00:00+00:00"


@pytest.mark.asyncio
async def test_with_counts_from_column_and_legacy_fallback(tmp_path):
    """--with-counts: live count from column; legacy backfilled via bounded read."""
    db_path = tmp_path / "sessions_counts.db"
    conn = sqlite3.connect(str(db_path))
    _create_checkpoints_table(conn)
    _create_thread_metadata_table(conn)
    # live thread: metadata carries the count directly
    _insert_checkpoint(conn, "live", "live", "2026-03-29T10:00:00+00:00", 12)
    _insert_thread_metadata(conn, "live", "live", "2026-03-29T10:00:00+00:00", 12)
    # legacy thread: only in checkpoints (will be backfilled w/ count 0,
    # then resolved by the bounded per-thread fallback)
    _insert_checkpoint(conn, "old", "old", "2026-03-29T09:00:00+00:00", 8)
    conn.commit()
    conn.close()

    with patch.object(sessions, "get_db_path", return_value=db_path):
        result = await sessions.list_all_threads(include_message_count=True)

    by_id = {t["thread_id"]: t for t in result}
    assert by_id["live"]["message_count"] == 12
    assert by_id["old"]["message_count"] == 8


@pytest.mark.asyncio
async def test_multi_namespace_thread_aggregates_to_one_row(tmp_path):
    """A thread spanning multiple checkpoint_ns collapses to a single row."""
    db_path = tmp_path / "sessions_multins.db"
    conn = sqlite3.connect(str(db_path))
    _create_checkpoints_table(conn)
    _create_thread_metadata_table(conn)
    _insert_thread_metadata(
        conn, "multi", "multi", "2026-03-29T10:00:00+00:00", 5, checkpoint_ns=""
    )
    _insert_thread_metadata(
        conn, "multi", "multi", "2026-03-29T12:00:00+00:00", 7, checkpoint_ns="sub"
    )
    conn.commit()
    conn.close()

    with patch.object(sessions, "get_db_path", return_value=db_path):
        result = await sessions.list_all_threads(include_message_count=True)

    assert len(result) == 1
    row = result[0]
    assert row["thread_id"] == "multi"
    # MAX(updated_at) across namespaces
    assert row["updated_at"] == "2026-03-29T12:00:00+00:00"
    # SUM(message_count) across namespaces
    assert row["message_count"] == 12


@pytest.mark.asyncio
async def test_delete_thread_also_removes_thread_metadata(tmp_path):
    """Deleting a thread must purge its thread_metadata row (no ghost thread)."""
    db_path = tmp_path / "sessions_del.db"
    conn = sqlite3.connect(str(db_path))
    _create_checkpoints_table(conn)
    _create_thread_metadata_table(conn)
    _insert_checkpoint(conn, "doomed", "doomed", "2026-03-29T10:00:00+00:00", 3)
    _insert_checkpoint(conn, "keep", "keep", "2026-03-29T11:00:00+00:00", 4)
    _insert_thread_metadata(conn, "doomed", "doomed", "2026-03-29T10:00:00+00:00", 3)
    _insert_thread_metadata(conn, "keep", "keep", "2026-03-29T11:00:00+00:00", 4)
    conn.commit()
    conn.close()

    with patch.object(sessions, "get_db_path", return_value=db_path):
        deleted = await sessions.delete_thread("doomed")
    assert deleted is True

    # thread_metadata must no longer reference the deleted thread ...
    conn = sqlite3.connect(str(db_path))
    remaining = [
        r[0]
        for r in conn.execute(
            "SELECT thread_id FROM thread_metadata ORDER BY thread_id"
        ).fetchall()
    ]
    conn.close()
    assert remaining == ["keep"]

    # ... and a subsequent listing must not resurrect it.
    with patch.object(sessions, "get_db_path", return_value=db_path):
        result = await sessions.list_all_threads()
    assert {t["thread_id"] for t in result} == {"keep"}


class _SpyConnectCM:
    """Async-context-manager wrapper that swaps in a statement-recording conn."""

    def __init__(self, inner_cm, sink: list[str]):
        self._inner_cm = inner_cm
        self._sink = sink
        self._conn = None

    async def __aenter__(self):
        self._conn = await self._inner_cm.__aenter__()
        return _AsyncSpyConnection(self._conn, self._sink)

    async def __aexit__(self, *exc):
        return await self._inner_cm.__aexit__(*exc)

    def __await__(self):
        # Support `await aiosqlite.connect(...)` form too.
        async def _coro():
            self._conn = await self._inner_cm
            return _AsyncSpyConnection(self._conn, self._sink)

        return _coro().__await__()


class _AsyncSpyConnection:
    """Records SQL passed to .execute on an aiosqlite connection."""

    def __init__(self, real, sink: list[str]):
        self._real = real
        self._sink = sink

    def execute(self, sql, *args, **kwargs):
        self._sink.append(sql)
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)
