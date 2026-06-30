# DESIGN: `deep-worker --list` slow — read pre-aggregated `thread_metadata`

Status: proposed
Branch: `fixes`
Issue: `deep-worker --list` runs ~1m50s on a real install.

## Problem (measured)

`deep-worker --list` → `agent_command.list_threads_command` →
`sessions.list_all_threads()` (`deep_worker_cli/sessions.py:260`).

For **every** `sessions_*.db` file it runs:

```sql
SELECT thread_id,
       json_extract(metadata, '$.agent_name'),
       MAX(json_extract(metadata, '$.updated_at'))
FROM checkpoints
GROUP BY thread_id;
```

Measured on the live box (`~/.deepagents/`):

- 104 `sessions_*.db` files, **17 GB total** (largest 3.3 / 2.4 / 1.7 GB).
- `checkpoints` holds one row per checkpoint per turn; each row carries a
  `checkpoint` BLOB averaging **~65–126 KB**. Example
  `sessions_<agent>.db` = 3.3 GB, **13 184 rows, only 2 threads**.
- `GROUP BY thread_id` + `MAX(json_extract(metadata…))` forces a full read
  of every row. `EXPLAIN QUERY PLAN` →
  `SCAN checkpoints USING INDEX sqlite_autoindex_checkpoints_1`: it walks
  the PK b-tree whose leaf pages are ~99% the unused `checkpoint` BLOB.
- Net: to answer "2 threads" SQLite streams 3.3 GB through the page cache.
  Across 104 files: **~35M block reads, 31% CPU, 1m50s** (mostly I/O-wait).

This is not a logic regression — it is a query that reads a BLOB-heavy table
to obtain two light columns, exposed by data growth.

## Key fact: the answer already exists pre-aggregated

Every delta-checkpointer DB already maintains a small table, updated on each
`aput` (`delta_checkpointer.py:163,688`):

```sql
CREATE TABLE thread_metadata (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    agent_name TEXT,
    updated_at TEXT,
    message_count INTEGER DEFAULT 0,
    checkpoint_count INTEGER DEFAULT 0,
    PRIMARY KEY (thread_id, checkpoint_ns)
);
```

One row per `(thread_id, checkpoint_ns)` — exactly the fields `--list` prints,
including `message_count` (so `--with-counts` no longer deserializes BLOBs).

Reading it is instant (measured 0.00s vs full scan on the 3.3 GB file).

### Consistency audit (live, read-only)

Across all 104 DBs:

- **0 DBs** have a `checkpoints` thread missing from `thread_metadata`
  (`LEFT JOIN … WHERE m.thread_id IS NULL` → 0 everywhere the table exists).
  So where the table exists it is **complete** — no thread is lost.
- DBs **without** the `thread_metadata` table at all: a set of old/legacy and
  throwaway test DBs (`sessions_<agent_a/b/c>.db`, `sessions_<web>.db`,
  `sessions_<test>.db`, `sessions_<unit>.db`, several `sessions_test_*`),
  most of which are empty. These are the only gap.

## Decision

**Read `thread_metadata` as the sole hot path. No slow fallback.**

Per the user's explicit instruction: a fallback to the `checkpoints` full
scan is unacceptable. If a DB lacks `thread_metadata`, we **backfill** the
table once (cheap, one-time), then read it — never scan `checkpoints` on the
`--list` path.

### Aggregation across `checkpoint_ns`

`thread_metadata` keys on `(thread_id, checkpoint_ns)`. The old query grouped
by `thread_id` only. To preserve identical output we aggregate in SQL:

```sql
-- no filter
SELECT thread_id,
       MAX(agent_name)    AS agent_name,
       MAX(updated_at)    AS updated_at,
       SUM(message_count) AS message_count
FROM thread_metadata
GROUP BY thread_id;
```

With an `agent_name` filter (RC-2 — fixed SQL form, applied in WHERE before
grouping, mirroring the old `WHERE json_extract(...agent_name) = ?`):

```sql
SELECT thread_id,
       MAX(agent_name)    AS agent_name,
       MAX(updated_at)    AS updated_at,
       SUM(message_count) AS message_count
FROM thread_metadata
WHERE agent_name = ?
GROUP BY thread_id;
```

`agent_name` is constant per thread (derived from thread_id), so `MAX` is a
safe selector — equivalent to the old implicit collapse of `GROUP BY
thread_id` over a per-thread-constant `agent_name`. `updated_at` `MAX` matches
the old `MAX(...updated_at)` over ISO strings.

`message_count` semantics (RC-1 clarification): the column holds the **total
message count per `(thread_id, checkpoint_ns)`** as written by the last
`aput` (it is a total, not an increment — see `delta_checkpointer.py:689`,
`msg_count = len(...messages)`). `SUM` across namespaces therefore yields
"total messages across all namespaces of the thread". Today there is exactly
one ns per thread, so `SUM` is identity; summing is the correct
generalization for multi-ns threads.

## Scope: BOTH list functions share the same slow query (RC-1)

The identical BLOB-heavy query exists in **two** places in `sessions.py`:

- `list_all_threads()` (`sessions.py:260`) — `deep-worker --list`.
- `list_threads()` (`sessions.py:159`) — the paginated path used by
  `deepagents threads list` (has `limit`, sorts by `updated_at` desc).

Both must be fixed via a **single shared helper** so the fix is complete:

```python
async def _list_threads_from_metadata(
    conn, agent_name: str | None
) -> list[dict]:
    """Return [{thread_id, agent_name, updated_at, message_count}] for one
    DB, reading the pre-aggregated thread_metadata table (no checkpoints
    scan). Caller must have run _ensure_thread_metadata(conn) first."""
```

`list_all_threads()` and `list_threads()` both call
`_ensure_thread_metadata(conn)` then `_list_threads_from_metadata(conn, …)`,
and keep their own dedup / sort / limit / `include_message_count` post-logic
(unchanged dict shape).

## Changes

### 1. `sessions.py` — backfill helper

New `_ensure_thread_metadata(conn) -> None`:

1. If `thread_metadata` table is absent → create it using the **DDL constant
   imported from `delta_checkpointer`** (RC-3 — single source of truth, same
   as `scripts/migrate_to_delta.py`; never hand-duplicate the DDL) — handles
   DBs that predate the table.
2. Backfill any `checkpoints` thread missing from `thread_metadata`:

   ```sql
   INSERT OR IGNORE INTO thread_metadata
       (thread_id, checkpoint_ns, agent_name, updated_at,
        message_count, checkpoint_count)
   SELECT thread_id,
          checkpoint_ns,
          json_extract(metadata, '$.agent_name'),
          MAX(json_extract(metadata, '$.updated_at')),
          0, COUNT(*)
   FROM checkpoints
   GROUP BY thread_id, checkpoint_ns;
   ```

   This runs **only** when a gap exists (guarded by a cheap `LEFT JOIN …
   LIMIT 1` existence probe), so the common case — table present & complete —
   pays nothing. `message_count` is left 0 for backfilled rows (no BLOB
   deserialization during `--list`); the live checkpointer corrects it on the
   next `aput`. `--with-counts` for legacy backfilled threads is addressed
   below.

   `INSERT OR IGNORE` (not `OR REPLACE`) so we never clobber accurate
   live-maintained rows. After backfill, issue an explicit `COMMIT`. Wrap the
   write in best-effort error handling: on `aiosqlite.OperationalError` (DB
   locked / read-only fs) skip the backfill for that file and fall through to
   reading whatever `thread_metadata` already holds — same best-effort posture
   as the existing corrupt-file `except OperationalError: continue` (RC-3).

Permissions: backfill writes to the existing DB file in place; we do not
create new files, so no new mode bits are introduced. The connection is
opened read-write only when a gap is detected; otherwise read-only.

### 2. `sessions.list_all_threads()` — read `thread_metadata`

Replace the per-file `checkpoints` query with:

1. `await _ensure_thread_metadata(conn)` (no-op when complete).
2. The aggregation query above (optionally filtered by `agent_name`).
3. `include_message_count`: take `message_count` straight from the column.
   For rows where it is 0 **and** the thread actually has checkpoints (legacy
   backfilled), fall back to `_count_messages_from_checkpoint` **for that
   single thread only** — bounded, not a table scan, and only for the
   `--with-counts` flag which is already documented "slow". The default
   `--list` path never touches a BLOB.

No change to `format_threads_text` / `format_threads_json` — same dict shape.

### 3. One-time live backfill of the gap DBs

The code change makes `--list` self-healing (first run backfills, subsequent
runs are instant). To guarantee the **very first** post-deploy `--list` is
also fast, run the backfill once over the live `~/.deepagents/sessions_*.db`
gap set as part of rollout (the integration test exercises the same helper).

## Out of scope / non-goals

- No schema/contract change to `checkpoints` or the checkpointer write path.
- No new index on `checkpoints` (rejected: build cost + disk + still BLOB
  `metadata`; `thread_metadata` is strictly better).
- No new threads, no polling.

## Tests (TDD)

Unit (`tests/unit_tests/` or `tests/unit/cli/`):

1. `thread_metadata` present & complete → **both** `list_all_threads` AND
   `list_threads` return correct rows and **issue no SELECT against
   `checkpoints`** (assert via a connection wrapper / query spy on both
   functions, RC-4) → proves the BLOB scan is gone on every list path.
2. `thread_metadata` table absent → helper creates it, backfills from
   `checkpoints`, result identical to legacy output.
3. Partial `thread_metadata` (some threads missing) → backfill fills only the
   gap, live rows untouched (`OR IGNORE` verified).
4. `--with-counts`: counts correct for both live rows (from column) and
   legacy backfilled rows (single-thread BLOB count fallback).
5. Multi-`checkpoint_ns` thread aggregates to one row (SUM counts, MAX ts).

Integration (no mocks): build a real multi-DB `~/.deepagents`-like dir with
the delta checkpointer, plus one legacy DB with no `thread_metadata`, run the
real `deep-worker --list` subprocess, assert output correctness and that a
second run does no backfill writes.

## Rollout

1. Implement + unit tests green.
2. Integration test green.
3. One-time backfill over live gap DBs.
4. Verify `time deep-worker --list` is sub-second on the live box.
