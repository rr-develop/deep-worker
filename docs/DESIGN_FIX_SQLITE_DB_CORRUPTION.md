# Design: Fix SQLite Database Corruption in Checkpointer

## Problem

`_handle_get_last_response()` in `agent_process.py` logs `"Error getting last response: database disk image is malformed"` when the per-agent checkpointer's SQLite database becomes corrupted. This error occurred 7+ times for a production agent.

**Root cause:** `DeltaSqliteCheckpointer` opens the SQLite database with **zero PRAGMA settings** — no WAL mode, no synchronous mode. Default `journal_mode=DELETE` means:
1. A crash/abort during a write leaves the rollback journal in an inconsistent state
2. The database file header can get corrupted on next open
3. No resilience to concurrent readers during write (WAL mode allows concurrent reads)

## Design

### Change 1: Prevention — Add WAL mode to checkpointer setup

**File:** the CLI's `delta_checkpointer.py`, method `setup()` (~line 337)

Add PRAGMA statements **before** table creation:

```python
async def setup(self) -> None:
    if self._is_setup:
        return
    # --- NEW: robust PRAGMA settings ---
    await self.conn.execute("PRAGMA journal_mode=WAL")
    await self.conn.execute("PRAGMA synchronous=NORMAL")
    # ------------------------------------
    async with self.conn.executescript(_CREATE_TABLES_SQL):
        pass
    await self.conn.commit()
    self._is_setup = True
```

**Why:**
- `journal_mode=WAL`: Write-Ahead Logging — allows concurrent readers during writes; crash recovery is automatic on reopen.
- `synchronous=NORMAL`: With WAL, NORMAL is safe (WAL syncs checkpoint, not every page). Default FULL is 2-3× slower on writes.
- These PRAGMAs persist in the database file — once set, all future connections use WAL.

**Does NOT change:**
- Architecture (same checkpointer class, same interface)
- Contracts (same tables, same query API)
- Policies (no polling, no threads, no 777/666)

### Change 2: Better error handling — Catch DatabaseError specifically

**File:** the CLI's `process/agent_process.py`, method `_handle_get_last_response()` (~line 1515)

Add specific `except` for both `aiosqlite.DatabaseError` and `sqlite3.DatabaseError` **before** the generic `except Exception` (the exceptions are from different class hierarchies):

```python
import sqlite3
import aiosqlite

...

try:
    state = await self.agent.aget_state(config)
    ...
except (aiosqlite.DatabaseError, sqlite3.DatabaseError) as e:
    logger.error(
        "Database corruption detected in checkpointer DB for agent '%s': %s. "
        "Run 'deep-worker --repair %s' or delete the agent's local "
        "checkpointer database to recover.",
        self.name, e, self.name,
    )
    return {"success": False, "error": f"Database corrupted: {e}"}
except Exception as e:
    logger.error(f"Error getting last response: {e}")
    return {"success": False, "error": f"Failed to get last response: {str(e)}"}
```

Similarly add the same handler to `_handle_get_history()` (~line 1398) — it has the same `except Exception` without DatabaseError-specific handling.

**Why:**
- Clear error message with recovery instructions instead of generic error
- Helps the user/admin fix the problem without code inspection
- Still returns `{"success": False}` — no behavior change for callers

### Change 2b: Exception form — catch both aiosqlite and sqlite3

Catch `(aiosqlite.DatabaseError, sqlite3.DatabaseError)` because `aiosqlite` wraps `sqlite3` exceptions but in rare cases the underlying `sqlite3.DatabaseError` may not be wrapped. Catching both covers all paths.

### What we do NOT add

- **No auto-repair**: Automatically repairing a corrupted DB could lead to silent data loss. Use explicit CLI command.
- **No polling**: No background health checks. Only react on actual DB access.
- **No new threads**: Everything stays on the existing asyncio event loop.

## Tests (TDD)

### Unit tests — `test_delta_checkpointer.py`

Add 2 test cases to the existing test file:

```python
# Test: setup() sets journal_mode=WAL
#   Create a DeltaSqliteCheckpointer with a temporary file.
#   Call setup().
#   Run PRAGMA journal_mode — verify it returns "wal".

# Test: setup() sets synchronous=NORMAL
#   Run PRAGMA synchronous — verify it returns 1 (NORMAL).
```

### Unit tests — `test_agent_process_last_response.py`

Add 3 test cases to the existing test file:

```python
# Test: DatabaseError in _handle_get_last_response → error response
#   Mock agent.aget_state so it raises aiosqlite.DatabaseError.
#   Assert: the result is {"success": False, "error": ...}
#   Assert: the error text contains "Database corrupted"

# Test: sqlite3.DatabaseError is also caught (without the aiosqlite wrapper)
#   Mock agent.aget_state so it raises sqlite3.DatabaseError.
#   Assert: the result is {"success": False, "error": "Database corrupted: ..."}

# Test: other Exceptions → generic error response (regression)
#   Mock agent.aget_state so it raises RuntimeError.
#   Assert: the result is {"success": False, "error": "Failed to get last response: ..."}
```

### Unit tests — `test_agent_process_history.py` (or similar)

Add similar tests for `_handle_get_history()`.

### Unit tests — `test_delta_checkpointer_setup.py`

Add tests for delta_checkpointer's setup method with PRAGMA settings (use temp file, not in-memory — in-memory journal_mode returns "memory" instead of "wal"):

```python
# Test: setup() includes WAL mode
#   Create a checkpointer with a temporary file and call setup().
#   Then read PRAGMA journal_mode through a new connection.
#   Assert: journal_mode == "wal"

# Test: setup() includes synchronous=NORMAL
#   Create a checkpointer with a temporary file and call setup().
#   Assert: synchronous == 1 (NORMAL)
```

### Unit tests — `test_agent_process_history.py`

Add 2 test cases to the existing test file for `_handle_get_history`:

```python
# Test: DatabaseError in _handle_get_history → error response
#   Mock agent.aget_state so it raises sqlite3.DatabaseError.
#   Assert: the result is {"success": False, "error": ...}
#   Assert: the error text contains "Database corrupted"

# Test: other Exceptions → generic error response (regression)
#   Mock agent.aget_state so it raises RuntimeError.
#   Assert: the result is {"success": False, "error": "Failed to retrieve history:"}
```

## Risks

- WAL mode creates a `-wal` and `-shm` file alongside the DB. These are cleaned up on last connection close. If the agent is SIGKILL'd, stale files remain but WAL replay on next open handles them correctly.
- Backward compatible: old DBs opened with WAL PRAGMA will switch to WAL mode automatically.
- No regression: `_handle_get_last_response` still returns `{"success": False}` on error — callers unchanged.
