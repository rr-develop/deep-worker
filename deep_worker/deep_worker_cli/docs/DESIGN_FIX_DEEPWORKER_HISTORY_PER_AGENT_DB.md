# Design: Fix `deep-worker --history` (and related offline commands) for Per-Agent DBs

**Status:** Implemented (unit + integration tests green)
**Author:** (proposed by assistant, approved)
**Date:** 2026-04-19

## Implementation Notes

- Helpers added to `sessions.py`:
  - `_agent_name_from_thread_id(thread_id)` — derive agent from thread_id.
  - `iter_all_db_paths()` — enumerate all `sessions_*.db` plus legacy
    global (respects `get_db_path(None)` so monkey-patched tests pick up
    the overridden directory).
  - `_resolve_db_path_for_thread(thread_id)` — per-agent, then legacy
    fallback, or `None`.
- Updated functions:
  - `get_thread_history` — uses `_resolve_db_path_for_thread` when
    `db_path` is not passed.  Also treats a missing `checkpoints` table
    as "not found" instead of letting the raw SQLite error leak.
  - `list_threads`, `list_all_threads` — iterate per-agent files +
    legacy, dedupe thread_ids, re-apply sort/limit after merge.
  - `clean_agent_history` / `clean_agent_history_db_only` — delete the
    per-agent DB file outright and additionally scrub stale rows from
    the legacy global.
  - `remove_thread` — routes to the correct DB via
    `_resolve_db_path_for_thread`.
  - `remove_threads_before` — iterates all per-agent files (or the
    explicitly passed `db_path`), aggregates counts.

## Tests

- Unit (all pass): `tests/unit/test_per_agent_db_resolution.py`
  covers `get_thread_history`, `list_threads`, `clean_agent_history`,
  `clean_agent_history_db_only`, `remove_threads_before` with real
  temp-dir `<agent-data-dir>` (no mocks of session/sqlite layer).
- Integration (all pass):
  `test_deep_worker_offline_per_agent_db.py`
  drives the real `deep-worker` CLI binary against a temp `HOME` with
  real per-agent DB files. Exact reproducer of the bug is covered
  (`test_dw_history_still_works_when_global_db_is_empty`).
- Rewrote `tests/integration_ai/test_thread_management_cli.py` to use
  the isolated-HOME pattern so it does not depend on the dev machine's
  <agent-data-dir> state.
- Full related test suites pass: **146 tests green**.

---

## 1. Problem

```
$ deep-worker --history lc4_test
Error: Error retrieving thread history: no such table: checkpoints
```

The agent `lc4_test` has **6660 checkpoints** stored in
`<agent-data-dir>/sessions_lc4_test.db` (3.8 GB). But `deep-worker --history`
reads `<agent-data-dir>/sessions.db` (the **legacy global** file), which on this
machine is empty (0 bytes, no tables). Hence the error.

This is not a data-loss bug — the data is intact. It's a **read-path bug**:
the offline CLI commands were never updated when the storage layout switched
from one global DB to one DB per agent.

## 2. Root Cause

The codebase already supports per-agent DBs:

- `get_db_path(agent_name)` → `<agent-data-dir>/sessions_{agent_name}.db`
- `get_db_path(None)` → `<agent-data-dir>/sessions.db` (legacy)
- `get_checkpointer(agent_name)` writes to the per-agent file and runs a
  one-time migration from the global file
  (`_migrate_from_global_db`, `sessions.py:401`)

However, four **offline** helper functions in
`deep_worker_cli/sessions.py` still call `get_db_path()` **without**
`agent_name`, so they only ever look at the legacy global file:

| Function | Line | Used by | Effect of bug |
|---|---|---|---|
| `get_thread_history` | `sessions.py:877` | `deep-worker --history <agent>` | Error `no such table: checkpoints` (reported) |
| `list_threads` | `sessions.py:115` | `deep-worker --list` (offline path) | Running agents may be visible via RPyC but offline threads are invisible |
| `clean_agent_history` / `clean_agent_history_db_only` | `sessions.py:567,635` | `deep-worker --clean`, `deep-worker -c` | Pretends to clean, but per-agent DB is left intact |
| `remove_threads_before` | `sessions.py:1014` | `deep-worker --remove-before` | Can't remove anything because target rows are in per-agent files |

The `dwcli --history` path is **different** — it goes through RPyC to the
running agent process and uses `get_history()` on the live checkpointer. That
path works (modulo the separate `int - NoneType` bug tracked elsewhere). So
the fix in this document is strictly for the **offline** commands of
`deep-worker`.

## 3. Why Tests Didn't Catch This

- Unit tests in `tests/unit/test_thread_management.py` all pass an explicit
  `db_path=str(tmp_path/"sessions.db")`, so `get_db_path()` is never exercised
  and the single-DB-vs-per-agent-DB distinction is invisible to them.
- Integration tests in `tests/integration_ai/test_dwcli_history.py` exercise
  `dwcli --history` (online, RPyC), **not** `deep-worker --history` (offline,
  direct DB read). Different code paths.
- There is **no** integration test that runs a real agent, lets it write to
  `sessions_{agent}.db`, stops it, and then runs `deep-worker --history
  <agent>` against the file on disk. That's the exact scenario that broke.

## 4. Thread-ID ↔ Agent-Name Mapping

Relevant invariants (observed in the live `sessions_lc4_test.db`):

```
thread_id | count
----------+------
lc4_test   | 6660
lc4_test:tm|  869
```

- The **main agent thread** uses `thread_id == agent_name`.
- The **TaskManager thread** uses `thread_id == f"{agent_name}:tm"`.

So for a given `thread_id`, the corresponding agent DB is:

```python
agent_name = thread_id.split(":")[0]
db_path    = get_db_path(agent_name)  # <agent-data-dir>/sessions_<agent>.db
```

This rule is already used inside `_migrate_from_global_db` (line 433):

```python
tids = [agent_name, f"{agent_name}:tm"]
```

So we're not inventing anything new — we're just applying the same mapping
on the read side.

## 5. Design

### 5.1 `get_thread_history(thread_id, db_path=None, format="text", agent_name=None)`

**New signature** (backward-compatible — `agent_name` is optional):

```python
async def get_thread_history(
    thread_id: str,
    db_path: str | None = None,
    format: str = "text",
    agent_name: str | None = None,  # NEW
) -> dict:
```

**DB resolution order** (applies only when `db_path is None`):

1. If `agent_name` is given → `db_path = get_db_path(agent_name)`.
2. Else derive from `thread_id`: `derived = thread_id.split(":")[0]`;
   `candidate = get_db_path(derived)`. If `candidate` exists → use it.
3. Else fall back to the legacy global `get_db_path(None)` (handles threads
   that never migrated — which in practice means DBs created before
   commit `_migrate_from_global_db` shipped).
4. If both candidate and global don't exist → return
   `{"success": False, "error": "No session database found for thread '{thread_id}'"}`.

**Behavior:** unchanged once the correct file is opened. The `checkpoints`
table check on line 882-887 still applies to the chosen file.

**Explicit `db_path` still wins** — important for unit tests that want
an isolated `tmp_path/"sessions.db"`.

### 5.2 `list_threads(agent_name=None, limit=20, include_message_count=False)`

**Change:** when `agent_name is None`, iterate over all per-agent files.

```python
if agent_name is not None:
    # Single file
    db_files = [get_db_path(agent_name)]
else:
    # All per-agent files + legacy global
    db_dir = Path.home() / ".deepagents"
    db_files = list(db_dir.glob("sessions_*.db"))
    legacy = db_dir / "sessions.db"
    if legacy.exists():
        db_files.append(legacy)
```

Then run the existing SELECT against **each** file, concatenate, sort by
`updated_at DESC`, apply `limit`.

**Edge case:** a file missing the `checkpoints` table is silently skipped
(the existing `_table_exists` check already does this).

### 5.3 `clean_agent_history(agent_name)` / `clean_agent_history_db_only(agent_name)`

**Two-step cleanup** (simpler and bulletproof):

1. If `<agent-data-dir>/sessions_{agent_name}.db` exists → **delete the file**.
   Returns `(checkpoints_deleted, writes_deleted)` by reading row counts
   beforehand. Deleting the file is atomic, avoids SQLite lock issues with
   a just-exited agent, and is exactly what the user expects from `--clean`.
2. Also run the current logic against the legacy global `sessions.db` so
   any leftover rows (from pre-migration runs) get cleaned too.

### 5.4 `remove_threads_before(date_str, db_path=None, dry_run=False)`

**Change:** when `db_path is None`, iterate over every
`sessions_*.db` file + the legacy global, and apply the current
`DELETE WHERE json_extract(metadata, '$.updated_at') < ?` query to each.

Aggregate counts from all files into the returned dict.

### 5.5 No change to CLI wrappers

`show_history_command`, `list_threads_command`, `remove_threads_before`
CLI handlers don't need to change — the fix is entirely inside
`sessions.py`. (If we later want to add a `--agent-name` flag to
`--history`, that's a separate enhancement; it's not required to fix the
reported bug.)

## 6. Backward Compatibility

- Signatures extended with optional parameters. Existing callers (including
  unit tests passing `db_path` explicitly) continue to work byte-identically.
- Behavior for users who only ever had a global `sessions.db` (legacy
  installs): **unchanged**, because step 3 of the resolution order falls
  back to the global file.
- Behavior for users with per-agent DBs: **now works** where before it
  failed.

## 7. Testing Plan (TDD — tests first, **real data, no mocks**)

### 7.1 Unit tests (fast, `tmp_path`)

Add to `tests/unit/test_thread_management.py`:

- `test_get_thread_history_uses_per_agent_db` — create a per-agent file at
  `tmp_path/"sessions_myagent.db"` with a checkpoint for thread `myagent`,
  monkey-patch `Path.home()` → `tmp_path`, call
  `get_thread_history("myagent")` with **no** `db_path`, assert
  success + messages.
- `test_get_thread_history_derives_agent_from_tm_thread` — same, but
  checkpoint under `thread_id="myagent:tm"`, call
  `get_thread_history("myagent:tm")`, assert the per-agent file is read.
- `test_get_thread_history_falls_back_to_global_db` — only
  `tmp_path/"sessions.db"` exists with a checkpoint; assert it's still
  found.
- `test_list_threads_aggregates_all_per_agent_dbs` — seed two per-agent
  files + global; assert `list_threads(None)` returns rows from all three.
- `test_list_threads_filters_by_agent_name` — same seed; assert
  `list_threads("agent_a")` returns only `agent_a`'s rows.
- `test_clean_agent_history_deletes_per_agent_db_file` — create
  `sessions_foo.db`; call `clean_agent_history("foo")`; assert file is
  gone.
- `test_remove_threads_before_removes_from_per_agent_dbs` — seed
  two per-agent files with old rows; assert both get cleaned.

### 7.2 Integration tests (**real agent, real CLI**)

Add `test_deep_worker_offline_history.py`:

1. **`test_deep_worker_history_reads_per_agent_db_after_agent_exit`**
   ```
   # Start agent in background, send a prompt, wait for reply, kill it.
   # Then run: deep-worker --history <agent>
   # Assert exit_code == 0 and output contains both USER and ASSISTANT lines.
   ```
2. **`test_dw_history_works_for_tm_thread`**
   ```
   # Same as above but for <agent>:tm thread_id.
   ```
3. **`test_deep_worker_list_shows_offline_per_agent_threads`**
   ```
   # Pre-populate sessions_test_offline.db with a checkpoint, no running agent.
   # Run: deep-worker --list
   # Assert test_offline appears.
   ```
4. **`test_deep_worker_clean_removes_per_agent_db_file`**
   ```
   # Create sessions_tmpclean.db with a checkpoint.
   # Run: deep-worker --clean tmpclean (or equivalent).
   # Assert sessions_tmpclean.db no longer exists.
   ```

These run against **actual** DB files on disk and actual CLI processes — no
mocks. They will catch this class of bug in the future.

### 7.3 Regression test for the reported failure

`test_dw_history_lc4_test_style` — the exact scenario: start an
agent whose history ends up in a per-agent file, exit, run
`deep-worker --history <agent>`, assert success. This is effectively a
generalized version of (1) above and is the key defence against
"works for one agent but not another" regressions.

## 8. Implementation Order (TDD)

1. Write all unit tests from §7.1 — they should **fail** against current
   code (red).
2. Implement changes in `sessions.py` (§5.1–§5.4) — unit tests go **green**.
3. Write integration tests from §7.2 — run, confirm green.
4. Run the full `deep-worker` / `dwcli` test suite to confirm no
   regressions.
5. Manual check: `deep-worker --history lc4_test` should now print the
   real history, not an error.
6. Update `docs/DESIGN_THREAD_MANAGEMENT_OPTIONS.md` and
   `docs/DESIGN_CHECKPOINTER_FIX.md` to note the per-agent-DB read path is
   now implemented end-to-end.
7. Commit with message along the lines of:
   ```
   fix(sessions): read per-agent DBs in offline --history/--list/--clean/--remove-before

   Offline commands still looked only at the legacy global sessions.db,
   silently failing (or silently succeeding with no data) once an agent
   started writing to its per-agent sessions_<name>.db. Reproducer:
     $ deep-worker --history lc4_test
     Error: no such table: checkpoints
   even though sessions_lc4_test.db had 6660 checkpoints.

   Resolution order is now: explicit db_path > per-agent file derived from
   agent_name (or thread_id prefix) > legacy global. list_threads and
   remove_threads_before iterate over all sessions_*.db files.
   clean_agent_history deletes the per-agent file outright.

   Adds unit tests for each branch and integration tests that drive the
   real deep-worker CLI against real DB files on disk.
   ```

## 9. Out of Scope

- Changing the online `get_history()` RPyC path (the `int - NoneType`
  bug). That's a separate fix.
- Verbosity levels / Full-Dialog UI work (tracked in
  `DESIGN_FULL_DIALOG_REVISED.md`).
- Adding a real migration that removes the legacy global `sessions.db`.
  (It's already effectively dead code after `_migrate_from_global_db`
  runs once; cleaning it up is a separate cleanup ticket.)
