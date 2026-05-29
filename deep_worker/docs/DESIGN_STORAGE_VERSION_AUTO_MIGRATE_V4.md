# Design: Auto-Migrate Agent Storage to Version 4 on Restart

## Problem

1. **New agents start on v3**, but should start on **v4** — the version that includes JSONL summarization history support.
2. **Restart does not migrate** existing agents. An agent created at v2 stays on v2 forever;
   an agent created at v3 stays on v3. There is no auto-migration path.
3. The version in `session_metadata.json` is read back verbatim by `detect_storage_version()`,
   locking the agent to its original version.

## Goals

1. `CURRENT_STORAGE_VERSION = 4` — new agents start on v4.
2. On every restart, if the agent is on v2 or v3, **auto-migrate** to v4 before the session manager initializes.
3. v4 uses the exact same physical directory layout as v3 (`~/.deep_worker/<name>/` with
   `threads.db` and `data/`). The version bump only signals "JSONL summarization is active".
4. Old `.md` summarization files are **never** migrated to `.jsonl` — they lack the
   structured metadata needed for conversion.
5. New summarization events always write `.jsonl` (already implemented in v4 summarization history format).

## Version Definitions

| Version | Name | Layout | DB path | Notes |
|---|---|---|---|---|
| 1 | `STORAGE_VERSION_GLOBAL_DB` | ~/.deepagents/sessions.db | Single global DB | Deprecated |
| 2 | `STORAGE_VERSION_DEEPAGENTS_PER_AGENT` | ~/.deepagents/sessions_\<name\>.db | Per-agent legacy DB | Legacy |
| 3 | `STORAGE_VERSION_AGENT_ROOT` | ~/.deep_worker/\<name\>/threads.db | Per-agent in agent root | Previous current |
| **4** | `STORAGE_VERSION_AGENT_ROOT_WITH_JSONL` | ~/.deep_worker/\<name\>/threads.db | Same as v3 | **New current** |

**Rollback note:** v2→v4 migration copies data and sets the new version. There is no automated rollback path — manual restoration from backup (or re-creation) is required if the user needs to revert.

## Migration Rules

| From | Physical changes | Metadata change |
|---|---|---|
| v1 → v4 | Not supported (v1 is ancient global DB) | — |
| v2 → v4 | Copy `sessions_\<name\>.db` → `threads.db` (atomic: temp dir + rename); copy `~/.deepagents/\<name\>/` → `~/.deep_worker/\<name\>/data/` | `storage_version` → 4 |
| v3 → v4 | None (same layout) | `storage_version` → 4 (atomic write: tmp + rename) |

## Implementation

### Start sequence in `agent_process.py`

```
Step 1: detect_storage_version(name) → reads from disk
Step 2: if version < CURRENT_STORAGE_VERSION → call migrate_agent_storage_to_v4(name)
          - v2→v4: copies legacy DB + data dir, sets version=4
          - v3→v4: no-op physically, writes version=4 to metadata
          Race condition is mitigated by PID file: two processes
          cannot run the same agent simultaneously, so v2→v4 file
          copy is safe.
Step 3: init SessionManager(data_dir, launch_config, ...)
```

### 1. `storage_paths.py`

```python
STORAGE_VERSION_AGENT_ROOT_WITH_JSONL = 4
CURRENT_STORAGE_VERSION = STORAGE_VERSION_AGENT_ROOT_WITH_JSONL
```

`migrate_agent_storage_to_v4()` delegates to `migrate_agent_storage_to_v3()` for the
physical copy, then overwrites `storage_version` → 4:

```python
def migrate_agent_storage_to_v4(agent_name, *, home=None):
    # v3 path: ensure agent_root exists with correct metadata
    if detect_storage_version(agent_name, home=home) >= 3:
        # No physical copy needed, just update version
        meta = metadata_path(agent_name, home=home)
        ...
    else:
        # v2→v4: reuse v3 copy logic, then bump version
        result = migrate_agent_storage_to_v3(agent_name, home=home)
        ...
    # Atomic write: write to .<pid>.tmp, then rename
    metadata["storage_version"] = 4
    ...
```

### 2. `agent_process.py` — auto-migration in `start()`

Between step 1 (detect) and step 3 (session manager init):

```python
# Step 1: detect (already in __init__)
# Step 2: auto-migrate if needed
if self.storage_version < CURRENT_STORAGE_VERSION:
    from deep_worker_cli.storage_paths import migrate_agent_storage_to_v4
    migrate_agent_storage_to_v4(self.name)
    self.storage_version = CURRENT_STORAGE_VERSION
# Step 3: init SessionManager
```

### 3. `migrate_agent_storage_to_v3` → kept as-is (used by v4 internally)

The existing `migrate_agent_storage_to_v3()` function stays — it handles the
physical DB+data copy from legacy layout. `migrate_agent_storage_to_v4()` calls
it and only differs in the metadata version it writes.

### 4. CLI `deep-worker migrate` → updated to v4

`agent_command.py` `migrate_storage_command()` calls `migrate_agent_storage_to_v4()`
instead of `migrate_agent_storage_to_v3()`.

### 5. Tests

Unit tests (`test_storage_paths.py`):
- `test_new_agent_starts_on_v4`: fresh agent → `detect_storage_version` returns 4.
- `test_v3_agent_auto_migrate_to_v4`: agent with v3 metadata → `migrate_agent_storage_to_v4` bumps to 4.
- `test_v2_agent_auto_migrate_to_v4`: agent with legacy DB → copies to new location + sets version 4.
- `test_v2_to_v4_preserves_data_dir`: verify `~/.deepagents/\<name\>/` content appears in `data/`.

Integration test (`test_storage_version_migration.py`, no mocks):
- Create a test agent via `deep-worker --restart test_agent_v4`. Stop it.
- Verify `session_metadata.json` has `storage_version: 4`.
- Verify `summaries/` directory exists.
- Simulate v3 metadata: edit `session_metadata.json` → set version=3, restart agent.
- Verify after restart `session_metadata.json` has `storage_version: 4` again.

## Non-goals

1. Do NOT convert old `.md` files to `.jsonl`.
2. Do NOT change the physical layout of the agent root directory.
3. Do NOT add polling, background threads, or file locks (PID file prevents concurrent restarts).
4. Do NOT support migration from v1 (global DB).
