# Design: Agent Storage v3

## Goal

Keep all data that belongs to one deep-worker agent under one root:

```text
~/.deep_worker/<agent_name>/
```

The agent root must contain metadata, thread history, summaries, provider state, and optional deepagents-compatible user data. TaskManager data must not create a sibling `<agent_name>:tm` root; if TaskManager needs its own files, they live under:

```text
~/.deep_worker/<agent_name>/tm/
```

All created agent storage directories must be private to the current OS user.

## Storage Versions

Persist a numeric `storage_version` in `session_metadata.json`.

| Version | Meaning |
|---|---|
| `1` | Old global thread DB in the legacy user tree (removed 2026-08-31; see POLICIES §16) |
| `2` | Metadata under `~/.deep_worker/<agent>/`, threads and user data under the legacy tree (removed 2026-08-31; see POLICIES §16) |
| `3` | New layout: all per-agent data under `~/.deep_worker/<agent>/` |

Compatibility rules:

- Existing metadata without `storage_version` is treated as version `2`.
- Existing agents continue using their existing version until explicitly migrated.
- A new agent with no existing metadata and no existing v2 thread DB starts as version `3`.
- Historically, if no metadata existed but a v2 DB was found in the legacy tree, the agent was treated as version `2` to avoid silently abandoning history. That tree is gone (POLICIES §16), so this fallback no longer applies.
- The legacy global DB path remains readable for version `1` and fallback offline commands.

## v3 Layout

```text
~/.deep_worker/<agent_name>/
  session_metadata.json
  threads.db
  summaries/
  codex_session_state.json
  claude_session_state.json
  data/
    AGENTS.md
    skills/
    agents/
  tm/
```

`threads.db` stores all LangGraph checkpoints for both the main agent thread and TaskManager thread. Thread IDs keep their existing values:

- main thread: `<agent_name>`
- TaskManager thread: `<agent_name>:tm`

The `:tm` suffix remains a thread-id convention only. It must not be used as a filesystem root.

## Path API

Add a small storage-path module, for example `deep_worker_cli/storage_paths.py`, with these responsibilities:

- validate/sanitize agent directory names using the existing agent-name rules;
- resolve `agent_root(agent_name) -> ~/.deep_worker/<agent_name>`;
- resolve `threads_db_path(agent_name, storage_version=None)`, returning:
  - v3: `~/.deep_worker/<agent_name>/threads.db`
  - v2/v1: legacy-tree paths, now unreachable (POLICIES §16)
- resolve `deepagents_data_dir(agent_name, storage_version=None)`, returning:
  - v3: `~/.deep_worker/<agent_name>/data`
  - v2: legacy-tree path, now unreachable (POLICIES §16)
- create private directories with mode `0700`;
- when creating files controlled by deep-worker, enforce mode `0600` after creation.

Keep `sessions.get_db_path(agent_name)` as a compatibility wrapper so existing call sites and external imports do not break. It should delegate to the new path resolver and default to version detection when an `agent_name` is present.

## Version Detection

Detection is synchronous and local filesystem based. It must not poll and must not introduce threads.

For an `agent_name`:

1. Read `~/.deep_worker/<agent_name>/session_metadata.json` if it exists.
2. If it contains integer `storage_version`, use that value.
3. If metadata exists without a version, treat as `2`.
4. Historically, a v2 DB in the legacy tree with no metadata meant version `2`; that tree is gone (POLICIES §16), so this step no longer fires.
5. Otherwise use `3`.

When `SessionManager` initializes metadata for a new v3 agent, it writes `storage_version: 3`. When it opens existing metadata without a version, it may persist `storage_version: 2` while preserving all existing fields.

The synchronous detection runs only during startup/CLI path resolution. Async request handling should use the already detected version/path instead of re-reading metadata. This avoids adding `asyncio.to_thread()` or other thread-pool work for version detection.

## Permissions

Every deep-worker-created agent directory must be owner-only:

- directories: `0700`
- SQLite DB files, metadata files, provider state JSON files: `0600`

Path helpers should call `mkdir(..., exist_ok=True)` and then `chmod(0o700)` for controlled directories. Metadata writes should chmod the final file after atomic replace. SQLite DB creation should chmod the DB path after the saver opens or during checkpointer setup.

Do not recursively chmod unrelated project config directories. The scope is user-level agent storage controlled by deep-worker.

## Migration Script

Add an explicit migration script/CLI command; do not auto-migrate existing agents.

Recommended command:

```bash
deep-worker --migrate-storage <agent_name> --to-version 3
```

Behavior:

1. Refuse to migrate if the agent is currently running.
2. Create `~/.deep_worker/<agent_name>/` and `data/` with `0700`.
3. Copy or move the v2 thread DB from the legacy tree to `~/.deep_worker/<agent_name>/threads.db`.
4. Copy useful files from the agent's legacy user-data directory to `~/.deep_worker/<agent_name>/data/` if they exist.

   Both steps applied only while the legacy tree still existed; it was removed 2026-08-31 (POLICIES §16), so a fresh install has nothing to migrate.
5. Update `session_metadata.json` with `storage_version: 3`.
6. Leave the old v2 files in place unless a later explicit cleanup flag is added.

## Implementation Plan

1. Add tests first for path resolution, version detection, permission modes, and no `<agent>:tm` root creation.
2. Add `storage_paths.py` and update `sessions.py` DB resolution to use it.
3. Update `SessionManager` to persist `storage_version` and private metadata file permissions.
4. Update config/user-data path helpers so v3 agents use `~/.deep_worker/<agent>/data/`.
5. Add integration coverage with a temporary HOME that starts a real local agent/checkpointer path without mocks and verifies `threads.db`, metadata version, permissions, and TaskManager thread storage.
6. Add the explicit migration command/script after the v3 runtime path is verified.

## Non-Goals

- No automatic migration of existing v2 agents.
- No polling.
- No new background threads.
- No change to thread-id semantics.
- No deletion of legacy data during normal startup.
