# DESIGN: agent databases under `~/.deep_worker/agents/`

**Date:** 2026-08-18 · **Status:** proposed — awaiting re-review (round 2)

> **Changelog**
> - Round 2: incorporated review feedback — see sections 3.4 (migration chain), 4.2 (full call-site table), 4.5 (read fallback helper), 5.1 (lazy-migration race safety), 5.2 (regex fix + keep-list early-skip), 6.5 (monkeypatch approach), 7 (non-goals), 8 (rollback precedence), 9 (resolved open questions).

## 1. Problem

`~/.deep_worker/` currently contains a mix of things:

| Category | Examples |
|---|---|
| **Agent databases** (per-agent trees: `threads.db`, `token_stats.jsonl`, `data/`, `inbox/`, `summaries/`, `session_metadata.json`) | `agent_a/`, `test_web/`, `rg_cross_file_extractor/`, `~250` others |
| **Product infrastructure** (NOT agents, never touched) | `providers.json`, `provider_availability.json`, `env.d/`, `logs/`, `lsp/`, `summaries_backup/`, `services_shared/`, `tasks_service_runtime/` |
| **Legacy / junk files** | `agents.db` (empty), `sessions.db` (empty), `agent_main/`, `agentX/`, `aX/`, `r1/`, `r2/`, `x/`, `restart_test_thread/`, `diag_tool/`, `fresh_*.txt/`, `list/` |
| **Test-created agents** (created by tests, safe to delete) | `e2e-*`, `dbg-*`, `acp_r6*`, `acptest*`, `acp_master_smoke`, `acpfix1`, `dg*`, `dgimg*`, `dgmut*`, `dw-*`, `dwtest_*`, `vc*`, `mj*`, `mc*`, `ca*`, `cn*`, `mm_cfg*`, `commit-editor*`, `ce-*`, `img*`, `acp_r6*`, `task_developer` |

All of this lies at **one level** (`~/.deep_worker/<entry>`), with no clear separation. Any CLI operation of the form "iterate over all subfolders of `~/.deep_worker/` and treat them as agents" produces false positives:

- `iter_thread_db_paths()` (`storage_paths.py:185`) already filters by the presence of `threads.db`, but not by location.
- If a user creates `~/.deep_worker/logs` or `~/.deep_worker/providers.json` next to an agent, and an agent by mistake has no `threads.db`, this code will skip it. The reverse is also bad: names like `tasks_service_runtime` **may** accidentally coincide with a new agent naming scheme.

The separation "agents vs everything else" must be **structural**, not conventional.

## 2. Solution (user requirement)

Move all agent databases from `~/.deep_worker/<name>/` to `~/.deep_worker/agents/<name>/`. The name of the subfolder is `agents/` (confirmed by the user).

- All **new** agents are created in `agents/`.
- All **existing** databases are moved there as well (lazy + bulk fallback).
- The old path `~/.deep_worker/<name>/` remains a **read-only fallback** for backward compatibility.
- **Test-created** folders are deleted (as a separate operation with explicit `--execute` + `--delete-test-dirs`).

## 3. Path structure

### 3.1. New layout

```
~/.deep_worker/
├── agents/                                # ← ALL agent databases live here
│   ├── agent_a/
│   │   ├── threads.db
│   │   ├── threads.db-shm
│   │   ├── threads.db-wal
│   │   ├── token_stats.jsonl
│   │   ├── session_metadata.json
│   │   ├── acp_session_state.json
│   │   ├── grok_session_state.json
│   │   ├── claude_session_state.json
│   │   ├── data/
│   │   ├── inbox/
│   │   └── summaries/
│   └── <other_agent>/
├── logs/                                  # infrastructure (NOT touched)
├── providers.json                         # infrastructure
├── provider_availability.json
├── env.d/
├── lsp/
├── summaries_backup/
├── services_shared/
└── tasks_service_runtime/
```

### 3.2. Compatibility with the old path

`agent_root()` reads the new path **first**, **then** (if nothing is found and `create=False`) the old one. This is needed for read operations: if the migration has not yet run on some host, a read must find the data at the old address rather than fail.

Writes go **only** to the new path. There is no fallback when creating files.

### 3.3. Storage version bump

The current `CURRENT_STORAGE_VERSION = 5` (`STORAGE_VERSION_WITH_PREFERRED_PROVIDER`). We introduce `STORAGE_VERSION_AGENTS_SUBDIR = 6`, meaning that the database lives under `agents/<name>/`. We update `detect_storage_version()`:

```
def detect_storage_version(agent_name, *, home):
    if agent_name is None:
        return STORAGE_VERSION_GLOBAL_DB

    validate_agent_name(agent_name)
    new_meta = deep_worker_home(home) / "agents" / agent_name / "session_metadata.json"
    if new_meta.exists():
        try:
            data = json.loads(new_meta.read_text())
            version = data.get("storage_version") if isinstance(data, dict) else None
            if isinstance(version, int):
                return version
            return STORAGE_VERSION_AGENTS_SUBDIR
        except (json.JSONDecodeError, OSError):
            return STORAGE_VERSION_AGENTS_SUBDIR

    # Legacy path — explicit chain detection
    legacy_meta = deep_worker_home(home) / agent_name / "session_metadata.json"
    if legacy_meta.exists():
        try:
            data = json.loads(legacy_meta.read_text())
            version = data.get("storage_version") if isinstance(data, dict) else None
            if isinstance(version, int):
                return version
        except (json.JSONDecodeError, OSError):
            pass
        return STORAGE_VERSION_WITH_PREFERRED_PROVIDER  # v5 (current best-guess)

    # v2 legacy global (sessions_<name>.db)
    if legacy_per_agent_db_path(agent_name, home=home, create_parent=False).exists():
        return STORAGE_VERSION_DEEPAGENTS_PER_AGENT  # v2

    # No data anywhere → caller is creating a new agent under the new layout
    return STORAGE_VERSION_AGENTS_SUBDIR
```

`migrate_agent_storage_to_v6(name, *, home)` moves `~/.deep_worker/<name>/` → `~/.deep_worker/agents/<name>/`:

```
1. ensure_private_dir(deep_worker_home(home) / "agents")   # ← mode 0o700, BEFORE first move
2. dst = deep_worker_home(home) / "agents" / name
   if dst.exists(): return {"skipped": "already-migrated"}
3. src = deep_worker_home(home) / name
   if not src.exists(): return {"skipped": "no-legacy"}
4. shutil.move(str(src), str(dst))   # may raise OSError(EXDEV) on cross-device
5. ensure_private_tree(dst)
6. meta = dst / "session_metadata.json"   # ← AFTER move, NOT via metadata_path()
7. Atomic metadata update: storage_version → 6
8. Return {"migrated": True, "dst": str(dst)}
```

`CURRENT_STORAGE_VERSION = STORAGE_VERSION_AGENTS_SUBDIR` (6).

### 3.4. Migration chain v2/v3/v4/v5 → v6

A single `v5 → v6` step does not cover old agents. The chain is: **v2 → v3 → v4 → v5 → v6**, via sequential calls to the existing `migrate_agent_storage_to_v{3,4}` + the new `migrate_agent_storage_to_v6`.

```
def migrate_agent_storage_to_current(name, *, home):
    """Walk the agent to the current storage version, one hop at a time."""
    current = detect_storage_version(name, home=home)
    if current == STORAGE_VERSION_GLOBAL_DB:
        if legacy_per_agent_db_path(name, home=home, create_parent=False).exists():
            migrate_agent_storage_to_v3(name, home=home)
            migrate_agent_storage_to_v4(name, home=home)
        else:
            raise ValueError(f"Cannot migrate: {name!r} has no legacy storage")
    elif current == STORAGE_VERSION_DEEPAGENTS_PER_AGENT:
        migrate_agent_storage_to_v3(name, home=home)
        migrate_agent_storage_to_v4(name, home=home)
    elif current == STORAGE_VERSION_AGENT_ROOT:
        migrate_agent_storage_to_v4(name, home=home)
    elif current in (
        STORAGE_VERSION_AGENT_ROOT_WITH_JSONL,
        STORAGE_VERSION_WITH_PREFERRED_PROVIDER,
    ):
        pass  # fall through to v6
    elif current == STORAGE_VERSION_AGENTS_SUBDIR:
        return {"skipped": "already-current"}
    else:
        raise ValueError(f"Unsupported storage version: {current}")

    return migrate_agent_storage_to_v6(name, home=home)
```

The lazy migration at process start and the CLI bulk migration both call `migrate_agent_storage_to_current()`, not v6 directly.

## 4. Code changes

### 4.1. `storage_paths.py`

- `agent_root(name, *, home, create=True)` → returns `deep_worker_home() / "agents" / name`.
- New `legacy_agent_root(name, *, home)` → returns `deep_worker_home() / name` (read-only fallback).
- New `iter_legacy_thread_db_paths(*, home)` — a **separate** function (NOT part of the main API). Used only by the migration CLI.
- `iter_thread_db_paths()` → iterates ONLY `~/.deep_worker/agents/*/threads.db`. Legacy is no longer in the main API.
- `detect_storage_version()` → new branch for v6 (see 3.3).
- New `migrate_agent_storage_to_v6()` (see 3.3) and `migrate_agent_storage_to_current()` (see 3.4).
- `CURRENT_STORAGE_VERSION = STORAGE_VERSION_AGENTS_SUBDIR`.

### 4.2. Direct compositions `deep_worker_home() / name` (ALL found)

This is the **complete** list of places where the agent path is built directly, bypassing `agent_root()`. All of them must switch to `agent_root()` for write operations, or to `read_agent_file()` (see 4.5) for read operations:

**Production code (CLI)**:

| File | Line | Current | Purpose | Becomes |
|---|---|---|---|---|
| `cli/agent_command.py` | 223 (`_get_agent_data_dir`) | `deep_worker_home() / name` | write | `agent_root(name)` |
| `cli/agent_command.py` | 3421 (start) | `deep_worker_home() / args.name` | write | `agent_root(args.name)` |
| `process/agent_process.py` | 132 (grok state) | `deep_worker_home() / agent_name / "grok_session_state.json"` | write | `agent_root(agent_name) / "grok_session_state.json"` |
| `process/agent_process.py` | 208 (codex state) | similar for codex | write | similar |
| `process/agent_process.py` | 591 (`self.data_dir`) | `deep_worker_home() / name` | write | `agent_root(name)` |
| `process/agent_stop.py` | 180 | `deep_worker_home() / name` | write | `agent_root(name)` |
| `tools.py` | 626 (inter-agent sender) | `deep_worker_home() / sender_name` | write | `agent_root(sender_name)` |
| `summaries.py` | 54 (`get_summaries_dir`) | `deep_worker_home() / agent_name / "summaries"` | read (subprocess check) | `agent_root(agent_name) / "summaries"` (the writer at line 61 already uses `agent_root`) |
| `sessions.py` | 1031 (delete agent) | `deep_worker_home(home_dir) / agent_name` | write — `shutil.rmtree` | `agent_root(agent_name, home=home_dir)` |
| `fork_agent.py` | 49 (src) | `deep_worker_home(home) / source_name` | read+write | `agent_root(source_name, home=home)` |
| `fork_agent.py` | 50 (dst) | `deep_worker_home(home) / new_name` | write | `agent_root(new_name, home=home)` |

**Production code (web UI)**:

| File | Line | Current | Purpose | Becomes |
|---|---|---|---|---|
| `app/services/agent_service.py` | 364 (load_metadata) | `deep_worker_home() / info.name / "session_metadata.json"` | read | `read_agent_file(info.name, "session_metadata.json")` |
| `app/services/agent_service.py` | 397 (docstring) | `<deep_worker_home>/<info.name>/token_stats.jsonl` | doc | update to `agents/<info.name>/token_stats.jsonl` |
| `app/services/agent_service.py` | 495 (load_snapshots) | `sp.deep_worker_home() / agent_name / STATS_FILE_NAME` | read | `read_agent_file(agent_name, STATS_FILE_NAME)` |
| `app/services/agent_service.py` | 535 (load_model) | similar | read | `read_agent_file(agent_name, STATS_FILE_NAME)` |
| `app/services/agent_service.py` | 2220 (load_launch_config) | `deep_worker_home() / agent_name` | read | `read_agent_file(agent_name, "session_metadata.json")` |
| `app/services/lifecycle.py` | 58 (`_read_status_file`) | `deep_worker_home() / agent_name / "session_metadata.json"` | read | `read_agent_file(agent_name, "session_metadata.json")` |
| `app/routers/agents.py` | 849 (`load_launch_config`) | `deep_worker_home() / agent_name` | read | `read_agent_file(agent_name, "session_metadata.json")` |

**Provider code** (state files):

| File | Line | Current | Purpose | Becomes |
|---|---|---|---|---|
| `providers/grok_provider/session_state.py` | 63 | `deep_worker_home() / _thread_base(thread_id)` | read+write | `agent_root(agent_name_from_thread_id(thread_id))` |
| `providers/acp/session_state.py` | 56 | same | same | same |
| `providers/claude_code_sdk_provider/session_state.py` | 50 | same | same | same |
| `providers/codex_provider/session_state.py` | 65 | same | same | same |

**Test fixtures** (do NOT break production, but block tests after the rename):

| File | Line |
|---|---|
| `tests/integration_ai/test_claude_api_error_e2e_subprocess.py` | 429, 580 |
| `tests/integration_ai/test_headless_to_tui_broadcast.py` | 103 |
| `tests/integration_ai/test_message_created_timestamps.py` | 152 |
| `tests/integration_ai/test_agent_process_lifecycle.py` | 56, 253 |
| `tests/integration_ai/clean_agent_fixture.py` | 137 |
| `tests/integration_ai/agent_fixture.py` | 148 |
| `tests/integration_ai/test_slow_tool_display.py` | 139 |
| `tests/integration_ai/test_tool_schemas_loaded_at_start.py` | 238 |
| `tests/integration_ai/test_taskmanager_tui.py` | 110 |
| `tests/integration_ai/test_agent_storage_v3.py` | 36, 49, 77 |
| `tests/integration_ai/test_token_stats_live.py` | 54 |
| `tests/integration_ai/test_claude_code_tool_then_text_hang.py` | 111 |
| `tests/integration_ai/test_codex_native_command_execution_crash.py` | 215 |
| `tests/integration_ai/test_taskmanager_progress.py` | 92 |
| `tests/integration_tools_schema/test_restart_self_sync_mechanism.py` | 77, 92 |
| `tests/integration_tools_schema/test_fork_agent_live.py` | 18, 74 |
| `tests/integration_tests/test_restart_with_fable_alias.py` | 330, 472, 578 |
| `tests/integration_tests/test_storage_v4_migration.py` | 110 |
| `tests/integration_tests/test_dev_launcher.py` | 93, 397 |
| `tests/unit_tests/test_daemonize_cgroup_detach.py` | 354 |
| `tests/unit_tests/test_cli_fast_return.py` | 114 |

### 4.3. `metadata_path` polymorphic warning

After the change to `agent_root`, the function `metadata_path(name, *, home)` automatically starts returning `~/.deep_worker/agents/<name>/session_metadata.json`. This is **desired behavior for reads**.

However, in `migrate_agent_storage_to_v6` you **must not** call `metadata_path(name)` BEFORE `shutil.move` — the function would return the new path (`agents/<name>/session_metadata.json`), which does not exist yet. Use the direct composition `dst / "session_metadata.json"` AFTER the `move` (see step 6 in 3.3).

### 4.4. Lazy migration at agent start

In `AgentProcess.__init__` (or in the CLI startup sequence):

```
def _ensure_agent_layout(name, *, home):
    """Idempotent: bring this agent's storage up to CURRENT_STORAGE_VERSION.
    Called once at agent-process startup, before any data access.
    """
    current = detect_storage_version(name, home=home)
    if current == CURRENT_STORAGE_VERSION:
        agent_root(name, home=home, create=True)
        return
    migrate_agent_storage_to_current(name, home=home)
```

Guarantees:
- `ensure_private_dir()` on `~/.deep_worker/agents/` runs **before** any `shutil.move`.
- `dst.exists()` is checked explicitly → we return `"already-migrated"` and **do not rely** on the semantics of `shutil.move`.
- Cross-device fallback (see 5.1): if `shutil.move` raises `OSError(EXDEV)`, we do `shutil.copy2 + shutil.rmtree`. If `rmtree` fails, we set an `mv_failed` flag in the metadata so that the next attempt tries again.
- Every migration is written to `~/.deep_worker/logs/migrate.log`.

### 4.5. Read fallback helper (new)

In `storage_paths.py`:

```python
def read_agent_file(name: str, filename: str, *, home: Path | None = None) -> bytes | None:
    """Read a file from the agent's dir, preferring the new layout.

    Lookup order:
      1. ``~/.deep_worker/agents/<name>/<filename>`` — preferred.
      2. ``~/.deep_worker/<name>/<filename>`` — legacy fallback.

    Returns the bytes if found, else ``None``. NEVER merges — the first hit
    wins (avoids accidentally reviving stale data after a partial migration).
    """
    new = agent_root(name, home=home, create=False) / filename
    if new.exists():
        return new.read_bytes()
    legacy = legacy_agent_root(name, home=home) / filename
    if legacy.exists():
        return legacy.read_bytes()
    return None
```

Used at all read-only call sites from table 4.2 (marked `read`). This is DRY instead of 11 copies of the fallback logic.

### 4.6. CLI: `deep-worker migrate-agent-dbs`

```
deep-worker migrate-agent-dbs
    [--dry-run]                # default — log only
    [--execute]                # actually move
    [--keep-legacy]            # do NOT delete legacy after the move (for verification)
    [--delete-test-dirs]       # delete test-created directories (requires --execute)
    [--freshness-days N]       # safety net: do not delete directories newer than N days (default 14)
    [--keep-list FILE]         # additional names to protect (union with registry)
```

Behavior (see 5.2 for classification):
1. Load `keep_list` = (names from `/tmp/deep_worker_registry.json` with a live PID via `os.kill(pid, 0)`) ∪ (`--keep-list` file).
2. Walk `~/.deep_worker/<entry>/` where `entry` is NOT in the **infrastructure_allowlist**.
3. Classify each `<entry>/` (5.2): `keep-move` / `test-candidate` / `fresh-unknown` / `move-unknown`.
4. Print progress: `moved=N, kept=N, deleted=N, skipped=N, freshness-skip=N`. Details go to `~/.deep_worker/logs/migrate.log`.

Argparse validation: `--delete-test-dirs` **without** `--execute` → `parser.error("--delete-test-dirs requires --execute")`.

### 4.7. `summaries_backup/`

`grep -rn 'summaries_backup' <repo>/deep_worker --include='*.py'` → **0 hits in production code**. This is **on-disk data only** (legacy backups from previous versions). It is neither written nor read. We leave it untouched. (An explicit grep confirmation after migration is added to the design, see 5.5.)

## 5. Safety

### 5.1. Migration

- `ensure_private_dir(deep_worker_home() / "agents")` (mode 0o700) **before** the first `shutil.move`.
- `if dst.exists(): return "already-migrated"` — an explicit guard **before** `shutil.move`.
- Cross-device (`OSError(EXDEV)`):
  ```
  try:
      shutil.move(str(src), str(dst))
  except OSError as e:
      if e.errno == errno.EXDEV:
          shutil.copytree(str(src), str(dst))
          ensure_private_tree(dst)
          try:
              shutil.rmtree(str(src))
          except OSError:
              meta = dst / "session_metadata.json"
              ... atomic write with "mv_failed": True
              return {"migrated": True, "partial": True}
      else:
          raise
  ```
- **Permissions**: after `shutil.move` (or `copytree`) we run `ensure_private_tree()` on the new location.
- **Idempotent**: `dst.exists()` check → skip.
- **Atomic metadata**: `session_metadata.json` via `tmp + replace` (as in `migrate_agent_storage_to_v4`).
- **Dry-run by default**. Real actions only with `--execute`.
- **Log**: every migration → stdout + `~/.deep_worker/logs/migrate.log`.
- **Progress logging**: ~250 dirs × move + chmod = minutes. We print `[N/total] processing <entry>...` every 5 entries.

### 5.2. Classification of test-created directories

**Step 1: keep-list early-skip** (before any regex check). `keep_list`:

```python
def build_keep_list(registry_path=Path("/tmp/deep_worker_registry.json")) -> set[str]:
    keep = set()
    if registry_path.exists():
        try:
            data = json.loads(registry_path.read_text())
            for name, info in data.items():
                pid = info.get("pid")
                if pid is not None:
                    try:
                        os.kill(pid, 0)   # alive check
                        keep.add(name)
                    except OSError:
                        pass  # dead, not in keep
        except (json.JSONDecodeError, OSError):
            pass
    return keep
```

CLI `--keep-list FILE` (text file, one name per line, `#` comments) — a union on top of the registry keep.

**Step 2: infrastructure allowlist** (never touched):

```python
INFRASTRUCTURE_ALLOWLIST = frozenset({
    "agents", "logs", "env.d", "lsp", "summaries_backup",
    "services_shared", "tasks_service_runtime",
    "providers.json", "provider_availability.json",
    "agents.db", "sessions.db",
})
```

**Step 3: test patterns** (regex, case-sensitive). A list, free of OR-precedence bugs:

```python
TEST_PATTERNS = [
    r"^e2e[-_]",
    r"^dbg[-_]",
    r"^acp_r6[a-e]$",
    r"^acptest\d*$",
    r"^acp_master_smoke$",
    r"^acpfix\d*$",
    r"^dg(img\d+|mut\d*|_imgtest[a-z0-9_]*|_probe[a-z0-9_]*|_dbg[a-z0-9_]*|big\d+|imglive\d+)$",
    r"^dw-[a-f0-9]{6,}$",
    r"^dwtest_[a-z0-9_]+$",
    r"^vc\d+_",
    r"^vc_reviewer$",
    r"^mj\d+_",
    r"^mc\d+_",
    r"^mcheck\d*$",
    r"^ca\d+_",
    r"^cn\d+_",
    r"^mm_cfg[a-z0-9_]*$",
    r"^commit-editor(-[a-z0-9]+|:.*)?$",
    r"^ce-(failover|manual)$",
    r"^img(test|verify|mutant)\d*$",
    r"^task_developer$",
    r"^manualcheck\d*$",
    r"^fresh_\d+_",
    r"^restart_test_thread$",
    r"^diag_tool$",
    r"^agent_main$",
    r"^agentX$",
    r"^aX$",
    r"^r[12]$",
    r"^x$",
    r"^list$",
    r"^test[_-]",
    r"^test[_0-9]",
]
```

`test_web` and other live agents do NOT match the test patterns, but they land in the keep_list — they are protected.

**Step 4: freshness heuristic**:

```python
def is_fresh(path: Path, days: int) -> bool:
    if not (path / "session_metadata.json").exists():
        return False
    try:
        data = json.loads((path / "session_metadata.json").read_text())
        started = data.get("launch_config", {}).get("started_at")
        if not started:
            return False
        ts = datetime.fromisoformat(started)
        return (datetime.now() - ts) < timedelta(days=days)
    except (json.JSONDecodeError, OSError, ValueError):
        return False
```

`--freshness-days N` (default 14). If `is_fresh(path, N)` → skip + log a warning, **do not delete**.

**Step 5: hard rule for doubtful cases**:

```
classify(entry):
    if entry in keep_list:        → "keep-move"
    if entry in INFRA_ALLOWLIST:  → "skip-infra"
    if matches(TEST_PATTERNS):    → "test-candidate" (move-only if NOT --delete-test-dirs)
    if is_fresh(entry, days):     → "fresh-unknown" (skip)
    else:                         → "move-unknown" (move only, never delete)
```

### 5.3. Permission checks

- Created directories — mode 0o700. Files — 0o600.
- `ensure_private_tree()` after every move.
- No 777/666. If a legacy dir has bad permissions, we fix them during the move.

### 5.4. Schema check for `session_metadata.json`

Before migration we check that the metadata does not contain hardcoded absolute paths that could become invalid after the move. Currently known fields in `launch_config`:

- `tree_root`: an absolute path to the **working_dir** (NOT to the data dir). It does not change on migration — this is fine.
- `pid`, `started_at`, `cli_args`: not paths.

There are **no** other path fields in the metadata. Verified: `launch_config.tree_root` is the only absolute field, and it is the working directory, not the data dir. Migration does not affect the metadata.

### 5.5. `summaries_backup/` explicit grep confirmation

After migration (even dry-run):

```
grep -rn 'summaries_backup' <repo>/deep_worker --include='*.py' | grep -v '\.pyc'
```

If 0 hits in production code → ok. If hits appear (new code uses this path) → fail the migration, alert.

## 6. Tests

### 6.1. Unit (`storage_paths.py`)

- `agent_root("foo")` → `~/.deep_worker/agents/foo/`.
- `legacy_agent_root("foo")` → `~/.deep_worker/foo/`.
- `read_agent_file("foo", "x.json")`: existing new → bytes; existing legacy only → legacy bytes; both → new bytes; neither → None.
- `migrate_agent_storage_to_v6("foo")` is idempotent: the second call is a no-op.
- `migrate_agent_storage_to_v6` with a mock `OSError(EXDEV)` → fallback to copy+rmtree. Mock `rmtree failed` → metadata with `mv_failed: True`.
- `detect_storage_version`: new path → 6. Legacy path → 5. Legacy v2 → 2. None → 6.
- `migrate_agent_storage_to_current` chain test (v2 → v6) for each variant.
- `iter_legacy_thread_db_paths` returns nothing from `agents/`. `iter_thread_db_paths` returns nothing from legacy.

### 6.2. Unit (read paths)

- `_load_context_snapshots` finds data in `agents/<name>/token_stats.jsonl` (new path).
- Same — fallback to `<name>/token_stats.jsonl` (legacy).
- If both files exist, **only the new one** is read (no merge).

### 6.3. Integration (lazy migration)

- Create `~/.deep_worker/<name>/threads.db` (legacy).
- Call `migrate_agent_storage_to_current(name)`.
- Verify: legacy moved to `agents/<name>/`, `threads.db` in place, mode 0o600.
- Call a second time → `{"skipped": "already-current"}`.

### 6.4. Integration (CLI migrate)

- Create an artificial layout in `tmp_path`:
  ```
  <tmp>/.deep_worker/
    test_xx/{threads.db, session_metadata.json}
    prod_yy/{threads.db, session_metadata.json}
    old_zz/{threads.db}
    logs/
    env.d/
  ```
- Mock `/tmp/deep_worker_registry.json` with `prod_yy` PID alive.
- `deep-worker migrate-agent-dbs --dry-run` → prints the plan, **does nothing**.
- `deep-worker migrate-agent-dbs --execute` → test_xx, prod_yy, old_zz moved to agents/; logs/ and env.d/ stay in place.
- `deep-worker migrate-agent-dbs --execute --delete-test-dirs` → test_xx and old_zz deleted, prod_yy moved, logs/ and env.d/ stay in place.
- **Progress logging**: every step is logged. ~250 dirs → we see progress, it does not appear hung.

### 6.5. Isolation via a monkeypatched accessor (per reviewer request)

We use `monkeypatch.setattr(storage_paths, 'deep_worker_home', lambda h=None: tmp_path / '.deep_worker')` for unit tests, and do **NOT** modify the `HOME` env var (async tasks may cache it, in which case the tests would not be isolated).

```python
def test_agent_root_under_agents(monkeypatch, tmp_path):
    fake_home = tmp_path / ".deep_worker"
    monkeypatch.setattr(
        "deep_worker_cli.storage_paths.deep_worker_home",
        lambda h=None: fake_home,
    )
    assert agent_root("foo") == fake_home / "agents" / "foo"
```

## 7. Non-goals

- We do **not delete** `~/.deep_worker/agents.db` (empty) and `sessions.db` (empty) — their history is separate, a separate feature.
- We do **not move** infrastructure directories (`logs/`, `lsp/`, `services_shared/`, `tasks_service_runtime/`).
- We do **not migrate** the ancestor's legacy tree (legacy global) — that was a separate branch of storage evolution; since then it has been removed and is forbidden to read (POLICIES §16).
- We do **not** introduce an env var for path overrides (there is already a `home=` parameter).
- We do **not** automatically delete test directories — only opt-in via `--delete-test-dirs`.

## 8. Rollback plan

**Cheapest**: `git revert` + redeploy the old code. If the new code has managed to write something to `agents/<name>/` (for example, between the merge and the rollback), then:

1. Stop all agents (`deep-worker stop --all`).
2. `git revert <merge_commit>` (or `git reset --hard <pre-merge_sha>`).
3. Redeploy the old code.
4. **Do NOT** run `mv ~/.deep_worker/agents/* ~/.deep_worker/` — the old code writes to `<name>/`, and if `agents/<name>/` exists it will NOT see that data.

**Reason**: git revert + redeploy leaves the data in `agents/`, but it is simply invisible to the old code (which reads from `<name>/`). This is safe. Moving it back (mv) is dangerous, because in the meantime new writes may have appeared in `agents/` that we would lose.

**Recovery log**: `~/.deep_worker/logs/migrate.log` keeps the list of all moved directories → can be restored manually if something goes completely wrong.

## 9. Open questions (resolved)

| # | Question | Decision (from the reviewer) |
|---|---|---|
| Q1 | Storage version name | `STORAGE_VERSION_AGENTS_SUBDIR = 6` — consistent naming. ✓ |
| Q2 | `iter_thread_db_paths` with `is_legacy` | **NOT in the main API**. A separate `iter_legacy_thread_db_paths()`. ✓ |
| Q3 | Default `--dry-run` or `--execute`? | `--dry-run` default. ✓ |
| Q4 | 14-day heuristic → configurable | `--freshness-days N` (default 14). ✓ |
| Q5 | keep_list source | **auto from registry + user `--keep-list` file (union)**. The registry is filtered via `os.kill(pid, 0)`. ✓ |

## 10. Changes outside the code

- Update `docs/ARCHITECTURE_OVERVIEW.md` — the section on storage paths.
- Update `docs/API_KEY_SECURITY_POLICY.md` — it mentions `~/.deep_worker/env.d/` (this does not change, but check the wording).
- Update `docs/DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md` — it has the path `{agent}/conversation_history/` in the ancestor's legacy tree (this is about the legacy global, it does not change).
- Update the comment in `storage_paths.py:32-41` — add a mention of the `agents/` subfolder.

## 11. Additional checks (per reviewer remarks)

| Remark | Resolution |
|---|---|
| 10 missed call sites | All 21 call sites are covered in 4.2. |
| Read fallback DRY | `read_agent_file()` in 4.5. |
| `metadata_path` polymorphic | Step 6 in 3.3 uses the direct composition `dst / "session_metadata.json"`, NOT `metadata_path()`. |
| Lazy migration race | Step 2 in 3.3: explicit `if dst.exists()`. |
| Cross-device partial state | Step 4 in 3.3 + the `mv_failed` flag in metadata. |
| `iter_thread_db_paths` legacy | Split: `iter_thread_db_paths()` + `iter_legacy_thread_db_paths()`. |
| Registry freshness | `os.kill(pid, 0)` in `build_keep_list()` (5.2 step 1). |
| `--delete-test-dirs` requires `--execute` | argparse validation in 4.6. |
| `session_metadata.json` schema | `tree_root` is the only path, and it is the working dir, not the data dir (5.4). |
| `summaries_backup/` grep confirmation | 5.5. |
| Rollback: cheapest = git revert | 8. |
| Test isolation via monkeypatch accessor | 6.5. |
| Do not migrate the ancestor's legacy tree | 7. |
| Progress logging | 4.6 + 5.1. |
