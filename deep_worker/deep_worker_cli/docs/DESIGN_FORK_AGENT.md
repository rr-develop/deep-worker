# DESIGN: `deep-worker --fork <old> [--from <message_id>]`

## Goal

Implement `deep-worker <new_name> --fork <old_agent_name> [--from <message_id>]` —
fork (snapshot-copy) an existing agent's data directory into a new agent, then start
the new agent with its own history.

## Semantics (from issue #110)

1. **New agent name**: positional arg (like normal `deep-worker <name>`), must NOT
   already exist (error otherwise).
2. **`--fork <old>`**: source must exist on disk (`~/.deep_worker/<old>/`). Source
   may be running — copying is read-only.
3. **What is copied**:
   - `session_metadata.json` → new agent's `launch_config`, with name/pid/started_at
     substituted.
   - History (`threads.db` + WAL/SHM) — via SQLite backup API (safe for running
     source). **Always full copy** (see `--from` semantics below).
   - Summaries (`summaries/`) — copied fully by default.
   - `data/` — recursive copy (working files).
4. **`--from <message_id>`**: Truncates only the `summaries/` JSONL files, keeping
   entries up to and including `message_id`. `threads.db` (checkpointer) is **always
   a full copy**. Rationale below.
5. **CLI overrides**: `--provider`, `--model`, `--dir`, `-sp`, `-tm`, `-m`, etc.
   override the copied `launch_config` (same merge pattern as `--restart`).
6. **`--clean` and `--fork` incompatible**: explicit SystemExit.
7. **After fork**: start the new agent (like `deep-worker <new>` with merged config).

## Where to implement

### 1. New module: `deep_worker_cli/fork_agent.py`

Contains the pure helper `fork_agent_data()` that can be unit-tested without
starting any agent.

#### Steps inside `fork_agent_data()`:

##### a) Validate
- `src_dir` must exist (FileNotFoundError).
- `dst_dir` must NOT exist (FileExistsError).

##### b) Create dst directory
- `dst_dir.mkdir(parents=True, exist_ok=False)` — raises if exists.
- `os.chmod(dst_dir, 0o700)`.

##### c) Copy `data/` directory
- `shutil.copytree(src_dir / "data", dst_dir / "data", dirs_exist_ok=True)`
  if source data/ exists. If no data/, skip.
- Post-copy: walk the tree and fix permissions (0o700 dirs, 0o600 files).

##### d) Copy `summaries/` directory
- If `message_id` is set:
  - Require `src_summaries/` exists (`ValueError` if not — agent has no summary history).
  - Iterate over `*.jsonl` files, filter entries up to and including `message_id`
    via `_copy_summaries_truncated()`. Raises `ValueError` if `message_id` not found.
  - `*.md` files are copied whole (no message_id semantics).
- Otherwise: full `shutil.copytree(...)` → dst if source summaries/ exists.
- Post-copy: fix permissions (0o700 dirs, 0o600 files).

##### e) Copy `session_metadata.json`
- Load src, update:
  - `pid` → current PID
  - `started_at` → current timestamp
  - `cli_args` → current CLI args (with new name)
  - Clear runtime status fields (`status`, `error_message`, etc.)
- Write to dst with 0o600 perms.

##### f) Copy `threads.db` via SQLite backup API
- `sqlite3.backup()` — safe for running source (WAL mode).
- After copy, set 0o600 on dst DB file.
- **Always full copy**, regardless of `--from`. Rationale:
  - The checkpointer is the agent's ground-truth conversation state.
  - On next boot, the agent loads its full checkpoint state, then the summarizer
    truncates older entries according to the (partially truncated) summaries.
  - The real "point of fork" is the summaries — they control which context survives
    into the new agent's working memory. A full threads.db ensures the agent can
    still access any checkpoint ID referenced prior to the fork point without error.

##### g) Partial-copy safety
- Caller wraps in try/except: `shutil.rmtree(dst_dir, ignore_errors=True)` on failure.

### 2. CLI changes in `deep_worker_cli/cli/agent_command.py`

#### `create_parser()` additions:
- `--fork OLD_NAME`: fork from an existing agent.
- `--from MESSAGE_ID`: truncate summaries up to this message_id (threads.db full copy).

#### New handler `do_fork(args: argparse.Namespace) -> int`:
1. Validate: `args.name` (new) dir does NOT exist → error if exists.
2. Validate: `args.fork` (old) dir exists → error if missing.
3. Validate: `--clean` + `--fork` → SystemExit with message.
4. If `--from` given: pass it to `fork_agent_data()` for summary truncation.
5. Load saved launch_config from source via `load_launch_config()` → error if None.
6. Merge CLI overrides via `merge_restart_config()`.
7. Build launch_config via `build_launch_config(effective)`.
8. Call `fork_agent_data(src_dir, dst_dir, args.name)` with cleanup wrapper.
9. Write launch_config to dst's `session_metadata.json`.
10. Call `run_agent()` with effective args (normal launch).

#### Dispatch order in `main()`:
- Place `--fork` check AFTER `--restart` but BEFORE `run_agent()` fallthrough.

### 3. Tests

#### Unit tests: `tests/unit_tests/test_fork_agent.py`
- `test_fork_full_copy`: mock source with all dirs.
- `test_fork_source_not_found`: src missing.
- `test_fork_dst_already_exists`: dst exists.
- `test_fork_no_data_dir`: source without data/.
- `test_fork_no_summaries`: source without summaries/.
- `test_fork_no_threads_db`: source without threads.db.
- `test_fork_permissions`: verify 0o600/0o700 perms.

#### Integration test: `tests/integration_tools_schema/test_fork_agent_live.py`
- Create fake source, run fork code path, verify dst structure.

## Scope

- `--fork <old>` — full copy of all agent data.
- `--from <message_id>` — truncates only summaries (JSONL) up to `message_id`;
  threads.db is always full copy (see rationale below).
- `--clean` + `--fork` → SystemExit.
- CLI overrides via `merge_restart_config()` + `build_launch_config()`.
- Partial copy cleanup on error.
- File permissions correct.

## Edge cases

| Condition | Behaviour |
|-----------|-----------|
| Source agent running | Safe: SQLite backup() reads without locking. data/ copy best-effort. |
| Source has no summaries | Fork succeeds, dst has no summaries. |
| Source has no data/ | Fork succeeds, dst has no data/. |
| Source has no threads.db | Fork succeeds, dst has no threads.db. Warning. |
| dst already exists | Error: agent name taken. |
| src does not exist | Error: source not found. |
| `--clean` + `--fork` | SystemExit. |
| `--from <id>` given | Truncates summaries JSONL up to `message_id`. threads.db full copy. |

## Why `--from` truncates only summaries (not threads.db)

The agent loads two things on startup:

- **State** from the checkpointer (`threads.db`) — internal checkpoint deltas,
  write buffers, thread metadata. This is an implementation detail.
- **Conversation context** from summaries (`summaries/<id>.jsonl`) — the
  LLM-generated narrative that the agent "remembers" from its past turns.

Truncating summaries up to `message_id` effectively resets the agent's visible
memory to that point. Copying `threads.db` fully preserves structural integrity
(no orphaned checkpoints) without leaking past LLM context into the new agent's
summarization window — the new agent will generate fresh summaries from that
point onward using the copied full state but truncated conversation memory.

The checkpointer's internal delta chain does carry `message_id` metadata, but
materialising a checkpoint as a root is an optimisation, not a correctness
requirement. The current approach (full threads.db + truncated summaries) is
correct, simpler, and safe for a running source agent.

## Permissions

After `shutil.copytree()`, files may carry source permissions. Post-copy fix:
- Walk dst tree, set dirs 0o700, files 0o600.
