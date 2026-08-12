# DESIGN: `--reindex-history` command (rebuild an agent's history-memory index on demand)

Status: Draft
Date: 2026-07-03
Worktree: development tree (branch `dev`)

## 1. Problem

The history-memory vector index (`~/.deep_worker/<agent>/summaries/.history_index/`)
is only (re)built at two moments:

1. **Incrementally**, via the `_on_summarized` callback, when the running
   agent offloads history (`agent.py:1167`).
2. **Catch-up at agent start**, when `_should_catchup_history_index()` finds
   the JSONL line-count exceeds `line_hashes.json` (`agent.py:1184`).

There is **no way to force a full rebuild on demand**. If the index is
deleted, corrupted, or a user wants a clean rebuild (e.g. after editing the
JSONL, or a partial-delete leaving `line_hashes.json` out of sync with a
missing `vectors.lance`), the only options today are "restart the agent" (only
does an incremental catch-up, and cannot fix a stale-hash / missing-vectors
desync) or manual `rm -rf` + restart. `dw-hist --reindex` is a documented but
**unimplemented placeholder** (`dw_hist/README.md:152`).

This design adds an explicit CLI command to rebuild the index on demand, with
a **progress display** (the operation can take a long time — embedding-model
load + re-embedding every JSONL line).

## 2. Goals

1. Add `deep-worker <name> --reindex-history` — force a **full** rebuild of the
   agent's history-memory index (clear + re-index all `summaries/*.jsonl`).
2. Show **progress** during the rebuild (it can be slow): a per-file progress
   bar for the offline path, and a spinner plus final summary for the
   running-and-idle path.
3. Work whether the agent is **not running** OR **running-and-idle** (the two
   states the user specified). Refuse cleanly when the agent is running and
   **busy** (not idle), so we never rmtree/rebuild an index a live turn is
   concurrently reading/writing.
4. Fail loud on any error (missing summaries dir, index build failure) with a
   non-zero exit code and a user-visible message.

## 3. Non-goals

1. NOT an incremental update — this is always a **full** clear+rebuild (that is
   the whole point: fix desync / corruption). Incremental catch-up already
   exists at start and via `_on_summarized`.
2. Do NOT change the index format, storage location, or the
   `index_all_history_files` / `clear_history_index` internals (reuse as-is).
3. Do NOT add reindex to `dw-hist` (separate tool; out of scope).
4. Do NOT reindex `project_memory` (code index) — history only.
5. NOT persisted to `launch_config` — it is a one-shot maintenance command,
   never a launch flag.

## 4. Design

### 4.1 CLI surface

A new offline-style subcommand flag on the existing parser, dispatched in
`main()` in the same shape as `--history` / `--list-summaries`:

```
deep-worker <name> --reindex-history
```

- `--reindex-history` (`action="store_true"`, `dest="reindex_history"`).
- Requires the agent `name` positional (like `--history`). Reject with exit 2
  if `name` is missing (falls through to the existing "agent name required"
  guard, but we also short-circuit early — see §4.4).
- Dispatched BEFORE the launch path in `main()`:
  ```python
  if getattr(args, "reindex_history", False):
      return asyncio.run(reindex_history_command(args.name))
  ```

### 4.2 Running-state guard (the core safety decision)

Reindex = `clear_history_index()` (rmtree of `.history_index`) + full rebuild.
Doing that from the CLI process while a **live agent** holds LanceDB open and
may fire `_on_summarized` concurrently risks a corrupted / half-written index
(rmtree under an open LanceDB table; a concurrent offload re-creating files
mid-wipe). So the guard is state-dependent:

`reindex_history_command(name)` first calls `ProcessRegistry().lookup(name)`:

- **Not running** (`lookup is None`) → the CLI performs the rebuild **directly**
  in-process (§4.3, offline path). Safe: nobody else touches the index.
- **Running** (`lookup` returns `AgentProcessInfo` with `rpyc_port`) → query
  the live status via `AgentClient.get_status()`. The returned status is
  normalized before comparison because `SessionInfo.status` can be enum-like or
  a raw string:
  ```python
  status_value = getattr(status, "value", status)
  normalized_status = str(status_value).lower()
  ```
  - `normalized_status == "idle"` → **delegate the rebuild to the agent** over RPyC
    (§4.5, agent-side path). The agent owns its index; it runs the rebuild
    under `_invoke_lock` so it is serialised against any turn and against
    `_on_summarized`. This is the correct behaviour
    for the "running-and-idle" case the user asked for — no CLI-side rmtree
    under a live process.
  - any other value (`running`, `waiting_input`, `interrupted`, `error`,
    `shutdown`, or unknown) →
    **refuse**: print `Error: agent '<name>' is busy (status: <s>); stop it or
    wait until idle before reindexing.` and return exit code 1. We never touch
    a busy agent's index.

Rationale for delegating (rather than CLI-side rebuild) when idle-but-running:
the single-source-of-truth architecture already funnels all index mutation
through the agent process (`_on_summarized`). Reusing that ownership avoids two
writers to one LanceDB table. The alternative — refuse for ANY running agent
and require a stop — is simpler but violates the user's explicit "or idle"
requirement, so it is rejected (see §5).

### 4.3 Offline rebuild path (agent not running)

`reindex_history_command` (in `agent_command.py`), for the not-running case:

```python
async def reindex_history_command(agent_name: str) -> int:
    from deep_worker_cli.summaries import get_summaries_dir
    from semantic_search.history_indexer import (
        clear_history_index, index_all_history_files,
    )
    summaries_dir = get_summaries_dir(agent_name)
    if not summaries_dir.exists():
        print(f"Error: no summaries directory for agent '{agent_name}' "
              f"({summaries_dir}) — nothing to index.", file=sys.stderr)
        return 1
    jsonl_files = sorted(summaries_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Agent '{agent_name}' has no conversation history "
              f"(no *.jsonl in {summaries_dir}); nothing to reindex.")
        return 0

    clear_history_index(summaries_dir)          # full wipe → forces from-scratch

    stats = _run_reindex_with_progress(summaries_dir, len(jsonl_files))
    print(f"Reindexed {stats.files_processed} file(s): "
          f"{stats.lines_read} lines → {stats.chunks_indexed} chunks "
          f"({stats.errors} error(s)).")
    return 1 if stats.errors else 0
```

`_run_reindex_with_progress` wraps `index_all_history_files(summaries_dir,
on_progress=...)` in a `rich.progress.Progress` bar keyed on file count — the
SAME progress-UI shape already used by the start-up catch-up
(`agent.py:1200-1225`), factored into a small helper so both call sites share
it. The `on_progress(filename)` callback advances the bar per file.

### 4.4 Early name guard + mutual exclusion

- `--reindex-history` requires `name`. Dispatch happens after the
  `--list`/`--history` blocks but BEFORE the "agent name required" fallthrough;
  we add an explicit check: if `reindex_history and not args.name` → print
  `Error: --reindex-history requires an agent name.` and return 2.
- `--reindex-history` combined with a launch is nonsensical → it short-circuits
  and returns (never proceeds to start the agent), same as `--history`.

### 4.5 Agent-side rebuild path (running-and-idle) — RPyC delegation

New `exposed_reindex_history` on `AgentService`, mirroring the maintenance-RPC
pattern of `exposed_reset_task_manager` (agent_service.py:715):

```python
async def exposed_reindex_history(self) -> str:
    if not self._agent_process:
        return json.dumps({"success": False, "error": "Agent not initialized"})
    try:
        stats = await self._agent_process.reindex_history_owned()
        return json.dumps({"success": True, **stats})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"success": False, "error": str(e)})
```

`AgentProcess.reindex_history_owned` runs the clear+rebuild **under
`_invoke_lock`** (same critical-section discipline as the startup-query work),
so it cannot interleave with a turn or an `_on_summarized` offload. Both the
`clear_history_index()` rmtree and the indexing work run in the worker thread;
the event loop is not blocked by large filesystem deletes or LanceDB writes:

```python
async def reindex_history_owned(self) -> dict:
    lock = self._invoke_lock  (lazy-init as elsewhere)
    async with lock:
        summaries_dir = get_summaries_dir(self.name)
        stats = await asyncio.to_thread(_clear_then_index_history, summaries_dir)
        return {"files": stats.files_processed, "lines": stats.lines_read,
                "chunks": stats.chunks_indexed, "errors": stats.errors}

def _clear_then_index_history(summaries_dir: Path) -> HistoryIndexStats:
    clear_history_index(summaries_dir)
    return index_all_history_files(summaries_dir)
```

Notes:
- **No loop blocking:** `clear_history_index` (directory rmtree) and
  `index_all_history_files` (embedding-model load + LanceDB writes) both run in
  `asyncio.to_thread`, exactly like `_on_summarized` offload does
  (`summarization.py:833`). The lock is held across the thread await so mutual
  exclusion holds, but the loop keeps ticking.
- **Running-path progress v1:** the CLI renders an indeterminate Rich spinner
  while the blocking RPyC call is in flight, then prints final counts from the
  RPC return. This satisfies visible progress without adding a new subscriber
  streaming protocol in v1.
- **Deferred streaming option:** if per-file progress is later needed for the
  running path, the sync `index_all_history_files(on_progress=...)` callback
  must bridge from the worker thread to the agent event loop explicitly:
  capture `loop = asyncio.get_running_loop()` before `to_thread`, call
  `asyncio.run_coroutine_threadsafe(notify_callback(event), loop)` or
  `loop.call_soon_threadsafe(lambda: asyncio.create_task(...))` inside the sync
  callback, and drain scheduled futures/tasks before returning if final-event
  ordering must be guaranteed. That is intentionally out of scope for v1.
- The agent's `history_memory` tool availability is re-evaluated on next start
  (middleware checks at `__init__`); a live reindex does not hot-swap the tool
  in the current process — out of scope, and the index is still queryable by
  the already-registered tool once rebuilt (same dir).

### 4.6 Progress display

- Offline path: `rich.progress.Progress` with `SpinnerColumn`, description
  (`Reindexing <file>...`), `BarColumn`, percentage; `total = len(jsonl_files)`;
  `advance=1` per `on_progress`. Identical component set to the start-up
  catch-up bar (factored into `_run_reindex_with_progress`).
- Running-and-idle path: indeterminate Rich spinner while waiting for
  `AgentClient.reindex_history()` / `exposed_reindex_history`, then the final
  counts from the RPC response. No subscriber stream in v1.
- Final line (both paths): `Reindexed N file(s): L lines → C chunks (E error(s)).`

### 4.7 Error handling

| Condition | Behaviour |
|---|---|
| `name` missing | `Error: --reindex-history requires an agent name.` exit 2 |
| summaries dir absent | `Error: no summaries directory ...` exit 1 |
| no `*.jsonl` files | message "nothing to reindex", exit 0 (not an error) |
| agent running & busy (not idle) | `Error: agent '<name>' is busy (status: <s>); stop it or wait until idle.` exit 1 |
| agent running & idle | delegate via RPyC; exit 0/1 per returned `errors` |
| `clear_history_index` raises | print `Error: failed to clear history index ...`; exit 1 / RPC `{success:false}` |
| `index_all_history_files` hard-raises | print `Error: failed to rebuild history index ...`; exit 1 / RPC `{success:false}` |
| `index_all_history_files` returns `stats.errors > 0` | print final stats including error count; exit 1 |
| RPyC call fails (running path) | `Error: failed to reach running agent '<name>': <e>` exit 1 |

## 5. Alternatives considered

- **Always refuse if the agent is running (require stop).** Simpler, no RPyC
  path, no concurrency concern. Rejected: violates the user's explicit "or
  idle" requirement.
- **CLI-side rmtree+rebuild even when running-and-idle.** Rejected: two writers
  to one LanceDB table (CLI rmtree vs. the agent's open handle / `_on_summarized`)
  → corruption risk. The agent must own the mutation.
- **Reuse `--clean`.** Rejected: `--clean` wipes checkpoints/DB and is a launch
  modifier, not a maintenance no-launch command; wrong semantics.
- **Progress via RPyC subscriber stream for running path.** Deferred from v1:
  it needs an explicit thread-to-loop bridge, lifecycle handling for subscribe /
  unsubscribe, subscriber-disconnect semantics, and final-event ordering. The v1
  running path uses spinner + final counts; the offline path still shows the
  full per-file bar.

## 6. TDD plan

RED tests first.

### 6.1 CLI parsing / dispatch (unit)
- `test_reindex_history_flag_parses`
- `test_reindex_history_requires_name` (exit 2)
- `test_reindex_history_dispatches_command` (main routes to reindex_history_command)

### 6.2 Offline rebuild path (unit, mocked indexer)
- `test_reindex_offline_calls_clear_then_index` (order: clear before index)
- `test_reindex_offline_no_summaries_dir_exit_1`
- `test_reindex_offline_no_jsonl_files_exit_0_message`
- `test_reindex_offline_progress_callback_invoked_per_file`
- `test_reindex_offline_errors_in_stats_exit_1`

### 6.3 Running-state guard (unit, mocked ProcessRegistry + AgentClient)
- `test_reindex_running_busy_refuses_exit_1` (status running → refuse, no clear)
- `test_reindex_running_idle_delegates_via_rpyc` (status idle → RPyC path,
  no CLI-side clear)
- `test_reindex_status_guard_accepts_enum_like_idle`
- `test_reindex_status_guard_accepts_string_idle`
- `test_reindex_status_guard_refuses_unknown_status`
- `test_reindex_not_running_uses_offline_path`

### 6.4 Agent-side reindex (unit, focused on AgentProcess)
- `test_reindex_history_owned_runs_under_invoke_lock` (serialised vs a
  concurrent turn — same pattern as the startup-query concurrency test)
- `test_reindex_history_owned_clears_then_indexes_in_thread` (single
  `asyncio.to_thread` critical section; clear before index)
- `test_reindex_history_owned_clear_failure_returns_rpc_error`
- `test_reindex_history_owned_returns_stats`

### 6.5 Deferred streaming tests (only if running-path streaming is added later)
- `test_reindex_progress_callback_from_worker_thread_schedules_async_notify`
- `test_reindex_progress_notify_tasks_drained_before_final_response`
- `test_reindex_progress_callback_has_no_unawaited_coroutine_warning`

### 6.6 Integration (no mocks, real filesystem)
- `test_reindex_offline_rebuilds_index_from_jsonl` (seed summaries/<agent>.jsonl,
  delete .history_index, run offline reindex, assert
  .history_index/lancedb/vectors.lance recreated and line_hashes.json present)
- `test_reindex_offline_fixes_stale_hash_desync` (leave line_hashes.json but
  delete lancedb → offline reindex clears+rebuilds → vectors present again)

## 7. Compatibility

- Purely additive: new flag + new command + new RPyC method. No existing
  signature changes.
- `index_all_history_files` / `clear_history_index` reused unchanged.
- The start-up catch-up progress UI is factored into a shared helper
  (`_run_reindex_with_progress`); the existing call site keeps identical
  behaviour (pure refactor covered by existing start tests).
- No launch_config change; flag is one-shot.

## 8. Roll-out

dev → review → TDD impl → code review → commit → FF to master.
