# Design: History Index MUST Catch Up On Every Agent Start

**Status:** Draft → ready to implement
**Date:** 2026-06-05
**Related (supersedes the relevant section of):**
- `DESIGN_HISTORY_MEMORY_RESTART_AND_PROGRESS.md` §3.4 ("Pre-Indexing Check")
- `DESIGN_HISTORY_MEMORY_TOOL.md`

---

## 1. Problem (observed)

After saving each summarization batch the agent's `summaries/<agent_id>.jsonl`
grew from 589 to 5285 lines over 8 batches (Jun 1 → Jun 5). The semantic
index under `.history_index/` stayed frozen at **589 indexed lines**, with
all index files showing `mtime = Jun 1 18:11`.

User runs `history_memory(...)` queries against perfectly relevant topics
discussed in batches 2-9 (an internal merger fix, uncommitted work, and a suppress fix). The tool returns 0 results from any of those
batches and only ever hits chunks from the first batch.

The tool is technically alive (queries succeed, return chunks) but
*useless*: it only knows the very first summarization snapshot the agent
ever produced.

## 2. Root cause

`deep_worker_cli/agent.py::create_cli_agent()` (lines 949-1006 as of a recent commit) guards the startup indexing block as:

```python
if jsonl_files and not history_index_exists(summaries_path):
    # ... index_all_history_files(...)
```

The guard is **wrong**. It re-indexes **only when no index exists at all**.
Once the very first run has built an index, every subsequent restart skips
indexing entirely, even when the JSONL has grown by thousands of lines
since.

The previous design (`DESIGN_HISTORY_MEMORY_RESTART_AND_PROGRESS.md`
§3.4) explicitly stated this behaviour as intentional, justifying it with:
*"If index exists but some files have new lines: the `_on_summarized`
callback handles this incrementally during the session."*

That justification is unsound for two independent reasons:

1. **Cross-process gap.** `_on_summarized` is in-process. While the agent
   is **not running** (between two starts), summarization cannot happen
   in this agent — but other writers (in this codebase: the
   summarization that happened *during the previous process's lifetime
   and was never indexed because the previous process either crashed,
   was killed mid-callback, or the callback raised silently*) can leave
   the JSONL ahead of the index. The next process must catch up.
2. **Silent-failure resilience.** If `_on_summarized` ever raises
   inside the summarization middleware, the index falls behind silently
   (current code in `summarization.py:675-676` and `:754-755` calls it
   without a try/except around it). The next agent start is the only
   opportunity to recover automatically.

## 3. Goal (single sentence)

**Whenever the agent starts with `history_memory` enabled, the index must
be brought fully up to date with all `*.jsonl` files in the summaries
directory before the `HistoryMemoryMiddleware` is constructed — regardless
of whether an index already exists.**

This is in addition to (not instead of) the existing on-summarization
incremental update.

## 4. Design

### 4.1 Behaviour matrix

| State on startup | New behaviour | Rationale |
|---|---|---|
| `--history-memory` off | Skip indexing | Tool not enabled, no work needed |
| flag on, no JSONL files | Skip silently | Nothing to index |
| flag on, JSONL exists, no index dir | Index everything, show progress | Same as before |
| flag on, JSONL exists, index dir present, no new lines | Skip silently (cheap line-count check) | No work needed |
| **flag on, JSONL exists, index dir present, new lines** | **Catch up: index delta, show progress if delta is large** | **THIS IS THE FIX** |

### 4.2 Code change

`deep_worker_cli/agent.py::create_cli_agent()`, in the
`if enable_history_memory:` block currently at lines 949-1017:

Replace:

```python
if jsonl_files and not history_index_exists(summaries_path):
    # ... full startup indexing with progress bar ...
```

with:

```python
if jsonl_files:
    pending = _count_pending_history_lines(summaries_path, jsonl_files)
    if pending > 0:
        # ... startup indexing with progress bar
        #     (same Rich progress code as today, no behavioural change
        #      for the "first-ever index" case) ...
```

Where the new helper:

```python
def _count_pending_history_lines(
    summaries_dir: Path,
    jsonl_files: list[Path],
) -> int:
    """Return total number of JSONL lines NOT YET indexed across all files.

    Reads ``.history_index/line_hashes.json`` (cheap, ~bytes) and
    counts lines in each JSONL file (file scan, but no parsing).

    Returns 0 when:
    * the index hash file doesn't exist (caller treats this as "index
      everything", which is the same behaviour as today)
    * every JSONL file's current line count equals its recorded count

    Treating 0 == "no work" lets the caller skip progress display
    entirely when the index is already current.
    """
```

Implementation lives in `deep_worker_cli/agent.py` (private helper). The
core function `index_history_file` already implements delta logic
correctly (uses `line_counts.get(path, 0)`). We only need to **stop
guarding the call site** and add a cheap pre-check so we don't even show
the progress UI when there's nothing to do.

Note: when `_count_pending_history_lines` returns `> 0` and **no index
dir exists yet**, this is identical to today's first-time path. When it
returns `> 0` AND **an index exists**, this is the new catch-up path.
Both go through `index_all_history_files(...)`, which is already
idempotent and incremental (`index_history_file` returns immediately
when `current_line_count <= last_line_count`).

### 4.3 Why not just always call `index_all_history_files`?

We could. But:

* Calling it imports `llama_index.embeddings.huggingface` and instantiates
  `HuggingFaceEmbedding`, which loads the embedding model from disk.
  That's seconds even when there's nothing to index.
* Showing a "Indexing conversation history..." spinner that finishes in
  100ms is more noise than signal.

The pre-check (`_count_pending_history_lines`) is a cheap file scan
(`sum(1 for _ in open(p))` per JSONL, plus one tiny JSON read for
`line_hashes.json`) — it takes microseconds and lets us skip both the
embedding load and the UI entirely when the index is current.

### 4.4 What `_on_summarized` keeps doing

Unchanged. `summarization.py` keeps calling
`_on_summarized(path)` → `index_all_history_files(summaries_dir)` after
each summarization batch. The new startup catch-up is **defence in
depth**, not a replacement — it covers:

* the gap while the agent is not running, and
* silent failures in the in-process callback.

### 4.5 Edge cases

* **Index dir exists but `line_hashes.json` missing** — treat all lines as
  pending (force full re-index). This handles half-built/corrupted index
  states.
* **Some JSONL files in `line_hashes.json` no longer exist on disk** —
  ignore those entries. They don't contribute to pending count.
* **JSONL file shrank** (file replaced rather than appended) —
  `index_history_file` already handles this gracefully (re-index from
  scratch). The pre-check just reports the full new line count.
* **Indexer raises mid-startup** — same as today: `try/except`
  → `logger.warning`, `history_index_exists()` check decides whether
  the middleware enables the tool.

## 5. Files to modify

| File | Change |
|---|---|
| `deep_worker/deep_worker_cli/deep_worker_cli/agent.py` | Remove `not history_index_exists(...)` guard; add `_count_pending_history_lines` helper; gate progress UI on `pending > 0` |
| `deep_worker/deep_worker_cli/tests/unit_tests/test_history_index_catchup_on_start.py` | New file: tests for catch-up behaviour |

## 6. Test plan (TDD — write these first, watch them fail, then fix)

1. **`test_catchup_when_jsonl_grew_since_last_index`**
   Build an index over a small JSONL (2 lines). Append 3 more lines.
   Build an agent with `--history-memory`. Assert that after start
   `line_hashes.json` records 5 lines (not 2).
2. **`test_no_work_when_index_is_current`**
   Build an index, then build an agent. Assert
   `index_all_history_files` is **not** called (cheap pre-check
   returns 0).
3. **`test_first_run_indexes_everything`** (regression guard for the
   existing happy path).
4. **`test_pending_count_helper_with_missing_hash_file`**
   Index dir exists but no `line_hashes.json` → helper returns total
   lines in all JSONL files.
5. **`test_pending_count_helper_ignores_stale_entries`**
   `line_hashes.json` references a file that no longer exists →
   helper does not count it (and does not raise).

## 7. Out of scope

* Investigating why `_on_summarized` historically silently failed
  between Jun 1 and Jun 5 for a particular agent. That's a separate diagnostic;
  this design fixes the symptom (stale index) by adding the catch-up
  path. The in-process callback continues to run; if it works, the
  catch-up does nothing.
* Concurrent indexing across multiple agent processes touching the
  same summaries dir. Each agent owns its own per-agent data directory tree.

---

## 8. Acceptance

After this design ships:

1. Stop the agent. Append messages to its JSONL. Restart it.
2. Immediately query `history_memory` for those new messages.
3. The new messages **must** appear in the results.

Today, step 3 fails. After the fix, step 3 passes.
