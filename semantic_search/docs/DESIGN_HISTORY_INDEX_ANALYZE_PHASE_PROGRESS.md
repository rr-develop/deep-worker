# DESIGN — Phase 1 "Analyzing history…" progress during chunking

## Status
Proposed — awaiting review by an internal reviewer.

## Context / problem
Progress of history re-indexing (`7ba222e`) is already two-axis and correct on the
EMBEDDING/WRITE phase:
- `total = lines_read` (the number of new messages) is known up front → the bar
  shows `X/N`, `%`, ETA from the first tick;
- embedding proceeds in slices of `embed_progress_step` (progress tick), writing —
  in batches of `index_batch_size` (1 transaction/batch), the axes are independent.

One MUTE stretch remains. Before embedding, `index_history_file` calls
`_pending_chunks_for_file → _chunk_messages`, which tokenizes and glues
all message groups into chunks. On a large history (an agent with ~8k
messages) this is ~87 s of PURE silence: the bar is not yet created / sits at zero,
and the user cannot tell whether the process is stuck. The user's specification:

> Phase 1 «Analyzing history…» must tick as chunking progresses (per group/
> record), so that a long pre-computation does not hang silently.

## Root cause
`_chunk_messages` iterates over `groups` (the result of
`_group_by_summarized_event`) silently — there is no callback to tick. The number
of groups is known BEFORE the loop (`len(groups)`), i.e. the denominator for phase 1
is available immediately, just like `lines_read` for phase 2.

## Design of the fix

### A separate "analysis" progress axis, tied to nothing else
Phase 1 (chunking/analyze) and Phase 2 (embed/write) are SEPARATE axes. Phase 1
ticks per message GROUP (the natural unit of the `_chunk_messages` loop), Phase 2 —
per message (as today). Both denominators are known in advance.

### Core: a callback in `_chunk_messages`
```python
def _chunk_messages(groups, *, effective_max, tokenize_fn,
                    on_group_progress: Callable[[int, int], None] | None = None):
    total_groups = len(groups)
    if on_group_progress is not None:
        on_group_progress(0, total_groups)
    for gi, group in enumerate(groups, start=1):
        ...  # existing logic unchanged
        if on_group_progress is not None and (gi % step == 0 or gi == total_groups):
            on_group_progress(gi, total_groups)
```
- `step` = `max(1, config.analyze_progress_step)` — NEW parameter
  `IndexConfig.analyze_progress_step: int = 32` (default "small step", per
  requirement; NOT equal to the write batch size 1000). Tick by modulo, plus a
  guaranteed final tick at `gi == total_groups`.
- Callback is optional: without it the behaviour is 1:1 with the old one
  (backward compatible with all current call sites).

### Plumb-through across layers
- `_pending_chunks_for_file(..., on_group_progress=None)` — forwards to
  `_chunk_messages`.
- `index_history_file(..., on_analyze_progress: Callable[[int,int],None]|None)`
  — forwards to `_pending_chunks_for_file`. The existing
  `on_message_progress` (phase 2) is untouched.
- `index_all_history_files(..., on_analyze_progress=None)` — forwards the
  per-file callback, as already done for `on_message_progress`.

### CLI (`agent_command._run_reindex_with_progress`)
One rich `Progress` with TWO tasks (or a reusable one):
- task «Analyzing {file}…» with `total=total_groups`, ticking from
  `on_analyze_progress`. Same columns (`MofNCompleteColumn`, `%`,
  `TimeElapsedColumn`, `TimeRemainingColumn`) — unit "groups".
- when analysis of a file finishes — task «Indexing {file}…» with `total=lines_read`
  (as today) from `on_message_progress`.

The start-up catch-up in `create_cli_agent` (`agent.py`) — optionally with the same
callback; touched minimally (a file-level bar already exists there).

## What does NOT change
- The phase 2 axis (embedding in slices / writing in batches) — unchanged.
- `line_range` relative to the slice, `total=lines_read` — already consistent
  (U+2028 fix). No 0→100% jump on the incremental tail (verified).
- The chunking/tokenization logic inside the loop — unchanged; only a tick
  at the end of the iteration is added.

## Testing plan (TDD, RED first)
**semantic_search unit** (`test_history_indexer_analyze_progress.py`):
- `_chunk_messages` with a callback: receives `(0, total_groups)` at the start,
  `(total_groups, total_groups)` at the end, grows monotonically, `done<=total`.
- Without a callback — previous behaviour (same list of chunks).
- `analyze_progress_step` controls the tick frequency (step 1 → tick on every
  group; step > total → only start+final).
- `_pending_chunks_for_file` forwards the callback.

**semantic_search integration** (real e5):
- `index_history_file` with `on_analyze_progress` — phase 1 reaches
  `(total,total)` BEFORE the first `on_message_progress` tick; the phase order
  is respected.

**CLI unit**:
- `_run_reindex_with_progress` creates the analyze task and updates it (mocked
  Progress/index_func); analyze ticks arrive before message ticks.

## Compatibility / rollback
Fully additive: a new optional parameter + a new config field with a
default. All existing calls without a callback work identically. Rollback —
a trivial revert. No data is touched.
