# DESIGN — Fix: single source of truth for JSONL line counting
# (`U+2028` infinite-reindex on agent start)

## Status
Proposed — awaiting review.

## Symptom (user repro)
`deep-worker --restart <agent>` does not appear in the web UI for "minutes":
the process is alive but stuck BEFORE the RPyC server comes up and registers
in the registry, burning 300–370% CPU in BERT embedding. `py-spy` shows:

```
index_all_history_files -> index_history_file -> get_text_embedding_batch
  -> SentenceTransformer.encode -> BERT.forward   (create_cli_agent, before RPyC)
```

Moreover, `--reindex-history` had just been run manually — the index was
already built, no reindexing was needed.

## Root cause — TWO different line counters for one file

`.history_index/line_hashes.json` for `<agent>` = **8029**, while the jsonl
physically has **8031** lines. The discrepancy is constant (+2) and is caused
by the file being counted **in two incompatible ways**:

| Who | Method | Separator | Result |
|---|---|---|---|
| startup cheap-check (`agent._count_pending_history_lines`) | `sum(1 for _ in open())` | `\n` only | **8031** |
| indexer (`history_indexer._read_jsonl_lines`) | `str.splitlines()` | ~8 unicode breaks | 8039 → after dropping unparsed fragments → **8029** |

`str.splitlines()` splits a string not only by `\n`, but also by `\r`, `\v`,
`\f`, `\x1c–\x1e`, `U+0085`, **`U+2028` (LINE SEPARATOR)**, `U+2029`. Two
history records (`<agent>`, lines 2380 and 2436) contain 8 `U+2028` characters
inside JSON values (legitimate content). `splitlines()` cuts those 2 valid
JSON records into fragments, `json.loads` chokes on the halves → the indexer
undercounts by 2 records (8029 instead of 8031).

### How this leads to infinite reindexing
1. cheap-check: `8031 (raw) − 8029 (recorded) = 2` → `needs_catchup=True`.
2. `_pending_chunks_for_file`:
   ```python
   current_line_count = sum(1 for _ in open())     # 8031
   lines_read = 8031 − 8029 = 2                     # > 0, continue
   all_entries = _read_jsonl_lines(...)             # 8029 (splitlines!)
   total_entries = 8029
   if last_line_count(8029) > 0 and total_entries(8029) > last_line_count(8029):
       # 8029 > 8029 == False  ->  else:
       entries = all_entries                        # the WHOLE file (8029 records!)
   ```
   → chunking + embedding of all 8029 records = **10065 chunks, 13.7 MB** through
   BERT on CPU — those very minutes.
3. At the end it writes `line_counts[file] = total_entries = 8029` again → the
   cheap-check on the next start sees `8031 vs 8029 = 2` again → **repeats forever**.

The root cause of the infinite loop is exactly the counter DESYNC (splitlines vs
`\n`), NOT the `>` comparison. The early `return` at `lines_read <= 0` (line ~440)
already correctly handles the "nothing changed" case, so `>` alone did not cause
an infinite re-index. Separately there is a degenerate case
`total_entries == last_line_count` with `current > last` (the file has
unparseable lines): there `>` would go into a full re-index, while `>=` is safer —
it takes only the tail. So `>`→`>=` is kept as a correctness strengthening, but it
is NOT the root cause of the symptom.

## Fix design

User requirements:
1. A single counting method — one source of truth.
2. It must be correct.
3. If the count hasn't changed — no reindexing.
4. Don't break the valid databases of other agents (at worst, carefully
   adjust their configs).

The user is explicitly against preliminary parsing for counting ("a `*.jsonl`
file — just count the lines correctly"). Therefore:

### Core: one cheap `\n`-based counter
A new public helper `history_indexer.count_jsonl_lines(path) -> int`:
counts **non-empty physical lines separated by `\n`** (NO JSON parsing).
Cheap even for large files. A record with `U+2028` inside a value is exactly
one line.

```python
JSONL_RECORD_SEPARATOR = "\n"
def _iter_jsonl_lines(text): return text.split(JSONL_RECORD_SEPARATOR)
def count_jsonl_lines(path):
    try: text = path.read_text("utf-8", errors="replace")
    except OSError: return 0
    return sum(1 for l in _iter_jsonl_lines(text) if l.strip())
```

### Parser fix
`_read_jsonl_lines` is also switched from `str.splitlines()` to splitting on
`\n` (`_iter_jsonl_lines`). Then `U+2028` stays INSIDE the value, `json.loads`
parses the record whole → the number of parsed records matches
`count_jsonl_lines`.

### Single source of truth at all three call sites
- `agent._count_pending_history_lines` (startup) — via `count_jsonl_lines`.
- `history_indexer._pending_chunks_for_file` — `current_line_count` via
  `count_jsonl_lines`.
- `history_indexer.index_all_history_files` (progress denominator) — via
  `count_jsonl_lines`.

### No-op on equality (strengthening, not the root cause)
No-op on "nothing changed" is ensured by the early `return` at
`lines_read <= 0` (it existed before the fix) — after the counters agree,
`lines_read` is now honestly 0. Additionally `>` → `>=` in the incremental
slice selection: a correctness strengthening for the degenerate case
`total_entries == last_line_count` (a file with unparseable lines) — take the
tail instead of a full re-index. The slice `all_entries[last_line_count:]` is
correct because the writer and the cheap-check now count in the same unit.

## Compatibility with other agents (requirement #4)

Checked on all 13 agents with real `line_hashes.json` (audit script
`count_jsonl_lines` vs recorded):

* **11 of 13: delta = 0** → after the fix cheap-check == recorded → **no
  reindexing**. They had no `U+2028`/`\r` — both counters already agreed.
* `<agent>`: delta **+2** — the fix's target. One-time catch-up of 2
  genuinely new tail records → writes 8031 → converges (pending=0).
* `<agent>`: delta **+3669** — NOT related to the fix: `line_hashes` was stale
  (30-05) while the jsonl grew until 23-06; recorded=383, actual=4052 with both
  counters (`U+2028` absent there). Such staleness would cause a catch-up in
  both old and new code; the fix only makes it correctly incremental (tail 3669,
  not the whole file).

No valid database is at risk of a spurious full re-index. No config edits were
needed.

## Scope, NOT touched (deliberately)
`long_text_embedding.py` contains a DOUBLE of `count_jsonl_lines`/`_read_jsonl` with
the same `splitlines()`. But the message-embedding-store subsystem uses THIS
counter symmetrically on both sides of the comparison (and `message_search`
intentionally walks `splitlines()` for physical line numbers under the frozen
store's watermark), so it has no desync like history's. Changing it here is a
risk of shifting the watermark semantics and breaking other agents (exactly
what #4 warns against). Marked as a separate latent issue, out of scope of
this fix.

## Test plan (TDD)
RED first (function missing / bug present), then GREEN.

**semantic_search unit:**
- `test_history_indexer_line_count_source_of_truth.py` — `count_jsonl_lines`
  counts records by `\n`; `U+2028` does not inflate the count; consistent with
  `_read_jsonl_lines`; empty/missing file; `_read_jsonl_lines` no longer
  splits a record by `U+2028`, but still drops genuinely broken lines.
- `test_history_indexer_incremental_noop.py` — equality = no-op (0 chunks,
  lines_read==0), growth = only the tail, shrink = no-op at this layer (no
  spurious re-index).

**semantic_search integration (real e5 embedder):**
- `test_history_indexer_integration.py::test_second_index_is_noop_with_u2028_records`
  — a file with `U+2028`, second pass `files_processed==0, chunks_indexed==0`.

**CLI unit:**
- `test_history_index_catchup_on_start.py::test_no_catchup_when_records_contain_u2028`
  — a fully indexed file with `U+2028` → pending==0.

## Backward compatibility / rollback
Additive. Existing `line_hashes.json` written with the old `splitlines`-
count, where it matched the `\n`-count (11/13), keep working without a re-index.
Where it didn't match — a one-time correct incremental catch-up. Rollback —
a trivial revert. Data (jsonl, LanceDB) is untouched.
