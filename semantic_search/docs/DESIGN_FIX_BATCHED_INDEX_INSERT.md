# DESIGN — Fix: batch node inserts into LanceDB (kill the per-node anti-pattern)

## Status
Implemented.

## Symptom
`deep-worker --reindex-history <agent>` crawled at ~0.8 chunks/sec; for ~10,065
chunks that is >2 hours. The progress bar "stuck" on every line.

## Diagnostics (py-spy, empirical)
- 12/12 samples of the main thread — in `lancedb/background_loop.py → table.add`
  (waiting for a write), NOT in embedding/torch.
- Baseline embedding e5-small on these cores: ~15 emb/sec. So embedding is not
  the bottleneck — the **per-chunk lancedb write call** was.
- On disk: `vectors.lance/_transactions/*.txn` and `data/` fragments = **one
  per chunk** (4651 at ~4700 inserted). Classic anti-pattern:
  one-by-one writes into an append-only columnar DB. Each `add()` creates a new
  data fragment + a manifest version + fsync; the cost grows with the table
  size → degradation to ~1/sec.

## Root cause
`index_history_file`:
```python
for doc in documents:
    index.insert(doc)     # 1 Document -> 1 insert_nodes -> 1 lancedb add -> 1 txn
```

## Fix
1. `IndexConfig.index_batch_size: int = 1000` — a single batch-size config.
2. Chunks are ALREADY final (we cut them ourselves in `_chunk_messages`), so we
   build `TextNode` directly (1 node = 1 chunk) — without a redundant
   SentenceSplitter pass and without the private `index._transformations`.
3. Insert in batches: `for start in range(0, n, batch): index.insert_nodes(
   nodes[start:start+batch])`. Each batch = ONE lancedb transaction →
   `ceil(n / batch_size)` writes instead of `n`.
4. Progress `on_chunk_progress(done, total)` is reported PER BATCH (done grows
   by the batch size), final `done == total == chunks_indexed`.

## Result (measurement)
- **10.4–11.6 chunks/sec** (was 0.8) — ~14× speedup; now bounded by
  embedding, as it should be.
- **1 lancedb transaction** per 300 chunks (was 300).
- `<agent>` (~10k chunks): ~2 h → **~15 min**.
- Index search verified: the marker chunk is found in top-3 (nodes are not
  lost in the Document→TextNode + batching transition).

## Contract / architecture
Unchanged: the same node metadata/IDs (`<jsonl>:L<range>`), the same vector store,
docstore, incremental by `line_counts`. The public indexer API is extended with
an optional `index_batch_size` (via IndexConfig) — fully backward compatible
(default 1000). No new threads/polling.

## Tests (TDD, real embed model, no mocks)
- `TestBatchedInsert`: the number of `.txn` == `ceil(chunks/batch)` (not `chunks`);
  a small batch → several batches and strictly < one-per-chunk; completeness (index
  exists, all chunks); progress monotonic and final == total.
- Updated `TestChunkLevelProgress`: progress is now per-batch (was per-chunk).

---

## Follow-up: progress in MESSAGES, decoupled from the write batch

Two UX defects remained after the initial batching fix:
- `0/None` / `0/?` at start — the denominator was the CHUNK count, only known
  after the slow (~87 s on <agent>) chunking pass, so the bar had no
  total and no ETA until then. A second pre-count pass doubled that cost.
- Progress jumped in steps of `index_batch_size` (1000) — the write batch size
  was wrongly used as the progress granularity. They are DIFFERENT axes.

### Fix
- **Progress unit = messages (jsonl lines)**, not chunks. The denominator is a
  plain line count — known up front (instant), so the bar shows `X/N msgs`,
  `%`, elapsed, and **ETA** from the first tick. "How far" maps back to
  messages via each chunk's `line_range` end.
- **Removed the chunk pre-count pass** (`_count_pending_chunks`) — no more
  duplicated ~80 s chunking; `total` comes from a cheap line count.
- **Decoupled the two knobs**:
  - `IndexConfig.embed_progress_step` (default 64) — PROGRESS granularity:
    nodes are embedded in slices of this size, ticking progress after each.
  - `IndexConfig.index_batch_size` (default 1000) — WRITE batch: nodes (already
    carrying embeddings) are written to LanceDB in these batches, one
    transaction each. Fine progress does NOT create more transactions.
- Two-phase per file: (1) embed in progress slices → set `node.embedding`;
  (2) `insert_nodes` in write batches (no re-embed, since `.embedding` is set).
- CLI bar columns: spinner, description, bar, `%`, `X/N msgs`, elapsed, `< ETA`.

### Tests
- `TestMessageProgress`: total == message count and constant from the first
  tick (never None); done monotonic 0..N; progress step independent of write
  batch (step=1 + batch=1000 → many ticks but a single LanceDB transaction);
  aggregation across files; None-callback backward compatible.
- `TestBatchedWrite`: txn count == ceil(chunks / index_batch_size); completeness.
