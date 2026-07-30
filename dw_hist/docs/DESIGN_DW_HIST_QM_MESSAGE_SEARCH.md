# dw-hist `-qm` — per-message embedding search + `--flag` filter

Status: design (proposed)
Scope: `dw_hist/` (CLI surface) + a small read-only query helper in
`semantic_search/` (loads the existing per-message embedding store and ranks).
No new dependencies, no new threads, no embedding recomputation, no schema
change to the store.

## 1. Motivation

`dw-hist <agent> -q <query>` searches the **chunked LanceDB history index**
(`history_retriever.search_history`): messages are merged/split into retrieval
chunks and ranked by the index's L2→`exp(-d)` score.

Some agents additionally have a **per-message embedding store** built by
`semantic_search.message_embedding_store` (`{summaries_dir}/.message_embeddings/`:
`embeddings.npy (N,dim)`, `flags.npy (N,)`, `message_indices.npy (N,)`,
`token_lengths.npy (N,)`, `manifest.json`). There each row is a single message,
already L2-normalised, with a `MessageFlag` bitmask.

`-qm` (q-message) is a second, complementary search mode over **that** store:

* embed the query once (`embed_long_text`, normalised),
* score every stored row by the **dot product** (== cosine, both sides are
  unit vectors) — this is the query's projection onto each message embedding,
* return the top-k rows by that projection (the projection IS the score),
* render them like `-q` does, plus the row's set flags,
* optionally restrict the scored set to rows that have **at least one** of a
  set of operator-named flags (`--flag`/`-f`, repeatable, logical OR).

This is exact (full inner product over all rows), not ANN — N is at most a few
thousand per agent, so a single vectorised matmul is instant.

## 2. CLI surface

### 2.1 New / changed flags

| flag | type | meaning |
|------|------|---------|
| `-qm`, `--query-messages` | str | message-embedding search query (mutually exclusive with `-q`) |
| `-f`, `--flag` | str, repeatable | restrict scoring to rows with ≥1 of these flags (OR). Only valid with `-qm`. |

`-q` / `-qm` are mutually exclusive; exactly one is required when an agent name
is given (same as today's "agent requires `-q`" rule, generalised). `-k` /
`--format` / `-v` apply to `-qm` unchanged.

**RC-2 (empty query).** `embed_long_text("")` / whitespace-only returns a
ZERO vector (not unit-norm), which would make every dot product 0.0 and yield k
arbitrary rows with a meaningless `score=0.0` "success". `-qm` therefore rejects
an empty / whitespace-only query as a usage error (exit 2) BEFORE loading the
model or the store. Verified by a unit test.

`--flag` values are the `MessageFlag` names, case-insensitive:
`USER_ROLE`, `TASK_MANAGER`, `PLAIN_USER`, `AGENT`, `AGENT_TO_AGENT`,
`TOOL_RESULT`, `TOOL_CALL`, `SYSTEM`, `SUMMARIZATION`, `AGENT_FINAL`.
An unknown flag name → usage error (exit 2) listing the valid names.
(`TOOL_CALL` is accepted but currently matches nothing — it is reserved/always 0;
that yields an empty result set, which is correct, not an error.)

### 2.2 Examples

```bash
dw-hist <agent> -qm "deadlock on restart"
dw-hist <agent> -qm "what did the user ask about flags" -k 10
dw-hist <agent> -qm "summary of the session" -f SUMMARIZATION
dw-hist <agent> -qm "my own questions" -f PLAIN_USER -f TASK_MANAGER --format json
```

### 2.3 Exit codes (reuse existing taxonomy)

| Code | `-qm` meaning |
|------|---------------|
| 0 | success (including zero results after a flag filter) |
| 2 | usage error: `-q`+`-qm` together, `-f` without `-qm`, bad `-k`, unknown agent, unknown flag name |
| 3 | agent has no **per-message embedding store** (distinct message from the `-q` "no history index") |
| 4 | runtime error (load/encode/shape failure) |

Exit 3 text guides the operator to build the store:
`No per-message embedding store for agent '<a>'. Build it with
semantic_search.message_embedding_store.build_message_embeddings(...)` (and
`build_message_metadata(...)` for flags).

## 3. Ranking semantics (`semantic_search`, new helper)

New module `semantic_search/message_search.py`, public function:

```python
@dataclass
class MessageSearchResult:
    content: str          # role-prefixed message text (same extraction as the store)
    score: float          # query·row dot product (cosine; in [-1, 1], ~[0,1] in practice)
    thread_id: str        # JSONL file stem
    line_range: str       # 1-based line number of the entry in its file (e.g. "42")
    timestamp: str        # entry "summarized_at"
    message_types: str    # entry "message_type" (human/ai/tool/system)
    file_path: str        # absolute path to the source JSONL
    flags: int            # MessageFlag bitmask for this row
    flag_names: list[str] # decoded set-flag names, stable order

def search_messages(
    *, query: str, summaries_dir: str | Path, top_k: int,
    flags_any: int = 0, embed_model: Any | None = None,
    config: IndexConfig | None = None,
) -> list[MessageSearchResult]: ...
```

Algorithm (all vectorised, numpy; torch never needed):

1. `emb, _lens = load_message_embeddings(summaries_dir, as_torch=False)` → `(N,dim)`.
   `idx, flags = load_message_metadata(summaries_dir, as_torch=False)` → `(N,)`,`(N,)`.
   Both raise `FileNotFoundError` if absent → CLI maps to exit 3.
   Assert `emb.shape[0] == flags.shape[0]` (the store's own 1-to-1 invariant);
   mismatch → `RuntimeError` (exit 4) telling the operator to rebuild metadata.
2. `q = embed_long_text(query, embed_model, config)` → normalised `(dim,)`.
   (Reuses the project embedder; same vector space as the store.)
3. Candidate mask: if `flags_any != 0`, `mask = (flags & flags_any) != 0`, else
   all rows. Apply the mask to the embedding matrix and remember the original
   row indices (`np.flatnonzero(mask)`), so results map back to the right rows.
   Empty candidate set → return `[]` (exit 0).
4. `scores = candidate_emb @ q` → `(M,)` dot products. Because both operands are
   unit-norm, this is cosine similarity and equals the requested projection.
5. Top-k by score **without a full sort**: with `M` candidates and `k_eff =
   min(top_k, M)` (NIT N3 — `argpartition(scores, -k)` raises when `k > M`, which
   happens easily after a `-f` filter), `np.argpartition(scores, -k_eff)` then
   sort just those `k_eff` descending (`O(M + k_eff log k_eff)`), tie-break by
   ascending row index for determinism. `M == 0` → `[]`.
6. For each selected row, recover its source message. The store's row order is
   *file-sorted, then embeddable-entry order* (exactly `build_message_metadata`).
   We rebuild that mapping once per call by walking the same sorted `*.jsonl`
   and recording, per embeddable entry, the
   `(file_path, stem, physical 1-based line, summarized_at, message_type, text)`.
   We then index that list by row to fill the result. This re-derivation reads
   the JSONL (cheap text parse, no embedding) and is verified to have exactly
   `N` entries (else `RuntimeError` → exit 4: store/history drift, rebuild).

   **RC-1 (physical line number).** `line_range` is the message's PHYSICAL
   1-based line in its `.jsonl`. The shared `_read_jsonl` (and thus
   `read_jsonl_entries` / `message_entries_from_file`) silently drops blank and
   undecodable lines and returns a *compacted* list, so a compacted-list index
   is NOT the physical line. Therefore the re-walk does its OWN pass over
   `path.read_text().splitlines()` keeping a physical line counter, parses each
   line itself (mirroring `_read_jsonl`'s skip rules so the *selection/order*
   stays identical to the store), and emits the physical counter as the line.
   Verified by a unit test with a blank line and an undecodable line before the
   target entry.

   **Mapping rationale (NIT).** `build_message_metadata` flattens the raw entry
   stream ACROSS files to give `classify_message` a cross-file look-ahead for
   `AGENT_FINAL`. This module needs only POSITIONAL data (file/line/text) and
   takes flags from `flags.npy` (never recomputed), so a per-file walk is
   equivalent for selection/order and is used here deliberately — a code comment
   records this so a future reader does not "fix" it toward a cross-file flatten.

`flag_names` decodes the bitmask via the same `MessageFlag` name list
`build_message_metadata` tallies, in that fixed order.

### 3.1 Efficiency notes

* One `embed_long_text` call (the only model work).
* One matmul `(M,dim)·(dim,)`; `argpartition` for top-k (no full sort).
* `embeddings.npy` is `mmap_mode="r"` loadable; we slice the candidate rows.
  For the common unfiltered case we matmul the whole `(N,dim)` — N≈7k, dim=384,
  ≈2.7M flops, sub-millisecond.
* JSONL re-walk for row→message is the same parse `build_message_metadata`
  already does and runs in well under a second for thousands of messages.

## 4. dw-hist integration

* `_build_parser`: add `-qm/--query-messages` and `-f/--flag` (`action="append"`).
* `main`: validation
  * `-q` and `-qm` mutually exclusive → exit 2.
  * agent given but neither `-q` nor `-qm` → exit 2 (message updated).
  * `-f` present without `-qm` → exit 2.
  * `-qm`/`-f` without an agent → exit 2 (same shape as `-q` today).
* New `_cmd_search_messages(...)` parallel to `_cmd_search`:
  * validate `-k` (1..20) and agent root (reuse helpers),
  * resolve flag names → bitmask (unknown → exit 2),
  * pre-check store presence via
    `message_embedding_store.message_embeddings_exist` AND
    `message_metadata_exist` (flags required for `-f` and for the `flags` column);
    absent → exit 3,
  * `_force_offline_quiet_env()` + lazy import `search_messages`, run inside the
    same stdout/stderr `/dev/null` redirect (unless `--verbose`),
  * map `FileNotFoundError`→exit 3, other `Exception`→exit 4.
* Lazy loader `_load_search_messages()` mirrors `_load_search_history()`.

## 5. Output format

Reuse `search.format_text` / `search.format_json` extended to carry `flags`:

* Text: append a `Flags:` segment to each result's header line, e.g.
  `--- Result 1 (score=0.842) ---` then
  `Thread: <agent>` / `Lines: 1187 | When: <ts> | Types: human | Flags: USER_ROLE,TASK_MANAGER`.
  The flags segment is omitted when the result object has no `flags` attribute
  (so the existing `-q` output is byte-unchanged — back-compat via duck typing).
* JSON: add `"flags": <int>` and `"flag_names": [...]` to each record when
  present. The `-q` path keeps emitting its current keys (no `flags` field),
  so existing scripts are unaffected.

`format_text`/`format_json` stay duck-typed on a `Protocol`; we widen the
protocol with **optional** `flags`/`flag_names` and branch on `hasattr`.

## 6. Testing

Unit (no model; inject a fake embedder / stub `search_messages`):

* `message_search.search_messages`:
  * ranking: hand-built tiny store (write `.npy`s with known unit rows + a fake
    embed_model whose `_model.encode` returns a chosen query vector) → asserts
    the row with the largest dot product ranks first, score == that dot product.
  * `flags_any` OR filter: only rows with a matching bit are scored; others
    never appear even if more similar.
  * empty candidate set → `[]`.
  * row→message mapping: correct text/line/type/thread for the top row.
  * store/metadata absent → `FileNotFoundError`; N-mismatch → `RuntimeError`.
* CLI (`test_cli.py`, stub `search_messages`):
  * `-q`+`-qm` → exit 2; `-f` without `-qm` → exit 2; unknown flag → exit 2;
  * `-qm` without store → exit 3 (stub `home` with agent root but no store);
  * happy path text + json shapes include flags;
  * `-k` bounds reuse existing tests.

Integration (real model, real store — by request "check with real calls"):
run `dw-hist <agent> -qm ...` with and without `-f`, in text and json, and
eyeball that the top hits are on-topic and flags are correct. Not a pytest
(needs the model + a built store), executed manually and reported.

## 7. Out of scope

* No ANN index for messages (exact matmul is fast enough at this N).
* No writing/rebuilding the store from dw-hist (read-only tool; build stays in
  `semantic_search`).
* e5 `query:` prefix is intentionally NOT added, to match how the store rows
  were embedded (no prefix) — consistent with the existing `-q` limitation
  noted in `history_retriever`.
