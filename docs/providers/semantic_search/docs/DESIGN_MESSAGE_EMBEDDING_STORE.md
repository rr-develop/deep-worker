# Design: per-message embeddings + incremental persistent store

Status: implemented (dev). Scope: `semantic_search/` only. No architecture or
contract changes to existing search/index APIs. Existing agents are never
touched.

## 1. Problem

The semantic-search index (`.history_index/`, LanceDB) stores embeddings for
**retrieval chunks** — adjacent messages merged and long messages split to fit
the embedder window. That is the right granularity for search, but it is **not**
a 1-to-1 map to messages, and it does not expose the embeddings as plain
tensors for downstream numerical work (clustering, drift/diff over time,
dedup, MMR, similarity matrices, etc.).

We need:

1. A way to embed text of **arbitrary length** (including > 512 tokens, the
   XLM-RoBERTa / e5 hard window) without silently dropping the tail.
2. **One embedding per message** (strict 1-to-1 with the history), plus each
   message's token length, as two tensors.
3. A **persistent, incrementally-extendable** store of (2) next to the existing
   search data, so the ~10-minute full pass is paid once and later runs only
   embed newly-appended messages (by request; no automatic background job).
4. No eager heavy-dependency imports in the agent's code path: `numpy`/`torch`/
   the embedder load only when these functions are actually called.

## 2. Embedding semantics

### 2.1 Arbitrary-length text — `embed_long_text`

Effective ceiling = `min(child_chunk_size, model_max_length - special_tokens)`
= **510** for e5-small (512 − `<s>`/`</s>`), reusing
`_tokenizer.embed_model_tokenizer` / `effective_max_tokens` so the ceiling and
vector space match the search index exactly (see
`DESIGN_FIX_TOKENIZER_MISMATCH.md`).

* Text ≤ 510 tokens → embedded directly.
* Longer text → split into **token-accurate windows** (slice raw token ids into
  ≤510 windows, decode each window back to text), embed each window, then
  combine as a **token-count-weighted, L2-normalised mean**:

  ```
  emb = normalize( w1*emb1 + w2*emb2 + ... )    # wi = #tokens in window i
  ```

  No tail is silently truncated; every token contributes. Window embeddings
  are L2-normalised by the SentenceTransformer; the weighted sum is
  re-normalised so the result is a unit vector in the same space.

### 2.2 Per-message embeddings — `embed_history_messages`

Message text is extracted **exactly** like
`history_indexer._chunk_messages` (role prefix `User:/Assistant:/Tool:` +
multi-modal text-part join), so embedded text is byte-identical to what the
search index was built from. Empty-content messages are skipped (they have no
text to embed); "message" therefore means *non-empty message*.

All windows across all messages are embedded in **one batched
`SentenceTransformer.encode` pass**; per-message pooling is a vectorized
`torch.index_add_` (scatter-add) followed by row normalisation. Output:

* `embeddings` — `float32 (N, dim)`, rows L2-normalised.
* `token_lengths` — `int64 (N,)`, full token count per message.

`N` = number of non-empty messages, in **file-sorted then line order** —
deterministic and append-only friendly.

The shared core `embed_message_texts(texts, st, eff_max, …) -> (np, np)` is used
by both the one-shot function and the persistent store, so there is a single
implementation of the windowing + weighted pooling.

## 3. Persistent store

Directory: `summaries_dir/.message_embeddings/` (mode `0700`, sibling of
`.history_index/`).

| File | Content |
|------|---------|
| `embeddings.npy` | `float32 (N, dim)` |
| `token_lengths.npy` | `int64 (N,)` |
| `message_indices.npy` | `int64 (N,)` — optional metadata, see §4 |
| `flags.npy` | `int64 (N,)` — optional metadata, see §4 |
| `manifest.json` | `schema_version`, `embedding_model`, `dim`, `total_messages`, `file_entry_counts`, `has_message_indices`, `has_flags` |

`file_entry_counts` maps each `*.jsonl` path to the number of **parsed entries**
already consumed (parsed, not physical lines, so blank/corrupt lines don't
shift the watermark). The store rows are append-only and ordered the same way
`embed_history_messages` orders them, so a row index is stable across updates.

### 3.1 Incremental update — `build_message_embeddings`

1. For each `*.jsonl`, compute current parsed-entry count; the delta beyond the
   stored watermark is the set of new entries.
2. Extract non-empty texts from the delta, embed them with the shared core, and
   **append** the new rows to `embeddings.npy` / `token_lengths.npy`.
3. Update `manifest.json` (new counts, new total).

Files are append-only, so this is correct. Writes go through a temp file +
`replace()` for crash-safety.

**Rebuild-from-scratch fallback** (loud, explicit — never a silent mismatch)
triggers when:

* `reset=True`, or
* the store doesn't exist / manifest is unreadable, or
* `schema_version` or `embedding_model` differs from the current config, or
* any tracked file **shrank** (count < stored watermark) — i.e. history was
  rewritten rather than appended.

`MessageEmbeddingStats.rebuilt` reports which path was taken.

### 3.2 Loading — `load_message_embeddings`

Reads the two `.npy` files only (no embedder, no torch unless
`as_torch=True`). This is the fast path that replaces the ~10-minute compute
with a sub-second load.

### 3.3 Per-message metadata tensors — `build_message_metadata` / `load_message_metadata`

Two **additional** tensors describe the already-computed embedding rows, built
by a separate, cheap pass that **never recomputes embeddings**:

* `message_indices.npy` — `int64 (N,)`. The global 0-based ordinal of the source
  message (file-sorted, then parsed-entry order) for each embedding row. This is
  the canonical, unique, append-only-stable **message id** that lets you map any
  embedding back to its message.

  *Why the ordinal, not the JSONL `message_index` field?* In the real history
  the `message_index` field is **not unique** (it repeats — e.g. 2417 distinct
  values across 8266 entries), so it cannot identify a row. The parsed-entry
  ordinal is unique, contiguous `0..N-1`, and matches the embedding build order
  exactly.

* `flags.npy` — `int64 (N,)`. A :class:`MessageFlag` **bitmask** per row (a
  message can set several bits). Bits:

  | bit | flag | meaning |
  |-----|------|---------|
  | 0 | `USER_ROLE` | `human`-role message (but never a tool result) |
  | 1 | `TASK_MANAGER` | `human` with a `TaskManager:>` prefix |
  | 2 | `PLAIN_USER` | `human` with no known prefix (genuine end-user) |
  | 3 | `AGENT` | `ai`-role (final or intermediate, incl. with tool calls) |
  | 4 | `AGENT_TO_AGENT` | carries an inter-agent `<a> -> <b> :>` prefix |
  | 5 | `TOOL_RESULT` | `tool`-role message (a tool's returned result) |
  | 6 | `TOOL_CALL` | the embedding is of a tool call itself (reserved; 0 — see below) |
  | 7 | `SYSTEM` | `system`-role, or a `System:>`-prefixed user message |
  | 8 | `SUMMARIZATION` | a condensed-summary message (see below) |
  | 9 | `AGENT_FINAL` | an `ai` message whose next message is a user/TaskManager turn |

  **1-to-1 correspondence.** The metadata pass selects exactly the messages the
  embedding pass selected (same non-empty `_message_text` criterion, same file
  and entry order). It asserts its row count equals the embedding row count and
  raises otherwise, so `message_indices[i]` / `flags[i]` always describe
  `embeddings[i]`. This is the correspondence the requester asked for, over the
  embeddings that exist (no recomputation).

  **`SUMMARIZATION` detection.** A condensed-summary message is an `ai`-role
  message whose body starts with the summary template heading
  `## SESSION INTENT` (emitted by the deepagents summarization middleware).
  Why content, not a structural marker:
  * `additional_kwargs.lc_source == "summarization"` (set by the middleware on
    its delivery `HumanMessage`) is **not persisted** into the offloaded JSONL —
    0 occurrences in stored history.
  * The wrapper text `A condensed summary follows` / `<summary>` appears **only**
    inside unrelated `tool` outputs that quoted prior history (system-prompt
    dumps, context-extraction prompts) — false positives, not summaries.
  * `## SESSION INTENT` is unique to the real summary bodies (4 on this agent's
    history; no other `ai` message uses that heading). If the template heading
    changes in a future deepagents version, this detector must be extended.
  A `SUMMARIZATION` row is also `AGENT` (it is an `ai` message).

  **`AGENT_FINAL` detection.** Set on an `ai` message whose **raw next** history
  entry (parsed-entry order, across the sorted file sequence — including
  non-embeddable entries) is `human`-role. This marks the agent's last message
  of a turn, i.e. it handed control back to the user / TaskManager. `ai`
  messages followed by a `tool` result (mid tool-loop) or by `system` are not
  final. The look-ahead uses the raw stream, so a text-less (non-embedded)
  tool-call-only `ai` that follows another `ai` correctly prevents the latter
  from being marked final. Built via `classify_message(entry, next_entry=…)`.

  **Reserved bit (0 on current data, documented for stability):**
  * `TOOL_CALL` — means "this embedding **is** of a tool call itself". The store
    does not (yet) build separate embeddings for tool calls — only for the
    agent's accompanying comment text when present — so this bit is **always 0**
    on current data. It is deliberately NOT used to annotate an `AGENT` comment
    that merely carries a tool call, because that embedding is of the comment,
    not of the call (annotating it would mislabel a comment embedding as a
    tool-call one). The bit position is reserved for when standalone tool-call
    embeddings are introduced.

**Staleness / invalidation.** Because metadata describes specific embedding
rows, any change to the embedding rows (`build_message_embeddings` appending or
rebuilding) **deletes** `message_indices.npy` / `flags.npy`. They must then be
rebuilt with `build_message_metadata` against the new rows — never left stale.

`load_message_metadata` reads the two `.npy` files only (numpy; torch only when
`as_torch=True`).

## 4. Lazy dependencies

`numpy`/`torch`/`HuggingFaceEmbedding` are imported **inside** the functions
that use them. Importing `semantic_search.long_text_embedding` or
`semantic_search.message_embedding_store` does not pull `torch`. (The
`semantic_search` package `__init__` already imports `numpy` transitively via
the indexer/retriever; that is pre-existing and unchanged.) `load_message_
embeddings(as_torch=False)` avoids torch entirely.

## 5. Naming note (pyright strict)

`resolve_sentence_transformer`, `embed_message_texts`,
`message_texts_from_file`, `count_jsonl_lines` are public (no leading
underscore) because they are imported across modules within the package and
pyright strict flags underscore-prefixed cross-module imports as
`reportPrivateUsage`. Same convention as `_tokenizer.embed_model_tokenizer`
(see `DESIGN_FIX_TOKENIZER_MISMATCH.md` §4.1). The store helpers used only
inside their own module keep the underscore.

## 6. Measured (one real agent history, e5-small, CPU 4 threads)

* Full build: ~6317 messages, ~2.25M tokens, ~10 min (one-time).
* 987 messages exceeded 510 tokens and went through the weighted-mean path
  (largest single message ≈ 25k tokens).
* Reload via `load_message_embeddings`: sub-second.
* Incremental no-op (history unchanged): seconds (parse + manifest compare, no
  embedding).
* `build_message_metadata` (indices + flags, **no embedding recompute**): ~0.55s
  for 6317 rows. Verified `message_indices` is contiguous `0..N-1` and row count
  matches `embeddings` exactly. Flag distribution:
  `USER_ROLE=1153` (`TASK_MANAGER=1051` + `PLAIN_USER=80` + `AGENT_TO_AGENT` on
  human=22), `AGENT=2225`, `AGENT_FINAL=1119`, `TOOL_CALL=0` (reserved — see
  §3; never annotated onto an `AGENT` comment), `TOOL_RESULT=2929`, `SYSTEM=10`,
  `AGENT_TO_AGENT=37` (22 human-transport + 15 system "incoming-from-agent"
  notices), `SUMMARIZATION=4`. `SUMMARIZATION` and `AGENT_FINAL` are both
  subsets of `AGENT`. The 1786 text-less "tool-call-only"
  AI messages produced no embedding (empty text) and so are not rows — which is
  why `N=6317 < 8266` total entries.

## 7. Out of scope (future, by request only)

* Automatic/triggered rebuild on summarisation (explicitly deferred — manual
  `build_message_embeddings` only for now).
* Optional e5 `passage:` input prefix (would diverge from the current index;
  needs a coordinated re-index).
* A `dw-hist`/CLI surface over these tensors.
