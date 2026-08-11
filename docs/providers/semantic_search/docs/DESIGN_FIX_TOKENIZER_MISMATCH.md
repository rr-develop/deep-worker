# Fix: align chunker tokenizer with the embedding model's tokenizer

**Status:** DRAFT (awaiting design review)
**Worktree / branch:** the project worktree on `dev`
**Scope:** `semantic_search/` (both `history_indexer.py` and `indexer.py`)
**Date:** 2026-06-30

---

## 1. Problem (the bug)

The chunkers in `semantic_search/` measure chunk size with **one** tokenizer and
the embedder encodes the chunk with a **different** tokenizer. Concretely:

| Stage | Tokenizer | Limit used |
|-------|-----------|------------|
| `_chunk_messages` (history) | `llama_index.core.utils.get_tokenizer()` = `tiktoken.cl100k_base` (`gpt-3.5-turbo` BPE) | `config.child_chunk_size` = 1024 |
| `_merge_small_chunks` (project code/docs) | same `tiktoken.cl100k_base` | `config.child_chunk_size` = 1024 |
| `HuggingFaceEmbedding(intfloat/multilingual-e5-small)` (both flows) | model's own `XLMRobertaTokenizer` | hard `model_max_length` = 512 (architectural limit of the model — `max_position_embeddings=512` in `config.json`) |

The two tokenizers do **not** agree even on rough order of magnitude.
Measured on this host (`HuggingFaceEmbedding._model.tokenizer` vs `tiktoken`):

| Sample text | tiktoken tokens | XLM-R tokens | Direction of error |
|-------------|-----------------|--------------|--------------------|
| Russian sentence (76 chars) | 29 | 21 | tiktoken **over**-counts |
| English sentence (66 chars) | 13 | 16 | tiktoken **under**-counts |

`tiktoken` is not just "coarser" than XLM-R — it is **systematically wrong in
both directions**, depending on the language.

**Consequence today.** A "1024-token" chunk produced by the chunker carries
anywhere from ~700 to ~1500 XLM-R sub-word tokens. Every chunk above 512
XLM-R tokens is **silently truncated** by `HuggingFaceEmbedding` (`SentenceTransformer`
defaults `truncation=True`): the tail of the chunk is stored in LanceDB's
`content` (and is returned to the user via `HistorySearchResult.content`), but
**the embedding vector is computed only over the first ~512 sub-word tokens**.
Ranking ignores the tail. There is no warning anywhere.

The user explicitly flagged this:

> I believe we must use the same tokenizer as the model.
> I don't understand at all why two different ones are used there.

This document closes that gap.

## 2. Goals

1. **Single source of truth for the tokenizer** — the chunker counts tokens
   with **the same instance** the embedder will use, so "size of the chunk in
   the chunker" and "size of the chunk in the embedder" are the same number.
2. **Hard cap matches the model** — no chunk handed to the embedder ever
   exceeds `tokenizer.model_max_length`. Silent tail-truncation goes away.
3. **No behaviour change for `config.child_chunk_size`** — the user said
   "don't change the chunk sizes for now". The config value stays `1024`. We add a
   technical ceiling `effective_max = min(child_chunk_size, model_max_length)`
   that is applied inside the chunkers. The day we switch to an 8 k-context
   embedder, the ceiling lifts automatically.
4. **Existing indexes keep working.** Old LanceDB tables produced by the
   tiktoken-based chunker stay readable; new lines added to the same JSONL
   are chunked with the new tokenizer (mixed-tokenizer index is fine — vector
   space is the same, only chunk boundaries differ between historical and
   fresh entries).
5. **No new threads, no contract changes, no architecture changes.** Pure
   internal refactor of two helper functions.
6. **Policy compliance.** No agent-initialisation state is touched. The pre-commit hook
   runs strictly (pyright on `semantic_search/`), no `--no-verify`.

## 3. Non-goals

- **Not** changing `child_chunk_size` or any other tunable in `IndexConfig`.
- **Not** switching the embedding model. (8 k-context candidates discussed
  with the user — `nomic-embed-text-v1.5`, `jina-embeddings-v3`,
  `gte-multilingual-base` — are out of scope here; they would require full
  reindex and quality re-calibration.)
- **Not** forcing reindex of existing agents. Operators MAY remove
  `<state_dir>/<agent>/summaries/.history_index/` to get a clean,
  uniformly-tokenized index — but it is opt-in.
- **Not** changing `dw-hist`. The user has not asked for token counts in the
  output yet; if requested, that becomes a follow-up.

## 4. Design

### 4.1 New helper: `embed_model_tokenizer(embed_model) -> (callable, int, int)`

> **Implementation note (post-review).** Two purely-mechanical deviations were
> made to satisfy `pyright --strict` on `semantic_search/`, neither changes the
> contract:
> 1. The helper is named `embed_model_tokenizer` (no leading underscore). It
>    lives in the package-internal module `_tokenizer.py`; the underscore on the
>    *module* marks it private. Pyright strict flags an underscore-prefixed
>    *function* imported across modules as `reportPrivateUsage`/
>    `reportUnusedFunction`, so the name is public within the package.
> 2. Attribute access uses `getattr(embed_model, "_model", None)` /
>    `getattr(inner, "tokenizer", None)` with explicit `RuntimeError` guards
>    (RC-1) instead of the bare `embed_model._model.tokenizer` shown below. The
>    only `# type: ignore[attr-defined]` is on the
>    `tokenizer.num_special_tokens_to_add(pair=False)` call (RC-2).

A single internal helper that takes the resolved `HuggingFaceEmbedding`
instance (the one already cached in `_embedding_cache`) and returns:

- `tokenize_fn`: a callable that returns a token-id list for a string.
  Implementation:
  `embed_model._model.tokenizer(text, add_special_tokens=False)["input_ids"]`.
  We pass `add_special_tokens=False` so the count is what a *fragment* of a
  chunk contributes, not what the encoded sequence will be (`<s>` + body +
  `</s>` adds 2 tokens for XLM-R; the exact number is computed by the
  tokenizer itself and returned as `special_tokens_reserve` below — see
  RC-2 in §4.2).
- `model_max_length`: `embed_model._model.tokenizer.model_max_length`
  (= 512 for e5-small).
- `special_tokens_reserve`: derived from the tokenizer itself via
  `tokenizer.num_special_tokens_to_add(pair=False)` (HF public API). For
  XLM-R this returns 2 (`<s>` + `</s>`); for hypothetical future models
  with different special-token policies (e.g. 0 for some plain BPE
  embedders, 3 for some others) it self-adjusts. We do not hard-code 2.

We touch `embed_model._model.tokenizer` directly. That is the same path the
embedder itself uses; if upstream ever renames it, the embedder breaks at
the same time as the chunker — there is no asymmetric risk. The alternative
(loading a second `AutoTokenizer.from_pretrained(model_name)`) would double
the cold-start cost, double the resident memory, and would still have to
match the embedder's tokenizer config exactly. **Rejected.**

**RC-1 (review-required): no silent fallback to tiktoken.** If
`getattr(embed_model, "_model", None) is None`, or `_model` has no
`tokenizer`, or the tokenizer object does not expose
`model_max_length` / `num_special_tokens_to_add` / a `__call__` returning
`input_ids`, `_embed_model_tokenizer` MUST raise
`RuntimeError("embed_model has no accessible tokenizer; see "
"DESIGN_FIX_TOKENIZER_MISMATCH.md §4.1")`. Falling back to `tiktoken` is
forbidden — it would silently resurrect the exact bug this design closes.
A unit test asserts the `RuntimeError` is raised when handed a fake
embed_model with `_model=None`, and a second test handed an object with
`_model` but no `tokenizer` attribute.

The helper lives in a new module `semantic_search/_tokenizer.py` so both
`history_indexer.py` and `indexer.py` import it (no cyclic risk; both
already depend on the embedder).

### 4.2 Effective ceiling: `effective_max_tokens(...)`

```python
def effective_max_tokens(
    config: IndexConfig,
    model_max_length: int,
    special_tokens_reserve: int,
) -> int:
    return min(
        config.child_chunk_size,
        model_max_length - special_tokens_reserve,
    )
```

For e5-small (`model_max_length=512`, `special_tokens_reserve=2`,
`child_chunk_size=1024`) this yields **510**. For a hypothetical future
8 k embedder with the same special-token policy it yields
`min(1024, 8190) = 1024` — `child_chunk_size` becomes the binding
constraint again, no code change.

**RC-2 (review-required): `special_tokens_reserve` is derived, not
hard-coded.** `_embed_model_tokenizer` reads it from the tokenizer via
`tokenizer.num_special_tokens_to_add(pair=False)` (HF public API). The
chunker only sees an integer — it does not encode the model-family
assumption "always 2" anywhere. A sentinel unit test asserts that for the
real `intfloat/multilingual-e5-small` tokenizer
`num_special_tokens_to_add(pair=False) == 2`, so the "510" we use in
boundary tests stays a consequence of the model, not a coincidence.

If a future tokenizer does NOT expose `num_special_tokens_to_add`
(unlikely — it is on every HF `PreTrainedTokenizerBase` subclass and on
`tokenizers.Tokenizer` wrappers), `_embed_model_tokenizer` falls under
RC-1 and raises — better loud failure than a silently wrong reserve.

### 4.3 Where it plugs in

#### `history_indexer.py::_chunk_messages`

Current (line 157): `tokenizer = get_tokenizer()` (tiktoken). Replace by:

```python
tokenize_fn, model_max_length = _embed_model_tokenizer(embed_model)
max_tokens = effective_max_tokens(config, model_max_length)
```

Where `embed_model` is the same instance threaded through from
`index_history_file` (currently it is resolved at call time via
`HuggingFaceEmbedding(...)`; we lift it into `_chunk_messages`'s signature so
the function is unit-testable with a fake tokenizer).

All three call sites inside `_chunk_messages` that compute
`len(tokenizer(text))` become `len(tokenize_fn(text))`. The `max_tokens`
argument of the function is renamed to `effective_max` to make the new
semantics explicit. The `SentenceSplitter` fallback for oversized single
groups is created with `chunk_size=effective_max` so its sub-chunks also
respect the model's limit. `chunk_overlap=20` stays as is (it is in the same
units now, which is actually a bonus — previously it was 20 tiktoken tokens
fed into a splitter measuring its own chunks in some third unit, which was
nonsense).

#### `indexer.py::_merge_small_chunks`

Identical change: replace `get_tokenizer()` at line 148 with
`_embed_model_tokenizer(embed_model)` (plumbed through from
`index_repository`), use `effective_max_tokens(...)` for the cap. The Phase-1
"node already too big → keep as is, don't merge" branch is preserved — it
was already designed to handle oversized leaf nodes, it just now triggers at
510 instead of 1024.

The Phase-2 small-chunk filter (`< min_chars`) is unchanged: that threshold
is a character count, not a token count, so the tokenizer swap does not
affect it.

#### `indexer.py::_split_document`

`CodeSplitter(count_mode="token", max_tokens=config.child_chunk_size, ...)`
at line 251 also needs the new ceiling, otherwise `CodeSplitter` emits
leaves whose tiktoken-count is ≤ 1024 but whose XLM-R count may exceed 510.
Pass `max_tokens=effective_max` instead. Same for the `SentenceSplitter`
fallback at line 267 (`chunk_size=effective_max`).

**RC-3 (review-required): hard XLM-R invariant for the code/docs path.**
`CodeSplitter` internally measures with its own tiktoken-backed counter,
not with the embedder's tokenizer. Even with `max_tokens=effective_max=510`
it may still hand back a leaf that, **measured in XLM-R**, exceeds 510
(tiktoken under-counts on English source code and on many
identifier-heavy texts). Passing `effective_max` to `CodeSplitter` is
therefore only **Phase 1 — advisory**.

To keep the ≤ `effective_max` invariant in XLM-R units (the units the
embedder actually cares about), `_merge_small_chunks` gets a **Phase 1.5
re-check** between the existing Phase 1 (merge by token budget) and
Phase 2 (drop micro-chunks):

```python
# Phase 1.5: any leaf that is still > effective_max measured by the
# EMBEDDER'S tokenizer is split with SentenceSplitter(chunk_size=
# effective_max, chunk_overlap=...) so the downstream embedder never
# sees a >model_max_length payload. Phase 1 already used effective_max
# as its budget, but CodeSplitter measures in its own (tiktoken) units,
# so a leaf may pass Phase 1 and still be over-limit in XLM-R units.
re_checked: list[BaseNode] = []
for node in merged_nodes:
    if len(tokenize_fn(node.get_content())) <= effective_max:
        re_checked.append(node)
        continue
    sub_nodes = SentenceSplitter(
        chunk_size=effective_max,
        chunk_overlap=min(20, effective_max // 8),
    ).get_nodes_from_documents([_node_to_doc(node)])
    re_checked.extend(sub_nodes)
merged_nodes = re_checked
```

Two consequences:

1. The integration test §5.4 invariant "every stored chunk ≤
   `effective_max` XLM-R tokens" is a **hard** invariant for both the
   history flow and the code/docs flow — no "best-effort" carve-out.
2. `SentenceSplitter` itself measures with `Settings.tokenizer`. We
   explicitly set `Settings.tokenizer = tokenize_fn` for the duration of
   the indexing call (inside the indexer's own setup, next to the
   existing `Settings.embed_model = ...` / `Settings.llm = None`), so its
   internal accounting also lines up with the embedder. This is a scoped
   per-call assignment, not a global side-effect that leaks across
   indexers (both indexers always set `Settings.embed_model` for
   themselves anyway).

This is the same pattern in both places: `config.child_chunk_size` is the
**advisory budget**, `effective_max_tokens(...)` is the **enforced
ceiling**, and Phase 1.5 makes the ceiling hold even when the upstream
splitter measured the input in foreign units.

### 4.4 No effect on `IndexConfig`

`IndexConfig.child_chunk_size: int = 1024` stays. We do **not** add an
`embedding_max_length` field — that is a property of the model, not of the
operator's intent. Reading it from the live embedder at chunk time is
correct and self-adjusting.

### 4.5 Backward compatibility with existing indexes

LanceDB tables created by the old tiktoken-based chunker hold:

- vectors in the same dimensionality (e5-small is unchanged — `dim=384`),
- `node.get_content()` text that may exceed 510 XLM-R tokens (truncated when
  embedded, but stored in full).

After this change:

- **Reads** (`search_history`, project memory): unchanged. The retriever
  pulls top-k nearest vectors and returns `node.get_content()`. We do not
  re-embed and we do not recompute lengths.
- **Writes** (new JSONL lines, new project files): the incremental path
  (`file_hashes.json` / `line_hashes.json`) re-runs the chunker only for new
  lines/files. Those new chunks will be ≤ 510 XLM-R tokens. The resulting
  index is a "mixed" set of old (≤ 1024 tiktoken) and new (≤ 510 XLM-R)
  chunks. Both are perfectly valid LanceDB rows in the same table; the only
  observable difference is shorter `content` strings on freshly-added rows.

No migration is required. If an operator wants a uniformly-tokenized
index they can `rm -rf <state_dir>/<agent>/summaries/.history_index/`
and the next summarisation cycle re-indexes from scratch. We document this
in the README; we do **not** do it automatically and we **never** touch
the agent-initialisation state.

## 5. Tests (TDD)

All tests live under `semantic_search/tests/`. Run with the project's
existing harness; pyright strict must pass on the changed files.

### 5.1 Unit — token counting and `_embed_model_tokenizer`

**`tests/unit/test_tokenizer_alignment.py`** (new file):

Tokenizer extraction & metadata:

- `test_embed_model_tokenizer_returns_callable_and_512_limit` — instantiate
  the real `HuggingFaceEmbedding(intfloat/multilingual-e5-small)`
  (offline, `local_files_only=True`), call `_embed_model_tokenizer(e)`,
  assert (i) `tokenize_fn` is callable and `tokenize_fn("hello world")`
  returns a non-empty `list[int]`, (ii) `model_max_length == 512`,
  (iii) `special_tokens_reserve == 2`.
- `test_special_tokens_reserve_matches_real_tokenizer` (RC-2 sentinel) —
  load the real e5-small tokenizer and assert
  `tokenizer.num_special_tokens_to_add(pair=False) == 2`. If this ever
  breaks, the "510" boundary in §5.2 / §5.3 changes accordingly and we
  notice.

Cross-tokenizer-disagreement sentinels:

- `test_xlmr_token_count_differs_from_tiktoken_on_russian` — assert
  `len(tokenize_fn("This is a long Russian-language sentence; running the tests is important."))` differs from the
  `tiktoken.get_tokenizer()` count on the same string by at least 25 %
  AND that the XLM-R count is the smaller one. This is the regression
  sentinel for the whole change: the two tokenizers MUST disagree —
  otherwise the bug we are fixing did not exist.
- `test_xlmr_token_count_differs_from_tiktoken_on_english` — same shape,
  English sentence, expect XLM-R to be the **larger** count (the other
  direction of the error).

`effective_max_tokens`:

- `test_effective_max_with_short_config` — `child_chunk_size=128`,
  `model_max_length=512`, `special=2` → 128 (config binds).
- `test_effective_max_with_long_config_and_e5_model` —
  `child_chunk_size=1024`, `model_max_length=512`, `special=2` → 510
  (model binds, reserve applied).
- `test_effective_max_with_hypothetical_8k_model_and_xlmr_specials` —
  `child_chunk_size=1024`, `model_max_length=8192`, `special=2` → 1024
  (config binds again).
- `test_effective_max_with_zero_special_tokens` —
  `child_chunk_size=1024`, `model_max_length=512`, `special=0` → 512
  (boundary against accidental off-by-special-reserve).

RC-1 — no silent fallback to tiktoken:

- `test_raises_when_embed_model_has_no_model_attr` — pass a `SimpleNamespace()`
  (no `_model`); assert `RuntimeError` whose message contains both
  `"embed_model has no accessible tokenizer"` and
  `"DESIGN_FIX_TOKENIZER_MISMATCH.md"`.
- `test_raises_when_inner_model_has_no_tokenizer` — pass
  `SimpleNamespace(_model=SimpleNamespace())` (inner `_model` present but
  no `tokenizer`); same `RuntimeError`.
- `test_raises_when_tokenizer_lacks_required_api` — pass a tokenizer stub
  that is not callable, or lacks `model_max_length`, or lacks
  `num_special_tokens_to_add`; same `RuntimeError`.

### 5.2 Unit — `_chunk_messages` boundary behaviour

**`tests/unit/test_history_indexer_xlmr_boundary.py`** (new file):

Use a **fake tokenizer** (callable that splits on whitespace and returns a
list of fake int IDs) plus a fake `model_max_length` so the test is
deterministic and does not require the real model.

- `test_group_under_effective_max_kept_whole` — one group of ~300 fake
  tokens, `effective_max=510`, asserts one chunk produced, no
  `SentenceSplitter` invoked, `chunk_index` metadata absent.
- `test_group_at_exact_effective_max_kept_whole` — boundary: exactly 510
  tokens, one chunk, no split.
- `test_group_above_effective_max_is_sentence_split` — 800 fake tokens, one
  group, asserts the result is multiple chunks, each ≤ 510 by the **same**
  fake tokenizer, and that `chunk_index` metadata is `"0", "1", ...`.
- `test_two_small_adjacent_groups_merge_until_effective_max` — three groups
  of ~200 tokens each, asserts they fit into two chunks (200+200 → flush
  before adding third), not three.
- `test_subchunk_below_min_chars_dropped_in_multi_chunk_case` — keeps the
  existing 50-char filter behaviour intact.

### 5.3 Unit — `_merge_small_chunks` boundary + Phase-1.5 invariant

**`tests/unit/test_indexer_xlmr_boundary.py`** (new file): mirror of 5.2
for the project-code/docs path. Uses fake nodes (no real CodeSplitter,
no model) with a fake `tokenize_fn` that returns one fake token per
word.

- `test_several_small_leaves_merged_up_to_effective_max_not_to_config` —
  three 200-fake-token leaves, `effective_max=510` (model-bound) —
  asserts they merge into two chunks (200+200, flush, 200), not into one
  big 600-token chunk.
- `test_leaf_above_effective_max_is_resplit_phase_1_5` (RC-3 invariant) —
  feed a single leaf of 800 fake tokens into Phase 1 with
  `effective_max=510`; assert the function returns multiple nodes, each
  of ≤ 510 fake tokens measured by the same `tokenize_fn`. **Before this
  test goes green, Phase 1.5 does not exist.** This is the hard-invariant
  test for the code/docs path.
- `test_leaf_already_under_effective_max_passes_through` — 300-token
  leaf, `effective_max=510`, returned unchanged (Phase 1.5 is a no-op for
  it).
- `test_child_chunk_size_in_config_unchanged` — assert
  `IndexConfig().child_chunk_size == 1024` (overlaps with §5.5 but kept
  here as a context-local sentinel).

### 5.4 Integration — real index, real embedder, no mocks

**`tests/integration/test_indexer_no_silent_truncation.py`** (new file):

Two-sided sentinel (reviewer note — RC-3 follow-up):

- **Pre-condition probe.** Run `_split_document` on the test input
  WITHOUT the new ceiling (call the old code path via a thin shim that
  rebuilds it from the pre-fix snapshot, or temporarily disables Phase
  1.5) and assert that at least one resulting leaf is > 510 XLM-R
  tokens. This proves the test input is large enough to actually
  exercise the bug — guards against the false-green "small test data,
  bug never tripped".
- **Post-condition assertion.** Run the real `index_repository(tmpdir)`
  end-to-end (real `HuggingFaceEmbedding`, real `LanceDBVectorStore`,
  real `_merge_small_chunks` with Phase 1.5) and assert that EVERY
  stored chunk has XLM-R-token-length ≤ 510 (computed with the same
  `tokenize_fn` from `_embed_model_tokenizer`).

Test input: a temporary repo with a single Markdown file built by
concatenating long Cyrillic + identifier-heavy English paragraphs (~3000
words). The "Cyrillic + identifier-heavy English" mix is deliberate
because that is the language profile under which XLM-R OVER-counts vs
tiktoken — exactly the regime where `CodeSplitter`'s tiktoken-based
`max_tokens=effective_max` advisory is the weakest, so Phase 1.5 has the
most work to do. The pre-condition probe verifies this empirically.

`chunk_overlap` sanity: while iterating chunks, assert that the
`SentenceSplitter(chunk_size=effective_max,
chunk_overlap=min(20, effective_max//8))` calls inside Phase 1.5 never
produce a sub-node > `effective_max` measured by `tokenize_fn` — reviewer
asked for this explicit check ("510+overlap must not overflow").

**`tests/integration/test_history_indexer_mixed_index_reads.py`** (new
file):

- Build a fake `.history_index/` populated with rows whose `content` is
  intentionally > 1024 tiktoken tokens (simulating the **pre-fix** state).
- Append two new JSONL lines that produce a couple of small chunks under
  the **new** code path.
- Call `index_history_file` to ingest only the new lines.
- Run `search_history` for a phrase present in the new lines and for a
  phrase present in the old long chunk; assert both queries return their
  respective rows (mixed index reads cleanly).

### 5.5 Regression — config knob unchanged

**`tests/unit/test_config.py`** (existing file, append):

- `test_child_chunk_size_default_unchanged_by_tokenizer_fix` — asserts
  `IndexConfig().child_chunk_size == 1024` (sentinel against accidental
  change during this work).

## 6. Migration notes

Documented in `semantic_search/docs/DESIGN_FIX_TOKENIZER_MISMATCH.md`
(this file) and in `dw_hist/README.md` (operator-facing):

> Existing history / project indexes built before this fix may contain
> chunks whose tail content was silently dropped at embedding time. New
> writes use the embedder's own tokenizer with a 510-token ceiling, so the
> problem does not recur. To get a uniformly-tokenized index for an agent,
> stop the agent and remove
> `<state_dir>/<agent>/summaries/.history_index/` and
> `<state_dir>/<agent>/.vector_index/`; the next summarisation /
> indexing cycle rebuilds them. **Do NOT do this for the
> agent-initialisation state.**

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Touching `embed_model._model.tokenizer` is private API of `HuggingFaceEmbedding` | The embedder itself relies on the same attribute; an upstream rename breaks both at once, no asymmetric risk. Wrapped in one helper (`_embed_model_tokenizer`) so a future fix touches one line. |
| Mixed-tokenizer index produces inconsistent scores | The vector space is unchanged (same embedder, same dim, same metric). Only chunk boundaries differ between old and new rows. Scores remain comparable; the worst case is that an old over-long chunk's stored `content` was longer than the part the embedder actually saw — exactly the pre-fix behaviour, just not getting *worse* with this change. |
| Cold-start cost of the chunker now touches the model | The embedder is already loaded for any indexing/search call; we reuse the same instance via `_embedding_cache`. No extra mmap, no extra weight load. |
| Pyright strict on private attribute access | Add a single `# type: ignore[attr-defined]` on `embed_model._model.tokenizer` inside `_embed_model_tokenizer` with a comment pointing back to this design. |
| Test on Russian/English specifically — what about other languages? | The unit tests are sentinels for "the two tokenizers disagree"; the integration test is what proves end-to-end correctness, language-agnostic. We do not enumerate languages. |

## 8. Out-of-scope follow-ups

These are not implemented here but are natural next steps:

- `dw-hist --show-tokens` — surface `(query_xlmr_tokens, content_xlmr_tokens,
  content_truncated_at_embed_time?)` per result. Requires only `dw-hist`
  changes; no `semantic_search` change.
- Investigate switching to an 8 k-context embedder
  (`nomic-embed-text-v1.5`, `jina-embeddings-v3`,
  `Alibaba-NLP/gte-multilingual-base`). Requires full reindex and quality
  re-calibration; explicitly deferred until the user asks.

---

### Sign-off

This document is the contract for the change. Implementation will follow it
literally; any deviation discovered during TDD will be brought back here as
an amendment and re-reviewed.
