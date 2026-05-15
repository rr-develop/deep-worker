# Process-wide caches in `semantic_search.retriever`

## Why this matters

`search()` is called many times per process — typically once per
`project_memory` tool invocation in deep-worker agents. The
straightforward implementation that re-creates `HuggingFaceEmbedding`
and `VectorStoreIndex` on every call leaks roughly **390 MB of RSS per
call** in production with the `multilingual-e5-small` model.

This document explains what is cached, why, and how to interact with
the caches if you ever need to.

## What is cached

Two module-level dicts in `semantic_search/retriever.py`:

```python
# model_name -> HuggingFaceEmbedding (or whatever embed implementation)
_embedding_cache: dict[str, Any]

# (repo_path_str, model_name) -> (VectorStoreIndex, StorageContext)
_index_cache: dict[tuple[str, str], tuple[VectorStoreIndex, StorageContext]]
```

The first call for a given key pays the full instantiation cost; every
subsequent call is a dict hit and reuses the same Python objects.

## Why naive re-instantiation leaks

A fresh `HuggingFaceEmbedding(model_name="...")` does three things:

1. Loads sentence-transformer weights into a torch storage. For
   `multilingual-e5-small` (~117M params) that is one ~128 MB
   anonymous arena per call.
2. `mmap`-s the model blob (`.../models--intfloat--multilingual-e5-small/blobs/<hash>`).
   Each instance gets its own private mapping (~450 MB virtual,
   ~80 MB resident).
3. Allocates tokenizer state, attention masks, and cached tensors.

The previous instance does **not** disappear when `search()` returns
because:

* `LanceDBVectorStore` (created inside `_load_storage_context`) keeps a
  reference to its `embed_model`.
* `lancedb` spawns a tokio thread pool (visible as `tokio-runtime-w`
  threads in the process). The pool holds a C++-side handle that
  references the Python embed_model. Python's GC does not see across
  the C++ boundary, so reference-counting-based cleanup never fires.
* `llama_index.core.Settings.embed_model` is a process-global; the
  pointer is overwritten each call but the previous Python object
  remains live through the chain above.

Net effect: each call adds another full set of (embedder + index +
storage + lance handle + tokio threads). In production agents run for
hours; the leak shows up as RSS growing linearly to multiple GB before
something forces the OS to act.

## Verification

Regression test:
[`tests/integration/test_search_no_leak_per_call.py`](../tests/integration/test_search_no_leak_per_call.py)

It runs five sequential `search()` calls against a small indexed repo
and asserts that post-warmup RSS growth stays below 50 MB. Without the
caches the test fails with a >1 GB delta over five calls.

Live agent verification — three sequential `project_memory` calls
against `test_vc1` (model = multilingual-e5-small):

| Probe                  | RSS       | ΔRSS    | mmap of model |
| ---------------------- | --------- | ------- | ------------- |
| baseline (fresh agent) | 811 MB    | —       | 0             |
| after 1st call (warmup)| 1 823 MB  | +1 012  | 1             |
| after 2nd call         | 1 838 MB  | +15 MB  | **1** (reused)|
| after 3rd call         | 1 849 MB  | +10 MB  | **1** (reused)|

Compare to the leaking pre-fix code where each non-warmup call added
~390 MB and another mmap of the model.

## Cache lifetime, eviction, invalidation

The caches live for the lifetime of the Python process. There is no
eviction. Rationale:

* The `(repo_path, model_name)` cardinality per process is at most a
  handful — agents typically work in one repo with one model.
* Eviction would re-trigger the very leak this design avoids: the
  evicted entry would be referenced by lance C++ retainers and never
  actually freed.
* Long-lived agents are restarted explicitly when the user wants a
  clean slate; that resets everything.

If you must invalidate a cache entry (e.g. you re-indexed a repo and
want the new index visible without a process restart), simply
`del _index_cache[(str(repo_path), model_name)]`. The next `search()`
call will rebuild it. Be aware that the OLD `LanceDBVectorStore` will
not be freed — the tokio threads and lance handle will continue to
live in the process.

## Caller-supplied embeddings

If a caller passes their own `embedding_model=...` argument to
`search()` (e.g. a unit test using `MockEmbedding`), we honour it and
bypass `_embedding_cache`. We still cache the resulting index by
`(repo_path, config.embedding_model_name)`, so:

* A test using `MockEmbedding` populates `_index_cache` keyed by the
  default model name. Subsequent test calls reuse the same cached
  index — the embedding model parameter is fixed by what was used to
  build the index, the override is only honoured the first time.

For production code paths (no caller-supplied embedding) this is
transparent.
