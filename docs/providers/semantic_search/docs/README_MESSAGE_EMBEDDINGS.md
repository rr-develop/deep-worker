# Per-message embeddings — usage

Length-robust embeddings for arbitrary text, and a persistent, incrementally
extendable store of one embedding per history message. Full design:
`DESIGN_MESSAGE_EMBEDDING_STORE.md`.

All heavy deps (`numpy`, `torch`, the embedder) load lazily — importing these
modules is cheap; `torch` is only loaded when you actually embed.

## Embed text of any length

```python
from semantic_search import embed_long_text

vec = embed_long_text("…any length, even > 512 tokens…")   # numpy (dim,), L2-normalised
```

Text longer than the embedder window (510 tokens for e5-small) is split into
token-accurate windows; the windows are combined as
`normalize(Σ tokensᵢ · embᵢ)` — no tail is dropped.

## One embedding per message (in memory)

```python
from semantic_search import embed_history_messages

embs, lens = embed_history_messages("/home/<user>/.deep_worker/<agent>/summaries")
# embs: torch.float32 (N, dim), rows L2-normalised
# lens: torch.int64  (N,)  — token length per message
```

This recomputes everything (minutes for a large history). For repeated use,
persist it (below).

## Persistent, incremental store (recommended)

Build once, then extend on demand as history grows. Data lives in
`summaries_dir/.message_embeddings/` (`embeddings.npy`, `token_lengths.npy`,
`manifest.json`).

```python
from semantic_search import (
    build_message_embeddings,   # create or incrementally extend
    load_message_embeddings,    # fast load, no recompute
    message_embeddings_exist,
)

summaries = "/home/<user>/.deep_worker/<agent>/summaries"

# First run: full build (slow, one-time). Later runs: only new messages.
stats = build_message_embeddings(summaries)
print(stats)   # total / new / files / rebuilt

# Fast load anywhere later — sub-second, no model loaded:
embs, lens = load_message_embeddings(summaries)            # torch tensors
embs_np, lens_np = load_message_embeddings(summaries, as_torch=False)  # numpy, no torch
```

### After history files grow

History `*.jsonl` files are append-only. Re-running `build_message_embeddings`
embeds **only the appended messages** and appends their rows:

```python
stats = build_message_embeddings(summaries)
print(stats.new_messages)   # 0 if nothing changed
```

### Rebuild from scratch

Pass `reset=True`, or it happens automatically (with `stats.rebuilt == True`) if
the embedding model / schema changed or a tracked file shrank (rewritten rather
than appended):

```python
build_message_embeddings(summaries, reset=True)
```

## Per-message metadata tensors (indices + flags)

Two extra tensors describe the already-computed embeddings **without
recomputing them**: which message each embedding came from, and what kind of
message it is.

```python
from semantic_search import (
    build_message_metadata,    # cheap pass over JSONL; NO embedding recompute
    load_message_metadata,
    message_metadata_exist,
    MessageFlag,
)

# After build_message_embeddings(...) has produced the embedding rows:
stats = build_message_metadata(summaries)
print(stats.total_messages, stats.flag_counts)

idx, flags = load_message_metadata(summaries)            # torch int64 (N,), (N,)
# idx[i]  -> global 0-based ordinal of the message behind embeddings[i]
# flags[i] -> MessageFlag bitmask for embeddings[i]
```

`message_indices` is the canonical message id (unique, contiguous `0..N-1`,
stable across appends). It is used instead of the JSONL `message_index` field,
which is **not** unique in real history.

### Reading flags (bitmask)

```python
import torch
from semantic_search import load_message_metadata, MessageFlag

_, flags = load_message_metadata(summaries)

is_plain_user = (flags & MessageFlag.PLAIN_USER) != 0      # bool tensor (N,)
is_agent      = (flags & MessageFlag.AGENT) != 0
is_tool_result = (flags & MessageFlag.TOOL_RESULT) != 0

# e.g. count genuine end-user messages:
print(int(is_plain_user.sum()))
```

Available bits: `USER_ROLE`, `TASK_MANAGER`, `PLAIN_USER`, `AGENT`,
`AGENT_TO_AGENT`, `TOOL_RESULT`, `TOOL_CALL`, `SYSTEM`, `SUMMARIZATION`,
`AGENT_FINAL`. A message may set several at once (e.g. `AGENT | AGENT_FINAL`, or
`AGENT | SUMMARIZATION`). `SUMMARIZATION` marks a condensed-summary message
(`ai` body starting `## SESSION INTENT`); `AGENT_FINAL` marks an agent message
whose next turn is the user/TaskManager. `TOOL_CALL` is **reserved and always 0**
on current data — it would mark an embedding *of a tool call itself*, which the
store does not build (only the agent's accompanying comment text), so it is never
annotated onto an `AGENT` comment. See the design doc.

### Staleness

Metadata describes specific embedding rows, so **changing the embeddings
invalidates it**: `build_message_embeddings` (append or rebuild) deletes
`message_indices.npy` / `flags.npy`. Re-run `build_message_metadata` afterwards.

## Notes

* Row order is **file-sorted then line order**, append-only — a row index is
  stable across incremental updates.
* "Message" = a non-empty message; empty-content entries are skipped (no text
  to embed), exactly as in the search indexer.
* Vectors are in the **same space** as the search index (same e5-small model,
  same 510-token ceiling), so they are directly comparable.
* Reusing a loaded embedder across calls: pass `embed_model=<HuggingFaceEmbedding>`
  to any of these functions to skip re-loading the model.
```
