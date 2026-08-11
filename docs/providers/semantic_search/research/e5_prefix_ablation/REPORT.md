# E5 `query:` / `passage:` prefix ablation — agent history retrieval

**Date:** 2026-06-30
**Status:** Complete. No code change recommended (see Conclusion).
**Branch:** dev

---

## 1. Background and motivation

`semantic_search/` uses `intfloat/multilingual-e5-small` to build the agent-side
history index (`history_indexer.py`, `history_retriever.py`) and the project-side
code index (`indexer.py`, `retriever.py`). Both indices currently pass **bare
text** to the embedding model on both the indexing side and the query side.

The official model card for the E5 family (see "Sources" below) is explicit:

> *"Do I need to add the prefix `query: ` and `passage: ` to input texts?"* —
> *"Yes, this is how the model is trained, otherwise you will see a performance
> degradation."*

The rule:
- **Asymmetric retrieval (open-QA, ad-hoc IR):** `query:` on the query, `passage:`
  on the document.
- **Symmetric tasks (semantic similarity, paraphrase, clustering):** `query:` on
  both sides.

Neither path is what we currently do — both sides go in bare. The question this
research answers:

> **For our actual corpus (agent conversation history) and our actual use-case
> (history-memory retrieval), does adopting the E5 prefix contract measurably
> improve retrieval quality?**

We deliberately do *not* try to predict what would change on public benchmarks
(BEIR, MTEB, Mr. TyDi). We measure the effect on our own data, because we found
**no published ablation** of "E5 with prefixes vs without" with numbers on those
benchmarks — only the authors' qualitative statement that performance will
degrade.

## 2. Method

### 2.1 What we test

Two pipelines over the **same** message corpus and the **same** model
(`intfloat/multilingual-e5-small`, the production choice):

- **A (baseline, current):** documents and queries are encoded as plain text.
- **B (prefixed, per E5 training contract):** `"passage: " + text` on documents,
  `"query: " + text` on queries.

For each query we record the rank at which the **gold passage** appears in the
top-50 of the cosine-similarity ranking over the full haystack. Embeddings are
L2-normalized (matches the E5 model card recipe), so cosine reduces to a dot
product.

We do **not** route through LanceDB / `history_retriever` for this ablation — the
production path has its own caching, chunking, and prefix wiring that would
muddy A vs B. We compute embeddings directly from the model so the only variable
between the two pipelines is the prefix.

### 2.2 Pair construction (no manual labels)

We have no relevance-labeled set. The agent JSONL summaries contain natural
(query, gold passage) pairs whose relevance is **built into the format** rather
than annotated by hand. Two pair types are extracted per the protocol agreed
with the user:

**T1 — user/TaskManager prompt → final agent answer.**
- *Query:* a `human` message. Both real human turns and TaskManager-injected
  turns (`content` starts with `TaskManager:>`) are included; from the
  retriever's point of view they are both "a request directed at the agent".
- *Passage (the "final answer"):* the **last** `ai` message in the contiguous
  run of messages between this `human` turn and the next `human` turn (or end
  of file). Tool calls / tool results in between are skipped.
- *Hard exclusion:* if that last `ai` message has a `tool_calls` entry whose
  `name` is `send_message_to_agent` (or `mcp__deepworker__send_message_to_agent`),
  it is **NOT** a reply to the user — it is a request to another agent. The
  pair is dropped.

**T2 — outbound `send_message_to_agent` → peer reply.**
- *Query:* `args.message` of an outgoing `send_message_to_agent` tool call. The
  target peer is `args.agent_name`.
- *Passage:* the next incoming `human` message whose content begins with
  `"<peer> -> <self> :>"` for the same `<peer>` (this prefix is how the
  inter-agent transport formats incoming peer messages — verified empirically
  in the source corpus).

### 2.3 Hygiene applied to every candidate pair

1. After `strip()`, both sides must be **≥ 50 characters** (drops `"ok"`,
   `"thank you"`, empty turns).
2. Truncate any side to **8000 characters** (E5 truncates at 512 tokens anyway;
   this bounds encoding memory and keeps the scoring deterministic).
3. Deduplicate by query text — first occurrence wins.
4. Each pair carries `message_index` of both sides so any single result can be
   spot-checked back to the source JSONL.

### 2.4 Haystack (distractor pool)

The **full** ai-content message set across all source agents (after the same
≥ 50 char filter) plus every incoming inter-agent message. Gold passages are
guaranteed to be inside the haystack. This is intentionally a realistic search
— the gold competes with thousands of other plausible documents from the same
agent's history.

### 2.5 Metrics

Per pipeline:
- **Recall@1, Recall@5, Recall@10** — fraction of queries where the gold
  passage is in the top-k.
- **MRR@10** — mean reciprocal rank, with rank > 10 contributing 0.

Comparison A vs B:
- Per-query reciprocal rank (capped at 10) → **Wilcoxon signed-rank** test
  (paired, two-sided). Reports `W` and `p`.
- Per-query rank delta count: in how many queries B beat A, vs A beat B, vs tied.

### 2.6 Validity caveats (documented up-front)

- **`ai` ≠ unique correct document.** Multiple `ai` messages in the corpus may be
  semantically relevant to the same query (e.g. several turns discussing the
  same incident). So *absolute* Recall@k is a lower bound in **both** pipelines.
  The **delta** A vs B remains interpretable because the same bias acts on both.
- **TaskManager:> prompts are highly stereotyped** (`"continue"`,
  `"don't stall — act"`). They are still legitimate queries but contribute
  high lexical-overlap pairs that may inflate Recall@1 in both pipelines
  equally.
- **T2 pairs are typically long** (review requests, design docs). The 8000-char
  cap may truncate; this affects A and B equally.
- **One model size** (E5-small). The direction of the effect could be different
  on `multilingual-e5-large` (1024-dim vs 384-dim).

## 3. Data sources

Source files (all read-only, **not** copied into the repo):

| Agent | JSONL path | Messages |
|------|------|-------:|
| agent_a | `~/.deep_worker/<agent_a>/summaries/<agent_a>.jsonl` | 12271 |
| agent_b | `~/.deep_worker/<agent_b>/summaries/<agent_b>.jsonl` | 10376 |
| agent_c | `~/.deep_worker/<agent_c>/summaries/<agent_c>.jsonl` | 8873 |
| agent_d | `~/.deep_worker/<agent_d>/summaries/<agent_d>.jsonl` | 6489 |
| agent_e | `~/.deep_worker/<agent_e>/summaries/<agent_e>.jsonl` | 5344 |
| agent_f | `~/.deep_worker/<agent_f>/summaries/<agent_f>.jsonl` | 4557 |
| agent_g | `~/.deep_worker/<agent_g>/summaries/<agent_g>.jsonl` | 4472 |
| agent_h | `~/.deep_worker/<agent_h>/summaries/<agent_h>.jsonl` | 4126 |
| agent_i | `~/.deep_worker/<agent_i>/summaries/<agent_i>.jsonl` | 4052 |

Each file is the `deepagents.summarization_history` JSONL produced for that
agent — multi-week conversation history with `human`/`ai`/`tool`/`system`
messages, intact `tool_calls` blocks.

Per-source extraction summary (after hygiene filters):

| Source | T1 pairs | T2 pairs | Distractors in haystack |
|------|-------:|-------:|-------:|
| agent_a | 80 | 2 | 530 |
| agent_b | 24 | 28 | 244 |
| agent_c | 121 | 0 | 678 |
| agent_d | 64 | 9 | 333 |
| agent_e | 32 | 12 | 564 |
| agent_f | 15 | 10 | 252 |
| agent_g | 9 | 20 | 262 |
| agent_h | 13 | 12 | 258 |
| agent_i | 52 | 4 | 527 |
| **Total** | **410** | **97** | **4086** |

(Numbers above are from per-source extractor passes; the unified de-duplicated
manifest reports `pairs_total=518` with T1=401, T2=117 after cross-source
de-dup of identical query texts.)

## 4. Tools and methodology used

All scripts in this folder. Single-pass, deterministic (no random sampling, no
non-determinism in the dot-product search).

- **`extract_pairs.py`** — JSONL parser that produces `pairs.jsonl`,
  `corpus.jsonl`, `manifest.json` per the §2.2/§2.3 protocol. Multi-source mode
  (`--jsonl path1 --jsonl path2 ...`) de-dups pairs across sources by query
  text and namespaces document ids by agent (`<agent>::<message_index>`) so
  identical `message_index` values in two different agents cannot collide.

- **`run_ablation.py`** — loads the model, encodes the corpus and queries
  twice (bare vs prefixed), runs the search, computes the per-query rank of
  the gold passage and the aggregate metrics, dumps `results.json`,
  `per_query_ranks.jsonl`. Caches the four encoding tensors under `_emb_cache/`
  so re-runs are fast. Wilcoxon falls back to a normal-approximation if SciPy
  is missing.

### Model and infrastructure

- Embedding model: `intfloat/multilingual-e5-small`, snapshot
  `c007d7ef6fd86656326059b28395a7a03a7c5846` from the locally cached
  llama_index snapshot used by the production retriever (local model cache).
- Encoding: mean-pool over `last_hidden_state` masked by `attention_mask`,
  L2-normalize. Matches the official E5 model card recipe.
- `max_length=512`, batch size 16, CPU inference. Wall-clock: ~10 minutes for
  4086 docs + 518 queries × 2 pipelines.
- No GPU was used or required for this dataset size.

### How to reproduce

```bash
# 1. Extract pairs from multiple agent histories
python3 semantic_search/research/e5_prefix_ablation/extract_pairs.py \
    --jsonl ~/.deep_worker/<agent_a>/summaries/<agent_a>.jsonl \
    --jsonl ~/.deep_worker/<agent_b>/summaries/<agent_b>.jsonl \
    --out-dir <output-dir>

# 2. Run ablation (writes results.json and per_query_ranks.jsonl into out-dir)
python3 semantic_search/research/e5_prefix_ablation/run_ablation.py \
    --pairs <output-dir>/pairs.jsonl \
    --corpus <output-dir>/corpus.jsonl \
    --out-dir <output-dir> \
    --model intfloat/multilingual-e5-small
```

Outputs are deliberately **not** in the repo — they are agent-specific data and
belong under `~/.deep_worker/<agent>/data/e5_prefix_ablation/`. The repo holds
only the protocol, the tools, and this report.

## 5. Results

### 5.1 Overall (n = 518)

| Metric    | A (bare) | B (prefixed) | Δ (B − A)  |
|-----------|---------:|-------------:|-----------:|
| Recall@1  | 0.0444   | 0.0444       |   0.0      |
| Recall@5  | 0.1293   | 0.1274       |  −0.0019   |
| Recall@10 | 0.1486   | 0.1564       |  +0.0077   |
| MRR@10    | 0.0790   | 0.0801       |  +0.0011   |

**Wilcoxon signed-rank (two-sided) on per-query reciprocal rank: p = 0.861.**
Out of 518 pairs, only 41 changed rank at all: B better on **19**, A better on
**22**, tied on **477**.

### 5.2 By pair type

**T1 — user/TaskManager prompt → final agent answer (n = 401)**

| Metric    | A      | B      | Δ        |
|-----------|-------:|-------:|---------:|
| Recall@1  | 0.0224 | 0.0274 | +0.0050  |
| Recall@5  | 0.0549 | 0.0599 | +0.0050  |
| Recall@10 | 0.0673 | 0.0698 | +0.0025  |
| MRR@10    | 0.0364 | 0.0409 | +0.0045  |

**T2 — outbound `send_message_to_agent` → peer reply (n = 117)**

| Metric    | A      | B      | Δ        |
|-----------|-------:|-------:|---------:|
| Recall@1  | 0.1197 | 0.1026 | −0.0171  |
| Recall@5  | 0.3846 | 0.3590 | −0.0256  |
| Recall@10 | 0.4274 | 0.4530 | +0.0256  |
| MRR@10    | 0.2251 | 0.2144 | −0.0107  |

## 6. Interpretation

1. **Differences are tiny and not statistically significant on this corpus.**
   Wilcoxon p = 0.86 means the data is consistent with no real difference
   between A and B. Of 518 pairs, **92 %** end up at the same rank under both
   pipelines (often *both* outside the top-10 — see caveat §2.6).
2. **The direction is mixed.** B helps slightly on T1 (consumer-style chat:
   short prompt → longer answer), which is exactly the asymmetric retrieval
   shape E5 was trained on. B does **not** help on T2, where both sides are
   long structured peer-to-peer messages (review requests, design docs) —
   that distribution looks more symmetric than asymmetric and the
   `passage: ` / `query: ` framing arguably introduces a mismatch.
3. **Absolute Recall@k is low** in both regimes. That is expected and was
   called out in §2.6: many ai-messages discuss overlapping topics, so the
   "gold" message is rarely the *uniquely* best match in the haystack of
   ~4000 distractors. The **delta** A vs B remains interpretable because the
   same bias hits both pipelines equally.
4. **One model size only** (E5-small). It is plausible — but not measured here
   — that `multilingual-e5-large` shows a larger gap because its representation
   capacity gives the prefix conditioning more leverage.

## 7. Conclusion

For *this corpus* and *this use-case* (agent-history retrieval with
`multilingual-e5-small`), **adopting the E5 `query:` / `passage:` prefix
contract does not produce a measurable improvement** (Recall@10 +0.77 pp,
MRR@10 +0.11 pp, p = 0.86). The training-contract argument from the model
authors is valid in principle, but on our data the prefix gap is not a
high-priority correctness issue.

**Recommendation:** do **not** prioritize a full re-index just for the prefix
change. If retrieval quality is later identified as a critical-path concern,
the much larger levers (in rough priority order) are:

1. Chunking long messages instead of single-shot 512-token encoding.
2. Trying `multilingual-e5-large` (or a stronger instruction-tuned variant)
   and re-running this same ablation to see whether the prefix gap grows
   with model capacity.
3. Investigating the per-query regressions in `per_query_ranks.jsonl` (the 22
   cases where B was worse than A) — most are T2 with long bodies; suggests
   truncation interaction.

## 8. Sources

### Authoritative model documentation
- [intfloat/multilingual-e5-large — Hugging Face model card](https://huggingface.co/intfloat/multilingual-e5-large) — FAQ explicitly states the prefix is required for the model's trained behaviour.
- [intfloat/multilingual-e5-small — Hugging Face](https://huggingface.co/intfloat/multilingual-e5-small) — model used in this study; same prefix contract.
- [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2), [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) — English variants, same contract.

### E5 papers
- [*Multilingual E5 Text Embeddings: A Technical Report*, Wang et al., arXiv 2402.05672 (2024)](https://arxiv.org/pdf/2402.05672) — describes training procedure and prefix convention for the multilingual line.
- [*Text Embeddings by Weakly-Supervised Contrastive Pre-training*, Wang et al., arXiv 2212.03533 (2022)](https://arxiv.org/pdf/2212.03533) — the original E5 paper.

### Community evidence on silent degradation
- [qdrant/qdrant#9024 — *Cloud Inference: E5 models embed text without required `query:` / `passage:` prefixes*](https://github.com/qdrant/qdrant/issues/9024) — reproducible script demonstrating that the same string with three different prefix variants produces three distinct embeddings (cos `(bare, passage) = 0.978`, `(bare, query) = 0.969`, `(passage, query) = 0.942`). Does **not** itself measure retrieval-quality delta on a labeled set.
- [intfloat/multilingual-e5-large — Discussion #34: Unclear query and passage prefix instructions](https://huggingface.co/intfloat/multilingual-e5-large/discussions/34) — author confirmations and community Q&A on when each prefix applies.

### Tools used (all in this folder)
- `extract_pairs.py` — pair extractor
- `run_ablation.py` — encoder + ranker + Wilcoxon

### Related local docs in this repo
- `semantic_search/semantic_search/history_indexer.py` — production agent-history indexer (still bare text on both sides at the time of this study).
- `semantic_search/semantic_search/history_retriever.py` — production agent-history retriever; the docstring already calls out the prefix gap as a known limitation (commit `621e586`).
- `semantic_search/docs/PROCESS_WIDE_CACHES.md` — describes the cached `multilingual-e5-small` model blob this study uses.
- `semantic_search/docs/DESIGN_FIX_PROJECT_MEMORY_OFFLINE_MODEL_LOAD.md` — confirms the model is present locally in `~/.cache/llama_index/...` and can be loaded offline.
