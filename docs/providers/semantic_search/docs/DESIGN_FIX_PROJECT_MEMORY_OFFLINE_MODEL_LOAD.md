## DESIGN: project_memory must load its embedding model strictly offline (never hit the network)

**Status:** design (no code changed). Branch: `dev`.
**Scope:** `semantic_search.retriever` (the engine behind the agents' `project_memory` tool). The offline behaviour is scoped to OUR embedder instantiation only — NOT a host-wide / process-global setting (explicit user constraint).

---

## 1. Incident — `agent_a` hung indefinitely in `status=running`

### Symptom
Agent `agent_a` (working_dir `/path/to/project`) appeared hung: `session_metadata.json` stuck at `status: "running"`, `last_activity` frozen, process alive (PID 598143) but idle (main thread in `ep_poll`, ~0% CPU; a torch/HF worker-thread pool parked in `futex_wait_queue`). No Python traceback in the log.

### Timeline (from `<agent-data-dir>/logs/agent_a.err.log`)
1. Agent started; one turn completed fine (final answer `'9'`).
2. First real `project_memory` call → `semantic_search.retriever.search` lazily instantiated `HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")`.
3. Log shows `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN ...` then `Loading weights: 0% … 7% (14/199)` and **stops**.
4. `ss` showed two external `:443` sockets to HF/CloudFront ranges (`<HF-ENDPOINT>`, `<HF-ENDPOINT>`) in **`CLOSE-WAIT`** with bytes stuck in `recv-q` — the remote closed mid-fetch and the loader neither finished nor errored.

### Root cause
- The embedding model `intfloat/multilingual-e5-small` **is present locally and complete** at `~/.cache/llama_index/models--intfloat--multilingual-e5-small` (llama_index's default `cache_folder = get_cache_dir()`). Verified: with the network disabled, `search()` over `/path/to/project` returns correct results from that cache.
- BUT `HuggingFaceEmbedding(model_name=...)` is instantiated **without forcing local-only resolution**, so `huggingface_hub` performs an **online revision/metadata check** (and, on a cache-dir mismatch, a full download) at load time. Normally this is milliseconds; at the moment `agent_a` ran, HF Hub was transiently unreachable (the `CLOSE-WAIT` sockets), and because:
  1. the model load runs **synchronously inside the agent turn** (`project_memory` is a sync `@tool`; the turn coroutine awaits it), and
  2. there is **no timeout** on the network part,
  the whole turn hung forever → status never returned to idle.

### What is NOT the cause (verified)
- **Not a `project_memory`/`semantic_search` regression.** `git log --since="36h"` shows NO commits touching `semantic_search/`, `tools/project_memory.py`, or `middleware/project_memory.py`. The tool itself is healthy (online and offline loads both succeed in a healthy environment).
- **Not Storage V3, and nobody "mocked the agent's LLM".** The log line `LLM is explicitly disabled. Using MockLLM.` comes from **llama_index itself** (`llama_index/core/llms/utils.py:115`), printed because `retriever.py` sets `Settings.llm = None` (semantic search needs only the embedder, not an LLM). It is a benign side-log of the first `project_memory` call, unrelated to the agent's real (deepseek) LLM and unrelated to the v3 storage layout. The Storage V3 commits (`7ac3770`, `9b8b570`) only touch per-agent storage layout/migration/permissions — they do not touch `semantic_search`, the embedding cache, HF resolution, cwd, or env affecting model loading.
- It is the activation of a long-standing latent vulnerability (network-dependent model load with no offline-forcing and no timeout) by a transient HF outage.

---

## 2. Requirement (user)

> `project_memory` must **NEVER, under any circumstances, hit the network**. The model is already downloaded and that local copy must be used.
> The offline setting must be **scoped to the agents / the `project_memory` tool**, NOT global to the whole host/process.

---

## 3. Fix — force local-only resolution at the embedder instantiation (scoped, not global)

### Mechanism (verified by code + live test)
`HuggingFaceEmbedding.__init__` forwards `**model_kwargs` into
`sentence_transformers.SentenceTransformer(...)`, and `SentenceTransformer.__init__`
accepts **`local_files_only: bool`**. So passing
`model_kwargs={"local_files_only": True}` makes THIS embedder instance resolve
the model strictly from the local cache — `huggingface_hub` performs no network
HEAD/metadata/download requests. This is per-instance; it does NOT set any
`os.environ` and does NOT affect any other library/process on the host.

Verified live (no `HF_HUB_OFFLINE` in env throughout):
- cached model → loads from local cache, 384-dim embedding, ~11s, `HF_HUB_OFFLINE` stayed `None`.
- genuinely-missing model → **fails fast (~6.6s) with a clear `OSError`** ("not a local folder and is not a valid model identifier"), does NOT hang on the network.

We do NOT pass an explicit `cache_folder`. Live-verified (see below): with just
`local_files_only=True` and no `cache_folder`, `HuggingFaceEmbedding` already
finds the model in llama_index's default cache (`~/.cache/llama_index`, where the
model actually lives and where healthy agents load it from). Omitting
`cache_folder` is the safest option (agent review Q2): it adds no dependency
on `get_cache_dir` (whose import path differs across llama_index versions —
it lives in `llama_index.core.utils`, not `llama_index.core`) and makes no
assumption about the cache directory beyond llama_index's own default — the exact
behaviour healthy agents already rely on. (User decision on Q1: rely on the
llama_index default rather than pinning/overriding the location.)

### Where
`semantic_search/semantic_search/retriever.py`, the single place that builds the
embedder (≈ line 219-221), inside the `_embedding_cache` miss branch:

```
cached_embed = HuggingFaceEmbedding(
    model_name=config.embedding_model,
    model_kwargs={"local_files_only": True}, # SCOPED offline — never network
)
```

This is the only embedder-construction site in the engine; both the
`project_memory` tool and any direct `search()` caller go through it. The result
is cached process-wide (existing `_embedding_cache`), so the local-only
resolution happens at most once per process.

### Why not a global env var
Setting `os.environ["HF_HUB_OFFLINE"]="1"` / `TRANSFORMERS_OFFLINE=1` would force
offline for the ENTIRE process and anything else that might legitimately use HF
on the host — exactly what the user forbade. The `model_kwargs` /
`local_files_only` argument is the surgical, instance-scoped equivalent.

### Behaviour on a genuinely-missing model (honest failure, never a hang)
With `local_files_only=True`, an absent model raises a clear `OSError`
immediately instead of hanging on the network. `search()` already wraps load
failures; `project_memory` surfaces a readable "Project memory not available …"
message. So the worst case becomes a fast, explicit error the agent can act on —
never an indefinite `status=running` hang. (A separate, optional hardening —
moving model materialization off the turn's await path / adding a timeout — is
out of scope here; forcing offline already removes the only network dependency,
which is the actual incident trigger and the user's explicit requirement.)

---

## 4. TDD plan

Unit tests (`semantic_search/tests/`), no mocks of the unit under test:
- The embedder is constructed with `local_files_only=True` in `model_kwargs`
  (assert on the kwargs passed to a real-signature spy / by constructing through
  the real path and asserting no network). Concretely: patch the network layer
  (`huggingface_hub` HTTP) to RAISE on any call, then assert `search()` against a
  locally-indexed repo with the cached model still succeeds — proving zero
  network. (We patch the network boundary, not the unit under test.)
- Negative: with the network boundary raising AND a non-cached model name,
  `search()` surfaces `IndexNotFoundError`/load error fast, does not hang.
- Regression-guard: assert the construction passes `local_files_only=True`
  (string/kwarg check) so a future refactor can't silently re-enable network.

Integration test (no mocks) — see §5.

---

## 5. Integration test (no mocks)

Hermetic, real engine, real local cache, network hard-disabled at the socket
layer (so the test proves "never touches the network" rather than trusting a
flag):
- Monkeypatch `socket.socket.connect` (or `huggingface_hub`'s session) to raise
  on ANY outbound connection — a real network kill switch, not a mock of our code.
- Run `search(query, repo_path=<a real indexed repo>, top_k=…)` using the real
  `HuggingFaceEmbedding` + real lancedb index + the already-cached e5 model.
- Assert: results come back correctly AND no outbound connection was attempted.
- Place under `semantic_search/tests/integration/`; skip only if the model is
  not present in the local cache (same guard the existing leak test uses).

---

## 6. Files touched (planned)

- `semantic_search/semantic_search/retriever.py` — add
  `model_kwargs={"local_files_only": True}` to the single
  `HuggingFaceEmbedding(...)` construction (no `cache_folder`); a short comment
  pointing here.
- NEW `semantic_search/tests/...` unit test(s) for the offline construction.
- NEW `semantic_search/tests/integration/...` no-mock network-kill test.
- docs: this file.

No change to public APIs, the index format, the cache design, agent/provider
code, or any global env. Scope is exactly the embedder used by `project_memory`.

---

## 7. Open questions for the user — RESOLVED

1. **Cache location → RESOLVED.** Rely on llama_index's default cache dir; do NOT
   pin `cache_folder` and do NOT add a config/env override. Live-verified the
   model is found with `local_files_only=True` alone. (User agreed to the
   recommendation; this is also an agent's safest option.)
2. **Missing-model behaviour → RESOLVED.** Absent model fails fast with a clear
   `OSError`, surfaced by `project_memory` as "Project memory not available …".
   Accepted as the terminal behaviour — no extra pre-flight check, no custom
   message. (User agreed.)
