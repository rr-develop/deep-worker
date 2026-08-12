# DESIGN: per-agent token and cache usage statistics

**Status:** DESIGN SIGNED OFF v5 (see §12)
**Scope:** `deep_worker_cli` (collection/storage/aggregation), `claude_code_provider` /
`codex_provider` / `grok_provider` (usage export), the web app (API + UI)

---

## 0. Summary

The system collects for each agent:

- **total number of requests** — each LLM call = one request (including every step
  of the tool loop within a single agent turn);
- **token counts** (input + output, total);
- **how many tokens hit the cache** (cache hit) and **how many did not** (cache miss);
- **the ratio of cached tokens to total tokens** (cache ratio);
- **the average number of tokens per request** (prompt) and **per response**
  (completion).

**The accounting unit and its guarantee (BLOCKER-2/3 from review v2) differ for the
two provider families and are fixed explicitly:**

| Family | Record unit | Guarantee | Mechanism |
|---|---|---|---|
| **LangChain providers** (deepseek, openai, anthropic, google, vertexai) | **one model invocation** (each `ainvoke`/`astream`, including the tool loop) | **exactly-once per observed invocation** | `AsyncCallbackHandler` (`TokenStatsCallbackHandler`) attached to the actual model instance; `run_id` = invocation key; `on_llm_start`→pending, `on_llm_end`/`on_llm_error`→closes exactly one record |
| **Custom providers** (claude-code, codex) | **one terminal provider event** (`ResultMessage` / `TurnCompletedEvent` / terminal failure) | **at-most-once per observed terminal event** | event-driven recording; tool sub-turns between two terminal events are packed into one record; state machine in §5.2 |

The old post-run points (`log_token_usage`) remain only for the compatibility
text log and are NOT a source of accounting.

Statistics are kept **per agent**: both **per provider** (an agent can be
restarted with different providers) and **aggregated across all providers**.
The provider/model are taken from the model actually invoked at invocation
time (after failover), not from a stale config.

The web app gains a "📊 Stats" button — clicking it opens **a separate browser
tab**; the page shows the agent's statistics snapshot **as of the `fetch` after
the tab opens** (the actual snapshot moment is the API request on page load,
not the button click; this wording is a minor from review v2).

---

## 1. Motivation

Agents run on paid LLM providers where the input price depends directly on
cache hit/miss (DeepSeek: hit ≈ $0.0028/Mtok vs miss ≈ $0.14/Mtok — a ~50×
difference). Today token usage is only logged to the text log
(`log_token_usage`); it is not persisted and not shown in the web UI. We need
a cumulative data source that survives agent restarts and a simple way to view
it in the web app.

---

## 2. Current state (analysis)

### 2.1 LLM call sites in the agent

All LLM calls happen inside the agent process (`deep_worker_cli`):

| Call site | Where | Comment |
|---|---|---|
| Agent main loop, stream | `deep_worker_cli/process/agent_process.py` `_handle_astream` | `log_token_usage(..., "stream")` already called |
| Agent main loop, non-stream | `agent_process.py` `_handle_invoke`, 2 places | `log_token_usage(..., "invoke")` already called |
| TaskManager (manager agent) | `deep_worker_cli/task_manager/task_manager.py` `_evaluate_with_manager` | `log_token_usage(..., "tm")` already called |
| Summarization (compaction) | middleware `deepagents/graph.py` `SummarizationMiddleware` | **INCLUDED in accounting** (see §5.4, MAJOR-4) |

**Problem with the existing points as an accounting source:** `agent_process.py:3552-3561`
and `task_manager.py:484-505` are called AFTER the graph run completes and see only
the last `AIMessage` (`messages[-1]`). Within a single agent turn the model is
called several times (tool loop: model → tool → model → …), and each such
call is a separate paid request to the provider. Post-run points systematically
undercount both requests and tokens. They therefore remain only for the text
log, while accounting moves to the model-invocation boundary (see §5).

### 2.2 What already exists: `log_token_usage`

`deep_worker_cli/utils.py:39` — reads `response_metadata["token_usage"]` from the
last `AIMessage`; understands two cache formats:
- `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` (DeepSeek, top-level);
- `prompt_tokens_details.cached_tokens` (OpenAI-compatible).

Writes **to the log only** (INFO), stores nothing. For providers without
`token_usage` in `response_metadata` (claude-code, codex, grok) it silently returns.

### 2.3 What providers currently return (usage lifecycle)

- **deepseek / openai / anthropic / google / vertexai** — standard LangChain
  models; usage arrives on EVERY model invocation, but **in two different
  forms depending on the call path** (live-verified 2026-08-03, deepseek-chat
  and deepseek-v4-flash, with and without tool_calls):
  - **non-streaming** (`ainvoke`; this is how the LangGraph node calls the
    model, `langchain/agents/factory.py`) → `AIMessage.response_metadata["token_usage"]`
    with the provider form (`prompt_tokens`/`completion_tokens`/
    `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` etc.);
  - **streaming** (`astream`) → `response_metadata["token_usage"]` is **completely
    absent** (`None`); usage exists ONLY in the cross-provider aggregate
    `AIMessage.usage_metadata` (`input_tokens`/`output_tokens`/`total_tokens` +
    `input_token_details.cache_read`).

  Both forms MUST be normalized (see §5.1). Ignoring the second form is the
  cause of the 2026-08-03 incident (§12): 196/196 records went to `no_usage`.

  Separately (a policy, NOT the incident cause): `stream_usage=True` is set
  explicitly in `create_model` for deepseek/openai. `stream_options={"include_usage":
  true}` goes on the wire only when `self.stream_usage` is enabled, and
  `langchain_openai` auto-enables it only when `openai_api_base is None`
  (`chat_models/base.py`); we always pass an explicit `api_base`. It was
  live-verified that DeepSeek returns usage even without the flag, so the flag
  is a guarantee for endpoints/configurations that would not send usage without it.
- **claude-code** (`claude_code_provider/chat_model.py`) — usage arrives in
  `ResultMessage.usage` (raw `dict[str, Any] | None`, `types.py:1017`; Anthropic
  format: `input_tokens`/`cache_read_input_tokens`/`cache_creation_input_tokens`/
  `output_tokens` — the keys are untyped, to be confirmed on a live CLI). In the
  tool branch the provider RETURNS `AIMessage(tool_calls=...)` BEFORE
  `ResultMessage` (you cannot wait for it — deadlock, `chat_model.py:1183-1190`);
  `ResultMessage` arrives later (after the tool result is sent). `ResultMessage`
  has stable `session_id` and `uuid` (`types.py:1014,1023`) — the dedup key
  (BLOCKER-2).
- **codex** (`codex_provider/chat_model.py`) — usage arrives in
  `TurnCompletedEvent.usage` (`input_tokens`, `cached_input_tokens`,
  `output_tokens`, `reasoning_output_tokens`). In the tool branch the provider
  exits to `pending_tool_calls` BEFORE `TurnCompletedEvent` (`chat_model.py:741-774`);
  the event arrives in the next `codex exec resume`. **`TurnCompletedEvent`
  contains ONLY `usage` + `type`** (`exec_runner.py:84-86`) — the event has NO
  stable id → an honest dedup strategy for codex (BLOCKER-2).
- **grok** (`grok_provider/chat_model.py`) — **live-red-discovery (2026-08-01,
  grok CLI 0.2.73)**: the ACP `usage_update` notification does NOT arrive in
  practice (among kinds there are only `agent_message_chunk`,
  `agent_thought_chunk`, `available_commands_update`, `user_message_chunk`).
  Per-turn usage arrives in the **RESULT of the `session/prompt` method → `_meta`**:
  `{"totalTokens", "inputTokens", "outputTokens", "cachedReadTokens",
  "reasoningTokens"}`. The `usage_update` branch in `_apply_session_update`
  (`chat_model.py:912-936`) is kept as a fallback; the main path is emission
  from the `_meta` RESULT in `_finalize_response` (§5.2). Cache is present
  (`cachedReadTokens` → cache hit).

---

## 3. Provider research: availability of token and cache data

Verified against provider documentation (and confirmed by the project code).

| Provider | Data channel | Usage fields | Cache hit | Cache miss (canonical prompt) |
|---|---|---|---|---|
| **deepseek** (non-stream) | `response_metadata.token_usage` | `prompt_tokens`, `completion_tokens`, `total_tokens` | `prompt_cache_hit_tokens` | `prompt_cache_miss_tokens` (prompt = hit + miss) |
| **deepseek / openai (stream)** | `AIMessage.usage_metadata` (live 2026-08-03; `token_usage` absent when streaming) | `input_tokens`, `output_tokens`, `total_tokens` | `input_token_details.cache_read` | `input_tokens − cache_read` (prompt = hit + miss) |
| **openai** (non-stream) | `response_metadata.token_usage` | `prompt_tokens`, `completion_tokens` | `prompt_tokens_details.cached_tokens` | `prompt_tokens - cached` |
| **anthropic** | `response_metadata.token_usage` | `input_tokens`, `output_tokens` | `cache_read_input_tokens` | `input_tokens + cache_creation_input_tokens` |
| **google / vertexai** | `response_metadata.usage_metadata` | `prompt_token_count`, `candidates_token_count`, `total_token_count` | `cached_content_token_count` | `prompt_token_count - cached` |
| **claude-code** | `ResultMessage.usage` (raw dict) | `input_tokens`, `output_tokens` | `cache_read_input_tokens` | `input_tokens + cache_creation_input_tokens` |
| **codex** | `turn.completed.usage` (Codex CLI) | `input_tokens`, `output_tokens` | `cached_input_tokens` | `input_tokens - cached_input_tokens` |
| **grok** | `session/prompt` RESULT `_meta` (live 0.2.73) | `totalTokens`, `inputTokens`, `outputTokens`, `reasoningTokens` | `cachedReadTokens` | `inputTokens − cachedReadTokens` |

**Canonical prompt (BLOCKER-3 v1):** for Anthropic/Claude Code
`cache_read_input_tokens` and `cache_creation_input_tokens` usually do NOT fall
inside the regular `input_tokens`, therefore:
`prompt = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`,
`hit = cache_read`, `miss = input + cache_creation`. Otherwise `hit` can exceed
`prompt`, and ratios/total become wrong. For the remaining providers prompt =
the value of the provider's prompt field. Normalization invariant: **hit + miss = prompt**.

Findings:

1. **DeepSeek and Claude Code — confirmed**: cache data is present. OpenAI/
   Anthropic/Gemini — also (for models with prompt caching).
2. **Grok — confirmed (live 0.2.73)**: per-turn usage exists in the `session/prompt`
   RESULT `_meta` (`inputTokens`/`outputTokens`/`cachedReadTokens`/`totalTokens`),
   cache hit = `cachedReadTokens`, miss = `inputTokens − cachedReadTokens`.
   The ACP `usage_update` (context gauge) does not arrive in practice — the
   `_apply_session_update` branch is kept as a fallback; `context_snapshot` is
   also a fallback when `_meta` usage is absent.
3. **claude-code / codex usage arrives with a delay** relative to the `AIMessage`
   return (tool branches) → recording happens event-driven when the terminal
   event arrives (state machine §5.2).

---

## 4. Data model

### 4.1 UsageRecord (one JSONL line = one accounting unit, see §0)

```json
{
  "schema_version": 1,
  "record_id": "3f2a…c9",                 // uuid4 hex — RECORD id (not the dedup key)
  "event_key": "claude-code|sess_123|msg_456",  // stable event key; null = not deduplicated (BLOCKER-1 v3)
  "session_cumulative": null,             // claude-code cumulative branch: raw session counters (baseline for the delta)
  "ts": "2026-08-01T12:34:56.789Z",       // record time (UTC)
  "provider": "deepseek",                 // from the model actually invoked
  "model": "deepseek-v4-flash",           // resolved model at call time
  "kind": "agent",                        // "agent" | "tm" | "other"
  "status": "ok",                         // ok | no_usage | failed (see below)
  "usage_semantics": "tokens",            // tokens | context_snapshot (fallback for grok without _meta)
  "provider_turn_id": null,               // where available: the provider's turn id
  "provider_session_id": "sess_123",      // claude-code: ResultMessage.session_id — ALWAYS (cumulative baseline); otherwise null
  "prompt_tokens": 3000,                  // canonical prompt (see §3); null when usage is absent
  "completion_tokens": 100,
  "cache_hit_tokens": 2500,               // null if the provider gives no cache
  "cache_miss_tokens": 500,
  "total_tokens": 3100                    // prompt + completion
}
```

- **`status` (MAJOR-2 from review v2):**
  - `ok` — usage obtained (tokens non-null);
  - `no_usage` — a terminal event arrived but usage is absent (tokens = null);
  - `failed` — terminal failure (TurnFailedEvent / usage-limit / refusal /
    a hard error that ended the exchange);
  - **`cancelled` is written ONLY for LangChain providers** (the callback
    `on_llm_error`/cancellation on an invocation). For custom providers
    cancellation is NOT recorded: there is no terminal event, the exchange is
    abandoned, there is nothing to attribute the status to — honest semantics
    (BLOCKER-2).
- `usage_semantics="context_snapshot"`: the record carries a context gauge —
  the separate `context_used` / `context_size` fields (instead of prompt/completion).
  Such records are **excluded from token aggregates and ratios** and shown in a
  separate section. For grok this is the fallback: the main path is tokens from
  the `_meta` RESULT (§5.2); a snapshot is written only when per-turn usage is
  absent (e.g. the `usage_update` branch with gauge data).
- **`event_key` (BLOCKER-2 from v2, BLOCKER-1 from v3)** — the stable event key
  for deduplication:
  - claude-code: `"claude-code|<session_id>|<uuid>"` — **ONLY when `uuid` is
    non-empty** (`ResultMessage.uuid` is typed `str | None`, `types.py:1023`).
    **Never deduplicate by `session_id|None`** — otherwise all
    `ResultMessage` records of one session without a uuid collapse into
    duplicates and the statistics are lost. If `uuid` is empty/absent →
    `event_key = null` (the record is considered unique; no dedup on it);
  - codex: `"codex|<thread_id>|<final_agent_message_id>"` — the id of the last
    `AgentMessageItem` before `TurnCompletedEvent` in the same processing loop
    (`AgentMessageItem.id`, `exec_runner.py:96`); if there was no
    `AgentMessageItem` before `TurnCompletedEvent` — `event_key = null` (we do
    not create an unstable key);
  - LangChain providers: `"langchain|<run_id>"` (run_id from the callback);
  - grok: **`event_key = null` always** — a timestamp is not a stable
    dedup key, and the `_meta` RESULT has no provider id (MINOR from v3).
  - **Dedup rule:** the aggregator deduplicates ONLY non-null `event_key`.
    Records with `event_key = null` are always considered unique.
- **`session_cumulative`** (BLOCKER-2 from v3): only for claude-code in the
  cumulative branch (see §5.2) — raw cumulative counters as returned by
  `ResultMessage` (`prompt_tokens`/`completion_tokens`/`cache_*` at event
  time). Used as a crash/restart-safe baseline for the delta.
  In the per-terminal branch the field is absent/null.
- **`provider_session_id`** (BLOCKER from review v5): an explicit provider
  session id — a separate field, it does NOT overload `provider_turn_id`. For
  claude-code it is written **always**: `ResultMessage.session_id`, regardless
  of `uuid`/`event_key` (including `uuid=None`/`event_key=null`). The
  cumulative-delta baseline is looked up by it (§5.2). For the remaining
  providers — `null` if no stable session id is available (codex: thread_id is
  a candidate at implementation time; grok / LangChain: none).
- `record_id` (uuid4) — unique, but **does not deduplicate**; dedup is by
  `event_key` (see §4.3 idempotent aggregation).
- `schema_version` — 1. The reader accepts only known versions.

### 4.2 Storage

- **File:** `<AgentProcess.data_dir>/token_stats.jsonl` — **the path is taken
  from the agent's actual `self.data_dir`** (`agent_process.py:559`), not
  recomputed as `~/.deep_worker/<name>` (MAJOR-3 from v1).
- **Permissions (MAJOR-3 from v2):** after `os.open` **explicitly
  `os.fchmod(fd, 0o600)`** — the mode passed to `os.open` applies only at
  creation and does NOT fix the mode of an existing file. The directory is
  `0o700` (`ensure_private_dir`).
- **Append contract (MAJOR-3 from v2):** `os.open(path, O_WRONLY|O_APPEND|O_CREAT)`
  + `os.fchmod(fd, 0o600)` + **one `os.write(fd, line.encode())`** with a return
  check: if `written != len(line)` — **a short write counts as corruption** of
  the line: close the fd, do NOT try to append uncontrolled (without a
  cross-process lock this could interleave lines); the reader will mark the line
  as partial/malformed (`skipped_records`/`partial_tail`). Inside
  `asyncio.to_thread` (the existing executor pattern; we create no new threads).
- **Lock guarantee (MAJOR-3 from v2):** a per-process `asyncio.Lock` serializes
  writes WITHIN the process; between processes (old/new after a restart) the
  guarantee is given ONLY by `O_APPEND` (offset atomicity). The contract
  promises nothing more.
- **fsync policy:** no fsync per write — on a crash the last in-flight record
  may be lost (the same reliability class as `session_metadata.json`; this is
  statistics, not billing). The policy is fixed in the recorder's docstring.
- **Rotation: deferred** (§10.4). No code/untested hooks
  (`MAX_STATS_FILE_BYTES`) are added in v1.

### 4.3 Aggregates (per agent, per provider, total)

A bucket = agent × provider (with a nested breakdown by model, see §6.1).
Records with `usage_semantics="context_snapshot"` (grok fallback) do not enter
the token aggregates (separate section).

Notation: `T_all` = Σ total over records with tokens; `P_all` = Σ prompt over
records with tokens; `H_known` = Σ hit, `P_known` = Σ prompt, `T_known` = Σ total
over records WITH cache data (`cache_hit_tokens != null`); `P_unknown` = Σ prompt
over records without cache data.

| Metric | Formula | Note |
|---|---|---|
| requests | `count(all records)` | each accounting unit (§0); for custom providers = terminal events |
| requests_with_usage (MAJOR-2) | `count(records: usage_semantics="tokens" AND tokens non-null)` | usually `status="ok"`; does NOT include no_usage |
| usage_coverage (MAJOR-2) | `requests_with_usage / requests` | share of requests with observed usage (null when requests=0) |
| prompt_tokens | `Σ prompt_tokens` | over records with tokens |
| completion_tokens | `Σ completion_tokens` | |
| total_tokens | `Σ total_tokens` | |
| cache_hit_tokens | `Σ cache_hit_tokens` | over records with cache data |
| cache_miss_tokens | `Σ cache_miss_tokens` | |
| **cache_ratio_total_observed** (BLOCKER-1) | `H_known / T_known` | conditional ratio over cache-observed records; "observed", NOT "lower bound" |
| **cache_ratio_total_lower_bound** (BLOCKER-1) | `H_known / T_all` | the true lower bound (unknown hits = 0) |
| **cache_ratio_total_upper_bound** (BLOCKER-1) | `(H_known + P_unknown) / T_all` | the true upper bound (all unknown prompt hit the cache; completion is not cached in the denominator) |
| cache_ratio_prompt_observed | `H_known / P_known` | secondary metric (spec invariant hit+miss=prompt) |
| **cache_coverage** | `P_known / P_all` | what share of prompt traffic actually observes the cache |
| avg_tokens_per_usage_observed_request (MAJOR-2) | `Σ prompt / requests_with_usage` | "per request" — only over requests WITH usage |
| avg_tokens_per_usage_observed_response (MAJOR-2) | `Σ completion / requests_with_usage` | "per response" — only over requests WITH usage |

**Why `observed` ≠ `lower bound` (BLOCKER-1):** `H_known / T_known` is a
conditional ratio on the cache-observed sample. If unknown traffic has 0 hits —
the overall ratio `H_known / T_all` is lower; if unknown traffic caches well —
higher. Therefore the API returns **both** metrics: `observed` (what was
computed) and the honest `lower_bound` / `upper_bound` of the overall ratio,
with `cache_coverage` alongside. In the UI the metrics are labelled
"observed", "lower bound", "upper bound", "coverage".

**A provider without cache data (grok `context_snapshot` fallback / unknown
schemas):** ratios = `null` (UI: "n/a"), not "0"; `has_cache=false`. For grok
the main path (tokens from the `_meta` RESULT) **does have** cache data
(`cachedReadTokens`), so `has_cache=true` in ordinary grok records.

**Null/0 rules:** division by zero → ratio = `null`; sums with no records = `0`;
averages when `requests_with_usage == 0` = `null`.

**Idempotent aggregation (BLOCKER-2 v2, MINOR v3):** the reader/aggregator
deduplicates records by **non-null `event_key`** (claude-code: `session_id|uuid`
with a non-empty uuid; LangChain: `run_id`). For codex and grok (no stable
key → `event_key = null`) no dedup is performed — the at-most-once per
observed terminal event guarantee is provided by the fact that terminal events
are consumed from the CLI stream exactly once and are not replayed after a
restart (see §5.2). **Duplicates are counted by a SEPARATE `duplicate_records`
counter** — they are not mixed with `skipped_records` (malformed/schema is a
different data quality).

**The `context` section (grok fallback, MINOR from v2/v3):** aggregating several snapshots —
**summation is not allowed** (a gauge is cumulative). Shown: `snapshots` (count),
`latest` (the last by **parsed UTC `ts` with a deterministic tie-break by record
order in the file** — not lexicographically over an unvalidated string),
`max_context_used` / `max_context_size` over the period.

---

## 5. Design: collection on the agent side (deep_worker_cli)

### 5.1 Instrumentation: a single mechanism chosen (BLOCKER-3 from v2, MAJOR 1-3 from v3)

**Choice: `TokenStatsCallbackHandler(AsyncCallbackHandler)`, attached to the
actual model instance in `create_model`** (`config.py:1448`). A Runnable
wrapper is REJECTED: it loses the `BaseChatModel` contract (`bind_tools`,
`with_structured_output`), breaks the existing checks
`isinstance(model, ClaudeCodeChatModel)` (`agent.py:600`), `_provider_of_model`
by `type().__name__` (`agent.py:744`), `setattr(model, '_working_dir', …)`
(`agent_process.py:1218-1224`, `task_manager.py:223-227`),
`set_resolved_model_callback` (`agent_process.py:1242`), `profile`
(`config.py:1792`).

**Precisely `AsyncCallbackHandler`, not `BaseCallbackHandler` with async methods
(MAJOR-1 from v3):** the project already has this pattern —
`StatusTrackingCallback(AsyncCallbackHandler)` (`deep_worker_cli/process/
status_callback.py:14`). An async handler is needed for
`await recorder.record(...)`; otherwise the behavior of coroutine callbacks
depends on LangChain's internal dispatch and you can easily get an un-awaited
coroutine / executor mismatch.

`TokenStatsCallbackHandler(AsyncCallbackHandler)`:

- **run_id = invocation key.** `on_llm_start` puts `run_id` into a pending-set;
  `on_llm_end`/`on_llm_error` takes it out and **closes exactly one record**.
  It fires on EVERY model invocation inside the tool loop (including stream —
  LangChain delivers the full `LLMResult` even in stream mode), exactly once.
- **Usage from the result:** the source order is —
  `message.response_metadata["token_usage"]` → `response_metadata["usage_metadata"]`
  → **`AIMessage.usage_metadata`** → `LLMResult.llm_output`; then
  `normalize_usage` (§5.3). Message-level `usage_metadata` is mandatory: on the
  streaming path `token_usage` is completely absent from `response_metadata` and
  it is the ONLY token source (§2.3, incident §12). The seam
  `message.usage_metadata → normalize_usage → record` is locked in by a separate
  test (`test_stats_callback_usage_metadata.py`), because normalize-only tests
  do not cover it.
- **Error/cancel (MAJOR-3 from v3):** `on_llm_error` distinguishes statuses via
  `isinstance(error, asyncio.CancelledError)` → `status="cancelled"`, otherwise
  `status="failed"`. **The pending run_id is removed on BOTH paths** (test §8).
- **Async scheduling:** `async on_llm_end`/`async on_llm_error` awaited call to
  `recorder.record(...)` (per-process `asyncio.Lock` + `asyncio.to_thread` +
  atomic append §4.2). **There is no buffer** — the write is performed and
  awaited immediately, so no separate flush-on-shutdown is needed (documented in
  the docstring). **Contract (MAJOR-3 from v3):** `on_llm_end` does not return
  before the append completes; recorder exceptions are logged and NOT re-raised —
  they must not break the LLM result. Tests §8.
- **kind main/TM:** `create_model` receives the optional parameters
  `kind: str | None` and `stats_callback_factory(kind, *, provider, model_name) ->
  AsyncCallbackHandler` (for standard providers) and
  `usage_event_callback_factory(kind, *, provider, model_name) ->
  UsageEventCallback` (for custom ones; injected via
  `model.set_usage_event_callback`); both call sites
  (`agent_process.py:1218-1224`, `task_manager.py:223-227`) pass
  `kind="agent"` / `kind="tm"`. **A single point** — both paths use one
  `create_model`. The factories are called AFTER the provider/model resolution
  and receive resolved values (not `settings`/config) — see below.
- **Resolved provider/model after failover:** the handler reads
  `provider`/`model` from the model instance it is attached to, at invocation
  time (not from `settings`/config) — after a hot failover every invocation
  writes to its own bucket.

**Custom providers (claude-code / codex / grok): dependency-inversion, WITHOUT a
LangChain handler (BLOCKER-3 v2, MAJOR-2 from v3):**

- **In `create_model` by provider branch:** standard LangChain providers
  (deepseek/openai/anthropic/google/vertexai) get `TokenStatsCallbackHandler`;
  **claude/codex/grok do NOT get it at all** — only
  `set_usage_event_callback(cb)`. A double source is **constructively**
  impossible (no private flag/weakref check in the handler — the callback event
  does not carry a model instance, such a check would be fragile). If a common
  callback list is needed for other purposes, only `TokenStatsCallbackHandler`
  is excluded from it.
- Providers do NOT import the `deep_worker_cli` recorder.
  `set_usage_event_callback` — **explicitly an async Callable**; the provider
  **must `await` it on the terminal event** (state machine §5.2). AgentProcess
  (or `create_model`) injects `recorder.record` as the callback.
- **Best-effort contract (MAJOR-3 from v3):** recorder errors (e.g. a disk
  error on append) are logged and **do NOT turn a successful provider turn into
  a failure** — otherwise a disk error would break the agent. The record is
  skipped, the agent keeps working. If a callback/pending remains without a
  terminal hook because the process was cancelled — a record is **not promised**.

### 5.2 Custom providers: record state machine (BLOCKER-2 from review v2)

**Review question: "is the next LangGraph ainvoke a new paid request or a
continuation of the previous provider turn?" — for accounting this is NOT
distinguishable from the provider data** (claude-code/codex do not return
per-sub-turn usage). Therefore the record unit for custom providers is the
**terminal event**, not the ainvoke. Tool sub-turns between two terminal events
are packed into one record (this is the documented granularity, §0).

State machine (per provider, per thread):

```
state = { pending_invocation: bool, session_key: str|None, baseline_loaded: bool }
# the baseline for the cumulative delta (branch B, §5.2) is NOT in memory —
# lazy-loaded from JSONL by provider_session_id at recorder initialization
# (crash/restart-safe, BLOCKER-2 v3; refined by review v5)

ainvoke START                    → pending_invocation = true        (no record written)
ainvoke END, tool branch         → pending_invocation stays true    (no record written)
                                   # usage will arrive later, in the terminal event
terminal event WITH usage        → ONE record written status="ok"
  (claude: ResultMessage;          covers ALL pending invocations since the last
   codex: TurnCompletedEvent)      terminal event; pending = false
terminal event WITHOUT usage      → ONE record written status="no_usage" (tokens null)
terminal failure                  → ONE record written status="failed"
  (codex: TurnFailedEvent;         pending = false
   claude: usage-limit/refusal/hard error
   that ended the exchange)
CANCELLED (user)                  → no record written (no terminal event;
                                   exchange abandoned; honest semantics, §4.1)
```

- **When no_usage/failed are admissible despite possible delayed usage:** only
  at the moment of the terminal event (usage arrived empty) or of the terminal
  failure — i.e. when we ALREADY hold the terminal event and know that delayed
  usage for this group of invocations will not come anymore. Between `ainvoke`
  and the terminal event **no** status records are written (no double counting).
- **Cumulative contract (claude-code, BLOCKER-2 from v3):** on a live CLI it is
  verified whether `ResultMessage.usage` is cumulative within a session
  (`num_turns`, `session_id` are available, `types.py:1013-1014`). **Both
  contract branches are fixed NOW** (the branch is chosen by a live-probe at
  the start of implementation — RED discovery; the other branch does not
  remain in the code):
  - **Branch A — per-terminal usage (the probe showed usage per single event):**
    write the usage as-is. **The delta branch is removed entirely** — no
    baseline, no `session_cumulative`. The simplest case.
  - **Branch B — cumulative usage (the probe showed session-level
    cumulativity):** each record stores the raw cumulative counters in
    `session_cumulative` (as returned by `ResultMessage`) and
    `provider_session_id` (`ResultMessage.session_id`). **The baseline for the
    delta is persisted in the records themselves**, not in memory: at recorder
    initialization it **lazy-loads from JSONL the last valid physical record**
    with `provider=claude-code`, the same `provider_session_id` and non-null
    `session_cumulative` (the file is append-only and crash/restart-safe).
    **The baseline is chosen by record order in the JSONL (the last matching
    one), NOT by `ts`; malformed/unknown lines are ignored** (non-blocking,
    review v5). **Dedup aggregation (by non-null `event_key`) does NOT affect
    this lookup** — the baseline is found over raw records, regardless of
    uuid/event_key. `delta = current - persisted_baseline`. **Guard on counter
    reset / negative delta:** if ANY component `current < baseline` — treat it
    as a reset of the **whole epoch**: baseline = current, write the full vector
    `delta = current` (a component-wise partial reset would break the invariant
    `hit + miss = prompt`; non-blocking, review v5). The behavior is defined and
    covered by the test in §8. A separate private in-memory baseline state is
    allowed as a cache, but the source of truth is the JSONL records.
  - **A test is mandatory (§8):** persisted old cumulative=100 → restart/resume →
    next cumulative=130 → **30** is written, not 130; counter reset (130→5) →
    5 is written, baseline=5. **Also (BLOCKER review v5):** a persisted record
    with `uuid=None` / `event_key=null` / `provider_session_id=S` /
    cumulative=100 → a new process, the same session S, `uuid=None`,
    current=130 → **delta=30** (the baseline is found by `provider_session_id`,
    not by event_key); a record of another session (S′, cumulative=500) does NOT
    become the baseline.
  For claude-code `ResultMessage.usage` — per-terminal (confirmed by a live-probe
  2026-08-01: one session, three turns, `input_tokens` stable at 2, `cache_read`
  20399→28599→28616, `cache_creation` 8200→17→17 — usage is NOT cumulative,
  §12), no deltas are needed (**branch A** — write usage as-is, without
  `session_cumulative` and baseline; branch B is not implemented).
  For codex `turn.completed.usage` — per-turn (confirmed on the host,
  `DESIGN_CODEX_PROVIDER.md` §4.1), no deltas are needed (branch A).
- **Dedup keys (stable, from actually available fields):**
  - claude-code: `session_id` + `uuid` (`ResultMessage`) — **only with a
    non-empty `uuid`** (`str | None`, `types.py:1023`); with `uuid=None` →
    `event_key = null`, **never deduplicate by `session_id|None`** (BLOCKER-1
    from v3, otherwise all ResultMessage records of one session collapse as
    duplicates);
  - codex: the id of the last `AgentMessageItem` before `TurnCompletedEvent`
    (`exec_runner.py:96`) + `thread_id`; if there was no `AgentMessageItem`
    before the event — `event_key = null` (we do not create an unstable key);
  - both: an in-process seen-set over the non-null key prevents a duplicate
    within the process; `event_key=null` is NOT put into the seen-set.
- **Replay/restart/crash:** terminal events are consumed from the CLI stream
  (stdio/event-stream) exactly once and are NOT replayed after a
  restart/crash (resume does not replay old events) → no duplicates on
  restart. A crash between the event arrival and our append = a lost record
  (at-most-once, documented). A crash between the append and anything else =
  the record is already on disk, the event will not repeat → no duplicate.
- **Code edit points:** claude-code — the `ResultMessage` handler
  (`chat_model.py:1384-1392`); codex — `TurnCompletedEvent` (`chat_model.py:772`);
  both call `usage_event_callback` (injected via `set_usage_event_callback`,
  §5.1). The deadlock invariant `chat_model.py:1183-1190` is NOT affected.

**grok** (`grok_provider/chat_model.py`): **live-red-discovery (2026-08-01,
grok CLI 0.2.73)** showed that the ACP `usage_update` does not arrive in
practice — per-turn usage is in the **RESULT of the `session/prompt` method →
`_meta`** (`totalTokens`/`inputTokens`/`outputTokens`/`cachedReadTokens`/
`reasoningTokens`). Implementation: `_emit_usage_event(..., usage=...)` +
`_finalize_response` emits token-usage from the `_meta` RESULT (main path);
the `usage_update` branch in `_apply_session_update` (`chat_model.py:912-936`)
is kept as a fallback (gauge → `usage_semantics="context_snapshot"`), and
`context_snapshot` is written only when `_meta` usage is absent. Normalization
via `normalize_usage`: grok `_meta` is mapped to the codex form
(`input_tokens`/`output_tokens`/`cached_input_tokens`), cache hit =
`cachedReadTokens`, miss = `inputTokens − cachedReadTokens` (§5.3).

SDK usage objects (e.g. `ResultMessage.usage` — already a raw dict) are **first
converted to a plain dict** (`vars()`/`asdict`/explicit copying) — we do not
assume JSON-serializability (MAJOR-4 from v1).

### 5.3 Normalization (shared by the recorder and the log)

`normalize_usage(provider, usage_dict) -> NormalizedUsage | None`:

- Input — a **plain dict** of provider usage (from
  `response_metadata["token_usage"]`,
  `response_metadata["usage_metadata"]`/`AIMessage.usage_metadata`, or from
  the provider's terminal event). The `response_metadata` of custom providers is
  **not overwritten**.
- Understands the formats from the table in §3; applies the canonical prompt
  (Anthropic: `input + cache_read + cache_creation`); invariant `hit + miss =
  prompt`. For LangChain providers BOTH forms are parsed: first the provider
  `token_usage` (`prompt_tokens`/`completion_tokens`, priority — byte-level
  compatibility with the legacy `log_token_usage`), and only when both metrics
  are absent — the streaming aggregate `usage_metadata` (`input_tokens`/
  `output_tokens`, hit = `input_token_details.cache_read`, miss =
  `input_tokens − cache_read`; without the `input_token_details` cache field
  they stay `null` — we invent nothing). Grok: the `_meta` RESULT
  (`totalTokens`/`inputTokens`/`outputTokens`/`cachedReadTokens`/
  `reasoningTokens`) is mapped to the codex form (`input_tokens`/`output_tokens`/
  `cached_input_tokens`); cache hit = `cachedReadTokens`, miss =
  `inputTokens − cachedReadTokens`.
- Missing cache fields → `cache_hit_tokens=None` (`has_cache=false`).
- Unknown schema/missing usage → `None` (record `status="no_usage"`).

**`log_token_usage` compatibility (MAJOR-4 from v1):** the function is
refactored onto this same normalizer, but **the log-line format and behavior
are preserved byte-for-byte**: empty usage is not logged, the label (`[TM]`,
`[invoke]`, `[stream]`) and the cache/no-cache lines do not change. The
existing 7 tests in `test_token_usage_logging.py` stay green without edits
(semantic + byte-level parity).

### 5.4 Summarization: deliberately included in accounting (MAJOR-4 from review v2)

Verified against the code: `SummarizationMiddleware` in `deepagents` receives
a **bind_tools COPY** of the model — `_bind_tools_for_middleware(model, tools)`
(`deepagents/graph.py:253`, for subagents :312), not the original object
(incident 2026-06-13, graph.py:231-252). LangChain copies the model config
(including `callbacks`) into the bound copy → the summary calls
`self.model.ainvoke(prompt)` from `_acreate_summary` **fall under the same
`TokenStatsCallbackHandler`**.

**Decision:** we include summarization calls in accounting **deliberately** —
they are real paid LLM calls of the same model, and the data boundary is simple
and provable: "all calls of the wrapped model are accounted for". They are
written with `kind="agent"` (the call happens in the context of the main agent)
and enter the overall aggregates. The only remaining separate task is the
**optional breakdown** `kind="summary"` (via run metadata) — see §10.5. Test:
a summary call is present in the file as a record (§8).

### 5.5 What happens to the old post-run points

`log_token_usage` in `agent_process.py` (3 places) and `task_manager.py`
(1 place) **stays as is** — only the text log, for compatibility. It is not a
source of request accounting (BLOCKER-1 from v1).

---

## 6. Design: web app (API + UI)

### 6.1 API: `GET /api/agents/{agent_name}/stats`

- **Contract (MINOR from v2):** the "known agent" criterion — **presence in
  the web service registry** (the same source as `list_agents`: scanning
  `agent_root` + `session_metadata.json` via `AgentService`/`ensure_connecting`).
  Unknown agent → `404`. Known agent without a statistics file → `200` with
  zero aggregates (no directory/file creation from GET).
- `validate_agent_name(agent_name)` — a `ValueError` on an invalid name is
  **converted to `400/422`, not `500`** (MINOR from v2).
- The path is read via `deep_worker_cli.storage_paths.agent_root(agent_name,
  create=False) / "token_stats.jsonl"` after `validate_agent_name(agent_name)` —
  **no raw path join from the URL**. A custom `data_dir` is a known
  limitation (the same as session_metadata today).
- **Reader/aggregator — a SINGLE implementation (MAJOR-1 from v2):** the
  dependency-free schema/reader/aggregate lives in
  `deep_worker_cli/stats/token_stats.py` (pure functions, no heavy
  dependencies); the web app **imports this module** (the web already imports
  `storage_paths` from `deep_worker_cli`). No duplication of formulas.
  Contract tests run against the single module.
- Reader: accepts only `schema_version == 1`; malformed lines and lines with an
  unknown version → into `skipped_records`; a partial trailing line without
  `\n` (concurrent writer / short write) → only the tail is skipped, the flag
  `partial_tail: true`; **duplicates by non-null `event_key` → a SEPARATE
  `duplicate_records` counter** (not mixed with `skipped_records` — MINOR
  from v3).
- Response:

```json
{
  "agent": "my-agent",
  "generated_at": "2026-08-01T12:35:00.000Z",
  "skipped_records": 0,
  "duplicate_records": 0,
  "partial_tail": false,
  "totals": { "requests": 42, "requests_with_usage": 40, "usage_coverage": 0.952,
              "prompt_tokens": 120000, "completion_tokens": 8000,
              "total_tokens": 128000, "cache_hit_tokens": 90000,
              "cache_miss_tokens": 30000,
              "cache_ratio_total_observed": 0.703,
              "cache_ratio_total_lower_bound": 0.699,
              "cache_ratio_total_upper_bound": 0.731,
              "cache_ratio_prompt_observed": 0.75, "cache_coverage": 0.97,
              "avg_tokens_per_usage_observed_request": 3000.0,
              "avg_tokens_per_usage_observed_response": 200.0,
              "has_cache": true, "n_usage": 40, "n_cache_observed": 39 },
  "by_provider": [
    { "provider": "deepseek", "totals": { "...": "as above" },
      "by_model": [ { "model": "deepseek-v4-flash", "totals": { "..." } } ] },
    { "provider": "grok", "totals": { "...": "requests, tokens, cache
      (cachedReadTokens), ratios over cache data" },
      "context": { "snapshots": 3,
                   "latest": { "ts": "...", "context_used": 53000, "context_size": 200000 },
                   "max_context_used": 53000, "max_context_size": 200000 } }
  ]
}
```

- **Grouping:** `by_provider` (aggregates per provider) + nested `by_model`
  (aggregates per model). The model is NOT mixed into the provider row.
- `has_cache` in a bucket = true if there is at least one record with cache data.
- The `context` section — for records with `usage_semantics="context_snapshot"`
  (grok fallback, when per-turn usage from `_meta` is absent): snapshot
  aggregation — **latest by parsed UTC ts (deterministic tie-break by record
  order) + max**, summation is not allowed (MINOR from v2/v3).
- Metric names correspond to §4.3 (`observed`/`lower_bound`/`upper_bound`/
  `coverage`) — BLOCKER-1.

### 6.2 UI: button + tab

- In `static/index.html`, in the agent control panel (next to "📜 Full Dialog"),
  a **"📊 Stats"** button is added; in `static/js/main.js` —
  `openAgentStats()`: `window.open('/static/stats.html?agent=' +
  encodeURIComponent(selectedAgent), '_blank')`.
- A new static page `static/stats.html?agent=<name>`: on `load` it does
  `fetch('/api/agents/<name>/stats')` (via `encodeURIComponent`) and renders
  tables (totals + by_provider + by_model + the context section for grok).
  **Rendering only via `textContent`/DOM API — no innerHTML with data**
  (HTML-escape). A manual "🔄 Refresh" is allowed; **no polling**.
- **Data snapshot — as of the `fetch` on tab load** (wording from a minor of
  review v2: this is actually the page-open moment, not the button-click
  moment).
- The UI shows: `cache_ratio_total_observed` + `lower_bound`/`upper_bound` +
  `cache_coverage` + `cache_ratio_prompt_observed`; for grok records with
  `context_snapshot` (fallback) — "n/a" for ratios and tokens, context
  used/size; ordinary grok token records render like the rest (with cache).

---

## 7. Files

**deep_worker_cli (the source of truth for both collection AND aggregation):**
- `deep_worker_cli/stats/token_stats.py` (new): `UsageRecord` schema, the
  recorder (`TokenUsageRecorder`: per-process lock, atomic append §4.2,
  `set_usage_event_callback` for custom providers), **reader + aggregator §4.3
  (pure functions, dependency-free)** — a single implementation for CLI and web
  (MAJOR-1 from v2).
- `deep_worker_cli/stats/callback.py` (new): `TokenStatsCallbackHandler`
  (BLOCKER-3 from v2).
- `deep_worker_cli/config.py` (`create_model`): attaching
  `TokenStatsCallbackHandler` **only for standard LangChain providers**;
  for claude/codex/grok — only `set_usage_event_callback` (MAJOR-2 from v3);
  an optional parameter for kind.
- `deep_worker_cli/task_manager/task_manager.py`: passing `kind="tm"`.
- `deep_worker_cli/process/agent_process.py`: recorder initialization with
  `self.data_dir`, callback injection; post-run `log_token_usage` remains.
- `claude_code_provider/chat_model.py`: `set_usage_event_callback` + the call on
  `ResultMessage` (§5.2).
- `codex_provider/chat_model.py`: the same on `TurnCompletedEvent` (§5.2).
- `grok_provider/chat_model.py`: `_emit_usage_event(..., usage=...)` from the
  `_meta` RESULT → tokens (§5.2); the `usage_update` branch →
  `context_snapshot` — fallback.

**Web app (imports the single reader/aggregator):**
- `app/routers/agents.py`: `GET /api/agents/{name}/stats` (404/400 contract,
  §6.1).
- `static/stats.html` (new) + `static/js/stats.js` (new).
- `static/index.html` + `static/js/main.js`: the "📊 Stats" button.

---

## 8. Tests

**unit (without LLM/RPyC):**
- `normalize_usage`: all formats of §3 (DeepSeek top-level, OpenAI
  cached_tokens, Anthropic canonical prompt `input+read+creation`, Gemini
  usage_metadata, codex cached_input_tokens), invariant hit+miss=prompt;
  empty/unknown → None.
- `log_token_usage` parity: the existing 7 tests unchanged.
- Recorder: atomic append (O_APPEND), **fchmod 0600 on an existing file**
  (MAJOR-3 v2), **short write → corrupted line, the reader marks it
  partial/malformed** (MAJOR-3 v2), schema_version, event_key.
- **State machine (BLOCKER-2):** event sequences ainvoke/tool/terminal →
  exactly N records without duplicates and without intermediate statuses;
  no_usage/failed only on terminal events; cancelled is not written for custom.
- **event_key = null (BLOCKER-1 from v3):** two `ResultMessage` records of one
  session with `uuid=None` → **two records / two requests** (the aggregator
  does not collapse them as duplicates); mutation test: one with uuid, one
  without — only the first is deduplicated.
- **Cumulative restart (BLOCKER-2 from v3, refined v5):** persisted
  cumulative=100 → restart/resume → next cumulative=130 → **30** is written,
  not 130 (baseline lazy-load from JSONL by `provider_session_id`); counter
  reset 130→5 → 5 is written, baseline=5; **a record with
  `uuid=None`/`event_key=null`/`provider_session_id=S`/cumulative=100 → a new
  process S, `uuid=None`, current=130 → delta=30**; another session S′
  (cumulative=500) does not become the baseline.
- **Callback contract (MAJOR-1/3 from v3):** `on_llm_end` does not return
  before the append completes; a recorder exception is logged and NOT re-raised
  into the LLM result; `on_llm_error` with `asyncio.CancelledError` →
  cancelled, otherwise → failed; **the pending run_id is removed on both
  paths** (no pending-set leak).
- **Cache-ratio math (BLOCKER-1):** observed/lower/upper on synthetic data —
  unknown traffic with 0 hits and with 100% hits; coverage.
- **MAJOR-2:** `requests_with_usage` only over records with tokens;
  `usage_coverage`.
- Aggregation: mixed deepseek+grok → ratios over cache-observed records,
  coverage, bounds; null ratios for grok-snapshot (fallback) records;
  null/0 rules; **context section: latest+max, not sum** (MINOR v2).
- Reader: malformed lines → skipped_records; partial tail → partial_tail;
  **duplicates by non-null event_key → a SEPARATE `duplicate_records` counter**
  (idempotent aggregation; not mixed with skipped_records — MINOR from v3).
- **Contract tests (MAJOR-1 v2):** the same functions are run from the web app
  tests (single implementation).

**integration (real providers, no mocks):**
- **Tool-flow state machine (BLOCKER-2):** one user turn → tool call → tool
  result → final answer; exactly N records by terminal events, without
  gaps and duplicates; claude-code (tool early-return before ResultMessage) and
  codex (tool early-return before TurnCompletedEvent) — separate cases.
- **Cumulative check (claude-code):** RED-discovery done on a live CLI —
  usage is NOT cumulative (one session, three turns, `input_tokens` stable at
  2, `cache_read` grows per-turn, §12). **Branch A is activated**
  (per-terminal, record usage as-is, no baseline/delta; `session_cumulative`
  remains reserved in the schema for future providers but is NOT filled).
  **The 100→130→30 restart test is only for branch B** — NOT implemented; for
  branch A it is tested that: two records of one session are independent events
  (130 as-is), no baseline/delta.
- **Failover/restart (MAJOR-3 v1):** one agent name, a restart with a
  different provider → two correct buckets in one file.
- **Grok tokens (live-red-discovery v5→GREEN):** per-turn usage from the `_meta`
  RESULT → an ordinary token record (provider=grok, prompt/completion/total +
  cache_hit/cache_miss); the **fallback** `usage_update` → `context_snapshot`;
  tokens/ratios = n/a only for snapshot records; latest+max aggregation.
- **Summarization (MAJOR-4 v2):** a summary call lands in the file as a record
  (the callback is inherited by the bind_tools copy).
- **Gate:** live tests are marked `@pytest.mark.integration_ai` and are
  **mandatory in the release job** (required gate, MINOR from v2) — otherwise
  the central at-most-once/state-machine invariants (§5.2) would remain
  optional.

**web UI (playwright, like the existing `test_ui_*`):**
- The "📊 Stats" button is visible; it opens the `stats.html?agent=…` tab.
- `stats.html` renders totals/by_provider without XSS (textContent); grok
  token records render like the rest; grok-snapshot (fallback) — n/a +
  context; labels observed/lower/upper/coverage.
- API: unknown agent → 404; known/no file → 200 zeros; invalid name → 4xx;
  skipped_records on a broken line.

---

## 9. Edge cases

- **Empty/missing file** → 200 with zeros, `has_cache=false`.
- **Partial line / short write** (concurrent writer) → skip tail,
  `partial_tail: true` / `skipped_records` (MAJOR-3 v2).
- **Unknown schema_version / malformed** → `skipped_records` (a counter, not a
  loss).
- **Duplicates by non-null event_key** → `duplicate_records` (idempotent
  aggregation, separate from skipped_records; MINOR from v3).
- **event_key = null (uuid=None on ResultMessage / codex without
  AgentMessageItem / grok)** → the record is unique, no dedup on it (BLOCKER-1
  from v3).
- **Mixed providers** → ratios: observed over cache-observed records + honest
  lower/upper + coverage (BLOCKER-1).
- **failed/cancelled calls (refined by review v5):** only **terminal
  failures** are counted (a record with `status="failed"` — in requests, not
  in tokens). LangChain cancellation is counted by the callback
  (`on_llm_error` → `asyncio.CancelledError` → `cancelled`). **A custom
  exchange without a terminal event is not observable and not counted**
  (cancelled is not written for custom — §4.1/§5.2; no contradiction with the
  state machine).
- **Crash between the event and the append** → loss of the last record
  (at-most-once, documented); a restart does NOT replay terminal events → no
  duplicates.
- **Agent dead / restarted** → the file remains, the web reads from disk
  (no RPyC).
- **Custom `data_dir`** — the recorder writes to it; the web reads the default
  path via `storage_paths` (a limitation, like session_metadata).
- **Grok snapshots (fallback)** — not summed; latest + max (§4.3, MINOR v2).
  The main grok path — token records from the `_meta` RESULT (live 0.2.73),
  cache is present (`cachedReadTokens`).

---

## 10. Resolved open questions — RESOLVED

Decisions are fixed (task owner, 2026-08-01; no user confirmation required).
Refinements from reviews v1–v5 are marked "(refined review v1)" /
"(refined review v2)" / "(review v5)" etc.

1. **Cache ratio (BLOCKER-1 v2):** per the spec — `cache_hit / total_tokens`.
   The API/UI return **three honest metrics**:
   `cache_ratio_total_observed` (H_known/T_known),
   `cache_ratio_total_lower_bound` (H_known/T_all), `cache_ratio_total_upper_bound`
   ((H_known+P_unknown)/T_all) + `cache_coverage` and
   `cache_ratio_prompt_observed`. The word "lower bound" is not applied to the
   observed metric (mathematically incorrect).
2. **Tab format** — a static `stats.html` + a JSON endpoint. Server-rendered
   HTML (the `dialog-text` style) is not used.
3. **Grok (refined by live-red-discovery 2026-08-01)** — the main path:
   `usage_semantics="tokens"` from the `session/prompt` RESULT `_meta`
   (`totalTokens`/`inputTokens`/`outputTokens`/`cachedReadTokens`), cache hit =
   `cachedReadTokens`, miss = `inputTokens − cachedReadTokens`, `has_cache=true`.
   Fallback: `usage_semantics="context_snapshot"` (`context_used`/`context_size`),
   `has_cache=false`, UI "n/a"; snapshots are not summed (latest + max).
   (Refined by review v1: a gauge is not mixed with token aggregates; refined
   v2: snapshot aggregation; live probe 0.2.73: `usage_update` does not arrive —
   the `_meta` RESULT is the main path.)
4. **JSONL rotation** — deferred; no code/hooks in v1.
5. **Summarization (MAJOR-4 v2)** — **INCLUDED in accounting** deliberately:
   summary calls go through the same callback (the bind_tools copy inherits
   callbacks) and are written with `kind="agent"` into the overall aggregates.
   The only remaining separate task is the optional breakdown
   `kind="summary"` by run metadata.
6. **Cost ($)** — strictly tokens per the spec; cost is **not added** to the
   schema/UI.
7. **Accounting guarantee (BLOCKER-2 v2)** — LangChain providers: exactly-once
   per observed invocation (callback, run_id). Custom providers: **at-most-once
   per observed terminal event** + idempotent aggregation by `event_key`; the
   record unit is the terminal event (tool sub-turns are packed); cancelled is
   not written for custom.
8. **Single reader/aggregator implementation (MAJOR-1 v2)** — a
   dependency-free leaf in `deep_worker_cli/stats/token_stats.py`; the web
   imports it. Formula duplication is forbidden; contract tests run against
   the single module.
9. **Instrumentation mechanism (BLOCKER-3 v2, refined v3)** —
   `TokenStatsCallbackHandler(AsyncCallbackHandler)` on the actual model
   instance, attached in `create_model` **only for standard LangChain
   providers** (MAJOR-2 v3: custom models do not get a LangChain stats handler
   at all — a double source is constructively impossible, no private flag);
   custom providers — via the async `set_usage_event_callback`
   (dependency-inversion, without importing deep_worker_cli; recorder errors
   are best-effort, logged and do not break the provider turn; MAJOR-3 v3).
10. **claude-code cumulative contract (BLOCKER-2 v3, refined v5)** — a
    live-probe at the start of implementation chooses branch A (per-terminal,
    no delta) or B (delta from a crash/restart-safe baseline persisted in the
    JSONL records via `session_cumulative` + `provider_session_id`;
    lazy-load of the last matching raw record at initialization; guard on a
    whole-epoch reset); both branches are fixed in §5.2, only the chosen one
    remains in the code.

---

## 11. Implementation order (after approval)

1. `deep_worker_cli/stats/token_stats.py`: schema + recorder (append contract,
   fchmod, short-write) + reader/aggregator (single implementation) + unit
   tests (TDD).
2. `normalize_usage` + `log_token_usage` parity tests.
3. `TokenStatsCallbackHandler` + attachment in `create_model` (main + TM,
   kind) + unit tests (run_id lifecycle; structural exclusion of custom models
   without a private flag — MAJOR-2 v3/v4, §5.1/§10.9).
4. Custom providers: `set_usage_event_callback` + state machine (§5.2) +
   integration tests (tool-flow, cumulative check, grok tokens from the `_meta`
   RESULT, summarization).
5. Web: endpoint (404/400 contract), `stats.html`, the button, UI/API tests.
6. Unit tests, code review, commit.

---

## 12. History

- v1 (2026-08-01): the first version of the design; sent for review.
- v1→v2 (2026-08-01): review v1 comments addressed — BLOCKER-1 (accounting
  unit = model invocation, exactly-once), BLOCKER-2 (event-driven usage
  recording for custom providers instead of response_metadata), BLOCKER-3
  (canonical prompt, cache coverage/lower bound), MAJOR-1 (grok gauge →
  context_snapshot), MAJOR-2 (schema_version/record_id, O_APPEND+lock,
  skipped_records, fsync policy, no inactive hooks), MAJOR-3 (path from
  `AgentProcess.data_dir` + `storage_paths`), MAJOR-4 (parity
  `log_token_usage`, plain-dict usage), all minors (404/200, grouping
  by_model, null/0, escape, targeted tests).
- v2→v2.1 (2026-08-01): the exact integration points were checked and fixed —
  `create_model` call sites (`agent_process.py:1218-1224`,
  `task_manager.py:223-227`), the choice of `BaseCallbackHandler` in
  `create_model` instead of a Runnable wrapper, the deadlock invariant
  `chat_model.py:1183-1190`, the `ResultMessage` (:1384) and
  `TurnCompletedEvent` (:772) handlers; SummarizationMiddleware receives a
  bind_tools COPY of the model (`graph.py:253`) — the question of accounting
  for summary calls reduces to the bound copy inheriting the callbacks.
- v2.1→v2.2 (2026-08-01): final fact checks — `ResultMessage.usage` in the
  installed SDK (`types.py:1006-1023`) — raw `dict[str, Any] | None` (has
  `model_usage`, `num_turns`, `session_id`, `uuid`); grok `usage_update` is
  ignored (`HANDLED_SESSION_UPDATE_KINDS`, `acp_types.py:22-31`;
  `_apply_session_update`, `chat_model.py:912-936`).
- v2→v3 (2026-08-01): review v2 comments addressed — BLOCKER-1 (cache-ratio:
  `observed`/`lower_bound`/`upper_bound` + coverage instead of the
  mathematically incorrect "lower bound"), BLOCKER-2 (state machine for custom
  providers: unit = terminal event, at-most-once per observed terminal event,
  stable `event_key` — claude `session_id|uuid`, codex `final
  AgentMessageItem.id`; cancelled not written for custom; behavior on
  replay/restart/crash), BLOCKER-3 (a single `BaseCallbackHandler` on the
  model instance chosen, run_id key, `set_usage_event_callback`
  dependency-inversion, guard against double recording), MAJOR-1 (single
  reader/aggregator implementation in `deep_worker_cli/stats/`; the web
  imports it), MAJOR-2 (`requests_with_usage` semantics = records with tokens;
  `usage_coverage`; avg renamed), MAJOR-3 (fchmod 0600; short-write →
  corrupted line, no uncontrolled appending; the exact lock guarantee),
  MAJOR-4 (summarization deliberately included, `kind="agent"`), all minors
  (context latest+max, snapshot wording, the known-agent criterion + 4xx,
  live-test gate in the release job).
- v3→v4 (2026-08-01): review v3 comments addressed — BLOCKER-1 (`event_key`
  only with a non-empty `uuid`; never session_id|None; uuid=None → two
  records/two requests; mutation test), BLOCKER-2 (claude cumulative contract:
  both branches A/B fixed, the baseline is persisted in `session_cumulative`
  and lazy-loaded from JSONL at initialization (crash/restart-safe), guard on
  counter reset; test persisted=100→restart→130→30), MAJOR-1
  (`AsyncCallbackHandler` — the `StatusTrackingCallback` pattern,
  status_callback.py:14; test: on_llm_end awaits the append, recorder errors
  do not break the LLM result), MAJOR-2 (custom providers do NOT get a
  LangChain stats handler — attachment by provider branch; a double source is
  constructively impossible, no private flag), MAJOR-3 (async
  `set_usage_event_callback`, the provider must await it; recorder —
  best-effort; `asyncio.CancelledError` → cancelled, otherwise → failed; the
  pending run_id is removed on both paths; a process cancellation without a
  terminal hook — no record promised), minor (duplicates →
  `duplicate_records` separate from skipped_records; grok event_key=null
  always; context latest by parsed UTC ts with a tie-break by record order).
- v4→v5 (2026-08-01): review v4 comments addressed — BLOCKER (an explicit
  `provider_session_id` field added: for claude-code `ResultMessage.session_id`
  is ALWAYS written regardless of `uuid`/`event_key`; the cumulative baseline
  is found by the last valid raw record `provider=claude-code` +
  `provider_session_id` + non-null `session_cumulative`, by record order in the
  JSONL, NOT by ts; dedup aggregation does not affect the lookup; test:
  `uuid=None`/`event_key=null`/`provider_session_id=S`/cumulative=100 →
  restart → 130 → delta=30; another session S′ does not become the baseline),
  REQUIRED editorial fixes (§8: the 100→130→30 restart test only for branch B;
  for branch A — no baseline/delta and usage recorded as-is; §9: terminal
  failures are counted, LangChain cancellation via the callback, a custom
  exchange without a terminal event is not observable and not counted),
  non-blocking (baseline by physical record order, malformed ignored;
  reset-guard — a whole-epoch reset and the full current vector, otherwise
  `hit + miss = prompt` would break).
- v5 → SIGNED OFF (2026-08-01): review v5 — **DESIGN SIGN OFF**, no
  architectural/correctness comments. Last nit: the §8 gate — "exactly-once
  invariants" → "at-most-once/state-machine invariants (§5.2)" — applied.
  Next step — RED/TDD; the claude-code branch A/B choice is fixed by the
  result of the live RED-discovery before GREEN (§5.2, §10.10). Implementation
  only after explicit user approval.
- v5 → LIVE RED-DISCOVERY (2026-08-01, implementation §11.1): a claude-code
  live-probe confirmed **branch A** — `ResultMessage.usage` is NOT cumulative
  within a session. Proof: one `ClaudeSDKClient`, one session
  (session_id=2722eb3a…), three consecutive turns (live probe script):
  `input_tokens` stably = 2 on each turn (with cumulativity it would grow
  together with the history), `cache_read_input_tokens` 20399→28599→28616
  (grows — this is per-turn reading of the accumulated context, not a sum),
  `cache_creation_input_tokens` 8200→17→17 (after the first turn the history
  is in the cache, only the new user message is created), `output_tokens`=5
  always, `iterations[]` — one element per turn. Conclusion: usage is per
  terminal event, record as-is, without `session_cumulative` and baseline
  (branch B and its 100→130→30 tests are NOT implemented). Branch A is fixed
  in §5.2 and §10.10.
- v5 → LIVE RED-DISCOVERY (grok, implementation §11.4, 2026-08-01): a live
  probe of grok CLI 0.2.73 showed that the ACP `usage_update` notification
  does NOT arrive in practice (kinds: `agent_message_chunk`,
  `agent_thought_chunk`, `available_commands_update`, `user_message_chunk`);
  per-turn usage arrives in the **RESULT `session/prompt` → `_meta`**:
  `{"totalTokens": 19370, "inputTokens": 19331, "outputTokens": 38,
  "cachedReadTokens": 128, "reasoningTokens": 33}` (grok-4.5,
  totalContextTokens=500000). Cache IS present (`cachedReadTokens`).
  Implementation reworked: `_emit_usage_event(..., usage=...)` +
  `_finalize_response` emits token-usage from the `_meta` RESULT (main path);
  the `usage_update` branch of `_apply_session_update` is kept as a fallback →
  `context_snapshot`; `normalize_usage`: grok `_meta` → codex form (cache hit
  = `cachedReadTokens`, miss = `inputTokens − cachedReadTokens`). The live test
  `test_grok_agent_records_token_usage` is GREEN (1 passed in 26.84s). The
  failover live test was redesigned: automatic failover (usage-limit →
  `_schedule_failover`) is honestly not provocable (a broken model gives
  `model_not_found`, not usage-limit; the claude-code usage-limit marker is
  `"You've hit your session limit"`); the test checks the failover effect —
  a claude-code agent writes a record, then the same agent (the same data_dir)
  is restarted on deepseek → in ONE file there are records from ≥2 providers
  (the recorder is bound to data_dir, not to the provider). GREEN.
  All 6 live tests of §11.4 are green.
- v5 → PRODUCTION INCIDENT (2026-08-03, agent `my_job_analyst`): a live
  agent's statistics turned out to be zero — `Requests: 196`,
  `Requests with usage: 0`, `usage_coverage 0.0%`, all tokens 0. The
  `token_stats.jsonl` file contained exactly 196 lines, ALL with
  `status="no_usage"`, `prompt/completion/total = null`, `provider=deepseek`,
  `model=deepseek-v4-flash`, the `event_key` prefix `langchain|` (the
  LangChain callback path) — i.e. the recorder worked, but usage was not
  extracted. **Root cause — a normalizer gap:** on the streaming path
  (`astream`) the final `AIMessage` does NOT contain
  `response_metadata["token_usage"]` at all; usage arrives only in the
  cross-provider aggregate `AIMessage.usage_metadata`
  (`input_tokens`/`output_tokens`/`total_tokens` +
  `input_token_details.cache_read`). `_from_langchain_normalized` understood
  only the provider `token_usage` form and returned `None` on the aggregate →
  the record degraded to `no_usage`. Proven by live probes (deepseek-chat and
  deepseek-v4-flash, with and without tool_calls: `token_usage=None`,
  `usage_metadata={'input_tokens': 359, 'output_tokens': 66, 'total_tokens':
  425, 'input_token_details': {'cache_read': 256}, ...}`) plus a direct run of
  the old normalizer on this form → `None`.
  **Fix:** `_from_langchain_usage_metadata` added — called as a fallback from
  `_from_langchain_normalized` exactly where there used to be an unconditional
  `return None` (neither `prompt_tokens` nor `completion_tokens`), so the
  priority and byte-level semantics of the legacy `token_usage` path are
  unchanged; cache hit = `cache_read`, miss = `input_tokens − cache_read`
  (invariant `hit + miss = prompt`); without the `input_token_details` cache
  field they stay `null`.
  **`stream_usage=True` (deepseek/openai in `create_model`) — a policy, NOT the
  root cause:** live-verified that DeepSeek returns `usage_metadata` even
  without the flag. The flag was set at the direct request of the user
  ("statistics are needed by default") as a guarantee for
  endpoints/configurations that would not send usage without
  `stream_options={"include_usage": true}` (langchain_openai auto-enables it
  only when `openai_api_base is None`, and we always pass an explicit
  `api_base`).
  **Tests:** unit — 4 `TestNormalizeUsage` cases (verbatim live-payload,
  cache_read split, missing cache details → `null`, priority of `token_usage`
  over `usage_metadata` when both forms are present); the new
  `test_stats_callback_usage_metadata.py` locks in the seam
  `message.usage_metadata → _extract_usage → normalize_usage → record` with a
  REAL `TokenUsageRecorder` (RED check: rolling back the parser fails exactly
  2 usage_metadata cases, the control `token_usage` and `no_usage` cases stay
  green); `test_stream_usage.py` — the flag policy (explicitly marked as a
  policy-guard, not a proof of causality); the live test
  `test_deepseek_agent_records_token_stats` is strengthened — it used to
  accept a record without tokens ("if prompt is present, check the invariant")
  and therefore could not distinguish working statistics from an all-`no_usage`
  file; now it requires ≥1 `status=ok` record with `prompt_tokens > 0`.
  **Limitation of the evidence base (deliberately documented):** a live repro
  of exactly the agent path with the `usage_metadata` form could not be
  obtained — the test agent went down the LangGraph `ainvoke` path
  (`langchain/agents/factory.py`: `output = await model_.ainvoke(messages)`) and
  got `token_usage`. The causal chain rests on 196 production `no_usage`
  records + a live astream-probe + a direct RED of the old normalizer.
