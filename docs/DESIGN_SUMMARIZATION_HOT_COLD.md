# DESIGN: summarization on a hot kv-cache (hot) and from a dump in a clean session (cold); immutable provider/model/usage/time stamps on messages

**Status**: design v1 — **DESIGN SIGN OFF** (reviewer, 2026-08-21) on this document. The code for stages 1–3 is written, but **CODE SIGN OFF has NOT been received**: do not commit, do not fast-forward master, do not restart the production agent on this tree. Live ACP verification (§9.4) remains a gate before the production ship (after the quota is lifted), not part of the design sign-off.
**Owner**: ACP maintainer. **Date**: 2026-08-19 (design), 2026-08-21 (implementation).
**Closes incident**: production agent, 2026-08-19 15:14Z (see `docs/DESIGN_HOT_SESSION_SUMMARIZATION.md` — analysis, timeline, cost; this document is the normative design).
**Related**: `docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md` (trigger/guard driven by `-l`, `a47af4b`), `docs/DESIGN_AGENT_TOKEN_STATS.md` (UsageRecord / `token_stats.jsonl`).

---

## 0. Owner requirements (normative)

| # | Requirement | Section |
|---|---|---|
| R1 | The creation/receipt time of a message is written ONCE and rewritten by NOBODY. The monotonic rewrite (`ensure_message_created_at`) is removed. | §2.1 |
| R2 | Provider and model are saved in the first checkpoint and afterwards ONLY on a real change. | §2.2 |
| R3 | Usage / kv-cache / tokens are saved on the message (per message), for provider analysis and for summarization. | §2.3 |
| R4 | Everything from R1–R3 survives the history dump to JSONL during summarization (offload) — nothing is lost. | §2.5 |
| R5 | Hot cache (same provider+model, less than N since the last provider response) ⇒ summarization FROM THE CONTENTS OF THE kv-cache: no dump. N is per-provider, configurable, default 5 min (proposal: 4 min, §4.3). | §4, §5.2 |
| R6 | Change of provider/model OR a pause > N ⇒ dump and summarization from the dump in a SEPARATE CLEAN session; the dump is never appended to the agent's open ACP session. | §5.3 |
| R7 | Later — an offline script over the checkpoints of all agents/providers: kv-cache hit statistics and the empirical cache TTL. The data for it = R1–R3. | §8 |
| R8 | Once implemented, the design must fix the 2026-08-19 incident. | §7 |

---

## 1. Terms

- **Provider context** — what the provider actually holds: for ACP (claude-code/codex) — the CLI session (`usage_update.used`), for API providers — the `messages[]` we send on every call.
- **kv-cache / prompt-cache** — the provider's prefix cache. Anthropic: `cache_creation_input_tokens` (paid write ×1.25) / `cache_read_input_tokens` (read ×0.1); TTL 5 min, extended by every hit (a 1h variant exists). OpenAI/codex: automatic, ~5–10 min (up to an hour off-peak). DeepSeek: automatic, "hours". xAI/Gemini: automatic, TTL not precisely documented. Any cache is tied to a BYTE-IDENTICAL prefix (system + tools + messages); changing tools/system/`tool_choice` = a miss of the whole prefix.
- **Hot** — provider and model have not changed since the last provider response AND `now − last_provider_message_at < N(provider)` AND (for a stateful provider) a resumable `session_id` exists.
- **Cold** — everything else.
- **Dump** — re-sending the history as text/`messages[]` into a session without a cache. FORBIDDEN on the hot path (HARD RULE from `DESIGN_HOT_SESSION_SUMMARIZATION.md` §1.1).

---

## 2. Data model: stamps on messages

All stamps live in `additional_kwargs` / `response_metadata` of LangChain messages, i.e. in the checkpoint and in the offload JSONL (`messages_to_dict`, `summarization_history.py:41`). They are NOT sent to the provider (§2.6) — zero cost in tokens.

### 2.1 `created_at` — immutable (R1)

- Same key: `additional_kwargs.created_at` (ISO-8601 UTC, microseconds). Semantics: **the moment of creation (Human/System/Tool) or of receipt of the last chunk (AI)**.
- Rule: **set-if-missing**. `ensure_message_created_at` (`message_timestamps.py:71`) loses the "clock.observe/next" overwrite parameter: if a stamp exists — do not touch it, whatever it is. `MonotonicUtcClock` remains only as a source of "now" for new stamps (monotonicity within one process is useful for ordering simultaneously created messages, but it does NOT give the right to overwrite someone else's stamp).
- Remove the re-stamping in offload (`summarization.py:707`, `:797`) — only set-if-missing is called there.
- `format_message_with_timestamp` (`:105`) does not mutate the message (reads the existing stamp, prints `?` for a missing one).
- Why it was there: the monotonic clock fixed the order in the summarizer buffer. Owner: "I don't understand why; I doubt it is useful to anyone." Order is defined by position in `state.messages`, not by the stamp.
- Consequence: the cascading rewrite of 721/722 messages within a single `before_model` (proven on the 11:07 checkpoint) disappears; the JSONL history receives real times.

### 2.2 `deep_worker_origin` — provider/model, only on change (R2)

`response_metadata["deep_worker_origin"]` on an **AIMessage**:

```json
{"provider": "claude-code", "model": "fable", "model_id": "claude-fable-5-20260801",
 "adapter": "@agentclientprotocol/claude-agent-acp@0.68.0", "since": "2026-08-19T15:14:22.151234+00:00"}
```

- `provider` — the deep-worker provider name (`claude-code`/`codex`/`grok`/`deepseek`/…, NOT the LangChain integration `openai`); `model` — as configured (`-m`); `model_id` — what the provider actually answered with (`_meta.modelId`/`response_metadata.model_name`), if known; `adapter` — for ACP (fingerprint from `acp_session_state.json`).
- **Set only if the origin differs from the effective origin of the previous AIMessage in the current `state.messages`** (delta-stamp) — i.e. on the first reply of a thread and on every real switch (failover, `-m` change, `model_id` change at the provider).
- **Slice invariant**: the first AIMessage of any "slice" of history (the current state after summarization; an offload portion of the JSONL) carries a full stamp. Ensured by: (a) the summarizer, when forming `[summary, *tail]`, copies the effective origin onto the summary HumanMessage (`additional_kwargs.deep_worker_origin`) AND onto the first AIMessage of the tail; (b) the offload writer resolves the effective origin for the first element of the portion and writes it into the portion's header line (`kind=deepagents.summarization_history.origin`) — see §2.5.
- The effective origin of message k = the nearest stamp at position ≤ k; the reader (`dw_hist`, the stats script, the summarizer) implements this in a single pass.
- **Rejected**: a `provenance` state channel in the checkpoint. The delta checkpointer (`delta_checkpointer.py`) does store a replace-delta only on change — the "write only on change" form would be achieved literally, but: attributing a message requires walking the chain of checkpoints; the channel does not reach the offload JSONL without a separate export; TM/non-graph model calls do not see it. A stamp on the message is self-describing and travels wherever the message travels.

### 2.3 `deep_worker_usage` — on every AIMessage (R3)

`response_metadata["deep_worker_usage"]`:

```json
{"received_at": "…", "request_started_at": "…", "latency_ms": 8421,
 "input_tokens": 12000, "output_tokens": 540,
 "cache_read_tokens": 465039, "cache_write_tokens": 4100,
 "context_used": 481811, "context_size": 1000000,
 "provider_session_id": "f44ee866-…", "provider_turn_id": null,
 "source": "api_usage_metadata | acp_meta | acp_usage_update | cli_transcript"}
```

- Fields are `null` if the provider does not supply them; `source` says where the data came from (for the stats script and the honesty of "n/a").
- Sources per provider:

| Provider | input/output | cache read/write | context gauge | Where we take it from |
|---|---|---|---|---|
| API (deepseek/openai-compat/anthropic/google) | LangChain `usage_metadata` | `usage_metadata.input_token_details.{cache_read,cache_creation}`; DeepSeek `prompt_cache_hit/miss_tokens` | — (we count = input) | already in the AIMessage; normalization as in `stats/token_stats.py` `normalize_usage` |
| grok (custom) | `_meta` / fallback | if present | `usage_update` | `grok_provider/chat_model.py` |
| ACP claude-code | `_meta.inputTokens/outputTokens` — mapping is implemented (`chat_model.py:1539-1557`); on adapter 0.68 the fields were observed empty → null; this is a property of THAT adapter version, not an absence of the channel | `_meta.cachedReadTokens` (same place; null on 0.68) | `usage_update.used/size` ✔ | `acp/chat_model.py:_finalize_response` |
| ACP claude-code — **enrichment** | ✔ | ✔ `cache_read_input_tokens`/`cache_creation_input_tokens` | — | **CLI transcript** `~/.claude/projects/<cwd-slug>/<session_id>.jsonl`: the last `assistant` record with `timestamp ≥ request_started_at` → `message.usage`. Best-effort post-turn hook of the profile (`AcpProfile.transcript_usage_reader`), does not block the reply, null if the file is missing |
| ACP codex | `_meta` if the adapter provides it | — | `usage_update` | same |

- `deep_worker_context` (`{used,size,first_id}`, `a47af4b`) **stays as is** — it is already read by the summarizer/guard (`observed_context.py`); `deep_worker_usage.context_*` duplicates its values (one write point, §3).

### 2.4 Non-AI messages

Human/System/Tool: only `created_at` (R1). A ToolMessage may additionally carry `tool_duration_ms` (the AIMessage already has `tool_call_created_at` — we do not touch it).

### 2.5 Offload JSONL (R4)

- `serialize_summarized_messages_jsonl` → `SCHEMA_VERSION=5`: before a portion of messages a header line is written
  `{"schema_version":5,"kind":"deepagents.summarization_history.segment","summarized_at":…,"thread_id":…,"origin":{effective origin of the first message of the portion},"reason":"hot|cold|requested","provider_session_id":…}`; message lines are as now (`messages_to_dict` carries all `additional_kwargs`/`response_metadata`/`usage_metadata`).
- `deserialize_summarized_messages_jsonl` understands v4 (no header) and v5 (skips/returns the header separately).
- Offload does NOT call anything that mutates messages (§2.1).

### 2.6 What is NOT sent to the provider

`additional_kwargs.created_at/tool_call_created_at/deep_worker_origin`, `response_metadata.*`, `usage_metadata` are not serialized into the request either by API providers (`DESIGN_HOT_SESSION_SUMMARIZATION.md` §5.2: `role/content/tool_calls/tool_call_id/name` are sent, DeepSeek + `reasoning_content`), or by ACP (only the Human tail goes to the CLI). Lock this invariant with a test per provider: the request body does not contain the string `deep_worker_`.

---

## 3. Single stamping point — `ProvenanceMiddleware` (deep_worker_cli)

- New `deep_worker_cli/middleware/provenance.py`, `AgentMiddleware.awrap_model_call`/`wrap_model_call`. Created in `agent.py` next to `ContextLimitGuardMiddleware` (`agent.py:1031`) with parameters `provider_name`, `model_name` (what `AgentProcess` knows; after a hot failover the graph is rebuilt with the new values — as is currently done for `TokenStatsCallbackHandler`).
- Order: **outside** `MessageTimestampMiddleware` and the provider stamps — it sees the final AIMessage with `deep_worker_context` (ACP) and `usage_metadata` (API).
- Algorithm: `t0=now` → `handler(request)` → `t1` → on `response.result[-1]` (AIMessage): `deep_worker_usage` (from `usage_metadata`/`response_metadata`/`deep_worker_context` + `received_at=t1`, `latency_ms`), `deep_worker_origin` if the effective origin of (`request.messages`) ≠ the current `(provider, model, model_id)`; `model_id` — `getattr(model, "resolved_model_name", None)` for ACP (`chat_model.py:312`, updated in `_maybe_report_model` BEFORE returning) or `response_metadata.model_name` (normalized against chunk concatenation `MiniMax-M3MiniMax-M3`).
- Enrichment from the CLI transcript (§2.3) is an asynchronous best-effort hook after return; if it makes it in time — it appends to the same AIMessage before the checkpoint commit; if not — `source` stays `acp_usage_update`.
- Persistent provider state (ACP `acp_session_state.json`, similarly codex/grok/claude-sdk) gets the fields `last_response_at`, `last_context_used`, `model_id` — written in the already existing `persist()` (`client_state.py:123`) on every turn. This duplicates the message stamp for the first turn after a restart (the checkpoint may be of the old format — §6).
- TM (`task_manager.py`) calls the model bypassing the graph — the same stamping function is called explicitly in TM (follow-up, does not block this design).

---

## 4. Hot/cold detector

### 4.1 Inputs (all persistent, survive a restart)

1. `last_provider_message_at` = `deep_worker_usage.received_at` of the last AIMessage in `state.messages` (fallback `created_at`); cross-check against `last_response_at` from the provider state — we take the **max** (a later provider response that is not yet in the checkpoint — e.g. an interrupted turn).
2. `last_origin` = effective origin of the last AIMessage; `current` = `(provider, model, model_id)` of the current graph.
3. For stateful (ACP/codex/claude-sdk/grok): `session_id` in the provider state and `session-capability: load` in the fingerprint.

   **Corrected 2026-08-24: grok is session-stateful, not stateless.** The earlier wording ("grok is classified as fully stateless in v1") contradicted the code: `GrokChatModel` sends only `messages[sent_count:]` and persists `grok_session_id`+`sent_count` across restarts, i.e. the history lives on the grok side. Because of the wrong classification `_is_session_stateful` returned False, the guard in `_ainvoke_identical_prefix` did not fire, and summarization went through a regular `ainvoke` — **a silent dump of the entire history into a fresh grok session on every summarization**. The test `test_grok_is_stateless_hot_without_session` did not catch the defect but locked it in; replaced with `test_grok_is_session_stateful`.
4. `N(provider)` — §4.3.

### 4.2 Rule

```
hot  := last_origin.provider == current.provider
     and last_origin.model    == current.model        # alias; a model_id change under the same alias → also cold
     and (now - last_provider_message_at) < N(provider)
     and (stateless(provider) or session_id is not None)
cold := not hot
reason := "provider_changed" | "model_changed" | "ttl_expired:<sec>" | "no_session" | "hot"
```

The decision is made **once** at the start of summarization and logged (`summarization.decision reason=… elapsed=… N=…`). On the hot path, after the reply the outcome is additionally recorded: `cache_read_tokens` ≥ 0.5×`context_used` ⇒ `hot_hit`, otherwise `hot_miss` (to the log and to `deep_worker_usage` of the summary message) — this is the raw material for §8.

### 4.3 N — configuration

- A single defaults table `deep_worker_cli/context_limit/hot_cache.py::HOT_CACHE_TTL_DEFAULTS_SEC`:

| provider | default N | rationale |
|---|---|---|
| claude-code, claude-code-sdk, anthropic | **240 s** | Anthropic TTL is 5 min; minus a margin for `session/load` of a large session (10–30 s) and network latency |
| codex, openai, grok, google | 240 s | documented/observed "minutes"; conservative |
| deepseek | 240 s (raise after §8) | cache is "hours", but there is no confirmation — measure first |
| `*` (any other) | 240 s | |

  Proposal to the owner: **4 min instead of 5** — precisely because the countdown runs from receipt of the last response, while the summarize prompt is sent after `session/load`/CLI startup; with N=5 min a "hot" decision at 4:50 actually fires at 5:10 and misses at the cost of a full prefix rewrite (2–2.5× more expensive than an honest cold dump, §5.5). After statistics accumulate (§8) the values are refined per provider.
- Override: CLI flag `--hot-cache-ttl <sec>` and `--hot-cache-ttl <provider>=<sec>` (repeatable; saved in the launch config and survives `--restart` — `run_agent(effective)`, lesson of `e7e6eb5`), env `DEEP_WORKER_HOT_CACHE_TTL_SEC[_<PROVIDER>]`, a key in the agent config. Priority: flag > env > config > table. `0` = always cold (emergency off switch for the hot path).

---

## 5. Summarization: protocol and two paths

### 5.0 A single interface for stateful providers (2026-08-24)

**Invariant: `provider ∈ SESSION_STATEFUL_PROVIDERS` ⟺ the model implements `SummarizerPort` (both halves).**

The port is declared as a `Protocol` in `deep_worker_cli/context_limit/summarizer_port.py`; the check is `implements_summarizer_port(model)` (looks at the **type**, not the instance, and requires **both** halves to be `async def`). The invariant is enforced by the test `test_summarizer_port_contract.py`, not by convention: a new stateful provider must appear in `PROVIDER_MODELS` and pass the structural check.

Before this, `SummarizerPort` was a **fiction**: the name appeared in four docstrings, but the type did not exist, and both call sites detected the port via `getattr`. A provider that did not implement the method or misspelled its name **silently** degraded to "no port". Three discrepancies had accumulated under this:

| provider | was | now |
|---|---|---|
| `codex` | stateful, no port → `SummarizationFailedError` on every summarization (production agent, 2026-08-23) | port on `codex exec` (hot = resume rollout, cold = one-shot) |
| `grok` | classified stateless, no port → **silent dump** of the entire history | stateful; port obtained by inheritance from `AcpChatModel` |
| `claude-code-sdk` | `asummarize` exists, `areset_after_summary` does not; halves named `_asummarize_*` versus ACP `_summarize_*` | both halves in place |

Why an interface and not a common base class: the concrete models already inherit `BaseChatModel` (pydantic with its own metaclass machinery), and the protocols differ in transport (ACP JSON-RPC versus `codex exec`). What is unified is the **call contract**, not the transport — an "if provider X, then…" branch does not appear in the calling code.

Separately: `_current_provider_model` read `getattr(model, "profile")`, but pydantic strips the `@property` off the class, and the instance returned the `property` object itself → `provider` resolved to `"unknown"` and `_is_session_stateful` was **always False for all ACP providers, including claude-code**. The guard was dead code for exactly the providers it was written for; it looked functional only because they implement the port and the dump did not occur. Fixed by reading `_profile`.

> **UPDATED 2026-08-25 (POLICIES §13).** Everywhere below where a heuristic
> is mentioned as the source of context length, read: the source is ONLY
> the provider's figure. The heuristic estimator (`token_estimator.py`) has been removed;
> the "floor over the heuristic" from §6 no longer exists — the persisted
> provider value is used directly, and in the absence of a report
> the quantity is considered UNKNOWN and summarization does not fire.

### 5.1 An explicit port instead of heuristic 4b

Today ACP learns about summarization via the heuristic `sent_count > len(state_messages)` (`chat_model.py:549-566`) inside the summarizer's own `ainvoke` → it drops the live session, `session/new`, dumps in the first prompt, and that becomes the agent's session. This is the incident. Replaced with an explicit port:

```python
class SummarizerPort(Protocol):               # implemented by the model/provider; duck-typed
    async def asummarize(self, *, thread_id: str, messages: list[AnyMessage],
                         instruction: str, decision: HotColdDecision) -> SummaryResult
    async def areset_after_summary(self, *, thread_id: str) -> None
```

- `SummarizationMiddleware._acreate_summary_for_prompt` (`summarization.py:1042`) calls `asummarize` if the model implements it; otherwise — a generic implementation for API providers (§5.2b). No `model.ainvoke(str)` with the agent's `thread_id`.
- `areset_after_summary` is called by the middleware **after** the new `state.messages = [summary, *tail]` has been formed: for ACP = `record_summarization` + `_cancel_pump`/`_adopt_pump` + **clearing `queued_text/queued_tool_calls/queued_response` and `pending_session_prompt_future`** (R3 of the analysis; currently `client_state.py:146` does not clear them) → the agent's next turn opens `session/new` with a compact first prompt (existing step 10a/11). Heuristic 4b is removed (a fail-loud log remains if `sent_count > len(messages)` still happens — that is a bug).
- `SummaryResult = {text, usage: deep_worker_usage-dict, provider_session_id, path: "hot|cold"}`; an empty `text` is an error (fail-loud, no "Error generating summary" in the history).

### 5.2 Hot path

**(a) Stateful ACP (claude-code, codex) — in-session prompt.**
1. Session not alive in the process (after a restart) → `session/load(session_id)` (the only permissible load on the hot path; cache_read is expected; the outcome is recorded per §4.2). Load failed → **transition to cold** (reason `load_failed`), NOT `session/new` with a dump into the agent's session.
2. If the turn is not finished (the state tail contains ToolMessages, the CLI has a `session/prompt` hanging waiting for a tool result, `pending_session_prompt_future`/pump are alive) → `session/cancel` (`acp_client.py:474`), wait for `stopReason=cancelled`, adopt/remove the pump. Without this a second `session/prompt` is impossible.
3. `session/prompt(instruction)` into the SAME session. **Change nothing before this**: no `session/load(mcp_servers=[])` (step 10b), no change of mode/tools/system — all of these change the prefix and kill the cache. The instruction requires a plain-text reply without tools; if the model nevertheless calls a tool — the proxy MCP answers with a short error "tools unavailable during summarization" (limited to 3 attempts), and the text is collected as usual.
4. The reply = the summary. This session's usage (transcript/`_meta`) → `SummaryResult.usage`.
5. After `areset_after_summary` the old session is abandoned (its cache expires on its own); the agent's session = a new, compact one (`session/new` + `[summary, *tail]` as the first prompt ≈ 10–30k).
   Optional (follow-up): the CLI's native compaction (`/compact` class, detected via `available_commands_update`) — gives the same cost, but diverges from the LangGraph state (722 messages remain with us); not used in v1.

**(b) Stateless API (deepseek/openai-compat/anthropic/google, grok) — identical prefix + instruction.**
- The summarization request = **the same system + the same tools (already `_bind_tools_for_middleware`, `graph.py:265`) + `state.messages` unchanged + HumanMessage(instruction)**. No `get_buffer_string` into a single string (different tokens ⇒ a guaranteed cache miss — that is what happens today). We do not set `tool_choice` (for Anthropic changing it invalidates the message cache).
- The agent's system prompt, for identity: the middleware remembers the `request.system_message` of the last `awrap_model_call` (stable within the process) and passes it to `asummarize`.
- For stateless, cold differs from hot only in the expected cost — the mechanism is the same (full `messages[]` + instruction); nothing "in a separate session" needs to be done, there is no session.

### 5.3 Cold path (ACP): dump in a separate clean session

1. An **ephemeral** `session/new`: NOT registered as the agent's session (`register_session` under the key `"{agent_thread_id}:summarizer"`, not persisted in `acp_session_state.json`, the agent's `ThreadState` is not touched until the §5.1 reset). Same `cwd`, same proxy MCP (so as not to wait for detach; tool calls are rejected as in §5.2a-3) or `mcp_servers=[]` — there is no cache here, it can be done without MCP; the choice: **without MCP** (shorter prompt, less temptation).
2. The first and only prompt = instruction + a dump of `messages_to_summarize` in the existing rendering format of the first prompt (step 11 `chat_model.py:813`; NOT `get_buffer_string` v1 — the format is unified with hot for API). Trimming to the model budget: `trim_tokens_to_summarize` is taken from the profile/`-l` (currently `None` everywhere — the dump is not cut; `AcpProfile.max_input_tokens` must reach `_compute_summarization_defaults`, R5 of the analysis).
3. The reply = the summary; the ephemeral session is closed (`session/cancel` + evict index). In the CLI transcripts it remains as a separate file — this is expected.
4. The agent's session with the old history is **never `session/load`-ed** on the cold path: its `used > -l` by definition (§6). After `areset_after_summary` the agent goes `session/new` with a compact state.
5. The provider changed (failover) — cold on the NEW provider (its model = the `summarization_model` of the rebuilt graph); the old provider is unavailable by definition of failover, "finish summarizing on the old one while it is hot" is impossible.

### 5.4 Instruction

- `HOT_SUMMARY_INSTRUCTION` — the current `DEFAULT_SUMMARY_PROMPT` without `{messages}` ("Summarize the conversation above … sections SESSION INTENT/SUMMARY/ARTIFACTS/NEXT STEPS; plain text; no tools").
- `COLD_SUMMARY_PROMPT` — the same + the dump. One template source, two renderings.
- The summary message to the agent: as now `HumanMessage(lc_source='summarization')` + a link to the JSONL; additionally `additional_kwargs.deep_worker_origin` (effective, §2.2) and a `response_metadata` analogue of `deep_worker_usage` of the summarizing call (`summary_usage`), so that the cost of summarization is visible in the checkpoint.

### 5.5 Cost (estimate for the incident numbers, used≈482k, dump≈200k)

| path | write | read | total (equiv. input tokens ×price) |
|---|---|---|---|
| hot-hit | ~5k | ~482k | ≈ 0.1×482k + 1.25×5k ≈ **55k** |
| cold (dump) | ~200k | ~0 | ≈ 1.25×200k ≈ **250k** |
| hot-miss (mistaken hot) | ~482k | ~0 | ≈ **600k** |
| incident (load + dump + TM) | 911k | 2.0M | ≈ **1.3M** |

Hence the conservative N (§4.3) and the mandatory recording of hot_hit/hot_miss.

### 5.6 Errors and fault tolerance

- hot: `session/load` failed / `session/cancel` not confirmed within T / the summarize prompt returned empty or `stopReason` is not `end_turn` ⇒ **one** transition to cold (log `hot_path_failed reason=…`). cold failed ⇒ `HistoryOffloadError`-class fail-loud → `AgentStatus.ERROR` (as today for offload). No silent "Error generating summary: …" in the state.
- Timeouts: the summarize turn is bounded by the same `_prompt_idle_timeout()` (`chat_model.py:113`); the dump session — by the same.
- User interruption during summarization: the ephemeral session is cancelled, the state is not rewritten (atomicity: `[summary,*tail]` is written only after a successful `SummaryResult`).
- Parallel subagents: own `thread_id` ⇒ own `ThreadState` and own ephemeral session; the shared `AcpClient` is not blocked (sessions are independent).

---

## 6. Trigger and the first turn after a restart (link with `DESIGN_CONTEXT_LIMIT_GUARANTEE.md`)

- The floor of "how much is occupied" on the first `before_model` after a restart, when the checkpoint has no `deep_worker_context` (pre-`a47af4b`) or the last stamp is old: **max(stamp, `last_context_used` from the provider state §3, the last `context_snapshot` by `provider_session_id` from `token_stats.jsonl`)**. This is exactly what was missing at 15:14:02 (heuristic 151k against a real 468k → `session/load` 466k).
- The trigger is driven only by `-l` (R5 of the analysis): `_compute_summarization_defaults` accepts an `AcpProfile` (dataclass) on a par with a dict profile; the fixed 170k does not apply when `-l` is set.
- The guard (`guard.py`), the summarizer and the web status bar read the same floor (R4 of the analysis).
- **Gauge invalidation on summarization (normative).** `areset` / `record_summarization` MUST zero `last_context_used` and `last_response_at` in the provider state (or replace them with a compact post-summary gauge) and **persist** this to disk in the same call. The gauge describes a session that no longer exists: if the pre-summary value is left in it (in the incident — 468k), then `extra_floor` keeps returning it after the compaction to `[summary, tail] ~40k`, `max(counted, floor)` gives 468k, the trigger fires again — a loop; `--restart` re-reads the same value from disk and stays blind in exactly the same way as at 15:14:02. Zeroing is part of the reset contract, not an optimization: without it §6 breaks itself.

---

## 7. How the design closes the 2026-08-19 incident (R8)

| Incident step | Behavior after implementation |
|---|---|
| 15:14:02 `before_model` blind (151k) → a normal turn | floor from `last_context_used`/`token_stats` = 468k ≥ 340k ⇒ summarization BEFORE any `session/load` (§6) |
| decision | `last_provider_message_at`=11:07, now=15:14 ⇒ elapsed 4h > N ⇒ **cold**, reason `ttl_expired` (§4.2) |
| 15:14:20 `session/load` 466k (cc=465,796) | does not happen: cold never loads an overflowing session (§5.3-4) |
| 15:14:50 dump into a `session/new` that became the agent's session; summary=""; the summary text went to the user; the user's question was not sent | dump in an **ephemeral** session (§5.3); `queued_*` cleared, pump removed (§5.1) ⇒ no race; summary → state; the agent's next turn — `session/new` with the compact state and the real question |
| 15:16:24 second summarization (170k trigger, session 224k = dump+summary) | agent session = compact ≈ 10–30k; trigger only from `-l` (§6) ⇒ no second summarization |
| TM evaluation cc=233k | TM does not receive the summary as "the agent's reply" (the summary is not part of the turn); TM's own context is a follow-up (§9) |
| cost ≈ 1.3M equiv. | ≈ 250k (cold, §5.5); on a restart < N — ≈ 55k |
| had there been no restart (hot, messages in a row) | in-session summarize on the 482k cache: cache_read, ~5k write; no dump |

---

## 8. Offline kv-cache statistics script (R7, after data has accumulated)

- Input: the agents' `threads.db` files in the agent data directory (checkpoints; read via the `dw_hist` reader or a `DeltaSqliteCheckpointer` walk), `…/conversation_history/*.jsonl` (offload v4/v5), `token_stats.jsonl`, for claude-code — CLI transcripts (if the §2.3 enrichment did not make it in time).
- For each AIMessage: effective origin (§2.2), `deep_worker_usage`, `gap = received_at − received_at(previous AIMessage of the same provider_session_id)`, hit-ratio = `cache_read/(cache_read+cache_write)`.
- Output: per `(provider, model_id)` a histogram of hit-ratio by gap bucket (0–1, 1–2, 2–4, 4–5, 5–10, 10–30, 30–60, >60 min) → empirical TTL = the bucket where hit-ratio drops below 0.5; + the share of hot_hit/hot_miss decisions (§4.2) ⇒ adjustment of `HOT_CACHE_TTL_DEFAULTS_SEC`.
- Location: `deep_worker_cli/stats/cache_ttl_report.py` (run from a neutral directory — `deep_worker_cli/types.py` shadows the stdlib). Not part of this design — only the data requirements.

---

## 9. Implementation plan (stages, each a separate review)

1. **Stamps** (§2–3): `ensure_message_created_at` set-if-missing; `ProvenanceMiddleware`; `deep_worker_origin`/`deep_worker_usage`; offload v5; provider state `last_response_at/last_context_used/model_id`; claude-code transcript enrichment. Tests: immutability (the stamp does not change after N passes of `before_model`/offload); delta-origin (set on the 1st and on change, NOT on every one); offload contains the origin header; "the request body does not contain `deep_worker_`" on every provider.
2. **Hot/cold detector + config** (§4): a pure function `decide(...)`, the N table, flag/env, persistence through `--restart` (a mutation test for field loss, lesson of `e7e6eb5`).
3. **SummarizerPort** (§5): generic API implementation (identical prefix), ACP hot (load/cancel/prompt-in-session), ACP cold (ephemeral session), `areset_after_summary` with clearing of `queued_*`, removal of heuristic 4b, floor §6, `AcpProfile` in `_compute_summarization_defaults`. Tests with a fake ACP client: which `session/*` went to which session (hot: exactly one `session/prompt` into the old session, zero `session/new` before reset; cold: zero `session/load` of the old one, one `session/new` of the ephemeral + one of the agent); mid-turn hot → `session/cancel` precedes the prompt; `queued_*` race (incident regression) — a repro test from the timeline.
4. **Live verification on ACP** (mandatory before sign-off, lesson of `a47af4b`): scenario A hot (agent > trigger, messages in a row) → transcript: `cache_read ≈ used`, `cache_creation < 20k`, `usage_update.used` after summarization ≈ the size of the compact state; scenario B cold (restart after > N) → zero `session/load` of the old session, one ephemeral, cost ≈ dump; scenario C failover → cold on the new provider; scenario D hot mid-turn. A report with numbers in the PR.
5. Follow-up outside this design: TM context under the same rules; native CLI compaction; passthrough of builtin tools for non-grok ACP (phantom `Bash`).

### 9.5 Observation: cost of identical-prefix on stateless API (observed, NOT normative)

On deepseek (a test agent, 2026-08-20) hot and cold produced a statistically identical request shape: A `prompt=9111 / cache_hit=2304 / cache_miss=6807` versus B `prompt=9153 / cache_hit=2304 / cache_miss=6849` — both miss-dominated, i.e. the "hot-hit is cheaper than cold" cost model of §5.5 did not manifest on this provider. This is a finding about **API providers**, not evidence for §9.4 and not grounds to change the ACP hot/cold protocol, the `N` defaults, or the requirements R1–R8. Claude figures (`cache_read ≈ used`) will supplement this section later; until then API-provider telemetry cannot be counted as passing §9.4.

---

## 10. Rejected alternatives

- **Keep the dump, but "smarter"** (trim/chunks) — does not solve the cost on the hot path and keeps the dump session as the agent's session.
- **Native `/compact` as the main hot path** — diverges the CLI context and the LangGraph state; kept as an option.
- **A state channel for provider/model** — see §2.2.
- **N = 5 min by default** — see §4.3 (4 min; easy to return to 5 via config; final — per the §8 statistics).
- **Detecting "hotness" by `session_live_in_process`** — a process flag, not a property of the provider's cache; after a restart the cache may be alive (restart < N) while the flag is not. The criterion is time + provider/model (+ resumable session).
