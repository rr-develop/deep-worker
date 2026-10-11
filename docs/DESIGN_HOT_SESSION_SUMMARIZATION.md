# Design: Summarization on a HOT provider session — no history dump (incident 2026-08-19)

**Status**: Analysis + requirements (v1). **No code changed** — the operator ordered "only the document".
**Design that implements the rules below**: `docs/DESIGN_SUMMARIZATION_HOT_COLD.md` — hot/cold decision by cache TTL `N(provider)`, in-session summarization on the hot path, dump in an ephemeral clean session on the cold path, immutable `created_at`, delta `deep_worker_origin`, per-message `deep_worker_usage`.
**Date**: 2026-08-19
**Related**:
- `docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md` (§3.3a observed-context floor, `a47af4b`)
- `deep_worker/providers/acp/chat_model.py` (`_agenerate` steps 4b/4c/10, `_finalize_response`, `_maybe_report_model`)
- `deep_worker/providers/acp/client_state.py` (`record_summarization`)
- `deepagents/deepagents/middleware/summarization.py` (`_acreate_summary_for_prompt`, `_compute_summarization_defaults`)
- `deepagents/deepagents/middleware/message_timestamps.py` (`MessageTimestampMiddleware`, `ensure_message_created_at`)
- `context_limit/observed_context.py`, `context_limit/guard.py` (CLI package)
- the web UI's agent service (`_fill_context_fields`, status bar)

---

> **UPDATED 2026-08-25 (POLICIES §13).** The incident below is described
> correctly, but its conclusion ("status was right; the summarizer was blind")
> was closed not with a crutch but by deleting the heuristic: both the guard
> and the summarizer now read the provider's number — the same one the status
> bar shows.

## 1. Operator requirement (verbatim, 2026-08-19)

> Bugs like this are a crime. … It is insane to try to write new functionality and breed new bugs where a standard summarization call was all that was needed.

> So the status is done correctly, and the summarizer is not (at least on this provider).
> And I insist that **on a hot session, where messages flow one after another with no change of provider/model, we must not do a dump, but must do summarization on the cached history.**
> Write this into the document — it is vital and costs us money and long outages.

### 1.1 HARD RULE (non-negotiable)

> **On a HOT session — same provider, same model, same live CLI/API session, messages flowing one after another — summarization MUST be performed *inside* the already-cached session (a short "compress" instruction on top of the cached prefix). A full history dump into a fresh session is FORBIDDEN on the hot path.**
>
> A history dump (re-sending the whole conversation as one new prompt) is allowed ONLY on the COLD path: provider or model changed, the provider session cannot be resumed (`session/load` unsupported/failed), or the process has no live session at all — and even then the dump goes **first, into a fresh session, before any `session/load` of the oversized history**.

Rationale (what it costs, measured below): the hot-path dump re-tokenizes the entire history as an uncached prompt (cache-write billed at 1.25×), on top of the load of the oversized session that preceded it. On 2026-08-19 one turn burned ~35% of a 5-hour quota in 2.5 minutes and produced no useful answer.

---

## 2. Incident 2026-08-19 (affected agent, `-l 400k`, claude-code/fable, commit `a47af4b`)

All timestamps UTC. Sources: the CLI session transcripts `{f44ee866,f94ef7a6,344d62ac}*.jsonl` (`message.usage`), the agent's `threads.db` checkpoints (11:07:13 → 15:16:16), and the agent's error log.

| Time | Session | What happened | Tokens |
|---|---|---|---|
| 11:07 | f44ee866 | Last turn before restart ends. Checkpoint 722 msgs. Status bar (`token_stats.jsonl`): **468.1k / 1000k, limit 400k** | cr=465,039 |
| 15:12:10 | — | `deep-worker --restart <agent>` (pick up `a47af4b`) | |
| 15:14:01 | — | TaskManager iteration 1: user question injected | |
| 15:14:02 | — | `before_model` #1: checkpoint carries **no** `deep_worker_context` stamp (pre-fix) → heuristic ≈151k < 340k → summarizer **skipped**, guard **passed** | |
| 15:14:20 | f44ee866 | `session/load` of the whole old session → full prompt-cache **re-write** of the 466k history | **cc=465,796**, cr=16,011 |
| 15:14:22 | f44ee866 | Model replies with a phantom `Bash` tool call → ToolNode "Bash is not a valid tool". AIMessage now stamped `used=481,811` | cr=481,807 |
| 15:14:22 | — | `before_model` #2: stamp 481,811 ≥ 340k → **summarizer fires** | |
| 15:14:50 | f94ef7a6 | Summarizer `ainvoke(prompt)` → ACP step 4b drops the live session → `session/new` → **the whole history dumped as the first user message** (489,053 chars) | **cc=199,730**, cr=19,417, out=5,271 |
| 15:14:43–15:15:47 | f44ee866 | Meanwhile the old CLI session keeps running on its own (3 more model calls, real answer produced at 15:15:47) — **discarded** | cr=481,807 / 483,653 / 486,535 |
| 15:15:56 | — | Race: `queued_*` of the dropped session makes `_consume_turn` return the stale `Bash` AIMessage as the "summary" (summary text = ""); the real summary text of f94ef7a6 is consumed as **the agent's answer to the user** (`## SESSION INTENT …`) | |
| 15:16:16 | — | TM evaluation → corrective prompt; stamp on the last AIMessage is `used=224,420` (dump+summary = the new session's context) ≥ 170k → **second summarization** | |
| 15:16:24 | 344d62ac | `session/new` again, dump of the compact state (20,930 chars) | cc=12,438, cr=19,417 |
| 15:16:38 | — | User interrupts | |

Totals for the turn: ≈ **666k cache-write + ≈2.0M cache-read + ~12k output** in 2.5 min. Two useful outputs: zero.

Context gauges at the end: old session 486k, dump session 224k (the dump **stays** in the new session's context — nothing was "reset"), compact-state session 32k.

---

## 3. Root causes (verified)

1. **Status bar and summarizer look at different numbers.** The web status bar reads `token_stats.jsonl` (last `usage_update` of the CLI — 468.1k). `SummarizationMiddleware` / `ContextLimitGuardMiddleware` read the `deep_worker_context` stamp on the latest AIMessage (`observed_context.find_context_anchor`) and fall back to the chars/4 heuristic when there is none. A pre-`a47af4b` checkpoint has no stamps → the first `before_model` after the restart ran on ≈151k → no summarization → a normal turn → `session/load` of 466k (§2, 15:14:20). *The status was right; the summarizer was blind.*

2. **The summarizer's `ainvoke` is routed through the agent's own thread state.** `summarization.py:1061-1068` calls `self.model.ainvoke(prompt, config={"configurable": {"thread_id": <agent thread>}, "metadata": {"lc_source": "summarization"}})`. The ACP model keys its per-thread state by `thread_id` (`chat_model.py:542`), sees `len(messages)=1 < sent_count` → step 4b (`chat_model.py:549-566`) **drops the live CLI session** and step 10 (`:678-699`) opens `session/new` with the dump as the first prompt. There is no separate/ephemeral channel for summarization and no in-session "compress" path. After the call that dump session *is* the agent session (subsequent turns send only the Human tail) — so the "summarized" context is dump + summary (224k), not ~12k.

3. **Dropped-session race.** `client_state.record_summarization` (`client_state.py:146`) resets `sent_count` but leaves `queued_tool_calls/queued_text/queued_response` of the dropped session; `_consume_turn` (`chat_model.py:928`) returns the stale queued AIMessage instantly. Result: empty summary, the summary text delivered to the user as the answer, the user's real question never sent.

4. **Trigger is 170k regardless of `-l`.** `_compute_summarization_defaults` (`summarization.py:144-149`) requires `isinstance(model.profile, dict)`; `AcpProfile` is a dataclass → the model-based trigger falls back to the fixed `("tokens", 170000)` which is OR-ed with `-l × 0.85`. Hence the second summarization at 224k.

5. **Phantom `Bash`.** ACP passthrough tools are bound only for `provider == "grok"` (`agent_process.py:1466`); claude-code emits `Bash` → ToolNode error loop (11.5k such lines in the agent's error log since 2026-08-17). Not the cause of the cost, but it is what turned a one-call turn into a multi-call one.

---

## 4. Required behaviour (design requirements, NOT implemented)

R1. **Compare the provider-reported gauge against `-l` BEFORE any `session/load`.** If the persisted provider state already says `used > -l` (or the last stamp / `token_stats` does), the turn must not load the oversized session at all. Cold path: dump → fresh session **first**. *(Today the pre-fix checkpoint gap makes the first turn blind; the provider's own persisted `usage_snapshot` / `token_stats.jsonl` must be used as the floor on the first turn.)*

R2. **Hot path = in-session summarization.** When the session is live and provider/model unchanged: send the summary instruction as the *next* `session/prompt` on the cached session (cache-read of the prefix, a few k cache-write), take the summary text, then open the fresh session with the compact state. Never re-send the history. For ACP CLIs that expose a native compaction (`/compact`-class) — prefer it; otherwise the in-session prompt. Either way the summarizer must NOT go through step 4b's drop-and-new machinery.

R3. **Summarization must not drop a mid-turn session and must clear `queued_*` on drop.** `record_summarization` (or its caller) must discard queued text/tool-calls/response futures of the dropped session; `_consume_turn` must never hand a dropped session's leftovers to the new turn. A summarization that starts while a pump is live must wait for / cancel that pump explicitly.

R4. **One source of truth for "how full is the context".** Status bar, summarizer trigger, guard and `--info` must read the same number (provider gauge with a documented fallback chain). Any divergence is a bug.

R5. **Trigger derived from `-l` only.** No fixed 170k fallback when `-l` is set; `AcpProfile` must expose `max_input_tokens` in whatever shape `_compute_summarization_defaults` accepts, or the defaults must take the limit explicitly.

R6. **Verification on a LIVE ACP agent before sign-off**: after a summarization the CLI's `usage_update.used` must drop to ≈ the compact-state size; the bill for the turn (cache_creation + cache_read) must be logged and bounded (no full-history cache-write on the hot path). Unit tests that pin "summarizer fired" are insufficient (that is exactly what `a47af4b` verified — and it was not enough).

---

## 5. Operator questions answered (state of the code as of `a47af4b`)

### 5.1 Do we know *when* the last message was processed (ours or server-side)?

- **Server side:** no. Neither ACP `session/prompt` results nor the OpenAI-compatible responses carry a processing timestamp/latency that we keep; ACP `usage_update` carries only `used/size`.
- **Our side:** `MessageTimestampMiddleware` writes `additional_kwargs.created_at` (ISO-8601 UTC) on every message — AI/Tool messages at receipt (`wrap_model_call`/`wrap_tool_call`), everything else at `before_model`. **It is not reliable as a processing time:**
  - `ensure_message_created_at` (`message_timestamps.py:71-80`) **overwrites** an existing stamp whenever it is not strictly newer than the previous message's stamp (monotonic clock). After a summarization the summary `HumanMessage` sits at index 0 with a *newer* stamp than the kept tail → from index 1 on every stamp fails `observe` and is **rewritten to "now"** (cascade).
  - Reproduced offline on the 11:07 checkpoint: one `before_model` pass re-stamps **721 of 722** messages. Verified in production: all **199** messages retained across the 15:14 restart carry `2026-08-19T15:14:22.15xxxx` in the 15:16 checkpoint, while the 11:07 checkpoint had their real times (e.g. `2026-08-18T23:16:39`, `2026-08-19T11:07:13`). The in-memory rewrite happens every turn; it is persisted whenever the `messages` channel is rewritten (summarization).
  - No duration/latency is recorded anywhere on the message.
- **Requirement (to do):** store on each AIMessage, persisted only (never serialized to the provider): `received_at`, `latency_ms` (request start → final chunk), provider `model_id`, provider session id; and fix the monotonic-clock overwrite so history stamps are immutable.

### 5.2 What exactly is sent to stateless API providers?

- JSON `messages[]` built by LangChain from `state.messages` each call (`ChatOpenAI`/`ChatDeepSeek` in the CLI package's `config.py`): per message `role`, `content` (text or content blocks), for AI: `tool_calls` (id/name/arguments), for Tool: `tool_call_id` (+`name`), optional `name`; DeepSeek additionally echoes `reasoning_content` (required by the API, see `config.py:1835-1851`). Plus `tools[]` JSON schemas every call.
- **Not sent**: message `id`, `additional_kwargs.created_at`, `tool_call_created_at`, `response_metadata` (incl. our `deep_worker_context` stamp), `usage_metadata`. Persisted-only attributes therefore cost no tokens — adding more of them (5.1, 5.3) is free on the wire.
- ACP providers: only the Human tail is sent per turn (the CLI keeps the session); the whole history goes over the wire only on dump (cold path / the forbidden hot-path dump of §2).

### 5.3 Can we tell on which provider/model a given server-originated message was produced?

**Today: not reliably — and for ACP not at all.** Verified on real checkpoints:

- **ACP (claude-code / codex)**: `AIMessage.response_metadata == {}` and `usage_metadata is None` for all 347 AIMessages of the affected agent (only the last ones carry `deep_worker_context` since `a47af4b`, which has no model name). The adapter *does* learn the real model every turn — `_meta.modelId` → `_maybe_report_model` → `_record_resolved_model` (`chat_model.py:1776-1781`, `:312-324`) — but it lands only on the model object and in `token_stats`, never on the message.
- **API providers (ChatOpenAI-compatible, ChatDeepSeek)**: LangChain stamps `response_metadata = {model_name, model_provider, finish_reason, service_tier, (token_usage)}` + `usage_metadata`, and it persists — **but**: (a) `model_provider` is the LangChain integration (`'openai'` for MiniMax/grok/any OpenAI-compatible endpoint), not deep-worker's provider name (`minimax`/`grok`/`deepseek`) and not the `base_url`; (b) streamed responses get their string fields **concatenated** across chunks — in another agent's checkpoint: `model_name='MiniMax-M3MiniMax-M3'`, `finish_reason='tool_callstool_calls'`, `service_tier='standardstandard'` (35 of 87 AIMessages; the other 52 — ACP leg after failover — are empty). So even where present it needs normalization.
- Mixed-provider threads (failover) therefore cannot be attributed message-by-message.

**Requirement (to do):** one stamping point for every provider (the same place `deep_worker_context` is applied) writing `response_metadata["deep_worker_origin"] = {provider, model_id, session_id, received_at, latency_ms}` — persisted only, never sent (5.2). The context-limit, failover and cost diagnostics all need it; the `-l` decision on the first turn after a restart (R1) needs `model_id` to know whether the session is hot.

---

## 6. Out of scope / open

- Native compaction support per ACP CLI (capability discovery) — needed for R2; unknown for codex.
- History-offload `ensure_messages_created_at` also re-stamps the offloaded slice (`summarization.py:707,797`) — the JSONL history archive inherits the corrupted stamps.
- Passthrough binding for non-grok ACP providers (phantom `Bash`) — separate fix.
