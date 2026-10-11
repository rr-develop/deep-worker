# Design: Context-Limit Guarantee (2026-08-18)

**Status**: Draft → review (v1)
**Date**: 2026-08-18
**Related**:
- `deep_worker_cli/cli/agent_command.py` (`-l` parsing, `merge_restart_config`)
- `deep_worker_cli/agent.py` (`create_cli_agent(context_limit=...)`)
- `deep_worker_cli/process/agent_process.py` (passes `launch_config["context_limit"]` through)
- `deep_worker_cli/stats/token_stats.py` (UsageRecord schema, token-stats JSONL)
- `deep_worker_cli/stats/callback.py` (LangChain callback wiring)
- `deepagents/deepagents/middleware/summarization.py` (`_compute_summarization_defaults`)
- web service `agent_service.py` (`_fill_context_fields`)
- a related internal backlog item (not included here): a bloated agent threads.db
- `docs/DESIGN_HOT_SESSION_SUMMARIZATION.md` (2026-08-19 incident: HARD RULE — no history dump on a hot session; summarizer/status single source; per-message provider/model/time stamps)

---

## 1. User requirement (verbatim, 2026-08-18)

> Yesterday I restarted the model with `deep-worker -r <agent> -pp claude-code --model fable -l 400k`, but despite the 400k cap on maximum context, it consumed more than 600k, burned through the limits and restarted on grok, where I had to force it to summarize. **THIS IS ABSOLUTELY UNACCEPTABLE.** An explicitly set context limit must survive restarts and anything else, except an explicit change of the context limit via the command.
>
> Also write the context limits in the agents' status line in the web UI (where their current context is shown) and in the output of `deep-worker --info`.

Two non-negotiable requirements:

1. **Persistence guarantee** — a `context_limit` set via `-l L` survives every restart path (CLI `--restart`, web-spawn, provider failover, `merge_restart_config`); the only thing that changes it is a subsequent `-l L2` invocation. This is the operator's explicit choice; it is **load-bearing**.
2. **Enforcement guarantee** — the limit is enforced at runtime: the agent never sends a request that, by its own approximation, exceeds `context_limit`. When the request would exceed, the agent self-stops with a structured error visible in the UI; the operator decides what to do.
3. **Observability** — the limit is visible in the web-UI status bar (`context_used / limit (NN%)`) and in `deep-worker --info`.

No opt-out flag is added (per the user mandate). The limit is a hard guarantee, not a configurable hint.

---

## 2. Root cause analysis (long-running agent episode)

### 2.1 What the system did today

1. `deep-worker -r <agent> -pp claude-code --model fable -l 400k` → `parse_context_limit("400k") = 400_000` (`cli/agent_command.py:124`).
2. `build_launch_config` writes `launch_config["context_limit"] = 400_000` into `session_metadata.json` (`cli/agent_command.py:817`).
3. `merge_restart_config` (`cli/agent_command.py:531-539`) restores `context_limit` from `session_metadata.json` on every restart.
4. `agent_process.start()` passes `launch_config["context_limit"]` into `create_cli_agent(context_limit=...)` (`process/agent_process.py:1539`).
5. `create_cli_agent` forwards into `create_deep_agent(context_limit=...)` (`agent.py:1384`).
6. `create_deep_agent` calls `_compute_summarization_defaults(model, context_limit=400_000)`, which sets `trigger = ("tokens", int(400_000 * 0.85)) = ("tokens", 340_000)` (`deepagents/middleware/summarization.py:156-167`).
7. LangChain `SummarizationMiddleware._should_summarize` checks `total_tokens >= 340_000` against `usage_metadata` from each model response (`langchain/agents/middleware/summarization.py:383-409`).

### 2.2 Three latent defects

| # | Defect | Why the agent bled past the limit |
|---|---|---|
| 1 | **`_should_summarize` depends on `usage_metadata.input_tokens`/`output_tokens` being present** (`langchain/agents/middleware/summarization.py:364-381`). `claude-code` (ACP layer) does **not** reliably populate `usage_metadata` for every turn. Without it, the trigger is silently inert. | Trigger never fires; context grows past 340 k tokens turn after turn, unchallenged. |
| 2 | **No pre-call guard.** No code path counts the request tokens *before* dispatching to the API. The trigger is post-hoc (after the previous response is in). A single tool result that is larger than `context_limit - current_state` blows past the limit in one turn. | The agent produced a tool output that, on its own, exceeded the remaining headroom — and because (1) already disabled the trigger, summarization never caught up. |
| 3 | **`merge_restart_config` preserves `context_limit` but the persistence is untested.** `tests/unit_tests/cli/test_restart_preserves_launch_config.py` covers `mcp_config_paths`, `mcp_server_inline`, `turn_ending_tools`, `dev_tree`, `web_search_policy`, `preferred_model_under_failover`, `clean` — but **not `context_limit`**. A future refactor of `merge_restart_config` could silently drop the field. | Latent regression: a future contributor, "tidying up unused fields", could remove `context_limit` from the merge without any test flagging it. |

### 2.3 What this design fixes

* **Pre-call guard** that is provider-independent (works whether or not `usage_metadata` is reported), positioned **before** the API call, and instrumented to give the operator the earliest possible signal.
* **Persistence tests** for `context_limit` across every restart shape (plain restart, provider-change restart, explicit override, `-l default` clear).
* **UI display** of `context_limit` in web-UI status bar and `deep-worker --info`.
* **Provider-max validation** at startup: if `model_max < context_limit`, warn loudly so the operator sees the disagreement before the API rejects.

---

## 3. Design

### 3.1 Architecture — three layers

The guarantee is delivered by three independent layers. Each is a defense against a different failure mode.

```
                ┌─────────────────────────────────────┐
                │  Layer 3: compact_history tool      │  ← agent-initiated,
                │  (proactive compact)                │    operator-visible
                └─────────────────────────────────────┘
                  ↑ invoked by the model when it sees
                    context-approaching-limit hint
                ┌─────────────────────────────────────┐
                │  Layer 2: post-call tracker         │  ← updates the
                │  (token_stats.jsonl, UI display)    │    authoritative
                └─────────────────────────────────────┘     context_used
                  ↑ fed by usage_metadata when present,
                    fallback to pre-call approximation
                ┌─────────────────────────────────────┐
                │  Layer 1: pre-call guard            │  ← single source of
                │  (provider-independent, mandatory)  │    trigger truth
                └─────────────────────────────────────┘
```

Layer 1 is mandatory (no opt-out). Layer 2 is opportunistic (it writes what it can). Layer 3 is the agent's voluntary escape hatch.

### 3.2 Layer 1 — pre-call guard

A new middleware, `ContextLimitGuardMiddleware`, sits ahead of `SummarizationMiddleware` in the agent-middleware list. On every model call it:

1. Reads the **context size the provider reported** for this conversation (3.2.1).
2. Compares to `context_limit` (the value the operator set; if `None`, the middleware is a no-op).
3. Acts per the three-band policy (3.2.2).

#### 3.2.1 Where the number comes from — the provider, always

**HARD RULE (owner directive 2026-08-25, POLICIES §13).** The guard and the
summarizer decide on the number the PROVIDER reported — the same number the
status bar shows — and on nothing else. There is no estimator, no
chars-per-token table, no tokenizer of our own. A fallback of any kind
requires the owner's direct approval; **no provider is approved for one.**

This section used to specify exactly the opposite: a per-model heuristic
table (`len(text) // 3`, `// 4`, `+200` per tool definition) living in
"calibration source". That table and that module are deleted. What it cost:

* **2026-08-19, a long-running agent** — table said ~150k while the CLI held 468k
  against `-l 400k`. Nothing fired; the next turn loaded 466k.
* **2026-08-25, a long-running agent** — table said 214613 against a real 90k
  (`-l 250k`). The guard's amber band fired on fiction, and the hint message
  it injects tripped the codex provider's shrink detector, which dropped a
  warm session and restarted its context from zero.

The estimate was wrong in both directions because it cannot see what is not
in the LangChain message list (the CLI's own prompt and tool schemas,
extended-thinking blocks, full-size built-in results) and cannot know a
provider's tokenizer. Two providers use different tokenizers for the same
text; only the provider knows its own number.

**The contract.** `deep_worker_cli/context_limit/observed_context.py` is the
single reader. Exactly two shapes count as a provider report:

| Shape | Written by | Read as |
|---|---|---|
| `response_metadata["deep_worker_context"]` = `{used, size, first_id}` | the chat model, from the CLI's own gauge (ACP `usage_update`, codex rollout `last_token_usage`, claude-code-sdk `usage`) | `used`, valid only while `first_id` matches the first non-system message of the current history |
| `usage_metadata.input_tokens` | LangChain, for every standard API provider | the prompt size that provider read for that call |

Anything else is UNKNOWN. **UNKNOWN stays unknown**: the guard passes the
turn through and logs a warning naming the limit, and the summarizer does
not fire. A provider that never reports is a defect **in that provider** —
the warning is how it becomes visible — never a licence to estimate.

Every shipped provider reports today: ACP (grok, claude-code) stamps from
`usage_update`; codex stamps the rollout's `last_token_usage` (the same
reading that drives the live status-bar gauge); claude-code-sdk stamps its
`usage`; standard API providers report through `usage_metadata`.

#### 3.2.2 Three-band policy

`request_tokens` = the size the PROVIDER reported (3.2.1). `L = context_limit`. When nothing has been reported the size is UNKNOWN and the request is passed through as Green with a warning — never estimated (POLICIES §13).

| Band | Threshold | Action |
|---|---|---|
| Green | `request_tokens < 0.85 × L` | Pass through unchanged. |
| Amber | `0.85 × L ≤ request_tokens < 0.95 × L` | Inject a synthetic `SystemMessage` immediately before the user-visible request: `"[context-guard] Approaching limit: ~NN% of L tokens used (NNk / Lk). Consider invoking the `compact_history` tool to summarise before continuing."` The model sees the hint and may choose to compact voluntarily. |
| Red | `0.95 × L ≤ request_tokens ≤ L` | **Block the request**, return a structured `ToolMessage` to the agent: `{"error": "context_overflow_warning", "current_tokens": N, "limit": L, "fraction": f, "action_required": "invoke compact_history tool or reduce scope"}`. The model gets **one** turn to self-correct. |
| Block | `request_tokens > L` | **Hard block.** Raise `ContextLimitExceeded` (a recoverable agent-process exception caught by the no-agent-error-state policy machinery). State transitions: `RUNNING → IDLE`, error log written, WebSocket pushes a structured event to the web UI (`{"type": "context_limit_exceeded", "current": N, "limit": L}`). No LLM call is made. No auto-restart — operator decides. |

**Rationale for two-phase enforcement (Amber + Red + Block)**: the agent is given **one** voluntary turn to compact. If it does, the next request is back in Green. If it does not, the next request triggers Block (no LLM call). This avoids the catch-22 of telling a model to "summarise first" inside a request whose size has already exceeded the limit.

#### 3.2.3 Where the middleware is wired in

`create_cli_agent(..., context_limit=L)` in `agent.py` adds the new middleware **first** in the middleware list, ahead of `SummarizationMiddleware`:

```python
agent_middleware.append(ContextLimitGuardMiddleware(context_limit=L, model=self._model_family))
agent_middleware.append(SummarizationMiddleware(...))   # existing
```

The middleware is skipped entirely when `context_limit is None` (operator did not set `-l`).

### 3.3 Layer 2 — post-call tracker

`deep_worker_cli/stats/token_stats.py::UsageRecord` gains a new field `context_size: int | None = None`, and `build_record` populates it from the resolved `context_limit` (read once at agent start, not per-call — the value is stable for the lifetime of the process unless `-l` changes via restart, in which case the new process picks up the new value).

`deep_worker_cli/stats/callback.py::TokenStatsCallbackHandler._build_record` writes `context_size` alongside `prompt_tokens` and `completion_tokens`. Web UI's `_fill_context_fields` reads it via the existing `latest_context_size` aggregator field.

When `usage_metadata` is present, **both** `prompt_tokens` (real) and `context_size` (limit) are written — the UI label reads `(real)`. When `usage_metadata` is absent, the UI label reads `(approx)` and uses the pre-call estimate for `prompt_tokens`.

### 3.3a Observed-context floor (addendum 2026-08-19, 468k vs `-l 400k`)

**Incident.** With `-l 400k` the agent reached a CLI-reported context of
468,146 tokens and neither the summarizer (trigger 340k) nor the guard
fired. Replay of the real checkpoint: 722 messages, `count_tokens_approximately`
= 151,617, guard estimator = 225,631, CLI `usage_update.used` = 468,146.
The heuristics cannot see extended-thinking blocks, the CLI's own system
prompt / tool schemas, or full-size built-in tool results — for a provider
that runs its own session (claude-code over ACP) they are off by 3× in the
dangerous direction.

**Fix — read the provider's number; never compute one.**

* **Producers** (every chat model): each returned `AIMessage` — final and
  mid-turn — is stamped with
  `response_metadata["deep_worker_context"] = {"used", "size", "first_id"}`
  from the CLI's own gauge. ACP (`grok`, `claude-code`) stamps from
  `usage_update`; codex stamps the rollout's `last_token_usage` (the same
  reading behind the live status-bar gauge); claude-code-sdk stamps its
  `usage`. `first_id` is the id of the first non-system message of the
  request, so a stamp is bound to the history it measured and survives
  restarts inside the checkpoint. Standard API providers need no stamp:
  LangChain already carries `usage_metadata.input_tokens`, which is the
  same fact.
* **Reader** (`deep_worker_cli/context_limit/observed_context.py`):
  `provider_context_tokens(messages)` returns the latest valid report, or
  `None` for UNKNOWN. There is no arithmetic on top of it — nothing is
  added for the tail, because anything added would be invented.
* **Consumers**: `ContextLimitGuardMiddleware._provider_context` and the
  summarizer's counter (`provider_context_token_counter`, wired in
  `agent.py`). Both see the same number, and it is the number in the UI.

**Validity (the loop guard).** After summarization the history is
`[summary(new id), *kept_tail]`; tail AIMessages still carry the old 468k
stamp. `first_id` no longer matches → ignored → the size is UNKNOWN until the
fresh session reports one. The same rule applies to the summarizer's
`messages[mid:]` tail slices (a slice never starts at
the real first message), so the keep budget is not poisoned by the floor.

**Replay on the real checkpoint**: before — 151,617, no
summarization; after stamp — 468,146 → summarize, cutoff 523/722, kept
tail ≈ 40k, post-summary count 40,018 → no re-trigger.

**Known gap**: a checkpoint written by pre-fix code has no stamps; the
first turn after upgrade has no report yet (UNKNOWN), the stamp lands on
that turn's AIMessage, and the limit is enforced from the second turn on.

### 3.4 Layer 3 — `compact_history` tool

A new built-in tool, `compact_history`, exposed via the standard tool registry. The agent invokes it voluntarily (in response to the Amber-band hint) or the operator invokes it from the web UI's right-click menu.

```python
@tool
def compact_history() -> str:
    """Summarise the conversation history in place.

    Returns a short ToolMessage confirming the new approximate token count.
    Idempotent: invoking twice in a row is a no-op the second time.
    """
```

Internally it delegates to the existing `SummarizationMiddleware.before_model` summarisation flow with `force=True`; the new layer-1 middleware is **skipped** for the summarisation turn (otherwise summarisation would itself trip the limit). After summarisation completes, the next user-facing turn re-enters Green.

The tool is registered automatically by `create_cli_agent` whenever `context_limit is not None`.

### 3.5 Persistence tests

`tests/unit_tests/cli/test_restart_preserves_launch_config.py` gains three tests (regression-net for `merge_restart_config`):

```python
async def test_restart_preserves_context_limit(_sandbox, monkeypatch):
    """-l 400k set on first launch must survive --restart verbatim."""

async def test_restart_with_different_provider_preserves_context_limit(_sandbox, monkeypatch):
    """Switching provider (claude-code -> grok, the original incident scenario)
    must NOT drop context_limit. Operator choice is provider-independent."""

async def test_restart_explicit_l_flag_overrides_saved_context_limit(_sandbox, monkeypatch):
    """Re-launching with -l 800k replaces the saved 400k."""

async def test_restart_l_default_clears_saved_context_limit(_sandbox, monkeypatch):
    """Re-launching with -l default clears the saved override; future
    restarts use the model default (None) again."""
```

Each test runs the full `restart_agent` path with a captured `AgentProcess.start` (the existing `_install_agent_process_spy` pattern) and asserts on `effective.context_limit` post-merge.

### 3.6 Provider-max validation at boot

`deep_worker_cli/context_limit/provider_max.py` exports:

```python
PROVIDER_MODEL_MAX_INPUT: dict[tuple[str, str], int] = {
    ("claude-code-sdk", "sonnet"):    200_000,
    ("claude-code-sdk", "opus"):      200_000,
    ("claude-code-sdk", "haiku"):     200_000,
    ("claude-code",     "sonnet"):    200_000,
    ("claude-code",     "opus"):      200_000,
    ("grok",            "grok-3"):    131_072,
    ("openai",          "gpt-4-turbo"): 128_000,
    ("openai",          "gpt-5"):     400_000,
    # ...
}
```

`AgentProcess.start()` resolves `(provider, model)`, looks up the entry, and:

* If `context_limit > max_input`: print a one-line WARNING to stderr with the override proposal (`"context_limit 400k exceeds claude-code+sonnet max 200k — clamping to 200k. Re-run with -l <X> < 200k to keep your setting."`). Then clamp and proceed.
* If no entry exists: skip (silent — the API will reject and the operator will learn).
* If `context_limit is None`: skip silently.

This is a **boot-time** check, not a runtime check — it surfaces the disagreement once, before any LLM call.

### 3.7 UI display

#### 3.7.1 Web UI status bar

The web service's `agent_service.py::_fill_context_fields` already populates `agent_context_used`, `agent_context_size`. The frontend (the web UI status-bar component — TBD exact path) renders:

```
Context: 234k / 400k (59%)   [██████████░░░░░░░░░░]
                  ^               ^       ^   ^
                  used            limit   %   bar
```

When `usage_metadata` was unavailable for the most recent turn, the suffix becomes `(approx)`. When `context_size is None` (operator never set `-l`), the bar collapses to `Context: 234k / model-default (59%)` and the cap is the model's intrinsic `max_input_tokens`.

The full-dialog "context budget" tool (§9 of `DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md`) gains a new field `limit` (the operator-set value, or `None` for default).

#### 3.7.2 `deep-worker --info`

`cli/info_command.py` (path TBD — see 3.7.3) reads `session_metadata.json` for the named agent and prints:

```
Agent:           <agent>
Provider:        claude-code
Model:           fable
Context limit:   400,000 tokens (-l 400k, operator-set)
Last used:       234,567 tokens (59% of limit)
Last summarisation: 2026-08-18 14:22:01 UTC
Provider max:    200,000 tokens (claude-code + fable)
```

#### 3.7.3 Backend status display

The CLI's own status line (printed under the prompt — `deep_worker_cli/app.py:TokenDisplay`) gains a `(limit)` suffix:

```
[234k / 400k] <agent>>
```

The existing `TokenDisplay.add(total_tokens)` is extended to `TokenDisplay.add(total_tokens, limit=...)`; the limit is set once at agent start from `launch_config["context_limit"]`.

### 3.8 What this design does NOT do

* **No opt-out flag.** The user mandate is unambiguous. If a workflow genuinely needs a context larger than what a model supports, the right answer is `-l default` and a model with a larger window — not a back-door bypass.
* **No auto-restart on overflow.** The user (operator) sees the structured error in the UI and decides. Auto-restart hides the failure and re-loads the same bloated history.
* **No migration of existing bloated agent DBs.** The related backlog item (a threads.db at 459 MB) is its own work item; this design does not touch it. A bloated DB will, on boot, trip the Red band on the first turn — which is exactly the early-warning signal the operator needs to address it.
* **No changes to `SummarizationMiddleware` itself.** The pre-call guard sits ahead of it; the LangChain-internal trigger is left alone (it remains a second-best fallback for when usage_metadata is present).
* **No per-tool-call estimation.** A `read_file(big.py)` whose single result is > context_limit is **out of scope** for this change (noted in review). Backlog candidate for a follow-up.

---

## 4. File layout (planned changes)

```
deep_worker/deep_worker_cli/deep_worker_cli/
├── context_limit/                          # NEW package
│   ├── __init__.py
│   ├── provider_max.py                     # PROVIDER_MODEL_MAX_INPUT table
│   └── guard.py                            # ContextLimitGuardMiddleware
├── middleware/
│   └── compact_history_tool.py             # NEW — agent-facing compact tool
├── stats/token_stats.py                    # +context_size field, +calibration file
├── stats/callback.py                       # wire context_size into the record
├── agent.py                                # mount guard + compact_history tool
├── process/agent_process.py                # provider-max validation at start
└── cli/info_command.py                     # NEW (or extend existing --info path)

deep_worker/deep_worker_cli/tests/unit_tests/
├── cli/test_restart_preserves_launch_config.py   # +4 persistence tests
├── context_limit/
│   ├── test_provider_max.py
│   └── test_guard_three_bands.py
├── middleware/test_compact_history_tool.py
└── stats/test_token_stats_writes_context_size.py

web service agent_service.py                      # expose context_size
web UI AgentStatusBar.vue                         # context/limit bar (UI-only)
docs/CI_CD_TESTS.md                                # unchanged (no new CI rules)
docs/CI_CD_TESTS.md                                # unchanged
docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md            # THIS FILE
```

---

## 5. Test plan

### 5.1 Unit tests (mandatory, run on every push)

| Test | Asserts |
|---|---|
| `test_provider_context_only` | the guard and the summarizer decide only on a provider report; UNKNOWN passes through and is logged; the estimator module cannot come back (structural check) |
| `test_token_estimator_openai_uses_tiktoken_if_available` | When `tiktoken` importable, exact `cl100k_base` count is returned. When not, falls back to `len // 4` without raising. |
| `test_token_estimator_minimax_grok_fallback` | Conservative over-estimate (`>=` observed) for the unknown families. |
| `test_guard_green_band_passes_through` | `request_tokens < 0.85 L` → no SystemMessage injection, no block. |
| `test_guard_amber_band_injects_hint_system_message` | `0.85 L ≤ request_tokens < 0.95 L` → one SystemMessage is prepended, model call proceeds. |
| `test_guard_red_band_returns_tool_message` | `0.95 L ≤ request_tokens ≤ L` → no LLM call, ToolMessage with `error: context_overflow_warning` returned. |
| `test_guard_block_band_raises_context_limit_exceeded` | `request_tokens > L` → `ContextLimitExceeded`, agent transitions to IDLE, no LLM call. |
| `test_guard_skipped_when_context_limit_is_none` | `context_limit=None` → middleware is a no-op (no overhead). |
| `test_compact_history_tool_is_idempotent` | Two consecutive invocations → second is a no-op. |
| `test_compact_history_tool_resets_to_green_band` | After compact, next request enters Green band. |
| `test_token_stats_writes_context_size_field` | UsageRecord.to_dict contains `context_size`. |
| `test_restart_preserves_context_limit` | `-l 400k` survives `--restart`. |
| `test_restart_with_different_provider_preserves_context_limit` | `-l 400k -pp claude-code` → restart with `-pp grok` keeps 400k. |
| `test_restart_explicit_l_flag_overrides_saved_context_limit` | `-l 400k` → restart with `-l 800k` → 800k. |
| `test_restart_l_default_clears_saved_context_limit` | `-l 400k` → restart with `-l default` → None. |
| `test_provider_max_clamps_at_boot` | `-l 500k -pp grok --model grok-3` → boot-time WARN, stored as 131_072. |
| `test_provider_max_unknown_model_silent` | `(claude-code, fable)` not in table → no warn, no clamp. |
| `test_calibration_file_persists_observed_ratios` | After 50 turns with usage_metadata, `context_limit_calibration.jsonl` has 50 records. |

### 5.2 Integration tests (run with `-m integration`)

These exercise the **real** agent-process boot path with a mock model — no live API.

| Test | Asserts |
|---|---|
| `test_guard_blocks_at_runtime` | Mock model that requests > L tokens: guard blocks, agent state → IDLE, no model invocation in mock. |
| `test_compact_history_tool_via_middleware_chain` | Mock agent fills context to Amber band, invokes compact_history tool via the middleware chain, next request returns to Green. |
| `test_web_ui_status_bar_includes_limit` | The `AgentInfo` returned by `AgentService.discover` carries `agent_context_size` populated from `launch_config["context_limit"]`. |
| `test_info_command_prints_context_limit` | `deep-worker --info <agent>` output contains `Context limit:` line with the operator's value. |

### 5.3 Manual smoke (documented in PR description)

```bash
# Boot an agent with -l 400k.
deep-worker -r <agent> -pp claude-code --model fable -l 400k

# Confirm status line shows "[...] / 400k".
# Confirm web UI status bar shows "Context: NNk / 400k".
# Force the agent to issue a 350k-token tool result (use a synthetic fixture).
# → Amber band: SystemMessage hint visible in next turn.
# Force a 500k-token tool result.
# → Block: agent transitions to IDLE, UI shows red banner.
```

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| A provider stops reporting its context size | The size is UNKNOWN: the guard passes through and WARNS on every such call, so the gap is visible in the log instead of being papered over by a guess (POLICIES §13). Fixing the provider is the only remedy. |
| tiktoken absent → fall back to `len // 4` | Documented; calibration file flags drift; over-estimate is safe. |
| Provider API rejects oversize request even though our guard says it should fit | Boot-time validation (3.6) surfaces the disagreement. |
| `compact_history` summarisation itself exceeds limit | The middleware is skipped for the summarisation turn (3.4). |
| Bloated history (459 MB) trips Block band on first turn | The related backlog item is independent; this design surfaces the problem earlier, which is the correct behaviour. The operator decides what to do. |
| Race between concurrent turns (agent + tm) | Each invocation re-estimates; per-turn guard is independent. Layer 2 (`prompt_tokens` from response) reconciles after the fact. |

---

## 7. Rollout

1. **Phase 1 — Pre-call guard + persistence tests + provider-max validation.** This is the safety net. Mandatory. Shipped as one PR with pre-commit hooks passing and CI green.
2. **Phase 2 — UI display (web status bar + `--info` + CLI status line).** Observability. Shipped as a second PR after Phase 1 is stable in production for one full business day.
3. **Phase 3 — `compact_history` tool.** The voluntary escape hatch. Shipped as a third PR once the agent population has learned the Amber-band hint and anecdotally wants to compact earlier.

---

## 8. Out of scope (explicit backlog candidates)

* Per-tool-call estimation for `read_file` / `web_search` / similar large-output tools.
* Lazy-load of `threads.db` to avoid loading 459 MB at boot (related backlog item).
* `context_limit` settings via the web UI (currently CLI-only).
* Cross-agent context sharing for multi-agent task topologies.
* Per-message token attribution for cost analytics.