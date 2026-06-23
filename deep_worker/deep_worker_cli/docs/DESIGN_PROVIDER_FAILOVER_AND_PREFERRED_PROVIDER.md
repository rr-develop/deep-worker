# DESIGN: Provider failover on usage limit + preferred-provider memory

**Status:** PROPOSAL (not implemented)
**Date:** 2026-06-05
**Scope:** `deep_worker_cli/cli/agent_command.py`, `deep_worker_cli/session/session_manager.py`,
`deep_worker_cli/process/agent_process.py`, `claude_code_provider/chat_model.py`,
`codex_provider/exec_runner.py`, `codex_provider/chat_model.py`, plus a new
provider-availability tracker.

## 0. Motivation

Two providers we use in production — `claude-code` (Claude Pro/Max via the
local `claude` CLI) and `codex` (OpenAI Codex via the local `codex` CLI on a
ChatGPT account) — periodically refuse work because the account-level usage
window is exhausted. When that happens:

- the agent transitions to `status=ERROR` (claude-code, via the typed
  `ClaudeUsageLimitError` we added);
- or it dies with a generic `CodexProtocolError` whose message contains the
  upstream marker (codex — currently NOT typed as a dedicated error class);
- in either case it stops doing the user's task until a human notices,
  manually restarts the agent on a different provider, and (optionally) puts
  the original provider back when the limit resets.

We want this to be automatic:

1. **Detect** the limit event the moment it surfaces.
2. **Restart** the agent on the next provider in a preference chain so it
   can keep working on the same task / conversation history.
3. **Remember** which provider the agent was meant to run on
   (`preferred_provider`) so an automatic failover doesn't silently
   overwrite the operator's choice.
4. **Track** when each provider becomes available again (the limit message
   tells us the reset time — parse it).
5. **Restore** the agent to its preferred provider once that provider is
   back, but only when the agent is **idle** (between tasks) so we don't
   abort work mid-turn.
6. (Stretch) **Hot-swap** the provider inside a running process without a
   full restart, when the underlying model state allows it.

This document records the markers we already have, the existing restart
mechanism, and the concrete plan for adding failover.

## 1. What we know — exact provider failure markers

Inventory from 10 days of logs across ~25 agents
(an ad-hoc log-scanning script):

### 1.1 claude-code: session-window exhaustion

**Raw marker emitted by the `claude` CLI:**

```
You've hit your session limit · resets <TIME> (<TZ>)
```

Observed reset values: `5:40pm (Europe/Amsterdam)`,
`11:10pm (Europe/Amsterdam)`, `7pm (Europe/Amsterdam)`.

Total occurrences in scanned logs: **317**.

**Already wired** in code:

- `claude_code_provider/chat_model.py:50` — `class ClaudeUsageLimitError(RuntimeError)`.
- `claude_code_provider/chat_model.py:66` — `_SESSION_LIMIT_MARKER = "You've hit your session limit"`.
- `claude_code_provider/chat_model.py:1049` — `ainvoke` detects the marker
  on a text-only turn and raises `ClaudeUsageLimitError`.
- `agent_process.py:2352–2356` — `_handle_invoke` catches by class name
  (provider-agnostic), sets `status=ERROR`, error_message =
  `Claude usage limit reached — <marker>`.

So a failover orchestrator can subscribe to the existing `error` event from
the agent and pattern-match on either:

```python
type(exc).__name__ == "ClaudeUsageLimitError"
# or, on the message
re.compile(r"You've hit your session limit\s*[·•]\s*resets?\s+(?P<reset>.+?)(?:\s*\(|$)")
```

### 1.2 codex: usage-cap exhaustion

**Raw marker emitted by the `codex` CLI (single canonical form, 45 hits):**

```
Codex stream error: You've hit your usage limit. Upgrade to Pro
(https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage
to purchase more credits or try again at <TIME>.
```

Observed reset values: `11:32 PM.`, `7:10 PM.`, `May 27th, 2026 3:15 AM.`,
`May 29th, 2026 1:23 AM.`, `May 31st, 2026 12:21 PM.`.

**Currently NOT typed** as a dedicated error class:

- `codex_provider/exec_runner.py:51` — only generic
  `class CodexProtocolError(RuntimeError)`.
- The marker lives in `str(exc)`. The existing
  `[codex_provider] resume of '<uuid>' failed (CodexProtocolError(...))`
  branch tries a fresh session and replays history — and then hits the
  same limit again and propagates the same string.

Detection without code changes:

```python
isinstance(exc, CodexProtocolError) and "hit your usage limit" in str(exc)
```

Detection regex on log lines:

```python
re.compile(
    r"Codex stream error: You've hit your usage limit\b.*?"
    r"try again at (?P<reset>[^.]+?)\."
)
```

### 1.3 NOT observed in our logs (must not invent markers for these)

| Category | Hits |
|---|---:|
| `claude` weekly_limit | 0 |
| `claude` `overloaded_error` / HTTP 529 | 0 |
| `claude` `RateLimitError` class | 0 |
| `claude` HTTP 429 | 0 |
| OpenAI `rate_limit_exceeded` / `insufficient_quota` | 0 (Codex CLI bypasses the API path) |
| `codex` `context_length_exceeded` | 0 |
| `codex` HTTP 401 / `invalid_api_key` | 0 |

If we hit these later, add markers then, with the same discipline.

### 1.4 NOT a provider limit (do NOT failover)

Codex misconfiguration looks similar but is not a limit — failover would
just burn the next provider's quota on the same broken config:

```
Codex stream error: {"type":"error","status":400,"error":{
  "type":"invalid_request_error",
  "message":"The '<MODEL>' model is not supported when using Codex with a ChatGPT account."}}
```

Detection:

```python
re.compile(r'invalid_request_error.*model.*not supported.*ChatGPT account')
```

Action: surface as ERROR with a config-fix hint, NOT failover.

## 2. Reset-time parsing

The two CLIs use different formats. We need a small parser that returns a
`datetime` (UTC) or `None` (= "unknown, retry never until something else
proves availability").

### 2.1 claude format

```
resets <H>[:MM][am|pm] (<TZ>)
```

Examples: `5:40pm`, `11:10pm`, `7pm`. Timezone always parenthesised
(observed only `Europe/Amsterdam`, but we should not hard-code it).

Algorithm:

1. Parse `<H>[:MM]<am|pm>` as a wall-clock time.
2. Parse the timezone label via `zoneinfo.ZoneInfo`.
3. Combine with **today's date** in that timezone; if the resulting moment
   is in the past relative to now, add 1 day (the reset is for the *next*
   such wall-clock).
4. Convert to UTC.

### 2.2 codex format

Two shapes:

- absolute: `May 31st, 2026 12:21 PM.`
- relative wall-clock: `11:32 PM.` / `7:10 PM.` (assumed in the agent's
  local TZ — the codex CLI gives no timezone label).

Algorithm:

1. Try to parse the absolute form (`%B %dst/nd/rd/th, %Y %I:%M %p` — strip
   the ordinal suffix first).
2. If that fails, parse as wall-clock in the agent host's local timezone
   (same `<H>[:MM]<am|pm>` logic as claude, +1 day if in the past).
3. Convert to UTC.

A reset moment that can't be parsed is **honest** as `None` — the agent
just won't auto-restore preference until the next probe (see §6).

## 3. Existing restart mechanism (what we already have)

`deep_worker_cli/cli/agent_command.py`:

- `--restart NAME [--provider X] [--model Y] [...]` already works.
- `load_launch_config(name)` reads `session_metadata.json` → `launch_config`.
- `_merge_launch_config(saved, args)` (lines 149+) merges saved fields with
  CLI overrides. Key rule (lines 159–162): when `provider` is overridden
  and `--model` is NOT explicitly given, the saved model is reset to None
  so the new provider's default kicks in (prevents `--model opus` leaking
  into a codex run, as in an earlier agent incident).
- `SessionManager.__init__(launch_config=...)` persists the new
  `launch_config` to `session_metadata.json` and resets transient state
  (status reset on a new launch, line 100+).

What's missing for failover:

- No notion of `preferred_provider`. Every `--provider` override
  overwrites the persisted `provider` field — there is no separate "what
  the operator originally wanted" slot.
- No automatic trigger. The operator must run `deep-worker --restart`
  manually after seeing the ERROR.
- No availability tracker.

## 4. Required additions — overview

| Component | Where | Adds |
|---|---|---|
| `preferred_provider` field | `launch_config` schema (storage_version bump) | A second slot, NEVER mutated by automatic failover |
| `--preferred-provider X` CLI arg | `agent_command.py` parser | Explicit way to set / change preference without rotating the active provider |
| `--no-update-preferred` CLI flag | `agent_command.py` parser | Restart without touching `preferred_provider` even though `--provider` is given (used by automatic failover) |
| Typed `CodexUsageLimitError` | `codex_provider/chat_model.py` | Subclass of `CodexProtocolError`, raised on marker match — symmetric with `ClaudeUsageLimitError` |
| `availability_tracker.py` | new module under `deep_worker_cli/runtime/` | In-process registry: `{provider_name: {"available_after_utc": dt|None, "last_failure_marker": str, "last_failure_at": dt}}` |
| Failover orchestrator | new method on `AgentProcess` OR external service | Subscribes to its own `error` events, parses reset, kicks off `--restart … --provider <next> --no-update-preferred` |
| Restore-on-idle scheduler | new method on `AgentProcess` OR external service | When agent goes idle AND `current_provider != preferred_provider` AND tracker says preferred is available → `--restart … --provider <preferred> --no-update-preferred` |
| (Stretch) hot-swap | `agent_process.py` + provider `replace_chat_model()` | Replace `chat_model` in the live agent graph without process kill |

## 5. Provider chain & failover order

The user-specified default order is:

```
claude-code  →  codex  →  grok  →  deepseek
```

Rules:

1. The chain starts from the agent's `preferred_provider`.
2. On failover, advance to the next provider in the chain that the
   availability tracker currently considers `available_after_utc <= now`
   (or `unknown`).
3. `deepseek` is the **terminal** provider — it doesn't have a usage cap
   we've observed, so an agent that reaches it stays there until either
   `preferred_provider` reopens or the operator manually intervenes.
4. If `preferred_provider == deepseek`, no failover is ever triggered
   (we have nowhere better to go — surface the ERROR honestly).
5. The chain is a **module constant**, not a per-agent setting (keep
   complexity in check; if a use case appears, add per-agent override
   later).

```python
# deep_worker_cli/runtime/provider_chain.py
PROVIDER_FAILOVER_CHAIN: list[str] = ["claude-code", "codex", "grok", "deepseek"]
```

## 6. Availability tracker

A single in-process singleton per `deep-worker` host (the same process
that supervises agents). Backed by a JSON file at
`provider_availability.json` under each user's data directory so that
supervisor restarts don't forget reset times.

Schema:

```json
{
  "claude-code": {
    "available_after_utc": "2026-06-04T18:10:00+00:00",
    "last_failure_marker": "You've hit your session limit · resets 7pm (Europe/Amsterdam)",
    "last_failure_at_utc":  "2026-06-04T13:55:11+00:00",
    "last_failure_agent":   "<agent_name>"
  },
  "codex": null,
  "deepseek": null
}
```

Update on every observed failure event (any agent on this host). All
agents read from the same tracker, so a usage cap that one agent triggers
is immediately known to all others (no point in each of N agents
discovering it independently).

API:

```python
def record_failure(provider: str, marker_text: str, agent_name: str) -> None: ...
def is_available(provider: str) -> bool: ...      # True if no record OR available_after_utc <= now
def next_available(provider: str) -> datetime | None: ...
def clear(provider: str) -> None: ...             # called when an invocation succeeds on that provider
```

## 7. The flow

### 7.1 On agent invocation that hits a limit

1. `chat_model` raises `ClaudeUsageLimitError` / `CodexUsageLimitError`.
2. `_handle_invoke` (agent_process):
   - calls `availability_tracker.record_failure(<provider>, <marker>, <name>)`
     with the parsed reset moment;
   - sets `status=ERROR` with the marker message (as today);
   - broadcasts the error event (as today);
   - **schedules a failover** (new): compute
     `next_provider = first(p for p in PROVIDER_FAILOVER_CHAIN[idx_of(current)+1:] if tracker.is_available(p))`.
     If `next_provider is None` → stop, the agent stays ERROR.
3. Failover runs `deep-worker --restart <name> --provider <next> --no-update-preferred`.
   `--no-update-preferred` makes the merger keep the old
   `preferred_provider` value while updating `provider`.
4. The new process starts on the next provider, restores history (each
   provider already has resume — claude session import, codex session
   uuid, deepseek SQLite checkpointer), and continues the task.

### 7.2 On agent going idle

A small loop (NOT a poll — event-driven, fired only on the existing
`status_change → idle/waiting_input` transition):

1. If `current_provider == preferred_provider`: nothing to do.
2. If `availability_tracker.is_available(preferred_provider)` is False:
   nothing to do.
3. Otherwise: `deep-worker --restart <name> --provider <preferred>
   --no-update-preferred`. Same restore-history path.

Hard invariants:

- This NEVER fires while `status` is `working`. The whole point of
  preferring "idle" is to avoid yanking the agent mid-turn.
- This NEVER fires for an agent whose `preferred_provider == "deepseek"`
  (no point — deepseek is terminal).
- This NEVER runs more than once per provider-reopen event. Once we've
  restarted, we clear the tracker entry for that provider; the next
  failure resets it.

### 7.3 Operator overrides

- `deep-worker --restart NAME --preferred-provider claude-code` →
  updates BOTH `preferred_provider` AND `provider`. Use when the operator
  explicitly wants a new preference.
- `deep-worker --restart NAME --provider codex` → updates ONLY
  `provider`, leaves `preferred_provider` intact. (This is the SAME
  behaviour as today, by design — operator restarts are usually
  experimental and shouldn't lose the long-term preference.)
- `deep-worker --restart NAME --provider codex --no-update-preferred` →
  same as above; the flag is redundant for operator-typed commands but
  required for the automatic failover trigger (so its intent is
  explicit and unambiguous in logs).

### 7.4 First-run defaulting

Agents created before `preferred_provider` existed have no value in the
persisted `launch_config`. The merger fills it in:

```python
launch_config["preferred_provider"] = launch_config.get("preferred_provider") \
                                       or launch_config["provider"]
```

This runs once on the next normal restart and is sticky thereafter.
`storage_version` is bumped to 5 to signal the new schema; no destructive
migration — the old field is left in place.

## 8. Hot-swap (stretch goal)

The straightforward path is "exit + restart". It costs one process spawn
(~1–2s) and a history replay (~5–15s for an agent with a large thread).
For agents in the middle of a long invocation that's significant — and
for agents using inter-agent messaging it can drop in-flight peer
messages.

A hot-swap would work like this:

1. New API on each provider: `chat_model.replace_underlying(new_provider_args)`
   that swaps the inner LangChain `BaseChatModel` while keeping the
   surrounding middleware (history, tool_call decoding, etc.) intact.
2. `AgentProcess` exposes `exposed_swap_provider(new_provider, new_model)`
   over RPyC. It awaits the current invocation (so it never preempts a
   working turn), creates a new `chat_model`, atomically swaps it under
   a lock, persists the new `launch_config`, and resumes.

Risks to validate before committing:

- **Tool registry mismatch.** Each provider has slightly different tool
  registration (claude_code has the proxy-MCP layer, codex has its own
  args). If we naïvely swap chat_model, the tools attached to the
  LangGraph agent stay the same, which may not be valid for the new
  provider. Need to check `create_cli_agent` carefully.
- **Resume state.** claude-code uses session import via a session file;
  codex uses a session uuid; deepseek uses SQLite checkpointer. A
  hot-swap has to either drop the resume info and replay full history
  (loses some context) or translate it (hard).
- **Streaming state.** A swap during a stream is impossible — must wait
  for `current_worker_task` to settle.

Recommendation: implement §7 (full restart-based failover) first. Do NOT
ship hot-swap in the same PR. Open a follow-up only after §7 is
production-stable for a week or two, and only if restart cost is shown to
be a real problem.

## 9. Acceptance criteria

For the §7 (restart-based) implementation:

1. New field `preferred_provider` exists in persisted `launch_config`,
   bumped `storage_version`, backfilled on next load.
2. New CLI args `--preferred-provider` and `--no-update-preferred` work
   as specified in §7.3, with unit tests covering all combinations of
   `(--provider given?, --preferred-provider given?, --no-update-preferred?)`.
3. New `CodexUsageLimitError(CodexProtocolError)` class, raised on marker
   match. `agent_process.py` catches it by class name (provider-agnostic,
   same pattern as `ClaudeUsageLimitError`) and writes a clean
   `Codex usage limit reached — <marker>` to `error_message`.
4. `availability_tracker` persists/restores correctly across supervisor
   restarts. The JSON file is honestly missing fields when reset can't
   be parsed (no synthesised guesses).
5. Failover trigger fires within one event of a limit error and runs the
   correct `--restart` invocation. End-to-end test: agent on claude-code
   with injected limit response → after the failover, the SAME agent
   process name is alive on codex with the same conversation thread.
6. Restore-on-idle fires only on real `→ idle` transitions, never on
   `working`, never twice per reopen. Log analysis 24 h after deploy
   should show: number of restore-on-idle restarts == number of
   tracker.clear() events.
7. Misconfig markers (§1.4) do NOT trigger failover. Test covers the
   `'unknown' model is not supported` line.
8. `preferred_provider == "deepseek"` agents never trigger failover and
   never trigger restore-on-idle.
9. The polling invariant from `DESIGN_EVENT_DRIVEN_STATUS.md` stays
   green — failover and restore-on-idle must both be triggered by
   existing events, never by `asyncio.sleep(N)` loops.

## 10. Out of scope (separate tasks if/when needed)

- Hot-swap (§8) — separate task after restart-based path proves stable.
- Per-agent override of the failover chain.
- Notifying the operator (Slack/email) when an agent has been on a
  non-preferred provider for > N hours.
- Cost-aware preference (e.g. "prefer codex during US business hours
  because Claude credits are scarce in that window") — interesting, but
  not motivated by current data.
- Automatic provider chain on the **first-ever** start of an agent (the
  operator picks; the chain only governs failover and restore).

## 10b. The TaskManager path must propagate fatal errors too (incident 2026-06-17)

The failover/ERROR transition above happens in the **invoke-handler**
(`agent_process.py` — `_handle_invoke`): it catches the provider usage-limit
by class name, calls `set_error` + `_schedule_failover`. That is the ONLY
place that transitions status and triggers the chain.

But an agent running with `-tm` (TaskManager) drives the manager agent through
`TaskManager.on_subordinate_response` → `_evaluate_with_manager`. If those
methods **swallow** the provider exception into `None`, it never reaches the
invoke-handler, so **no failover and no ERROR happen** — the agent settles to
a false `running`.

This is exactly the 2026-06-17 codex incident: TaskManager re-raised only
`ClaudeUsageLimitError` and swallowed `CodexUsageLimitError` (and any other
provider's usage-limit). Fix: both TaskManager `except` blocks now re-raise
**any** fatal exception via the SINGLE provider-agnostic registry
`deep_worker_cli.exceptions.is_fatal_exception` (which already contains every
`*UsageLimitError`, `ProviderAuthFatalError`, etc.), instead of hard-coding one
provider's class name. Non-fatal (transient) errors are still swallowed to
`None` so the TaskManager loop stays resilient.

Contract: **adding a new provider's usage-limit to `FATAL_EXCEPTION_TYPES` is
sufficient** — it then propagates through BOTH the invoke-handler AND the
TaskManager path automatically. No second list to update.

See a related internal incident analysis (not included here).
Tests: `tests/unit_tests/task_manager/test_session_limit_propagates.py`
(codex + any-registered-fatal propagate; non-fatal still `None`) and
`tests/unit_tests/process/test_handle_invoke_session_limit.py`
(`test_codex_session_limit_sets_error_and_schedules_failover`).

## 11. References

- Log inventory of provider failure markers: this document §1 (raw data
  collected by an ad-hoc log-scanning script).
- Existing typed exception: `claude_code_provider/chat_model.py:50,66,1049`.
- Existing handler: `deep_worker_cli/process/agent_process.py:2352-2356`.
- Existing restart merger: `deep_worker_cli/cli/agent_command.py:149+`.
- Persistence layer: `deep_worker_cli/session/session_manager.py` (the
  storage-version bump must be coordinated with `LAUNCH_CONFIG_PERSISTENCE.md`).
- Related design (event-driven, no polling):
  the event-driven status design notes.
- Codex error class: `codex_provider/exec_runner.py:51`.
- Codex resume-fallback (which today re-fails on the same limit):
  `codex_provider/chat_model.py` (search for "resume" + "failed").
