# DESIGN: per-turn usage mapping (codex + claude-code-sdk)

Status: DESIGN — awaiting reviewer verdict. No code, no tests written.
Date: 2026-08-22.
Predecessor: MAPPING VERDICT (reviewer, "APPROVE with corrections").

This document implements the mapping table **exactly in the reviewer's
edition**. Where my own live probe contradicts a premise of that verdict,
the contradiction is called out explicitly in §2.1 rather than silently
"fixed" — it changes the size of the codex work item and needs the
reviewer's ack before implementation.

---

## 0. Problem

`prompt_tokens` for some providers is **cumulative** (session-to-date), not
per-turn. A cumulative number written into a per-turn field makes every
downstream consumer wrong: per-turn cost, context-occupancy display, and
the summarization trigger. Live evidence of the inflation:

- codex row (an agent's token stats, 18:38:33.222Z): `prompt=3736695 hit=3600640
  out=5212 total=3741907` — the exact `total_token_usage` blob, 4/4 fields,
  while the same instant's `last_token_usage.input_tokens` was `113457`.
  A `window` of `258400` makes `3736695` structurally impossible as
  occupancy: `tot_in` exceeds the window in 87/101 samples, `last_in` in 0/101.
- claude-code-sdk `kind=agent` rows on one agent: 59/81 with `prompt>250k`,
  29/81 `>1M`, max `17236028`. No `AssistantMessage` in the project jsonl is
  in that magnitude.

Goal: `prompt_tokens` becomes **per-turn** for the two in-scope sources,
with no change to `NormalizedUsage` and no change to the arithmetic.

---

## 1. Mapping table (reviewer's edition, verbatim intent)

Two **separate axes**. Crossing them is the defect class this whole task
exists to prevent (the rg1 "2265.4k as Context" incident).

- **Spend** (token fields): `prompt_tokens`, `completion_tokens`,
  `cache_hit_tokens`, `cache_miss_tokens`, `total_tokens`.
- **Gauge** (occupancy): `context_used`, `context_size`.

A gauge value MUST NEVER be written into a token field, and a token field
MUST NEVER be written into a gauge.

### 1.1 Token fields

| our field | codex | claude-code-sdk | grok |
|---|---|---|---|
| `prompt_tokens` | `last_token_usage.input_tokens` | last `AssistantMessage.usage`: `input + cache_read + cache_creation` | `_meta.inputTokens` |
| `completion_tokens` | `last.output_tokens` | `usage.output_tokens` | `_meta.outputTokens` |
| `cache_hit_tokens` | `last.cached_input_tokens` | `cache_read_input_tokens` | `_meta.cachedReadTokens` |
| `cache_miss_tokens` | `input − cached` | `input + cache_creation` | `inputTokens − cachedReadTokens` |
| `total_tokens` | `last.total_tokens` | `prompt + output` if absent | `_meta.totalTokens` |

### 1.2 Gauge (separate pair)

| our field | codex | claude-code-sdk | grok |
|---|---|---|---|
| `context_used` | `last.input_tokens` | same Anthropic SUM as `prompt` | NOT recorded today; if ever added → `usage_update.used`, **never** `prompt` |
| `context_size` | UNRESOLVED #1 | UNRESOLVED #2 (sdk usage carries no window) | UNRESOLVED #3 |

Note on codex: `last.total_tokens = in + out` (`113457 + 83 = 113540`) is
**spend**, marginally above occupancy. It is `total_tokens`, not
`context_used`. `model_context_window` is **not** a token field.

### 1.3 Verbatim payloads

**codex** — `~/.codex/sessions/2026/08/22/rollout-…-01a02a1c-….jsonl`,
`event_msg.payload.type == "token_count"`, ordinal 19 (2nd turn), verbatim:

```json
{"type": "token_count",
 "info": {
  "total_token_usage": {"input_tokens": 35260, "cached_input_tokens": 16128,
                        "cache_write_input_tokens": 0, "output_tokens": 208,
                        "reasoning_output_tokens": 0, "total_tokens": 35468},
  "last_token_usage":  {"input_tokens": 18662, "cached_input_tokens": 16128,
                        "cache_write_input_tokens": 0, "output_tokens": 126,
                        "reasoning_output_tokens": 0, "total_tokens": 18788},
  "model_context_window": 258400}}
```

Mapping reads the **`last_token_usage`** sub-object: `prompt=18662`,
`completion=126`, `hit=16128`, `miss=18662−16128=2534`, `total=18788`.
`total_token_usage` (`35260`) is what we record today — the bug.

**claude-code-sdk** — Anthropic `usage` on an `AssistantMessage`.
Live-verified shape on session `1b982327`: `input_tokens=2` on **every**
assistant message, with the bulk in `cache_read_input_tokens`. So
`prompt = 2 + 111924 + 10441 = 122367`, `miss = 2 + 10441 = 10443`,
`hit = 111924`. This matched `kind=tm` rows 12/12 to the token
(122367, 124617, …, 159491). Those are `num_turns≈1` turns where
`Result == Assistant`, so — per the reviewer — they do **not** prove Result
is safe; they only validate the *formula*.

**grok** — `session/prompt` RESULT `_meta`, mapped through the codex-shaped
path (`_from_codex`). Sample n=60: prompt min/med/max
`17716 / 112352 / 234157`, all under a ~256k window, `hit ≤ prompt` in 60/60.
Per-turn already. **No change.** Included in the table for completeness only.

---

## 2. Code change points

### 2.1 (a) codex — where `last_token_usage` comes from

Current chain:

- `deep_worker/providers/codex_provider/exec_runner.py:167-168`
  ```python
  if t == "turn.completed":
      return TurnCompletedEvent(usage=obj.get("usage", {}) or {})
  ```
- `deep_worker/providers/codex_provider/chat_model.py:852-861` — on
  `TurnCompletedEvent`, calls `_emit_usage_event(usage=event.usage, …)`.
- `deep_worker_cli/stats/token_stats.py:296-299` → `_from_codex` →
  `prompt_tokens = usage["input_tokens"]`.

**⚠ FINDING THAT CONTRADICTS A PREMISE OF THE VERDICT.** The verdict states
that what `turn.completed` carries "is `total_token_usage`, not last", and
that obtaining `last` is an open transport problem. I ran a live probe on
`codex-cli 0.149.0` and that is **not** what the exec stream emits.

Single-turn probe, `codex exec --json --skip-git-repo-check`, verbatim
`turn.completed`:

```json
{"type": "turn.completed",
 "usage": {"input_tokens": 22439, "cached_input_tokens": 11008,
           "cache_write_input_tokens": 0, "output_tokens": 5,
           "reasoning_output_tokens": 0}}
```

Two-turn probe on one thread (`codex exec` then `codex exec resume <thread>`),
thread `01a02b15-9bd5-79d1-872f-af8375d20641`:

```
TURN 1 turn.completed.usage: {"input_tokens": 21824, "cached_input_tokens": 11008, "output_tokens": 5, ...}
TURN 2 turn.completed.usage: {"input_tokens": 21839, "cached_input_tokens": 21248, "output_tokens": 5, ...}
```

If this were `total_token_usage`, turn 2 would be ≈ `21824 + 21839`. It is
not — it is flat. The matching session rollout for the same thread confirms
`LAST == TOTAL` on both turns (each `codex exec` invocation is a fresh
accounting scope):

```
LAST in=21824 cached=11008 out=5 | TOTAL in=21824 cached=11008 out=5 | win=258400
LAST in=21839 cached=21248 out=5 | TOTAL in=21839 cached=21248 out=5 | win=258400
```

Consequences:

1. `turn.completed.usage` is **already per-turn** and already in exactly the
   `_from_codex` shape (`input_tokens` / `cached_input_tokens` /
   `output_tokens`). No transport change is needed to obtain per-turn data.
2. It carries **no `total_tokens` key**. `_from_codex` already handles this
   (`token_stats.py:166-168`: `total = inp + out` when absent) — so
   `total_tokens` is *derived*, not read. That is consistent with the
   table's "`prompt + output` if no total" fallback and needs no code.
3. ~~**The inflated `3736695` row therefore did NOT come from
   `turn.completed`.**~~ **WITHDRAWN — see §2.1-bis.** This inference was
   wrong: it generalized from a spawn mode that is not the one prod uses.
   The row DID come from `turn.completed`.

**This is the one thing I need the reviewer to arbitrate before I touch
codex.** My probe used a fresh `codex exec` per turn; the long-lived
provider may drive the CLI differently (persistent thread / `resume`), and
in a persistent thread `TOTAL` legitimately diverges from `LAST` — which
would put the inflation back inside `turn.completed` under conditions my
probe did not reproduce. Options:

- **A. Locate the real writer first (proposed).** Add no mapping code until
  I can point at the exact line that produced `prompt=3736695`. Cheapest and
  it is the only option that cannot fix the wrong thing. Risk: one more
  investigation round before any code.
- **B. Defensive read at the mapping boundary.** If a payload carries
  `last_token_usage`/`total_token_usage` sub-objects, prefer `last`;
  otherwise use the flat dict as-is. Handles both shapes. Risk: adds a
  branch for a shape we have not proven reaches this function — speculative
  generality, and it would mask A rather than answer it.
- **C. Change transport to consume `token_count`.** Rejected on present
  evidence: `token_count` is **not** on the exec JSONL (probe types seen:
  `thread.started`, `turn.started`, `item.completed`, `turn.completed`), it
  exists only in the session rollout, and an unknown type would raise
  `CodexProtocolError` (`exec_runner.py:189+`). Reading the rollout file
  from the provider would add a filesystem dependency to a stream parser.

I recommend **A** and have not written code for any of them.

---

### 2.1-bis RESOLVED BY EVIDENCE — option A executed, no arbitration needed

Option A was an *investigation*, not a code change, so it required no
permission. Executed. **The question is now closed by proof: the inflated
row DID come from `turn.completed`, and my §2.1 inference was wrong.**

**Step 1 — the record itself (verbatim, an agent's `token_stats.jsonl`):**

```json
{"cache_hit_tokens": 3600640, "cache_miss_tokens": 136055,
 "completion_tokens": 5212, "context_size": null, "context_used": null,
 "event_key": "codex|01a02a2e-968d-7103-9d31-c36afe005682|item_40",
 "kind": "agent", "model": "gpt-5.6-sol", "prompt_tokens": 3736695,
 "provider": "codex", "provider_session_id": "01a02a2e-968d-7103-9d31-c36afe005682",
 "provider_turn_id": "item_40", "record_id": "4deb63c4105744f38ff621a5cd6d5475",
 "schema_version": 1, "session_cumulative": null, "status": "ok",
 "total_tokens": 3741907, "ts": "2026-08-22T18:38:33.222Z",
 "usage_semantics": "tokens"}
```

`provider="codex"` (not ACP, not another provider). `provider_turn_id="item_40"`
is set **only** at `chat_model.py:860` (`turn_id=last_agent_message_id`) on the
`TurnCompletedEvent` branch.

**Writer enumeration — codex has exactly ONE usage writer.**
`grep -n "_emit_usage_event" codex_provider/chat_model.py` → `344` (the
definition), `855` (`TurnCompletedEvent`, `usage=event.usage`), `866`
(`TurnFailedEvent`, `usage=None`). Line 866 cannot produce tokens. Custom
providers receive **only** `set_usage_event_callback` and no LangChain
callback handler (`config.py:2101-2102`, comment: *"a double source is
impossible structurally"*). No ACP/`_meta` writer exists on this path.
**Therefore `chat_model.py:855` is the line that wrote `3736695`.**
(That identifies the **writer**, not the repair site: per §2.1-ter
*PLACEMENT*, `:855` keeps passing `event.usage` and the override lands in
`_emit_usage_event`, which both terminal call sites funnel through.)

**Step 2 — the arithmetic test, and its honest limits.**
`3736695 + 5212 = 3741907` — **exactly** the record's `total_tokens`. So
`total` was *derived* by `_from_codex` (`token_stats.py:166-168`), not
received. But this does **not** discriminate between sources: `turn.completed`
never carries `total_tokens` in *either* mode, so the total is always derived.
The test rules out "a ready-made `total_token_usage` blob was passed through
verbatim" — it does not identify which `input_tokens` was read. What
discriminates is the **2/2 field match against the rollout**, below.

**Step 3 — how prod actually spawns, and the reproduction.**
`ExecArgs.session_uuid` (`exec_runner.py:233`) → `codex exec resume <uuid>`.
But the accumulation does **not** come from `resume`: 5 turns over one thread
via `resume` gave `21824 / 21840 / 21857 / 21874 / 21891` — flat, and the
rollout showed `LAST == TOTAL` on all 5 (each process = a fresh accounting
scope).

The real mechanism is `chat_model.py:604-614`:

```python
is_mid_turn = (
    (tool_resolved or saw_native_resume)
    and state.current_exec is not None
    and not user_text_parts
)
if is_mid_turn:
    handle = state.current_exec
    return await self._consume_turn(state, agent_thread_id, handle)
```

**One `codex exec` process spans MANY LangGraph turns** whenever tool calls
occur — the handle is reused, so the CLI's `total_token_usage` keeps
accumulating across every tool round-trip, and `turn.completed` at the end of
that long-lived exec reports the accumulated figure. My probes never issued a
tool call, so each process saw exactly one model call and `LAST == TOTAL` —
which is precisely why they looked "already per-turn".

**Live proof from the very session in question** (rollout
`…-01a02a2e-968d-7103-9d31-c36afe005682.jsonl`, 101 `token_count` lines):

```
15:54:56.914 LAST in=16407   out=137  | TOTAL in=16407    out=137   | win=258400
15:54:59.980 LAST in=19713   out=83   | TOTAL in=36120    out=220   | win=258400
15:55:10.053 LAST in=23284   out=139  | TOTAL in=59404    out=359   | win=258400
15:55:48.038 LAST in=60495   out=81   | TOTAL in=500089   out=1511  | win=258400
15:56:47.803 LAST in=82234   out=298  | TOTAL in=1335246  out=3381  | win=258400
15:58:34.892 LAST in=91213   out=46   | TOTAL in=2399172  out=5638  | win=258400
16:03:02.593 LAST in=92375   out=133  | TOTAL in=92375    out=133   | win=258400   <-- fresh spawn, TOTAL reset
17:27:25.843 LAST in=95736   out=105  | TOTAL in=1140024  out=1338  | win=258400
18:34:04.227 LAST in=102041  out=301  | TOTAL in=1088520  out=2213  | win=258400
18:37:04.256 LAST in=104239  out=259  | TOTAL in=2327173  out=3547  | win=258400
18:38:30.110 LAST in=112887  out=185  | TOTAL in=3623238  out=5129  | win=258400
18:38:33.188 LAST in=113457  out=83   | TOTAL in=3736695  out=5212  | win=258400  <-- our row
```

Our record at `18:38:33.222Z` (34 ms later) is `prompt=3736695 out=5212` —
the **TOTAL** blob, 2/2 fields, while LAST was `113457 / 83`. Inflation
**32.9×**.

Corroboration from the same file: our only other codex row for this session,
`item_0` at `16:03:02.629`, is `prompt=92375 out=133` — and the rollout line
34 ms earlier is the **fresh-spawn** point where `TOTAL == LAST == 92375`.
That row looked correct only because the two quantities coincided at a
process boundary. Same code path, same bug, invisible.

**Conclusion.** `turn.completed.usage` carries `total_token_usage` semantics
whenever the exec process spans more than one model call — i.e. in every
tool-using turn, which is the normal case in prod. The reviewer's original
premise was **correct**; my §2.1 "already per-turn" claim was an artifact of
tool-free probes. I withdraw it.

This makes the codex work item real again, and option **C** is back on the
table since the per-turn number is not on the exec stream at all. Restating
the options with the new evidence:

- **C′. Emit `last_token_usage` onto the exec JSONL.** The correct fix, but
  `token_count` is genuinely absent from the exec stream (probe types:
  `thread.started`, `turn.started`, `item.completed`, `turn.completed`) — this
  needs an upstream CLI change or a flag we do not control. Not actionable by us alone.
- **D. Delta-track across the exec process.** Keep the previous
  `turn.completed.usage` per live handle and record the difference. Correct
  per-turn spend from data we already receive, no transport change. Risks:
  the first `turn.completed` of a reused handle has no predecessor
  (equals LAST legitimately); `cached_input_tokens` deltas can be
  non-monotonic; and a handle reset must clear the baseline or the next
  delta goes negative.
- **E. Read the rollout file for `last_token_usage`.** Accurate, but adds a
  filesystem dependency and a path/parse coupling to a stream parser. I
  consider this worse than D.

I have **not** written code for any of them and am not choosing between
D/C′/E unilaterally — that is a design call I want the reviewer on, since D
changes what `prompt_tokens` *means* for a multi-call turn.

---

### 2.1-ter DECIDED — D rejected, C′ unavailable, E adopted with a correction

Reviewer's ruling, plus verification I ran on each branch
before writing it down.

**D — REJECTED. Ruling confirmed against the code.** `_consume_turn` emits
usage **only** on `TurnCompletedEvent` (`:852-863`). Tool rounds `break`
earlier (`:821`, `:840`) and keep `current_exec` alive. So one exec lifetime
→ exactly one `turn.completed` → one record, and that payload is the process
TOTAL. There is **no predecessor on the handle to delta against**; D
degenerates to writing `3736695` — the bug itself. And even with
intermediate terminals, `delta(total)` = the sum of the inner `last.input`
values packed onto one row, which is the ResultMessage defect in Codex
clothing. The table specifies last-call occupancy
(`last_token_usage.input_tokens`), **not** "spend of the whole exec". D is
out; `prompt_tokens` does not change meaning.

**C′ — UNAVAILABLE, and I did not invent a flag.** I enumerated the real
surface of `codex-cli 0.149.0` rather than assuming. `codex exec --help`
exposes no token/usage flag; the only output-shaping options are `--json`,
`--experimental-json`, `--output-schema`, `-o/--output-last-message`.
Critically I re-probed under the **exact prod argv**
(`exec --experimental-json --dangerously-bypass-approvals-and-sandbox
-s workspace-write --config approval_policy="never" --ignore-user-config
--skip-git-repo-check`) — event types emitted were
`thread.started`, `turn.started`, `item.completed`, `turn.completed`, and
`grep -c token_count` on that stream returned **0**. C′ needs an upstream
CLI change. **The sdk half (§2.2) is not blocked on it.**

**E — ADOPTED, but the reviewer's formulation of it is unsafe as written and
I am correcting it rather than implementing it verbatim.**

The instruction was to read "the latest `token_count.info.last_token_usage`
for this `codex_session_uuid` at emit time". **Reading the *latest* line is
wrong**, and it fails on the very row this task exists to fix:

```
rollout LAST token_count ts=2026-08-22T18:39:45.741Z last_in=114460
our record             ts=2026-08-22T18:38:33.222Z prompt=3736695
correct per-turn value                              = 113457
```

The rollout is **append-only for the whole session across many exec
processes**, so at any later emit the "latest" line belongs to a *subsequent*
turn. Four `token_count` lines land at/after our record's timestamp; taking
the last one yields `114460`, not `113457`. Silently wrong by a small margin
— far more dangerous than the 32.9× error, because it looks plausible.

**Correction — correlate on the `task_complete` boundary, not on recency.**
The rollout carries `event_msg.task_complete`, one per exec lifetime (5
`task_started` / 3 `task_complete` in this file, matching the spawn count).
The `token_count` immediately **preceding** each `task_complete` is exactly
the turn's last call. Verified across every terminal in the file:

```
task_complete  16:03:02.604 | preceding token_count 16:03:02.593 last=92375  total=92375
task_complete  17:27:28.424 | preceding token_count 17:27:28.413 last=95864  total=1235888
task_complete  18:38:33.192 | preceding token_count 18:38:33.188 last=113457 total=3736695
```

The third line is our row: `last=113457` — the correct value, 11 ms before
`task_complete`, which is itself 30 ms before our record. So the datum
**does** exist and is deterministically addressable; it just is not "the
last line of the file".

**E as it must actually be specified:** at emit time, read the rollout for
`state.codex_session_uuid`, scan for the **last `task_complete` at or before
the emit instant**, and take the `token_count.info.last_token_usage`
immediately preceding it. Fail-closed per the reviewer's floor.

**PLACEMENT — decided (reviewer must-fix #2b, 2026-08-22).** An earlier
revision of this section left the lookup at `:855` (the
`TurnCompletedEvent` branch of `_consume_turn`) while the RED tests drove
`_emit_usage_event` directly. That contradiction had to be resolved, or a
correct implementation would be blocked by a permanently-red test and a
wrong one could pass. **Option (b) is adopted: the lookup lives inside
codex `_emit_usage_event`.** Rationale — that method already receives
`session_id` and `turn_id` (`chat_model.py:344-352`), so (b) needs no new
plumbing, and it puts the fail-closed rule at the single point where every
terminal record is written (both the `TurnCompletedEvent` and
`TurnFailedEvent` call sites, `:855` and `:866`, funnel through it).

Consequences, stated precisely so code and test agree:
- `:855` **keeps** passing `usage=event.usage`; it is not the fix site.
  The line stays as-is.
- codex `_emit_usage_event` **overrides** whatever `usage` it was handed
  with `last_token_usage_from_rollout(session_id, <emit instant>)`, and
  records `no_usage` when that returns `None` — it must **never** persist
  the TOTAL blob it received.
- Pinned by `TestCodexEmitSiteUsesLookup` (behaviour) and
  `TestProductionWiring::test_codex_terminal_emit_consults_the_rollout_helper`
  (AST: the helper is actually called from that method).

Costs, stated as costs (not virtues), per the reviewer's instruction:
- Adds a **filesystem dependency and path coupling** (`CODEX_HOME/sessions/
  <yyyy>/<mm>/<dd>/rollout-*-<uuid>.jsonl`) to what is otherwise a pure
  stream parser. The date-sharded path must be globbed by uuid, not
  constructed from the current date (a turn can cross midnight).
- Adds a **parse dependency on an internal log format** that is not a
  contract — `token_count`/`task_complete` shapes can change without notice.
- **`--ephemeral` would break it entirely** (no session file written). We do
  not pass it today (verified in `_build_argv`), but E silently depends on
  that and must fail-closed, not guess, if the file is absent.
- Requires a **bounded read** (the file reaches 101+ `token_count` lines and
  grows across a long session); reading it on every emit is O(file).

**MANDATORY FLOOR (reviewer, adopted verbatim):** never persist a Codex
`prompt_tokens` that is the TOTAL blob. If `last_token_usage` cannot be
obtained — file missing, unreadable, stale, no `task_complete` at/before the
emit instant — record **`no_usage`**. Do **not** fall back to
`turn.completed.usage`. The magnitude/window check is a safety rail, not the
mapping.

**B** stays a defensive parser branch only (prefer `last_*` sub-objects when
present); it does **not** address today's flat-TOTAL exec payload and is not
the fix.

**Open question I am NOT deciding:** E makes `prompt_tokens` depend on a file
the provider does not own. If the owner would rather accept `no_usage` for
codex until C′ exists upstream than take that coupling, that is a legitimate
call and it is theirs, not mine. Flagged, not resolved.

### 2.2 (b) claude-code-sdk — Result → last Assistant

- Source today: `chat_model.py:1535` — `await self._emit_usage_event(event)`
  where `event` is the `ResultMessage`; `_emit_usage_event`
  (`chat_model.py:276-299`) reads `getattr(event, "usage", None)`.
- The correct value **is already captured**: `chat_model.py:1422-1424`
  ```python
  _usage = getattr(event, "usage", None)
  if _usage is not None:
      last_usage = _usage
  ```
  on every `ClaudeAssistantMessage`. It is currently used only by the
  API-error detector and is ignored for stats. That is the bug.
- `chat_model.py:1528-1532` already treats `ResultMessage.usage` as a
  **fallback** for `last_usage` (`if _u is not None and last_usage is None`).
  The stats path simply does not honour that precedence.

Change: emit stats from `last_usage` (the last `AssistantMessage.usage`,
with `ResultMessage.usage` retained only as the existing no-assistant
fallback) instead of unconditionally from `ResultMessage.usage`. Routing
stays `normalize_usage("claude-code-sdk", …)` → `_from_anthropic`
(`token_stats.py:294-295`). **Not** `ResultMessage`. **Not**
`sum(assistants)` — summing across turns onto one turn is the same defect
in a different costume.

Arithmetic in `_from_anthropic` (`token_stats.py:130-152`) is unchanged.

---

## 3. Guard comments (both sites)

To be added verbatim at `_from_codex` and `_from_anthropic` so a future
"unification" cannot merge them:

> `cache_miss` is computed DIFFERENTLY per provider **by requirement**, not
> by accident. Codex/grok: `input_tokens` **includes** the cached prefix, so
> `miss = input − cached`. Anthropic: `input_tokens` is only the **uncached
> remainder** (live-verified `input_tokens=2` with ~122k of cache on session
> `1b982327`), so `miss = input + cache_creation` and
> `prompt = input + cache_read + cache_creation`. Applying the codex formula
> to an Anthropic payload yields a **negative** miss (`2 − 111924`). DO NOT
> unify these two branches.

Plus, at both emit sites: a note that `prompt_tokens` is per-turn spend and
that occupancy/window values must never be written into it.

---

## 4. NON-SCOPE (explicit)

- **ACP (`claude-code`) is OUT OF THIS SLICE** — scope, not capability.
  ACP **does** map per-turn usage: `acp/chat_model.py:1539-1557` reads
  `session/prompt` RESULT `_meta` (`inputTokens`/`outputTokens`/
  `cachedReadTokens`/`totalTokens`) into the codex-shaped dict
  (`input_tokens`/`output_tokens`/`cached_input_tokens`/`total_tokens`) and
  emits it at `:1576`, together with the `usage_update` gauge
  (`context_used`/`context_size`) as ONE hybrid record. So records are
  `usage_semantics=tokens` whenever `_meta` carries token fields, and
  `context_snapshot` only when it does not.
  My earlier "claude-code formula verified" evidence was
  `claude-code-sdk kind=tm` rows, not ACP — that withdrawal stands.
  Normalisation was repaired separately (2026-08-23): `normalize_usage` had
  no `claude-code` branch, so ACP payloads fell through to
  `_from_langchain_normalized` — tokens survived (`input_tokens`/
  `output_tokens` share names) but `cached_input_tokens` was unknown, so
  BOTH cache fields silently became `None`. Root cause was the `b164482`
  rename: it moved the NAME `claude-code` from the SDK provider to the ACP
  one and correctly retargeted the anthropic branch to `claude-code-sdk`,
  but no branch was added for the new owner of the name. `claude-code` now
  routes to `_from_codex` alongside `codex`/`grok`.
- **No third-provider piggyback.** grok is documented above for
  completeness; its code path is not touched.
- **No jsonl rewrite.** Historical records stay as they are.
- **No new fields in `NormalizedUsage`** (`token_stats.py:109-117`).
- **No change to any arithmetic** in `_from_codex` / `_from_anthropic`.

---

## 5. UNRESOLVED — owner-only

No default is chosen here. Choosing one silently would *be* the decision.

- **#1 codex `context_size`.** `model_context_window` (live `258400`) vs the
  `-l` limit. Tradeoff: the provider value is truthful about the real model
  window and tracks model changes automatically, but it overrides the
  operator's deliberately smaller budget, so the gauge would read "plenty of
  room" while the agent is at its configured ceiling. `-l` respects operator
  intent but goes stale/wrong when the model's real window is smaller.
  Cross-provider comparability differs between the two.
- **#2 claude-code-sdk `context_size`.** sdk usage carries **no window at
  all**. Options: leave `context_size` unset (gauge unavailable, honest), or
  fall back to `-l` (a gauge that is operator-intent, not model truth).
  Consistency with #1 matters more than either choice alone.
- **#3 grok gauge.** Whether to record `context_used`/`context_size`
  (currently 0/60 — `grok_provider` emits tokens-only and drops the
  snapshot); if yes, `1_000_000` vs `-l`. Same tradeoff as #1.

---

## 6. Verification plan

### 6.1 RED tests — must fail on the *source*, not the formula

Design constraint: a test that feeds a hand-made dict into `_from_anthropic`
and asserts the arithmetic would pass **today** — it cannot express this
bug. Each RED test must therefore drive the **emit path** and assert on
which object was read.

1. **sdk — Result vs last Assistant.** Drive the receive loop with a scripted
   stream: `Assistant(usage=A1)`, `Assistant(usage=A2)`, `Result(usage=R)`,
   where `R` is numerically distinct from both and **not** equal to
   `A1 + A2` (so neither "Result" nor "sum" can pass by coincidence).
   Assert the emitted record equals `_from_anthropic(A2)`. Fails today
   because `chat_model.py:1535` passes the `ResultMessage`.
   Anti-tautology: `A2` values chosen so the correct answer is not derivable
   from `R` by any of {identity, max, sum}.
2. **sdk — no-assistant fallback preserved.** Stream `Result(usage=R)` only.
   Assert the record equals `_from_anthropic(R)`. Passes today and must keep
   passing — this pins the existing `last_usage is None` fallback so the fix
   cannot regress it.
3. **codex — UNBLOCKED (E adopted, §2.1-ter).** Writer proven
   (`chat_model.py:855`), mechanism proven (handle reuse,
   `chat_model.py:604-614`), path decided (E + fail-closed floor).
   Fixture: a `turn.completed` carrying an **accumulated** TOTAL payload
   (`input_tokens=3736695`, `cached_input_tokens=3600640`,
   `output_tokens=5212`) plus a rollout fixture whose pre-`task_complete`
   `token_count` holds `last_token_usage.input_tokens=113457`. Assert the
   emitted record is `prompt_tokens=113457`, not `3736695`. Fails today by
   **32.9×**.
   A single-model-call fixture is **useless** here — that is exactly the
   blind spot that made my first probe wrong, and it is why the live
   `item_0` row (`92375`) looked correct while being produced by the same
   defect. The fixture MUST have TOTAL ≠ LAST.
4. **codex fail-closed floor.** Same accumulated `turn.completed`, but the
   rollout is missing / unreadable / has no `task_complete` at or before the
   emit instant. Assert the record is **`no_usage`** and that `3736695`
   is **never** persisted. This pins the floor independently of E's happy
   path — without it, a later "graceful fallback" refactor silently
   reintroduces the 32.9× bug.
5. **codex recency trap (regression pin).** Rollout fixture with a
   `token_count` *after* the turn's `task_complete` (values `114460`,
   mirroring live data). Assert the emitted value is the pre-`task_complete`
   `113457`, **not** the latest line. This pins the §2.1-ter correction so a
   future simplification to "read the last token_count" cannot land — that
   variant is wrong by a *small, plausible-looking* margin and would
   otherwise pass every magnitude/window sanity check.

### 6.2 Live verification (~5 turns per provider)

Per provider, run ≥5 real turns in **one** session and read the emitted
records:

- **Monotonicity check (the actual bug signature):** `prompt_tokens` must
  **not** be non-decreasing-by-construction across turns. Cumulative data
  rises every turn; per-turn data fluctuates with the prompt. Assert at
  least one turn-over-turn **decrease**, or that the spread is inconsistent
  with accumulation.
- **Window sanity:** `prompt_tokens ≤ context window` on every turn. Today
  codex violates this in 87/101 samples; after the fix it must be 0/N.
- **Cross-check against the provider's own ledger:** for codex, each
  emitted `prompt` must equal that turn's `last_token_usage.input_tokens` in
  the session rollout; for sdk, the last `AssistantMessage` SUM in
  `~/.claude/projects`.
- **Non-negativity:** `cache_miss_tokens ≥ 0` on every record for both
  providers — the direct regression signal for a formula unification.

Results to be reported as raw rows, not summaries.

---

## 7. Sequencing

1. Reviewer arbitrates §2.1 (codex: option A/B/C, and whether my probe
   invalidates the codex work item as scoped).
2. Only then: RED tests per §6.1.
3. Then implementation; then live verification per §6.2.

No code will be written before step 1 returns.
