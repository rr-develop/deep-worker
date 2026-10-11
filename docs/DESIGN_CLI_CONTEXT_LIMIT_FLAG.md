# DESIGN: `--context_limit` / `-l` — operator-set effective context window

Owner directive (2026-08-17), verbatim:

> We need functionality for explicitly setting the maximum context limit at
> which summarization is invoked. […] If the provider changes, for example
> because the limits for the given provider were reached, the configured
> maximum context size MUST be preserved. It must likewise be preserved
> across an agent restart, etc., until it is set to a different value or
> explicitly reset to the default. This is done primarily to reduce token
> costs when working with models that have a large allowed context window.

Plus: add `-r` as an alias for `--restart`.

---

## 1. Problem

Models with very large context windows (1M tokens) are billed per input
token on every turn. Summarization currently triggers at a *fraction of the
model's own window* (`("fraction", 0.85)` of `max_input_tokens`), so a 1M
model runs to ~850k tokens before compacting — and every turn until then
re-sends the whole transcript. The operator has no way to say "compact at
200k regardless of what this model could technically hold".

## 2. Scope decisions (settled 2026-08-17)

**2.1 The limit defines the EFFECTIVE WINDOW, not just the trigger.**
Both `trigger` and `keep` are computed from it. Rationale is in the
directive itself: the goal is *lower* spend. If `keep` stayed a fraction of
the model's real window (10% of 1M = 100k) while `trigger` moved to 200k,
then every summarization would leave 100k in place — half the budget — and
the next trigger would arrive almost immediately. Cost would go UP. With
`--context_limit 200k` the agent must behave exactly like a model whose
window is 200k.

**2.2 Subagents inherit the limit.** The owner framed it as a property of
the agent that survives provider switches and restarts. Subagents are part
of that agent's work; leaving them uncapped would deliver partial savings,
i.e. not satisfy the requirement.

## 3. What already exists (verified, not assumed)

* `SummarizationMiddleware.trigger` accepts **a list** of `ContextSize`
  conditions and `_should_summarize` ORs them — first one to fire wins.
  This is exactly the "whichever comes first" semantic the directive asks
  for, with no branching of our own.
* `_compute_summarization_defaults(model)` (`deepagents/middleware/
  summarization.py`) is the single producer of `trigger`/`keep`/
  `truncate_args_settings`, consumed at **three** `SummarizationMiddleware`
  sites in `graph.py`: general-purpose subagent (:267), user subagents
  (:320), main agent (:371). One change point covers all three.
* Failover (`agent_process.py:3645`) re-launches via
  `deep-worker --restart <name> --provider <candidate>`. **The provider-switch
  requirement therefore rides on the restart path** — no separate code. It
  still gets its own test, because the requirement is explicit.
* `test_every_build_launch_config_input_is_handled_by_merge_restart_config`
  already guards persist-vs-restore symmetry. It will fail if the new field
  is persisted but not restored (the exact class of defect that wiped six
  fields in `--restart` on 2026-08-14).
* `config.py:503 get_model_context_limit` / `:551 get_summarization_config`
  are **dead code** — nothing in production calls them. Implementing there
  would produce a feature that looks wired and silently does nothing.
  Implementation goes in the live path.

## 4. Design

### 4.1 CLI surface

```
--context_limit VALUE   (alias: -l)
-r                      (alias for --restart)
```

`VALUE` grammar (strict — rejected values fail at argparse `type=`, so
the error is a clean usage message, not a traceback):

| Input      | Meaning                                    |
|------------|--------------------------------------------|
| `200000`   | 200 000 tokens                             |
| `200k`/`200K` | 200 000 tokens                          |
| `1M`/`1m`  | 1 000 000 tokens                           |
| `default`  | clear the override; use the model's own default |

Rejected: zero, negative, non-numeric, empty, float (`1.5k`), unknown
suffix (`200g`). Rejecting floats keeps the grammar unambiguous and
reversible in logs.

Stored form is always `int | None` (tokens). Suffixes are a CLI
convenience; `session_metadata.json` writes `"context_limit": 400000`
(or `null` after `default`). The grammar may grow (`g`) later without
a storage migration.

Alias safety: `-ls` and `-rm` already exist. argparse resolves an **exact**
option string before prefix matching, so `-l`, `-r`, `-ls`, `-rm` all remain
unambiguous — verified by running the parser, not by reading docs. A guard
test pins this on **both** `create_parser()` and `_explicit_cli_flags()`
(the detector must declare `-ls` next to `-l`, or `-ls` is swallowed as
`-l` + value `s` — the same class of defect as `-p`/`--preferred-provider`
producing 35 phantom failures).

### 4.2 Semantics — the disjunction

Fractions are the ones already hardcoded in
`_compute_summarization_defaults` (`summarization.py`):

* `trigger_fraction = 0.85`
* `keep_fraction = 0.10`

Given operator limit `L` and the model's own default trigger `D`:

* No limit set → today's behaviour, untouched.
* `L` set → pass `trigger=[("tokens", int(L * 0.85)), <model default>]`.
  `_should_summarize` ORs them, so summarization fires at
  `min(L*0.85, D)`. This directly satisfies "if the default for the current
  model is already lower than context_limit, summarization must be invoked
  on reaching the default limit". **We do not compute the min ourselves** —
  comparing a `fraction` against a `tokens` value would require resolving
  the model profile at construction time, which is exactly the fragile
  coupling the list form exists to avoid.
* `keep` becomes `("tokens", int(L * 0.10))` — the effective-window
  decision from §2.1.
* `truncate_args_settings` is recomputed from the same `L` (same
  fractions, token form):
  `{"trigger": ("tokens", int(L * 0.85)), "keep": ("tokens", int(L * 0.10))}`.
  `TruncateArgsSettings.trigger` is a single `ContextSize`, not a list,
  so we cannot OR it with the model default the way `trigger` does.
  Recomputing from `L` is the consistent effective-window policy: after
  `--context_limit 200k` the agent must not keep truncating args against
  the old 1M profile (850k) while summarization already fires at 170k.
  When `L` is *looser* than the model window, args truncation may fire
  later than summarization — that is fine, because summarization has
  already evicted those messages.

`--context_limit 200k` on a 1M model therefore produces the same 0.85 /
0.10 ratios a genuine 200k model would get, for trigger, keep, **and**
truncate_args.

### 4.3 Plumbing (5 hops, each an explicit test)

1. **CLI** → `--context_limit` parsed to `int | None` (`None` = default).
2. **Persist** → `build_launch_config` writes `context_limit`.
3. **Restore** → `merge_restart_config` reads it back
   (`effective.context_limit = saved.get("context_limit")`); an explicit
   CLI value overrides via `_explicit_cli_flags()` (the detector **must**
   list `--context_limit` / `-l`, or the "explicit" bit is lost and
   `default` cannot beat a saved int). `default` must be distinguishable
   from "not passed": passing `--context_limit default` stores `None` and
   that `None` must WIN over the saved value, otherwise the reset is
   impossible (this is the one place where the usual "saved is the
   default" rule inverts). `--restart` / `-r` is a one-shot action, not a
   persisted field — it is **not** added to the detector.
4. **Config** → carried on the agent config object to `create_cli_agent`.
5. **Graph** → `create_deep_agent(..., context_limit=...)` →
   `_compute_summarization_defaults(model, context_limit=...)` → all three
   middleware sites.

### 4.4 Why not a per-provider setting

The limit is a *cost policy of the operator*, not a property of a provider.
Storing it per provider would resurrect the bug the directive explicitly
forbids: switching provider on a usage cap would silently drop it.

## 5. Test plan (RED first)

| # | Test | Pins |
|---|------|------|
| 1 | `200k`/`1M`/`200000`/`default` parse correctly | grammar |
| 2 | `0`, `-5`, `1.5k`, `200g`, `abc`, `""` rejected | fail-loud |
| 3 | limit above model default → default still fires | disjunction |
| 4 | limit below model default → limit fires | disjunction |
| 5 | `keep` derives from the limit, not the model window | §2.1 |
| 6 | all three middleware sites get it (incl. subagents) | §2.2 |
| 7 | `merge_restart_config(saved={context_limit:400000}, args(no-flag), explicit={"context_limit"})` → None; `explicit=set()` → 400000. Pins the mechanism (`effective.context_limit = saved.get("context_limit")`), not just the restart effect. Generic guard `test_every_build_launch_config_input_is_handled_by_merge_restart_config` still required. | directive + mechanism |
| 8 | survives failover provider switch (`--restart NAME --provider X` does not pass `--context_limit`, saved value is restored) | directive |
| 9 | `--context_limit default` / `-l default` clears a saved value (`_explicit_cli_flags` detects both forms) | reset |
| 10 | Primary parser AND `_explicit_cli_flags` detector: `-l 200k` → 200000 / dest present; `-ls x` is a different dest or an error; `-r` → restart; `-rm` unaffected | alias safety |
| 11 | field passes the persist/restore symmetry guard | regression |

Every implementation edit must break at least one named test (mutation
testing). A mutation that survives means the test pins nothing.

## 6. Risks

* **Silent no-op.** Mitigated by testing the live path and asserting on the
  actual `trigger`/`keep` values handed to the middleware, not on our own
  helper's return value in isolation.
* **Subagent divergence.** Three construction sites; a change to two of them
  passes a naive test. Test 6 asserts all three.
* **Reset impossible.** `default` → `None` must beat the saved value.
  Test 9 exists because the natural implementation gets this backwards.
