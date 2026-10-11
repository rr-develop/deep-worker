# DESIGN: Named, operator-configurable failover chains

**Status:** DESIGN — awaiting review sign-off
**Date:** 2026-09-03
**Owner directive:** chains must be configurable, addressable by name, and
**never hard-coded**. The default chain is named `default` and matches today's
behaviour. Other agents may use other chains.
**Supersedes:** §5 of `DESIGN_PROVIDER_FAILOVER_AND_PREFERRED_PROVIDER.md`
(chain as a module constant).

## 0. Problem

`runtime/provider_chain.py:13` hard-codes the failover order:

```python
PROVIDER_FAILOVER_CHAIN: list[str] = ["claude-code", "claude-code-sdk", "codex", "grok", "deepseek"]
```

Three consequences the owner has ruled unacceptable:

1. **Not configurable.** Reordering, extending, or giving one agent a
   different order requires editing Python and shipping a commit.
2. **Config-driven providers are excluded.** `failover_candidates()` returns
   `[]` for any name absent from the constant
   (`provider_chain.py:109-112`), so an agent on `openrouter` or `minimax`
   hits a 429, records the limit, and **stays in ERROR forever** — it can
   neither leave nor be arrived at. Nine such providers exist in the live
   `providers.json`.
3. **No model dimension.** A chain step is a provider; the model is whatever
   the provider defaults to. There is no way to say "same provider, cheaper
   model" or "opus first, then sonnet".

## 1. Requirements (from the owner, 2026-09-03)

| # | Requirement | Consequence for this design |
|---|---|---|
| R1 | Chains configured in files, **never** hard-coded | No chain literal in Python. §3 |
| R2 | Chains have **names**, referenceable from commands and agent config | `--failover-chain`, `launch_config.failover_chain`. §5, §6 |
| R3 | Default chain named `default`, same composition as today | Ships as config, not as a constant. §3.3 |
| R4 | Per-agent chain selection | Name resolved per agent at failover time. §6 |
| R5 | A step names a provider; model = provider default or `preferred_model` | §4.1 |
| R6 | A step **may** name a model explicitly; explicit wins | §4.1 |
| R7 | A model may be a **family** (e.g. `opus`) → current model of that family | §4.2 |
| R8 | **No artificial restrictions** on which providers may appear | §4.3 |
| R9 | Only validate that a named provider is really configured | §4.3, §7 |
| R10 | `claude-code` / `claude-code-sdk` are two forms of one provider — no point failing between them | §4.4 |

## 2. Design principles

- **Config is the source of truth.** Python contains no chain, no order, no
  provider list. The absence of config is an error to report, not a default
  to invent — except for `default` itself, which is *materialised as a config
  file* on first use (§3.3), never as a literal consulted at runtime.
- **Fail loud on bad config, degrade only where a limit is being handled.**
  A malformed chains file at agent start is a startup error (`sys.exit(1)`,
  matching `providers.json` handling at `config.py:1511-1530`). A malformed
  file *at failover time* must not crash a running agent — it logs and leaves
  the agent in ERROR, which is the honest outcome.
- **No model names in Python** (`feedback_no_model_hardcode_2026_08_15`).
  Family resolution asks the provider/backend, never a whitelist. §4.2.

## 3. Configuration file

### 3.1 Location and shape

`failover_chains.json` in the Deep Worker home directory — sibling of `providers.json`, same
loader conventions (`storage_paths.deep_worker_home()`, so `--data-dir` and
tests redirect it together).

```json
{
  "version": 1,
  "chains": {
    "default": {
      "description": "Quality-first order across all subscription accounts.",
      "steps": [
        {"provider": "claude-code"},
        {"provider": "codex"},
        {"provider": "grok"},
        {"provider": "deepseek"}
      ]
    },
    "cheap": {
      "description": "Free OpenRouter tiers before anything metered.",
      "steps": [
        {"provider": "or-glm-52"},
        {"provider": "or-nemotron-super"},
        {"provider": "deepseek"}
      ]
    },
    "claude-tiers": {
      "steps": [
        {"provider": "claude-code", "model": "opus"},
        {"provider": "claude-code", "model": "sonnet"},
        {"provider": "codex"}
      ]
    }
  }
}
```

### 3.2 Step schema

| Field | Required | Meaning |
|---|---|---|
| `provider` | yes | Provider name. Built-in or from `providers.json`. |
| `model` | no | Exact model id **or** family name. Overrides the provider default and `preferred_model` for this step. |

A bare string step (`"codex"`) is accepted as sugar for
`{"provider": "codex"}` — chains are hand-edited, and the common case has no
model.

### 3.3 The `default` chain and first run

R3 says `default` must exist and match today's order. R1 forbids a literal.
Resolution: on first load, if `failover_chains.json` is absent, the loader
**writes** it with the `default` chain seeded from the *migration source*
(§8) and a header comment pointing at this document, then loads it normally.

The seed value lives in the migration module, is used exactly once to create
the file, and is never consulted afterwards — the runtime path always reads
config. After that write the operator owns the file, including the right to
change `default` itself.

If the file exists but has no `default` chain, that is a config error, not a
silent fallback: an agent that asked for a chain by name must not be given a
different one.

`claude-code-sdk` is absent from the seeded `default` per R10 — see §4.4.

## 4. Semantics

### 4.1 Model precedence per step

For the step the agent is failing over **to**:

1. Step's `model`, if present (R6) — after family resolution (§4.2).
2. Agent's `preferred_model`, if it applies to that provider
   (today: `agent_process.py:1054` passes it only when restoring to
   `preferred_provider`).
3. Provider default — `ProviderSpec.resolve_model()` for config-driven
   (`provider_registry.py:127`), `get_default_model_name()` for built-ins
   (`config.py:1490`).
4. `None` — provider's own CLI decides (correct for `claude-code`, `codex`,
   `grok`, whose defaults are already `None` at `config.py:1457-1487`).

### 4.2 Model families (R7)

A step's `model` may be a family (`opus`) rather than a full id
(`claude-opus-5`), and it is passed to the provider **verbatim**. There is no
family table in Python and no resolution step at failover time.

This is not a shortcut — it is where family aliases are actually resolved:

1. `claude-code` / `claude-code-sdk` accept `--model opus` and resolve it
   against their own picker. The picker's answer *is* the current model of
   that family, by definition and at the moment of launch.
2. Every other provider receives the string unchanged and rejects a bad id
   with a real vendor error — the documented policy for model names
   (`feedback_no_model_hardcode_2026_08_15` rules 2 and 4).

**Rejected alternative (revising review R2).** The first draft proposed an
optional `resolve_model_family()` on the provider profile, and review asked
only that its absence not raise. Implementation showed the premise was wrong:
no profile registry keyed by provider name exists, and the one mechanism that
*can* turn `opus` into a dated id — `context_limit/provider_max.py` — resolves
it by **spawning a transient CLI subprocess** (`provider_max.py:226`).

Paying that cost here would be wrong twice over: failover runs while an agent
is already failing, and spawning a second process to learn a name we can
simply forward is precisely the pattern the owner forbade on 2026-09-01. The
context-limit path has its own reason to spawn (it needs a number the CLI will
not otherwise tell us); choosing a launch model does not.

Consequence to state plainly: a chain step naming `{"provider": "codex",
"model": "opus"}` sends `opus` to Codex, which will reject it. That is correct
— the operator named a model the provider does not have, and the vendor's
error says so. We do not validate model names locally (rule 2).

### 4.3 Validation (R8, R9)

The **only** membership check is: does the named provider exist and is it
configured? Concretely, for each step at load time:

- name is a built-in (`SUPPORTED_PROVIDERS`) **or** present in
  `providers.json`; otherwise `FailoverChainConfigError`;
- for a config-driven provider, `has_credentials()` is checked at
  *failover time*, not load time — a missing env var should not prevent the
  CLI from starting, but must stop us restarting an agent onto a provider
  that cannot authenticate ("don't send it to grandfather's village", i.e. to an address that cannot answer).

Explicitly **not** enforced: no account-group filtering of chain
composition, no "terminal provider" rule, no ban on config-driven providers,
no ordering constraints, no dedup requirement. The operator's chain is walked
as written.

### 4.4 Same-provider steps (R10)

`claude-code` and `claude-code-sdk` are two forms of one provider on one
subscription. Failing from one to the other is pointless, so the seeded
`default` lists only `claude-code`.

This is **not** re-imposed as a rule: an operator may list both if they have
a reason (R8). What remains is the *availability tracker's* account identity
(`account_group_for`, `provider_chain.py:61`) — it already records limits per
account, so a chain that walks into a same-account step will simply find it
unavailable and move on. Behaviour, not prohibition.

### 4.5 Walking the chain

`failover_candidates(after)` becomes chain-aware and keeps its contract:
steps strictly after the agent's current position, in order. Position is
matched on `provider`; for a chain with repeated providers and different
models (`claude-tiers`), the match uses the current `(provider, model)` pair
when the agent's model is known, falling back to first occurrence of the
provider. The account-group skip stays (§4.4) — it is availability logic, not
composition policy.

If the agent's current provider is absent from its chain, the walk starts at
the chain's **first** step rather than returning `[]`. This is the fix for
the `openrouter` dead-end: today an off-chain provider can never fail over.

## 5. CLI surface (R2)

| Flag | Meaning |
|---|---|
| `--failover-chain NAME` | Chain this agent uses. Persisted in `launch_config`. Default: `default`. |
| `--list-failover-chains` | Read-only: print configured chains, their steps, and per-step validity. |

`--failover-chain` follows the `-pp` precedent: it is preserved across
`--restart` via `launch_config`, and appears in `_explicit_cli_flags()`
(`agent_command.py:506-595`) so "not passed" is distinguishable from
"passed as default".

## 6. Agent config (R2, R4)

`launch_config` gains `failover_chain: str` (default `"default"`),
`storage_version` 5 → 6. Migration: absent key reads as `"default"`, which
preserves today's behaviour exactly for every existing agent.

Read at failover: `agent_process._schedule_failover` resolves the agent's
chain name → chain → candidates. Read at restore: `_maybe_restore_preferred_provider`
is unchanged in policy (it targets `preferred_provider`, not a chain step).

## 7. Modules

| Module | Change |
|---|---|
| `runtime/failover_chains.py` | **New.** `FailoverStep`, `FailoverChain`, `load_chains()`, `get_chain(name)`, `FailoverChainConfigError`, first-run materialisation. |
| `runtime/provider_chain.py` | `PROVIDER_FAILOVER_CHAIN` **deleted**. `failover_candidates()` gains a chain argument. `account_group_for` / `PROVIDER_ACCOUNT_GROUPS` / `SUPPORTED_PROVIDERS` stay — they are identity and menu, not order. |
| `process/agent_process.py` | `_schedule_failover` (:4296) resolves the chain by name; per-step model flows into the `--restart` command. **Both** `deepseek` hard-codings are deleted — see §7.1. |
| `cli/agent_command.py` | Two flags, `build_launch_config`, `_explicit_cli_flags`, `merge_restart_config`. |
| `runtime/availability_tracker.py` | `_PROVIDER_FALLBACK_HOURS` (:50) — see §9, open question. |

### 7.1 The two `deepseek` hard-codings (review R4)

Terminality is hard-coded in **two** places, not one. Review caught the second;
the first draft of this document named only the first.

| Site | Today | Decision |
|---|---|---|
| `agent_process.py:4331` | `if provider == "deepseek": return` — no failover *off* deepseek | **Delete.** Terminal = last step of the chain. |
| `agent_process.py:1037` | `if preferred_provider == "deepseek": return` — no restore *to* deepseek | **Delete**, plus the docstring claim at `:1018`. |

Rationale for deleting the second: it encodes the same "deepseek is special"
assumption. Keeping it produces asymmetric behaviour the moment chains are
operator-defined — a chain like `["or-glm-52", "deepseek"]` would let an agent
fail over *into* deepseek but never restore *back* to it when deepseek is the
agent's `preferred_provider`, stranding it on a provider it was never meant to
run on.

Nothing is lost by removing it: restore is already gated by
`is_restore_eligible` (`availability_tracker.py:290`), which returns `False`
when the reset window is unknown. An agent whose `preferred_provider` is
deepseek and which never left it also never reaches this code — the
`preferred_provider == current_provider` guard at `:1035` returns first.

## 8. Migration

One-shot, in `runtime/failover_chains.py`: seed `default` as
`claude-code → codex → grok → deepseek`. This is today's constant minus
`claude-code-sdk` (R10) — a deliberate, owner-directed behaviour change,
called out here so review sees it rather than discovering it.

Existing agents get `failover_chain="default"` on next launch-config write;
no agent changes behaviour except that a `claude-code` agent now fails over
to `codex` instead of `claude-code-sdk` — which today is a wasted step into
the same exhausted subscription.

## 9. Open question for review

`_PROVIDER_FALLBACK_HOURS` (`availability_tracker.py:50`) hard-codes cooldowns
per provider name, including `"minimax": 3` for a config-driven provider. It is
adjacent to this work (same "no hard-coding" directive) but not required by
R1–R10, and the owner did not name it. **Decision (review R6): out of scope
here, recorded as a follow-up.** Flagging rather than silently widening scope.

**FOLLOW-UP (open, not scheduled).** `_PROVIDER_FALLBACK_HOURS`
(`availability_tracker.py:50`) is a hard-coded cooldown table keyed by
provider name — `claude-code`/`claude-code-sdk`/`codex` 6h, `grok` 24h,
`minimax` 3h. Two problems: (a) it is the same class of hard-coding this
document removes from chains; (b) `"minimax": 3` names a *config-driven*
provider, so an operator adding a provider to `providers.json` cannot give it
a cooldown without editing Python. A provider missing from the table gets
`available_after_utc=None`, which `is_available` treats as available
(`:281-283`) but `is_restore_eligible` treats as ineligible (`:301-302`).
Recorded here so it survives this session; needs an owner decision before
anyone acts on it.

## 10. Test plan

- **Config**: valid load; unknown provider rejected; missing `default`
  rejected; bare-string step sugar; first-run materialisation writes a file
  that then loads identically; malformed JSON → typed error.
- **Semantics**: model precedence (step > preferred > provider default >
  None); family passthrough when the provider has no resolver; family
  resolution when it does.
- **Walking**: candidates after a step; off-chain current provider starts at
  step 0; same-account step skipped by availability, not by composition;
  repeated-provider chain (`claude-tiers`) advances by `(provider, model)`.
- **Off-chain + same account (review R3)**: agent's provider is absent from
  the chain **and** the chain's *first* step shares its `account_group`. The
  account-group skip must still fire, or the very first candidate is a
  provider we already know is exhausted. This is the one interaction between
  "start at step 0" (§4.5) and account affinity (§4.4), so it gets its own
  test rather than being assumed.
- **No hard-coding guard**: an AST/grep test asserting no chain literal
  remains in `runtime/` — mutating the config must change behaviour, which is
  the only proof the config is actually read (per
  `feedback_ast_guard_proves_callers_not_helper_2026_08_31`, mutate **both**
  the config and the call site).
- **Integration**: `openrouter` agent hits a simulated 429 and fails over —
  the case that is impossible today.
- **CLI**: flag parsed, persisted, survives `--restart`, listed by
  `--list-failover-chains`.

Bounded per `feedback_integ_tests_bounded_2026_07_28`: no real provider
calls, no spawned agents, ≤60s budget, temp `deep_worker_home`.

### 10.1 Existing tests that break (review R7)

Deleting `PROVIDER_FAILOVER_CHAIN` breaks exactly two tests. Both are
**rewritten**, not deleted or weakened — a vanished test loses the pin it was
holding.

**`tests/acp_provider/test_acp_cli_wiring.py:114`** — the more important one:

```python
assert set(PROVIDER_FAILOVER_CHAIN) <= set(SUPPORTED_PROVIDERS)
```

This invariant **directly contradicts R8**. `SUPPORTED_PROVIDERS`
(`provider_chain.py:37`) holds built-ins only; none of the nine config-driven
names in the live `providers.json` (`minimax`, `openrouter`, `or-glm-52`, …)
appear in it. Any chain containing `openrouter` fails this assertion. So the
old invariant is not a rule we are breaking by accident — it is the very rule
the owner has revoked.

Replacement invariant, which is §4.3 stated as a test: **every chain step
must name a provider in `all_provider_names()`** (built-in ∪ `providers.json`,
`provider_chain.py:123`). Strictly weaker on membership, strictly stronger on
coverage — it now validates operator config, which the old assertion never
did. The risk to guard against during implementation is quietly relaxing this
to something vacuous; the test must fail on a chain naming a provider that
exists nowhere.

**`tests/unit_tests/process/test_failover_issue77.py:235-244`** — pins the
literal chain and `[-1] == "deepseek"`. Rewritten to pin the **seeded
`default`** against the migration source (§8): after first-run
materialisation, the loaded `default` chain equals
`claude-code → codex → grok → deepseek`. Without this rewrite the composition
of `default` would become untested the moment the constant disappears.

Note the second assertion (`[-1] == "deepseek"`) is deliberately **not**
carried over as a general rule — it is a property of the seeded `default`,
not of chains, and §7.1 removes the code that treated it as one.
