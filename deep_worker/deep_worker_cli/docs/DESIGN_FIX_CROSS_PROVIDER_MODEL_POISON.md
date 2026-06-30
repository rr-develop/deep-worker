# Design: Fix cross-provider stale-model poison on restart (a related internal incident)

Status: Proposed
Branch: dev

## Symptom

An agent was restarted onto `codex` (`-pp codex`) but kept booting with
`model = grok-composer-2.5-fast` — a grok model name — both displayed
(`dwcli --status` → `Provider: codex / Model: grok-composer-2.5-fast`) and
*executed* (`create_model(provider="codex", model_name="grok-composer-2.5-fast")`,
so the codex CLI was handed a foreign model string as `ExecArgs.model`).

Manually nulling `launch_config["model"]` in `session_metadata.json` did **not**
stick: after a plain `deep-worker --restart <agent>`, `model` came back as
`grok-composer-2.5-fast`.

## Root cause (two independent layers)

### Layer 1 — config-merge reset is too narrow (partially fixed by 4cd1bd5)

`merge_restart_config` (`cli/agent_command.py`) resets the saved model to `None`
on an explicit `-pp`/`--provider` switch **only when the provider actually
changes**:

```python
if not model_explicit and effective.provider != saved.get("provider"):
    effective.model = None
```

The original `-pp` bug had already persisted `provider = codex` alongside the
stale `model = grok-composer-2.5-fast`. So a subsequent `-pp codex` sees
`effective.provider (codex) == saved.provider (codex)` → the reset never fires →
the foreign model is carried forward. The guard keys off *provider change*, but
the invariant we actually want is **"no explicit `--model` ⇒ the provider's own
default"** (operator's categorical rule).

### Layer 2 — the display-model resolver re-pins a foreign model (the re-injection source)

`AgentProcess.start()` (`process/agent_process.py`) wires a
`set_resolved_model_callback` and, for a codex/grok agent launched with
`config.model is None`, eagerly seeds the display from the provider's own
session log:

```python
if self.config.provider == "codex" and self.config.model is None:
    persisted_model = _resolve_persisted_codex_display_model(self.name)
    if persisted_model:
        await _on_resolved_model(persisted_model)
```

`_on_resolved_model` calls `SessionManager.update_launch_config_model(name)`,
which **writes the resolved value into `launch_config["model"]` and persists it**
(`session/session_manager.py`). That is the second source: even after we null
`launch_config["model"]`, the codex rollout for the resumed session
(`~/.codex/sessions/.../<uuid>.jsonl`) still carries
`turn_context.payload.model = "grok-composer-2.5-fast"` (because the *first*
broken run fed codex that string and codex echoed it into its own rollout). The
resolver reads it back and re-pins `launch_config["model"]` on every restart.

The design `DESIGN_CODEX_RESOLVED_MODEL_DISPLAY.md` deliberately persisted the
resolved model into `launch_config["model"]` for "display continuity". That
**conflates display state with execution config**: `launch_config["model"]` is
also the value fed to `create_model()` → `ExecArgs.model` on the next boot. A
display-only value must never become an execution-pinning value.

## Principle

> If no `--model` was specified, the provider's own default MUST be used.
> `launch_config["model"]` stays `None` for an unpinned agent.

This is the operator's categorical rule. It is also exactly what heals a
cross-provider poison without needing any provider↔model-name table (we have
none; codex and grok both resolve their default as `None` and the CLI picks).

## Fix

### Fix 1 — unconditional model reset on explicit provider declaration

In `merge_restart_config`, when the operator explicitly passes `-pp`/`--provider`
and does **not** pass `--model`, reset `effective.model = None` regardless of
whether the provider string changed. An explicit provider declaration without an
explicit model means "use that provider's default."

- `--provider <P>` (no `--model`)  → `effective.model = None`
- `-pp <P>` (no `--model`)          → `effective.model = None`
- plain `--restart` (no provider/model flags) → **unchanged**: keep the saved
  model (a flat restart must not silently wipe a deliberately-pinned model).
- explicit `--model M` always wins and is preserved.

`preferred_model` MUST track the reset **unconditionally** whenever the Fix 1
gate fires (RC-1 from internal review). Today `preferred_model` only resets on
a true pp-switch (`pp_switched`, agent_command.py:389-393); on same-provider `-pp`
the `elif saved.get("preferred_model")` branch (line 394-395) would carry a stale
`preferred_model` forward and contradict the operator rule (and break failover
restore-on-idle which reads `preferred_model`). So: when
`(pp_explicit or provider_overridden) and not model_explicit`, also set
`effective.preferred_model = effective.model` (i.e. `None`).

Distinguishing signal: the reset is gated on `pp_explicit or provider_overridden`
(an explicit provider declaration this invocation), NOT on `provider != saved`.

Implementation (RC-3, consolidation, non-blocking): the two narrow reset blocks
(model-reset at agent_command.py:294-298 and the pp model-reset at 369-370) are
replaced by a single consolidated block placed AFTER the pp-block so the provider
assignment is finalized, then a single
`if (pp_explicit or provider_overridden) and not model_explicit: effective.model = None`
plus the matching `preferred_model = None`.

### Fix 2 — resolver must not re-pin execution config for an unpinned agent

In `AgentProcess.start()`, the startup display-seed path
(`_resolve_persisted_codex_display_model` / `_resolve_persisted_grok_display_model`
→ `_on_resolved_model`) runs precisely when `self.config.model is None`. For an
agent launched with no pinned model, the resolved value is **display-only** and
MUST NOT be persisted into `launch_config["model"]` (doing so re-pins execution
and defeats Fix 1 / the principle).

Implementation: `_on_resolved_model` gains a `persist: bool` parameter.
- The **startup session-log seed** calls it with `persist=False` → updates the
  live in-memory display (`set_dynamic_display_model`, when present) only.
- The **live runtime callback** (the model reporting its real model id from an
  actual response, i.e. `set_resolved_model_callback` firing during a turn) keeps
  `persist=True` — but only persists when it does not contradict an unpinned
  launch. To keep the change minimal and provider-agnostic, when the agent was
  launched with `config.model is None` we still do **not** write the resolved
  value into `launch_config["model"]`; we only update the live display. This
  guarantees `launch_config["model"]` stays `None` for an unpinned agent — the
  durable execution config is never silently pinned from a session log.

Net effect: an unpinned codex/grok agent always boots at its provider default;
the web/status display may still show the dynamically resolved model via the
in-memory display path, but the persisted `launch_config["model"]` is never
poisoned.

> Note: this narrows `DESIGN_CODEX_RESOLVED_MODEL_DISPLAY.md` §65 ("passing this
> known value as `--model` is acceptable"). That assumption is what allowed a
> cross-provider value to be re-pinned. Display continuity is preserved via the
> in-memory display path; execution config is no longer mutated from a log.

## What this does NOT change (no architecture/contract changes)

- No new threads, no polling.
- `update_launch_config_model` keeps its existing signature and the `"unknown"`
  guard; we change only *whether* the resolver invokes the persist path for an
  unpinned agent.
- The `"unknown"` legacy heal in `merge_restart_config` stays.
- Explicit `--model` continues to pin and persist normally.

## Live agent (escalated separately, NOT part of this code change)

The code fix stops *us* from feeding/persisting a foreign model. The already
running agent resumes codex session `019f1332…`, whose rollout still bakes in
the grok string codex-internally. Fully healing the live agent needs an operator
decision: (A) reset the agent's codex `session_uuid` (fresh codex session at the
real default; our threads.db history is untouched, codex-side continuity lost),
or (B) leave it and let it self-heal on the next fresh codex session. No
destructive action without explicit approval.

## Tests (TDD)

`tests/unit_tests/process/test_failover_issue77.py` (real parser, no MagicMock
for the parse path):

1. `test_merge_restart_config_pp_same_provider_no_model_resets_to_default`
   (NEW / REPLACES `test_pp_same_provider_keeps_saved_model`): saved
   `provider=codex, model=grok-composer-2.5-fast`; args `-pp codex` no `--model`
   → `merged.model is None`. (Encodes the operator rule; supersedes the prior
   assumption that same-provider `-pp` keeps the saved model.)
2. `test_merge_restart_config_provider_flag_no_model_resets`: `--provider codex`
   no `--model`, same saved provider → `merged.model is None`.
3. `test_merge_restart_config_plain_restart_keeps_model`: no provider/model
   flags → saved model preserved.
4. `test_merge_restart_config_explicit_model_wins`: `-pp codex --model X` →
   `merged.model == "X"`.

`tests/unit_tests/process/` resolver-guard test:

5. `test_resolver_seed_does_not_persist_for_unpinned_agent`: with
   `config.model is None`, the startup seed updates display but
   `update_launch_config_model` is NOT called with the resolved value (assert via
   a fake SessionManager recording calls).
6. `test_merge_restart_config_pp_same_provider_clears_preferred_model` (RC-1):
   saved `provider=codex, preferred_model=grok-composer-2.5-fast`; args
   `-pp codex` no `--model` → `merged.preferred_model is None`.

Pre-existing MagicMock/argv failures in
`test_failover_issue77.py`/`test_restart_option.py`/`test_dev_flag.py` are
out of scope (identical with and without this change).

## §65 amendment (RC-2)

`DESIGN_CODEX_RESOLVED_MODEL_DISPLAY.md` §65 originally said: *"`launch_config["model"]`
can be updated from `None` to the concrete Codex model only after Codex reports
it. … On restart, passing this known value as `--model` is acceptable because it
reflects the previous actual session model."* — In the same PR as this fix, that
sentence is narrowed: **for an agent launched with `config.model is None`, the
resolved value is display-only and is NOT written to `launch_config["model"]`.**
Only an explicit operator `--model M` pins execution. This eliminates the
re-pinning vector while preserving display continuity via the in-memory display
path.
