# DESIGN: fix silent drop of Claude Code model aliases + wrong hardcoded default

Status: revision 3 — SIGNED OFF (review agent, round 2 conditions applied)
Owner: (internal agent)
Reviewer: (round 1: APPROVE WITH CHANGES; round 2: APPROVE
WITH CHANGES — "after the four required documentation corrections,
DESIGN SIGNED OFF; no further architecture review needed")
Related incident (2026-07-28) — a user started
`deep-worker --restart <agent> --model fable`, agent booted on `sonnet`
without any warning. Persisted `launch_config.model = "sonnet"`.

## 1. Problem

Two independent defects in the same code path silently substitute the model
an operator passes on the CLI:

### 1.1 `model_can_belong_to_provider` uses an outdated whitelist

`deep_worker_cli/config.py:1382-1418`. For `provider="claude-code-sdk"` the
predicate only accepts these tokens:

```python
if provider_name == "claude-code-sdk":
    return (
        model_name in ("sonnet", "haiku", "opus", "inherit")
        or "claude" in model_name
    )
```

The list is stale. Per official
[Claude Code — Model configuration](https://code.claude.com/docs/en/model-config)
(fetched 2026-07-28), Claude Code CLI accepts the following short aliases:
`default`, `best`, `fable`, `sonnet`, `opus`, `haiku`, `sonnet[1m]`,
`opus[1m]`, `opusplan`. Any of these plus any `claude-*` full ID or
`claude-*[1m]` variant is legitimate. Whitelist only knows four tokens.

The mirror predicate in the web frontend
(`app/routers/agents.py:41-74`) has the same stale list.

Concrete failure: `--model fable` → predicate returns False → predicate is
used from `merge_restart_config` (agent_command.py:396-409) to nullify
`effective.model` → `build_launch_config` falls back to
`get_default_model_name("claude-code-sdk")` → `"sonnet"`. The user got Sonnet
without asking, without notice.

### 1.2 Silent fallback (no warning)

`merge_restart_config` at agent_command.py:396-409:

```python
if (
    effective.model
    and (
        ...
        or not model_can_belong_to_provider(effective.provider, effective.model)
    )
):
    effective.model = None
```

No `logger.warning`, no `print` to stderr, no `sys.exit`. The rejected value
is discarded silently. Same in the web `_model_can_belong_to_provider` path
(agents.py:727).

### 1.3 Hardcoded `sonnet` default for `claude-code-sdk` is wrong

`config.py:1423-1439`:

```python
_PROVIDER_DEFAULT_MODELS = {
    ...
    "claude-code-sdk": (None, "sonnet"),
    "codex": ("CODEX_MODEL", None),   # ← let CLI decide
    "grok":  ("GROK_MODEL",  None),   # ← let CLI decide
}
```

Sonnet is only the default for a subset of Claude accounts (Pro / Team
Standard / Enterprise subscription seats — per
[Claude Code default-model setting](https://code.claude.com/docs/en/model-config#default-model-setting)).
For Anthropic API / Max / Team Premium / Enterprise pay-as-you-go / Bedrock
/ AWS Claude Platform / Google Cloud Agent Platform the real default is
Opus 5. For Microsoft Foundry it's Sonnet 4.5.

Codex and Grok already have the correct pattern: env-var only, hardcoded
default = `None`, meaning "do not pass `--model`, let the CLI pick its own
default". Claude Code must do the same — it is a CLI wrapper exactly like
codex/grok.

## 2. Requirements (user, verbatim)

Quote:

> Fix everything. It must follow the same logic as codex/grok.

Quote (earlier turn):

> if the model is not found and there is a fallback to something else, a
> warning must be printed. The default for the claude code provider must be
> the provider's real default (what claude code itself considers the
> default, not sonnet).

Derived requirements:

R1. `--model X` for `claude-code-sdk` where `X` is a known short alias
    (`default`, `best`, `fable`, `sonnet`, `opus`, `haiku`, `sonnet[1m]`,
    `opus[1m]`, `opusplan`) or any `claude-*` full ID must NOT be dropped.

R2. When a model reset happens (stale-provider mismatch, unknown token,
    or bare provider declaration), a WARNING must reach the operator on
    stderr exactly once, naming the concrete (provider, dropped model,
    *actual* replacement) triple. No silent substitution. Honouring a
    valid alias is not a reset and must stay quiet.

R3. Claude Code's `_PROVIDER_DEFAULT_MODELS` entry must behave exactly
    like codex/grok: `("CLAUDE_CODE_SDK_MODEL", None)`. `None` means the
    wrapper does not pass `--model` to `claude` and the CLI resolves its
    own account-type default. No hardcoded `"sonnet"`.

R4. Both call sites of the predicate (CLI `merge_restart_config` and web
    `_spawn_subprocess`) must gain the warning + widened allow-list. They
    must not diverge again (single source of truth).

R5. All existing tests continue to pass. New TDD tests cover: `fable`
    accepted, silent-drop replaced by warning, `claude-code-sdk` default
    resolves to `None` (like codex).

## 3. Non-goals

- Do NOT change what `claude` CLI itself considers its default. We just
  stop lying about it and let the CLI answer.
- Do NOT add a HTTP call to Anthropic to enumerate models. The CLI is the
  authoritative source; we defer.
- Do NOT introduce `--strict-model` / fail-fast on unknown tokens. Warning
  is sufficient per R2. (If operator really wants to reject unknown, they
  can grep the log — the message is unambiguous.)
- Do NOT touch `claude_code_sdk_provider/chat_model.py` — it already handles
  `model=None` correctly (`ClaudeAgentOptions(model=...)` is only created
  when `model is not None`; a `None` model means the CLI resolves its own
  default). Verified: `claude_code_sdk_provider/chat_model.py:434-436`.
- Do NOT touch the local `claude_sdk` source checkout — it's the Anthropic
  Python SDK (`claude_agent_sdk` package). No model-list registry there;
  nothing relevant.

## 4. Design

### 4.1 Widen the allow-list (single source of truth in `config.py`)

Replace the four-token check with the actual documented alias set, plus
a permissive fallback for any full `claude-*` model ID (with or without
the `[1m]` context-window suffix):

```python
_CLAUDE_CODE_SHORT_ALIASES = frozenset({
    "default", "best", "fable", "sonnet", "opus", "haiku",
    "sonnet[1m]", "opus[1m]", "opusplan",
    # Legacy internal alias kept for backward compat with older configs
    # persisted by earlier deep-worker versions.
    "inherit",
})

def _is_claude_code_model(model_name: str) -> bool:
    if model_name in _CLAUDE_CODE_SHORT_ALIASES:
        return True
    # Full model ID: claude-sonnet-5, claude-opus-4-8, claude-fable-5,
    # claude-haiku-4-5-20251001, and their ``[1m]`` context-window
    # variants (claude-sonnet-5[1m]) — the bare ``startswith`` already
    # covers the suffixed forms, so no extra branch is needed.
    return model_name.startswith("claude-")
```

Source: `code.claude.com/docs/en/model-config` §"Model aliases" table.

Review note (review agent): an earlier draft had a second
`startswith("claude-") and endswith("[1m]")` branch after the plain
`startswith` check. It was unreachable and therefore misleading; it has
been removed. `[1m]`-suffixed full IDs are accepted by the single
`startswith` test.

### 4.2 Change `claude-code-sdk` default entry

```python
_PROVIDER_DEFAULT_MODELS = {
    ...
    # Claude Code: do NOT pin a model. Let the claude CLI resolve its own
    # account-type default (Opus 5 on Anthropic API / Max / Team Premium
    # / Enterprise PAYG / Bedrock / AWS / GCP; Sonnet 5 on Pro / Team
    # Standard / Enterprise seats; Sonnet 4.5 on Microsoft Foundry — see
    # code.claude.com/docs/en/model-config#default-model-setting).
    # CLAUDE_CODE_SDK_MODEL env var pins it explicitly if the operator wants.
    "claude-code-sdk": ("CLAUDE_CODE_SDK_MODEL", None),
    ...
}
```

This mirrors codex/grok exactly. `ClaudeCodeChatModel(model=None)` is
already a safe no-op (`chat_model.py:434-436` — no `ClaudeAgentOptions`
is constructed, so no `--model` arg is added to the subprocess), so the
change is transparent to the provider layer.

`build_launch_config` already persists `model=None` correctly for codex
(comment at agent_command.py:483-490). The same code path now applies to
claude-code-sdk.

### 4.3 Emit a WARNING on any model reset

#### 4.3.1 Exact state transitions being instrumented

An earlier draft said "three reset sites". That is not what the code
does. The precise transitions in `merge_restart_config` are:

| # | Location | Transition | Trigger |
|---|----------|------------|---------|
| T1 | agent_command.py:393-394 | `effective.model` → `None` | `provider_decl_unpinned` — operator declared `-pp`/`--provider` with no `--model` and not `--no-update-preferred` |
| T2 | agent_command.py:398-409 | `effective.model` → `None` | saved `model_provider` differs from `effective.provider`, **or** predicate rejects the token |
| T3 | agent_command.py:454-463 | `effective.model` → `effective.preferred_model` (a **restore**, not a reset) | `--no-update-preferred` + `provider_overridden` + failover return to the preferred provider |

Only T1 and T2 drop a model and therefore warrant a WARNING. T3 puts a
model *back* and must stay silent. The design instruments T1 and T2.

#### 4.3.2 The helper

```python
def _warn_model_reset(reason: str, provider: str | None,
                      dropped: str | None, replacement: str | None) -> None:
    """Emit exactly one operator-visible WARNING when a model is dropped.

    Reasons: 'unknown-alias' (predicate rejected the token),
    'stale-provider' (saved model belonged to a different provider),
    'provider-declared' (operator declared a provider without --model).

    ``replacement`` is the model that will actually be used, as resolved
    by :func:`get_default_model_name`. ``None`` means the provider pins
    no default and the downstream CLI resolves its own.
    """
    if not dropped:
        return  # nothing was actually dropped, don't spam
    repl_txt = (
        "provider default (resolved by the provider CLI)"
        if replacement is None
        else repr(replacement)
    )
    msg = (
        f"[deep-worker] WARNING: model {dropped!r} was reset "
        f"(reason={reason}, provider={provider!r}); "
        f"using {repl_txt} instead."
    )
    print(msg, file=sys.stderr, flush=True)
```

**Single emission path (review point 2).** The draft emitted both
`print(..., file=sys.stderr)` and `logger.warning(msg)`. In a normal CLI
invocation the root logger has a `StreamHandler` on stderr, so the
operator would see the same sentence twice. We therefore emit **one**
guaranteed stderr line and no logger call from this helper.

**Where that line actually lands (review R2-3).** An earlier draft
claimed the single `print` is "captured exactly once in both the
interactive terminal and the on-disk `.err.log`". That is **not proved**
and is not claimed here. `merge_restart_config` runs in the *restart
invoker* — the foreground `deep-worker` process the operator typed —
**before** the daemon's stderr redirection is installed. So:

- **CLI path (the requirement):** the warning appears on the invoking
  terminal's stderr. This is exactly what the user asked for ("a warning
  must be printed") and is fully satisfied.
- **Agent `.err.log`:** *not* guaranteed by this mechanism. If durable
  on-disk capture of the reset is wanted later, it needs a separate,
  explicitly-designed hook inside the daemon — out of scope here and not
  asserted by any test in §5.
- **Web path:** `print(..., file=sys.stderr)` from the web frontend's
  `app/routers/agents.py` reaches the **web server's logs**, not the
  browser. The honest contract is therefore: *the web
  layer emits a server-side warning log on model reset*. The HTTP
  response body and the UI are **not** extended by this change; surfacing
  the warning to a browser operator would require extending the endpoint
  response/UI and is explicitly a non-goal (§3). Tests assert the
  server-side emission only.

#### 4.3.3 Correct `reason` computation (review point 1)

The draft's `reason = "stale-provider" if saved_model_provider else
"unknown-alias"` is wrong: a truthy `saved_model_provider` that **equals**
the current provider still labels a pure predicate rejection as
`stale-provider`. Compute the mismatch explicitly and reuse it as both
the branch condition and the reason selector:

```python
saved_model_provider = saved.get("model_provider")
provider_mismatch = bool(
    isinstance(saved_model_provider, str)
    and saved_model_provider
    and saved_model_provider != effective.provider
)
if effective.model and (
    provider_mismatch
    or not model_can_belong_to_provider(effective.provider, effective.model)
):
    reason = "stale-provider" if provider_mismatch else "unknown-alias"
    ...
```

**Explicit `--model` never inherits the saved token's provenance.** When
`model_explicit` is true, `effective.model` is a token the operator just
typed; the saved `model_provider` describes the *previous* token and says
nothing about the new one. Attributing it would mislabel
`--model bogusXYZ` as `stale-provider`. So:

```python
if model_explicit:
    provider_mismatch = False   # the new token has no saved provenance
```

placed before the branch. A typo'd explicit model is therefore always
reported as `unknown-alias`, which is the truth.

#### 4.3.4 Reporting the real replacement (review point 2)

`replacement=None` with the text "provider default (resolved by CLI)" is
true for `claude-code-sdk` / `codex` / `grok` (all three map to a `None`
hardcoded default after §4.2) but **false** for `deepseek`, `openai`,
`anthropic`, `google`, whose `_PROVIDER_DEFAULT_MODELS` entries carry a
concrete hardcoded string that `build_launch_config` materialises. Saying
"resolved by CLI" there would be a lie in the very message meant to stop
lying to operators.

The reset sites therefore resolve the replacement the same way
`build_launch_config` will:

```python
from deep_worker_cli.config import get_default_model_name
replacement = get_default_model_name(effective.provider)
_warn_model_reset(reason, effective.provider, effective.model, replacement)
effective.model = None
```

so a hardcoded-default provider reset names its concrete model, while a
`claude-code-sdk` reset prints `using provider default (resolved by the
provider CLI) instead.`

The test must not hardcode the expected string: `_PROVIDER_DEFAULT_MODELS`
values drift (as of config.py:1424 the `deepseek` default is
`deepseek-v4-flash`, not the `deepseek-chat` an earlier draft quoted —
that alias is now only a context-window key at config.py:423). Test (e)
therefore asserts the warning contains
`get_default_model_name("deepseek")` looked up at runtime, so it stays
correct when the pinned default changes.

#### 4.3.5 Wired call sites

```python
# T1 — provider declared without --model
if provider_decl_unpinned:
    if effective.model:
        _warn_model_reset(
            "provider-declared", effective.provider, effective.model,
            get_default_model_name(effective.provider),
        )
    effective.model = None
    model_reset_cause = "provider-declared"   # see §4.5

# T2 — stale provenance or rejected token
if effective.model and (
    provider_mismatch
    or not model_can_belong_to_provider(effective.provider, effective.model)
):
    reason = "stale-provider" if provider_mismatch else "unknown-alias"
    _warn_model_reset(
        reason, effective.provider, effective.model,
        get_default_model_name(effective.provider),
    )
    effective.model = None
    model_reset_cause = reason               # see §4.5
```

Same helper is added to the web layer (`agents.py`) where
`_model_can_belong_to_provider` is consulted (line 727) — either mirror
the print, or (better) import `model_can_belong_to_provider` from
`deep_worker_cli.config` after confirming it doesn't drag in heavy deps.

Check the transitive imports: `config.py` top of file — if importing it
in the web spawn path pulls langchain / transformers, we keep the local
mirror BUT with the widened allow-list and add the same warning. This
was already the reason a local mirror exists (see the note in
a related internal incident analysis (not included here), part 1 of
session summary).

Decision: keep local mirror in `agents.py`, factor the predicate literal
into a small standalone module `deep_worker_cli/model_predicate.py`
(zero heavy deps) that both `config.py` and `agents.py` import. This
prevents future divergence.

Alternative considered and rejected: put the predicate into a package
that already exists as leaf (`deep_worker_cli/agent_config_pkg/`) —
rejected because that package is imported by other heavy modules; the
web layer's whole point was to avoid that.

### 4.4 Storage-format compatibility

`launch_config.model = None` was already the norm for codex/grok. The
web reader and CLI reader both handle it (`agents.py:_spawn_subprocess`
skips `--model` when saved model is falsy; verified by existing tests
the web frontend's tests
(`tests/unit/test_spawn_subprocess_model.py`)).

Backward compat: an existing agent whose `launch_config.model` was
persisted as `"sonnet"` (e.g. one launched earlier with `--model sonnet`)
will keep that value on restart because saved config wins over provider
default in
`merge_restart_config` when `--model` is not passed. That is deliberate:
we do not silently re-model a running fleet. Two operator actions clear
it, and they behave differently with respect to R2:

| Operator action | Effect on `model` | WARNING? |
|---|---|---|
| `--model default` (or `opus`, `fable`, …) | replaced by the typed alias | **No** — `default` is a documented alias (§4.1), the predicate accepts it, nothing is dropped |
| bare `--provider claude-code-sdk` (no `--model`) | reset to `None` → provider default | **Yes** — T1 fires, `sonnet` was dropped |

Review correction (review agent): the draft claimed "Both paths now emit
WARNING". That was wrong. Passing a valid alias such as `--model default`
is an *honoured* request, not a fallback, and warning on it would train
operators to ignore the warning. Only the bare provider declaration
drops a value and warns. This is asserted by a test in §5 (case `d`).

### 4.5 Interaction with `preferred_model`

#### 4.5.1 Why the draft's placement could not work (review point 3)

The draft appended a `if effective.preferred_model == dropped_model:`
check immediately after line 409. That is dead code:

- At line 409 `effective.preferred_model` has **not yet been restored**
  from `saved`. The whole preferred cascade begins at line 421. The
  comparison therefore reads `None` and never fires.
- Even if it did fire, lines 440-450 run afterwards and would happily
  re-adopt `saved["preferred_model"]` or `saved["model"]`, undoing the
  reset.

So the reset intent must be carried **forward into** the cascade, not
executed before it.

#### 4.5.2 Mechanism: an explicit reset-cause flag

T1/T2 (§4.3.5) set a local `model_reset_cause: str | None`, initialised
to `None` before T1. The cascade at lines 431-452 then gains a branch
keyed on that flag rather than re-deriving intent from now-stale state:

```python
model_reset_cause = None      # set by T1 / T2 above
explicit_model_rejected = model_explicit and model_reset_cause == "unknown-alias"

...

if provider_decl_unpinned:
    effective.preferred_model = effective.model          # existing RC-1
elif explicit_model_rejected and effective.provider == effective.preferred_provider:
    # The operator typed a model for the CURRENT provider and it was
    # rejected. Do not silently resurrect the previous preference:
    # both active and preferred collapse to the provider default.
    if saved.get("preferred_model"):
        _warn_model_reset(
            "preferred-reset-after-invalid-explicit",
            effective.preferred_provider,
            saved.get("preferred_model"),
            get_default_model_name(effective.preferred_provider),
        )
    effective.preferred_model = None
elif pp_switched and not model_explicit:
    effective.preferred_model = effective.model
elif saved.get("preferred_model") and model_can_belong_to_provider(
    effective.preferred_provider, saved.get("preferred_model")
):
    effective.preferred_model = saved.get("preferred_model")
elif ...  # unchanged tail
```

The `effective.provider == effective.preferred_provider` guard is what
keeps the failover case safe: a stale *saved active* model that gets
dropped while the agent is temporarily on a different provider does not
touch the preference belonging to the preferred provider.

#### 4.5.3 Contract matrix (fixed by tests)

| # | Scenario | `effective.model` | `effective.preferred_model` | WARNING |
|---|---|---|---|---|
| C1 | explicit invalid `--model bogusXYZ`, provider == preferred_provider, saved preferred `opus` | `None` (provider default) | `None` — old preference NOT resurrected | 2: `unknown-alias` (for `bogusXYZ`) + `preferred-reset-after-invalid-explicit` (for `opus`) |
| C2 | explicit valid `--model fable` on claude-code-sdk | `"fable"` | `"fable"` (line 465-466 tail) | none |
| C3 | failover: `--provider codex --no-update-preferred`, saved active model `sonnet` is stale for codex, saved preferred_provider `claude-code-sdk` + preferred_model `opus` | `None` (codex default) | `"opus"` — preserved, provider differs from preferred | 1: `stale-provider` |
| C4 | restore-on-idle: `--provider claude-code-sdk --no-update-preferred`, preferred_provider `claude-code-sdk`, preferred_model `opus` | `"opus"` via T3 restore | `"opus"` | none (T3 is a restore) |
| C5 | saved `preferred_model` itself invalid for preferred_provider (legacy poison) | per other rules | `None` — dropped by the existing predicate guard on line 440 | none required (pre-existing silent guard, out of scope) |
| C6 | bare `--provider claude-code-sdk`, saved model `sonnet` | `None` | `None` (RC-1) | 1: `provider-declared` |

C3 and C4 are exactly the `--no-update-preferred` restore/failover paths
the reviewer asked to pin down; both get dedicated tests in §5.

**Why C1's second reason is not `unknown-alias` (review R2-4).** In C1
the dropped preference is `opus` — a perfectly valid claude-code-sdk alias.
It is cleared not because the token is bad but because the operator's
explicit replacement request was invalid, and honouring the *old*
preference while rejecting the *new* active model would silently
resurrect a preference the operator was in the middle of replacing. The
reason string is therefore `preferred-reset-after-invalid-explicit`,
which states the actual cause. Labelling it `unknown-alias` would tell
the operator that `opus` is unrecognised, which is false and would send
them debugging the wrong thing. Test (g)/C1 asserts the exact reason
string and asserts that the message for `opus` does **not** contain
`unknown-alias`.

## 5. File touch-list

Production (modified):
- `deep_worker/deep_worker_cli/deep_worker_cli/model_predicate.py` — NEW,
  standalone predicate + short-alias set (no heavy deps).
- `deep_worker/deep_worker_cli/deep_worker_cli/config.py` — rewrite
  `model_can_belong_to_provider` to delegate to `model_predicate`; change
  `_PROVIDER_DEFAULT_MODELS["claude-code-sdk"]` to `("CLAUDE_CODE_SDK_MODEL", None)`.
- `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py` —
  add `_warn_model_reset` helper, wire it at transitions T1/T2 (§4.3.1),
  add `model_reset_cause` and the preferred cascade branch (§4.5.2).
- the web frontend's `app/routers/agents.py` — replace local
  `_model_can_belong_to_provider` with import from `model_predicate`;
  emit warning on drop.
- `deep_worker/deep_worker_cli/deep_worker_cli/agent_config_pkg/agent_config.py`
  — **docstring only** (lines 109-116), see below.

#### 5.1 Consumers of the predicate (review point 5)

`model_can_belong_to_provider` has four consumers besides its own
definition. All of them inherit the widened allow-list automatically
because they import from `config.py`, which delegates to the new leaf
module. No behavioural edit is required in the two below — but they are
listed here because the behaviour *change* they inherit is
operator-visible and therefore needs regression coverage.

**`rpyc_service/agent_service.py:913-924`** — filters the model reported
in agent metadata; a token failing the predicate is hidden from the UI.
Before this change, a claude-code-sdk agent running `fable` would report a
blank/stale model in the web UI. After delegation it reports `fable`.
No code edit; **regression test added** (§5.2, `test_resolved_model_metadata.py`)
asserting claude-code-sdk + `fable` is surfaced, not suppressed.

**`agent_config_pkg/agent_config.py:123-131`** (`set_dynamic_display_model`)
— same delegation, same automatic fix. Its docstring at lines 109-116
currently states that a `None` model is a codex-specific condition. After
§4.2 that is stale: `None` is now the normal resting state for
`claude-code-sdk` and `grok` too. Docstring updated accordingly; **regression
test added** asserting `fable` yields a display model rather than being
discarded.

**`cli/agent_command.py`** and **the web frontend's `app/routers/agents.py`**
are the two sites that actually change logic (warning + import), covered
above.

#### 5.2 Tests (TDD — written first, must be RED before the fix)

- `tests/unit_tests/config/test_claude_code_short_aliases.py` — NEW.
  Cases: `fable`, `best`, `default`, `opusplan`, `sonnet[1m]`,
  `opus[1m]`, `claude-fable-5`, `claude-haiku-4-5-20251001`,
  `claude-sonnet-5[1m]` accepted; `gpt-5` / `grok-4` / `""` / `None`
  rejected. Also asserts the leaf module imports with **no** heavy
  dependency (`sys.modules` free of `langchain*`/`transformers` after a
  fresh subprocess import) — this is the property the web layer relies
  on.
- `tests/unit_tests/config/test_default_model.py` — extend:
  `get_default_model_name("claude-code-sdk") is None` (line 53 changes from
  `== "sonnet"`); `CLAUDE_CODE_SDK_MODEL=X` env var respected.
- `tests/unit_tests/cli/test_model_reset_warning.py` — NEW. Cases:
  - (a) `--model fable` on saved `sonnet` → no reset, **no** warning.
  - (b) `--model bogusXYZ` → reset, warning contains `bogusXYZ` and
    `reason=unknown-alias` (**not** `stale-provider`, even though
    `saved["model_provider"] == "claude-code-sdk"` is truthy — pins review
    point 1).
  - (c) bare `--provider claude-code-sdk` over saved `sonnet` → reset,
    warning contains `sonnet` and `reason=provider-declared`.
  - (d) `--model default` → **no** warning (pins §4.4 correction).
  - (e) **hardcoded-default provider**: invalid `--model` on a provider
    whose `_PROVIDER_DEFAULT_MODELS` entry has a concrete string (e.g.
    `deepseek`) → warning contains `get_default_model_name("deepseek")`
    **looked up at runtime** (never a hardcoded literal), and does
    **not** say "resolved by the provider CLI" (pins review point 2).
  - (f) exactly **one** occurrence of the warning sentence in captured
    stderr — no double emission (pins review point 2).
  - (g)-(l) the six rows of the §4.5.3 contract matrix C1-C6, including
    both `--no-update-preferred` paths (C3 failover, C4 restore). C1
    additionally asserts the second warning's reason is
    `preferred-reset-after-invalid-explicit` and that the message naming
    `opus` does **not** contain `unknown-alias`.
- `tests/unit_tests/rpyc_service/test_resolved_model_metadata.py` —
  extend: claude-code-sdk + `fable` is surfaced in metadata, not suppressed.
- `tests/unit_tests/agent_config_pkg/` — extend/add: `set_dynamic_display_model`
  keeps `fable` for claude-code-sdk.
- the web frontend's `tests/unit/test_spawn_subprocess_model.py` — extend
  with `fable` case (spawn must include `--model fable`, no drop, no
  fallback to sonnet).

#### 5.3 Integration test (no mocks) — revised per review point 6

`tests/integration_tests/test_restart_with_fable_alias.py` — NEW.

The draft was not executable as written. Revised specification:

**Setup — isolation via `HOME`, not a nonexistent override (review R2-1).**
An earlier draft claimed the test would point `DEEP_WORKER_HOME` at a
sandbox. **No such variable exists**: `grep -rn DEEP_WORKER_HOME` over
`deep_worker_cli` returns zero hits. Agent state is rooted at
`Path.home()` unconditionally — verified sites include
`process/agent_process.py:555` (`data_dir = deep_worker_home()/name`),
`process/agent_process.py:123,198` (provider session state),
`process/agent_stop.py:63`, `config.py:846` (at the time of writing, the
legacy user-config tree that POLICIES §16 has since banned; the call site
now resolves through `deep_worker_home()`), `config.py:1027` (`~/.agents`),
and `history_memory_tool.py:151,157`.

Isolation is therefore achieved by running the child with **`HOME=<tmp_path>`**
in its environment, which relocates `~/.deep_worker/<name>` and the
registry wholesale. Because `Path.home()` honours `HOME` on POSIX, this
is sufficient and is the mechanism existing isolated-HOME tests already
use. Two consequences must be handled explicitly, as those tests do:
- Dependency discovery breaks if the interpreter resolves user site
  packages under the new `HOME`; the test preserves the original
  `PYTHONPATH` and sets `PYTHONUSERBASE` to the real user base so imports
  still resolve.
- `claude` CLI credentials/config live under the real `HOME`. The test
  either copies the minimal credential path into the sandbox or exports
  the provider's config-dir variable pointing at the real location —
  whichever the installed CLI version supports. This is stated as a
  precondition because it determines whether the test can run at all.

The concrete sandbox layout asserted by the test:
`~/.deep_worker/<agent>/session_metadata.json` (the sandbox `HOME` is
pointed at `<tmp>`).

Sequence:
1. Genuine first launch (`deep-worker <name> --provider claude-code-sdk
   --dir <tmp>`) under the sandbox `HOME`, letting the product seed
   `session_metadata.json` + registry entry through its own code path.
2. Stop that agent.
3. `deep-worker --restart <name> --model fable`.

**A live `claude` child is not implied by a restart (review R2-2).**
`ClaudeSDKClient` is constructed and `connect()`ed **lazily** — verified
at `claude_code_sdk_provider/client_manager.py:555-569`, inside the
per-thread client-acquisition path. An agent that restarts and sits idle
has **no** `claude` subprocess, so asserting `/proc/<pid>/cmdline` right
after startup would inspect a process that does not exist (or, worse,
match an unrelated `claude` belonging to the operator's own session).

The test therefore **drives a real query** after restart — it sends an
actual message to the agent (trivial prompt, e.g. `"ping"`) so the
provider client is genuinely constructed and connected, and only then
inspects argv.

**Robust child identification.** The test must not `pgrep claude`
globally. It resolves the child by walking the process tree **down from
the known daemon PID** (read from the sandbox registry/metadata):
enumerate `/proc/*/stat` field 4 (`ppid`) to find descendants of that
PID, and select the one whose `/proc/<pid>/cmdline` argv[0] basename is
`claude`. This guarantees the inspected process belongs to *this* test's
agent. If zero such descendants exist after the query completes, the
test **fails** (it does not silently pass) — absence of the child means
the alias was never forwarded.

**Waiting — event-driven, no polling.** Polling a log file is forbidden
by project policy. The test consumes the child's **stdout stream
directly**: `subprocess.Popen(..., stdout=PIPE, text=True)` and a bounded
`for line in proc.stdout:` read that blocks in the kernel until data
arrives and stops at the `[STARTUP]` line. The bound is enforced by an
outer `concurrent.futures` timeout on the single read call, not a sleep
loop. Readiness for the follow-up query uses the existing event-driven
registry readiness signal that `--restart` itself consumes, not a
re-read of any file.

**Assertions.** Not just the log text — the test asserts *actual startup
state*:
- The persisted `~/.deep_worker/<name>/session_metadata.json`
  `launch_config.model == "fable"` (proves the merge kept the alias).
- After the driving query, the `claude` **descendant of this agent's
  daemon PID** has `--model fable` in `/proc/<pid>/cmdline`, proving the
  alias reached the provider process and was not swallowed downstream.
- stderr contains **no** `WARNING: model` line.

**Cleanup.** All process teardown lives in `try/finally`: the daemon is
SIGTERM'd and, after a bounded wait, SIGKILL'd, and the registry sandbox
is removed — executed even when an assertion fails mid-test, so a failing
run cannot leak a live agent.

**Skip policy.** `skipif(shutil.which("claude") is None)` remains only as
a developer-machine convenience. Per the reviewer: a skipped test proves
nothing. Acceptance for this change requires the test to be observed
**passed, not skipped**, in the release environment; the run output is
attached to the commit review. A `DEEP_WORKER_REQUIRE_INTEGRATION=1` env
var turns the skip into a hard failure so CI cannot silently degrade to
green-by-skipping.

## 6. Open questions

None significant. Two minor design choices made unilaterally per user's
"use common sense":

Q1. Fail-fast vs WARNING on unknown alias.
    User asked for WARNING explicitly ("a warning must be printed"). We
    do WARNING. Fail-fast can be added later if needed.

Q2. Storage: keep `model: null` (like codex/grok) vs materialise the
    account-type default. We keep `null` — matches codex/grok, matches
    user's "follow the same logic", and correctly reflects the fact that
    "default" is account-dependent and should be re-resolved at each
    launch.

## 7. Rollout

1. Land in `dev` after review sign-off. TDD, tests first.
2. Existing agents with `launch_config.model = "sonnet"` keep that value
   until the operator acts. The two actions differ (see §4.4):
   - `--model default` (or `opus`/`fable`/…) — an *explicit replacement*.
     The value changes to what was typed. Nothing is dropped, so **no
     warning**.
   - bare `--provider claude-code-sdk` — a *reset*. `sonnet` is dropped and
     the provider default applies, **with** a warning.
3. New agents launched without `--model` on Claude Code start with
   `launch_config.model = null`, and the running `claude` CLI picks
   its own default per account type.
4. Fast-forward to `master` (production tree) after commit lands
   on `dev`.

## 8. Review log

### Round 1 — reviewer, verdict APPROVE WITH CHANGES

Architecture accepted: leaf predicate module, documented alias set,
`("CLAUDE_CODE_SDK_MODEL", None)` default entry, and preserving an existing
persisted `"sonnet"` until the operator acts. All required changes are
applied in revision 2:

| # | Reviewer finding | Resolution |
|---|---|---|
| 1 | `reason` mislabels predicate rejection as `stale-provider` when saved `model_provider` is truthy but equal to the current provider | §4.3.3 — explicit `provider_mismatch` boolean drives both branch and reason; explicit `--model` forces `provider_mismatch = False` so a typo is never attributed to the old token's provenance. Test (b) pins it. |
| 2 | Warning must name the real replacement; `None` + "resolved by CLI" is false for hardcoded-default providers. Also: don't `print` *and* `logger.warning` | §4.3.4 — replacement resolved via `get_default_model_name(provider)`, test (e) covers `deepseek`. §4.3.2 — single guaranteed stderr emission, no logger call; test (f) asserts exactly one occurrence. |
| 3 | Preferred cascade in the proposed position cannot work (`preferred_model` not yet restored at line 409; lines 440-450 would undo it) | §4.5.1 explains the dead code; §4.5.2 replaces it with a `model_reset_cause` flag consumed *inside* the cascade; §4.5.3 fixes the C1-C6 contract matrix including both `--no-update-preferred` paths. |
| 4 | `--model default` is a valid alias and must not warn; "Both paths now emit WARNING" is wrong | §4.4 — corrected with an explicit two-row table; test (d) pins it. |
| 5 | Consumers list incomplete (`rpyc_service/agent_service.py`, `agent_config_pkg/agent_config.py` + its stale docstring) | §5.1 — both listed with the behaviour they inherit; docstring 109-116 updated; regression tests added in §5.2. |
| 6 | Integration test not executable as described; polling forbidden; skip ≠ coverage | §5.3 — real seeded launch via the product's own first-launch path, event-driven bounded stream read instead of log polling, argv + persisted-config assertions, `try/finally` SIGTERM→SIGKILL cleanup, and `DEEP_WORKER_REQUIRE_INTEGRATION=1` so CI cannot pass by skipping. |
| extra | Unreachable `startswith && endswith("[1m]")` branch | §4.1 — removed. |
| extra | "three reset sites" doesn't match the code | §4.3.1 — replaced by an explicit T1/T2/T3 transition table; T3 is a restore and stays silent. |

### Round 2 — reviewer, verdict APPROVE WITH CHANGES → SIGNED OFF

Reviewer accepted the merge/preferred logic outright: explicit
`provider_mismatch` provenance, concrete replacement lookup,
exactly-one CLI warning, the T1/T2/T3 model, the guarded
`model_reset_cause` cascade with C1-C6, consumer coverage, default
backward compatibility, and the leaf predicate design. Four
documentation corrections were required; all are applied in revision 3,
each verified against the source rather than accepted on assertion:

| # | Reviewer finding | Verification | Resolution |
|---|---|---|---|
| R2-1 | `DEEP_WORKER_HOME` does not exist; state the real isolation mechanism | `grep -rn DEEP_WORKER_HOME` over `deep_worker_cli` → **0 hits**; `Path.home()` confirmed at `agent_process.py:123,198,555`, `agent_stop.py:63`, `config.py:846,1027`, `history_memory_tool.py:151,157` | §5.3 — isolation is `HOME=<tmp_path>`, with `PYTHONUSERBASE`/`PYTHONPATH` preserved for dependency discovery and the `claude` credential path called out as a precondition. Concrete sandbox layout stated. |
| R2-2 | A restart does not guarantee a live `claude` child; argv assert could hit a phantom or an unrelated host process | `client_manager.py:555-569` confirmed: `ClaudeSDKClient(options=...)` + `await client.connect()` inside the lazy per-thread acquisition path | §5.3 — test now **drives a real query** after restart to force client construction, then identifies the child by walking `/proc/*/stat` ppid **down from this agent's daemon PID** (never a global `pgrep claude`). Zero descendants ⇒ test **fails**, not passes. |
| R2-3 | Don't claim unproved dual terminal+disk persistence; define the web contract honestly | `merge_restart_config` runs in the foreground restart invoker, before daemon stderr redirection | §4.3.2 — claim removed. CLI stderr (the actual user requirement) is guaranteed; `.err.log` capture is explicitly **not** claimed and would need a separate daemon-side hook. Web contract stated as **server-side warning log**; response body/UI unchanged and a non-goal. |
| R2-4 | `unknown-alias-preferred` is misleading — `opus` is valid | — | §4.5.2/§4.5.3 — reason renamed to `preferred-reset-after-invalid-explicit`; rationale added; C1 test asserts the exact string and that `opus`'s message does not contain `unknown-alias`. |

Non-blocking cleanups also applied:
- §4.3.4 — the `deepseek-chat` example was stale (`config.py:1424` pins
  `deepseek-v4-flash`; `deepseek-chat` survives only as a context-window
  key at `config.py:423`). The example no longer hardcodes a value and
  test (e) asserts the runtime lookup instead.
- §7 rollout — now distinguishes explicit replacement (`--model default`,
  changes the value, no warning) from reset (bare `--provider`, drops the
  value, warns).
