# Design: Web-Restart Preserves ALL Persisted Boolean Options (opt-OUT, not opt-IN)

**Status:** Draft
**Date:** 2026-06-05
**Related:**
- `deep_worker_web/app/routers/agents.py::_spawn_subprocess`
- `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py::merge_restart_config`
- `deep_worker/deep_worker_cli/docs/LAUNCH_CONFIG_PERSISTENCE.md`
- `deep_worker/deep_worker_cli/docs/DESIGN_NO_FLAGS_FOR_RESTART.md`

---

## 1. Problem

When the web UI restarts an agent, persisted boolean launch options are silently dropped.

Concrete symptom (2026-06-05): a live agent was running with `--history-memory` (`-m`). After a web-UI restart, `launch_config.history_memory` became `false`, the `history_memory` startup catch-up never ran, and the index stayed stale — despite the user never disabling the option.

### 1.1 Root cause

`deep_worker_web/app/routers/agents.py::_spawn_subprocess` (lines 671-697) hand-builds the relaunch CLI as a sequence of opt-in conditionals:

```python
cmd = ["python3", "-m", "deep_worker_cli.cli.agent_command", agent_name,
       "--provider", ..., "--dir", ...]
if config.get("skip_permissions"): cmd.append("-sp")
if config.get("task_manager"):     cmd.append("-tm")
if config.get("memory_only"):      cmd.append("--memory-only")
if config.get("use_sqlite"):       cmd.append("--use-sqlite")
if clean:                          cmd.append("--clean")
```

Notably absent: `history_memory`, `auto_approve_send_message`, `dev_tree` (the last is partially handled via env var, but the `--dev` CLI flag is NOT re-emitted).

Because the CLI never sees `-m`, the spawned process treats `history_memory` as "not given" → parser default `False` → that `False` gets persisted into the new `launch_config`, overwriting the previous `True`. Same shape for any other boolean.

### 1.2 Why this keeps regressing

This is an **opt-in registry**: every new boolean must be remembered in TWO independently-maintained places:
1. `agent_command.py::merge_restart_config` (for CLI `--restart`)
2. `agents.py::_spawn_subprocess` (for web restart)

Add one and forget the other → silent regression. We have the symmetric problem in `merge_restart_config` itself: each option needs (a) a `saved.get(...)` line, (b) an `if _is_override(...)` line. There is no compile-time / test-time link from "this option exists in `_PARSER_DEFAULTS`" to "it is preserved across restart".

---

## 2. Goal

1. Preserve **every** persisted boolean across restart **by default**, regardless of which path (CLI or web) drove the restart.
2. Make adding a new boolean a **single-list change**; the rest follows automatically.
3. Make a **forgotten preservation** a hard test failure, not a runtime regression that surfaces months later.

Out of scope:
- Changes to `merge_restart_config` semantics. It is already opt-out correct in spirit (every key has BOTH a `saved.get(...)` and an `_is_override(...)` line). The fix here is to **derive** those lines from a shared registry so the link is enforced.
- Web `RestartRequest` schema changes for new per-request overrides — that is a separate feature, not a regression fix.

---

## 3. Design

### 3.1 Spawn paths inventory

| # | Path | Code | Currently opt-in? | Fix required |
|---|------|------|---|---|
| 1 | CLI `deep-worker --restart NAME` | `agent_command.py::merge_restart_config` | Soft (correct today, no enforcement) | Derive from registry + property test |
| 2 | Web "Restart" button | `agents.py::_spawn_subprocess` | **Yes — bug** | Iterate registry |
| 3 | `restart_agent` LLM tool | `tools.py::make_restart_agent_tool` → shells out to `deep-worker --restart` | No — delegates to path #1 | None (covered by #1) |
| 4 | Internal fix service | Does NOT spawn agents (verified via grep over its source) | n/a | None |
| 5 | Internal tasks service | Does NOT spawn agents (verified via grep over its source) | n/a | None |
| 6 | Bash launcher `deep-worker` | Dispatches to `python3 -m deep_worker_cli.cli.agent_command "$@"` — verbatim arg forwarding | n/a | None |

Only paths #1 and #2 need code changes. Path #6 (bash) is verbatim-forwarding; nothing to preserve there.

### 3.2 The `_BOOLEAN_FLAGS` registry

Single source of truth, placed in `agent_command.py` next to `_PARSER_DEFAULTS`:

```python
# (dest_name, cli_flag_to_pass_when_true, cli_flag_to_pass_when_false_or_None)
# The "false" form is the --no-* counterpart for restart-time disabling.
_BOOLEAN_FLAGS: tuple[tuple[str, str, str | None], ...] = (
    ("skip_permissions",          "-sp",                       "--no-skip-permissions"),
    ("task_manager",              "-tm",                       "--no-tm"),
    ("auto_approve_send_message", "-asm",                      "--no-auto-approve-send-message"),
    ("memory_only",               "--memory-only",             "--no-memory-only"),
    ("history_memory",            "-m",                        "--no-m"),
    ("use_sqlite",                "--use-sqlite",              "--no-use-sqlite"),
    ("dev_tree",                  "--dev",                     "--no-dev"),
)
```

Explicitly **NOT** preserved (the "exclusions" the user mentioned):

```python
_NON_PRESERVED_BOOLEAN_OPTIONS: frozenset[str] = frozenset({
    "clean",                # Never inherited — explicit per-launch action, not a persistent mode.
    "no_update_preferred",  # One-shot failover flag, not a persistent mode.
})
```

`clean` is documented in `merge_restart_config` (line 229: `effective.clean = False  # NEVER inherit --clean`). `no_update_preferred` is a per-invocation failover override (see DESIGN_PROVIDER_FAILOVER_AND_PREFERRED_PROVIDER.md) — restarting an agent must not silently re-apply a previous one-shot intent. Listing both in `_NON_PRESERVED_BOOLEAN_OPTIONS` makes every exclusion auditable.

### 3.3 Coverage rule (the regression guard)

Property held by tests:

> Every boolean key in `_PARSER_DEFAULTS` is EITHER listed in `_BOOLEAN_FLAGS` (preserved) OR in `_NON_PRESERVED_BOOLEAN_OPTIONS` (explicitly excluded). No third option.

Concretely:

```python
def test_every_boolean_default_is_classified():
    bool_keys = {k for k, v in _PARSER_DEFAULTS.items() if isinstance(v, bool)}
    preserved = {dest for dest, _t, _f in _BOOLEAN_FLAGS}
    classified = preserved | _NON_PRESERVED_BOOLEAN_OPTIONS
    missing = bool_keys - classified
    assert not missing, (
        f"Boolean option(s) {missing} added to _PARSER_DEFAULTS but neither "
        f"listed in _BOOLEAN_FLAGS (to preserve across restart) nor in "
        f"_NON_PRESERVED_BOOLEAN_OPTIONS (to explicitly exclude). "
        f"This will cause a silent restart-preservation regression."
    )
```

Adding a new boolean to `_PARSER_DEFAULTS` without classifying it → CI red. That is the structural guarantee the user demanded.

### 3.4 `_spawn_subprocess` becomes a loop over the registry

```python
from deep_worker_cli.cli.agent_command import _BOOLEAN_FLAGS  # single import

# ... after assembling cmd's name / provider / dir / model:
for dest, true_flag, _false_flag in _BOOLEAN_FLAGS:
    if config.get(dest):
        cmd.append(true_flag)
# We only emit the "true" form. A persisted False is the parser default;
# omitting the flag is correct. The "false" forms are for users who actively
# disable a saved option via --no-X on the CLI; web restart never disables.
```

The hard-coded `if config.get("skip_permissions"): cmd.append("-sp")` block is removed and replaced by the loop. `clean` stays as a separate explicit parameter — matches current call signature (`clean=False` arg, only appended when caller passes `True`).

### 3.5 Edge cases

| Case | Behavior |
|---|---|
| `config.get(dest)` returns `None` (key missing) | Falsy → flag not appended. Identical to current behavior. |
| `config.get(dest)` returns `False` | Falsy → flag not appended. Correct. |
| `config.get(dest)` returns `True` | Flag appended. Fix realized. |
| New boolean added to `_PARSER_DEFAULTS`, forgotten in `_BOOLEAN_FLAGS` | Coverage test fails. CI red. |
| New boolean intentionally excluded (another `clean`-like) | Author adds it to `_NON_PRESERVED_BOOLEAN_OPTIONS`. Coverage test passes. The intent is now visible in code review. |
| Web `RestartRequest` (`model`/`provider`/`clean`/`grace_seconds`/`force`) | Untouched. `model`/`provider`/`clean` are per-request overrides (passed as explicit args to `_spawn_subprocess`); `grace_seconds`/`force` belong to the stop phase, not the spawn. |
| `dev_tree` env-var path | `_spawn_subprocess` already sets `DEEP_WORKER_DEV_TREE=1` based on `config.get("dev_tree")`. Emitting `--dev` on the CLI in addition is harmless (the bash launcher filters `--dev` out before forwarding; the python parser accepts it). Decision: emit `--dev` via the registry so the CLI line is self-describing in logs / `cli_args`. |
| `clean` per-request | Untouched. `clean` is in `_NON_PRESERVED_BOOLEAN_OPTIONS`, so the loop never reads `config.get("clean")`. The explicit `clean` parameter of `_spawn_subprocess` still appends `--clean` when caller passes `True`. |
| `RestartRequest.clean=True` interaction | Web route reads `RestartRequest.clean`, forwards it as the explicit `clean=` kwarg to `_spawn_subprocess`. Persisted `config.clean` (almost always `False` per `merge_restart_config` line 229) is never read by the loop. No double-emit, no conflict. |
| `_BOOLEAN_FLAGS` order changes | Order matters only for log readability of the CLI line. Registry tuple → ordered iteration → deterministic. |

### 3.6 Why a registry and not BooleanOptionalAction

Already addressed in `DESIGN_NO_FLAGS_FOR_RESTART.md` §2: `argparse.BooleanOptionalAction` does not work with short aliases (`-sp`, `-tm`, `-m`, `-asm`). We keep `store_true` / `store_const False` parser definitions and bridge them via the registry.

### 3.7 None vs False in saved config

`session_metadata.json` always serializes booleans as JSON `true`/`false`, so `config.get(dest)` returns `True` / `False`. The `None` case can only arise when:
- Legacy metadata predates the option (key absent → `.get(dest)` returns `None`).
- The metadata file was hand-edited.

Both cases are handled identically by `if config.get(dest):` — falsy → flag omitted → parser default `False` applies. Behavior is equivalent to "option was not enabled", which is the safe default for a legacy / partial config.

---

## 4. Test Plan

All tests new; no existing tests modified. Two files:

### 4.1 `deep_worker/deep_worker_cli/tests/unit_tests/cli/test_boolean_flags_registry.py`

Property-level guard (these are the tests that prevent future regressions):

1. **`test_every_boolean_default_is_classified`** — the coverage rule from §3.3. Adding a new bool to `_PARSER_DEFAULTS` without listing it → fails. This is THE regression guard.
2. **`test_preserved_and_excluded_sets_are_disjoint`** — `{dest for dest in _BOOLEAN_FLAGS}` ∩ `_NON_PRESERVED_BOOLEAN_OPTIONS` = ∅. Catches a typo that would put the same option in both lists.
3. **`test_registry_dests_exist_in_parser_defaults`** — typo guard: every registry entry's `dest` exists in `_PARSER_DEFAULTS`.
4. **`test_registry_true_flag_is_known_to_parser`** — for every `(dest, true_flag, _)`, the real `create_parser()` accepts that flag and stores `True` under `dest`.
5. **`test_registry_false_flag_is_known_to_parser`** — for every `(dest, _, false_flag)` where `false_flag is not None`, the real `create_parser()` accepts that flag and stores `False` under `dest`.

### 4.2 `deep_worker_web/tests/unit/test_spawn_subprocess_preserves_booleans.py`

End-to-end behavior (follows existing `_spawn_subprocess` test conventions — mock `subprocess.Popen` + `builtins.open`):

6. **`test_spawn_preserves_history_memory`** — `config={"history_memory": True, ...}` → spawn cmd contains `-m`. Regression for the actual 2026-06-05 bug.
7. **`test_spawn_preserves_auto_approve_send_message`** — `-asm` forwarded when persisted.
8. **`test_spawn_preserves_dev_tree_via_cli_flag`** — `--dev` appears in spawn cmd (in addition to the env-var path that already exists).
9. **`test_spawn_does_not_emit_clean_from_saved_config`** — `config={"clean": True}` MUST NOT emit `--clean`. The registry skips it; only the explicit `clean=` argument controls emission.
10. **`test_spawn_emits_clean_only_when_explicit_argument`** — `_spawn_subprocess(..., clean=True)` → `--clean` in cmd; `_spawn_subprocess(..., clean=False)` → `--clean` absent.
11. **`test_spawn_omits_flag_when_value_is_false`** — falsy values don't append flags.
12. **`test_spawn_omits_flag_when_key_missing`** — `None` / missing keys behave identically to `False`.
13. **`test_spawn_parametrized_for_every_registry_entry`** — `pytest.mark.parametrize` over `_BOOLEAN_FLAGS`: each entry verified end-to-end with `value=True` and `value=False`. New boolean added to the registry → parametrize picks it up automatically → no per-flag test maintenance.

### 4.3 Regression suite gate

Full `deep_worker_cli` + `deep_worker_web` unit suites must remain green. Existing `test_spawn_subprocess_dev_tree.py` and `test_spawn_subprocess_model.py` exercise the same call site and must keep passing without modification — the loop replaces the conditionals 1-for-1 in behavior for the flags they cover.

---

## 5. Files Modified / Added

| File | Change |
|---|---|
| `deep_worker/deep_worker_cli/deep_worker_cli/cli/agent_command.py` | Add `_BOOLEAN_FLAGS` and `_NON_PRESERVED_BOOLEAN_OPTIONS` constants near `_PARSER_DEFAULTS`. No behavior change in `merge_restart_config` — the existing opt-out logic stays as-is; the registry just becomes the documented source of truth. |
| `deep_worker_web/app/routers/agents.py` | Replace the 5 hard-coded `if config.get(...)` lines in `_spawn_subprocess` with a single `for dest, true_flag, _ in _BOOLEAN_FLAGS` loop. The `clean` handling stays separate. |
| `deep_worker/deep_worker_cli/tests/unit_tests/cli/test_boolean_flags_registry.py` | NEW. 5 property tests (§4.1). |
| `deep_worker_web/tests/unit/test_spawn_subprocess_preserves_booleans.py` | NEW. 8 behavior tests (§4.2). |

Net: 2 prod files touched, 2 test files added, ~30 LOC of prod change, ~150 LOC of tests.

---

## 6. Resolved Decisions

1. **Registry shape: tuple of tuples vs. dict.** Tuple of tuples — ordered iteration matters for log readability of `cli_args`, and we want the registry to be append-only (adding entries at the end keeps existing CLI lines visually stable in logs).
2. **`--no-*` form stored in the registry.** Even though `_spawn_subprocess` only emits the `True` form, recording the `False` form in the same place keeps both halves of "this is a preserved boolean" together and lets a future CLI-introspection tool list them.
3. **`clean` excluded explicitly.** Adding it to `_NON_PRESERVED_BOOLEAN_OPTIONS` rather than just relying on its absence from `_BOOLEAN_FLAGS` — the coverage test (§3.3) demands a positive classification for every bool, which forces a code-review moment when a future option also needs exclusion.
