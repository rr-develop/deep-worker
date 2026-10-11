# Design: `--restart` must preserve every persisted launch-config field

**Status:** rev 3 (2026-08-14). Awaiting DESIGN + CODE SIGN OFF from the reviewer. rev 3 corrects the affected-field count from three to **six**, replaces the over-promising build-vs-merge guard, de-vacuums the two-restart round-trip test, adds a `--clean` regression, and records the clear-all limitation as an open owner decision (§8 rev 3). rev 2 rewrote `test_restart_preserves_dev_tree` after it was caught being tautological (§8 rev 2).
**Branch:** to be implemented on `dev`, FF to `master` after CODE SIGN OFF.
**Reported by:** an agent user (own MCP config `mcp_config_paths` wiped on every `restart_self`; a manual re-edit is also overwritten on the next restart — they got stuck in this loop twice).
**Explicit owner constraint (translated from the original):** "Configs must not be overwritten on restart, period!!!! Options specified at restart may be added or removed, but otherwise there must be no overwriting whatsoever!!!!"

## 1. Problem

`deep-worker --restart <name>` (used by the `restart_self` tool, and by `--restart` from the shell) silently discards **SIX** persisted launch-config inputs, even when the on-disk config that fed them (`~/.deep_worker/agents/<agent>/mcp.json`) is intact.

Three are wiped to `[]` and are directly visible in `session_metadata.json`:

- `mcp_config_paths` — the reported symptom
- `mcp_server_inline`
- `turn_ending_tools` (shipped in `aa70703`, has the same defect — my own recent work)

One reverts to the parser default, also directly visible:

- `web_search_policy` → `"auto"`, discarding an operator's explicit `enabled` / `disabled`

Two more are lost from the dataflow but **masked on disk**, because `build_launch_config` re-derives them from other state — which is why a naive "read the persisted file" test cannot see them:

- `dev_tree` — re-derived from `_repo_root` (`agent_command.py:704-714`). Masked whenever the process runs from the dev tree, which includes every test run.
- `preferred_model` — re-derived from `model` whenever `provider == preferred_provider` (`agent_command.py:735-740`). The loss is observable, and damaging, only in the **failover shape** `provider != preferred_provider`: while a TaskManager failover has moved the agent onto a fallback provider, the operator's preferred model (`opus`) is what restore-on-idle uses to put them back. Wiping it means restore-on-idle boots the preferred provider on its default model instead.

The masking is the reason the affected-field count was under-reported as three in rev 1–2: the two masked fields do not show up in the obvious symptom, and the tests that would have caught them were written in the shape the masking defeats.

Symptom (verified live 2026-08-14 on a probe agent):

```
first launch  --mcp-config /tmp/wk/mcp.json           -> mcp_config_paths=['/tmp/wk/mcp.json']
deep-worker --restart wk_probe                        -> mcp_config_paths=[]
```

External MCP tools disappear from the graph after every restart. The consumer's workaround — hand-editing `session_metadata.json` — survives exactly one lifetime and is then wiped again.

## 2. Root cause (verified by execution, not by reading)

`restart_agent()` in `deep_worker_cli/cli/agent_command.py` runs a five-step sequence:

1. `saved = load_launch_config(name)` — read `session_metadata.json`.
2. `effective = merge_restart_config(saved, args)` — merge saved + CLI overrides. **This step restores `mcp_config_paths`, `mcp_server_inline`, `turn_ending_tools` from `saved` correctly** (verified).
3. Stop the running agent if needed.
4. Handle `--clean`.
5. **Copy `effective` back into `args` FIELD-BY-FIELD, by an enumerated list, then call `run_agent(args)`.**

The step-5 list currently reads:

```python
args.provider = effective.provider
args.model = effective.model
args.dir = effective.dir
args.skip_permissions = effective.skip_permissions
args.task_manager = effective.task_manager
args.memory_only = effective.memory_only
args.history_memory = effective.history_memory
args.use_sqlite = effective.use_sqlite
args.auto_approve_send_message = effective.auto_approve_send_message
args.clean = False
args.preferred_provider = getattr(effective, "preferred_provider", args.provider)
```

**Six of `build_launch_config`'s inputs are missing from this list.** For those — `mcp_config_paths`, `mcp_server_inline`, `turn_ending_tools`, `web_search_policy`, `dev_tree`, `preferred_model` — `args.*` remains the raw argparse default, even though `effective.*` already holds the correct restored value.

Confirmed by execution against the pre-fix code, failover shape (`saved.provider=deepseek`, `saved.preferred_provider=claude-code-sdk`, `saved.preferred_model=opus`, `saved.web_search_policy=enabled`):

```
build_launch_config(effective):  preferred_model = 'opus'  | web_search_policy = 'enabled'
build_launch_config(args):       preferred_model =  None   | web_search_policy = 'auto'     ← persisted
```

Then `run_agent(args)` calls `build_launch_config(args)`, which honestly writes those empty lists to `session_metadata.json`, overwriting the saved ones. Verified in isolation:

```
effective AFTER merge:   mcp_config_paths = ['/tmp/wk/mcp.json']
args -> run_agent:       mcp_config_paths = []
build_launch_config():   ['/tmp/wk/mcp.json'] -> []   ← persisted
```

Asymmetry inside the same file makes the defect obvious once you see it:

- Fork path (`agent_command.py:907`) calls `build_launch_config(effective)`.
- Restart path (`agent_command.py:1819`) calls `build_launch_config(args)`.

The fork path is correct; the restart path relies on a hand-written copy-back list that is not the complete set of fields `build_launch_config` reads.

## 3. Design choice

Two families of fix. The owner's constraint ("there must be no overwriting whatsoever" plus "options specified at restart may be added or removed") rules out anything that requires human maintenance of a copy-back list, because the class of failure is «someone added a persisted field and forgot to add it to the list». Any fix that preserves that list is one refactor away from the same bug reappearing.

### Option A (chosen): pass `effective` straight to `run_agent`, drop the copy-back

`run_agent(args)` currently reads `args.provider`, `args.model`, `args.dir`, `args.mcp_config_paths`, etc. Nothing about it needs the ORIGINAL user-typed `args`; it needs the EFFECTIVE, post-merge namespace. Pass that.

```python
# in restart_agent(), replace steps 5:
return await run_agent(effective)
```

That is the whole fix at the call site. `merge_restart_config` already produces a full `Namespace` with every field `run_agent` consumes, plus every field `build_launch_config` persists — because that is exactly what it was designed to do. The copy-back was a workaround dating from before `merge_restart_config` covered everything.

Consequence: `build_launch_config` will be called with a namespace where all persisted fields are correct, INCLUDING future fields nobody has invented yet. The class of bug goes away, not just this instance.

### Option B (rejected): extend the copy-back list

Adding six lines fixes today's symptom for today's six fields — and the fact that the list was under-reported as three until review (§8 rev 3) is itself evidence of how badly this option scales: the maintenance list is exactly as easy to get wrong when *fixing* the bug as it was when introducing it. The next field added to `build_launch_config` — someone building yet another CLI switch — is one omission away from the same silent-wipe. The user's constraint is a class-level one ("no overwriting, period"); Option B is instance-level.

### Option C (rejected): make `build_launch_config` merge with the on-disk saved config

Would fix the persistence side but leave `run_agent` seeing empty lists (so external MCP tools would still not be started, even if the file kept the paths). Consumers care about behavior, not just the file. Also, it hides the class-of-bug — someone reading `restart_agent` would still see a suspicious copy-back and rediscover the defect on the next persisted field.

### Consequence of Option A: one asymmetry inside `merge_restart_config` to verify

`merge_restart_config` sets `effective.dir` as a `Path` object (`agent_command.py:379`, `Path(saved["working_dir"])`). `run_agent`'s `args.dir` handling accepts `Path`, so this is already the shape `run_agent` receives on the fork path via `effective`. Verified in `test_agent_command_working_dir.py` (fork path). No regression expected.

## 4. Testing (TDD, RED first)

New file: `tests/unit_tests/cli/test_restart_preserves_launch_config.py`. Every test drives `restart_agent()` (not just `merge_restart_config` — that is what the existing shipped tests do, and it is why the defect slipped through) and asserts against `session_metadata.json` written by the second-life `run_agent()`.

**Loss-of-field tests — one per known persisted field (RED without the fix):**

- `test_restart_preserves_mcp_config_paths` — first launch with `--mcp-config /path/a.json`; then simulate `deep-worker --restart <name>` (no flags); assert persisted `mcp_config_paths == ["/path/a.json"]`, not `[]`.
- `test_restart_preserves_mcp_server_inline` — same, with `--mcp-server 'srv={...}'`.
- `test_restart_preserves_turn_ending_tools` — same, with `--turn-ending-tool foo:bar`. This one is the regression on my own aa70703 work.
- `test_restart_preserves_dev_tree` — dev-tree flag has been touched by past incidents (2026-06-18). MASKED field: asserted at the `run_agent` seam, before `build_launch_config` re-derives it from `_repo_root` (§8 rev 2).
- `test_restart_preserves_web_search_policy` — saved `enabled` must not revert to the parser default `auto`.
- `test_restart_preserves_preferred_model_under_failover` — MASKED field. Written deliberately in the failover shape `provider != preferred_provider` (running on `deepseek`, preferring `claude-code-sdk`/`opus`), because that is the only shape where the loss is observable at all: with `provider == preferred_provider`, `build_launch_config` re-derives `preferred_model` from `model` and the test would pass regardless of the defect. Same masking class as `dev_tree`.

**CLI-add / CLI-remove semantics (owner's "options specified at restart may be added or removed"):**

- `test_restart_flag_replaces_saved_mcp_config_paths` — saved has `["/a.json"]`; restart passes `--mcp-config /b.json`; result is `["/b.json"]` (replace-wholesale, as `--mcp-config` already documents). Guards that the fix did not accidentally make CLI values additive.
- `test_restart_flag_replaces_saved_turn_ending_tools` — same for `--turn-ending-tool`.
- **Clear-all is NOT possible today — stated plainly, not glossed.** The list flags replace wholesale, so a restart can remove *selected* saved entries by passing a new NON-EMPTY list that omits them. It cannot empty a list: passing no flag INHERITS the saved list (that is the whole point of this fix), and there is no `--no-mcp-config` / `--no-turn-ending-tool` switch to express "clear it". So an operator who wants zero MCP configs after having had one has no CLI path; hand-editing `session_metadata.json` is the only route, and that is the very workaround this fix makes durable rather than eliminating. This is an **open owner decision**, recorded in §6 — not something this urgent preservation fix resolves.

**`--clean` one-shot semantics (regression guard):**

- `test_restart_clean_runs_once_and_is_not_persisted` — Option A introduces a specific hazard the copy-back did not have. `merge_restart_config` sets `effective.clean = True` so step 4 fires; passing `effective` straight through would then PERSIST `clean=True`, and every subsequent restart would silently wipe the agent's history. The product code resets `effective.clean = False` immediately before `run_agent`; this test pins both halves — cleanup invoked exactly once, and the rebuilt `launch_config` carries `clean=False`.

**Class-of-bug guard — the point of Option A over Option B:**

- `test_no_field_by_field_copy_back_in_restart_agent` — AST guard. Parse `restart_agent`'s function body; assert it does NOT contain `args.X = effective.X` assignments (or that the count is <= 1, for a single legitimate copy if the reviewer insists). This test is what prevents the fix from being reversed by a future refactor that «restores clarity» by putting the list back.

- `test_every_build_launch_config_input_is_handled_by_merge_restart_config` — structural guard, **rewritten in rev 3**. Two AST scans: every `args.X` / `getattr(args, "X", …)` that `build_launch_config` READS, and every `effective.X = …` that `merge_restart_config` ASSIGNS. Assert the first set is a subset of the second.

  What it proves and what it does not — stated precisely, because the rev 2 version over-promised:
  - **Proves (structural):** no field can be added to `build_launch_config` and silently inherit whatever the bare CLI namespace holds, because `merge_restart_config` is forced to make a deliberate decision about it — restore from `saved`, or explicitly reset (`clean` is assigned `False` on purpose and is therefore covered, not excluded). There is no hand-written list of known fields and no skip-list, so unlike rev 2 it cannot silently pass over a field its author forgot. Mutation-verified: adding a hypothetical `"new_flag": getattr(args, "new_flag", None)` to `build_launch_config` without touching `merge_restart_config` turns it red with `uncovered == ['new_flag']`.
  - **Does NOT prove (semantic):** that the assignment restores the RIGHT value. A future contributor could assign `effective.new_flag = None` and satisfy this guard. That layer is covered by the six per-field behavioral `restart_agent` tests, one for each currently persisted field. The two layers are complementary; neither is claimed to subsume the other.

  The rev 2 version claimed future-field protection it did not deliver: it built a hand-written `saved` dict and did `if field not in saved: continue`, so a new persisted field missing from that dict was silently skipped — the same maintenance-list class the fix is meant to eliminate — and it excluded `preferred_model`, one of the six genuinely lost fields. See §8 rev 3.

**Behavioural round-trip on the real code path:**

- `test_two_consecutive_restarts_preserve_mcp_config_paths` — first launch → save → restart 1 (no flags) → restart 2 (no flags) → `mcp_config_paths` still original. This is the exact loop the reporter got stuck in.

  **De-vacuumed in rev 3.** The rev 2 form installed a spy whose `start()` was a no-op, so `session_metadata.json` was never rewritten and BOTH restarts re-read the original seed file. The second restart therefore could not fail independently of the first, and the test proved nothing about restart-over-restart behaviour — precisely the property the original bug report is about. The spy's `start()` now persists the captured `launch_config` to `session_metadata.json`, mirroring what the real `AgentProcess.start()` does via `SessionManager(data_dir=…, launch_config=…)` (`session_manager.py:100-107, 226-227`). The test additionally asserts the file really was rewritten by restart 1 (fresh `pid`, versus the seed's `1234`), so the round-trip cannot silently degrade back into a seed re-read.

  Independently verified non-vacuous: corrupting ONLY restart-1's persisted output (setting `mcp_config_paths` to `[]` between the two restarts) makes restart 2 observe `[]`, whereas with restart-1's output left clean restart 2 observes `['/etc/mcp.json']`. With a no-op `start()` both branches would have returned the seed value.

## 5. Risk / rollback

- **Risk 1: `effective` differs in shape from the raw `args` in some field `run_agent` does not currently consume from `effective`.** `merge_restart_config` builds `effective = argparse.Namespace(**vars(args))` (`agent_command.py:368`) and then overrides fields, so it is a strict superset of `args` — no field of `args` is missing from `effective`. Fields added by `merge_restart_config` that `run_agent` did not previously see: `preferred_provider`, `preferred_model`, `dir` as Path, all the persisted lists. `run_agent` already reads these in the fork path. Verified.
- **Risk 2: The consolidated one-liner hides which fields flow through.** Mitigated by the AST guard test above, plus a comment at the call site pointing to `merge_restart_config` as the single source of truth for the restart-effective namespace.
- **Risk 3: Existing agents on disk have `session_metadata.json` where the affected fields are already wiped.** After the fix, a first restart without CLI flags will keep them at their wiped values (nothing to restore). Consumers must reintroduce the flag ONCE (`--mcp-config /path` on the next restart); from then on, it persists. Documented in the fix commit. Note this applies to `web_search_policy` too: agents restarted under the defect have `"auto"` persisted and must re-pass `--web-search-policy` once if they had a non-default one.
- **Risk 4 (introduced by Option A): `--clean` becomes persistable.** Under the copy-back, `args.clean` was hard-set to `False`. Passing `effective` through would persist `clean=True` and make every later restart wipe history. Mitigated by an explicit `effective.clean = False` before `run_agent`, pinned by `test_restart_clean_runs_once_and_is_not_persisted`. This is the one place where Option A required a deliberate line rather than removing one.
- **Rollback:** revert the single fix commit. No schema change, no data migration.

## 6. Owner questions — ONE OPEN

- Owner constraint recorded verbatim in §1. Option A honours the preservation half of it in full: nothing is overwritten unless the operator passed the corresponding option on the restart command line.

- **OPEN — explicit clear-all has no CLI expression.** The owner's constraint has two halves; the second is "Options specified at restart may be added or removed". Option A fully delivers *add* and *replace*, and delivers *remove-selected* (pass a new non-empty list omitting the unwanted entries). It does **not** deliver *remove-all*: with the list flags' wholesale-replace semantics, passing nothing means INHERIT, so there is no way to say "I want zero MCP configs now". Emptying a list currently requires hand-editing `session_metadata.json`.

  I am **not** extending this fix to cover it: adding `--no-mcp-config` / `--no-turn-ending-tool` (or a sentinel such as `--mcp-config ''`) is new CLI surface with its own naming, interaction and precedence questions, and the reporter's agent is broken today. But it must not be recorded as settled, because the owner's wording may well have intended clear-all when it said "removed".

  **Decision needed from the owner:** is inherit-unless-replaced sufficient, or is an explicit clear switch required? If required, it is a separate design + change on top of this one. Until answered, this section stays open — §6 previously read "none open", which was wrong.

## 7. Implementation order (TDD)

1. RED: write all tests in `test_restart_preserves_launch_config.py`. Loss-of-field tests fail on current `dev`, the AST/introspection guards fail because they exercise the same failing behaviour.
2. GREEN: replace step 5 of `restart_agent()` with `return await run_agent(effective)`. Delete the copy-back list. Add a comment pointing to `merge_restart_config` as the single source of truth.
3. Full suite `env -u PYTHONPATH python3 -m pytest tests/unit_tests -q` on the current baseline (whatever it is at HEAD `b419354`). Zero regressions required.
4. CODE REVIEW REQUEST → CODE SIGN OFF → single commit to `dev` → FF to `master`.
5. Notify the reporter. They then run one restart with `--mcp-config` on their agent to reintroduce the path, and every subsequent restart keeps it.

## 8. Corrections log

### rev 3 (2026-08-14) — five reviewer findings (DESIGN + CODE CHANGES REQUESTED)

**1. Affected scope was under-reported: three fields, actually six.** `web_search_policy` and `preferred_model` are also dropped by the copy-back. I missed them because I derived the field list from the reported symptom (empty lists in `session_metadata.json`) instead of from `build_launch_config`'s actual input set. `preferred_model`'s loss is masked whenever `provider == preferred_provider`, so even a test would have hidden it unless written in the failover shape. §1, §2, §4 corrected; the product-code comment at the call site corrected from three fields to six with the masking noted. Two behavioral tests added, the `preferred_model` one deliberately in the failover shape (verified pre-fix: `'opus' → None`).

**2. The two-restart round-trip test was vacuous.** Its spy's `start()` was a no-op, so `session_metadata.json` was never rewritten and both restarts re-read the same seed file — the second restart could not fail independently of the first. Fixed by making the spy persist the captured `launch_config`, mirroring the real `SessionManager` write. Independently verified non-vacuous by corrupting only restart-1's output and confirming restart 2 observes the corruption. A guard assertion (fresh `pid` after restart 1) now prevents silent regression back into a seed re-read.

**3. The generic build-vs-merge guard over-promised.** It advertised protection against future persisted fields while doing `if field not in saved: continue` against a hand-written `saved` dict — so a new field missing from that dict was silently skipped, reproducing the very maintenance-list class of bug the fix removes. It also excluded `preferred_model`, one of the six real losses. Replaced with a pure AST subset check (`args` attrs read by `build_launch_config` ⊆ `effective` attrs assigned by `merge_restart_config`) which needs no hand-maintained list, and its docstring now separates the structural property it proves from the semantic one it does not. Mutation-verified against a hypothetical new field.

**4. `--clean` regression test was promised but not written.** Added. It matters specifically because Option A *introduces* the hazard: unlike the copy-back, passing `effective` through would persist `clean=True`. Recorded as Risk 4 in §5.

**5. §6 claimed "none open" while clear-all was unresolved.** The owner's "removed" may mean explicit clear-all, which is impossible today (inherit-unless-replaced cannot express an empty list). §4 and §6 now state this plainly and record it as an open owner decision rather than a settled point. The urgent fix is deliberately not extended to cover it.

Lesson worth pinning, common to findings 1–3: each was a case of a claim being broader than the evidence behind it — a field list derived from a symptom rather than from the code's input set, a round-trip test whose second half could not fail, and a guard advertising future-proofing it did not implement. "The suite is green" was true in all three cases and diagnostic in none of them.

### rev 2 (2026-08-14) — TaskManager finding on `test_restart_preserves_dev_tree`

The rev 1 form of this test read the persisted `session_metadata.json` and asserted `dev_tree=True`. It passed BEFORE the fix, which is what triggered the review question.

Root cause: `build_launch_config` re-derives `dev_tree` from `_repo_root` (the running code tree) as a THIRD source, beyond `args.dev_tree` and `DEEP_WORKER_DEV_TREE`. Tests are executed from the dev tree checkout, so `dev_tree=True` on disk was inevitable regardless of what `restart_agent` handed to `run_agent`. The test was tautological — a textbook test-theater case.

Mutation-verified: replacing `effective.dev_tree = saved.get("dev_tree", False)` with a hard `False` in `merge_restart_config` did NOT turn the rev 1 test red (the `_repo_root` re-derivation masked the mutation). So the test was measuring the tree the interpreter was running from, not the CLI dataflow it claimed to guard.

Fix: the test now monkey-patches `agent_command.run_agent` and asserts the `dev_tree` attribute on the `effective` namespace `restart_agent` hands to it — the single observable point in the CLI dataflow BEFORE `build_launch_config` re-derives the value. Re-run of the same mutation now turns the test red (verified). Restored code, verified green.

Lesson worth pinning: if a persistence layer has a documented "recompute from environment" fallback, a "read the persisted value" assertion cannot prove the dataflow you want to guard reached that layer with the right input. Assert at the seam before the fallback runs.
