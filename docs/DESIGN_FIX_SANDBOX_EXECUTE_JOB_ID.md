# DESIGN: the fourth source of `job_id` — proxy-MCP providers park under `toolu_…`

Status: **DESIGN SIGNED OFF** (2026-07-31) by the reviewer — the DESIGN is
approved, but not the start of work.
**Do not start implementation until the user approves the priority** (a direct
instruction in the task statement; the reviewer confirmed this constraint).
Branch `dev`. Reviewer's decisions — §10.

Related documents (the same defect class, three already-closed sources):
`docs/DESIGN_FIX_JOB_ID_NAMESPACE_STALL.md` (sources 1–2, commit `6bd042b`),
`docs/DESIGN_FIX_STAGE_A_FALSE_PROMISE.md` (source 3, commit `7074243`),
`docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md` (invariants of the single wait point).

---

## 1. Symptom: verbatim from the current session

The defect reproduced **twice in one working day**, after the three previous
sources had already been closed and merged into `master`.

First occurrence:

```
execute(command="cd /tmp && timeout 600 python3 /tmp/klsp_probe.py 180 2>&1 | tail -80")
→ Tool 'execute' is still running after 120s. status: in_progress.
  job_id: toolu_01N7dZDYtTgfao5CafcDfTzD.
  actions: wait_for_job(job_id="toolu_01N7dZDYtTgfao5CafcDfTzD", timeout=<seconds>)
  or kill_job(job_id="toolu_01N7dZDYtTgfao5CafcDfTzD").
```

Here `wait_for_job` unexpectedly **worked**, but answered with a foreign identifier:

```
wait_for_job(job_id="toolu_01N7dZDYtTgfao5CafcDfTzD", timeout=180)
→ Job 'job_1' (tool 'execute') finished.
```

Second occurrence, same day, same tool:

```
execute(command="sleep 120; cat /tmp/xfile2.out")
→ Tool 'execute' is still running after 120s. status: in_progress.
  job_id: toolu_012BF1Js6reQLQU5cFmwPuu4. actions: wait_for_job(...)

wait_for_job(job_id="toolu_012BF1Js6reQLQU5cFmwPuu4", timeout=120)
→ Error: no background job with id 'toolu_012BF1Js6reQLQU5cFmwPuu4'.
```

So the behavior is **nondeterministic**: sometimes the id resolves via the
alias, sometimes it does not. For the model this is the worst case — the
contract looks functional and fails selectively.

## 2. Why this is NOT a regression of the three previous fixes

The first suspicion — "the process was started on old code". Checked and **rejected**:

```
$ ps -o lstart= -p 677168
Fri Jul 31 16:07:50 2026

$ git log -3 --format="%h %cd %s" --date=format:'%a %b %d %H:%M:%S %Y'
7074243 Fri Jul 31 16:00:12 2026 fix(providers): stage-A timeout ...
6bd042b Fri Jul 31 14:58:59 2026 fix(jobs): emit a real job_N ...
b568a6d Fri Jul 31 12:50:37 2026 fix(lsp): remove false reportMissingImports ...
```

The process started at 16:07:50 — **7 minutes later** than the latest commit.
The code in the working worktree also contains the fix:

```
$ cd <repo-root>    # the master checkout that the agent's PYTHONPATH points to
$ ls deep_worker/deepagents/deepagents/middleware/parked_job_registry.py
deep_worker/deepagents/deepagents/middleware/parked_job_registry.py
$ grep -c "ParkedJobRegistry\|build_dispatch_timeout_notice" \
     deep_worker/deepagents/deepagents/middleware/bounded_tool_call.py
7
```

So `toolu_` was emitted **not** by `BoundedToolCallMiddleware`. This is a
separate, fourth source.

## 3. Root cause

Searching for all producers of this phrase in the code yields four points:

```
$ grep -rn "is still running after" --include=*.py . | grep -v /tests/
./deep_worker/deepagents/deepagents/middleware/bounded_tool_call.py:129   ← fixed (6bd042b)
./claude_code_sdk_provider/proxy_mcp_server.py:543                            ← NOT fixed
./codex_provider/proxy_mcp_server.py:329                                  ← NOT fixed
./grok_provider/proxy_mcp_server.py:380                                   ← NOT fixed
```

The three provider proxy-MCP servers park work **in their own registry** and
advertise `call_id` (that is, `toolu_…`) as the `job_id`.

`claude_code_sdk_provider/proxy_mcp_server.py:507-554`:

```python
except asyncio.TimeoutError:
    # Park the live Future (work is NOT cancelled) ...
    self._jobs[call_id] = {                      # ← key = toolu_…
        "future": future,
        "tool_name": tool_name,
        "args": args,
        "agent_name": self._agent_name,
    }
    ...
    # Default job-handle.
    return {"content": [{"type": "text", "text": (
        f"Tool '{tool_name}' is still "
        f"running after {wait_seconds:g}s. "
        f"status: in_progress. "
        f"job_id: {call_id}. "                   # ← toolu_… goes to the model
        f"actions: wait_for_job(job_id="
        f"\"{call_id}\", timeout=<seconds>) "
        f"or kill_job(job_id=\"{call_id}\")."
    )}]}
```

The same defect in the second message of the same file (the
`mode == "timeout_notice"` branch, lines 519-536).

**Here `self._jobs` is a third, isolated registry.** It is declared as
`self._jobs: dict[str, JobHandle] = {}` (`:243`) and has nothing in common with
either `CLIShellBackend._jobs` or the process-wide `ParkedJobRegistry`. And
`wait_for_job` / `kill_job` resolve the identifier through exactly two points:

* `CLIShellBackend._resolve_job` — its own registry + `toolu_→job_N` aliases;
* `CLIShellBackend._resolve_parked` → `get_parked_job_registry().resolve(...)` —
  the shared `ParkedJobRegistry`.

The proxy-MCP server's registry is visible to **neither** of them.

### 3.1. Where the nondeterminism from §1 comes from

Both layers have the same 120 s bond and start almost simultaneously, so the
outcome depends on which of them fired first:

| Who fired first | What happened |
|---|---|
| `CLIShellBackend` managed to create `job_N` and the `toolu_→job_N` alias | `wait_for_job("toolu_…")` resolves via the alias → the answer "Job 'job_1' … finished" (first case in §1) |
| proxy-MCP emitted the notice before the backend created `job_N` | the alias is in neither registry → `no background job` (second case in §1) |

This is exactly the same race class described in
`DESIGN_FIX_JOB_ID_NAMESPACE_STALL.md` §2 — but on a layer **not covered
then**.

## 4. Why the source was not closed together with the three previous ones

The `6bd042b` fix was point-specific: it brought the parking in
`BoundedToolCallMiddleware` to `job_N`, and `7074243` removed the false
job-handle promise in the providers' stage-A (`chat_model.py`). Neither of
them touched `proxy_mcp_server.py`, where a **standalone** parking lives.

Process takeaway: a point-specific edit to "the place where the symptom was
observed" does not close the defect class. An automatically checked invariant
is needed (§7, T4).

## 5. Proposed solution (sketch; details after the priority is approved)

Bring all three provider layers to the same invariant already established in
`build_in_progress_notice`:

> `job_id` MUST be a canonical `job_N` from `ParkedJobRegistry`.

Specifically, for each of the three `proxy_mcp_server.py` files:

1. at parking time take the canonical id: `registry.allocate()` from the shared
   `ParkedJobRegistry`, and register `call_id` as an **alias**
   (`registry.register(job_id, ..., alias=call_id)` — the mechanism already
   exists and is used in `bounded_tool_call._park_task`);
2. store the local entry under the canonical key instead of `toolu_…`, keeping
   alias resolution for incoming `resolve_tool_result(call_id, …)` calls;
3. build the text **via the shared** `build_in_progress_notice(...)` rather than
   a private f-string — then the message physically cannot diverge from the
   contract;
4. bring the `timeout_notice` branch to the same place (it has its own f-string
   with the same defect).

Key idea: **remove the text duplication**, not just fix the value. As long as
the code has four independent f-strings with `job_id:`, a fifth source is a
matter of time.

### 5.1. Alternatives

| Option | Assessment |
|---|---|
| Keep `toolu_` but teach `wait_for_job` to resolve the proxy-MCP registry | Rejected: it would create a fourth resolution point and entrench two namespaces instead of one. Directly contradicts the "one namespace" decision made in `DESIGN_FIX_JOB_ID_NAMESPACE_STALL.md`. |
| Remove parking from proxy-MCP entirely, leaving only the middleware | Tempting (minus a whole layer), but this is an **architecture change**: the layers have different bonds and different application points. Requires separate agreement; not part of this fix. |
| Emit an honest "not_started" notice, as in `7074243` | Not suitable: here the work is **really parked** and the Future is alive — the job-handle promise is truthful, only the identifier is wrong. |

## 6. Risks

* **`resolve_tool_result` compatibility.** Currently it looks up `call_id` in
  `self._jobs` (`:670`). After switching the key to `job_N`, resolution must go
  through the alias, otherwise the parked tool's result would be lost — a
  regression worse than the original. Mandatory test (§7, T3).
* **Three files — three copies.** Editing the three providers at once raises
  the risk of divergence. Mitigation: a shared helper + the T4 invariant test.
* **Grok/codex not verified live.** The defect in them was found by reading the
  code (the symptom was observed on claude-code-sdk). They will need a separate
  live check or an explicit "fixed by analogy, not reproduced" note.

## 7. TDD plan (RED → GREEN + mutation checks)

| # | Test | RED without the fix |
|---|---|---|
| T1 | Parking in `proxy_mcp_server` registers an entry in the shared `ParkedJobRegistry`, and `_resolve_parked(job_id)` finds it | currently the registry is isolated → `None` |
| T2 | The notice text contains `job_id: job_<N>` and does **not** contain `toolu_` (both branches: `job_handle` and `timeout_notice`) | currently `toolu_…` |
| T3 | `resolve_tool_result(call_id=toolu_…)` still completes the parked Future after the key is switched to `job_N` | regression guard |
| T4 | **Invariant test:** no `.py` outside tests builds the `job_id:` string with its own f-string — the only source is `build_in_progress_notice` (AST/text check across all providers) | currently 3 violations |
| T5 | The race from §3.1: if the backend has already created `job_N`, re-parking does not produce a second id | — |

**Mutation checks** for T1, T2, T4 (following the precedent of the previous
fixes): revert `call_id` in the notice — T2 and T4 must go red; disable
registration in the shared registry — T1 must go red.

**Integration test without mocks:** a real agent, a tool with a guaranteed
bond overrun, then `wait_for_job` with the received id — it must return a
status, not `no background job`. A hard time budget (the rule "integration
tests must not hang").

## 8. Scope

**In scope:** `claude_code_sdk_provider/proxy_mcp_server.py`,
`codex_provider/proxy_mcp_server.py`, `grok_provider/proxy_mcp_server.py`;
a shared text helper; the §7 tests; updating
`DESIGN_FIX_JOB_ID_NAMESPACE_STALL.md` (marking that there were four sources).

**Out of scope:** removing the parking layer from proxy-MCP (an architecture
change, §5.1); a private worktree copy — not touched.

## 9. Open questions

1. **Priority.** The symptom is not blocking (no work is lost; it keeps
   running in the background), but the model loses control of the task and
   orphaned processes accumulate. **Remains open — the user decides.**
2. **Codex/grok.** Closed by the reviewer — see §10.
3. **The T4 invariant in pre-commit.** Closed by the reviewer — see §10.

## 10. Reviewer's decisions (2026-07-31)

* **§5.1 — fully agreed.** Do NOT remove parking from proxy-MCP in this fix
  (an architecture change; an explicit user OK is needed); do NOT add a fourth
  registry in `wait_for_job`; do NOT use `not_started` (the work is real). The
  correct path: `ParkedJobRegistry.allocate` + alias=`call_id` + **only**
  `build_in_progress_notice`.
* **§6 — accepted as critical.** After switching the local storage key to
  `job_N`, an incoming `call_id` is resolved via the alias (or a dual-key
  lookup). **T3 is blocking**: a red T3 means stop. Document the dual-key in
  the code: the notice and the registry operate on `job_N`, while
  `resolve_tool_result` keeps accepting `toolu_…`.
* **§9.2 — fix all three providers in one scoped change.** One defect class,
  one contract; T4 must pass on all three. Live reproduction on claude-code-sdk
  only is acceptable; mark codex/grok in the commit message as "by analogy +
  unit/AST". Phased — only if the user demands a minimal blast radius.
* **T4 — mandatory** as a unit/AST invariant with mutation checks; this is the
  real protection against a fifth f-string.
* **Pre-commit — optional follow-up, NOT a blocker.** First ship T4 in the
  unit suite; extending the hook only separately. Do not hold the fix back for
  hook work.
