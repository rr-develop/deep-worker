# DESIGN: a unified job-id namespace for tool parking (`toolu_…` → `job_N`)

Status: IMPLEMENTED (2026-07-31). Branch `dev`. Design signed off
(DESIGN SIGNED OFF); the reviewer's answers to §8 are incorporated in the
implementation — see §9.
Related documents: `docs/DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md` (invariants of
the single wait point), a past incident (2026-06-17) — the same defect class
on another provider.

## 1. Symptom (observed; reproduced repeatedly in an earlier session)

```
execute(...)
→ Tool 'execute' is still running after 120s. status: in_progress. job_id: toolu_01ABC…
wait_for_job(job_id="toolu_01ABC…")
→ Error: no background job with id 'toolu_01ABC…'.
kill_job(job_id="toolu_01ABC…")
→ Error: no background job with id 'toolu_01ABC…'.
```

The model loses control of the task: the work keeps running in the background,
but it can neither be awaited nor killed. Live child `claude` CLI processes
were observed accumulating (4 per hour).

## 2. Root cause (established from the code, not by guesswork)

There are **TWO independent background-job registries with different id
namespaces**, and **both timers share the same 120 s default**.

| Layer | File:lines | Timer | Registry key | What the model receives |
|---|---|---|---|---|
| `BoundedToolCallMiddleware` (outer, mandatory per §7.1) | `deepagents/middleware/bounded_tool_call.py:228-246` | `DEFAULT_TOOL_WAIT_SECONDS = 120.0` | `self._jobs[tool_call_id]` | `job_id = tool_call_id` = **`toolu_…`** |
| `CLIShellBackend.execute` (inner, `execute` only) | `deep_worker_cli/backends.py:272-307` | `DEFAULT_EXECUTE_TIMEOUT = 120` | `self._jobs["job_N"]` + alias `toolu_→job_N` | `job_id = job_N` |

The `wait_for_job` / `kill_job` tools (`backends.py:365`, `:410`) resolve ids
**exclusively** through `CLIShellBackend._resolve_job` — that is, they see
**only the backend registry**. The middleware registry
(`BoundedToolCallMiddleware._jobs`) has no readers at all:
`get_job_registry()` is never called anywhere in production.

The race: the outer bond covers stage-A (dispatch wait of the CLI provider)
**plus** execution, the inner one covers execution only. With both at 120 s on
claude-code-sdk/codex the outer one expires **first**. Then:

1. the middleware parks the task in its own registry under the `toolu_…` key;
2. the backend has NOT yet reached its timeout → `job_N` is not created, the
   `toolu_→job_N` alias is not written;
3. the model receives `toolu_…` and calls `wait_for_job` → the backend registry
   is empty for that key → `no background job`.

The alias mechanism from the alias fix (`backends.py:288-293`) only covers the
reverse case (the model itself supplied `toolu_…` **after** the backend had
already created `job_N`). It is powerless when `job_N` is never created at all.

**This is NOT "the model mixed up the namespace".** The system itself emitted
`toolu_…` in the `job_id` field — a promise it cannot keep.

## 3. User requirement (verbatim)

> **The correct fix** — an edit in **claude_code_sdk_provider/** or in the
> middleware that registers backgrounded jobs (a real `job_N` must be created,
> not `toolu_XXX` substituted in).

The fix goes into the middleware (the single parking point per §7.1), because
the defect is provider-independent: `BoundedToolCallMiddleware` is shared by all
providers, and the race is identical on codex/grok.

## 4. Invariants the fix does NOT break

From `DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md` §2:

1. Exactly one function awaits tool execution — `awrap_tool_call`. **Unchanged.**
2. Applies to all providers through the shared middleware stack. **Unchanged.**
3. One time budget for the whole tool interval. **Unchanged.**
4. Timeout → NOT `is_error`, work continues in the background (no cancel) +
   `in_progress` notice. **Unchanged.**
5. No bare `await` without a bond. **Not added.**
6. No visible-block injection into pending. **Not added.**

Additionally we keep: the "polling forbidden" policy (event-driven solution,
no polling), "no new threads" (we add no threads).

## 5. Solution

### 5.1 Essence

When parking, `BoundedToolCallMiddleware` must emit a **canonical `job_N`** —
the same namespace as the backend — and its registry must be **visible to
`wait_for_job` / `kill_job`**.

A single shared registry contract is introduced: **`ParkedJobRegistry`** — the
process-wide source of truth for background-job ids.

```
deepagents/middleware/parked_job_registry.py   (new module)

class ParkedJobRegistry:
    """THE ONLY source of job-ids for parking. Exactly one namespace: job_N."""
    def allocate(self) -> str                      # "job_1", "job_2", …
    def register(self, job_id, *, kind, handle, alias=None) -> None
    def resolve(self, job_id: str) -> ParkedJob | None   # by job_N OR by alias
    def drop(self, job_id: str) -> None                  # + purge aliases
```

- `allocate()` — a single counter; `itertools.count(1)` within the same process.
- `register(alias=tool_call_id)` — the `toolu_… → job_N` alias is written
  **always**, at parking time. Backward compatibility: a model that supplied
  `toolu_…` still reaches the task.
- `resolve()` — unified logic for the two namespaces, moved from
  `CLIShellBackend._resolve_job` (behavior preserved 1-to-1).

### 5.2 Changes by file

**A. `deepagents/middleware/bounded_tool_call.py`**

In `awrap_tool_call` / `wrap_tool_call`, when the bond fires:

```python
except asyncio.TimeoutError:
    job_id = registry.allocate()                  # "job_7" — NOT tool_call_id
    registry.register(job_id, kind="tool", handle=task, alias=tool_call_id)
    return ToolMessage(
        content=build_in_progress_notice(tool_name, job_id, wait_seconds),
        tool_call_id=tool_call_id,                # this field is the LangChain protocol;
        status="success",                          # we do not touch it
    )
```

Key point: `build_in_progress_notice` receives `job_id` (`job_N`), not
`tool_call_id`. The `ToolMessage.tool_call_id` field stays `toolu_…` — that is a
LangChain protocol requirement and is unrelated to job ids. Mixing these two
things is exactly what created the defect.

`_ParkedToolJob` and the local `self._jobs` are removed — their role is taken
over by `ParkedJobRegistry`. `get_job_registry()` is kept as a thin delegate for
the existing tests.

**B. `deep_worker_cli/backends.py`**

- `CLIShellBackend` no longer keeps its own `_job_counter` and `_job_aliases`;
  `execute()` takes an id via `registry.allocate()` and registers the task in
  the shared registry with an alias to the current `tool_use_id` (the
  `_current_tool_use_id` logic is preserved unchanged).
- `_resolve_job` delegates to `registry.resolve`, additionally checking that the
  found task belongs to the shell namespace (`kind == "shell"`) — otherwise
  `wait_for_job` could not return an `ExecuteResponse`.
- `wait_for_job` / `kill_job`, upon encountering a task with `kind == "tool"`
  (parked by the middleware, not by a shell command), return a **meaningful
  non-error** response: the task was found, it is the tool `<name>`, status
  running/finished; killing such a task is done by cancelling its `asyncio.Task`.

The single counter guarantees that `job_N` from the middleware and from the
backend never collide.

**C. Race "both timers fired"**

If the backend still reaches its own timeout after the middleware has already
parked the task, it allocates a **second** `job_N` for the same command. This
is acceptable and safe: both entries resolve and both point to live work. To
avoid duplicates, the backend checks at registration whether a task already
exists under the current `tool_use_id` alias and reuses its id.

### 5.3 What is NOT done

- We do not change the timeout values and do not spread them apart (120/120).
  Spreading masks the defect instead of fixing it: with
  `DW_LONG_RUNNING_WAIT_SECONDS` smaller than `DEFAULT_EXECUTE_TIMEOUT`, the
  race would return.
- We do not change the `wait_for_job` / `kill_job` contract (signatures, names,
  and semantics stay the same).
- We do not touch the stage-A dispatch bridge or the CLI sequential protocol.

## 6. TDD plan

**RED tests (must fail on the current code):**

1. `test_middleware_parks_under_job_n_namespace` — run `awrap_tool_call` with
   a tool that does not fit within the bond; assert that the `in_progress`
   notice text contains `job_<N>` and does **not** contain `tool_call_id`.
   Currently fails: `toolu_…` is emitted.
2. `test_wait_for_job_resolves_middleware_parked_job` — a task parked by the
   middleware is found via `wait_for_job` by the issued id. Currently fails:
   `no background job`.
3. `test_wait_for_job_resolves_by_tool_use_alias` — the same task is also found
   by `toolu_…` (backward compatibility with the alias fix).
4. `test_shell_and_middleware_job_ids_never_collide` — both layers allocate ids
   from a single counter; no overlaps.
5. `test_kill_job_cancels_middleware_parked_task` — `kill_job` by the id of a
   middleware task actually cancels its `asyncio.Task`.

**Regression tests (must stay green):**
`tests/unit_tests/test_job_id_tooluse_alias.py`,
`tests/unit_tests/test_background_jobs.py`,
`claude_code_sdk_provider/tests/test_job_handle_by_default.py`,
`claude_code_sdk_provider/tests/test_job_handle_integration.py`,
`claude_code_sdk_provider/tests/test_stage_a_bounded_single_source.py`.

**Integration test without mocks:** bring up a real agent, give `execute` a
command that is deliberately longer than the bond with
`DW_LONG_RUNNING_WAIT_SECONDS=5`, take the issued `job_id` from the response,
call `wait_for_job`, and verify the task is found and its output is retrievable.
Hard budget ≤60 s, guaranteed SIGTERM→SIGKILL in `finally` (see the rule
"integration tests must not hang").

## 7. Risks

- **Test compatibility.** `BoundedToolCallMiddleware._jobs` / `_ParkedToolJob`
  appear in existing tests — we keep `get_job_registry()` as a delegate and fix
  the tests if needed (they are tests, not a contract).
- **Process-wide registry.** A single counter per process — an agent has
  exactly one process and one backend, so there are no collisions. The registry
  is NOT a global mutable variable in the sense of state shared between agents:
  each process has its own.
- **Leak of parked tool tasks.** Middleware tasks must fall out of the registry
  after completion — we attach `add_done_callback(registry.drop)`
  (event-driven, no polling).

## 8. Open review questions

1. Location of `ParkedJobRegistry`: `deepagents/middleware/` (next to the
   consumer) or top-level `deepagents/`? The former is proposed.
2. Is cancelling the `asyncio.Task` enough for `kill_job` over a tool task, or
   is a stricter contract needed (wait for `CancelledError`)?
3. Should `wait_for_job` be able to return the result of a tool task (not only
   shell output), or is a status enough? Proposed: status + result if the task
   has already finished.

## 9. What was actually implemented (after the design sign-off)

### 9.1 Files

| File | Change |
|---|---|
| `deepagents/middleware/parked_job_registry.py` | **new.** `ParkedJobRegistry` + `get_parked_job_registry()` (a module-level singleton per process), `ParkedJob(job_id, kind, handle, tool_name, aliases)` |
| `deepagents/middleware/bounded_tool_call.py` | async branch: `_park_task()` — allocates `job_N`, registers `kind="tool"` + alias, `add_done_callback` → `drop`. sync branch: parks a `concurrent.futures.Future` under the same `job_N`. `get_job_registry()` now returns a **copy** |
| `deep_worker_cli/backends.py` | `_allocate_job_id` / `_register_shared` / `_resolve_parked` / `_wait_parked_tool` / `_kill_parked_tool` / `_render_tool_result`; `_drop_job` also cleans the shared registry |

### 9.2 Reviewer answers incorporated verbatim

1. **Placement** — `deepagents/middleware/parked_job_registry.py`, exported via
   `get_parked_job_registry()`. There are no private middleware instances
   (exactly what had created the two registries).
2. **`kill_job` over a tool task** — `task.cancel()` + **bounded** wait
   (`wait_for(shield(task), 2.0)`); no unbounded `await CancelledError`. If it
   did not stop, we honestly report "cancellation requested, not stopped yet".
3. **`wait_for_job` over a tool task** — status **and** result: a finished task
   returns its content via `_render_tool_result`, a running one returns an
   `in_progress` response with the canonical `job_N`.
4. **Process-wide registry** — §7.1 is not violated: this is not a second wait
   point but a shared identity table; awaiting still happens only in
   `awrap_tool_call`.

**Non-blocker #1 (sync path).** Real parking was implemented, not just
cosmetics: the sync branch registers a `Future` under `job_N`. A thread cannot
be cancelled, so `kill_job` over such a task honestly reports that it will
finish on its own — without a fake "killed".

**Non-blockers from code review — applied before the commit:**

1. The lazy `deepagents` import was narrowed from `except Exception` to
   `except ImportError`. A missing package (isolated tests) is still acceptable,
   but **a breakage inside the registry module itself now surfaces** instead of
   silently disabling the bridge — otherwise exactly the "no background job"
   symptom being fixed here would have returned.
2. The `exit_code` of a finished tool task is derived from `ToolMessage.status`
   (`_tool_exit_code`): a failed tool no longer looks like a clean success just
   because the *job* finished. Content is returned in both cases. Locked in by
   two tests (`test_failed_tool_job_reports_nonzero_exit_code`,
   `test_successful_tool_job_reports_zero_exit_code`).
3. The stage-A false promise is a separate task, see §9.4 (agreed with the
   reviewer; does not block the current delta).

### 9.3 Proof

Mutation checks (every mutation is caught):

| Mutation | Tests caught |
|---|---|
| middleware again returns `tool_call_id` instead of `job_N` | 5 |
| backend blind to middleware tasks (`_resolve_parked` → `None`) | 3 |
| backend back to a private counter | 1 |

Runs:
- `deepagents/tests/unit_tests` — **695 passed**, 70 skipped, 3 xfailed, 0 failed
- `deep_worker_cli/tests/unit_tests` (without `cli/`) — **1906 passed, 8 failed**;
  the same 8 fail on a clean `HEAD` too (pre-existing: `preferred_provider`,
  checkpointer, launch-config) — **zero regressions**
- Contract regressions: `test_job_id_tooluse_alias.py` + `test_background_jobs.py`
  — 15 passed; `claude_code_sdk_provider` job-handle/stage-A — 8 passed
- Mock-free integration (`test_parked_job_id_live.py`) — **3 passed in 7.8 s**;
  with the middleware reverted to the old code the test goes red

Live confirmation of the defect was obtained right during the work: an agent
workflow (old code) emitted `job_id: toolu_019qXvMXxtMv5afCKG6BBndk`, and
`wait_for_job` answered `no background job` — exactly the scenario being fixed.

### 9.4 Deliberately NOT done

The stage-A timeout in `claude_code_sdk_provider/chat_model.py:1486` and
`codex_provider/chat_model.py:727` also calls `build_in_progress_notice` with a
non-job identifier (`tool_use_id` / `it.id`). But there **no work is parked at
all** — there is nothing to wait for or kill, and therefore no `job_N` to hand
out. The correct fix is to remove the `wait_for_job`/`kill_job` promise from
that text, not to substitute a fictitious id. This is a separate, lower-risk
defect: it was kept out of this change to avoid mixing it with the parking fix.
It requires a separate agreement on the wording.
