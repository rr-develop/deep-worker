# DESIGN: remove the false wait_for_job/kill_job promise from the stage-A timeout

Status: IMPLEMENTED (2026-07-31). Branch `dev`. Continuation of the fix in
`DESIGN_FIX_JOB_ID_NAMESPACE_STALL.md` (§9.4 — "Deliberately not done").
DESIGN SIGNED OFF (2026-07-31).

## 0. Requirement (verbatim from the user)

> We make fixes not for the sake of fixes, but so that the functionality
> works. […] The point mentioned above of course also needs to be fixed. […]
> Restart yourself and then fix it so there are no such dead ends.

The dead end in question: `wait_for_job(job_id="…")` → `no background job with
id '…'.` The original fix (`6bd042b`) closed the case where
`BoundedToolCallMiddleware` parked a task in a foreign namespace. Here —
the third source of **the same symptom** ("no background job"), but with a
different root: the work was not even parked, yet the promise was already given.

## 1. Facts from the code

### 1.1 Location 1 — claude-code-sdk provider
`claude_code_sdk_provider/chat_model.py:1477-1490`. When the CLI declared a
`ToolUseBlock` with a `tool_use_id`, but the control request never arrived
within the single timeout (`resolve_tool_wait_seconds()`, default 120 s),
`wait_for_pending_call_by_id` raises `DispatchTimeout`. The handler does:

```python
response_text = build_in_progress_notice(
    tool_name=expected_proxy_tool_name or "tool",
    job_id=expected_proxy_tool_call_id,       # this is toolu_…, NOT job_N
    wait_seconds=resolve_tool_wait_seconds(),
)
```

The comment in the code at lines 1470-1476 admits it itself: "There is NO
pending call to execute, so we do NOT fabricate a tool_call". But
`build_in_progress_notice` meanwhile promises the model
`wait_for_job(job_id="toolu_…")` and `kill_job(job_id="toolu_…")` — by an id
for which there is neither a backend-job nor a middleware-park.

### 1.2 Location 2 — codex provider
`codex_provider/chat_model.py:717-733`. Symmetrically: an `ItemStartedEvent`
arrived with `it.status == "in_progress"`, `wait_for_pending_calls` did not
receive the control request → `asyncio.TimeoutError` → `job_id=getattr(it,
"id", None) or it.tool`. The same lie — a promise without a park.

### 1.3 What happens to the task next
In neither claude nor codex is work physically **created**. In claude
`wait_for_pending_call_by_id` only waits for the pending-call registration —
if it never arrives, no `Future`/`Task`/`proc` is started. In codex —
similarly: `_handle_tool_call` would have arrived later and created a pending
call, but it did not arrive. `wait_for_job` and `kill_job` objectively have
nothing to wait for / kill.

### 1.4 Classification relative to the original defect
The same **symptom** ("no background job"), a different **root cause**: not
"the system substituted a foreign namespace", but "the system fabricated a
promise for nonexistent work". The shared `ParkedJobRegistry` from `6bd042b`
will not help: allocate/register without a real task is semantically incorrect
and would simply move the lie one level down.

### 1.5 Other places where an in_progress-notice is built
`grep build_in_progress_notice`:
- `deep_worker/deepagents/deepagents/middleware/bounded_tool_call.py:186,242`
  — async and sync parking branches after `6bd042b`. The task is REALLY
  parked in the shared registry, the notice is honest.
- Tests.
Only the two stage-A places lie. Verified.

## 2. Invariants (do not break)

1. The single tool-wait point — `BoundedToolCallMiddleware.awrap_tool_call`
   (POLICIES §7.1). We touch neither waits nor timers here.
2. The invariants of `DESIGN_UNIFIED_BOUNDED_TOOL_WAIT.md §2` are preserved:
   zero bare awaits, zero second timers, zero polling.
3. A timeout is never `is_error`. The string stays NON-error assistant text.
4. No fake-park: do not allocate a `job_N`, do not register a nonexistent
   `handle` — otherwise we would reproduce exactly the rake that `6bd042b`
   just buried.

## 3. Solution architecture

### 3.1 A new function in `bounded_tool_call.py`
Next to `build_in_progress_notice` — a specialized function:

```python
def build_dispatch_timeout_notice(tool_name: str, wait_seconds: float) -> str:
    """Stage-A dispatch timeout: CLI announced the tool call but the control
    request never arrived. NO job was parked (there is no work to wait on or
    kill). NEVER an error — the assistant text ends the turn and the model
    can retry. See DESIGN_FIX_STAGE_A_FALSE_PROMISE.md."""
    return (
        f"Tool '{tool_name}' did not start within {wait_seconds:g}s "
        f"(the CLI announced the call but the control request never arrived). "
        f"status: not_started. No background job was created. "
        f"The turn ends; retry the tool call."
    )
```

The key difference from `build_in_progress_notice`:
- no `job_id` field;
- no `wait_for_job` / `kill_job` promise;
- `status: not_started`, not `in_progress` — unambiguous to the model that
  the work has not started;
- an explicit retry request.

`build_in_progress_notice` is **not changed** and **not removed** — it
correctly serves real parks in the middleware.

### 3.2 Targeted call-site edits
`claude_code_sdk_provider/chat_model.py:1481-1490`:

```python
except DispatchTimeout:
    from deepagents.middleware.bounded_tool_call import (  # noqa: PLC0415
        build_dispatch_timeout_notice,
        resolve_tool_wait_seconds,
    )
    response_text = build_dispatch_timeout_notice(
        tool_name=expected_proxy_tool_name or "tool",
        wait_seconds=resolve_tool_wait_seconds(),
    )
```

`codex_provider/chat_model.py:722-733`:

```python
except asyncio.TimeoutError:
    from deepagents.middleware.bounded_tool_call import (  # noqa: PLC0415
        build_dispatch_timeout_notice,
        resolve_tool_wait_seconds,
    )
    final_text = build_dispatch_timeout_notice(
        tool_name=it.tool,
        wait_seconds=resolve_tool_wait_seconds(),
    )
    turn_done = True
    break
```

`turn_done = True; break` is kept (confirmed by the reviewer): the semantics
"end the turn without a phantom tool_call" is already correct, changing it is
out of scope.

### 3.3 What is not done and why
- **No retroactive parking of the task.** There is no process, no `Future`,
  no `Task`. A fictitious entry in `ParkedJobRegistry` would be the same lie
  under a different sauce.
- **No touching of the timer, `DispatchTimeout`, `wait_for_pending_call_by_id`,
  `wait_for_pending_calls`.** Their behavior is correct.
- **No removal of `build_in_progress_notice`.** Its real consumers — lines
  186 and 242 in `bounded_tool_call.py` — work correctly.

## 4. TDD

### 4.1 Units
1. `deepagents/tests/unit_tests/test_dispatch_timeout_notice.py`
   - `test_notice_has_no_job_id_or_wait_kill_promise` — the result of
     `build_dispatch_timeout_notice(…)` contains no `job_id`, `job_`,
     `wait_for_job`, `kill_job` substrings.
   - `test_notice_includes_tool_name_and_seconds` — contains the tool name and
     the number of seconds.
   - `test_notice_marks_status_not_started` — contains `not_started`,
     no `in_progress`, no `error`.
   - **Regression**: `test_in_progress_notice_still_has_job_id_and_actions`
     — `build_in_progress_notice(…)` still contains `job_id`,
     `wait_for_job`, `kill_job`. Guarantees the fix does not break real parks.
2. `claude_code_sdk_provider/tests/test_stage_a_no_false_job_promise.py`
   - `test_dispatch_timeout_uses_dispatch_timeout_notice` — we run ainvoke
     with a proxy that never resolves the control request; `response_text`
     contains `not_started`, lacks `wait_for_job` and `toolu_`.
3. `codex_provider/tests/test_stage_a_no_false_job_promise.py`
   - symmetric for codex.

### 4.2 Mutation checks
- Revert `build_in_progress_notice` at the call-site → stage-A tests fail.
- Remove `not_started` from the new notice → `test_notice_marks_status_not_started`
  fails.
- Add `job_id` to the new notice → `test_notice_has_no_job_id_or_wait_kill_promise`
  fails.

### 4.3 Integration level
By agreement with the reviewer (answer 3): the unit level is sufficient.
An integration test "bring up a real provider with an artificially lost
control-request" is expensive, fragile, and adds no guarantees beyond the
contract test of the string.

## 5. What was actually implemented (post-ship, 2026-07-31)

- `deepagents/middleware/bounded_tool_call.py`: added
  `build_dispatch_timeout_notice(tool_name, wait_seconds)`; the docstring of
  `build_in_progress_notice` was supplemented with a warning — to use only
  after a REAL park under a canonical `job_N`.
- `claude_code_sdk_provider/chat_model.py`: `except DispatchTimeout` now calls the
  new notice instead of `build_in_progress_notice`. `response_text` no longer
  contains `toolu_…` in the `job_id` field.
- `codex_provider/chat_model.py`: `except asyncio.TimeoutError` — the same.
  `final_text`, `turn_done = True; break` — preserved.
- Tests (RED→GREEN + mutation checks):
  * `deep_worker/deepagents/tests/unit_tests/test_dispatch_timeout_notice.py`
    — 5 tests: notice contract (no `job_id`/`wait_for_job`/`kill_job`,
    contains the tool name and seconds, `status: not_started`, a retry
    request) + a regress-guard that `build_in_progress_notice` still carries
    `job_id` and actions;
  * `claude_code_sdk_provider/tests/test_stage_a_no_false_job_promise.py` — AST
    call-site + end-to-end via `wait_for_pending_call_by_id` + skew clock;
  * `codex_provider/tests/test_stage_a_no_false_job_promise.py` — AST
    call-site (symmetric to claude).

All four mutations are caught: (1) reverting the claude call-site to
`build_in_progress_notice`, (2) substituting `not_started`, (3) inserting
`job_id` into the new notice, (4) breaking the `build_in_progress_notice`
contract.

Runs: `deepagents` 700 passed (+5); `claude_code_sdk_provider` 300 passed
(1 pre-existing flake checked on baseline); `codex_provider` 144 passed;
`deep_worker_cli` unchanged vs baseline. No regressions.

Design + code review: (DESIGN SIGNED OFF, CODE SIGNED OFF).

## 6. Boundaries

Not in this fix:
- Diagnosing the causes of the lost-dispatch itself (that is a CLI/proxy bug,
  separate).
- Automatic retry inside the provider (complicates the protocol; the model
  will decide on its own based on the hint in the notice).
- Changes to `_default_wait_seconds` / `DW_LONG_RUNNING_WAIT_SECONDS`.
