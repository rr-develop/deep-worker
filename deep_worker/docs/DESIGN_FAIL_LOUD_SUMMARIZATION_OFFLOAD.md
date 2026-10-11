# Design: Fail-loud on summarization history offload failure

Status: Draft
Date: 2026-07-02
Related policies: `docs/POLICIES.md §1` (No-Agent-Error-State Policy),
`deep_worker/docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md`

## Problem

`SummarizationMiddleware._offload_to_backend` /
`_aoffload_to_backend` (`deepagents/middleware/summarization.py`) silently
tolerates offload failures. When the backend `write` / `edit` returns `None`
or `result.error`, or an exception is thrown during the write:

1. `_offload_to_backend` catches the exception, logs a **`warning`**, and
   returns `None`.
2. The caller (`before_model` / `abefore_model`, `summarization.py:842-849`)
   emits `warnings.warn(...)` and then **continues summarization anyway**:
   `_create_summary` is called, the old messages are dropped from the active
   thread, and `_build_new_messages_with_path(summary, None)` produces a
   short summary message **without any file path reference**.

Result: the pre-summarization history is lost, the summary message doesn't
even mention the loss, and the agent stays in the RUNNING / IDLE flow as if
everything went fine. This is exactly the "silent failure" the user
prohibited: recoverable-looking flow that irrecoverably loses conversation
history.

The docstring of `_offload_to_backend` claims "abort summarization if this
fails to prevent data loss", but the code doesn't. The comment is a lie.

## Goals

1. **Fail loud**: if the offload write cannot persist the pre-summarization
   history (backend error, permission denied, disk full, file locked,
   backend returns `None`, exception during write), the summarization step
   MUST raise. It must NOT swallow the failure with `warnings.warn`.
2. **The raise must transition the agent to `AgentStatus.ERROR`** per the
   existing No-Agent-Error-State Policy. This means adding the new
   exception class to `FATAL_EXCEPTION_TYPES` in
   `deep_worker_cli/exceptions.py`.
3. **The error message must be actionable**: it must name the target path
   and the underlying cause (backend error string, exception class + args)
   so the operator can immediately see whether it's permissions, disk
   space, or a backend bug.
4. **No changes to other policies**: recoverable-vs-fatal classification
   for anything else stays the same. No new sleeps, threads, polling. No
   changes to the RPyC surface. No changes to how summarization is
   triggered.

## Non-goals

1. Do not add a "skip offload" flag / opt-out. Silent skip is exactly what
   we're removing.
2. Do not attempt automatic retry inside the middleware. The middleware
   runs on the agent's event loop; a retry loop there is the wrong place.
   If retry is desired later, it belongs in the backend implementation
   (backend is the concrete resource owner).
3. Do not change the JSONL format, the summary-message shape, or the
   backend protocol.
4. Do not deduplicate the three copies of `FATAL_EXCEPTION_TYPES` in
   `exceptions.py` (that's a separate cleanup).

## Design

### 1. New exception

Add to `deepagents/middleware/summarization.py`:

```python
class HistoryOffloadError(RuntimeError):
    """Raised when the pre-summarization history offload cannot be persisted.

    Fatal — proceeding with summarization would silently discard the
    history that is about to be evicted from the active thread. Handled by
    the agent runtime's fatal-exception catch-all
    (``FATAL_EXCEPTION_TYPES``) → ``AgentStatus.ERROR``.
    """
```

The class is defined in the summarization module (not in
`deep_worker_cli.exceptions`) so `deepagents` stays independent of the
runtime CLI package. Classification is by class name, so registering the
name in `FATAL_EXCEPTION_TYPES` is enough.

### 2. Register as fatal

Add `"HistoryOffloadError"` to every copy of `FATAL_EXCEPTION_TYPES` in
`deep_worker_cli/exceptions.py`. (Yes, three copies; that's a bug but not
mine to fix in this ticket.)

### 3. `_offload_to_backend` / `_aoffload_to_backend`

Both methods change their return type from `str | None` to `str` (the
path). All error paths inside these methods **raise** instead of
returning `None`:

- Existing `try/except` around `backend.download_files` for reading
  existing content stays as-is (it's already logged at DEBUG for
  file-doesn't-exist-yet). This is not a failure — we treat missing file
  as empty existing content.
- `try/except` around the actual write:
  - `result is None or result.error` → raise `HistoryOffloadError` with
    message `f"Backend failed to write history to {path}: {error_msg}"`
    from the `result.error` string (or `"backend returned None"`).
  - Any exception during write → wrap with
    `HistoryOffloadError(f"Exception writing history to {path}: {type(e).__name__}: {e}") from e`.

### 4. `before_model` / `abefore_model`

Remove the `if file_path is None: warnings.warn(...)` guard. The call to
`_offload_to_backend` now either returns a valid path or raises. The raise
propagates through `before_model` up to the runtime catch-all, which
matches `HistoryOffloadError` in `FATAL_EXCEPTION_TYPES` and sets
`AgentStatus.ERROR`.

The `_build_new_messages_with_path(summary, None)` branch remains in the
code (it's still called from elsewhere in tests / future refactors), but
in production it's never hit on offload failure because we raise before
reaching it.

### 5. Tests to update / add

Existing tests (`tests/unit_tests/middleware/test_summarization_middleware.py`)
pin the old silent-warn contract with `pytest.warns(UserWarning, match="Offloading conversation history to backend failed")`
at 8 call sites. They need to be inverted:

- Replace `pytest.warns(UserWarning, ...)` with
  `pytest.raises(HistoryOffloadError, match=...)`.
- Assert that the raised exception's message contains the target path and
  the backend error text.
- Assert that the state was **not** mutated: since the raise happens
  before `_create_summary` is called, no summary text is produced and no
  history is dropped.

New TDD tests (failing on current main, passing after fix):

- `test_offload_backend_returns_none_raises_history_offload_error`
- `test_offload_backend_result_error_raises_history_offload_error`
- `test_offload_write_exception_wrapped_in_history_offload_error`
- Async variants of the three above.
- `test_history_offload_error_is_in_fatal_exception_types`
  (unit test in `deep_worker_cli/tests/`).

Integration test:

- Extend `tests/integration_tests/test_summarization.py` with
  `test_summarization_offload_write_denied_raises` that uses a
  write-denied filesystem backend and asserts `HistoryOffloadError` is
  raised.

## Impact on other components

- **Agent runtime** (`deep_worker_cli/process/agent_process.py`): none.
  The existing catch-all at `_handle_invoke` already routes fatal
  exceptions to `AgentStatus.ERROR`. Adding `HistoryOffloadError` to
  `FATAL_EXCEPTION_TYPES` is sufficient.
- **Providers** (claude-code-sdk, codex, grok, langchain, a2a): none. The
  raise happens in the middleware, above provider code.
- **Web UI**: agents that hit this now show as ERROR. The error banner
  will include the underlying cause. This is a UX improvement — today
  the user has no idea the history was lost.
- **Backwards compat**: agents that were quietly losing history will now
  fail loudly. This is the whole point.

## Risks

- **Flakiness under transient backend errors**: if the backend is flaky
  (e.g. NFS hiccup), summarization now hard-fails instead of continuing.
  Mitigation: agent runtime already surfaces the error, operator can
  `continue` after fixing the backend. Retry-inside-middleware is
  explicitly a non-goal (see above).
- **Existing agents mid-conversation** with a broken backend will
  transition to ERROR the next time summarization is triggered instead of
  continuing to leak history. This is desired.

## Rollout

Single commit on `dev`, then FF to `master`. No feature flag needed —
this is a policy correction, not a new feature. TDD: failing tests
first, then implementation.
