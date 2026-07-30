# Fix: `TOOL_ERROR` must not flip agent to ERROR; Grok arg-key/value compat

**Status**: Draft → review
**Date**: 2026-07-30
**Branch**: `dev`
**Related**:
- `docs/POLICIES.md` §1 (No-Agent-Error-State Policy)
- `docs/DESIGN_NO_AGENT_ERROR_STATE_POLICY.md`
- `deep_worker/deep_worker_cli/deep_worker_cli/process/status_callback.py`
- `deep_worker/deep_worker_cli/deep_worker_cli/session/session_manager.py`
- `deep_worker/deep_worker_cli/deep_worker_cli/exceptions.py` (`is_fatal_exception`)
- `grok_provider/chat_model.py` (`_map_tool_call`, `_correlate_proxy_tool_calls`)
- `grok_provider/acp_types.py`

## 1. User requirement (verbatim)

> Yes, we need to fix at least defects A and B. Follow the procedure
> add_feature → design → review → implement → commit → ff to master.

From the previous analysis (see the agent report):

- **DEFECT A (UX deception):** `StatusTrackingCallback.on_tool_error` writes
  `AgentStatus.ERROR` to the agent's global status because of a
  pydantic ValidationError from a single tool call. In the UI the agent is almost
  always "error", although it is actually working.
- **DEFECT B (grok tool naming):** Grok sends our tools with foreign
  argument names (`read_file(target_file=...)` — Cursor style)
  or wrong types (`execute(timeout='20s')`), which makes pydantic
  fail validation, and (through A) the agent is shown as "in error".

## 2. Where the bugs live (root cause)

### 2.1 DEFECT A

`deep_worker_cli/process/status_callback.py:65-91`:

```python
async def on_tool_error(
    self,
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    ...
) -> None:
    fields = [f"error={error}", f"run_id={run_id}"]
    if parent_run_id is not None:
        fields.append(f"parent_run_id={parent_run_id}")
    print(
        f"[{datetime.now(UTC).isoformat(timespec='microseconds')}] [TOOL_ERROR] {' '.join(fields)}",
        file=sys.stderr,
    )
    if self.session_manager:
        await self.session_manager.set_status(AgentStatus.ERROR)   # ← violates POLICIES §1
```

LangChain runtime invokes this hook on *any* tool exception —
including pydantic `ValidationError` on argument coercion, network
errors from a single MCP tool, etc. None of these are fatal per
`FATAL_EXCEPTION_TYPES` in `deep_worker_cli/exceptions.py`. Policy §1
says: recoverable → tool returns/appends an error message; the agent
stays `IDLE`/`RUNNING`. Global `ERROR` state is reserved for
`is_fatal_exception(exc) == True`.

Effect chain (from live logs, 2026-07-30):

1. `on_tool_start` → `WAITING_INPUT` (persist + notify).
2. tool arg validation fails → `on_tool_error` → **`ERROR`** (persist + notify).
3. LangChain reports error to model as a `ToolMessage` (recoverable);
   the model retries with corrected args.
4. Next successful tool: `on_tool_start` → `WAITING_INPUT` (overwrites `ERROR`).
5. Between (2) and (4) — 5–30 s of "model thinks" — the web UI shows
   red `error`. If validation errors are frequent, the UI is red
   almost the entire time.
6. On the very last tool call of a turn (no follow-up), `ERROR` stays
   until the top-level `finally` in `_handle_invoke` sets `IDLE`.

Additionally, `SessionManager.clear_error()` (session_manager.py:162)
is defined but **never called from any code path** — dead code that
would have masked the misdesign.

### 2.2 DEFECT B

`grok_provider/chat_model.py:_map_tool_call` surfaces the ACP
`tool_call.rawInput` verbatim as the LangChain tool args:

```python
title = upd.get("title")                       # e.g. "read_file"
raw_input = upd.get("rawInput") or {}          # e.g. {"target_file": "/x"}
proxy_tool = self._proxy_tool_from_update(upd)
if proxy_tool is not None:
    name, args = proxy_tool                    # forwarded unchanged
    if name:
        return {"id": tcid, "name": name, "args": args, ...}
```

Grok's model regularly emits Cursor-style keys (`target_file`) and
human-readable values (`timeout="20s"`). Our filesystem tools
(`middleware/filesystem.py:_create_read_file_tool` etc.) declare the
schema with `file_path: str`, `timeout: int` — mismatch → pydantic
`ValidationError` inside `StructuredTool` → `on_tool_error`.

Live-observed mismatches across several agent runs:

| Tool          | Wrong key / value              | Correct  |
|---------------|--------------------------------|----------|
| `read_file`   | `target_file=`                 | `file_path=` |
| `execute`     | `timeout='20s'` (string)       | `timeout=20` (int) |

Other Cursor-style aliases seen in grok's `unknown built-in` route
(`grok_provider.chat_model:1179`) — not blockers today but likely
future collisions if we ever add a real tool with the same name:
`list_dir`, `run_terminal_command`, `todo_write`, `write`,
`Web search:`, `search_tool`. Those are already benign (they fall
through to `GrokBuiltin`) and out of scope for this fix.

## 3. Fix — design

### 3.1 A: honor policy in `on_tool_error`

**Change** `StatusTrackingCallback.on_tool_error` to:

1. **Always** log the structured `[TOOL_ERROR]` line to stderr
   (existing behavior — kept for post-mortem parity with all other
   agents).
2. Classify the exception via
   `deep_worker_cli.exceptions.is_fatal_exception(exc)`.
3. If **fatal** — call `session_manager.set_error(str(exc))` (the
   existing path that turns the agent red and stores
   `error_message`). This is the *only* time this callback should
   surface `ERROR`.
4. If **recoverable** (default: pydantic ValidationError, generic
   Exception, MCP `-32602`, etc.) — call
   `session_manager.set_status(AgentStatus.RUNNING)` so the UI
   returns to the working state immediately (the model is about to
   see a `ToolMessage(status="error")` and will retry). Do **not**
   flip to `ERROR`. Do **not** touch `error_message`.

Rationale for `RUNNING` (not `WAITING_INPUT`): after a failed tool
the LLM will receive a `ToolMessage` and think — that is exactly
`RUNNING` (model is processing), matching the mirror in
`on_tool_end`.

**Why not clear_error?** In steady state the agent should never have
`error_message` set from this path (fatal cases persist it
deliberately). We do not need to call `clear_error` because we no
longer set an error message for recoverable failures.

**No new threads. No polling. Event-driven — pure callback.**

This is a strict tightening of the callback; nothing outside this
function changes. It is fully aligned with §1 of POLICIES.md and
the existing `is_fatal_exception` registry, which is the
single canonical policy anchor already used by
`agent_process._handle_invoke`.

### 3.2 B: Grok arg-key/value compatibility layer

**Scope.** Fix only the two live-observed patterns above. Do **not**
build an open-ended "Cursor compatibility" translator — that is a
`FALLBACKS` slope (POLICIES §7). Any future mismatch that surfaces
in logs will be added explicitly, with the offending
verbatim-`rawInput` shape captured from a real run
(POLICIES §9 / feedback_review_real_payload_shape).

**Where.** New pure module `grok_provider/arg_compat.py`:

```python
# Verbatim table of {tool_name: {wrong_key: right_key}} observed in
# production grok logs. Extend only with evidence.
KEY_ALIASES: dict[str, dict[str, str]] = {
    "read_file": {"target_file": "file_path"},
}

# Verbatim table of {tool_name: {key: coercer}} for known type drift.
def _duration_to_int_seconds(v):
    """'20s' | '2m' | '500ms' | int | float → int seconds; anything else → v."""
    ...

VALUE_COERCERS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "execute": {"timeout": _duration_to_int_seconds},
}

def normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with aliases renamed and coerced values applied.

    Preserves any keys not covered by the tables. Never raises.
    """
```

**Where it plugs in.** `grok_provider/chat_model.py:_map_tool_call`,
CASE A (proxy tool) branch — the *only* place raw grok `rawInput`
becomes LangChain `args` for our tools:

```python
proxy_tool = self._proxy_tool_from_update(upd)
if proxy_tool is not None:
    name, args = proxy_tool
    if name:
        args = normalize_tool_args(name, args)         # ← added
        return {"id": tcid, "name": name, "args": args, ...}
```

Built-in path (CASE B) is untouched — passthrough is verbatim by
design (`builtin_tool_passthrough.py`).

**Alias collision rule.** If the incoming `args` contain **both** an
alias key and the canonical key (e.g. `{'target_file': '/x',
'file_path': '/y'}`), `normalize_tool_args` keeps the canonical key
verbatim and **drops** the alias without touching its value. We do
not merge, do not overwrite, do not raise. Covered by a dedicated
unit test. Rationale: leaving both risks silent overwrite depending
on dict-insertion order; picking the canonical key matches the
schema contract.

**Coercion contract.** `_duration_to_int_seconds` accepts:
`int/float → int(x)`; strings ending in `s`/`m`/`h`/`ms` → converted;
bare numeric string `'20'` → `int('20')`; anything else (including
None or garbage) → returned unchanged so pydantic surfaces the real
type mismatch. Rounding rule for `ms`: `math.ceil(x/1000)` (minimum
1 s). Rationale: our `execute` tool's `timeout` parameter is in
whole seconds, and truncating a sub-second budget to zero silently
disables the timeout.

**No coupling to A.** B narrows the *cause*; A stops the *symptom*
even for future mismatches we haven't tabulated yet.

## 4. Explicitly out of scope

- Redesigning `AgentStatus` (no new enum values). `WAITING_INPUT` /
  `RUNNING` / `IDLE` / `ERROR` continue to mean the same thing.
- Rewriting `on_tool_end` / `on_tool_start` — they are correct.
- Alias tables for anything other than the two live-observed cases.
  New mismatches must arrive with a real-log payload.
- Changing `SessionManager.clear_error` — leaving it as-is.
  Verified: no production caller today (only unit-tests reference
  it). It stays as a public API for future use; cleanup is a
  separate scope item.
- claude-code / codex provider changes — no evidence they emit
  wrong-shape tool args (`target_file` did not appear in their
  `[TOOL_ERROR]` lines; their errors were genuine `execute`
  timeouts/backend errors that A already handles correctly).

## 5. TDD plan

### 5.1 A — `tests/unit_tests/process/test_status_callback_error_logging.py`

New tests (RED first):

- `test_recoverable_pydantic_error_keeps_running_status` — feed a
  `ValidationError`, assert `set_status` called with `RUNNING` (not
  `ERROR`); assert `set_error` **never** called.
- `test_recoverable_generic_exception_keeps_running_status` — same
  for a bare `RuntimeError`.
- `test_fatal_exception_transitions_to_error` — feed an instance of
  each `FATAL_EXCEPTION_TYPES` class (a stub class with matching
  `__name__` is enough for classification); assert `set_error`
  called with `str(exc)`.
- `test_stderr_line_still_emitted_for_recoverable` — regression:
  post-mortem log line format unchanged.
- Update `test_tool_error_log_includes_utc_timestamp_and_run_ids`:
  its `set_status.assert_awaited_once_with(AgentStatus.ERROR)`
  becomes `RUNNING` (documenting the new contract).

### 5.2 B — `grok_provider/tests/test_arg_compat.py` (new file)

Pure unit tests (no I/O, no mocks):

- `test_read_file_target_file_renamed_to_file_path`
- `test_read_file_no_op_when_correct_key`
- `test_execute_timeout_string_seconds_coerced`
- `test_execute_timeout_string_minutes_coerced`
- `test_execute_timeout_string_ms_coerced_ceil_at_least_1`
- `test_execute_timeout_int_passes_through`
- `test_unknown_tool_returns_args_unchanged`
- `test_unknown_key_preserved_alongside_alias_rename`
- `test_garbage_timeout_string_passthrough` (pydantic surfaces the real type error)

### 5.3 B — `grok_provider/tests/test_chat_model.py` extension

- `test_map_tool_call_proxy_normalizes_read_file_target_file` — feed
  a synthetic `session/update` dict with
  `{title: 'read_file', rawInput: {'target_file': '/x'}}`; assert
  `_map_tool_call` returns `args={'file_path': '/x'}`.

### 5.4 Integration (no mocks) — OPPORTUNISTIC ONLY

**Mandatory bar** for CI is §5.2 + §5.3 (pure Python, always run) plus
the §5.1 A-side unit tests. Do **not** gate correctness on model
whim.

Optional live-grok integration (skip if the grok CLI is unavailable):
subscribe to the agent's notify stream during a normal turn and,
**if** the grok model happens to emit a `tool_call` with
`title == "read_file"` whose `rawInput` contains `target_file`,
assert that (a) the resulting LangChain tool_call args carry
`file_path` (not `target_file`), (b) no `AgentStatus.ERROR` is
observed in the stream. If no such call is emitted, the test is a
no-op (not a failure) — prevents non-deterministic CI flakes on
prompt engineering.

Non-mocked integration for A on its own: the existing
`tests/unit_tests/process/test_status_callback_error_logging.py`
uses `AsyncMock(session_manager)` — that's the layer boundary the
callback is *supposed* to test (the callback owns no I/O). A "real"
integration for A would need a live provider round-trip; it is
covered opportunistically by the same live-grok observation above.

### 5.5 Implementation housekeeping

- Update the `StatusTrackingCallback` class docstring: it currently
  says `on_tool_error: Set status to ERROR` (line 20 of
  `status_callback.py`) — must reflect the new
  fatal/recoverable split.
- Preserve the stderr `[TOOL_ERROR]` line format **bit-for-bit**
  (post-mortem tooling and grep patterns depend on it). Only the
  *status transition* changes.
- Optional coverage addition per reviewer note: unit test that
  `execute(timeout="20")` (bare numeric string, no suffix) coerces
  to `int(20)`. Already covered by the coercer contract in §3.2
  ("bare numeric string `'20'` → `int('20')`") — just add the test.

## 6. Rollout

- One commit for A + B (both belong to the same user-facing symptom).
- Pre-commit hook must pass (POLICIES §5).
- `dev` → fast-forward `master` per user's default workflow.

## 7. Open questions for the user

None — the two fixes follow existing policy and existing extension
points; no architectural change is required and no external
contract shifts.

## 8. Non-goals (do not creep)

- Do not "clean up" `clear_error` in the same commit.
- Do not add a broad Cursor→deepworker alias translator.
- Do not touch the `_correlate_proxy_tool_calls` `wait_for_pending_calls`
  timeout — that is a separate open defect.
