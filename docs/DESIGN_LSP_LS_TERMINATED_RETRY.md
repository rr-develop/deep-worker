# DESIGN: retry after the language server dies (LS terminated)

Status: **DESIGN SIGNED OFF** (round 3)
Base commit: `9624db2`
Related documents:
- `docs/DESIGN_LSP_TRANSIENT_CANCEL_RETRY.md` — retry of transient cancels (`-32800/-32801`)
- `docs/DESIGN_LSP_SERENA_PIDFD_RESTART.md` — auto-restart of the backend by the supervisor
- `docs/DESIGN_LSP_OBSERVABILITY.md` — counters in `ProjectStatus`

---

## 1. Problem

### 1.1 Observed incident

An agent working with the target project (Kotlin+Python) received, in
response to a symbol request, raw text:

```
LanguageServerTerminatedException: Language server stdout read process terminated unexpectedly
```

The agent interpreted this as "you cannot use global search /
you cannot use absolute paths" and narrowed its working repertoire.
Both interpretations are wrong.

### 1.2 The real root cause (upstream, outside our code)

Confirmed from the live logs in the local Serena logs directory — **3 crashes
out of 3** have an identical signature:

1. `solidlsp/language_servers/kotlin_language_server.py:227` declares to the
   server the client capability `"diagnostics": {"refreshSupport": True}`
   (as well as `inlineValue.refreshSupport` and `inlayHint.refreshSupport`).
2. The handler for the `workspace/diagnostic/refresh` request is meanwhile
   **not registered**: the Kotlin server has only three `on_request`
   handlers (`client/registerCapability`, `workspace/executeClientCommand`,
   `window/workDoneProgress/create`), lines 477-481.
3. The Kotlin LSP trusts the declared capability and sends the request. It
   gets a refusal and crashes with an unhandled exception in a JVM coroutine:
   ```
   com.jetbrains.lsp.implementation.LspException:
       method 'workspace/diagnostic/refresh' not handled on client.
   ```
4. The JVM process of the language server dies. `solidlsp/ls_process.py:613`
   sees `process.poll() is not None`, the reader thread exits, and a
   `LanguageServerTerminatedException` is born.

A reference fix already exists in the same Serena codebase —
`solidlsp/language_servers/svelte_language_server.py:594`:
```python
self.server.on_request("workspace/diagnostic/refresh", do_nothing)
```
The Kotlin server has no such registration. This is an upstream bug.

**Important:** the JVM process of the language server crashes, inside Serena,
two process boundaries away from us. Our service process never crashed
(verified: `intellij-server` processes live >31 hours, the service is
alive). There is no "uncaught exception crashing our process" in our code,
and never was.

### 1.3 Why this is NOT related to what the agent thought

| Crash | Tool before death | `relative_path` |
|---|---|---|
| `08-01 12:32` | `find_symbol` | `''` (global) |
| `08-01 13:25` | `find_referencing_symbols` | narrow, a specific file |
| `08-02 20:30` | `get_diagnostics_for_file` | narrow, a specific file |

2 of 3 crashes happened on a **narrow** `relative_path`. Narrowing the path
does not protect. Absolute paths are irrelevant: `serena/project.py:309
validate_relative_path` correctly normalizes an absolute path inside the
project and gives an honest `ValueError` for a path outside the project —
the server does not crash from that.

### 1.4 What already works and what remains a defect

Serena itself can recover — `serena/tools/tools_base.py:380-390`
catches `SolidLSPException`, restarts the LS and **retries the call**:

| Moment | `Cancelling N pending` | Outcome for the agent |
|---|---|---|
| `08-01 12:32` | 1 pending | restart + retry → result delivered |
| `08-01 13:25:06` | 1 pending | restart + retry → result delivered |
| `08-02 20:29:37` | 0 pending, but an in-flight tool | restart → correct response |
| `08-02 20:30:02` | 0 pending | nobody affected (background death) |

The residual defect is **narrow but real**: the `else` branch in
`tools_base.py:391-394` — if `affected_language` cannot be determined,
Serena logs `affected language is unknown. Not retrying.` and re-throws the
error outward. Plus any refusal of the restart itself. In these cases the
raw `LanguageServerTerminatedException` reaches the agent.

Our side today (`serena_backend.py:664-675`) on a transport error does
`_mark_lost()` and immediately returns to the agent
`serena {tool} transport error: {e}` — **without a single retry attempt**,
even though the supervisor is at that very moment bringing up a new backend.

### 1.4-bis Second root cause — ours, found by an integration test (round 5)

The first version of the feature passed all unit tests, but the integration
test `test_kill_during_in_flight_call_triggers_retry` consistently failed:
`restarts_total` grew (the supervisor honestly brought up a new backend),
but `ls_terminated_retries_total` stayed at **zero**, and the call hung for
exactly `_DISPATCH_BUDGET_S`. That is, in the most important scenario —
"Serena died while the call was ALREADY in flight" — the retry never started
at all.

The cause turned out to be not in the retry logic but in the transport
`_OwnedStdioSession`. It considered death to be **only EOF on the child's
`stdout`**. But Serena launches language servers as its own subprocesses,
and they **inherit the same stdout/stderr pipe**. After Serena dies, the
writing end of the pipe stays open in the grandchildren → `readline()` never
returns EOF → `ClientSession` does not learn of the break → the call waiting
for a response hangs until the end of the budget. Upward went a **timeout**,
not "backend terminated", so the classifier `_is_backend_terminated` did not
fire and there was no retry.

`asyncio.Process.wait()` is **not suitable** for the death signal for the
same reason: asyncio resolves it only when the process has exited **AND**
all stdio pipes are disconnected (`BaseSubprocessTransport._try_finish`),
and the pipes are held by the grandchildren. Measured: `wait()` did not
return within 5 s after `SIGKILL`.

**Solution.** In `_OwnedStdioSession.__aenter__` a death watcher was added
via `pidfd` — the same primitive already used by
`SerenaSupervisor._arm_monitor` (`os.pidfd_open` + `loop.add_reader` +
`asyncio.Event`, zero polling). When it fires, the read-stream is closed and
the MCP session normally fails all pending requests with
`Connection closed` (`mcp/shared/session.py:431-441`) — the text that
`_is_backend_terminated` recognizes. Additionally `_teardown` closes the
subprocess transport, otherwise each such teardown uselessly burned timeouts
on `wait()` for the corpse and left an unfinished transport (whose `__del__`
then fired on an already-closed loop).

Measured on a fake MCP server that deliberately spawns a grandchild with an
inherited stdout: before — the call hangs ≥10 s (the test fails by timeout),
after — the call fails in 0.76 s with recognizable text.

**The second layer of the same root cause: a request that missed the death.**
One watcher was not enough — the integration test kept failing. A stack
dump of all tasks on the live service (diagnostics capture, 12 s after
`SIGKILL`) showed: the watcher fires instantly, the supervisor brings up a
replacement in ~1.5 s, but the `_dispatch_tool` task still stood on the
first attempt for all 110 s.

Cause: failing pending requests is the job of the MCP session's
`_receive_loop` in its `finally`. It fails **only those requests that are
already registered**. When death lands in the window between "the call task
is created" and "the request is written into the session" (and this is the
typical case: `create_task` does not start execution until the nearest
`await`), our request is registered **after** that loop has exited — there
is nobody to fail it, and no response will ever come. The call lived to the
end of the budget, the dispatcher saw `timeout`, classified it as "not
death" (and correctly so, given the available data), and did no retry.

**Solution.** `SerenaBackend` got a `_lost_event`, armed in `_mark_lost`,
and `call_tool` waits for the response in a race with it (`_call_or_death`).
The transport propagates the child's death into `_mark_lost` via
`_OwnedStdioSession.set_on_death`. The event is reset in `start()`,
otherwise a restarted backend would inherit the death of the previous life
and fail its first healthy call.

Measured on the live service (diagnostics capture, two rounds in a row):
before — `call 110.0s`, `is_error=True` (text about the budget),
`ls_retries=0`; after — `call 1.8s`, `is_error=False`, `retries=1`,
i.e. the call is transparently completed on the new backend.

### 1.4-ter Adjacent finding: mutating tools return non-JSON (NOT fixed here)

While finalizing the integration test about ambiguous commit, the following
was discovered. Serena's mutating tools (`insert_after_symbol` and
neighbors) respond with the text `OK`, not JSON. The service honestly puts
it into `ToolResult.content_json` (`_to_tool_result` → `_flatten_content`),
and the bridge unconditionally does `json.loads` (`bridge_loop.py:216-220`)
and raises `ToolCallFailed: service returned malformed JSON`.

Measured directly via gRPC on a clean service: `is_error=False`,
`content_json='OK'`, the edit is present on disk. That is, an agent that
successfully applied an edit gets an error.

**This is a separate defect that existed before this feature**, and here it
is deliberately NOT fixed: the fix would change the contract "what the
bridge returns to the agent" for all tools at once, and such changes require
separate agreement (see the policy). Recorded as an independent task. The §6
integration test does not depend on the call outcome: its invariant lives on
disk (the marker occurs exactly once), so the finding does not affect the
feature.

### 1.5 Task formulation

The agent must not see this class of failures. If the LS/backend died but
the infrastructure (Serena inside itself, or our supervisor) recovered —
the call must be transparently repeated. If recovery failed — the agent
gets an **explanatory** text, not the raw
`LanguageServerTerminatedException`, and the text explicitly says that the
cause is not concurrent calls and not absolute paths.

---

## 2. Solution boundaries

**In scope:**
- a new retry class "LS terminated / transport error" in our service,
  **only for 6 read-only RPCs** (§4.2.0);
- a ban on automatic replay for 5 mutating RPCs + an honest message about
  ambiguous commit;
- event-driven waiting for the readiness of the new backend
  (generation protocol, no lost wakeup, §4.3);
- a server-side budget `_DISPATCH_BUDGET_S = 110 s` for the whole tool-RPC
  (both attempts wrapped in `wait_for`), so the retry fits within the
  bridge's client deadline (§4.4, option A);
- a **deliberate reduction** of the healthy timeout 120 → 110 s and a
  separate response `_DISPATCH_TIMEOUT_MSG` for that case;
- a counter `ls_terminated_retries_total` in `ProjectStatus`;
- agent-facing text without the raw exception.

**Explicitly out of scope:** a durable idempotency key / result journal
that would allow safely repeating mutating calls too.

**Out of scope (a separate step, after landing):**
- a PR to upstream Serena with `on_request("workspace/diagnostic/refresh", do_nothing)`
  and the two other refresh-capabilities. Upstream code is not touched now.

**We do not change:** the architecture, the gRPC contract (only an additive
field), the single-spawn-point policy for Serena (POLICY §8), the "polling
forbidden" policy.

---

## 3. Key architectural decision: where the retry lives

Two places were considered.

### Option A (rejected): inside `SerenaBackend.call_tool`

`call_tool` is a method of **that very object** that just died. After
`_mark_lost()` its `_session is None` forever: the supervisor does not
resurrect the old object, it creates a **new** one via `self._factory()`
and re-sets the handler
(`serena_supervisor.py:271-281` → `server.attach_backend(new_backend)` →
`grpc_server.py:248 set_tool_handler(backend.call_tool)`).

A retry inside the dead object would be doomed: it would re-call
`self._session`, which is already `None`, and return
`serena backend not running`. That would worsen the message, not fix the
problem.

### Option B (accepted): in `LSPServiceServer._dispatch_tool`

`grpc_server.py:525` is the only funnel through which all 11 tool-RPCs pass
(lines 716-766). Critically: it reads `self._tool_handler` **anew on each
attempt**, and the supervisor on a successful restart re-sets it to the
`call_tool` of the new live backend.

So a retry at this level naturally lands on the new backend without any
knowledge of the backend object's internals. This same place already owns
`_backend_ready_event` — a ready-made event primitive for readiness.

**Solution: the retry is implemented in `_dispatch_tool`.**

### 3.1 Two different recovery paths (review, REQUIRED)

The first revision described everything as "switching to a new backend".
That is true for only one of the two paths. The difference matters because
what exactly we wait for before repeating depends on it.

**Path (a) — death of Serena's MCP transport.**
Symptom: `transport error` / `broken pipe` / `backend not running`.
Mechanics: `SerenaBackend.call_tool` catches the exception → synchronously
`_mark_lost()` → `on_lost` → `mark_backend_not_ready()` (event replaced,
`_backend_ready=False`) → the supervisor creates a **new** backend object →
`attach_backend` → `set_tool_handler(new call_tool)` →
`mark_backend_ready()` (generation +1).
The repeat must go to the **new** handler — this is exactly where placing it
in `_dispatch_tool` is not merely convenient but necessary.
Wait: for a generation change (§4.3).

**Path (b) — `LanguageServerTerminatedException` as a normal
`is_error`-result from a LIVE Serena.**
Symptom: the text `LanguageServerTerminatedException: ...` inside a normal
MCP response (exactly what the agent saw in the original report).
Mechanics: the MCP transport is intact, Serena is alive, `_mark_lost` is
**not** called, `_backend_ready` stays `True`, the generation does not
change, the handler is the same.
The repeat goes to the **same** backend — and that makes sense: the next
call re-triggers Serena's internal recovery
(`tools_base.py:380-390` restarts the LS and repeats), and our repeat covers
exactly the case where its own retry did not work
(the `affected language is unknown. Not retrying.` branch).
Wait: **no** wait for a generation change — immediately the second attempt.

Distinguishing in code: after a terminated-result we look at `_backend_ready`
and the generation. If `_mark_lost` has already run (`_backend_ready is False`)
— path (a), wait for recovery. Otherwise — path (b), repeat immediately.

Both paths obey the replay ban for mutating RPCs (§4.2.0) and the common
budget (§4.4).

---

## 4. Detailed design

### 4.1 Classification: what counts as "LS terminated"

A new predicate `_is_backend_terminated(message: str) -> bool` in
`serena_backend.py` (next to `_is_transient_cancel`, exported).

We match by text — there is no structured code on the MCP surface, exactly
as in the transient case (§1.4 of that design). Markers:

```python
_TERMINATED_MARKERS = (
    "languageserverterminatedexception",
    "language server stdout read process terminated",
    "language server process terminated",
    "transport error",
    "broken pipe",
    "serena backend not running",
    "serena backend torn down mid-call",
)
```

The match is case-insensitive, by substring. Justification of each marker:
the first three are direct forms from `solidlsp/ls_process.py:595,634,642`;
`transport error` and `broken pipe` — our wrapper
(`serena_backend.py:674`) and `_send_payload:681`; the last two are states
of our own backend between death and restart (`serena_backend.py:603,624`).

**Strictness (following the A1 example of the transient design):** the
marker list is closed and explicit. We do NOT retry "any error" — see §6,
mutation MUT-T2. In particular, `ValueError` from `validate_relative_path`
(a path outside the project), "symbol not found", timeout, and transient
`-32800/-32801` do not fall under this predicate.

Classification order in `_dispatch_tool`: transient retries have already run
**inside** the backend (they are cheaper and do not require a restart); only
the final result arrives here.

### 4.2 Algorithm in `_dispatch_tool`

#### 4.2.0 Replay is allowed only for read-only RPCs (review R1, blocker)

The first design revision allowed automatic replay for all 11 RPCs based on
the argument "an edit is applied after a successful LS response, so when the
LS dies no edit happened". **The argument is wrong**, and this is confirmed
by reading the upstream code:

`serena/tools/symbol_tools.py:638-641` (`InsertAfterSymbolTool.apply`):
```python
with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
    code_editor = self.create_code_editor()
    code_editor.insert_after_symbol(...)      # 1. the edit is ALREADY written to disk
    return diagnostics_context.format_result(SUCCESS_RESULT)   # 2. but HERE...
```
`DiagnosticsContext.format_result` (`tools_base.py:474-493`) **after**
writing the file talks to the language server for diagnostics
(`PublishedDiagnosticsSnapshot`, `DiagnosticsDiff`). It is exactly at this
step that the LS dies in our §1.2 scenario.

Hence a real **ambiguous commit** window: the edit is on disk, but the MCP
`ToolResult` never reached us. The same is caused by Serena's transport
death (`transport error` / `broken pipe`) after an edit was applied.
The loss of the acknowledgement is not cured by addressing via `name_path`.

Consequences of a blind replay: `insert_before_symbol` /
`insert_after_symbol` are **not idempotent** — a repeat would duplicate the
`body`. `rename_symbol`, `replace_symbol_body`, `safe_delete_symbol` also
give no guarantee of a safe repeat.

**Solution:**

| Class | RPC | Behavior on terminated |
|---|---|---|
| read-only (6) | `FindSymbol`, `FindReferencingSymbols`, `FindDeclaration`, `FindImplementations`, `GetSymbolsOverview`, `GetDiagnosticsForFile` | automatic replay (§4.2.1) |
| mutating (5) | `RenameSymbol`, `ReplaceSymbolBody`, `InsertBeforeSymbol`, `InsertAfterSymbol`, `SafeDeleteSymbol` | **replay forbidden**; exactly one call; return `_LS_TERMINATED_MUTATING_MSG` (§4.5) |

The mutating-RPC set is defined by an explicit constant
`_MUTATING_RPCS: Final[frozenset[str]]` in `grpc_server.py`. It is derived
from `_RPC_TO_TOOL` not automatically but by enumeration — a new mutating
tool must require a deliberate addition.

Idempotency via a durable idempotency key / result journal would allow a
safe replay for mutating calls too, but that is a separate subsystem with
persistent state — **out of scope** for this change.

#### 4.2.1 Algorithm

```
deadline = now + _DISPATCH_BUDGET_S            # 110 s, §4.4

attempt 1: result = await wait_for(self._tool_handler(rpc, args), remaining)
    TimeoutError -> return _DISPATCH_TIMEOUT_MSG (§4.4), NOT terminated
if result is NOT terminated      -> return as is (common case)
if result is terminated:
    if rpc in _MUTATING_RPCS:
        # ambiguous commit: the edit may have been applied
        -> return _LS_TERMINATED_MUTATING_MSG (§4.5), WITHOUT a second call
    if path (a) (§3.1):                       # _backend_ready == False
        wait for a GENERATION CHANGE (event-driven, §4.3),
        bounded by min(_LS_TERMINATED_READY_WAIT_S, remaining)
        did not wait -> return _LS_TERMINATED_MSG (§4.5)
    # path (b): nothing to wait for, the handler is the same
    if remaining < _MIN_ATTEMPT_S -> return _LS_TERMINATED_MSG (R2a)
    note_ls_terminated_retry()          # counter, §4.6
    attempt 2: result = await wait_for(self._tool_handler(rpc, args), remaining)
        TimeoutError -> return _LS_TERMINATED_MSG
    return result (success OR _LS_TERMINATED_MSG if terminated again)
```

`self._tool_handler` is re-read on **every** attempt (§3, MUT-T8).
Both attempts are wrapped in `wait_for` under one common `deadline` — this
is option A from §4.4 (MUT-T15).

**Exactly one retry.** Rationale: LS death is not "server busy"
(where a loop of 3 attempts is justified) but an event that is either fixed
by a restart or not. A second-third pass over dead infrastructure would only
burn the client deadline and delay a clear message.
`_MAX_RESTARTS` of the supervisor already limits the restart frequency.

**Exceptions from `_tool_handler` are classified too.** Today `except
Exception` on line 535 immediately does `context.abort(INTERNAL)`.
If the exception text passes `_is_backend_terminated`, the behavior is the
same as for an `is_error`-result — **including the replay ban for mutating
RPCs**: for them the exception is converted to
`_LS_TERMINATED_MUTATING_MSG` without a second call. Otherwise — the
previous `abort`, unchanged.

### 4.3 Readiness wait — event-driven, no polling

The **existing** `self._backend_ready_event` is used
(`grpc_server.py:147`). The mechanics are already right for our task:
`mark_backend_not_ready` (line 197) **replaces the event with a new one**,
and `mark_backend_ready` (line 183) sets it.

Sequence on death:
`_mark_lost()` → `on_lost` → `mark_backend_not_ready()` → a fresh
non-signaled event; then the supervisor → `attach_backend` →
`mark_backend_ready()` → the event is signaled.

**Error of the first revision (review, REQUIRED).** The design claimed that
a snapshot of the event reference "survives the replacement". That is false
and was reproduced experimentally: a waiter took `E1`;
`mark_backend_not_ready` replaced the attribute with `E2`;
`mark_backend_ready` sets `E2` — the waiter on `E1`
**never wakes up** (lost wakeup, caught only by timeout).

**Why the production path is nevertheless usually safe today.** On
transport death, `_mark_lost()` is called **synchronously inside**
`SerenaBackend.call_tool` (`serena_backend.py:671`) — i.e. the event
replacement happens **before** `_dispatch_tool` receives the `ToolResult`
and starts waiting at all. The snapshot is already taken from `E2`. Verified
experimentally: this order wakes up correctly. But relying on this as an
invariant is impossible — the order hangs on details of a third-party
callback path.

**Waiting protocol (generation-based, event-driven).**

A monotonic counter `_backend_ready_generation: int` is added,
incremented in `mark_backend_ready()`. The wait:

```python
deadline = loop.time() + _LS_TERMINATED_READY_WAIT_S
gen0 = self._backend_ready_generation
while True:
    if self._backend_ready and self._backend_ready_generation != gen0:
        return True                      # a NEW backend came up
    remaining = deadline - loop.time()
    if remaining <= 0:
        return False
    event = self._backend_ready_event    # re-read the CURRENT one each iteration
    try:
        await asyncio.wait_for(event.wait(), timeout=remaining)
    except asyncio.TimeoutError:
        return False
```

Key point: `mark_backend_not_ready` additionally **wakes the old event
before replacing it** (`old.set()`), so a waiter parked on `E1` is
guaranteed to wake, re-read the current `_backend_ready_event`, and continue
waiting on `E2` within the same single `deadline`.

This is **not polling**: there are exactly as many iterations as there were
real replacements/wakeups (normally 1-2), no tick, no `sleep`; each
iteration is a blocking wait on an event under one common bounded deadline.

Comparing generations also closes the reverse race: if the backend was
already ready and `_backend_ready` is true with the same generation, we do
not count that as recovery and wait specifically for a generation change.

**Case (b) from §3.1** (live Serena, `_mark_lost` not called, the event
already signaled, the generation unchanged) is handled by a separate branch
— there is no generation-change wait at all there, see §3.1.

### 4.4 Budget

The constant `_LS_TERMINATED_READY_WAIT_S: Final = 30.0` is the ceiling for
waiting for the new backend to come up.

Rationale for the value: per the logs, a cold start of the Kotlin LS from
`Starting language server process` to `Kotlin LSP ready` takes
~3.5-4 s (`08-02 20:29:37,333` → `20:29:41,051`); the Serena process
comes up faster. 30 s gives a ~7x margin for a slow machine and at the same
time is noticeably less than the client deadline.

**The budget problem.** The bridge's client deadline
`bridge_loop.py:56 _CALL_TIMEOUT_S = 120.0` is **exactly equal** to the
budget of one attempt `SerenaBackend.call_timeout_s = 120.0`
(`serena_backend.py:383`). Even the first attempt can exhaust the whole
client deadline; then the retry will not be delivered — the bridge falls off
with `DEADLINE_EXCEEDED` and turns that into `ToolCallFailed: DEADLINE_EXCEEDED`
(`bridge_loop.py:204-207`), i.e. the agent again gets a raw infrastructure
failure instead of an explanation.

#### Decision: option A — a hard dispatch cap

The review (round 2) rightly pointed out: `_DISPATCH_BUDGET_S` by itself
limits nothing while the first attempt is not wrapped. Therefore **every**
`await` to the handler — both the first attempt and the second — is wrapped
in `asyncio.wait_for(..., remaining)` under one common deadline
`_DISPATCH_BUDGET_S: Final = 110.0`.

Option B (do not reduce the healthy timeout; retry only when there is
reserve) was rejected: it leaves a hole exactly at the deadline boundary —
where failure is most likely — and the §4.7 guarantee becomes conditional.

**This is a deliberate change to the observable contract, and we record it
explicitly:** a healthy call that could legally last 110–120 s before will
now be cancelled at 110 s. The phrasing "the healthy path does not change"
from the previous revision of §4.4 and R2 is **wrong and removed**.

Why a 10 s reserve is an acceptable product contract (on data, not
feelings): across all the logs in the local Serena logs directory for
08-01/08-02 — **3003 measurements** of Serena task durations, the maximum
is **20.7 s**, the next values are 19.4 / 18.6 / 16.4 / 15.9 s. The range
110–120 s is empirically **empty**: no real call even came close to the
ceiling. We give away 10 s from an unreachable zone in exchange for the
guarantee that the explanation is delivered. If such a call does appear
anyway, it gets a correct separate response (below), not a raw gRPC failure.

#### A separate result for the dispatch-budget timeout

The timeout of the **first** attempt must not be called "LS terminated" —
no server death was observed at that moment. A third constant is
introduced, `_DISPATCH_TIMEOUT_MSG`:

```
serena {tool}: the call exceeded the service-side budget of {budget:.0f}s
and was cancelled. The language server did not report a crash — it is
most likely still indexing a large project. This is NOT a limit on
concurrent tool calls and NOT caused by an absolute path. Retry in a few
seconds, or narrow the request (pass relative_path to scope it to one
file).
```

Dispatch response classification:
- the first attempt exhausted the budget, no terminated observed →
  `_DISPATCH_TIMEOUT_MSG`;
- terminated observed, read-only, retry did not help → `_LS_TERMINATED_MSG`;
- terminated observed, mutating → `_LS_TERMINATED_MUTATING_MSG`.

#### Budget spending rules

- the common deadline starts at the beginning of `_dispatch_tool`;
- first attempt: `wait_for(handler(...), remaining)`;
- readiness wait: `min(_LS_TERMINATED_READY_WAIT_S, remaining)`;
- before the second attempt: if the remainder is less than
  `_MIN_ATTEMPT_S`, no retry — immediately `_LS_TERMINATED_MSG` (a delivered
  explanation is worth more than an undelivered retry, R2a);
- second attempt: `wait_for(handler(...), remaining)`.

Result: the worst-case wall time of any tool-RPC `≤ _DISPATCH_BUDGET_S = 110 s`
with a client deadline of 120 s. The bridge's `_CALL_TIMEOUT_S` and the
backend's `call_timeout_s` are unchanged — the gRPC contract is untouched;
the backend's internal budget (120 s) simply becomes unreachable because the
outer cap is stricter.

#### Cancellation and state cleanliness

`wait_for` cancels the `SerenaBackend.call_tool` coroutine. Its body runs
under `async with self._lock` (`serena_backend.py:619`);
`asyncio.Lock` is released on unwinding `CancelledError`, so the lock does
not "stick". The session belongs to the long-lived `_session_task` and is
not torn by the caller's cancellation. This is verified by a test
(`test_cancelled_attempt_leaves_backend_usable`): after a budget-cancelled
attempt, the next call on the same backend passes normally.

### 4.5 Agent-facing message

When the retry did not help (did not wait for readiness, or the second
attempt was terminated again) — `pb.ToolResult(is_error=True, ...)` is
returned with the constant text `_LS_TERMINATED_MSG` (in `serena_backend.py`,
next to `_TRANSIENT_EXHAUSTED_MSG`):

```
serena {tool}: the language server process died and did not come back in
time. This is an infrastructure fault on our side, not a problem with
your request: it was NOT caused by concurrent tool calls, and NOT by
using an absolute path or a broad search. The known upstream trigger is
the Kotlin language server crashing on an unhandled
'workspace/diagnostic/refresh'. Retry the same call in a few seconds; if
it keeps failing, check `deep-worker lsp status`.
```

Text requirements (following the §4.3 example of the transient design):
1. names the cause (the LS process died);
2. says this is our infrastructure, not the agent's request;
3. **explicitly** dismisses both false hypotheses — concurrent calls and
   absolute paths / broad search (exactly what the agent invented in the
   report);
4. gives an action (retry; look at `lsp status`);
5. **does not contain** the raw `LanguageServerTerminatedException` —
   pinned by mutation MUT-T3.

The raw text of the original error goes to the log (`_log.warning`), not to
the agent.

**A separate text for mutating RPCs** — `_LS_TERMINATED_MUTATING_MSG`.
Here we cannot say "just retry": the edit may have been applied (§4.2.0),
and a blind repeat would duplicate the insertion.

```
serena {tool}: the language server process died while this edit was in
flight, so the outcome is UNKNOWN — the edit may or may not have been
written to the file. It was NOT retried automatically, on purpose,
because repeating it could apply the same change twice. This is an
infrastructure fault on our side: it was NOT caused by concurrent tool
calls, and NOT by using an absolute path or a broad search. Read the
affected file/symbol first to see whether the change is already there,
then decide whether to repeat the call.
```

Requirements for it: (1) honestly declares the outcome unknown; (2)
explains why the repeat was not done automatically; (3) dismisses the same
two false hypotheses; (4) gives a safe action — check the file first;
(5) without the raw exception.

### 4.6 Observability

A new field in `lsp_service.proto`:
```proto
int64 ls_terminated_retries_total = 18;   // StatusReply
```
Additive; number 18 is free (the last occupied one is 17,
`transient_retries_total`). Requires regenerating `_pb2.py`.

- counter `LSPServiceServer._ls_terminated_retries_total`;
- method `note_ls_terminated_retry()` — unconditional increment, following
  the example of `note_transient_retry` (`grpc_server.py:276`);
- propagated into `ProjectStatus` (`grpc_server.py:649`) and into the
  output of `deep-worker lsp status` (`service_cli.py:146,185`) as
  `ls_terminated_retries=N`.

The increment happens **before** the second attempt, exactly once per
triggered retry — we count "how many times we covered an LS death".

### 4.7 Invariant (the main acceptance criterion)

> **Read-only RPC:** if Serena or our supervisor handled it themselves —
> the agent **sees nothing at all**; it gets a normal result.
> If they did not — the agent gets `_LS_TERMINATED_MSG`.
>
> **Mutating RPC:** a repeat is never performed; the agent gets
> `_LS_TERMINATED_MUTATING_MSG` with an honest "outcome unknown, check the
> file".
>
> **A slow healthy call** (no termination observed, budget exhausted)
> gets a third, separate text `_DISPATCH_TIMEOUT_MSG` — it must not be
> passed off as an LS death.
>
> **In all cases** the agent never sees the raw
> `LanguageServerTerminatedException` / `transport error` /
> `DEADLINE_EXCEEDED`, and any tool-RPC fits within
> `_DISPATCH_BUDGET_S = 110 s` < the client deadline of 120 s.

---

## 5. Risks

**R1. Re-calling a mutating tool — CLOSED, not "accepted".**
The first revision accepted this risk based on a wrong argument
(analysis in §4.2.0: `code_editor` writes the file BEFORE talking to the LS
for diagnostics, plus the transport can die after the edit is applied —
ambiguous commit). The risk is removed constructively: for 5 mutating RPCs
automatic replay is **forbidden** (§4.2.0), the agent gets an honest
"outcome unknown" (§4.5). The residual part — the agent may repeat
manually, but deliberately and after checking the file, which is strictly
better than a blind auto-repeat.

**R2. Changing the observable contract: healthy timeout 120 → 110 s.**
Option A (§4.4) wraps `wait_for` around **both** attempts, including the
first. Consequence: a healthy call that could previously last 110-120 s is
now cancelled at 110 s and gets `_DISPATCH_TIMEOUT_MSG`.
The previous revision's phrasing "the healthy path does not change" was
wrong — the risk is real and deliberately accepted.

Rationale for acceptability — on data: 3003 measurements of Serena task
durations in the logs for 08-01/08-02, maximum 20.7 s; the range 110-120 s
is empty. In exchange we get the guarantee that the explanation reaches the
agent instead of `DEADLINE_EXCEEDED`. Pinned by the test
`test_healthy_slow_call_gets_dispatch_timeout_not_terminated`.

**R2b. A budget cancellation must not corrupt the backend.** `wait_for`
cancels `call_tool` under `async with self._lock`. Verified
experimentally: on unwinding `CancelledError`, `asyncio.Lock`
is released and taken again; the session belongs to a separate long-lived
task and is not torn. Pinned by
`test_cancelled_attempt_leaves_backend_usable`.

**R2a. Starvation of the budget by the first attempt.** If the first
attempt ate almost all of `_DISPATCH_BUDGET_S`, there is no retry at all —
the agent gets `_LS_TERMINATED_MSG` without a second attempt. This is a
deliberate compromise: a delivered explanation is worth more than an
undelivered retry. Pinned by the test
`test_no_retry_when_budget_exhausted`.

**R3. A false-positive predicate match** → an extra retry of a healthy
error. Mitigated by the closed marker list (§4.1) and mutation MUT-T2.

**R4. Drift of upstream texts.** If Serena changes its wording, the
predicate stops matching and we return to today's behavior (an error to the
agent) — a soft degradation, not a breakage. The integration test on live
Serena (§6) will catch the drift.

---

## 6. Verification plan

### Unit (`tests/unit_tests/lsp/test_serena_ls_terminated_retry.py`)

1. `test_terminated_result_is_retried_and_succeeds` — the first attempt is
   terminated, the second is ok → the agent gets a success, `is_error=False`.
2. `test_terminated_exception_is_retried` — the handler **raises**
   an exception with terminated-text → retry, success.
3. `test_retry_uses_the_new_handler` — key: after death the handler is
   replaced (as the supervisor does), the retry must go to the **new** one.
4. `test_waits_for_backend_ready_before_retry` — path (a): the retry does
   not start until the readiness generation changes.
5. `test_wait_survives_event_replacement` (replacing the erroneous test 5) —
   the waiter is parked on `E1`; `mark_backend_not_ready` wakes `E1` and
   sets `E2`; `mark_backend_ready` sets `E2` → the waiter wakes and
   retries. **Without** the wake-protocol the test must fail by timeout
   (lost wakeup reproduced experimentally).
5a. `test_multiple_replacements_no_lost_wakeup` — two or three
    replacements in a row within one deadline → the retry still happens
    exactly once.
5b. `test_same_generation_is_not_treated_as_recovery` — the event is
    signaled, the generation did not change → this is not path (a).
6. `test_exhausted_returns_explanatory_message` — both attempts terminated
   → the text `_LS_TERMINATED_MSG`, without `LanguageServerTerminatedException`
   in the text.
7. `test_ready_wait_timeout_returns_explanatory_message` — the backend did
   not come up within the budget → the same explanatory text.
8. `test_non_terminated_error_is_not_retried` — "symbol not found",
   a transient cancel, a path `ValueError` → exactly 1 handler call.
9. `test_only_one_retry` — terminated always → exactly 2 calls.
10. `test_counter_incremented_once_per_retry` — `note_ls_terminated_retry`
    is called exactly once.
11. `test_predicate_is_strict` — synchronous: the predicate is True on all
    markers; False on "not found"/"-32800"/empty string.
12. `test_background_death_without_agent_call` — the `08-02 20:30` scenario:
    the LS dies between calls, no RPC in flight → neither a retry nor a
    counter; the next call passes normally.
13. `test_in_flight_tool_at_moment_of_death` — the `08-01 12:32` scenario:
    a call in flight at the moment of death → the agent gets the result.

**Mutating-RPC (review R1) — mandatory:**

14. `test_mutating_rpc_is_never_replayed_on_terminated_result` —
    parameterized over all 5 mutating RPCs; a terminated-result → the
    handler is called **exactly once**.
15. `test_mutating_rpc_is_never_replayed_on_exception` — the same, but the
    handler **raises** a terminated-exception → exactly one call.
16. `test_mutating_rpc_returns_ambiguous_commit_message` — the text
    `_LS_TERMINATED_MUTATING_MSG`: contains "unknown"/"not retried",
    the advice to check the file, does not contain the raw exception.
17. `test_rpc_classification_is_total_and_disjoint` (strengthened per
    review): we take the **full** set of registered RPCs — the keys of
    `_RPC_TO_TOOL` (11 items) — and assert that
    `read_only | mutating == all RPCs` and `read_only & mutating == empty`.
    Thus a new RPC cannot silently bypass the guard: it would fall into
    neither set and the test would fail. Additionally the composition of
    `_MUTATING_RPCS` is fixed (5 names).

**Budget — option A (hard dispatch cap):**

18. `test_retry_fits_client_deadline` — the total time of an RPC with a
    retry does not exceed `_DISPATCH_BUDGET_S`.
18a. `test_first_attempt_is_capped_by_dispatch_budget` (key per review):
    the first-attempt handler hangs longer than the budget → it is
    **cancelled** by the common deadline, not sat through to
    `call_timeout_s`. Proves that the cap applies to the first attempt, not
    only to the wait and the second.
19. `test_no_retry_when_budget_exhausted` — the first attempt ate the budget
    → no second attempt.
19a. `test_healthy_slow_call_gets_dispatch_timeout_not_terminated` —
    a slow **non**-terminated handler, cancelled by the budget →
    `_DISPATCH_TIMEOUT_MSG` is returned; a false `_LS_TERMINATED_MSG`
    is forbidden and the retry counter does not grow.
19b. `test_cancelled_attempt_leaves_backend_usable` — after a
    budget cancellation the lock is free, the session is alive, and the
    next call passes.

**Path (b) (§3.1):**

20. `test_live_serena_terminated_retries_same_handler` — `_mark_lost` was
    not called, `_backend_ready` True → the retry is immediate, into the
    same handler, without a generation wait.

Metrics (`test_grpc_server_metrics.py`): `ls_terminated_retries_total` in
`ProjectStatus` and its propagation.

### Mutation checks

| ID | Mutation | Must fail |
|---|---|---|
| MUT-T1 | do not retry terminated (remove the branch) | 1, 2, 3, 13 |
| MUT-T2 | retry any exception / any `is_error` | 8 |
| MUT-T3 | return the raw error text instead of `_LS_TERMINATED_MSG` | 6, 7 |
| MUT-T4 | skip the readiness wait (retry immediately) | 4 |
| MUT-T5 | allow unlimited retries | 9 |
| MUT-T6 | increment the counter on every attempt | 10 |
| MUT-T7 | do not wake the old event on replacement (lost wakeup) | 5, 5a |
| MUT-T8 | read `_tool_handler` once before the loop | 3 |
| MUT-T9 | allow replay for mutating RPCs | 14, 15 |
| MUT-T10 | return the plain `_LS_TERMINATED_MSG` for mutating | 16 |
| MUT-T11 | treat the mutating set as empty | 14, 17 |
| MUT-T12 | ignore `_DISPATCH_BUDGET_S` | 18, 19 |
| MUT-T15 | do not wrap `wait_for` around the FIRST attempt | 18a |
| MUT-T16 | return `_LS_TERMINATED_MSG` on dispatch-timeout | 19a |
| MUT-T17 | reduce the RPC classification to an incomplete set | 17 |
| MUT-T13 | on path (b) wait for a generation change | 20 |
| MUT-T14 | compare only `_backend_ready`, without the generation | 5b |

### Integration (`tests/integration_tests/lsp/test_lsp_ls_terminated_retry_e2e.py`)

Following the example of `test_lsp_transient_retry_e2e.py`, on live Serena,
without mocks.

**The review's note about the race is accepted.** The naive scenario "kill
Serena, then call a tool" is nondeterministic: the supervisor may recover
before the call, then the call passes without a retry and the counter stays
at zero — the test would be green without checking anything. Therefore two
separate tests:

1. `test_background_kill_then_call_succeeds` — kill Serena, wait for the
   **event** of recovery (not sleep), then call a tool. Checks only the
   call's success. The counter is **not** checked: there may have been no
   retry, and that is correct.
2. `test_kill_during_in_flight_call_triggers_retry` — deterministic:
   a barrier on the handler — the call enters the first attempt, the test
   waits for that fact, only then `SIGKILL` on the real pid of Serena,
   after which the barrier is released. Guarantees that death landed
   **inside** the call window. Both the successful result and
   `ls_terminated_retries >= 1` in `lsp status` are checked.
3. `test_mutating_rpc_not_replayed_live` — on live Serena, a mutating-RPC
   is not repeated on backend death: the file contains no duplicate
   insertion.

### Regression

A full unit run, `ty`, integ lsp with `-m integration`.

---

## 7. Affected files

| File | Change |
|---|---|
| `lsp/service/serena_backend.py` | + `_is_backend_terminated`, `_TERMINATED_MARKERS`, `_LS_TERMINATED_MSG`, `_LS_TERMINATED_MUTATING_MSG`, `_DISPATCH_TIMEOUT_MSG`; + pidfd-watcher of the child's death in `_OwnedStdioSession` and transport closing in `_teardown` (§1.4-bis) |
| `lsp/service/grpc_server.py` | the retry in `_dispatch_tool` with `wait_for` on both attempts, `_MUTATING_RPCS`, `_DISPATCH_BUDGET_S`, the generation counter + waking the old event in `mark_backend_not_ready`, `note_ls_terminated_retry`, propagation into the status |
| `lsp/service/proto/lsp_service.proto` | + field 18 |
| `lsp/service/proto/lsp_service_pb2.py` | regeneration |
| `lsp/service/service_cli.py` | the `ls_terminated_retries=` output |
| `tests/unit_tests/lsp/test_serena_ls_terminated_retry.py` | new |
| `tests/unit_tests/lsp/test_grpc_server_metrics.py` | + counter tests |
| `tests/unit_tests/lsp/test_serena_backend_child_death.py` | new: RED/GREEN for §1.4-bis (a fake MCP server with a grandchild holding stdout) |
| `tests/integration_tests/lsp/test_lsp_ls_terminated_retry_e2e.py` | new |

---

## 8. Open question for the user

A PR to upstream Serena (`on_request("workspace/diagnostic/refresh",
do_nothing)` + `inlineValue`/`inlayHint` refresh in
`kotlin_language_server.py`, following the example of
`svelte_language_server.py:594`) removes the root cause for all Serena
users. It is being prepared as a **separate step** after this feature
lands: a draft patch and PR text as a file in our repository; upstream is
not touched. The decision "send or not" is the user's.
