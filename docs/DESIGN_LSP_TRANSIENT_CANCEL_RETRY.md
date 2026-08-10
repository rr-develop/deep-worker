# DESIGN — LSP: retrying transient language-server cancellations

Status: draft for review
Branch: `dev`
Related: `DESIGN_LSP_SERENA_BACKEND_ATTACH.md`,
`DESIGN_LSP_MULTI_AGENT_STRESS.md`, `DESIGN_LSP_OBSERVABILITY.md`

## §1. Problem

The agent got a `cancelled (-32800)` error while calling the symbol tools
and drew the conclusion: "serena calls must be made strictly
sequentially; parallel calls cancel each other".

The conclusion is wrong, but **the defect is real and it is ours**.
Analysis:

1. `-32800` (`RequestCancelled`) is an LSP-protocol code. The language
   server cancels its own in-flight requests while a background
   reindex / workspace mutation is running. This is a **transient**
   state: the server is literally saying "busy, retry".
2. We have **no retry**. `SerenaBackend` hands the cancellation to the
   agent as an ordinary tool error.
3. The error text speaks of the protocol (`cancelled (-32800)`) rather
   than the state ("indexing in progress"). The agent cannot understand
   that a simple retry is all that is needed, and builds false theories
   about concurrency.

Parallelism does **not** create the problem: a single call that lands in
the same mutation window fails exactly the same way. Parallelism only
raises the chance of hitting it.

### User requirements (verbatim)

- If we can retry the call — retry it.
- Otherwise return a **specific** error to the agent: indexing is in
  progress, the server is not ready yet, it can be retried later.
- The agent **must not** conclude that two tools cannot be called at
  once.
- The client must send the **full** set of what it is ready to handle at
  the handshake. There are no restrictions here.

## §2. How the cancellation actually reaches us (established by reading the code)

This is the key fact that determines where the fix goes. Verified against
the solidlsp sources (inside Serena) and our code.

The error path:

1. The language server answers with a JSON-RPC error with `code = -32800`.
2. solidlsp `ls_process.py:380` wraps it:
   `SolidLSPException("Error processing request ...", cause=result.error)`.
3. `LSPError.__str__` (`lsp_protocol_handler/server.py:82-83`) yields the
   format `"<message> (<code>)"` — hence the literal `cancelled
   (-32800)` in the agent's report.
4. Serena catches this in its MCP layer and returns a **successful**
   MCP response with `CallToolResult(isError=True, content=[text])`.
5. On our side this lands **not** in an `except` but in `_to_tool_result`
   (`serena_backend.py:603-619`), which sees `isError=True` and
   returns `pb.ToolResult(is_error=True, error_message=<text>)`.

**Conclusion: there is no transport exception.** The cancellation arrives
as an ordinary result with an error flag. So the fix must work on the
**text** of `error_message`, not on the exception type. The
`except Exception` branch (`:556`) is irrelevant to this scenario — it is
about pipe death.

Practical consequence: the current `_mark_lost()` does **not** fire on a
cancellation, and that is correct — the backend is alive. That behavior
must not change.

## §3. What is NOT in scope

- We do not change the single-flight (`asyncio.Lock`) — the
  serialization is correct and pinned by tests.
- We do not change the gRPC RPC contracts. **Amendment A3:** one
  **additive** `StatusReply` field +17 (`transient_retries_total`)
  is still added — see §4.5/§5. Wire-compat is preserved: a new
  number, no renumbering, no RPC change.
- We do not touch `bridge_loop`/`bridge_stdio` on the protocol side.
- We do not patch Serena/solidlsp — we work on our own side.
- We add no threads (policy).

## §4. Solution

### §4.1. Recognizing transient cancellations

A new predicate module in `serena_backend.py`:

```
_TRANSIENT_CODE_RE: Final = re.compile(r"\(-3280[01]\)|\B-3280[01]\b")
def _is_transient_cancel(message: str) -> bool
```

Matching is by the code in the text (`-32800` / `-32801`), not by the
word `cancelled`: the word is localizable and appears in foreign
contexts, while the code is a stable part of the protocol
(`LSPErrorCodes`).

**Amendment A1 (strict pattern).** A bare substring `-32800` is
unacceptable: it would match inside a larger number (e.g. `-328001`).
We match:

1. the parenthesized form `(-32800)` — exactly what
   `LSPError.__str__` produces (`server.py:82-83`), including inside
   `caused by ... (-32800)`;
2. or a standalone code with word boundaries.

Tests must cover both real forms (`cancelled (-32800)` and
`... (caused by cancelled (-32800))`) and the negative case `-328001`.

If solidlsp ever starts returning a structured error code over MCP, we
switch to it; for now the text is the only available signal.

Why `-32801` (`ContentModified`) is included: the same nature (the
workspace changed under the request), the same correct reaction. solidlsp
has a generic retry for `-32801`, but it is **opt-in per method list**,
and it is not registered for Kotlin symbol navigation — so `-32801` also
reaches us.

### §4.2. Retry inside `call_tool`

The retry is placed **inside** `async with self._lock`, around
`session.call_tool`.

**Amendment A4 (conversion site).** Today `_to_tool_result` is called
**outside** the lock (`serena_backend.py:568`). Since the decision
"is this transient" is made from the already-converted `error_message`,
all three steps move inside the lock: `call_tool` → `_to_tool_result` →
`_is_transient_cancel` → (on a match) the next attempt. Otherwise we
would have to take the lock again, giving up our place in the queue.
Pinned by the test `peak_concurrent == 1`.

Rationale for the location:

- Releasing the lock between attempts is not allowed: the call would go
  to the end of the queue and lose its already limited budget.
- Holding the lock is correct: the server is busy with a mutation anyway;
  handing over the queue is pointless.

Parameters:

| Constant | Value | Rationale |
|---|---|---|
| `_TRANSIENT_MAX_ATTEMPTS` | 3 | 1 original + 2 retries |
| `_TRANSIENT_BACKOFF_S` | 0.25, 0.5 | exponential, ≤0.75 s total |

The wait is `await asyncio.sleep(...)` (not `time.sleep`): the lock is
held, but the event loop is not blocked. No new threads.

**Time budget.** Total backoff ≤0.75 s against a client deadline of 120 s
(`bridge_loop.py:56`) — 0.6 %. There is no risk of "instead of a clear
error the agent gets DEADLINE_EXCEEDED".

**Interaction with `wait_for` — amendment A2 (series-wide budget).**

The original variant "`wait_for(call_timeout_s)` per attempt, no overall
ceiling" had a hole: 3 attempts × 120 s + backoff under the lock give up
to ~6 minutes. Both consequences are unacceptable:

- the client deadline (`bridge_loop.py:56`, 120 s) would expire first,
  and the agent would see `DEADLINE_EXCEEDED` instead of the
  explanatory text this whole thing exists for;
- the whole queue would stall for minutes because the lock is held.

The correct scheme is a **per-attempt timeout under a series-wide
ceiling**:

```
deadline = loop.time() + self._call_timeout_s   # one budget per call
for attempt in range(1, _TRANSIENT_MAX_ATTEMPTS + 1):
    remaining = deadline - loop.time()
    if remaining < _MIN_ATTEMPT_S:              # 0.05 s
        → exit as "attempts exhausted" (§4.3)
    result = await wait_for(session.call_tool(...), timeout=remaining)
    ...
    # backoff is also deducted from the remainder; if it does not fit —
    # we do not sleep, but end the series with an explanatory error
```

Invariants:

- **one agent-visible tool call = one budget** `call_timeout_s`,
  consistent with the per-tool-call timeout (POLICIES §7.1);
- budget exhaustion on transient cancellations yields the same
  explanatory text (§4.3), not `DEADLINE_EXCEEDED`;
- `_MIN_ATTEMPT_S = 0.05` — the threshold below which an attempt is
  obviously pointless.

**What is NOT retried (A2):** `asyncio.TimeoutError` and transport
errors. A timeout means "Serena did not answer within the budget";
retrying it would only double it; a transport error means backend death →
the existing `_mark_lost`. Retry is strictly for `isError=True` + a code
match.

### §4.3. A clear error when attempts are exhausted

If all attempts returned a transient cancellation, `error_message`
is replaced with an explanatory text (instead of the raw
`cancelled (-32800)`):

```
serena <tool>: language server is busy reindexing the project
(LSP -32800/-32801 after 3 attempts); this is temporary — retry the
same call in a few seconds. This is NOT a limit on concurrent tool
calls: calls from multiple agents are queued and executed one by one.
```

Wording requirements (straight from the user's request):

1. The reason is named — indexing is in progress.
2. It says it is temporary and can be retried later.
3. The false conclusion about parallelism is **explicitly refuted**.

The original technical text is kept in the log (`_log.warning`) — it is
needed for diagnostics and harmful to the agent.

**Amendment A6 (pinning the strings).** Tests must check the presence of
all three semantic elements and the absence of the raw code as the only
explanation:

| Check | Why |
|---|---|
| has `reindexing` / `busy` | the reason is named |
| has `retry` + a temporariness hint | it says what to do |
| has `NOT a limit on concurrent tool calls` | the false conclusion from the report is refuted |
| the raw `cancelled (-32800)` is not the only text | otherwise the agent starts guessing again |

The same explanatory text is also produced when the **time budget** is
exhausted (A2), not only the attempt count: for the agent these are the
same state, "the server is busy indexing".

**Amendment A7 (actual attempt count, not the ceiling).** Found at code
review. The series ends for two reasons: attempts ran out OR the budget
ran out. In the second case the number of actually performed calls may be
below the ceiling (e.g. one), and substituting
`_TRANSIENT_MAX_ATTEMPTS` would give the agent **false diagnostics** —
exactly the defect this message exists to eliminate.

We count `attempts_made` (increment only after an actually received
response) and substitute it. The wording is neutral — `attempts made:
N` — so it reads correctly at N = 1 without singular/plural branching.
Pinned in the budget test (time cut-off → the actual number in the text)
and by mutation MUT-A9.

### §4.4. Handshake: the full set of retry methods (user requirement)

User: "The client must send the full set of what it is ready to handle at
the handshake. I see no restrictions here at all."

The restriction here is technical, and it must be named honestly:
`retryOnContentModified` lives in the **client** capabilities
(`general.staleRequestSupport`), which solidlsp forms inside the Serena
process — on its side, before our code. Our `SerenaBackend` talks to
Serena over **MCP**, not LSP; the MCP handshake (`initialize`) does not
forward LSP capabilities.

Options, honestly weighed:

| Option | Assessment |
|---|---|
| A. Patch solidlsp | Rejected: editing an external dependency; the policy demands escalation, not DIY |
| B. Our own LSP client instead of Serena | Rejected: rewriting the architecture for one error code |
| C. Retry on our side (§4.2) | **Accepted**: covers both `-32800` (which solidlsp's generic layer does not have at all) and `-32801`, regardless of what a foreign client declared |

Option C is **strictly stronger** than a full handshake declaration: the
declaration helps only for `-32801` and only for the listed methods,
while our retry covers both codes and all 11 tools. That is, the user
requirement ("handle everything, no restrictions") is met, but at the
layer where we actually have power.

If the user wants specifically a handshake edit in solidlsp — that is an
external-dependency change; I escalate it separately (see §8).

### §4.5. Observability

The metrics of commit 12 gain a counter:

- `LSPServiceServer.note_transient_retry()` — increments
  `_transient_retries_total`;
- the field `transient_retries_total` in `StatusReply` (**new number 17**,
  wire-compat);
- output in `service-status` in the `metrics:` block.

Why: today the hypothesis "it failed because of reindexing" is not
verified live. The counter turns it into a measurable quantity.

`SerenaBackend` gets an optional hook `on_transient_retry`
(analogous to `set_on_lost`), so the server is not pulled into the
backend.

**Amendment A5 (hook semantics).**

- The hook is called **once per retry**, not once per attempt: the first
  (original) attempt is not a retry. Three attempts with a transient
  cancellation → exactly 2 hook calls.
- An exception from the hook **must not break `call_tool`**: we wrap it
  in try/except with logging — the same tolerance as `on_lost`
  (`serena_backend.py:387-390`) and `note_restart`
  (`serena_supervisor.py`). Observability has no right to crash the
  working path.

## §5. Delta

| File | Change |
|---|---|
| `service/serena_backend.py` | `_is_transient_cancel`, the retry loop in `call_tool`, text replacement, the `on_transient_retry` hook |
| `service/grpc_server.py` | `_transient_retries_total`, `note_transient_retry()`, hook installation in `attach_backend`, the field in `ProjectStatus` |
| `service/proto/lsp_service.proto` | `StatusReply` +17 `transient_retries_total` |
| `service/proto/lsp_service_pb2.py` | regeneration |
| `service/service_cli.py` | output of `transient_retries` in `metrics:` |

Unchanged: the lock, `_mark_lost`, `refresh`, bridge, watcher, policy_guard.

## §6. Tests (TDD)

### Unit `test_serena_transient_retry.py` (new)

1. `test_transient_cancel_is_retried_and_succeeds` — the session returns
   `isError=True` + `-32800` on the first attempt, a valid result on the
   second. Result: `is_error=False`, exactly 2 session calls.
2. `test_transient_retry_exhausted_returns_explanatory_error` — all 3
   attempts are transient. Result: `is_error=True`, the text contains
   "reindexing" and "NOT a limit on concurrent tool calls", no raw
   `-32800`.
3. `test_content_modified_is_also_retried` — the same for `-32801`.
4. `test_non_transient_error_is_not_retried` — `isError=True` with foreign
   text → exactly 1 call, the text is not substituted.
5. `test_transient_retry_does_not_release_lock` — during the retry a
   parallel call does not get in (`peak_concurrent == 1`).
6. `test_transient_retry_notifies_hook` — `on_transient_retry` is called
   as many times as there were retries.
7. `test_backoff_is_awaited_not_slept` — `asyncio.sleep`
   is monkeypatched; we check that it is called (the event loop is not
   blocked).
8. `test_code_pattern_is_strict` (A1) — both real forms are recognized
   (`cancelled (-32800)` and `... (caused by cancelled (-32800))`),
   and `-328001` is NOT recognized (protection against a substring inside
   a larger number).
9. `test_series_budget_is_bounded_by_call_timeout` (A2) — the session
   returns a transient cancellation, but each attempt is "slow";
   the total series time does not exceed `call_timeout_s`, and the result
   is an explanatory error, not `DEADLINE_EXCEEDED`.
10. `test_timeout_and_transport_errors_are_not_retried` (A2) —
    `asyncio.TimeoutError` and a transport error give exactly 1 session
    call; the transport error still calls `_mark_lost`.
11. `test_hook_exception_does_not_break_call` (A5) — the hook raises,
    the call still returns a correct result.

### Unit `test_grpc_server_metrics.py` (extension)

12. `test_transient_retries_total_in_status` — `note_transient_retry()`
    ×2 → `StatusReply.transient_retries_total == 2`.

### Integ `test_lsp_transient_retry_e2e.py` (new, no mocks)

13. **Concurrent calls do not fail** — a real service, real Serena,
    6 simultaneous calls from two `BridgeSession`s in one `gather`.
    Invariant: **zero exceptions**, all returned a payload. This is a
    direct check of the user requirement.
14. `service-status` prints `transient_retries=<N>` (N ≥ 0).

Test 13 is exactly the "two tools at once" that the agent believed
impossible.

Honest caveat (accepted from the reviewer as a non-blocker): integ test 13
does not guarantee that `-32800` actually occurs during the run, and
`N ≥ 0` is a weak assertion. This is a deliberate smoke test: the
evidentiary load is carried by the unit tests and mutations; the integ
catches a concurrency regression on the live stack.

## §7. Mutations

- **MUT-A1**: remove the retry (one attempt) → test 1 fails.
- **MUT-A2**: retry any error, not only transient → test 4 fails.
- **MUT-A3**: keep the raw text on exhaustion → test 2 fails.
- **MUT-A4**: release the lock between attempts → test 5 fails.
- **MUT-A5**: do not increment the counter → test 12 fails.
- **MUT-A6** (A1): relax the pattern to a substring `-32800` → test 8
  fails on `-328001`.
- **MUT-A7** (A2): remove the series-wide budget (per-attempt timeout
  without a ceiling) → test 9 fails.
- **MUT-A8** (A2): retry `TimeoutError` → test 10 fails.
- **MUT-A9** (A7): report `_TRANSIENT_MAX_ATTEMPTS` instead of the
  actual `attempts_made` → test 9 (budget) fails.

All nine were run, all nine were killed. Two (MUT-A4, MUT-A7) survived
the first pass and required stronger tests — `peak_concurrent` did not
catch the lock release/reacquire, and the chosen timings did not catch
the missing series-wide budget.

## §8. Open questions for the user

1. **Editing solidlsp/handshake.** §4.4 closes the requirement on our
   layer and covers more cases. An external-dependency edit
   (`retryOnContentModified` with the full method list) would give less
   and requires changing foreign code. I propose **not doing it**; if
   needed — I escalate it as a separate task.
2. **Attempt count.** 3 attempts / ≤0.75 s were chosen conservatively
   (0.6 % of the budget). If observations show long indexing windows,
   the parameters are tunable without changing the architecture.

## §9. Compatibility

- The new proto field is only a new number; old clients do not break.
- Behavior on non-transient errors does not change.
- Successful calls are unaffected (retry only on error).
- No latency regression: backoff only on a real cancellation.
