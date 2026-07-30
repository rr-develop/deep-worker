# Project Policies

This document collects the strict, non-negotiable policies that every contributor (human or agent) must follow when changing this codebase. Reviewers must reject diffs that violate them.

## 1. No-Agent-Error-State Policy

> An agent process must never transition to `AgentStatus.ERROR` for any failure that can be reported back to the model as a tool-call or system message. The default response to a failed tool call, a failed middleware, an unhandled provider exception, an empty-response from the model, an exceeded iteration cap, or any other unhandled internal failure inside the agent invocation pipeline is:
>
> * convert it to a textual tool / system message,
> * append it to the conversation,
> * transition the agent to `IDLE` (so the user can simply type "continue"),
> * and persist a structured record to the error log.
>
> The `ERROR` status is reserved for truly fatal, non-recoverable conditions — usage / billing limit, corrupted checkpoint, unrecoverable provider auth failure, agent not initialised. The set of fatal exceptions is enumerated in `deep_worker_cli/exceptions.py` as `FATAL_EXCEPTION_TYPES`; everything else is recoverable by default.
>
> Tools must not raise exceptions to signal expected failure modes. They must return a structured result whose payload contains an `error` field plus a human-readable explanation. If the failure represents an unexpected / internal condition (bug, OS-level error, invariant violation) the tool may *additionally* log it — but it must still return, not raise.
>
> If a tool nevertheless raises (because someone wrote a sloppy tool), the agent runtime is responsible for catching that exception at the tool-dispatch boundary, converting it into a structured `ToolMessage` with `status="error"` visible to the model, logging it at `ERROR` level, and continuing. This catch is mandatory; bypassing it is a policy violation.
>
> Diagnostic `print(...)` calls inside async hot paths (middleware, tool dispatch, chat-model invoke, the invoke catch-all itself) are forbidden, because `stderr` blocking under load (`BlockingIOError`) has caused the very failure mode this policy outlaws. Use the standard `logging` module with non-blocking handlers.
>
> Anyone breaking these rules writes bad code. Reviewers must reject such diffs.

Full design: [`DESIGN_NO_AGENT_ERROR_STATE_POLICY.md`](./DESIGN_NO_AGENT_ERROR_STATE_POLICY.md).

## 2. Event-driven only — no polling

Polling is strictly forbidden. All wait-for-X loops must be event-driven (asyncio events, condition variables, queue.get, signal handlers, etc.). `while True: …; await asyncio.sleep(N)` patterns that exist to wait for a condition are policy violations.

## 3. Prefer non-blocking, no new threads

Always prefer non-blocking approaches. Do not add new threads without explicit user approval. Existing thread workers (e.g. the logging `QueueHandler` worker) are grandfathered; net-new threads require an issue and a sign-off.

## 4. Filesystem permissions

Files and directories created by project code must use safe permissions. `0o777` and `0o666` are forbidden. Defaults: files `0o600`, directories `0o700`, unless a stricter mode is required.

## 5. Pre-commit hooks must pass

`do_commit` rule: bypassing pre-commit checks (`git commit --no-verify`, etc.) is forbidden. If a hook reports a problem, the fix is to address the problem, not to skip the hook.

## 6. History rendering must include system prompts

**Rendering full conversation history WITHOUT system prompts is misleading and is an outright lie. It is ABSOLUTELY UNACCEPTABLE and is forbidden.**

System messages carry the agent's base system prompt AND runtime middleware hints injected mid-conversation (most notably the inter-agent reply hint from `InterAgentReplyMiddleware`, which tells a receiving agent to reply via `send_message_to_agent`). A history view that hides them misrepresents what the agent actually saw.

Any history-rendering surface (web Full Dialog, CLI, exports) MUST render system messages at every verbosity level **at or above `text`** ("All assistant text") and at every more detailed level (`full`). They appear under a dedicated `--- System ---` header, in chronological order within their turn. They are intentionally omitted only from the most compact `final` view, whose contract is strictly "one final assistant text per user turn".

This is enforced in two places that every history view depends on: the agent-side serializer (`deep_worker_cli/process/message_dict_cache.py::serialize_message` — must emit `content` for `SystemMessage`) and the renderer (`deep_worker_web/app/services/dialog_formatter.py` — must render `--- System ---` blocks at `text`/`full`). Producing a `text`/`full` history view that hides system messages is a regression, not an optimisation, and reviewers must reject it.

Full design: `DESIGN_FULL_DIALOG_FINAL_RESPONSE_PAIRS.md` (web app docs) §"System messages are first-class in history rendering".

## 7. No arbitrary code — find the root cause, never plaster fallbacks

> **Arbitrariness** is a decision made WITHOUT genuine understanding —
> taken at random, in the hope that it will work. In code it manifests
> as workarounds and fallbacks written where the right, working code
> should have been written from the start.
>
> Arbitrariness is also produced by **false data**, especially the
> kind widely circulated in society and rarely noticed as false — the
> kind contained in clever books, articles, conference talks. The
> test is simple: **does this datum actually let you solve the
> problem, or does it only create the appearance of a solution
> without removing the underlying cause?** If it does not really
> solve the problem, or proposes a harmful remedy that only leads to
> further degradation, it is arbitrariness. Possibly a very harmful
> one.
>
> **Law: arbitrarinesses breed new arbitrarinesses; they accumulate
> like a snowball.**
>
> When you see a multitude of fixes attempting to solve the same or
> closely related problem, and they apparently never quite solve it
> or keep spawning new bugs, you can be sure you are facing
> arbitrariness. **Do not keep plastering fixes on top of fixes,
> workarounds on top of workarounds, fallbacks on top of fallbacks.
> Find what was done incorrectly at the very beginning — the
> decision that triggered the chain reaction of arbitrarinesses.
> That is the real root cause. That is what must be found and
> corrected.**

### Concrete prohibitions that follow from this policy

* No `time.sleep(N)` / `asyncio.sleep(N)` "to give X time to settle".
  Either synchronise on the real event (`asyncio.Event`, `Future`,
  notification), or document that the wait is policy
  (e.g. per-tool execution timeout) with a citation that justifies
  the value. "Arbitrary settle times" are forbidden.
* No silent `try / except: pass`, no silent `if not isinstance(x, T):
  continue`, no silent retry loops. Either succeed, or fail loudly
  with a message that names what the contract was and why it broke.
  Past incidents that landed in production from silent skips: MCP
  servers lost after a restart because bare functions were filtered
  out, and double execution of tool calls (both analysed in related
  internal incident analyses, not included here).
* No "TTL caches" / "completed-id sets" / "FIFO eviction" introduced
  to mask a duplicate-execution symptom. Find the missing unique
  identifier on the original path and use it. Past incident: the
  tool-call double-execution root cause (the `8d0ae52` revert chain —
  6 attempts on one site in 5 days).
* No timeouts on infrastructure waits ("our `Future` waited too long
  → tell the LLM the tool failed"). If the wait represents real tool
  work, let it run; if it represents a deadlock in our own plumbing,
  raise an infrastructure error and surface it as an internal failure,
  not as a tool error to the model. Phantom timeouts mislead the LLM
  into retries that duplicate side effects.
* No two registries / caches / state stores for the same thing.
  When a second one shows up "for defence in depth", it is almost
  always a symptom of not having identified what the first one was
  trying to track. Past incidents: `_jobs` in `CLIShellBackend` vs
  the proposed `JobRegistry` in `ProxyMCPServer` — they must be one.

### How reviewers enforce this

When reviewing a diff that adds a workaround, a fallback, a retry, a
TTL, a dedup cache, a sleep, or a timeout, the reviewer MUST ask:

1. **What is the contract this is trying to defend?** Quote it or
   write it. If the contract cannot be stated, the change is
   arbitrariness — reject.
2. **What was the first decision that made this defence necessary?**
   Find the original commit or design choice. If the diff does not
   correct that decision, the change is plaster — reject.
3. **What is the loud failure mode this code takes instead of silent
   wrong behaviour?** If there is none, the change buries information
   that the next debugger will need — reject.

The full worked example of this policy in action is a related internal
incident analysis (not included here) of the tool-call double execution:
six attempted fixes in five days on one site, none of them addressed
the root cause (two-worlds identity in `proxy_mcp_server.py`
`call_id = f"{tool_name}_{counter}_{id(args)}"` vs Claude's
`tool_use_id` from MCP `_meta.claudecode/toolUseId`).

### 7.1 No tool may hang the agent forever (HARD POLICY, 2026-06-16)

EVERY tool call MUST be bounded. This is provider-agnostic and admits no
exceptions — it applies to every existing and future tool, on every
provider.

* **Job-handle mechanism is the DEFAULT for every tool.** After a
  configured wait (`long_running_wait_seconds`, default **120 s** /
  "~2 minutes", env-overridable via `DW_LONG_RUNNING_WAIT_SECONDS`), if
  the tool has not finished, the handler parks the live work as a job
  (the real work is NOT cancelled) and returns a non-error
  `in_progress` result with a `job_id` and the `wait_for_job` /
  `kill_job` actions. The agent decides whether to wait, kill, retry, or
  move on.
* **A bare, unbounded wait (`await future` with no timeout) is
  FORBIDDEN for any tool.** It is exactly what hung `edit_file` forever
  (observed in a past incident).
* **Opting out is EXPLICIT and must carry its own timeout.** A tool may
  decline the job-handle mechanism only by explicitly setting
  `long_running=False`, and that opt-out MUST provide its own `timeout`.
  Opting out WITHOUT a timeout is rejected at construction. A default
  (implicit) opt-out is strictly prohibited.
* **A timeout NEVER becomes an agent error.** It is never
  `is_error=true`. The agent is told what happened (still running,
  job_id, actions) and simply continues / retries. (The old
  `is_error=true` "timeout" lie caused retries that duplicated side
  effects — observed in a past incident.)

Claiming this mechanism "works" while no tool actually engages it (e.g.
`long_running` set on nothing, so the gate is always False) is **false
data** under this section and is treated as arbitrariness. The contract
is enforced, **identically on every provider**, by `ProxyMCPServer.__init__`
(rejects timeout-less opt-out) and the handler (bounded wait for all
tools), and is pinned by matching tests on each provider:
`claude_code_provider/tests/test_job_handle_by_default.py` +
`test_job_handle_integration.py` **and**
`codex_provider/tests/test_job_handle_by_default.py` +
`test_job_handle_integration.py`. Both providers' integration tests prove
the production 120 s wait by mocking ONLY the clock (no other mocks).

## 8. External MCP tools are not provider ProxyMCPServer tools

External MCP servers are executable third-party services configured per user,
project, agent, or launch override. Their discovered tools enter LangGraph as
ordinary `MCPToolAdapter` instances and execute through the external MCP client
session. Each `AgentProcess` owns and closes its own client pool.

Provider `ProxyMCPServer` instances serve a different purpose: they bridge
LangGraph-owned tools into provider CLIs and their handlers never execute a
tool directly. External MCP tools MUST NOT be inserted into those proxies or
written into global provider configuration. Both paths remain subject to the
single bounded-tool-wait policy. With `-sp` / `--skip-permissions`, external
MCP tools run without HITL; otherwise they require approval by default unless
the server explicitly declares `auto_approve: true`.

Full design: [`../deep_worker/deep_worker_cli/docs/DESIGN_EXTERNAL_MCP_TOOLS.md`](../deep_worker/deep_worker_cli/docs/DESIGN_EXTERNAL_MCP_TOOLS.md).

## 9. No silent API errors — every provider error is visible in the UI (HARD POLICY, 2026-07-04)

> **A provider / model API error must NEVER be swallowed.** Every error
> returned by any provider's API (claude-code, codex, grok, deepseek, or any
> future provider) MUST be surfaced to the user in the UI. Producing an empty
> assistant message, a benign "system: type continue" note, or a toast that
> auto-dismisses after a few seconds *in place of* the actual API error is a
> policy violation. This is provider-agnostic and admits no exceptions.

Concrete requirements:

* **Fatal API error** (account-terminal: usage/billing cap, revoked auth) →
  the agent MUST transition to `AgentStatus.ERROR` with the real API error text
  in `error_message`, which the web renders in the **persistent error panel**
  (never auto-dismissed, copyable). Where a provider chain exists, the fatal
  error additionally triggers failover to the next provider — but the error is
  still recorded and shown; failover does not hide it.
* **Recoverable API error** (transient/unexpected provider failure the runtime
  can keep the conversation alive through) → the agent MAY stay `IDLE`, but the
  real API error text MUST be surfaced as an **error** event (`type="error"`),
  routed into the **persistent** UI surface — NOT a `type="system"` note and
  NOT a self-dismissing toast. It must also be appended to the conversation
  history so it is inspectable after the fact. A fleeting message that
  disappears after a second is explicitly forbidden.
* **The detector must read the whole error object.** Provider error replies are
  frequently nested (e.g. grok JSON-RPC: the generic `error.message ==
  "Internal error"` with the real `"402 Payment Required …"` buried in
  `error.data.message`). Classification (fatal vs recoverable, usage-limit vs
  generic) MUST examine every level of the error payload, not just the
  outermost field. Matching only the top-level message is how the 402 →
  deepseek failover silently broke (observed in a past incident).

Root cause of the class of bug this outlaws: `grok_provider/acp_client.py`
`raise_for_error_reply` read only `reply["error"]["message"]` (`"Internal
error"`) and never `reply["error"]["data"]["message"]`, so a real HTTP 402
was classified as a generic `GrokAcpClientError` (not `GrokUsageLimitError`),
failover never fired, and the recoverable path disguised the API error as a
`type="system"` "please type continue" note. Analysed in a related internal
incident analysis (not included here).

Anyone breaking these rules writes bad code. Reviewers must reject such diffs.

## 10. Every fix and every task ships with honest integration tests (HARD POLICY, 2026-07-04)

> **Completing any task or fixing any bug MUST be accompanied by honest
> integration tests in every case where such testing is possible.** Unit tests
> alone are insufficient and, when they stand in for integration coverage, are
> "fake testing" — they can pass while the real end-to-end path is broken.

Requirements:

* **No mocks in integration tests** — the test must exercise the real objects
  on the real path (real agent process, real RPyC, real graph, real provider
  wiring). The ONLY mock permitted in an integration test is a **wait/clock
  mock** whose sole purpose is to avoid spending real wall-clock time on a
  timeout or sleep. Mocking away the very code under test, or the transport, or
  the provider boundary, disqualifies the test as an integration test.
* **Prefer test agents.** Integration tests run against real test agents
  (spawned via the normal `deep-worker` lifecycle), not against hand-built
  in-memory stand-ins.
* **Reproduce the real payload shape.** When a bug was caused by a specific
  real-world payload (e.g. a nested provider error reply), the integration test
  MUST drive that exact payload shape through the real path — not a flattened /
  idealised version. A prior grok 402 fix passed only because its
  tests fed a *flat* `error.message` the real API never sends; the real nested
  reply still broke in production. That is the failure mode this policy exists
  to prevent.
* **"Where possible" is narrow.** The only acceptable reason to omit an
  integration test is a genuinely uncontrollable external precondition (e.g.
  you cannot force a third-party account into a billing state on demand). In
  that case you fault-inject the real *reply* at the real boundary (not by
  mocking your own function) and you document in the test why the external
  trigger could not be reproduced directly.

Reporting a fix as "done and verified" on the strength of unit tests that never
executed the real failing path is a false report. Reviewers must reject diffs
whose only tests are mocked unit tests when an honest integration test was
feasible.
