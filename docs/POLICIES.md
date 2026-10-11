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

## 6. CI test pipeline must stay green (GitHub Actions)

Every push to `master`/`main` runs the `Unit tests` workflow on Python 3.11 and 3.12 (`.github/workflows/unittest.yml`). Both versions must pass. A unit test that depends on a host-specific path (an absolute path on a developer machine, a developer-only mount, a fixed `/tmp/<name>`, etc.) is a red-pipeline bug the moment it lands, because the CI runner uses `$GITHUB_WORKSPACE` and a fresh `ubuntu-latest` image — not the developer's machine. Required portability rules (full version, with the exact reproduction command and the `gh` CLI recipes for inspecting logs): [`docs/CI_CD_TESTS.md`](./CI_CD_TESTS.md). The repository-side rule of thumb is the same as §5: **fix the cause, never bypass the check.** If a test fails only in CI, do not silence it; the section "Procedure when CI is red" in that doc walks through the right diagnostic.

## 7. History rendering must include system prompts

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
`claude_code_sdk_provider/tests/test_job_handle_by_default.py` +
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
> returned by any provider's API (claude-code-sdk, codex, grok, deepseek, or any
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

---

## 11. A failing test is a defect — never an "expected" or "known" failure (HARD POLICY, 2026-08-23)

**The test suite is green or the work is not done.** There is no such thing as
an acceptable persistent failure. A red test is a defect report: either the
product is broken, or the test is broken. Both are fixed, not annotated.

### Absolutely forbidden

Do NOT write, commit, or report any of the following — in documentation, in
commit messages, in review reports, in test docstrings, or in messages to the
owner or to peer agents:

* "known failure" / "known failing test" / "the 2 known failures"
* "pre-existing failure" / "fails on baseline too" / "fails identically with
  the delta stashed"
* "env-coupled" / "tree-location artifact" / "environment-specific failure"
* "expected failure" / "acceptable failure" / "unrelated to this change"
* any table, list, or footnote that enumerates failing tests together with a
  rationale for leaving them red

Establishing that a failure predates your change is legitimate **diagnosis** —
it tells you the defect is not yours to have caused. It is NOT a disposition.
The only permitted next steps are: fix it, or escalate it to the owner as an
open defect with a concrete repair plan. Recording it as a tolerated state is
forbidden, because that is how a red suite becomes the normal background and
real regressions stop being visible.

### Why this is hard policy

A failure labelled "known" stops being read. Two failures on the target tree
were carried across multiple review rounds under the labels "env-coupled" and
"tree-location artifacts", passed from agent to agent as an established fact,
and reported to the owner as if they were a property of the environment. They
were neither known to the owner nor environmental: both were ordinary test
defects, repairable in minutes. The label was doing the work that a fix should
have done.

### What a green run means

"Full suite: N passed, 0 failed" is the only acceptable verification statement.
If a run is not green, the report says what is broken and what is being done
about it — it never presents the red count as a baseline to be matched.

### How reviewers enforce this

* A diff that leaves the suite red does not get sign-off, regardless of whether
  the redness predates it.
* A report quoting a non-zero failure count with an explanation attached is
  rejected as a false green.
* Any document introducing a new "known failures" enumeration is rejected; the
  entries must become fix commits or owner-facing open defects.
* Skipping or xfailing a test to make the suite green is the same violation
  wearing a disguise, unless the skip encodes a genuinely unavailable external
  precondition and says so at the skip site.

## 12. Summarization must never fail silently (HARD POLICY, 2026-08-23)

Summarization is not allowed to end with an error. If it does end with an
error, that error is **fatal and visible** — never a string quietly placed in
the conversation where a summary was supposed to be.

### The rule

* A failure anywhere in the summarization path raises a class listed in
  `deep_worker_cli.exceptions.FATAL_EXCEPTION_TYPES`
  (`SummarizationFailedError`, `HistoryOffloadError`). The runtime
  catch-all transitions the agent to `AgentStatus.ERROR` with the message
  surfaced in the CLI and the web UI.
* The failure text MUST NOT be returned in place of the summary. A helper
  returning `(text, meta)` may not put an error message into the `text` slot:
  the caller cannot distinguish it from a real summary, and neither can the
  agent that later reads it.
* When summarization fails, history is left **intact**. No
  `RemoveMessage(REMOVE_ALL_MESSAGES)`, no eviction, no replacement of
  `state.messages`. Erasing the history behind a suppressed error is what makes
  the failure unrecoverable.
* This applies to the automatic path (`abefore_model`) exactly as it applies to
  the explicitly requested path. The automatic path is the one that runs
  unattended, so a silent failure there is strictly worse.

### A test that pins this behaviour is itself the defect

If a test asserts that an error string lands in the summary — as
`test_automatic_model_failure_soft_errors_after_offload` did — that test is not
evidence of intended behaviour. It is the defect, written down. Delete or
rewrite it; do not preserve it for continuity.

**When a test contradicts a design document about data integrity, the design
document wins.** The test is changed to match the document, never the reverse.

### Verifying a fix here: mutate the ASYNC copy

`summarization.py` keeps near-identical sync and async implementations of the
offload (`_offload_to_backend` / `_aoffload_to_backend`), each with its own
`raise HistoryOffloadError(...)`. The agent runtime only ever reaches the async
one, via `abefore_model`.

A mutation applied to the sync copy therefore changes nothing observable and
the test stays green — which reads as "the test does not detect this", when in
truth the mutation was never executed. Confirm the mutated line is on the async
path before concluding a guard is weak. The same duplication means a fix
applied to one copy silently misses the other.

### Upstream compatibility is not a justification for data loss

Matching a base class's error-swallowing is not a reason to swallow. If the
upstream behaviour loses user data, we diverge from upstream and say so at the
divergence site.

This is not hypothetical. `SummarizationMiddleware._create_summary` is NOT
overridden by default, so it resolves to the langchain base class, which ends:

```python
except Exception as e:
    return f"Error generating summary: {e!s}"
```

The sync `before_model` fed that return value straight into
`_build_new_messages_with_path` and returned it next to
`RemoveMessage(REMOVE_ALL_MESSAGES)` — a second, complete copy of the incident,
latent only because production reaches `ainvoke`/`astream` and never the sync
path. "Currently uninvoked" is not a fix. `_create_summary` is now overridden
to raise, and the override — not the call site — is the right seam: every
present and future caller inherits the guarantee, including one added upstream
inside a base-class method we do not override.

Note the shape of that failure: because the base class catches internally, the
failure arrives as a RETURN VALUE, not an exception. A `try/except` wrapper
alone does not catch it — the override must also detect the marker string and
convert it back into the fault it always was. Mutation-check both branches
separately; disabling only the marker branch still reproduces the incident.

### A guard that scans the wrong file gives false confidence

The AST guard forbidding the marker in `summarization.py` passed for the entire
time the sync path was broken, because the offending literal lives in the
INSTALLED BASE CLASS, not in our module. A source-scanning guard only proves
something about the source it scans. Where the risk is inherited behaviour,
the test must drive the behaviour.

### Why this exists

The soft-return inherited the langchain base class's error-swallowing. When
`e3dc5dd` (2026-07-29) added the user-requested summarization path, it
introduced a `fail_loud` flag and set it `True` for that path only — its own
commit message records the choice: *"Automatic path keeps soft model-error
compatibility; requested path fails loud."* So the path a human watches was
made loud, and the path that runs unattended was deliberately left silent.
The asymmetry was then locked in by a passing test
(`test_automatic_model_failure_soft_errors_after_offload`) asserting the error
string lands in the replacement message.

`DESIGN_SUMMARIZATION_HOT_COLD.md` already forbade exactly this ("No silent
«Error generating summary: …» in state"). The test contradicted the
design document and the test won.

The fix removes the flag rather than flipping its default: a knob whose only
purpose is to choose between "loud" and "loses data" has one correct setting,
and leaving it configurable invites the next caller to pick wrong.

The result: agents woke up after a restart with
`<summary>Error generating summary: …</summary>` in place of their context,
their real history already deleted, the agent status `IDLE`, and nothing
anywhere reporting a fault.

Fourteen corruptions are confirmed on this host — occurrences where the marker
sits inside the summary wrapper itself (`"A condensed summary follows"`), i.e.
the error text WAS the restored context, not a quotation of it. They are spread across several agents, in both main and
task-manager sessions, plus the live session that surfaced the incident. Counting raw
grep hits instead overstates it — most `Error generating summary` strings in
those files are agents *discussing* the bug. Four distinct agents lost context
silently, so this was never a single-agent anomaly.


## 13. Context size comes from the provider — never from an estimate (HARD POLICY, 2026-08-25)

**Rule.** Any decision that depends on how full the context is — the
pre-call guard's bands, the summarization trigger, the status bar — uses the
number the PROVIDER reported. Computing, approximating or inferring that
number is forbidden.

**No unapproved fallback.** A fallback for a specific provider requires the
owner's direct, explicit approval, granted per provider. As of 2026-08-25 no
provider has such approval. "The provider does not report" is a defect to
fix in that provider, not a reason to guess.

**When there is no report yet**, the answer is UNKNOWN and must stay
unknown: the guard passes the turn through and logs it; the summarizer does
not fire. Silence is honest; a fabricated number is not.

**Why this is a hard policy.** It was violated twice, in both directions, by
the same chars-per-token table:

* 2026-08-19 (one agent): estimate ~150k vs real 468k against `-l 400k` —
  summarization never fired and the next turn loaded 466k;
* 2026-08-25 (one agent): estimate 214613 vs real 90k against `-l 250k` —
  the guard fired on fiction, and the message it injected tripped the codex
  provider's shrink detector, dropping a warm session mid-work.

The estimate cannot see what is not in the LangChain message list, and
tokenizers differ per provider. There is no version of it that is safe.

**Enforcement.** `deep_worker_cli/context_limit/observed_context.py` is the
only source; `token_estimator.py` is deleted. Structural tests in
`tests/unit_tests/context_limit/test_provider_context_only.py` fail if the
estimator module reappears, if the guard or the reader imports one, or if
`create_cli_agent` builds a counter that is not the provider-only one.

**Cadence: after EVERY model call, not at the end of the turn** (owner
directive, 2026-08-25). A provider turn is not one model call — grok has
been measured at 98 inferences inside a single turn, growing 18525 → 97368
tokens and beyond. A number that only arrives when the turn is over cannot
guard that turn, cannot move the status bar while the agent works, and
arrives too late to summarize. Every provider must therefore publish a
context reading as each model call completes:

* **acp / grok** — `LiveContextWatcher` tails the CLI's own unified log and
  emits on every `shell.turn.inference_done`
  (`providers/acp/context_watch.py`);
* **codex** — `RolloutContextWatcher` tails the session rollout and emits on
  every `token_count` event (`providers/codex_provider/context_watch.py`);
* **claude-code-sdk** — the receive loop emits from the `AssistantMessage`
  branch, which is the only site that sees each call. The terminal
  `ResultMessage` is NOT sufficient and never was: a tool-calling turn
  breaks out of the loop before it by design, so that path reported nothing
  at all.

A reading must never be de-duplicated against its siblings and must never
be persisted as spend — both mistakes silently restore the frozen gauge.

**Enforcement (cadence).** `tests/unit_tests/context_limit/`
`test_every_provider_reports.py::TestEveryProviderReportsPerModelCall` is
the census: a provider that reports only at the end of a turn fails there.

**One gauge, both consumers** (2026-08-25). The guard and the summarizer
must read the SAME provider number. They did not: the summarizer had a
`floor` on the provider's persisted gauge (`acp_session_state.json` →
`last_context_used`) and the guard had only the message stamp. For a
CLI-session provider that asymmetry is total, because the conversation
lives inside the CLI and our LangGraph state holds just the current
exchange — no stamp is ever anchored to it. Measured on one agent (grok,
`-l 250000`): the stamp carried `first_id=b50f955b`, the state the guard
saw began at `4d10180b`, the guard logged "passing the turn through
UNGUARDED", and the provider's own number was 275704 — 110% of the limit.
`create_cli_agent` now builds one `_persisted_context_floor` and hands it
to both. A persisted provider number is still the provider's number; a
live stamp anchored to the current history always wins over it.

## 14. One owner per rule — no duplicated logic across providers or layers (HARD POLICY, 2026-08-28)

**A rule lives in exactly one place: the component that owns the thing the
rule is about.** If the same decision is written down twice, the system has
two sources of truth, and they will drift — not in theory, but into a zoo of
per-provider bugs where the same feature is subtly broken in three different
ways.

### The rule

Logic that governs a shared resource belongs INSIDE the component that owns
that resource — never in its callers. A caller may ask the owner to act. It
may not re-implement the owner's rules, and it may not keep its own parallel
copy of the owner's state.

Concretely, forbidden:

* the same non-trivial logic copy-pasted into two or more providers
  (`acp`, `claude_code_sdk`, `codex`, …) — if all providers need it, it is not
  provider code, it is shared code they call;
* a second lookup table shadowing an authoritative registry, so a lookup can
  succeed in one table and fail in the other;
* a caller deciding when a shared record is created, retired or destroyed,
  when that lifetime is the registry's own invariant;
* "almost the same" copies — divergent by a line, a guard, or an argument.
  These are worse than identical copies: they read as intentional.

### Why this is hard policy

The 2026-08-28 incident was ONE defect written five times.
`ParkedJobRegistry` stored parked background jobs, but did not own their
lifetime, so five separate call sites each decided for themselves when a job
record dies. All five wrote the same line — `add_done_callback(… drop(job_id))`
— and all five were wrong the same way: they destroyed the record when the
work FINISHED, i.e. exactly when its result had become available and nobody
had collected it yet.

The consequence was silent and severe: `report_infrastructure_feedback` wrote
its report to disk, and the agent was then told "no background job with id".
The work succeeded; the confirmation was unreachable. Three of the five copies
lived in three provider proxy servers — the same code pasted three times, so a
fix in one left the other two broken.

Fixing five copies took three review rounds. Fixing one owner would have taken
one.

### What this requires in practice

* Before adding a call site for a shared resource, check whether the rule you
  are about to write already exists. If it does, call it. If it exists but is
  in the wrong place, MOVE it into the owner — do not add a sixth copy.
* When you find yourself writing the same block into a second provider, stop:
  that block belongs in shared code, and the providers call it.
* A registry owns its records' full lifetime — creation, retirement,
  destruction — and exposes intent-named operations. Callers must not be able
  to leave it in a state its own invariants forbid.
* If duplication is genuinely unavoidable, it needs an explicit owner decision
  recorded at the site, not a silent copy.

### Detection

The tell is a review that says "fixed in N places" or "the same fix applied to
each provider". That sentence is not a report of thoroughness — it is a report
that the code has no owner for the rule, and it is grounds to reject the change
and relocate the logic instead.

## 15. Verification reads the WORKING TREE, not the index (HARD POLICY, 2026-08-30)

`./scripts/type-check.sh`, `pytest`, and every other checker read the files as
they sit on disk. They do not read the git index, and they do not read the
commit you are about to make. A green run therefore certifies the working tree
— nothing else.

### The rule

* An uncommitted change belonging to ANOTHER topic is part of your verification
  result whether you meant it to be or not. A neighbouring topic that is
  mid-edit, or type-broken, blocks a commit that has nothing to do with it.
* Before running verification for a commit, know what else is in the tree.
  `git status --short` is the check. If a second topic is present, either
  finish it, stash it, or state plainly in the report that the run covered both
  — do not present the result as if it certified your topic alone.
* Never report `git diff --cached` or "what I am committing" as the thing that
  was verified when the runner saw the working tree.

### A check can leave behind an artifact that fails the next check

Verification is not read-only. `./scripts/type-check.sh` creates a `.venv` and
a `*.egg-info`, and either can block a later, unrelated commit. Clean up **both**
— removing only the `.venv` is the easy half.

Three properties make this hard to spot, and none is covered by the rule above:

* **`.gitignore` does not protect you.** The hooks look at the disk, not the
  index. `*.egg-info/` is ignored, so `git status` stays clean while the
  artifact sits there — a clean status is not evidence the tree is clean.
* **The radius is every worktree of the repository, not the one you commit
  from.** `pip show <pkg>` resolves against any directory carrying an
  `.egg-info`, wherever it lives. An artifact in the master worktree failed a
  commit made from the dev worktree.
* **The failure names the wrong thing.** The hook reported "FORBIDDEN PIP
  INSTALLATION DETECTED" when nothing had been pip-installed; `pip show` had
  merely resolved to a repository directory. Read what the check actually
  inspected before acting on what it claims.

Incident 2026-08-31: a `deepagents.egg-info`, created by a `type-check.sh` run
one minute earlier in the sibling worktree, rejected an unrelated commit as a
forbidden pip installation.

### Snapshot the bytes LAST, and freeze them

The corollary bites hardest during review. A reviewer signs off on specific
bytes; the author must make sure those are the bytes on disk.

* Take the `sha256sum` as the **last action** before sending it. A hash taken
  before a subsequent edit is a false statement about the tree, not a stale
  detail.
* Once hashes are sent, the files are FROZEN until the verdict arrives. Editing
  a file while a reviewer is running against it means the two of you sign
  different artifacts — and a mutation test can "pass" purely because the
  author's write clobbered the reviewer's.
* If the bytes must change after sending, say so and resend. The reviewer's
  verdict names a hash; a verdict for bytes that no longer exist is void.

### Rolling back a mutation must not touch anyone else's work

Mutation testing edits the working tree, and the edit must be undone from
**your own copy, taken before the mutation** — absolute paths, then re-verified
by sha256.

`git checkout -- <path>`, `git restore`, `git stash`, and `git checkout .` are
unfit for this. They return the file to HEAD and destroy any uncommitted work
in it, including the author's work you are reviewing at that moment. That work
is in neither the index nor git's object store, so **there is nothing to
recover it from**. The file afterwards looks untouched — `git diff` is empty —
so the loss reads as "the edit vanished" rather than "I deleted it".

Incident 2026-08-30: a reviewer rolled back a mutation with `git checkout --`
and erased the author's uncommitted fix. It survived only because the reviewer
still held a backup and could reassemble the bytes back to the signed hash.

The corollary binds the author too: until a change is committed it exists in
exactly one copy — on disk, in a tree the reviewer also writes to.

`cp` deserves its own caution, independent of the above: it resolves the
destination against the current working directory, so the same command run from
a different directory writes somewhere else entirely and silently leaves the
target untouched. Use absolute paths for both operands, and re-verify by hash
rather than assuming the copy landed.

### Diagnose the cause you can prove, not the one you assume

When work disappears from a shared tree, the party that lost it is usually not
the party that removed it. Naming the wrong cause is worse than naming none: a
postmortem that blames a mistyped `cp` hardens a rule against a defect class
that did not occur, while the real one — a reviewer reverting an author's
uncommitted work with `git checkout` — stays unguarded. Establish who wrote to
the file and when before writing the lesson down.

### Why this exists

Each of these produced a real incident in one review round: a neighbouring
topic's type errors blocking an unrelated commit, three successive rounds where
the hash named by the author did not match the disk, a reviewer's mutation
being overwritten mid-run by the author's concurrent write (so the mutation
test passed and proved nothing), and a mutation rolled back with a git command
that destroyed another party's uncommitted work. None were detectable from the
commit message or the diff — only from the tree.

---

## 16. Configuration lives in `~/.deep_worker` — never in `.deepagents` (HARD POLICY, 2026-08-31)

deep-worker descends from deepagents. It has not been compatible with them for a
long time, and it is a separate product. A path naming `.deepagents` as a
**configuration or data directory** is therefore not a harmless leftover: it
points our runtime at a foreign project's namespace.

### The rule

**Nothing in this project may WRITE to `.deepagents`. Not MCP config, not
skills, not agent memory, not session state, not databases — nothing, at any
scope.**

The single runtime root is `~/.deep_worker`, resolved through
`storage_paths.deep_worker_home()`. Never rebuild the literal at a call site.

| Scope | Correct path |
|---|---|
| user | `~/.deep_worker/mcp.json` |
| project | `<project_root>/.deep_worker/mcp.json` |
| per-agent | `~/.deep_worker/agents/<name>/mcp.json` |

Per-agent files belong under `agents/<name>/`, never as a top-level sibling —
the top level is reserved for infrastructure (`logs/`, `env.d/`, `lsp/`, …)
per `INFRASTRUCTURE_TOP_LEVEL_DIRS`.

### Reading is forbidden too — there is no legacy fallback

**Reading from `.deepagents` is banned on the same terms as writing.** There is
no "read-only legacy fallback", no migration shim, and no compatibility path.
The directory is not ours, and code that consults it can silently make a stale
foreign config authoritative — an agent then boots with configuration the
product no longer owns, and nothing about that is visible from the outside.

A resolver must therefore return exactly one path per scope. If configuration
is missing, the correct behaviour is to treat it as absent, not to go looking
for it in the ancestor's tree.

Migrating an existing install is a **one-time, out-of-band** action (move the
files, then delete the directory), never a branch inside the product's own
lookup path.

### Two different things share the name

Do not "fix" these by pattern-matching the string `deepagents`:

* `deep_worker/deepagents/` — the **vendored Python package**. A legitimate
  import path and `sys.path` entry. Leave it alone.
* `~/.deepagents/` and `<project>/.deepagents/` — the **configuration
  directory**. This is what must be eradicated.

Renaming the first breaks every import while fixing nothing.

### Migration must move the config, not just repoint the code

Repointing the resolver without moving existing files is a silent
configuration wipe: every agent restarts with its per-agent MCP servers gone
and reports no error, because an absent layer is indistinguishable from an
empty one. The two halves ship together — move the files, then repoint the
code — and per-agent files must be moved **individually**. They are not copies
of a shared template; at the time this policy was written all 32 of them
differed.

### Detection

```bash
# Config-dir references in runtime code. Expected: only legacy_* helpers.
grep -rn '"\.deepagents"\|/\.deepagents\|\.deepagents/' --include=*.py . \
  | grep -v '/tests/' | grep -v 'deep_worker/deepagents/'
```

A hit outside a `legacy_*` helper is a defect, not a style issue.

### Why this exists

This was raised, worked on, and still regressed: a `session_metadata.json`
written in August 2026 recorded `~/.deepagents/<agent>/mcp.json` as
a live config path. The reason it survived earlier cleanups is that the
resolvers were spread across six modules — `storage_paths.py`,
`external_mcp/config_store.py`, `process/agent_process.py`, `cli/mcp_command.py`,
`cli/lsp_command.py`, and `config.py` — each rebuilding the literal itself. Fixing
one and leaving the others is what produced the regression, so the accessor is
the only permitted spelling.
