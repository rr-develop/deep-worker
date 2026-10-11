# Design — `restart_self` must not fabricate a user denial

Status: implemented (2026-08-28)
Branch: `dev`

## Problem

Every `restart_self` call was recorded in the Claude CLI transcript as if
the **user had refused it**:

```
tool_result: "The user doesn't want to proceed with this tool use. The tool
use was rejected ... STOP what you are doing and wait for the user to tell
you how to proceed."
toolDenialKind: "user-rejected"
```
followed by `[Request interrupted by user for tool use]`.

No user ever refused. Measured over the whole transcript history: **5 of 5**
`restart_self` calls (2026-08-15 ×2, 2026-08-27 ×2, 2026-08-28) carry
`toolDenialKind: "user-rejected"`, and every one of them was followed by
300–870 further records — i.e. the restart succeeded and work continued.

In the 2026-08-28 case the denial is timestamped `10:30:13.190Z` while the
replacement process announced itself at `10:30:13.020` — the "user rejected
this" record was written **170 ms after the operation had already
succeeded**.

### Why this is not cosmetic

1. The CLI replays the transcript on `resume`, so both fabricated records
   re-enter the next process's context. The model reads a statement about
   the user's intent that the user never expressed.
2. `"STOP what you are doing"` is an instruction. Acting on it after a
   *successful* operation is a real behavioural error — in the incident that
   surfaced this, the agent went on to apologise for a "violation" that had
   not occurred and repeated that claim to a reviewing agent.
3. Any analysis keyed on `toolDenialKind` (denial statistics, permission
   tuning) is silently poisoned.

## Root cause

`restart_self` ended with `os._exit(0)` (`tools.py`, step 4). The MCP server
that carries tool results is **in-process** — `create_sdk_mcp_server`
(`proxy_mcp_server.py`) runs inside the very process being killed. The
result of a tool call reaches the CLI only via `resolve_call`
(`chat_model.py`), which LangGraph invokes *after* the tool handler returns.

`os._exit` inside the handler therefore guarantees the result can never be
delivered: transport dies mid-call. The CLI sees a `tool_use` with no
`tool_result`, and its denial classifier has no category for "the server
died" — the dangling call falls into `user-rejected`, the default branch.

Note the CLI does distinguish `user-rejected` / `permission-rule` /
`automode-*`, and the agent runs with `permission_mode="bypassPermissions"`
(`client_manager.py`) precisely so no prompts are shown. There were no hooks
and no permission rules in any settings file. The classification was purely
an artefact of the transport dying.

### Why the hard-exit was there

`deep-worker --restart` calls `stop_agent`, which refuses to stop an agent
whose status is `running`:

```python
if status == "running":
    raise RuntimeError("Cannot stop agent ... it is actively running")
```

An agent calling `restart_self` is mid-turn **by construction**, so it is
always `running`. Without the hard-exit the driver would refuse and nothing
would restart. The hard-exit was a way around the driver, not a deliberate
transport decision.

It also served a second purpose (incident 2026-06-12): stop the old process
before it finalises the turn, so the replacement's auto-continuation is the
*same* logical turn rather than a fresh idle one.

## Fix

**Remove the reason for bypassing the driver, instead of bypassing it.**

1. `stop_agent` accepts `self_requested: bool = False`. It waives **only**
   the "actively running" guard. The guard exists to stop a *third party*
   from interrupting a turn it knows nothing about; an agent is never a
   third party to its own turn.
2. `deep-worker --restart` accepts a hidden `--self-requested`
   (`argparse.SUPPRESS`) and forwards it to `stop_agent`.
3. `restart_self` **arms** `deep-worker --restart <name> --self-requested
   --self-requested-token <tok>` and **returns normally**. The result
   travels the ordinary path (ToolNode → `resolve_call` → CLI), so the
   transcript records what actually happened. `tools.py` no longer imports
   `os` at all.

§4 and §5 below close two holes found in review; without them the fix is
incomplete.

## §4 — Delivery barrier

Returning from the tool handler does **not** establish a happens-before
edge with delivery of the result. The driver is a separate process:
spawned inside the handler, it is free to pass `stop_agent` and land
SIGTERM while the result has not yet been delivered. Losing that race
reproduces the original defect exactly — a dangling `tool_use`,
`toolDenialKind: "user-rejected"`, a fabricated denial — just through a
narrower window. "It usually wins" is not a fix.

**What delivery means.** A tool result is delivered when, and only when, it
has been written into the model's context — that is, persisted to the
corresponding checkpoint. Nothing else is delivery. In particular, none of
the following establish it, and none may be used as a proxy for it:

* a successful write to a transport (`transport.write` returning);
* a hook or callback attached to any transport-level event;
* the absence of an interrupt, a stop, a restart, or an exception;
* teardown or end-of-turn state;
* any acknowledgement produced by a provider, CLI, or MCP transport.

The reason is not stylistic. A transport can accept bytes that never reach
the model's context, and a result can reach the context by a path that
never touched the transport being observed. A barrier keyed to the
transport therefore certifies something other than what it claims to
certify, and reintroduces precisely the defect it was built to close.

So the order is inverted. `restart_self` **arms** the restart
(`process/self_restart_barrier.py`); the driver is spawned only once the
result has been recorded in the model's context:

```
restart_self ──arm_self_restart(argv, call_id)──▶ [pending]
result → ToolNode → aput → await conn.commit()   ← delivery
        commit listener ──fire_armed_self_restart(ids)──▶ Popen(driver)
```

The firing point is the checkpointer's commit listener
(`DeltaSqliteCheckpointer.add_commit_listener`), which runs strictly after
`await self.conn.commit()` and carries the `tool_call_id` of every
`ToolMessage` in the committed checkpoint. That checkpointer is the single
persistence path for every provider, so the barrier needs no transport
hook and no per-provider gate. See
`DESIGN_DELIVERY_BARRIER_ON_CHECKPOINT.md` for the mechanism.

### §4.1 Correlation — which record released the barrier

Tool calls can be committed concurrently, so a barrier that fired on the
first commit of any kind would spawn the driver while `restart_self`'s own
result was still unrecorded — the same dangling `tool_use`, reached by a
different route. The barrier is therefore identity-bound: it stores the
`tool_call_id` of the `restart_self` call — injected into the tool by
LangChain, so it is known at arming time — and fires only when that exact
id appears among the committed ids. A commit that does not carry it leaves
the restart armed, which is the safe direction.

**End of turn — disarm, never spawn.** If the turn ends with the restart
still armed, the result was never recorded. `disarm_undelivered_self_restart`
drops it and logs an ERROR; it does **not** spawn.

An earlier revision fired the driver here, reasoning that "the turn is
over, so nothing is in flight". That does not follow: the turn can end
*precisely because* the result never got recorded, and the transcript may
already hold a dangling `tool_use`. Stopping the process at that point is
what makes the CLI fabricate the denial. An agent that visibly failed to
restart is recoverable; a fabricated denial replayed into the next process
is not. The model is free to call `restart_self` again.

**Error reporting.** Deferring the spawn also defers spawn *errors* past
the point where the tool can report them. `restart_self` therefore
pre-flights the driver (`_is_executable`) and returns a structured error
instead of a cheerful "success" followed by silence. A failure at fire time
is logged at ERROR and leaves nothing armed.

**Every provider is covered.** There is no provider gate: the signal is a
checkpoint commit, and every provider persists through the same
checkpointer. `restart_self` works on codex, grok, ACP, the standard API
providers and the Claude Code SDK alike, and the vendored SDK patch that
the old transport hook required is gone.

What the tool still checks is that a durable commit exists to fire on. A
process running on an `InMemorySaver` (`--memory-only`, tests) has none, so
a restart could never be confirmed there; the tool returns
`success: false, error: "no_delivery_signal"` **before** writing the
notice, minting the token, or arming the restart. Failing before any side
effect is what keeps that case from reporting success and then silently
disarming.

## §5 — `--self-requested` is a capability, not a claim

The waiver bypasses the active-turn guard, so it must not be reachable by
simply *asserting* it: `deep-worker --restart <busy> --self-requested` from
any shell must not work. Hiding the flag from `--help` is obscurity, not
authorization.

The sanction therefore travels as a one-time capability
(`process/self_restart_token.py`):

1. `restart_self`, running **inside** the target process, mints a
   `secrets.token_hex(32)` and writes it to `.self_restart_token.json` in
   that agent's own data dir, mode `0600`. Only something already executing
   as the agent can create it.
2. The token is passed to the driver as `--self-requested-token`.
3. `stop_agent` consumes it: the file is deleted **first**, then compared
   with `secrets.compare_digest`. Valid exactly once.

   Successful deletion is a **precondition of authorization**, not
   best-effort cleanup: if `unlink` fails the waiver is refused even when
   the value matches. Authorizing a token that survives on disk would leave
   a replayable capability and make "valid exactly once" a false claim.
   Missing file, unreadable file, failed unlink, or mismatch — all deny.

Without a matching unconsumed token the waiver does not apply and a busy
agent is refused exactly as if `self_requested` were False.

**Boundary.** This does not defend against an attacker who can already
write to the agent's private tree as the agent's uid — such an attacker has
many other routes to restart the agent. It defends against the reachable
case: an unprivileged external invocation asserting the flag.

### The 2026-06-12 invariant is preserved

Continuation does **not** depend on severing the turn. `AgentProcess.start`
consumes `.restart_notice.json` and schedules
`_auto_continue_after_restart` on the notice's presence alone
(`agent_process.py` §7.1) — it never inspects whether the previous turn
committed. The notice is written *before* the driver is spawned, and a
failed notice write aborts the restart.

The old process is terminated by the driver's SIGTERM (escalating to
SIGKILL under `force=True`), as for any other stop.

## Scope of the waiver

`self_requested=True` is set on exactly one path: the `restart_self` tool,
and it is honoured only when backed by a valid unconsumed token (§5). An
external `deep-worker --stop` / `--restart` of a busy agent is refused
exactly as before — pinned by
`test_external_stop_of_running_agent_still_refused` and
`test_external_self_requested_without_token_is_refused`.

## Tests

* `tests/unit_tests/process/test_self_requested_stop.py` — waiver is narrow;
  external stop still refused; idle path untouched; CLI→driver→`stop_agent`
  plumbing; notice-before-arm ordering; no `_exit(` anywhere in `tools.py`;
  the tool arms rather than spawns. Plus:
  * `TestSelfRequestedRequiresCapability` — bare flag refused, forged token
    refused, genuine token accepted, token consumed once, an **unconsumable
    token is refused twice** (fail closed), file is `0600`, driver forwards
    the token.
  * `TestDeliveryBarrier` — arming does not spawn, firing spawns exactly
    once, firing unarmed is a no-op, spawn failure disarms. Correlation:
    **an unrelated tool's result must not fire the restart** (and leaves it
    armed), an unbound arm cannot fire, first binding wins. End of turn:
    **disarm never spawns**, and a late delivery cannot resurrect a
    disarmed restart.
* `tests/unit_tests/tools/test_restart_self_tool.py` — returns a success
  result, no spawn before delivery, `--self-requested` + matching token,
  token file private, result contains no denial language.
* `tests/integration_tools_schema/test_restart_self_terminates_current_process.py`
  and `test_restart_self_sync_mechanism.py` — rewritten for the new
  contract; failure paths still return structured errors and spawn nothing;
  nothing is left armed after a failure.

### Mutation coverage

Each mutation is required to turn the suite RED:

* waiver/plumbing (6): waiver removed / waiver leaks to external stops /
  driver drops the sanction / flag not passed / hard-exit reintroduced /
  notice no longer precedes the arm.
* §4 barrier (3): inline spawn / eager arm / fire-never-spawns.
* §4.1 correlation (4): correlation dropped so any delivery fires / unbound
  arm may fire / proxy stops binding / binding not scoped to
  `restart_self`.
* §4.1 end-of-turn (2): disarm spawns anyway / `agent_process` fires
  instead of disarming.
* §5 capability (5): token check removed / value not compared / unlink
  failure no longer fails closed / driver drops the token / file not made
  private.

A reintroduced `os._exit` kills the pytest process itself, printing no
summary and exiting 0 — which naively reads as a pass. The harness
therefore treats "no summary line" as a detected regression.

## Already-poisoned transcripts

This fix stops **new** false records. The four records already present in
the affected Claude Code session transcript (under `~/.claude/projects/`) remain and will
continue to be replayed on resume of that session. Purging them is a
separate, deliberate operation on CLI-owned state and is **not** performed
here.
