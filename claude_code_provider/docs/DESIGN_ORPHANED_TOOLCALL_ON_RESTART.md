# Design — Orphaned tool calls must not deadlock the agent after restart

Status: design + TDD implementation (2026-06-15)
Branch: `fix/orphaned-toolcall-on-restart-2026-06-15`

## Problem

When a deep-worker agent is killed/restarted **while a tool call is in
flight**, the next invoke after restart deadlocks forever. Observed live
on `mj1_init` (2026-06-15): an invoke parked >7 min on tool
`history_memory`; the agent's event loop stayed alive and served other
RPyC requests, but that one turn never completed.

### Root cause (verified by gdb dump of live asyncio tasks)

The claude-code provider resumes the CLI session on (re)start
(`chat_model.py` — `resolve_restore_session`). On resume the CLI
**replays the persisted assistant transcript**, which re-emits the prior
turn's `ToolUseBlock` in the message stream — carrying the **old
generation's `tool_use_id`** (e.g. `toolu_01J5KP3G…`). But the *fresh*
control-channel `tools/call` the CLI dispatches for execution gets a
**new id** (e.g. `toolu_01MBPn8…`). So:

* Stream side (`chat_model.py:1264`) captures
  `expected_proxy_tool_call_id = first_proxy_block.id` =
  `…01J5KP3G…` and then `await proxy_mcp.wait_for_pending_call_by_id(…)`
  (`chat_model.py:1328`) — **no timeout** (intentional: a timeout there
  previously caused "3 copies of one block" / phantom retry).
* Control side registers the pending call under `…01MBPn8…` and parks at
  `proxy_mcp_server.py:413` `return await future` — also no timeout.

The two ids never match → both sides wait forever. The whole correctness
of the wait rests on the **per-call** equality `ToolUseBlock.id (stream)
== claudecode/toolUseId (control _meta)` (`chat_model.py:1290`,
POLICIES.md §7). **Restart-mid-tool-call breaks that equality.**

> ⚠️ TERMINOLOGY (2026-06-16): here `ToolUseBlock.id` means the
> per-call `toolu_*` (`content[].id` of one `tool_use`), i.e. the
> `tool_use_id` itself — NOT the assistant message-block id (`msg_*`).
> Do NOT read this as "message/block id == tool_use_id": that is FALSE
> (measured 0/46683; one message block holds up to 52 distinct
> `tool_use_id`s). See `FACTS_TOOL_USE_ID_VS_BLOCK_ID.md`. The mj1_init
> 2026-06-15 deadlock was NOT a restart-orphan but a *different-call*
> id (built-in `execute` vs our `mcp__deepworker__execute`) waited on
> as if it were ours — same FACTS doc, "the trap that still bites".

There is **no** restart/resume-aware handling of in-flight tool calls.
The existing restart machinery (`agent_process.py` restart-notice +
`_auto_continue_after_restart`) only *prompts the model to continue*; it
does not reconcile dangling tool calls. The single safety net that used
to exist (the `wait_for` timeout) was deliberately removed and **not
replaced** for the id-mismatch case.

## Principle (user mandate)

> After a restart the agent must NOT wait for a tool that will never
> return. It must understand it was restarted and, if there were pending
> calls, be **informed** about them — but never be made to wait.

This is to be solved **as such** — by reconciling the interrupted call —
**not** with fallbacks or timeouts. A timeout is a band-aid that fires
*after* the agent has already been stuck; the correct fix removes the
dangling call *before* anything can wait on it.

## Source of truth

There are TWO places a tool call's completion is recorded, and the
authoritative one for this bug is the CLI transcript (verified live
2026-06-15 — see below).

1. **The Claude CLI session transcript** (`~/.claude/projects/<slug>/
   <uuid>.jsonl`) — AUTHORITATIVE. Each assistant `tool_use` block has an
   `id`; its result is a user `tool_result` block with matching
   `tool_use_id`. A tool call is **orphaned** iff a `tool_use` id has no
   matching `tool_result`. This is the file the CLI REPLAYS on `--resume`,
   so this is where the dangling block that deadlocks the turn actually
   lives.

2. **The LangGraph conversation history** (`messages` passed to
   `ainvoke`) — an `AIMessage.tool_calls[i].id` with no matching
   `ToolMessage(tool_call_id=id)`. Useful as a secondary signal.

**Live evidence (2026-06-15, mj1_init repro):** an agent killed mid
`execute(sleep 10)` then restarted produced a CLI transcript with **8
`tool_use` blocks and 0 `tool_result`** (a dangling tool_use), while its
LangGraph checkpoint contained **only text turns, no tool_calls at all**
— LangGraph never checkpointed the interrupted tool turn. Therefore
orphan detection MUST read the resumed CLI transcript; LangGraph history
alone misses the orphan. Both sources are unioned for robustness.

Detection is deterministic, runs *before* the resumed turn issues any
wait, and uses no heuristics, no timing, no timeout.

## Design

### 1. Detect orphaned tool calls (pure functions)

In `claude_code_provider/orphaned_tool_calls.py`:

* `find_orphaned_tool_calls_in_transcript(transcript)` — AUTHORITATIVE.
  Scans CLI `SessionMessage` blocks: collect `tool_use` ids and
  `tool_result` `tool_use_id`s; orphans = `tool_use` ids with no
  `tool_result`.
* `find_orphaned_tool_calls(messages)` — secondary. Scans
  `AIMessage.tool_calls` vs `ToolMessage.tool_call_id` in LangGraph
  history.

Both are pure, synchronous, fully unit-testable with constructed inputs.

### 2. Reconcile on the first turn after restart (inform, don't wait)

Inside `ChatModel.ainvoke`, on the **restart turn** — the same branch
that already runs restore: `not already_connected and has_prior_turns
and _has_persisted_session` — before the CLI resume and before the send
loop:

* Compute `orphans = find_orphaned_tool_calls(messages)`.
* For each orphan, **synthesize a `ToolMessage`** with
  `tool_call_id = orphan.id` and content:
  `"⚠ This tool call ('{name}') was interrupted by an agent restart and
  produced no result. It was not executed to completion. Do not wait for
  it; decide whether to retry based on the task."`
* Inject these synthetic ToolMessages into the message stream the send
  loop consumes, so:
  - the conversation is **well-formed** — every `tool_use` has a
    matching `tool_result` (Claude requires this; a dangling `tool_use`
    is itself an API error on the next real turn), and
  - the model is **informed** the call was interrupted, and
  - **nothing ever calls `wait_for_pending_call_by_id` for an orphan**,
    because the orphan is resolved as a (synthetic) ToolMessage up
    front rather than re-entered as a live pending call.

The resolution path is the existing one: a synthetic `ToolMessage`
flows through the normal send loop. We do **not** invent a new wait or a
new timeout. We remove the dangling call from the set of things anyone
could wait on.

### 3. Why the deadlock cannot recur

* The stream-side `wait_for_pending_call_by_id` is only reached for a
  **freshly emitted** proxy `ToolUseBlock` in the *current* turn. After
  reconciliation, the orphaned call from the *previous* turn is already
  a resolved ToolMessage in history — Claude does not re-emit a
  `tool_use` for an id that already has a `tool_result`. So there is no
  replayed block whose id can mismatch a fresh control request.
* No timeout is introduced anywhere. The fix is structural: the
  ill-formed "tool_use without tool_result" state is repaired at the
  one boundary where the history is known and authoritative.

## What this design explicitly does NOT do

* No `asyncio.wait_for` / timeout on `wait_for_pending_call_by_id` or on
  `proxy_mcp_server.py:413` — those were removed for good reasons and
  this design does not bring them back.
* No polling, no "wait a bit and give up".
* No change to the live (non-restart) tool round-trip path — orphan
  reconciliation only runs on the first turn after a restart with
  persisted session state, exactly where restore already runs.

## Test plan

### Unit (mocks allowed)

`tests/test_orphaned_tool_calls.py`:

* `find_orphaned_tool_calls`:
  - AIMessage with a tool_call + matching ToolMessage → no orphan.
  - AIMessage with a tool_call + NO ToolMessage → one orphan.
  - Multiple tool_calls in one AIMessage, partially resolved → only the
    unresolved are orphans.
  - No AIMessages / empty history → no orphans.
  - Several AIMessages across turns → correct per-id matching.

`tests/test_orphan_reconcile_in_ainvoke.py` (mocked CLI client):

* Restart turn with an orphaned tool_call: ainvoke synthesizes a
  ToolMessage for the orphan and NEVER calls
  `wait_for_pending_call_by_id` for that id; ainvoke returns instead of
  hanging.
* Non-restart / fully-resolved history: no synthetic ToolMessage; normal
  path unchanged.

### Integration — reproduces the real deadlock

Two layers:

1. **Provider-level, deterministic (authoritative)** —
   `claude_code_provider/tests/test_orphaned_toolcall_deadlock_integration.py`.
   Drives the EXACT deadlock through a **real `ProxyMCPServer`** and the
   **real `ChatModel.ainvoke`** (no mock of the proxy or the wait logic).
   Only the CLI message stream is supplied — which is exactly what the
   real CLI does on resume: it REPLAYS the dangling `ToolUseBlock`.
   * `test_orphaned_toolcall_deadlock_is_reconciled` — with the fix, the
     resumed turn completes (no hang).
   * `test_without_reconcile_the_same_setup_would_deadlock` — control:
     with reconciliation neutralized, the identical setup PARKS on
     `wait_for_pending_call_by_id` (caught by a `wait_for` tripwire in
     the test). This proves the test exercises the real deadlock and the
     fix is what removes it.

2. **Real-agent end-to-end (full stack)** —
   `deep_worker_cli/tests/integration_ai/test_orphaned_toolcall_restart_e2e.py`.
   Spawn a real claude-code agent (`--use-sqlite -m` so history survives
   restart). Warm-up turn, then ask it to call `execute` running
   `sleep 10 && echo hello`; SIGKILL mid-call; restart; probe. It must
   respond, not hang. The wall-clock bound is an *assertion ceiling for
   the test*, not a production timeout. Because whether the CLI-resume
   path fires is environment-sensitive (session-restore tail-match), this
   test SKIPS with a clear reason if the run does not reach a genuine
   resume+reconcile — it never asserts on a path it did not exercise. The
   provider-level test guarantees deterministic coverage of the deadlock.

## Files

* `claude_code_provider/orphaned_tool_calls.py` — new pure detector.
* `claude_code_provider/chat_model.py` — call the detector + inject
  synthetic ToolMessages on the restart turn.
* `claude_code_provider/tests/test_orphaned_tool_calls.py` — unit.
* `claude_code_provider/tests/test_orphan_reconcile_in_ainvoke.py` —
  unit (mocked CLI).
* `claude_code_provider/tests/test_orphaned_toolcall_restart_integration.py`
  — no-mock integration reproducer.
* `claude_code_provider/docs/DESIGN_ORPHANED_TOOLCALL_ON_RESTART.md` —
  this document.

## Related

* `project_mj1_tooluseid_mismatch_deadlock_2026_06_15` (root-cause memo).
* `DESIGN_HISTORY_RESTORE_ON_RESTART.md` — the resume/restore mechanism
  this hooks into.
* POLICIES.md §7 (No-Arbitrariness; the id invariant) and incident
  2026-06-14 (tool_use_id unification) — this is the resume-time gap
  that unification did not cover.
