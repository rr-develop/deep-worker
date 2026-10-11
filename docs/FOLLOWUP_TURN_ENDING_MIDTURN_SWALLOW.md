# Follow-up: mid-turn resume must not swallow a tail HumanMessage

Status: IMPLEMENTED on dev worktree, UNCOMMITTED, awaiting code review.
Scope: the second defect of the same *class* as the already-closed `sum_query`
wipe (`ee15be8`) and the same-tick `tool_call`+`end_turn` drop (`182b66b`),
but a **distinct code path** — `chat_model.py` step 9, the mid-turn resume
branch. Those two are referenced below only as "same class, different slice".

Reporter: an internal worker agent (2026-08-22), 3/3 live captures.

## Symptom

A turn ends with a turn-ending tool_call (`send_message_to_agent`,
`submit_relations`, any turn-ender). The next message the user sends to
that agent is **silently and permanently lost**. The agent answers
instantly with the trailing text of the *previous* turn, and the new
question is never seen — not late, not degraded: never.

Live signature on the recipient side:

- reply arrives in the same second the request landed;
- `has_tool_calls=False`, `id=None`, `usage=None`;
- exactly ONE `session/prompt` in `token_stats` for the whole exchange;
- the "answer" is the tail of turn N, not a response to turn N+1.

## Mechanism

Turn N ends with a turn-ending tool_call. LangGraph completes the turn on
its side, but the ACP side leaves `state.pending_session_prompt_future`
alive, because the CLI is still holding `session/prompt` open waiting for
the tool result it just asked for.

Turn N+1 arrives with the tail `[ToolMessage, HumanMessage("new question")]`:

1. **step 7** walks the delta. The `ToolMessage` → `resolve_call` →
   `tool_resolved = True`. The `HumanMessage` → appended to
   `user_text_parts` (`chat_model.py:669-674`).
2. **step 8** (`:688-690`) persists `sent_count = len(state_messages)`
   and flushes to disk — *before* anything is sent. The brand-new
   HumanMessage is now marked "sent" on disk.
3. **step 9** (`:709`) sees `tool_resolved and
   pending_session_prompt_future is not None` → takes the mid-turn
   resume shortcut: `_adopt_pump` + `_consume_turn` on the OLD future,
   and **returns**. `user_text_parts` is never read — it is only
   consumed by the new-turn path further down (step 11).

So the fresh user turn is dropped on the floor while step 8 has already
recorded it as delivered. There is no retry path: `sent_count` never
goes backwards, so the message is unrecoverable.

## 3/3 live captures

Every one is the FIRST incoming message after a turn that ended with a
turn-ending tool_call:

| # | When | Where | Detail |
|---|------|-------|--------|
| 1 | 2026-08-22 10:23:24Z | worker agent | "Thanks for the invoice…" — reached the agent only after TaskManager retransmit |
| 2 | 2026-08-22 11:16:41Z | worker agent | "Thanks for both points…" — same, retransmit only |
| 3 | 2026-08-22 (extractor) | comparison agent | repo_graph extractor, every document after the first |

Signature identical in all three: `has_tool_calls=False`, instant
tail-of-previous echo, single `session/prompt`.

## Fix

`chat_model.py` step 9. The mid-turn resume shortcut is kept — it is
load-bearing (a built-in-only resume MUST take it, otherwise we fire a
second concurrent `session/prompt`, an ACP violation) — but it now has an
explicit exception: when the tail carries a fresh HumanMessage
(`user_text_parts and not text_only_turn`), we do not resume. Instead:

1. `session_cancel` the abandoned turn, so the CLI is not left waiting
   forever on a tool result LangGraph already handled on its side;
2. `_adopt_pump` + `_discard_pending_prompt` + `clear_queued_turn_state`
   — drain and clear, mirroring the `_prompt_for_summary` retry path, so
   the next `session_prompt` cannot collide with a live one;
3. fall through to step 10 case (c) (session already live) → step 11
   builds a proper SECOND `session/prompt` carrying the new user text.

`text_only_turn` is excluded because step 7c rebuilds `user_text_parts`
from the whole history for that path; it has no pending future to resume.

### On `sent_count`

The original plan called for moving the step-8 persist to after the
actual send. That is **not** part of this fix, deliberately.

State the invariant precisely — it is narrower than "every path that
reaches step 8 goes on to send". It does not: the built-in / tool-only
resume, where `user_text_parts` is empty, still returns via
`_consume_turn` without firing a new prompt, and that is correct
(firing one would be the concurrent-`session/prompt` ACP violation).

The invariant that actually matters is:

> no path that collected a **non-empty** `user_text_parts` can return
> without sending it (crash excepted).

That is what the step-9 exception establishes, and it is what makes the
early persist safe: `sent_count` can no longer be advanced over an
unsent *new user turn*. Moving the persist after the await would
re-open the crash-mid-await double-send hole that the `:688` comment
exists to prevent. Left as-is on purpose.

## Regression test

`tests/acp_provider/test_turn_ending_midturn_swallow.py` +
`turn-ending-open` scenario in `fake_acp_cli.py`.

The fake CLI emits a proxy-MCP tool_call on turn 1 and leaves
`session/prompt` **unresolved** — exactly what a real CLI does while
awaiting a tool result. Turn 2 resolves turn-1 and answers whatever
second prompt the client sends. Every prompt payload is appended to
`FAKE_ACP_TURN_ENDING_PROMPTS`, so the assertions read the wire, not
internal state.

Asserts: a second `session/prompt` was fired; its payload contains
`"brand new user question"`; the returned content is the answer to the
NEW question, not turn-1's tail.

### MUT-check (test proven load-bearing)

With the fix reverted and the test unchanged, the failure is the
designed assertion with the wire evidence attached:

```
AssertionError: turn 2 deadlocked: mid-turn resume swallowed the new
HumanMessage and no second session/prompt was fired; prompts recorded:
['[{"type": "text", "text": "please send delivery"}]']
```

Only turn 1's prompt ever reached the CLI — `"brand new user question"`
never did. That is the dropped `user_text_parts` specifically, not an
incidental failure. Restoring the fix turns it green.

Note on the RED manifestation: under the defect the mid-turn resume
awaits a future the CLI is holding open with nothing to unblock it, so
the coroutine deadlocks; the test bounds that with a 5 s timeout and
re-raises as the assertion above rather than hanging.

## Verification

Both trees, the working tree and the target tree:

- `tests/acp_provider`: **295 passed**, both trees.
- codespell on all three changed files + this doc: clean.
- `mypy` and `ty check` on `chat_model.py`: diagnostics **identical to
  the `19cae44` baseline** (mypy 12 errors before and after, all
  pre-existing and outside the changed range; `ty` diff empty).

`tests/unit_tests` differs per tree, and every failure was checked
against clean `19cae44` with this delta fully removed:

At the time of this slice the run was not green on either tree, and the
red tests were recorded here as tolerated "pre-existing / tree-location
artifacts". That disposition was wrong and is retracted — see
`POLICIES.md` §11: a failing test is a defect, never a known or expected
failure. Establishing that a failure predates a delta is diagnosis, not
a disposition.

The failures carried under that label were ordinary test defects and are
tracked as such:

* `test_dev_flag.py::…::test_not_dev_tree_when_running_from_prod_tree` —
  the test hardcoded a `DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE` path
  and then asserted that tree is NOT the dev tree, contradicting itself
  whenever the suite ran from that path. Fixed 2026-08-23.
* `test_policy_guard.py::test_this_repository_passes_the_guard` — scans the
  working directory rather than tracked files, so an untracked local
  `.deep_worker/mcp.json` fails the run. Being addressed together with the
  §8 violation that config actually contains.
* `test_provider_lazy_roots.py::test_no_top_level_claude_agent_sdk_import_in_production`
  — open defect, not a property of the tree.

No failure listed above was introduced by this slice; that fact explains
their origin and does not license leaving them red.

## Not to conflate

- `sum_query` wipe desync (`ee15be8`) — different code path, fixed.
- `_finalize_response` same-tick `tool_call`+`end_turn` drop
  (`182b66b`) — different code path, fixed.
- N1/N2 open items on the finalize path — different code path, still
  open, non-blocking.

## Operational note (obsolete once this ships)

Until this lands, the first message sent to an ACP agent right after a
turn that ended with `send_message_to_agent` is dropped with
probability ~1 (3/3). Workaround was: send twice, or rely on TaskManager
retransmit. This fix removes the need for both.
