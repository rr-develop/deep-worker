# Follow-up: `_finalize_response` drops mapped `tool_calls` (ACP)

Status: IMPLEMENTED on a development worktree, UNCOMMITTED, awaiting code review.
Scope: separate from the hot/cold summarization delta (`07354c3` + `3561111`).
Reviewer instruction (code review of the 3-file delta):
"_finalize_response tool_call drop — SEPARATE. Do not fold."

## Symptom

A normal agent turn (no summarization involved) where the CLI streams a
`sessionUpdate: tool_call` and resolves `session/prompt` in the SAME event
loop iteration loses the tool call. The turn returns an `AIMessage` whose
`content` is the accumulated text and whose `tool_calls` is empty, so the
tool is never executed.

## Localization

`deep_worker/providers/acp/chat_model.py`

1. `_consume_turn`, branch `if response_fut in done:` (~1210-1236).
   It absorbs `get_task` and drains `notif_q`, calling `_apply_update(...)`
   with the LOCAL list `pending_tool_calls`. `_map_tool_call` →
   `_append_tool_call_bounded` correctly lands mapped calls in that list.
   Then it returns `await self._finalize_response(response,
   final_text_parts, state)` — `pending_tool_calls` is never passed.

2. `_finalize_response` (~1472) builds
   `final_message = AIMessage(content="".join(text_acc))`.
   No `tool_calls` argument exists on this path, so anything accumulated in
   step 1 is discarded.

Contrast: the mid-turn path (~1248, `if pending_tool_calls:`) does surface
them, but it is only reached when `get_task` completes while `response_fut`
is still pending.

## Why it is rare in production

A real CLI that emits a `tool_call` normally does NOT resolve
`session/prompt` with `stopReason=end_turn` on the same turn — it leaves the
prompt pending while it waits for tool results. Both futures therefore
rarely complete in one tick. The code nevertheless allows it, and
`asyncio.wait(..., FIRST_COMPLETED)` reports both as done when a fake (or a
CLI that batches) emits them together; the `response_fut` branch is checked
first and wins.

## Not a summarization concern

`_prompt_for_summary` no longer depends on this path: the summarize retry
test leaves the turn-1 future PENDING (matching real CLI behaviour), so the
mid-turn branch surfaces the tool_calls and the cancel+re-prompt sequence
runs. Discarding tool_calls during summarization is intentional — the ACP
session is reset right afterwards by `areset_after_summary`, so unfinished
calls from the old session must not carry over.

## Delivered slice

1. RED test — `tests/acp_provider/test_tool_call_same_tick_as_end_turn.py`.
   Drives the REAL `fake_acp_cli.py` subprocess through a new
   `tool-then-end` scenario: it emits a proxy-MCP `tool_call`
   (`mcp__<server>__<tool>`, server name taken from the `session/new`
   `mcpServers` block, not hardcoded) and writes the `end_turn` prompt reply
   back-to-back without blocking on `tools/call`. Both land on the same
   event loop iteration. Verified RED before the fix:
   `AIMessage(content='', tool_calls=[])`.
2. Fix — `chat_model.py`, `response_fut in done` branch: after the drain, if
   `pending_tool_calls` is non-empty, `_correlate_proxy_tool_calls` then
   return `AIMessage(content=..., tool_calls=...)` stamped via
   `_stamp_context_observation`, instead of falling through to
   `_finalize_response`. Mirrors the ~1248 mid-turn branch, including the
   `state.text_only_turn_active` drop rule. Differs deliberately: the turn
   is already over, so NO `_start_pump` and no pending prompt future is kept
   (`pending_session_prompt_future` is already cleared, `cleared_normally`
   already True).
3. Mutation check — fix excised, test FAILED on the intended assertion
   ("tool_call ... was dropped"); restored verbatim (+27 chat_model.py).
4. Verification — `tests/acp_provider` 294 passed (293 + this one);
   `codespell` exit 0; type-check `deep_worker_cli` and `deepagents` both
   pass (`env -u PYTHONPATH deep_worker/scripts/type-check.sh`).

Uncommitted files: `deep_worker/providers/acp/chat_model.py` (+27),
`tests/acp_provider/fake_acp_cli.py` (+55, new scenario + docstring),
`tests/acp_provider/test_tool_call_same_tick_as_end_turn.py` (new).

Not committed: awaiting code sign-off.

## Committed

`182b66b` (dev + master, pushed). Code sign-off received.

## Still open — raised by the reviewer as non-blocking, NOT fixed here

N1. The new branch returns before `_finalize_response`, so it also skips
    `raise_for_error_reply`, the cancelled/refusal handling and the
    token-stats emit. Harmless for `tool_call` + `end_turn`. If the same
    tick ever carries `tool_call` + `cancelled`, we would run the tool
    instead of raising. Cheap fix later: inspect error/stopReason first,
    then surface tool_calls.

N2. Sibling hole on the pump path, same class of bug, not this branch:
    `_consume_turn`'s `if queued_response is not None:` (chat_model.py:1139)
    still returns `_finalize_response` and would drop pump-accumulated
    `queued_tool_calls` if the pump observed the same same-tick batch.
    Rarer (pump-only). Needs its own test + slice.
