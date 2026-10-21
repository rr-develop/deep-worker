# Follow-up: sum_query wipe must reset ACP session state

Status: IMPLEMENTED on dev worktree, UNCOMMITTED, awaiting code review.
Scope: separate from the summarization port and from the turn-ending /
tool-resolve HumanMessage-swallow defect. Both are stated below only as
"same root cause, distinct code path — different slice".

Reported: 2026-08-22.

## Symptom

`the code-graph server` extractor uses `sum_query` for every document: each doc dispatch
is `{"command":"invoke","message":"","sum_query":<prompt-with-document>}`.
On a live agent, the `exposed_invoke_agent` idle send-through routes into
the same `_run_startup_or_send_query` used at startup. That path:

1. Reads current messages via `aget_state`.
2. Offloads them to `<agent>.jsonl` (pure producer).
3. `aupdate_state({"messages": [RemoveMessage(REMOVE_ALL_MESSAGES),
   summary_shape_msg]})` — the LangGraph channel is now length 1.
4. Runs the sum-query turn.

The ACP client state on disk (`acp_session_state.json`) still carries
`sent_count = k > 1` and the pre-wipe `session_id` from the previous
document. First `ainvoke` after the wipe hits the `chat_model.py:571`
guard:

    if state.sent_count > len(state_messages):
        ...  # AcpSummarizationResetRequired

The invoke burns. The guard's side-effect (`record_summarization`) then
zeros `sent_count`/`session_id`, so the NEXT invoke starts a fresh CLI
session and works. Its terminal persist writes `sent_count = k'` back to
disk. On the next document (or the next restart), `k' > 1` again → burn.
Alternating burn / ok, per document, deterministically.

Reproduction on a local test setup (test Neo4j instance) with the extractor:
`err.log` "sent_count=14 > len(state_messages)=1", `token_stats.jsonl`
one session/prompt per burn, the summaries log segment
`summarized_at=2026-08-22T09:40:46Z` = dispatch moment, 7 s before the
burnt invoke.

## Why the guard is right (do not weaken it)

`chat_model.py:571` is the fail-loud net that closed the 2026-08-19
incident (variant 4b). Making it silently continue would re-open the
"dump-into-new-session" path (`state_messages` still contains every
HumanMessage → the first prompt of the new session becomes a full dump).
The right fix is the OTHER side: the code that shrunk `state_messages`
must also reset ACP state, so `state.sent_count` never exceeds the new
`len(state_messages)` in the first place.

## Where to fix

`_run_startup_or_send_query` (agent_process.py, ~2246). It already
locates `SummarizationMiddleware` via `_find_summarization_middleware`;
add the symmetric hook for the model: locate the compiled agent's model
instance, and — if it exposes `areset_after_summary(thread_id=...)` —
call it right after the `aupdate_state` wipe (step 3 above), BEFORE
`_invoke_turn([])` runs.

Capability check by attribute, not by class name (API providers do not
carry ACP state and simply do not implement the method — the call is
skipped). Matches the codebase convention (`_find_summarization_middleware`
capability descent through `bound / func / runnable / __self__`).

## Delivered slice

1. RED tests — `tests/unit_tests/process/test_sum_query_acp_reset.py`
   (new). Five tests, two of which are the behavioural pins that were
   RED before the fix:
   - `test_sum_query_wipe_calls_areset_after_summary_on_the_model` —
     `areset_after_summary(thread_id=<agent>)` is awaited exactly once.
   - `test_sum_query_reset_happens_after_wipe_and_before_invoke` — the
     event log records `["wipe", "reset:<agent>", "invoke"]` in that
     order. Presence alone would not have caught a "reset AFTER invoke"
     regression.
   Three satellites (skip-if-model-lacks-port, query-only-does-not-reset,
   offload-failure-skips-reset) stay GREEN on either side of the fix and
   pin capability + atomicity.

2. Fix — `deep_worker_cli/process/agent_process.py`
   `_run_startup_or_send_query`, in the sum_query branch, right after
   `aupdate_state`:

       model = getattr(self, "_model", None)
       areset = getattr(model, "areset_after_summary", None)
       if asyncio.iscoroutinefunction(areset):
           await areset(thread_id=self.name)

   `self._model` is the accessor set at agent_process.py:1341 — no graph
   descent needed. `iscoroutinefunction` (not `callable`) is required
   because sibling tests wire `self._model` as a plain `MagicMock`; a
   `callable`-based check would auto-produce a non-awaitable stub and
   break them (verified empirically). `AsyncMock` passes
   `iscoroutinefunction`; a real `MagicMock` does not.

3. Mutation checks
   - Entire fix block removed → both behavioural tests FAIL (`awaited
     once` and `["wipe", "reset:ag7", "invoke"]`).
   - `iscoroutinefunction` weakened to `callable` → 6 sibling tests in
     `test_auto_run_startup_query.py` FAIL with `MagicMock can't be used
     in 'await' expression`. Left as a permanent guard against the
     weaker check.

4. Verification — `tests/unit_tests/process/*sum_query_acp_reset*` +
   `test_auto_run_startup_query` + `tests/acp_provider`: 310 passed;
   codespell exit 0; type-check both packages clean under
   `env -u PYTHONPATH deep_worker/scripts/type-check.sh`.

Uncommitted files:
- `deep_worker_cli/process/agent_process.py` (+21 including the block
  comment that documents the fail-loud guard reference and the reason
  behind `iscoroutinefunction` vs `callable`)
- `tests/unit_tests/process/test_sum_query_acp_reset.py` (new)
- `docs/FOLLOWUP_SUM_QUERY_ACP_RESET.md` (this file)

## Note on session_id (follow-up question)

`record_summarization` (client_state.py:192-193) clears BOTH
`sent_count` and `session_id`, and also evicts the id from the
reverse-index (`_evict_session`). So step 10 in `chat_model.py` cannot
`session/load` the stale pre-wipe session id — it must go through
`session/new`. The tests pin this indirectly (the `AsyncMock` records
the `areset_after_summary` call; the effect on the state slot is unit-
tested in `test_record_summarization_clears_queue.py` and
`test_context_gauge_persisted.py` under `tests/acp_provider`).

## Explicitly NOT in this slice

- The `chat_model.py:571` guard stays as-is (fail-loud safety net).
- The turn-ending-tool / tool-resolve mid-turn `HumanMessage` swallow
  (`chat_model.py` step 7/9 interaction, `sent_count` persisted before
  actual send). Same root class ("ACP state desynced from LangGraph
  surgery") but a different code path and a different test. Filed as a
  separate follow-up.

Do not commit the implementation without code review sign-off.
