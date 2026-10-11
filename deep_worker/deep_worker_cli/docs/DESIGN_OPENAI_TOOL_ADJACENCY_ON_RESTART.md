# Design — OpenAI/DeepSeek tool-call adjacency after restart

**Status:** rev 2 (2026-08-14). Implementation landed locally (uncommitted). Awaiting design and code sign-off. rev 2 answers review: the adjacency contract is independently implemented in both middlewares (not a shared primitive); move-not-copy is pinned by identity tests on both.

**Incident:** one agent, 2026-08-14. After `-pp deepseek` the process is alive
(`status=idle`) but every invoke dies with:

```
openai.BadRequestError: 400 —
An assistant message with 'tool_calls' must be followed by tool messages
responding to each 'tool_call_id'.
```

Owner mandate (verbatim intent): agents MUST continue after restart regardless
of incomplete pre-restart tool calls. That policy already exists. This is a
regression in how closers are *placed*, not a missing feature. Do not wait for
the tool. Do not re-execute it. Close it, and leave history valid for every
provider — including OpenAI-compatible DeepSeek.

Error-state on unrecoverable 400 is **out of scope** (owner: secondary).

## 1. Problem (verified against live checkpoint)

The affected agent's `threads.db`, latest checkpoint, 147 messages. Three
`AIMessage(tool_calls=[restart_self])` sit immediately before a restart NOTICE:

| idx | tool_call_id | next messages |
|---|---|---|
| 20 | `toolu_018o1xiy5kuypbqwCBKvVRJc` | `SystemMessage` NOTICE — **no ToolMessage anywhere** |
| 34 | `toolu_01RWZRqKapPcLa4ZV8tjtqRD` | `SystemMessage` NOTICE — **no ToolMessage anywhere** |
| 69 | `toolu_01Rf5siKsK8Dq9gz1sKPxS45` | NOTICE, Human continuation, **then** idx 72 `ToolMessage` with Patch's "cancelled - another message came in" text |

OpenAI/DeepSeek rule: every `tool_call.id` on an `AIMessage` must be answered
by a **contiguous** run of `ToolMessage`s **immediately after** that
`AIMessage`. `SystemMessage` / `HumanMessage` in between is a 400.

Grok (12:49–13:03 on this same thread) does not enforce that rule, so the
agent answered. DeepSeek on the same checkpoint 400s. The policy layer that
should have made history valid for every provider did not.

Replay of those 147 messages through current `PatchToolCallsMiddleware`:
it inserts closers for idx 20 and 34, **leaves idx 69 untouched**. That one
leftover violation is enough for the 400.

## 2. Root cause

Policy is implemented in two provider-agnostic middlewares (Claude-only
`orphaned_tool_calls.py` is not on the DeepSeek path):

### Bug A — `DanglingToolCallMiddleware` appends

`deep_worker_cli/middleware/dangling_tool_calls.py:144`:

```python
return {"messages": closers}
```

`AgentState.messages` uses `add_messages`. That **appends** the closers to the
end of the list. After `_build_invoke_messages` (`agent_process.py:3863-3890`)
has already inserted `SystemMessage(NOTICE)` + `HumanMessage(continuation)`
after the dangling `AIMessage`, the closer lands **after** those messages.

It also treats *any* later `ToolMessage` with the same id as resolved
(`resolved_ids` is a global set). So idx 69 (closer already exists, just in
the wrong place) is a no-op.

### Bug B — `PatchToolCallsMiddleware` treats any later ToolMessage as resolved

`deepagents/middleware/patch_tool_calls.py:64-68`:

```python
corresponding_tool_msg = next(
    (m for m in messages[i:]
     if isinstance(m, ToolMessage) and m.tool_call_id == tool_call["id"]),
    None,
)
```

A `ToolMessage` anywhere after the `AIMessage` counts. Idx 69 is "resolved".
No `Overwrite`. History stays `AI → System → Human → Tool`. DeepSeek 400.

### Bug C — high-water mark skips a still-poisoned same-length list

`patch_tool_calls.py:53-54` + `:91`. After a scan, `_scanned_len = len(input)`
even when the result was an `Overwrite` that the caller did not apply. The
next `before_agent` on the **same** still-dangling list returns `None`.

`8c2e28d` (delta-cache) added this. Correctness of the restart policy outranks
the cache shortcut.

## 3. Fix

One **contract** of "closed", implemented independently in each middleware
(Patch lives in `deepagents`, Dangling in `deep_worker_cli`; a shared
primitive would couple the packages for a 40-line walk. Drift is pinned
by the same move-not-copy / adjacency tests run against both):

> A `tool_call.id` is closed iff it appears on a `ToolMessage` in the
> contiguous run of `ToolMessage`s immediately following its `AIMessage`.
> A later non-adjacent `ToolMessage` is **not** closed.

### Patch (`before_agent`)

Walk the list. After each `AIMessage` with `tool_calls`:

- Consume the already-adjacent `ToolMessage` run (keep them).
- For each missing id: if a later `ToolMessage` with that id exists, **move**
  it to immediately after the AI (do not leave a duplicate later). Else
  synthesize the existing "cancelled - another message came in" closer.
- Skip consumed later `ToolMessage`s when they are reached.

Return `Overwrite(patched)` iff the order changed; else `None` (delta-cache
append path unchanged for already-adjacent history).

HWM: set `_scanned_len` **only** when returning `None`. If we return
`Overwrite`, leave the mark alone so a retry on the unapplied input still
patches.

### Dangling (`before_model`)

Same adjacency definition. Same `Overwrite` rewrite (not append). Newly
synthesized closers keep the existing WARNING content +
`deep_worker_dangling_tool_call` / `deep_worker_warning` markers. Moved
existing `ToolMessage`s keep their original content.

If Patch already rewrote, Dangling sees adjacent history and returns `None`.
Dangling remains the safety net when Patch is skipped or HWM misfires.

### Not in this change

- Claude CLI transcript orphan reconcile (already works on the Claude path).
- Flipping this 400 to `status=error` (owner: secondary).
- New CLI flags.

## 4. Tests (already RED on current code)

New file `tests/unit_tests/middleware/test_dangling_openai_adjacency.py`:

- `test_patch_tool_calls_makes_openai_adjacent_after_restart_shape` — GREEN
  today (Patch inserts when **no** later closer exists).
- `test_dangling_middleware_makes_openai_adjacent_after_restart_shape` — RED
  (append).
- `test_patch_hwm_does_not_skip_still_dangling_same_length` — RED (Bug C).
- `test_patch_does_not_treat_nonadjacent_toolmessage_as_resolved` — RED (Bug B).
- `test_rg1_2026_08_14_exact_checkpoint_shape_still_poisons_deepseek` — RED
  (exact idx 20/34/69 shape from the live dump).

Existing `test_dangling_tool_calls.py` helpers that assume
`result["messages"]` is a list of *only* closers must read `Overwrite.value`
and filter by the dangling marker. Semantics of those tests (one closer per
id, markers, idempotency) stay.

Existing `TestPatchToolCallsMiddleware` cases where the AI is followed by
Human (no ToolMessage) already expect insert-immediately-after-AI via
Overwrite — they stay GREEN.

Mutation: reverting either "any later ToolMessage counts" or "append closers"
must turn the new adjacency tests red.

## 5. Risks

- **Duplicate ToolMessages.** Mitigated by moving, not copying, an existing
  later closer.
- **Delta-cache.** `Overwrite` only when order actually changes. Clean
  adjacent history still returns `None`.
- **Existing agents.** Next invoke after the fix rewrites the poisoned
  checkpoint in place. No manual history edit. The affected agent should answer
  DeepSeek on the next turn without `--clean`.

## 6. Owner questions

None. Clear-all / error-state are separate.

## 7. Key Decisions

1. **Adjacency, not mere existence**, is the source of truth. Existence-only
   is why DeepSeek 400s on a closer that is already in the list.
2. **Move** a non-adjacent closer next to its AI rather than synthesize a
   second one. Avoids an extra trailing `ToolMessage` that OpenAI may also
   reject.
3. **HWM only after a clean scan.** An unapplied Overwrite must not arm the
   skip. Cache must not defeat the policy.
4. **Both middlewares, same contract, two implementations.** Patch is
   first (`before_agent`); Dangling is the `before_model` backstop. One
   layer failing must not leave DeepSeek poisoned. They are not a shared
   helper: id-validation already differs slightly (`truthy` vs
   `isinstance(str)`). Parity tests on both make drift fail the suite.

## 8. Corrections log

### rev 2 (2026-08-14) — review

1. Added identity tests: a non-adjacent existing closer is moved once
   (`is` the original object, content/kwargs preserved, no trailing
   duplicate). Run against **both** Patch and Dangling.
2. Design no longer claims a single shared rewrite primitive. Same
   contract, two implementations, parity tests catch drift.

## 9. PR Plan

**PR 1 — `fix(middleware): close pre-restart tool_calls adjacent to the AIMessage`**

Files:

- `deepagents/middleware/patch_tool_calls.py`
- `deep_worker_cli/middleware/dangling_tool_calls.py`
- `deep_worker_cli/tests/unit_tests/middleware/test_dangling_openai_adjacency.py` (new)
- `deep_worker_cli/tests/unit_tests/middleware/test_dangling_tool_calls.py` (helper only)
- `deep_worker_cli/docs/DESIGN_OPENAI_TOOL_ADJACENCY_ON_RESTART.md`

No dependency. One commit on `dev`, FF `master`. Exclude `uv.lock`.
