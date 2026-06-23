# DESIGN: drop the ephemeral `__pregel_tasks` channel from delta checkpoints

**Status:** Landed (2026-05-30) — `aput` paths (delta, full-snapshot,
`channel_versions`, `versions_seen`). Branch: `dev`. Touches
`deep_worker_cli/delta_checkpointer.py`.

> **2026-06-23 follow-up:** the second ingress point — `aput_writes` —
> was missed in this fix and leaked the same channel into the SQLite
> ``writes`` table. Observed live on a live agent: 2.43 GB of
> `__pregel_tasks` rows in `writes` (96 % of a 2.84 GB `threads.db`).
> See a related internal incident analysis (not included here) and the
> `TestAputWritesDropsTasks` test class.

## 1. Symptom (forensic)

Live observation on an agent (incident 2026-05-30):
- `threads.db` reached **1.6 GB** while the next-largest agent sat at
  658 MB and a fresh agent at 5.7 MB.
- The DB physically corrupted under heavy write load (B-tree page numbers
  pointed past EOF — `PRAGMA integrity_check` failed); UI lost the agent's
  final response.
- After a `.recover`-style copy (15 439 of 15 447 checkpoints survived),
  decomposing the checkpoint BLOBs showed:

| field of delta record | total weight | share |
|---|---:|---:|
| `delta_channel_values.__pregel_tasks` | **782 MB** | **92 %** |
| `delta_channel_values.messages` | 5.5 MB | 0.6 % |
| `channel_versions` / `versions_seen` / everything else | ≈ 60 MB | 7 % |

So **92 % of the database is one channel that should not be there**.

## 2. Why this happens

`DeltaSqliteCheckpointer._compute_delta` (delta_checkpointer.py) special-cases
**only** the `"messages"` channel (append-delta). Every OTHER updated channel
falls into the `else` branch:

```python
else:
    delta_values[channel] = new_val   # full replace
    delta_meta[channel] = "replace"
```

For `__pregel_tasks` that means the full payload is rewritten in every
checkpoint that touches the channel. On the affected agent the payload is a list of
2-3 `Send` objects, each ≈ 500 KB (a `Send` carries the args/state passed to a
node — when fanning out to subagents this is essentially a full message
history). The "delta" is therefore a near-full snapshot of `TASKS` on every
fan-out — exactly the quadratic growth the docstring (`12 × 800KB = ~10MB per
exchange, growing quadratically`) promised to avoid.

## 3. Why dropping `__pregel_tasks` is safe

The channel is a LangGraph internal:

- `langgraph._internal._constants.TASKS = "__pregel_tasks"`.
- Reserved by Pregel: `pregel/main.py` raises `"Channel TASKS is reserved
  and cannot be used in the graph."` if a user state graph defines it.
- Built as `Topic[Send](accumulate=False)` (`pregel/main.py`). Topics with
  `accumulate=False` are **ephemeral**: the value is consumed within a single
  super-step and is NOT carried over.
- Pregel produces the `Send` list, reads it back in the same super-step
  (`channels[TASKS].get()` in `pregel/_algo.py`), routes the work, and the
  channel re-arms empty.
- On resume from a checkpoint, `pregel/_loop.py:758` only TOUCHES TASKS if it
  happens to be in `channel_values`; absence is a normal, supported state.

Implication: persisting TASKS is **not required for correct resume**. The
user's intuition is right: at best the *last* value of TASKS could matter (if
a process died mid-super-step before the Sends were dispatched), but
deep_worker checkpoints are taken at the **end** of each super-step boundary,
when TASKS has already been consumed. In practice the value we keep writing is
already stale by the time it lands on disk.

## 4. The fix

Make `_compute_delta` and the full-snapshot path BOTH strip `__pregel_tasks`
before serialisation. The reconstruction path (`_reconstruct_full`) already
treats a missing channel correctly — `channel_values.get(channel, ...)` and
the dict-merge based replay will simply omit it.

Single source of truth: a module-level frozenset

```python
# Channels never persisted by the delta checkpointer (LangGraph-internal,
# ephemeral, regenerated each super-step). Keeping them in the DB inflated
# the affected agent's threads.db to 1.6 GB (92% of weight) — see
# docs/DESIGN_FIX_DELTA_CHECKPOINTER_DROP_TASKS.md.
_EPHEMERAL_CHANNELS = frozenset({"__pregel_tasks"})
```

Applied in three places:

1. **`_compute_delta`** — skip `channel in _EPHEMERAL_CHANNELS` when iterating
   `updated`, and ensure the channel is absent from `delta_channel_values` and
   `delta_meta`.
2. **Full-snapshot path in `aput`** — pop the channel from the snapshot copy
   before serialisation (so existing v1/full-snapshot serialisers also stop
   writing it).
3. **`channel_versions` / `versions_seen`** — these dicts are written wholesale
   today; we ALSO strip the ephemeral key from them so its presence doesn't
   keep growing the delta. (Safe: when the channel reappears at resume, Pregel
   re-assigns a version; LangGraph treats absence as "never seen".)

What we deliberately do NOT change:
- The on-disk format/version — older checkpoints that contain TASKS still
  reconstruct correctly (we read what's there; we just stop adding more).
- The reader (`aget_tuple` / `_reconstruct_full`). It must remain
  backward-compatible: pre-fix DBs (including the recovered agent's one)
  may have TASKS in some rows; reads still work and the data is harmlessly
  inherited into the reconstructed state only for those legacy rows. New rows
  no longer carry it.

## 5. TDD plan

`tests/unit_tests/test_delta_checkpointer_drop_tasks.py` (new file, real
`DeltaSqliteCheckpointer` against an `aiosqlite` in-memory DB — no mocks of
the checkpointer itself):

1. `test_tasks_excluded_from_delta`: a sequence of `aput`s where one
   checkpoint marks `__pregel_tasks` in `updated_channels`; the serialised
   row contains NO `__pregel_tasks` in `delta_channel_values`.
2. `test_tasks_excluded_from_full_snapshot`: a root checkpoint (the
   full-snapshot path) likewise serialises without `__pregel_tasks`.
3. `test_tasks_excluded_from_channel_versions_and_versions_seen`: TASKS
   doesn't leak in via the always-written `channel_versions` /
   `versions_seen` maps.
4. `test_resume_after_aput_aget_round_trip_ignores_tasks`: a real
   `aput` → `aget_tuple` round-trip reconstructs every USER channel, and
   `__pregel_tasks` is simply absent from the reconstructed checkpoint.
5. `test_messages_delta_still_works`: append-delta on `messages` is
   unaffected (regression guard).
6. `test_pre_fix_row_with_tasks_still_reads`: hand-craft an old-style row
   containing `__pregel_tasks` in `delta_channel_values`, verify `aget_tuple`
   returns it intact (backward compat).

Tests follow the existing style in `test_delta_checkpointer.py`.

## 6. Verification on a real agent

After the fix lands:

1. Start a fresh test agent (`probe_drop_tasks`) on dev code, claude-code
   provider, local shell, working dir the development codebase.
2. Drive a sequence of safe operations that historically caused TASKS bloat
   (fan-out to subagents via the `task` tool — e.g. a small parallel grep).
3. Open the test agent's `threads.db` (under each user's data directory) and assert:
   - `PRAGMA integrity_check` → `ok`.
   - For every row in `checkpoints` containing the channel-values blob, the
     key `__pregel_tasks` is **not present** under `delta_channel_values`
     (parsed via msgpack, like the diagnostic in the incident note).
   - Database growth per super-step drops by ≈ the historical TASKS share
     (compare `length(checkpoint)` distribution against the pre-fix agent).
4. Stop & clean up (`--stop` + remove probe dir, like prior incidents).

## 7. Risk + rollback

- The fix is additive (a constant-driven skip) and has zero effect on any
  USER channel.
- Pre-fix rows remain readable because the reader code path is untouched.
- If a future LangGraph release makes TASKS required across super-steps
  (currently it does not), `_EPHEMERAL_CHANNELS` is the single line to
  revisit. The protective comment at its definition names that constraint.
- Rollback = revert the patch; old rows stay readable, new rows resume
  carrying TASKS again.
