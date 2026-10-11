# Delivery barrier on the checkpoint write

Status: design, r1. Supersedes the transport-hook mechanism described in
`DESIGN_RESTART_SELF_FALSE_DENIAL.md` §4 while keeping that document's
definition of delivery unchanged.

## The criterion (unchanged)

A tool result is delivered when, and only when, it has been written into
the model's context — persisted to the corresponding checkpoint. A
successful `transport.write`, a hook on a transport-level event, the
absence of an interrupt, teardown state, or an acknowledgement from a
provider or CLI are **not** delivery and may not stand in for it.

## Why the current mechanism must go

HISTORICAL — this section records the mechanism this design replaced. It
is no longer present in this tree: `register_post_delivery_hook`, the
provider gate and the vendored SDK patch are all gone (see "Provider gate
removal"). It is kept because the three defects below are what the
current shape is answerable to.

`restart_self` **used to** fire from `register_post_delivery_hook`, which
the vendored SDK ran immediately after `await transport.write(...)`.
That is the second entry on the forbidden list. It produced three
defects:

1. **Wrong moment.** Between the transport write and the checkpoint
   commit the process can be stopped. The bytes left for the CLI, the
   result never entered the context, and the resumed session replays a
   dangling `tool_use` — the fabricated `user-rejected` denial this work
   exists to remove.
2. **One provider only.** The seam existed solely in the patched Claude
   Code SDK, so `restart_self` had to be gated to `claude-code-sdk` and
   returned `unsupported_transport` on codex, grok, ACP and the standard
   API providers. A universal base tool that works on one transport is a
   regression.
3. **A vendored patch to maintain.** `_deepworker_post_delivery.py` and
   the `query.py` edit existed only to expose that seam.

## The signal

`DeltaSqliteCheckpointer` (`delta_checkpointer.py:190`) is the single
checkpointer for every provider — `agent_process.py:1624` constructs it
with no provider branch. Its `aput` already establishes exactly the
required ordering:

```
delta_checkpointer.py:717   await self.conn.commit()      ← durable
delta_checkpointer.py:736   self._fire_listeners(...)     ← after commit
```

The existing comment states the invariant: *"Listeners fire AFTER commit
so derivative caches observe a durable state."* This is the delivery
point, and it is provider-independent.

### Correlation — resolved

The open question was whether a listener can identify *which* call was
persisted, given `CacheListener = Callable[[str, str, int], None]`
carries no messages.

It can, without widening that signature. `aput` holds the full
`checkpoint` (line 731), whose `channel_values["messages"]` contains the
`ToolMessage` objects with their `tool_call_id`. Verified empirically
against the real class: a checkpoint carrying
`ToolMessage(tool_call_id="toolu_ABC")` fires the listener after commit
with that id readable from the committed entry.

**Decision: do not extend `CacheListener`.** Instead add a dedicated
`add_commit_listener` seam carrying the committed tool_call_ids:

```python
CommitListener = Callable[[frozenset[str]], None]
```

Reasons the existing listener list is the wrong home:

* `add_cache_listener` is documented as a cache-invalidation hook whose
  callbacks "MUST be cheap and MUST NOT await"; spawning a restart driver
  is neither in spirit nor in cost what that contract promises.
* `_fire_listeners` swallows every exception. A silently dropped restart
  is the failure mode this design must not have; the commit listener logs
  loudly instead.
* Nothing registers a cache listener in production today, so the two
  seams cannot be conflated by accident later.

The barrier's fire path becomes:

```
restart_self ──arm_self_restart(argv, call_id)──▶ [pending]
result → ToolNode → aput → commit ──▶ commit listener
        tool_call_id ∈ committed ids ──fire_armed_self_restart(id)──▶ Popen(driver)
```

### Arming becomes bound

With the id read from the checkpoint, `bind_self_restart_call_id` and the
proxy-side binding disappear. The tool no longer needs the MCP handler to
supply an identity it cannot see: `restart_self` knows its own
`tool_call_id` from the LangChain tool-call context, and arms *bound*.

Correlation semantics are unchanged and still mandatory: only an exact
match fires, a foreign id leaves the arm in place, and an unmatched arm
can never fire.

## Preserved properties

These are signed-off invariants; the change must not regress them.

| Property | How it is preserved |
|---|---|
| exact-id correlation | fire requires `armed_id ∈ committed_ids` |
| foreign delivery does not fire | non-matching ids leave the arm pending |
| end-of-turn disarm-only | teardown still calls `disarm_undelivered_self_restart`, never fires |
| capability token fail-closed | untouched — `self_restart_token.py` unchanged |
| notice before arm | ordering in `tools.py` unchanged |

## Provider gate removal

Once the signal is the checkpoint commit, every provider has it. Removed:

* the `provider != "claude-code-sdk"` test and the
  `unsupported_transport` branch in `tools.py`;
* the hook registration in `proxy_mcp_server.py`;
* the vendored SDK patch — `_deepworker_post_delivery.py` and the
  `query.py` edit are reverted in the vendored SDK fork.

Kept, and re-purposed rather than removed:
`self_restart_delivery_available()` / `mark_self_restart_delivery_available()`.
They no longer mean "this provider is claude-code-sdk"; they mean "this
process has a committing checkpointer". Under `--memory-only` /
`InMemorySaver` there is no commit to fire on, so `restart_self` must
fail closed instead of reporting a success it can never confirm. Deleting
them would have turned that case into a silent no-restart.

Both `KNOWN DIVERGENCE` blocks (DESIGN §4 and `proxy_mcp_server.py`) are
removed: they document a gap that no longer exists, and keeping them
would itself be false.

## Inbox

The same criterion governs the inbox delete boundary: a spool record is
dropped only after the commit that carried its message into the context.

The subject differs, so the signal does too. `CommitListener` reports
committed `tool_call_id`s; the inbox needs committed *inbox* messages, so
`DeliveryCommitListener` reports the `delivery_id` that
`_next_user_message` stamps into `additional_kwargs` for every
inter-agent `HumanMessage`. Both fire from the same post-`commit()`
point in `aput`. Overloading one listener for both would have forced the
barrier to reason about ids that are not tool results.

What this replaces: the boundary was `result.get("success")` — a claim
about control flow, not about the model's context. The turn can return
between its last write and the commit; a death in that window unlinked a
message the sender had already been told was accepted, and nothing
reported the loss.

Two bounds keep retention from becoming its own defect:

* **Retention is not ledgered.** A retained record's `delivery_id` is
  never appended to `delivered.log` — ledgering suppresses an id
  permanently, which would convert "retained for retry" into a silent
  drop.
* **Retention is bounded by the TTL.** A commit proof that never arrives
  would otherwise pin a record forever, trading a lost message for an
  immortal one. The existing TTL check in `_drain_inbox_once` retires it
  to `expired/` — moved, never deleted.

`--memory-only` / `InMemorySaver` has no commit and therefore no possible
proof. There the turn result stays the boundary, and only there:
retaining every record would replay the whole spool on each wake. The
fallback is explicit (`_inbox_commit_signal_available`), not implicit in
a missing attribute.

## Tests

* **Provider matrix, inverted.** For *every* name in
  `all_provider_names()`: zero spawns before the commit, exactly one
  after it. The former "everything except claude-code-sdk →
  unsupported_transport" expectation is deleted, not adjusted.
* **Kill between write and commit.** The window the transport hook
  mis-certified. For the barrier: a commit that does not carry the armed
  `tool_call_id` leaves it armed and spawns nothing. For the inbox: a
  turn that returned `success` without a commit keeps its record, so a
  kill in that window loses nothing.
* **Correlation.** A commit carrying only a foreign `tool_call_id` does
  not fire and leaves the arm pending; the matching commit then fires
  exactly once.
* **Disarm-only.** Turn teardown with an armed restart disarms and logs
  ERROR, never spawns.
* **Inbox** (`test_inbox_commit_boundary.py`, 8 tests). A turn returning
  `success` without a commit retains its record; the commit carrying its
  `delivery_id` releases it; a foreign `delivery_id` does not; a retained
  record is never ledgered and still ages out to `expired/`; the
  memory-only fallback drops on the turn result. The 62 pre-existing
  inbox tests continue to pass unchanged.

Full unit suite is required, not the focused subset.

Green tests are not the evidence — a test that cannot fail proves
nothing. Each new invariant was checked by mutation: reverting the
delete boundary to `result.get("success")` fails 5 of the 8 inbox tests;
disabling `delivery_id` correlation fails exactly the foreign-commit
test; disabling `tool_call_id` correlation fails the barrier's
foreign-commit test; a direct `fire_armed_self_restart(...)` call on the
teardown path trips the AST guard.
