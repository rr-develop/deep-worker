# DESIGN: durable inter-agent message delivery

> Fixes the defect analysed in
> a related internal incident analysis (not included here).
> Status: **design, revision 5** — addresses review round 4
> (1 BLOCKER: user interrupt would kill the drain consumer; plus stale-prose
> cleanup). Changes from r4 are marked **[r5]**; earlier changes remain marked
> **[r2]** / **[r3]** / **[r4]**.

## 1. Problem restated

`exposed_enqueue_message` returns `{"success": true}` immediately after
`asyncio.create_task(_bg_turn())` (`rpyc_service/agent_service.py:643-648`).
At that moment the message exists only as a captured local inside a coroutine
that has not started. `_bg_turn` → `_handle_invoke` parks on `_invoke_lock`
(`process/agent_process.py:3204`); the `HumanMessage` is not built until
`_next_user_message` (`:3299`) and not persisted until `astream` runs.

Two defects:

* **D1 — ack precedes durability.** The sender is told the message arrived
  while it can still be lost, so loss is silent.
* **D2 — no durable queue.** In-flight messages live only in process memory.
  `restart_self` hard-exits via `os._exit(0)` (`tools.py:617`), destroying every
  parked `_bg_turn` with no flush and no trace.

Measured effect: 2 of 6 messages lost (33 %), survivors delayed 39–44 min.

## 2. Goals

Numbered to match "What a fix must satisfy" in the incident document.

1. **Success means durable.** `success: true` only after the message is on
   disk in a form that survives abrupt process death (`SIGKILL` /
   `os._exit`). **[r2]** Durability scope is stated precisely in §3.2.1.
2. **Loss is loud.** Failure to persist ⇒ sender sees `success: false`.
   The tool already propagates this (`tools.py:890`), so no tool change.
3. **Delivery wakes the recipient.** Arrival schedules processing; it must not
   wait for the recipient to start a turn for unrelated reasons.
4. **Survives restart.** **[r2]** Narrowed per review MAJOR 5. Precisely: *a
   message that was durably acked before the recipient process died is
   processed after the recipient restarts.* A message sent while the recipient
   is **down** cannot be delivered at all — the sender's RPyC connect fails and
   it gets `connection_failed` (`tools.py:864-869`), which is loud and already
   correct. Closing that gap needs a sender-side outbox with retry and is an
   explicit non-goal here.
5. **Honest regression test.** Real A→B round-trip over the real transport,
   including recipient busy and recipient killed. No mocks on the delivery path
   (POLICIES §9/§10).

### Non-goals

* Not fixing the separate reply-processing incident (a *processing* failure
  with the message already in storage). Different stage, separate fix. This design
  must not make that one harder to fix.
* No change to the `send_message_to_agent` tool contract or to the
  `<sender> -> <receiver> :>` wire format.
* No new threads, no polling (project policy).

## 3. Approach

Persist before acknowledging, then process from the durable record.

The precedent already in the codebase is `.restart_notice.json`
(`_load_and_consume_restart_notice`, `agent_process.py:1623`): a file written
by one process and consumed by the next at startup. This design generalises
that pattern from one slot to an ordered queue.

### 3.1 Storage

A per-agent spool directory:

```
<agent data_dir>/inbox/          mode 0o700
  <seq>-<uuid4hex>.json          mode 0o600
```

* `data_dir` is the existing per-agent directory used by the restart notice,
  so no new path convention is introduced.
* Modes follow the established private-storage convention
  (`storage_paths.py:42`, `lsp/service/registry.py:43`) — never 0777/0666.
* Filename carries a zero-padded `seq` so lexicographic order is arrival
  order; the uuid suffix makes the name collision-free.

#### 3.1.1 `seq` allocation **[r2 — review MAJOR 1]**

r1 left `seq` undefined. Specified now:

* `seq` is a 12-digit zero-padded integer, allocated **only** by the
  recipient's own `AgentProcess`, never by the sender.
* Allocation and publish are serialised by a dedicated
  `asyncio.Lock` (`_inbox_seq_lock`), **separate from `_invoke_lock`** — it is
  held only across allocate+publish (microseconds) and must never be held
  across a turn.
* All RPyC calls land on one event loop in one process, so this lock is
  sufficient; there is no cross-process writer to this spool.
* On startup, `next_seq = max(seq of every file present) + 1`, scanning
  pending **and** quarantined entries so a quarantined record can never have
  its name reused. Empty spool ⇒ `next_seq = 0`.
* Wraparound/overflow is not a concern at 12 digits.

Test (per review): two concurrent enqueues with forced interleaving preserve
acknowledged order and never overwrite each other.

#### 3.1.2 Record body **[r2 — review MAJOR 2]**

r1 included a `sender` field. **Removed.** The review is correct: the wire
carries only `(message, reply_back)` (`agent_client.py:270`,
`agent_service.py:599`), and the sender name exists solely inside the already
prefixed display text built at `tools.py:887`. Recovering it would mean
parsing `<sender> -> <recipient> :>` back out of human-facing text — a bad
protocol, and it would break the moment the format changes.

Since this design promises no wire/signature change, v1 records carry no
`sender`:

```json
{
  "v": 1,
  "delivery_id": "<uuid4hex>",
  "ts": "<ISO-8601 UTC>",
  "message": "<full delivered content, already prefixed>",
  "reply_back": true
}
```

`v` is a schema version so a later change can migrate rather than misparse.
`delivery_id` is the stable delivery identity required by the review for
duplicate diagnosis (§3.3.1).

Adding a first-class `sender` is a genuine contract delta (optional arg on
`exposed_enqueue_message`, backward-compatible default). It is deliberately
**deferred** — it is not needed to fix loss, and bundling a contract change
into this fix would widen the review surface. Recorded in §7 as follow-up.

### 3.2 Durable write

Standard atomic-publish sequence, so a torn file is never observable:

1. write body to `inbox/.tmp-<uuid>` (`0o600`)
2. `f.flush()`, `os.fsync(f.fileno())`
3. `os.replace()` to the final `<seq>-<uuid>.json`
4. `os.fsync()` the `inbox` directory fd, so the rename itself is durable

Only after step 4 returns does `exposed_enqueue_message` return
`success: true`. Any exception ⇒ `success: false` with the error, satisfying
goals 1 and 2.

#### 3.2.1 Durability scope and directory creation **[r2 — review MAJOR 4]**

The review is right that r1 was inconsistent: the fsync steps claim OS-crash
durability, but a lazily-created `inbox/` would leave its own directory entry
unsynced, so after a power loss the whole queue could vanish despite every
per-record fsync.

Resolved by creating the spool **eagerly at startup**, not on first enqueue:

* `inbox/` is created during agent startup (§3.4) with `mkdir(mode=0o700)`,
  then `data_dir` is fsynced once so the directory entry is durable.
* By the time any enqueue runs, `inbox/` already exists and is durable, so the
  per-record sequence in §3.2 is sufficient on its own.
* This also removes the create-race between concurrent first enqueues.

Stated durability contract, deliberately not overclaimed:

* **Guaranteed:** survives abrupt process death — `SIGKILL`, `os._exit(0)`
  from `restart_self`, uncaught crash. This is the incident's actual failure
  mode and what the tests will prove.
* **Best-effort:** survives OS crash / power loss. The fsync chain is designed
  for it, but it depends on the filesystem and storage stack, and the tests do
  not prove it. Not claimed as a guarantee.

`.tmp-*` leftovers from an interrupted write are removed during the startup
scan (they are by construction not published, so discarding them cannot lose
an acked message). Quarantine rename (§3.3.5) uses the same
`os.replace` + directory-fsync discipline.

### 3.3 Consumption

`_bg_turn` no longer carries the payload. It becomes a *drain* of the spool.

#### 3.3.0 Lock discipline **[r2 — review BLOCKER 1]**

r1 said "drain takes `_invoke_lock`, then invokes the turn" without naming the
call. That would have deadlocked: `_handle_invoke` (`agent_process.py:3198-3211`)
acquires the same **non-reentrant** `asyncio.Lock` itself. This exact trap is
already documented in the codebase at `agent_process.py:2033-2037`, where
`_run_startup_or_send_query` calls the `_owned` variant for this very reason.

**Decision [revised in r5] — the drain does NOT take the lock; each record runs
in its own child task calling the public `_handle_invoke`:**

```
child = asyncio.create_task(self._handle_invoke(cmd, ...))   # takes _invoke_lock itself
await child
```

r2–r4 had the drain acquire `_invoke_lock` itself and call
`_handle_invoke_owned` inline. That is replaced — see §3.3.0b for why it was
unsafe. With a child task calling the **public** `_handle_invoke`, the lock is
acquired by the child exactly once, so there is no recursion and no deadlock:
the drain itself never holds the lock.

Pinned by a unit test with a timeout, so a future edit that reintroduces an
outer `async with _invoke_lock` around the child hangs the test instead of
hanging production.

#### 3.3.0b Interrupt safety: the turn must not be the consumer **[r5 — review BLOCKER]**

`_handle_invoke_owned` records the running turn for interrupt support
(`agent_process.py:3275-3276`):

```python
self.is_running = True
self.current_worker_task = asyncio.current_task()
```

`exposed_interrupt_agent` / `cancel_current_worker_task` (`:2446-2465`) then
cancel exactly that task. In r2–r4 the drain awaited the owned path **inline**,
so `asyncio.current_task()` was the *drain task itself*. A user interrupting a
routine inter-agent turn would therefore cancel the **only consumer**. Since
the design deliberately does not swallow `CancelledError`, the consumer would
end permanently: every later arrival would publish durably and set the event,
with no listener until the process restarts.

That is a new silent processing outage — the same failure class this document
exists to remove. The review is right, and this is the one defect in r4 that
would have shipped.

**Fix: one short-lived child task per record.**

```
child = asyncio.create_task(self._handle_invoke(cmd, ...))
await child
```

`current_worker_task` then points at the **child** (the turn), never at the
consumer, so an interrupt cancels one delivery and leaves the drain alive. This
also composes correctly with the existing cleanup at `agent_process.py:4398`,
which clears `is_running`/`current_worker_task` only when
`current_worker_task is asyncio.current_task()` — true for the child, so the
identity check keeps working unchanged.

##### Two kinds of `CancelledError`, disambiguated

The drain must distinguish a cancelled *turn* from a cancelled *consumer*:

1. **Child cancelled by a user interrupt.** The `await child` raises
   `CancelledError`, but the drain task itself was never cancelled. Contract:
   catch it, **retain the record** (delete boundary §3.3.1 is not reached),
   log, and continue to the next snapshot entry; the consumer stays alive and
   eventually parks on `event.wait()`.
2. **Drain cancelled by shutdown** (§3.3.4). Contract: cancel the active child,
   `await` it under `suppress(CancelledError)` so no orphan is left behind,
   then **re-raise** so `shutdown()` proceeds to close the checkpointer.

Disambiguation uses `asyncio.current_task().cancelling() > 0` (Python 3.12 —
confirmed available on this interpreter) to test whether *this* task was the
cancellation target, rather than inferring it from the child. A drain-level
`_shutting_down` flag set by `shutdown()` before `cancel()` is the explicit
secondary signal, so the contract never depends on timing.

In both branches the child is awaited before moving on, so a cancelled turn can
never outlive the record that spawned it — no orphan child, no turn running
against a closing checkpointer.

Tests: interrupting an active inbox turn cancels that turn, retains its record,
leaves the drain task identity **unchanged**, and a subsequent arrival is still
processed; shutdown during an active child cancels and drains the child before
the checkpointer closes. Mutation: inline `_handle_invoke_owned` so
`current_worker_task` is the consumer ⇒ RED.

#### 3.3.0a Lock scope: per-record, ordering guarantee stated honestly **[r3 — review MAJOR 1]**

r2 was self-contradictory: §3.3.0 claimed "acquire once" and rejected
per-record reacquire, while the §3.3.1 pseudocode acquired *inside* the loop.
The review is right that this must be decided, not left ambiguous.

**Decision: per-record locking (acquired by the child turn, not the drain —
§3.3.0/§3.3.0b), and the guarantee is weakened to match.**

Guarantee provided **[r4 — review MAJOR 2, weakened again]**:

* **Attempts begin in `seq` order within each wake.** The snapshot is sorted by
  `seq` and attempted in that order.
* **Later records may overtake a retained failure.** If `seq1` returns
  `success: false` it stays on disk while `seq2` succeeds and is unlinked; on a
  later wake `seq1` is re-attempted *after* `seq2` was already delivered.
* **Not** mutual exclusion against unrelated turns: a user-initiated turn (or a
  startup query) may interleave *between* two queued records.

r3 called this "FIFO among spool records", which the review correctly rejects
as too strong — the retained-failure case above is not FIFO delivery.

Rejected alternative: stop the batch at the first retained failure, restoring
strict FIFO. Rejected because one permanently failing record would then block
every later message from the same peer indefinitely — head-of-line blocking
that converts a single failed turn into a total delivery outage. Progress and
fairness matter more here than strict ordering, especially since §3.3.0a
already concedes ordering against unrelated turns. Delivery is at-least-once
and unordered-on-retry; that is the honest contract and it is what the tests
assert.

Rationale for choosing this over batch-holding: holding `_invoke_lock` across
an entire batch makes a large inbox monopolise the agent — every user message
would block behind N queued inter-agent turns of unbounded length. The review
correctly notes that batch-holding would additionally require a bounded-batch
and fairness policy. Interleaving an unrelated turn between two records is
harmless: **[r5]** attempts still *begin* in `seq` order within a wake (subject
to the overtake caveat above), and inter-agent messages have never been ordered
against user messages anyway. Fairness comes free, since the drain never holds
the lock between records (§3.3.0).

This weakened guarantee is deliberate and is what the tests assert.

#### 3.3.1 Drain loop and delete boundary **[r2 — review MAJOR 3]**

```
async def _drain_once():
    entries = sorted(inbox pending entries by seq)   # snapshot, taken once
    for path in entries:
        record = parse(path)
        if record is None:
            quarantine(path); continue              # §3.3.5
        # [r5] child task → current_worker_task is the TURN, not us (§3.3.0b).
        # _handle_invoke (public) takes _invoke_lock itself; drain holds nothing.
        child = asyncio.create_task(self._handle_invoke(
            {"command": "invoke", "message": record["message"],
             "stream": True, "reply_back": record["reply_back"],
             "inter_agent_delivery": True,
             "delivery_id": record["delivery_id"]},
            connection=None,
            notify_callback=self._inbox_notify,     # §3.3.3 — read per record
            approval_callback=None,
        ))
        try:
            result = await child
        except asyncio.CancelledError:
            if self._shutting_down or asyncio.current_task().cancelling() > 0:
                await _cancel_and_drain(child)      # no orphan
                raise                               # shutdown path — §3.3.4
            log(delivery_id, "turn interrupted")    # user interrupt — §3.3.0b
            continue                                # retain record, stay alive
        except Exception:
            log(delivery_id)                        # retain record, keep going
            continue                                # next record in snapshot
        # DELETE BOUNDARY — reached only on normal return of the await above
        if _delivery_succeeded(result):
            unlink(path)
            log structured: delivery_id, arrival ts, processing duration
        # else: record stays on disk, but is NOT retried in this wake — §3.3.1a
```

#### 3.3.1a Retry policy **[r3 — review BLOCKER]**

> **AMENDED 2026-08-27** by
> [`DESIGN_INBOX_STALE_REDELIVERY.md`](DESIGN_INBOX_STALE_REDELIVERY.md).
> The no-self-rearm rule below stands. What this section got wrong is
> *retention*: a record retained here is retried on **every** future arrival,
> forever, ahead of fresh traffic — so one lost `unlink` degrades the channel
> permanently rather than producing a single documented duplicate (see the
> crash-window table below, observed live; see
> a related internal incident analysis, not included here).
> Superseded by: a `delivery_id` dedup ledger, an attempt cap, a record TTL,
> retirement to `expired/`, and fresh-before-retained attempt order.

r2 combined "a failed record stays on disk" (§3.3.1) with "re-arm if any
pending entry exists" (§3.4.1). Together those form a **hot retry loop**: a
deterministically failing record re-arms the event immediately, with no new
external input, and burns unbounded LLM turns, side effects and API cost while
starving everything else. The review is right; this is worse than the bug being
fixed, because it is self-sustaining.

**Policy: each record is attempted at most once per wake.**

* `_drain_once()` works from a **snapshot** taken at the start of the pass. A
  record that fails is not re-read within that pass.
* A retained (failed) record **never** re-arms the event by itself.
* Retry happens only on a genuinely new edge:
  1. **a new arrival** (producer publishes and sets the event), or
  2. **process startup** (§3.4), or
  3. an explicit manual retry, if one is ever added.

So a permanently failing record costs exactly one attempt per external event,
never a spin. In the worst case it sits on disk — visible, greppable, and
never silently lost, which is the property this whole design exists to
guarantee.

Consequence, stated plainly: a failed record may wait until the next inbound
message. That is acceptable because failure here means the recipient's own
turn failed, and retrying it in a tight loop would not fix it.

Test (per review): one `{"success": false}` produces exactly one attempt and
the drain task then parks on `event.wait()` — asserted by attempt count plus
the task being blocked, not by a sleep.

Observable delete criterion, as the review demanded:

* **[r5]** the child task (`_handle_invoke`) **returned normally** *and*
  `result.get("success")` is truthy ⇒ unlink.
* **[r5]** Cancellation is handled at two levels (§3.3.0b):
  * **child cancelled by a user interrupt** (drain itself not cancelled) — the
    `CancelledError` is deliberately **absorbed**, the record is retained, and
    the pass continues with the next entry;
  * **drain itself cancelled** (shutdown) — the active child is cancelled and
    awaited, then the error **propagates** and ends the task.
* Any other `BaseException` propagates and is never swallowed.
* In every one of those paths the record is **not** unlinked: `unlink` sits
  after the await on the normal path only, never in a `finally`.
* A returned `{"success": false}` (e.g. a turn that reported an interrupt
  rather than raising) leaves the record for retry.

`delivery_id` is threaded into the turn and lands in
`HumanMessage.additional_kwargs` next to the existing `inter_agent_delivery` /
`reply_back` keys (`agent_process.py:3303-3307`), so a post-crash duplicate is
linkable to its original delivery in the recipient's own history.

**Crash windows, both never-zero:**

| Crash point | Outcome |
|---|---|
| after ack, before `astream` checkpoint | record present, turn never ran ⇒ processed after restart, no duplicate |
| after checkpoint/return, before `unlink` | record present, turn already in history ⇒ exactly one documented duplicate, same `delivery_id` |

Two properties fall out:

* **At-least-once, never-zero.** A duplicate is visible, linkable by
  `delivery_id`, and recoverable; a loss is silent and already corrupted one
  investigation.
* **Attempt order, not delivery order [r5].** Attempts begin in `seq` order
  within each wake — better than today's `create_task` free-for-all, but a
  retained failure may be overtaken by a later record (§3.3.0a). Delivery order
  is explicitly **not** guaranteed.

#### 3.3.3 Ownership seam for the notify callback **[r3 — review MAJOR 2]**

r2's pseudocode sat in `AgentProcess` but passed `self._notify_subscribers`.
That method does not exist there — it is defined on `AgentService`
(`agent_service.py:1423`) and is exactly why today's `_bg_turn` lives in the
service (`agent_service.py:624-643`). The reference was to a non-existent
member.

**Decision: the spool and the drain live in `AgentProcess`; the notify
callback is injected by `AgentService`.**

Rationale for keeping the drain in `AgentProcess` rather than the service: the
drain must run whenever the *process* is alive, including the startup edge that
recovers records left by a restart — which happens before/independently of any
subscriber. The service owns fan-out to subscribers; the process owns the turn
lifecycle. Injection keeps that split intact and avoids `AgentProcess` reaching
upward into the service.

##### Exact startup order **[r4 — review MAJOR 1]**

r3 was ambiguous: it said `AgentService` calls `start_inbox_drain(callback)`
*and* that a process without a service still drains with `None`, without
saying who starts what. Worse, combined with the §3.3.4 "already exists ⇒
no-op" guard, a later service call would have been swallowed and
`_inbox_notify` would have stayed `None` forever. Pinned down as two distinct
operations:

1. **`AgentProcess` starts the drain, exactly once**, during startup once the
   spool, the agent graph and the checkpointer are ready (§3.3.4). At this
   point `_inbox_notify` is `None`. This is what recovers records left by a
   restart, and it is why a process running without a service still drains —
   `_handle_invoke_owned` already tolerates `notify_callback=None`
   (`agent_process.py:3185`).
2. **`AgentService` injects the callback** when it attaches, via
   `set_inbox_notify(self._notify_subscribers)`. This **only assigns
   `self._inbox_notify`**; it never creates a task and is safe to call at any
   time.

So the two concerns are split across two methods rather than overloaded onto
one:

* `start_inbox_drain()` — creates the task, guarded, idempotent, no callback
  argument.
* `set_inbox_notify(cb)` — assigns the callback, no task side effects.

The drain reads `self._inbox_notify` **at each record attempt**, not once at
task creation, so a callback injected after the task started is picked up by
the very next record without restarting anything. Task identity never changes.

Tests (per review): a record pending at startup is processed before/without any
callback; a callback injected later is used by the next record; the drain task
identity is unchanged across injection.

#### 3.3.4 Drain task lifecycle **[r3 — review MAJOR 3]**

r2 said the task is "tracked in `_detached_invoke_tasks` exactly as the startup
task". Verified: that set (`agent_process.py:637, 1601-1604`) only holds
references — nothing cancels or awaits its members, and `shutdown()` closes the
checkpointer at step 4 (`:2370` → `cleanup():2317`) after stopping the RPyC
server at step 2. A long-lived consumer tracked that way could be mid-turn when
the checkpointer closes. Specified now:

**Creation** — by `AgentProcess` itself, exactly once, and only after the spool
directory, the agent graph and the checkpointer are all ready. Guarded by an
`self._inbox_drain_task is not None` check so a second call is a no-op (test:
no duplicate task). **[r4]** The guard is safe precisely because callback
injection is a *separate* method (`set_inbox_notify`, §3.3.3) — a no-op'd
`start_inbox_drain()` can no longer silently discard a callback.

**Shutdown** — a new step inserted in `shutdown()` **between step 2.1 (A2A
stop) and step 3 (persist state)**, i.e. after inbound acceptance has stopped
and strictly before `cleanup()` closes the checkpointer:

1. RPyC server already closed ⇒ no new records can be accepted.
2. **[r5]** set `self._shutting_down = True` **before** cancelling, so the
   drain can tell shutdown from a user interrupt (§3.3.0b).
3. `self._inbox_drain_task.cancel()`, then `await` it with
   `contextlib.suppress(asyncio.CancelledError)`. The drain, on its way out,
   cancels and awaits its active child, so **no orphan turn** survives into
   checkpointer teardown.
4. Only then proceed to persist state and close the checkpointer.

Cancellation mid-turn leaves the record on disk (§3.3.1 delete boundary), so it
is recovered on the next start.

**Fault containment — per record, not per pass [r4 — review MAJOR 3].**
r3 wrapped the whole `_drain_once()` pass in one `try/except`. The review is
right that this strands the rest of the batch: an unexpected exception on
`seq1` would exit the pass and leave `seq2..N` — already triggered, already
durable — unprocessed until some unrelated arrival or a restart. That is
head-of-line stranding and it weakens goal 3.

Corrected to two nested levels:

* **Per record** (inner, shown in §3.3.1): `except Exception` → log with
  `delivery_id`, retain the record, `continue` to the next entry in the
  snapshot. One broken record can no longer take down the rest of the batch.
* **Per pass** (outer, in the drain loop): a defensive `except Exception`
  remains so an error in the scan/snapshot machinery itself cannot kill the
  only consumer while durable records are pending.

Both levels preserve the §3.3.1a invariant — each record is still attempted at
most once per wake, and neither path re-arms the event, so a reproducible
exception cannot become a hot loop. Recovery is the same edge: next arrival or
next startup.

**[r5] Cancellation, precisely.** These `except Exception` clauses never catch
`CancelledError` (it derives from `BaseException` in 3.8+), so cancellation is
governed solely by §3.3.0b:

* **child cancelled, drain not cancelled** (user interrupt) — absorbed by the
  dedicated `except asyncio.CancelledError` branch; record retained, consumer
  survives and continues the pass;
* **drain itself cancelled** (shutdown) — the active child is cancelled and
  awaited, then the error is **re-raised**, ending the task; this is the
  shutdown path.

Any other `BaseException` is swallowed at neither level.

**Abrupt death** (`os._exit`, `SIGKILL`) simply kills the task; the spool is
durable and the startup edge recovers it. That is the incident scenario.

Lifecycle tests: no duplicate task on repeated start; graceful shutdown cancels
and awaits the drain *before* the checkpointer closes; one unexpected
per-record failure does not kill the consumer and does not spin.

#### 3.3.5 Corrupt records **[r2, renumbered r3]**

A record that fails to parse is renamed into `inbox/corrupt/` (same atomic
discipline) and logged. It is never retried and never re-scanned as pending,
so it cannot wake the drain forever or block later records — the review's
"corrupt must not spin" requirement.

> **AMENDED 2026-08-27** by
> [`DESIGN_INBOX_STALE_REDELIVERY.md`](DESIGN_INBOX_STALE_REDELIVERY.md) §3.4.
> This path also swallows **foreign files**: `pending_records()` returns any
> non-`.tmp-` entry, so a file another component wrote into the spool is read,
> fails to parse, and is quarantined with no notice to its author — observed in
> practice (a `0644` review-request `.md`). Intake is now name-gated to
> `<seq>-<uuid4hex>.json`; foreign files are reported at WARNING and left in
> place. `corrupt/` keeps its meaning here: a correctly-named record whose
> *contents* are unparseable.

### 3.4 Waking the recipient (goal 3)

Drain is triggered by two edges, both event-driven, no polling:

* **On arrival** — `exposed_enqueue_message`, after the durable write, sets an
  `asyncio.Event`. This is the fast path and preserves today's latency when the
  agent is idle.
* **On startup** — the spool directory is created/scanned (§3.2.1) and the
  event is set if anything is pending. Placed right after
  `_load_and_consume_restart_notice` (`agent_process.py:~1102`), reusing that
  established startup slot. This is goal 4.

One drain task, created near the existing startup task
(`agent_process.py:1598`). **[r5]** It is held in its own
`self._inbox_drain_task` attribute and is **not** managed like
`_detached_invoke_tasks` — that set only holds references and nothing cancels
or awaits its members, which is precisely the lifecycle gap §3.3.4 closes.

#### 3.4.1 Wakeup protocol **[r2 — review BLOCKER 2]**

r1 said "re-check the directory before clearing", which is the wrong order and
loses a wakeup: drain sees empty → producer publishes and sets → drain clears →
the record sits unprocessed until an unrelated message or a restart. Given
this bug is *about* silent waiting, that was a serious hole.

Corrected protocol — **clear before draining**, never after:

```
# producer (exposed_enqueue_message)
publish_durably(record)     # §3.2, fully fsynced
event.set()                 # strictly AFTER durability
return {"success": True}    # ack strictly AFTER set

# consumer (single long-lived drain task)
while running:
    await event.wait()
    event.clear()           # BEFORE reading the directory
    try:
        await _drain_once() # snapshot + process (§3.3.1)
    except Exception:
        log()               # §3.3.4 — no re-arm on this path
    # NO unconditional self-rearm — see §3.3.1a
```

**[r3]** r2 ended this loop with `if pending_entries_exist(): event.set()`.
That is removed: combined with "a failed record stays on disk" it re-armed on
its own output and spun forever (review BLOCKER, §3.3.1a). Re-arming is now
solely the producer's job.

Why this is still race-free without the self-rearm: the ordering
`publish → set` in the producer versus `clear → scan` in the consumer covers
both interleavings.

* A publish that lands **after** `event.clear()` calls `event.set()` itself, so
  the next `await event.wait()` returns immediately — even if the scan already
  missed the file.
* A publish that lands **before** `event.clear()` is already on disk, so it is
  in the snapshot that `_drain_once()` takes right after the clear.

The `clear()` can only ever discard a notification for work that the very next
scan is about to pick up. No durable record is left with the event unset.

This is **not** polling: the loop makes no timed wakeups and blocks
indefinitely on `event.wait()` when the spool is empty. There is no periodic
directory scan — a scan happens only in response to a producer event or the
startup edge.

Deterministic tail-race test **[r3, reformulated per review]**: a barrier
releases the producer's `publish + event.set()` exactly at the point where
`_drain_once()` has finished its snapshot, then asserts the record is processed
**because the producer set the event** — not because the consumer re-armed on
finding a pending file. Formulating it this way keeps the test honest: it
cannot be satisfied by an unconditional self-rearm, so it does not mask the
hot-loop defect. Mutation: move `clear()` after `_drain_once()` ⇒ RED.

**[r5]** Because the child turn still contends for `_invoke_lock` (the drain
holds nothing, but `_handle_invoke` acquires it), a message arriving mid-turn is
still processed only at turn end — the 39-minute latency is not eliminated by
this change. What changes is that it can no longer be *lost*, and the record is
externally visible while it waits. Reducing the latency itself means
interrupting or preempting a running turn, which is a behavioural change to the
turn model and out of scope here; noted in §7 for the reviewer.

## 4. Files touched

| File | Change |
|---|---|
| `deep_worker_cli/rpyc_service/agent_service.py` | `exposed_enqueue_message`: durable publish → `event.set()` → ack; `_bg_turn` and its `create_task` removed; **[r4]** injects `_notify_subscribers` via `set_inbox_notify` on attach — never starts the task (§3.3.3) |
| `deep_worker_cli/process/agent_process.py` | eager `inbox/` creation + `.tmp-*` cleanup + `next_seq` init at startup (§3.2.1); **[r4]** `start_inbox_drain()` (task, guarded) and `set_inbox_notify(cb)` (callback only) as two separate methods (§3.3.3); drain task + **[r3]** cancel/await in `shutdown()` between steps 2.1 and 3 (§3.3.4); **[r5]** per-record child task via public `_handle_invoke` + `_shutting_down` flag for interrupt/shutdown disambiguation (§3.3.0b); `delivery_id` into `additional_kwargs` |
| new: `deep_worker_cli/process/inbox_spool.py` | spool: atomic publish, seq allocation, scan, quarantine — isolated for unit testing |

No change to `client/agent_client.py`, to `tools.py`, to the RPyC signature, or
to the wire format. **[r2]** `tools.py:890` already surfaces a false ack, so
goal 2 needs no tool-side edit.

## 5. Test plan (TDD — RED first)

**Unit** (`tests/unit_tests/process/test_inbox_spool.py`)

1. write → file exists with mode `0o600`, dir `0o700`
2. atomic publish: no `.tmp-` file is ever visible as a queue entry
3. ordering: drain yields records in `seq` order
4. corrupt record: quarantined to `inbox/corrupt/`, does not block later
   records, and does **not** re-wake the drain forever **[r2]**
5. record survives simulated abrupt exit (write, then read from a fresh object)
6. `.tmp-*` leftovers are cleaned at startup and never treated as pending **[r2]**
7. `next_seq` on startup = max over pending **and** quarantined + 1 **[r2]**
8. concurrent enqueue with forced interleaving: acknowledged order preserved,
   no overwrite **[r2 — review MAJOR 1]**

**Unit** (`tests/unit_tests/process/test_inbox_drain.py`) **[r2]**

9. **no deadlock [inverted in r5]:** the drain holds **no** lock and each
   record runs via a child task calling the public `_handle_invoke`, which
   takes `_invoke_lock` exactly once. Test carries a timeout so a reintroduced
   outer `async with _invoke_lock` fails loudly instead of hanging.
   **[was: "calls `_handle_invoke_owned`, never `_handle_invoke`" — r5 reverses
   this; see §3.3.0b]**
10. **event tail-race [r3, restated r5]:** the barrier releases the producer's
    `publish + event.set()` exactly after `_drain_once()` has taken its
    snapshot ⇒ the record is processed **because the producer set the event**.
    There is no re-arm to test against — the phrase "between the emptiness
    check and the re-arm" is obsolete since the self-rearm was removed (§3.4.1).
11. delete boundary: unlink only after normal return **and** truthy `success`
12. **interrupt safety [r5 — review BLOCKER]:** interrupting an active inbox
    turn cancels the **turn**; the record is retained, the drain task identity
    is **unchanged**, and a subsequent arrival is still processed. Mutation:
    inline `_handle_invoke_owned` so `current_worker_task` is the consumer ⇒
    the drain dies ⇒ RED.
13. **no hot retry loop [r3 — review BLOCKER]:** one `{"success": false}`
    produces exactly **one** attempt, the record stays on disk, and the drain
    task then parks on `event.wait()`. Asserted by attempt count + task blocked,
    never by a sleep. Mutation: restore the unconditional self-rearm ⇒ the
    attempt count grows without bound ⇒ RED.
14. `delivery_id` reaches `HumanMessage.additional_kwargs`
15. startup with pre-existing records drains them without any new arrival **[r2]**
16. **ordering [r3, tightened r4]:** attempts begin in `seq` order within a
    wake; an unrelated turn interleaving between two records is permitted;
    **and** a retained `seq1` failure followed by a successful `seq2` is
    asserted as *allowed* — pins the overtake semantics of §3.3.0a so the test
    cannot drift back to claiming FIFO
17. **notify seam [r3, split r4]:** `AgentProcess` starts the drain with
    `_inbox_notify = None` and a startup-pending record is still processed;
    `set_inbox_notify(cb)` injected later is used by the **next** record;
    drain task identity unchanged across injection; `start_inbox_drain()`
    called twice creates no second task **and** does not clear the callback
18. **per-record fault containment [r4 — review MAJOR 3]:** `seq1` raises an
    unexpected exception, `seq2` succeeds **in the same wake**; `seq1` is
    retained, `seq2` is unlinked, the consumer survives and parks
19. **lifecycle [r3 — review round 2 MAJOR 3, extended r5]:** repeated start
    creates no duplicate task; `shutdown()` cancels **and awaits** the drain
    before the checkpointer closes; **[r5]** shutdown during an *active child*
    cancels and drains that child too (no orphan turn) and the drain re-raises
    so shutdown proceeds; one unexpected per-record exception leaves the
    consumer alive, parked, and not spinning

**Unit** (`tests/unit_tests/rpyc_service/`)

20. persist raises ⇒ `success: false` (goal 2) — RED against current code,
    which always returns `success: true`
21. ack ordering: durable write completes, then `event.set()`, then ack **[r2]**

**Integration** (`tests/integration_tests/`, real transport, no mocks)

22. A→B with B idle: round-trip, message in B's history
23. A→B with B's `_invoke_lock` held by a **test-only real-process barrier**:
    prove ack returned *and* the spool file exists on disk *before* B's turn
    ends, then release and assert processing. **[r2]** Ordering is proven by
    observed events (file stat vs ack return), never by `sleep` — same
    discipline as the Kotlin post-write barrier in `8c8bcb2`.
24. **Loss reproduction:** hold B's lock via the same barrier, send, assert the
    file exists, `SIGKILL` B (equivalent to `restart_self`'s `os._exit`),
    restart B, assert the message is processed. Exact incident scenario;
    fails against current code. **[r2]** The kill happens only after the
    file-exists assertion, so the test cannot pass by timing luck.

Mutation checks per project practice — each must turn a test RED:

* ack-before-write (test 20)
* `event.clear()` moved after the drain pass (test 10)
* `unlink` moved into a `finally` (tests 11–12)
* startup drain removed (test 15)
* seq allocation outside the lock (test 8)
* **[r3]** unconditional self-rearm restored (test 13)
* **[r3]** shutdown not awaiting the drain before checkpointer close (test 19)
* **[r4]** the per-record `try` moved back to per-pass, stranding the rest of
  the batch (test 18)
* **[r4]** `set_inbox_notify` folded back into `start_inbox_drain`, so the
  idempotence guard discards a late callback (test 17)
* **[r5, inverted]** the turn inlined via `_handle_invoke_owned`, so
  `current_worker_task` becomes the consumer and a user interrupt kills the
  drain (test 12 — consumer identity)
* **[r5]** an outer `async with _invoke_lock` reintroduced around the child, so
  the child's own acquire in `_handle_invoke` deadlocks (test 9 — timeout)

  **Note:** production must call the *public* `_handle_invoke` from a child
  task; the pre-r5 phrasing "drain calling `_handle_invoke` instead of the
  owned path" described the opposite and is obsolete.
* **[r5]** the shutdown branch absorbing `CancelledError` instead of
  re-raising, or skipping the child cancel-and-drain, leaving an orphan
  (test 19)
* **[r5]** the user-interrupt branch re-raising instead of absorbing, killing
  the consumer (test 12)

## 6. Policy compliance

* No polling — arrival event + startup edge only.
* No new threads — one asyncio task on the existing loop.
* Permissions 0o700/0o600, never 0777/0666.
* No architecture/contract change: same RPyC method, same signature, same
  return shape, same `_invoke_lock` turn model.
* Work on `dev`, then fast-forward to `master`.

## 7. Review outcomes and deferred work

### Settled in review round 1 **[r2]**

1. **At-least-once vs at-most-once** — CONFIRMED at-least-once, on condition
   that duplicates carry a stable delivery identity and are diagnosable.
   Implemented as `delivery_id` (§3.1.2) threaded into
   `HumanMessage.additional_kwargs` and the crash-window table (§3.3.1).
2. **Retention** — RESOLVED per reviewer recommendation: immediate `unlink`
   after a successful child turn (public `_handle_invoke`, §3.3.0b), plus a
   structured log line
   (`delivery_id`, arrival ts, processing duration). No `inbox/done/`: keeping
   full message bodies would enlarge sensitive-data retention and require cap
   machinery, while a pending record + agent history + stable `delivery_id`
   already give forensics. `inbox/corrupt/` is retained deliberately — it is
   small, bounded, and by definition unparseable.
3. **Latency/preemption stays out of scope** — CONFIRMED. This fix closes loss
   and the lying ack; it must not change the turn model.
4. **The separate reply-processing incident stays separate** — CONFIRMED. The
   spool deliberately does not try to repair a graph parked at `next = []`;
   it only guarantees the message exists and is offered to a turn.
5. **Durable ack without an end-to-end processing receipt** — CONFIRMED
   sufficient for this iteration.

### Deferred follow-ups (not in this fix)

* **First-class `sender` field** (§3.1.2). Needs a backward-compatible
  optional arg on `exposed_enqueue_message` — a real contract delta, so it
  gets its own design/review rather than riding along here.
* **Sender-side outbox with retry** — the only way to cover a recipient that
  is fully down (§2 goal 4).
* **Turn preemption** — would address the 39–44 minute latency.
