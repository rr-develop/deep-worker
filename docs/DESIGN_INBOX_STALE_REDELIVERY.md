# DESIGN: stale inter-agent re-delivery — dedup, attempt cap, TTL

> Fixes the defect analysed in a related internal incident analysis (not
> included here).
> Amends `docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md` (signed off, r6) —
> specifically §3.3.1a (retry policy) and §3.3.5 (corrupt records).
> Status: **design, revision 4** — addresses review round 3 from the reviewer
> (1 blocker: `fstat`-based create detection is not implementable and would
> re-admit the r2 silent-loss window; `O_EXCL` is now the only specified
> branch). Rounds 1–2 (6 blockers) resolved in r2/r3 and accepted. Changes are
> marked **[r2]** / **[r3]** / **[r4]**.

## 1. Problem restated

The durable spool guarantees *at-least-once*. It has no machinery to bound the
"more than once" half, so a record that is delivered but not unlinked replays
**verbatim, forever, ahead of fresh traffic**.

Three independent gaps, all in the consumer:

* **D1 — no dedup.** `delivery_id` exists in the record (`inbox_spool.py:187`)
  and is threaded into `HumanMessage.additional_kwargs` (r6 §3.3.1a) purely so
  a duplicate is *diagnosable*. Nothing ever *suppresses* one. r6 §7.1 accepted
  at-least-once "on condition that duplicates carry a stable delivery identity
  and are diagnosable" — that condition was met, but the reviewer's premise was
  that duplicates would be rare crash artefacts, not a steady state.
* **D2 — no attempt cap and no TTL.** `_deliver_inbox_record()`
  (`agent_process.py:3112-3127`) unlinks **only** on
  `result.get("success")`. Three outcomes retain the record forever: user
  interrupt (`:3104`), exception (`:3110`), `success=False` (`:3123`). r6
  §3.3.1a deliberately refused a self-rearm to avoid a hot loop — correct — but
  paired that with unbounded retention, so "retry on the next arrival" means
  "replay on **every** future arrival, indefinitely".
* **D3 — the spool directory is unguarded.** Any file dropped into
  `inbox/` is picked up by `pending_records()` and, if unparseable, silently
  moved to `corrupt/` with no log and no notice to anyone.

### 1.1 Live evidence (this is not theoretical)

The inbox of one agent (`agent_a`) — two records from **2026-08-14**,
pending for 13 days:

| seq | delivery_id | body |
|---|---|---|
| `000000000044` | `828e9570…` | `reviewer -> agent_a :> DESIGN + CODE CHANGES REQUESTED …` |
| `000000000045` | `5340d5a9…` | `reviewer -> agent_a :> DESIGN + CODE SIGN OFF …` |

Both `delivery_id`s are present in that agent's `threads.db` — `checkpoints` (4 and
2 rows) and `writes` (2 and 1). **The turns already ran and are in history**;
only the `unlink` was lost. This is exactly the second row of the r6 crash-window
table, in the wild. `agent_a` is currently not running, so both records are armed
to replay as fresh turns on its next start — a stale `CHANGES REQUESTED`
delivered *before* a stale `SIGN OFF`, thirteen days after the review closed.

These files are **evidence and must not be hand-deleted** until the fix ships
(explicit owner instruction). The fix must migrate them, not require a human to
clean up.

### 1.2 Observed symptoms this explains

From the eight-round two-agent review cycle: verdicts lagging a
round; byte-identical repeats of already-withdrawn `CHANGES REQUESTED`;
multiple `SIGN OFF`s for one snapshot; each stale arrival interrupting an
in-flight `send_message_to_agent`. All follow from a retained record being
replayed ahead of fresh traffic by `_drain_inbox_once()` (`:3037`), which walks
a whole-directory snapshot in `seq` order.

## 2. Goals

1. A record whose success has been **durably ledgered** is never delivered
   again, across restarts. See §2.1 for the precise, deliberately narrow
   statement of this guarantee.
2. A record that cannot be delivered is retired after a bounded number of
   attempts and a bounded lifetime — **never silently**.
3. A record that has already been attempted never overtakes fresh traffic.
4. Foreign files in the spool are rejected loudly, not silently quarantined.
5. Existing pending records (the `agent_a` pair) are handled correctly by the
   new code with no manual intervention.

### 2.1 The exact guarantee **[r2 — review BLOCKER 3]**

r1 claimed "delivered once is never delivered again" and "at-most-once
observable delivery". **Both were false**, and contradicted by r1's own §3.1,
which admits a crash after a successful turn but *before* the ledger append
still replays. The reviewer is right; stating it precisely:

| seam | guarantee |
|---|---|
| turn succeeded **and** id fsynced to the ledger | duplicate is **suppressed** — at-most-once from here on, across restarts |
| turn succeeded, crash **before** the ledger append | **at-least-once** — replays exactly once more, same `delivery_id`, as in r6 |
| turn did not succeed | retained, bounded by attempts/TTL, retired loudly |

So the change does **not** convert the channel to at-most-once. It closes the
*unbounded* replay (the actual defect) and leaves a single-duplicate window
that is strictly narrower than r6's. The pre-ledger seam stays at-least-once on
purpose: the alternative is ledgering before the turn is known to have
succeeded, which trades a visible duplicate for a silent loss — forbidden by
POLICIES §9/§12 and by the incident this subsystem exists to fix.

### Non-goals

* Sender-side outbox, turn preemption, first-class `sender` field — still
  deferred (r6 §7).
* Changing the RPyC contract, the ack semantics, or the `_invoke_lock` turn
  model. The publish→event→ack ordering is correct and stays.
* **Exactly-once, and at-most-once in general.** Only the post-ledger seam is
  at-most-once (§2.1). Anyone reading this design as "duplicates are now
  impossible" has misread it.

## 3. Approach

### 3.1 Dedup ledger (D1)

**Where the set lives.** A separate append-only journal beside the spool:

```
<data_dir>/inbox/
    <seq>-<uuid4hex>.json     0o600   pending           (unchanged)
    corrupt/                  0o700   unparseable       (unchanged)
    expired/                  0o700   retired, NEW
    delivered.log             0o600   dedup ledger, NEW
```

`delivered.log` is one `delivery_id` hex per line (32 chars + `\n`, fixed
width). It is **not** JSON: it must be appendable with a single `write` that
cannot half-corrupt a prior entry, and readable even if the last line is torn.

**Write ordering — this is the load-bearing part.** The ledger entry is
appended and made durable **before** the `unlink`:

```
turn returns success
  → append delivery_id to delivered.log, make durable      (1)
  → unlink the record                                      (2)
```

A crash between (1) and (2) leaves the record pending *and* the id in the
ledger — the next drain sees the record, finds the id already delivered, and
skips + unlinks it. That converts the r6 crash-window duplicate into a
**suppressed** duplicate. A crash before (1) is unchanged from r6: the turn ran
but is not recorded, so it replays once — at-least-once is preserved, which is
deliberate. We never move loss into the ledger's failure mode.

#### 3.1.1 What "make durable" means, precisely **[r3 — review BLOCKER 1]**

r2 wrote "append … fsync" and treated that as one atomic obligation. It is
three, and the reviewer found a **silent-loss window** in the one r2 skipped:
`fsync` on a newly created file does not make its **directory entry** durable.
So on the very first delivery for an agent:

```
create delivered.log → write+fsync the file → unlink the record → CRASH
```

can recover with the record gone and the ledger **pathname absent** — the
message is lost with no trace. That is precisely the failure class this whole
subsystem exists to prevent, and r2 would have shipped it.

**Create detection must be race-free [r4 — review BLOCKER].** r3 said "note
whether `O_CREAT` actually created the file (stat the fd, or attempt `O_EXCL`
first and fall back)". The first of those is **not implementable**: `fstat` on
the fd cannot tell whether *this* `open` created the file or opened an existing
one. An implementer taking that branch would silently reopen the r2
silent-loss window — the design would have permitted its own defect back in.
The `O_EXCL` branch is the *only* correct one, so it is now the specified one,
not an alternative.

The append is therefore specified as exactly this sequence:

1. **Try create:** `os.open(ledger, O_WRONLY | O_CREAT | O_EXCL | O_APPEND |
   O_CLOEXEC, 0o600)`.
   * **Success** ⇒ `created = True`. This process created the file; there is no
     ambiguity and no race.
   * **`EEXIST`** ⇒ `created = False`; open the existing file (step 1b).
   * **Any other error** ⇒ raise. There is no fallback path: an unexpected
     `open` failure must not be recovered into a weaker branch.
2. **1b — open existing:** `os.open(ledger, O_WRONLY | O_APPEND | O_NOFOLLOW |
   O_CLOEXEC)`, then `fstat` the fd and **fail closed** unless it is a regular
   file, owned by us, mode `0o600`, `st_nlink == 1`. The check runs **before
   any write**, so a rejected file is never appended to and the record stays
   pending. (Same posture as `storage_paths._verify_not_swapped`.)

   **Scope of this check [r4 — reviewer].** It catches *accidental* corruption
   of the spool — a symlink, a directory, a hardlink, a wrong-mode file left by
   another tool or an earlier bug. It is **not** an adversarial guarantee: it
   does not defend against a hostile process running as the same UID that
   replaces the pathname after the `open`, and no claim to the contrary should
   be read into it. Same-UID integrity here rests on the single-owner /
   single-writer precondition recorded below, not on these checks. Defending
   against that adversary would need file locking and a proper threat model,
   which is a separate design and review — deliberately out of scope.
3. **Write** all 33 bytes (`32 hex + \n`) in a **loop**, handling short writes
   and retrying `EINTR`.
4. **`os.fsync(fd)`** — file contents durable.
5. **If and only if `created`:** `fsync` the **inbox directory** fd, so the
   ledger's directory entry is durable. This is the step r2 lacked.
6. Only after 1–5 return successfully does the caller `unlink` the record.

If **any** of 1–5 raises or the write is incomplete, the record is **not**
unlinked. The delivery then replays once on the next pass — a visible
duplicate, never a loss. That direction of failure is chosen deliberately and
is the same trade the pre-ledger seam makes (§2.1).

The steady-state append (the `EEXIST` branch) needs **no** directory fsync: the
entry is already durable, so step 4 alone is sufficient.

**Single writer per agent — stated as a precondition [r4].** `O_APPEND` makes
each individual `write` atomic with respect to other appenders, but a write
loop resuming after a short write could in principle interleave with another
appender and tear a line. That cannot happen here: the ledger is appended
**only** from `_deliver_inbox_record`, which runs on the single long-lived
drain task, and exactly one agent process owns a given `data_dir`. So there is
one writer per agent by construction and no lock is required. This is now a
**recorded precondition**, not an accident: if a second appender is ever added
(a sender-side outbox, an admin tool), it must take an exclusive lock or the
torn-line reasoning in this section stops holding. A torn line is still
recoverable (it is discarded on load and its record is still pending), so the
failure mode of violating this precondition is a duplicate, not a loss.

**No dir-fsync is needed after the `unlink`.** If the unlink is not durable and
the record resurrects after a crash, it is suppressed by the ledger — which is
durable by construction at that point. This asymmetry is why the ordering is
ledger-first: the ledger protects the record, so the record's own removal does
not need protecting.

**Torn final line — why dropping it is safe [r3].** A truncated last line is
discarded on load, and that is sound for exactly one reason: a torn line means
the write loop (step 3) or the file fsync (step 4) did not complete, so the
caller never reached the `unlink` and the
**pending record is still on disk**. Dropping the id therefore costs at most
one replay of a record that still exists — never a loss. If the unlink could
precede a confirmed fsync, this reasoning would collapse and the torn line
would be indistinguishable from a lost message. Test 10 pins the discard;
mutation "unlink before fsync returns" must turn it RED.

**Which outcomes count as "delivered" for dedup purposes.** Only the outcome
that r6 already treats as terminal:

**Two independent counters [r2 — review BLOCKER 1].** r1 used one `attempts`
field for two unrelated jobs: deciding retirement, and deciding attempt order.
Because a user interrupt deliberately does not increment it, an interrupted
record stayed at `attempts == 0` and would be classed **fresh forever** — so it
would keep overtaking new traffic, which is the exact bug this design exists to
kill. The reviewer caught a real defect. The record therefore carries:

* `attempted` (bool) — "this record has been handed to a turn at least once".
  Set on **every** outcome below except drain-cancellation. Drives **ordering
  only** (§3.3).
* `failure_attempts` (int) — drives **retirement only** (§3.2). Never
  incremented by a user interrupt.

| outcome | ledger | record | `attempted` | `failure_attempts` |
|---|---|---|---|---|
| child returned, `success` truthy | **append** | unlink | — | — |
| child returned `success=False` | no | retain | `True` | `+1` |
| child raised | no | retain | `True` | `+1` |
| child cancelled by user interrupt | no | retain | **`True`** | **unchanged** |
| drain cancelled (shutdown) | no | retain | unchanged | unchanged |

The interrupt row is the whole point of the split: the record **is** demoted
out of the fresh lane (it was attempted, so it must not overtake new mail), but
it is **not** charged toward retirement — r6 §3.3.0b worked hard to make an
interrupt kill the turn rather than the consumer, and charging the record would
let a user who interrupts three times silently expire a message nobody read.

Shutdown-cancellation touches neither counter: the turn never got a fair
attempt, and the process is going down anyway.

Both fields are absent on records written by the old code;
`record.get("attempted", False)` / `record.get("failure_attempts", 0)` is the
compat path the two live `agent_a` records take.

**Bounding the ledger [r2 — review BLOCKER 2].** r1 stated the right principle
("keep ids that can still suppress something") and then implemented something
different: "keep the last `LEDGER_KEEP` ids". Those are not equivalent, and the
gap is a live re-arming of the bug. If a ledgered record survives its `unlink`
(precisely the `agent_a` failure) and more than `LEDGER_KEEP` deliveries follow,
its id is evicted by recency while the record is **still on disk** — and it
replays, defeating goal 1. Recency is the wrong key.

**Compaction is by relevance, not recency.** An id can only ever suppress a
record that is still pending, so the retained set is:

```
keep = {ids in ledger} ∩ ({delivery_id of each pending record} ∪ in_flight)
```

where `in_flight` is the id currently being delivered, if any — it has been
ledgered but its `unlink` has not run yet, so evicting it mid-delivery would
resurrect it. Compaction runs at startup (§3.6), after the sweep, and rewrites
via the same tmp→fsync→rename→dir-fsync chain `inbox_spool.publish` uses.

This is exact, not heuristic: **every** id that can still suppress something is
kept, and every id that cannot is dropped. Size is bounded by the number of
pending records rather than by an arbitrary constant, so the steady state on a
healthy agent is an empty ledger — pending is empty, so the intersection is
empty.

**The cap must never evict a suppressing id [r3 — review BLOCKER 2].** r2 kept
`LEDGER_CAP` as a "safety cap … truncates by recency, logged at WARNING". The
reviewer is right that this directly contradicts test 14: an old ledgered id
whose record is still pending can sit **beyond** the cap by position, and
truncating by recency would evict it and re-enable the replay. A cap that can
break correctness is not a safety measure. Worse, r2 was incoherent on its own
terms — if truncation only logs a WARNING without removing anything, it is not
a cap at all.

The algorithm is therefore ordered so correctness never depends on position:

1. Collect `pending_ids` from the post-sweep pending scan (plus `in_flight`).
2. **Stream** the ledger line by line — never load it whole — and keep every id
   in `pending_ids`, **regardless of its position in the file**. This set is
   the correctness set and is never truncated.
3. Discard every id not in `pending_ids`; they can no longer suppress anything.
4. Write the correctness set back atomically.

The result is bounded by `len(pending_ids)`, not by `LEDGER_CAP`, and step 2's
"regardless of position" is the property test 14 pins.

`LEDGER_CAP` is retained **only** as a fail-loud tripwire on the
*pre-compaction* file size (default **50000** lines). Exceeding it means
appends are outrunning compaction — a real anomaly worth surfacing — so it logs
at ERROR and, if the correctness set itself somehow exceeds the cap, the process
**fails loud rather than truncating**: dropping a suppressing id to save disk
would trade a bounded file for an unbounded replay. It never silently discards.
Streaming in step 2 means a large ledger costs I/O, not memory, so there is no
pressure to truncate for safety in the first place.

### 3.2 Attempt cap and TTL (D2)

**Both counters live in the record**, not in memory — they must survive
restart, which is the whole point.

`build_record` gains `"attempted": false` and `"failure_attempts": 0` (§3.1).
Updating either rewrites the record via the same atomic tmp→rename chain,
preserving the filename (so `seq` and therefore arrival order are untouched).
Records written by the old code have neither key and default to
`False` / `0` — the compat path the two live `agent_a` records take.

Retirement is driven by `failure_attempts` **only**; `attempted` never retires
anything.

Defaults, configurable (§3.5):

| knob | default | meaning |
|---|---|---|
| `INBOX_MAX_FAILURES` | **5** | failed delivery attempts before retirement |
| `INBOX_RECORD_TTL` | **72h** | wall-clock age (from record `ts`) before retirement |

Either bound tripping retires the record. 5 failures because one is consumed
only by a genuine turn failure and only one per external arrival — a message
that fails five separate turns is not going to succeed on the sixth.

**72h [r2 — reviewer's call].** r1 proposed 24h, reasoning from review traffic.
The reviewer overruled: the spool carries more than review verdicts, and
delivery after a weekend outage is legitimate. Accepted — and the 1h stale
marking (below) already protects the recipient's judgement, so a longer TTL
costs little. The `agent_a` pair at 13 days is far past either threshold.

**Retirement is loud and non-destructive.** The record is moved to
`expired/` (never deleted) and logged at **WARNING** with, per the owner's
instruction, `delivery_id`, `seq`, sender, age and attempt count:

```
[INBOX] RETIRED delivery_id=828e9570… seq=44 sender=reviewer
        age=13d2h failure_attempts=0 attempted=false reason=ttl
        — moved to expired/, NOT delivered
```

`sender` is parsed best-effort from the `<sender> -> <recipient> :>` prefix of
the message body **for the log line only**. r6 §3.1.2 correctly refuses to make
that a protocol field; using it in a diagnostic string carries no such
obligation, and it degrades to `sender=?` if the prefix is absent.

**Stale-but-delivered records are marked.** If a record is delivered while
already older than `INBOX_STALE_MARK` (default **1h**), the message body handed
to the turn is prefixed:

```
[STALE MESSAGE — queued 13d2h ago, delivered late; may be obsolete]
reviewer -> agent_a :> DESIGN + CODE SIGN OFF …
```

This is what makes the recipient *model* able to discount the content, which no
amount of logging achieves. The prefix is applied at delivery time only; the
stored record is never rewritten with it.

### 3.3 Delivery order (D3 / goal 3)

r6 §3.3.1a states delivery order is explicitly not guaranteed and that a
retained failure may be overtaken. That stays true, but the current behaviour is
the *opposite* of what we want: the snapshot is walked in ascending `seq`, so a
retained stale record is always attempted **first**.

**New rule: within a pass, never-attempted records go before attempted ones.**
`_drain_inbox_once()` partitions the snapshot on **`attempted`** (not on the
retirement counter): `attempted == False` first, in `seq` order among
themselves, then `attempted == True`, also in `seq` order.

**[r2 — review BLOCKER 1]** r1 partitioned on `attempts`, the same field a user
interrupt deliberately leaves alone. An interrupted record therefore stayed
`attempts == 0`, was classed fresh **forever**, and kept overtaking new
traffic — the original bug, preserved intact for exactly the records most
likely to hit it. Partitioning on `attempted` fixes this: an interrupt demotes
the record out of the fresh lane while still not charging it toward retirement
(§3.1).

This is a strict improvement — a never-attempted record is never delayed by a
known-bad one — and it weakens no r6 guarantee, because r6 guaranteed nothing
about order.

### 3.4 Foreign files in the spool (D3)

`pending_records()` currently returns any non-`.tmp-` file. A foreign file
therefore reaches `read_record()`, fails to parse, and is silently moved to
`corrupt/`. That is how
a review-request `.md` file in another agent's `inbox/corrupt/`
(mode `0644`, written directly by an agent) got there with nobody told.

Two changes:

* **Name-gated intake.** `pending_records()` returns only files matching
  `_RECORD_RE` (`<12 digits>-<32 hex>.json`). Anything else is not a candidate
  record and is never read or parsed.
* **Loud rejection, once per filename per process [r2 — reviewer].**
  Non-matching entries are reported at WARNING (`[INBOX] foreign file in spool:
  <name> mode=0644 — not a delivery record, ignored`) and **left in place**.
  Not moved: moving another component's file is how you lose someone else's
  data. `corrupt/` keeps its r6 meaning — a well-named record whose *contents*
  are unparseable.

  r1 said "once per pass", which the reviewer correctly rejected: a file we
  deliberately refuse to move would then emit a WARNING on **every** arrival,
  forever — log spam that trains readers to ignore the channel. Each distinct
  filename is warned **once per process**, tracked in an in-memory
  `set[str]` on `AgentProcess`; a restart warns again, so the condition stays
  visible to anyone reading a fresh log without being repeated within a run.
  The set is keyed by filename and bounded by the directory's own contents.

This closes the D3 quarantine hole and removes the "unparseable names sort
last but are still returned" branch in `pending_records`, which existed solely
to feed the quarantine path.

### 3.5 Configuration

All four knobs are module constants in `inbox_spool.py`, overridable by
environment variable, read once at process start:

| constant | env var | default |
|---|---|---|
| `MAX_FAILURES` | `DEEP_WORKER_INBOX_MAX_FAILURES` | `5` |
| `RECORD_TTL` | `DEEP_WORKER_INBOX_RECORD_TTL` | `259200` (s = 72h) |
| `STALE_MARK` | `DEEP_WORKER_INBOX_STALE_MARK` | `3600` (s) |
| `LEDGER_CAP` | `DEEP_WORKER_INBOX_LEDGER_CAP` | `50000` (lines, fail-loud tripwire) |

`LEDGER_CAP` is **not** a correctness knob and **never truncates the
correctness set** (§3.1). Compaction keeps exactly the ids that can still
suppress a pending record, regardless of their position in the file. The cap
only detects appends outrunning compaction: exceeding it logs at ERROR, and a
correctness set larger than the cap fails loud rather than dropping a
suppressing id.

Env vars, not CLI flags: this is operator-level tuning for a subsystem with no
per-agent semantics, and adding four CLI flags that must survive `--restart`
would drag in the launch-config copy-back machinery that already lost six
fields once (an earlier restart defect). An invalid value
**fails loud at startup** rather than silently falling back to the default.

### 3.6 Startup sequence

`init_inbox_spool()` (`agent_process.py:2936`), in order:

1. `ensure_spool()` — now also creates `expired/` at `0o700`.
2. `cleanup_tmp()` — unchanged.
3. `next_seq()` — now scans pending ∪ `corrupt/` ∪ `expired/`, so a retired
   record's `seq` can never be reused (same reasoning as r6 §3.1.1).
4. **Ledger load** — read `delivered.log`, drop a torn final line. No
   compaction yet.
5. **Sweep** — for each pending record: if its id is in the ledger, unlink it
   (this is the crash-window cleanup); else if TTL-expired, retire it to
   `expired/`. Both paths log at WARNING/INFO respectively.
6. **Ledger compact** — *after* the sweep, because the sweep is what removes
   pending records and therefore determines the retained set. Keep exactly
   `{ledger ids} ∩ {delivery_id of each still-pending record}` (§3.1) and
   rewrite atomically if it changed.

Compacting **before** the sweep (as r1 had it) would keep ids for records the
sweep is about to delete — harmless but pointless — while the real hazard is
the reverse: compacting by recency at any point can evict an id whose record is
still pending. Ordering 5→6 makes the retained set exact by construction.

Step 5 is what resolves the `agent_a` pair without human intervention: neither
id is in the ledger (the ledger did not exist when they ran), but both are 13
days old, so both are retired to `expired/` with the loud TTL log line — and
never delivered. The evidence is preserved on disk, the replay is disarmed.

Priming stays ALL-OR-NOTHING (`_inbox_seq_primed`): any `OSError` in 1–6
leaves the flag false and publishing refuses, per r6.

## 4. Files touched

| file | change |
|---|---|
| `deep_worker_cli/process/inbox_spool.py` | `expired/` dir; ledger append (loop-until-complete, fsync, dir-fsync on initial create) / streaming compact-by-intersection; `attempted` + `failure_attempts` in `build_record`; atomic counter update; name-gated `pending_records` + foreign-file report; `retire()`; TTL/failure config |
| `deep_worker_cli/process/agent_process.py` | `_deliver_inbox_record` delete boundary → ledger-then-unlink; per-outcome counter updates (§3.1 table); retirement check; stale prefix; `_drain_inbox_once` partition on `attempted`; `init_inbox_spool` sweep-then-compact; foreign-filename warn-once set |
| `tests/unit_tests/process/test_inbox_redelivery.py` | new — tests 1-18 incl. 16b/16c (§5), 20 total |
| `docs/DESIGN_DURABLE_INTER_AGENT_DELIVERY.md` | note in §3.3.1a and §3.3.5 pointing here |
| a related internal incident analysis (not included here) | new incident write-up |

## 5. Test plan (TDD — RED first)

Per POLICIES §10, integration coverage where possible; per the owner's
instruction, the repro drives the **public** path (`publish_inbox_message` +
the real drain), never private helpers.

**RED repros (must fail on current code):**

1. `test_delivered_record_is_not_replayed` — publish, drain to success, then
   *simulate the lost unlink* by restoring the record file, drain again:
   currently delivers twice, must deliver once.
2. `test_dedup_survives_restart` — as above but with a fresh `AgentProcess`
   over the same `data_dir`, proving the ledger is on disk and not in memory.
3. `test_stale_record_retired_not_delivered` — record with `ts` 13 days old
   (the `agent_a` shape, with neither counter key present) is retired to
   `expired/`, never
   handed to a turn, and logged at WARNING with id/seq/age.
4. `test_failure_cap_retires_after_max_failures` — a record whose turn returns
   `success=False` is retired on the 5th failure, not retained forever.
5. `test_user_interrupt_does_not_consume_failure_attempt` — cancelled child
   leaves `failure_attempts` unchanged.
6. `test_fresh_record_attempted_before_retained` — attempt order assertion.
7. `test_foreign_file_is_reported_and_left_in_place` — a `0644` `.md` file in
   the spool is logged, not read, not moved to `corrupt/`.
8. `test_stale_delivery_is_marked_in_body` — delivered-but-old record reaches
   the turn with the `[STALE MESSAGE …]` prefix.

**Added in r2, one per reviewer blocker:**

12. `test_interrupted_record_does_not_overtake_new_message` **[blocker 1]** —
    publish A, interrupt its turn, publish B; assert B is attempted **before**
    A on the next pass. This is the test r1 lacked, and it fails not only on
    today's code but on r1's own proposed design.
13. `test_interrupt_sets_attempted_but_not_failure_attempts` **[blocker 1]** —
    pins the two counters as *independent*, so a future "simplification" that
    re-merges them goes RED.
14. `test_ledgered_pending_id_survives_more_than_cap_deliveries`
    **[blocker 2]** — one ledgered record whose unlink was lost, then >
    `LEDGER_CAP` further deliveries; assert its id is still in the ledger and it
    is still suppressed. Fails on r1's recency compaction.

**Added in r3, one per reviewer blocker:**

16. `test_initial_ledger_create_fsyncs_directory_before_unlink`
    **[blocker 1]** — the *first* delivery for an agent (no `delivered.log`
    yet), i.e. the `O_EXCL`-succeeds branch. Asserts the inbox **directory** fd
    is fsynced, and that it happens **before** the record is unlinked. Without
    it, a crash after the unlink recovers with the record gone and the ledger
    pathname absent — silent loss.
16b. `test_existing_ledger_append_does_not_fsync_directory` **[r4 — reviewer]**
    — the `EEXIST` branch. Asserts the directory is **not** fsynced when the
    ledger already exists, pinning the two branches as genuinely distinct.
    Without this, an implementation that unconditionally dir-fsyncs (or one
    that never does) passes test 16 alone, and the create/open distinction is
    untested.
16c. `test_ledger_open_fails_closed_on_symlink_or_swapped_inode`
    **[r4 — reviewer]** — the `EEXIST` open rejects a symlink, a non-regular
    file, wrong ownership/mode, or `st_nlink != 1`. Asserts the failure happens
    **before any write and before the unlink**: nothing is appended to the
    suspect file and the record stays pending. Rejecting *after* writing would
    leak a delivery id into an attacker-visible file and still risk the unlink.
17. `test_incomplete_ledger_append_leaves_record_pending` **[blocker 1]** —
    a short write / `EINTR` / failing fsync during the append must leave the
    record **not** unlinked, so the delivery replays rather than vanishing.
18. `test_oldest_pending_id_beyond_cap_is_retained` **[blocker 2]** — the
    suppressing id sits **beyond** `LEDGER_CAP` by position; compaction must
    keep it anyway. This is the case r2's recency truncation would have
    silently dropped, re-arming the replay it was supposed to prevent.
15. `test_stale_prefix_preserves_reply_routing` **[blocker 4]** — a stale-marked
    delivery still yields correct sender extraction and `reply_back` handling
    through `InterAgentReplyMiddleware`. Verified against the real parser:
    `_is_inter_agent_message` (`inter_agent_reply.py:47-49`) and
    `_extract_sender` (`:361-364`) scan **per line**, so a prefix line is
    tolerated — but each has a `^`-anchored whole-content fallback (`:51`,
    `:365`) that a future refactor could promote to the only path, silently
    breaking reply routing. This test pins the behaviour at that seam.

**Ledger durability:**

9. `test_crash_between_ledger_and_unlink_suppresses` — append, skip the unlink,
   restart: record is unlinked without a turn.
10. `test_torn_final_ledger_line_is_dropped` — truncated last line does not
    poison the set, **and** its record is still pending (the property that
    makes the discard safe, §3.1.1).
11. `test_ledger_compaction_keeps_only_pending_ids` — after the sweep, the
    ledger contains exactly the ids of still-pending records; ids of unlinked
    records are gone and the file shrinks.

**Mutation checks** (proving the tests pin behaviour, per
an earlier per-turn usage fix): invert ledger-then-unlink to
unlink-then-ledger → test 9 RED; make the interrupt path increment
`failure_attempts` → test 5 RED; partition on `failure_attempts` instead of
`attempted` → test 12 RED; sort retained-first → test 6 RED; swap compaction
back to last-N-by-recency → tests 14 **and** 18 RED; anchor the middleware
parser to whole-content only → test 15 RED; **drop the initial-create directory
fsync** → test 16 RED; **unlink before the append/fsync is confirmed** → tests
10 and 17 RED. A test that stays green under its mutation is a fake test and
gets rewritten.

**[r4] Create-detection mutations.** Two more, because this is where the design
itself nearly re-admitted the defect: replace the `O_EXCL` probe with
`O_CREAT` + `fstat`-based detection → test 16 must go RED (it cannot tell
create from open, so the dir-fsync is skipped or spurious); and
unconditionally dir-fsync on every append → test 16b must go RED. If either
mutation stays green, the branches are not actually being distinguished.

**Isolation** (POLICIES, the rule that tests never touch production state):
every test uses `tmp_path`; no `AgentProcess` without an explicit `data_dir=`.
The real user data directory is `stat`-ed before and after the suite to prove it
was untouched. Every wait is bounded (`_BOUND = 5.0`), per the existing drain
tests.

## 6. Policy compliance

* §2 event-driven — no polling added; TTL is evaluated on the existing wake
  edges, never on a timer.
* §3 no new threads — all work on the existing drain task.
* §4 permissions — `expired/` `0o700`, `delivered.log` `0o600`.
* §9 / §12 no silent failure — retirement is WARNING with full identity;
  foreign files are WARNING; nothing is ever dropped without a trace.
* §11 no known failures — full suite green before sign-off is requested.
* Work on `dev`, FF to `master` after CODE SIGN OFF.

## 7. Review outcomes

### Round 1 — the reviewer, verdict on `e5fc36b` / design sha `644bd2055fc1768c`

**CHANGES REQUESTED, 4 blockers. All accepted; none disputed.**

1. **Ordering field vs interrupt — real defect.** r1 partitioned attempt order
   on the same counter a user interrupt deliberately skips, so an interrupted
   record stayed "fresh" forever and kept overtaking new traffic — the original
   bug, preserved for exactly the records most likely to hit it. Fixed by
   splitting `attempted` (ordering) from `failure_attempts` (retirement),
   §3.1/§3.3; new tests 12, 13.
2. **Recency compaction re-arms the bug — real defect.** r1 wrote the correct
   principle and then implemented "last N ids", which are not equivalent: a
   ledgered record that survives its unlink (the `agent_a` failure) loses its id
   after `LEDGER_CAP` further deliveries and replays. Fixed by compacting to
   `{ledger} ∩ {pending ids}`, §3.1, and by moving compaction after the sweep,
   §3.6; new test 14. `LEDGER_CAP` demoted to a disk-safety cap — which round 2
   showed was still wrong, and r3 replaced with a fail-loud tripwire that never
   truncates the correctness set.
3. **Overclaimed guarantee.** r1's goal 1 and non-goals said "never delivered
   again" / "at-most-once observable", contradicted by r1's own admitted
   pre-ledger crash window. Fixed by §2.1, which states the seam explicitly:
   at-most-once only once success is durably ledgered; at-least-once before it.
4. **Middleware regression coverage.** Accepted. Verified against the real
   parser — `inter_agent_reply.py:47-49` and `:361-364` scan per line, so the
   prefix is tolerated today, but the `^`-anchored fallbacks at `:51`/`:365`
   are the fragile seam the reviewer names. New test 15 pins it.

**Judgement calls, decided by the reviewer:**

* **TTL 72h, not 24h** — accepted. The spool carries more than review verdicts
  and weekend-offline delivery is legitimate; 1h stale marking already protects
  the recipient's judgement.
* **`success=False` keeps consuming a failure attempt**, cap 5 — accepted,
  confirming r1's lean.
* **Foreign-file warning once per process, not per pass** — accepted; a
  deliberately-unmoved file would otherwise spam a WARNING on every arrival
  forever (§3.4).

### Open for round 2

Nothing outstanding from round 1. The one thing I would still like challenged:
§3.1's claim that ledger-then-unlink has **no** window other than the admitted
pre-ledger one. The reviewer called it sound for the post-append case; if there
is a torn-append or dir-fsync gap I have hand-waved, it is the highest-value
thing left to find, because every other guarantee here rests on it.

### Round 2 — the reviewer, verdict on `94dc84a` / design sha `09c293f7…`

**CHANGES REQUESTED, 2 blockers.** Round 1's blockers 1/3/4 accepted as fixed.
The challenge posed just above is what broke — which is why it was posed.

1. **Missing directory fsync on initial ledger create — silent loss.** r2 wrote
   "append … fsync" as a single obligation. `fsync` on a *newly created* file
   does not make its directory entry durable, so the first delivery for an
   agent had the window `create → write+fsync → unlink → crash`, recovering
   with the record gone and the ledger pathname absent. That is a silent loss —
   the exact failure class this subsystem exists to prevent — and r2 would have
   shipped it. Fixed in §3.1.1: open/create detection, a write loop handling
   short writes and `EINTR`, file fsync, and a **directory fsync when and only
   when `O_CREAT` created the file**, all before the `unlink`; any failure in
   that chain leaves the record pending. New tests 16, 17. Also recorded the
   reviewer's two correct refinements: the steady-state append needs no
   dir-fsync, and no dir-fsync is needed after the `unlink`, because a
   resurrected record is suppressed by the by-then-durable ledger.
2. **Size cap could evict a suppressing id.** r2's `LEDGER_CAP` truncated by
   recency, contradicting test 14: an old ledgered id whose record is still
   pending can sit beyond the cap *by position*. r2 was also incoherent on its
   own terms — a cap that only logs a WARNING without removing anything is not
   a cap. Fixed in §3.1: compaction **streams** the ledger and keeps every id in
   `pending_ids` **regardless of position**; the cap never truncates the
   correctness set, logs at ERROR, and fails loud rather than dropping a
   suppressing id. New test 18 places the surviving id beyond the cap.

Also per the reviewer, the **torn-line rule is now justified rather than
asserted** (§3.1.1): discarding an incomplete final line is safe *only* because
an unconfirmed append means the `unlink` never ran and the record is still
pending. Test 10 now asserts that second half too, since it is what makes the
first half sound.

### Round 3 — the reviewer, verdict on `66dec51` / design sha `640eaa97…`

**CHANGES REQUESTED, 1 blocker.** The round-2 cap/streaming fix was accepted
(§3.1 keeps the correctness set regardless of position; tests 14/18 pin it),
and the durability ordering was accepted apart from one point.

1. **`fstat`-based create detection is not implementable.** r3 offered two
   ways to tell whether `O_CREAT` created the file: stat the fd, or probe with
   `O_EXCL`. The first cannot work — `fstat` on the fd says nothing about
   whether *this* `open` created the file — so an implementer following that
   branch would skip the initial-create directory fsync and **silently reopen
   the r2 silent-loss window**. A design that permits its own defect back in is
   a design defect. Fixed in §3.1.1: `O_EXCL` is now the *only* specified
   branch (success ⇒ `created=True` ⇒ dir-fsync; `EEXIST` ⇒ open existing with
   `O_NOFOLLOW` + fail-closed `fstat` on type/owner/mode/`st_nlink`; any other
   error raises, no fallback). Tests split per the reviewer: 16 proves the
   create branch dir-fsyncs before the unlink, 16b proves the pre-existing
   branch does **not**, 16c pins the fail-closed open. Two new mutations
   target exactly this: `fstat`-based detection → 16 RED, unconditional
   dir-fsync → 16b RED.

**Non-blocking point, adopted.** The reviewer noted that `O_APPEND` atomises
each individual `write`, so a loop resuming after a short write could in
principle interleave with another appender. Recorded in §3.1.1 as an explicit
**precondition**: the ledger is appended only from `_deliver_inbox_record` on
the single drain task, and one process owns a given `data_dir`, so there is one
writer by construction and no lock is needed — but if a second appender is ever
added it must take an exclusive lock. Noted also that violating the
precondition yields a torn line, which is a duplicate and not a loss.

### Open for round 4

Nothing outstanding. Seven blockers across three rounds; the two that mattered
most (initial-create dir-fsync, and this round's non-implementable detection of
it) were both in the same few lines, which is itself the argument for the
mechanism being specified at syscall level rather than described in prose.
