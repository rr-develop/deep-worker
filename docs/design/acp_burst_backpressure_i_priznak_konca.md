# ACP notification queue overflow: root cause and solution

**Date:** 2026-10-07 (UTC)
**Scope:** `deep_worker/providers/acp/acp_client.py`, `chat_model.py` — the shared
ACP transport, i.e. **all** agents on ACP providers.
**Supersedes:** the drafts `acp_stream_end_sentinel.md` (the "everything has been read" signal) and
`acp_notification_backpressure.md` (naive backpressure) — both were refuted
by measurements and were never committed to the repository.

## 1. What happens during a burst, step by step

1. The agent produces a long reply. The CLI does not send it whole, but as hundreds of small
   notifications — a separate frame for each text fragment.
2. Frames arrive faster than we process them and pile up in the read buffer.
   About 41,900 frames of 200 bytes fit there (8 MiB limit).
3. The reader is built as "take a line → unpack it → take the next one", and **nowhere
   in this loop does it yield control**. While the buffer has data,
   "take a line" returns it instantly, without suspending.
4. The unpacked frames are put into a box with 1000 slots. The reply collector is supposed to take them out —
   but it **never gets control even once** while the reader grinds through the buffer.
5. On the 1001st frame the box is full. The old code honestly noticed this and did not silently drop data —
   it **killed the whole turn** with an `AcpSinkOverflowError` error.

The agent's reply was lost entirely. Not because of memory, but because two pieces of code
did not alternate.

## 2. Root cause: quadratic text accumulation

Turn text accumulation was called on **every** notification and every time
recomputed the whole volume from scratch (`sum(len(c) for c in acc)`). Measured on the real
function, with 200-byte chunks:

| chunks accumulated | cost of handling one notification |
|---|---|
| 0 | 2.4 µs |
| 1000 | 40.4 µs |
| 4000 | 153.8 µs |
| 8000 | **300.7 µs** (×125) |

The total cost of a turn grew **×4.08 when the input doubled** — quadratic. The culprit
is unambiguously attributed: that very recomputation took **97–101 %** of the cost of
the call, while the `append` itself took 0.15 µs.

This gives both symptoms:

* **Overflow.** Receiving a frame costs 3.4 µs. Starting from ~1000 accumulated chunks
  the collector is an order of magnitude slower than arrival — it loses the race, and the box
  reaches 1000. That is why increasing the capacity only moved the cutoff: capacity N
  survived a burst < N and died at ≈ N+500.
* **"A trivial grep hangs for 120 s".** The drain holds the event loop with no `await`:
  a plain `await` waited **1441 ms** on a turn of 8000 notifications, and at
  the limit of 10,000 chunks — **3.49 s** of continuous CPU. During that time nothing
  else in the process runs.

**An important boundary established by measurement:** linear accumulation is **not enough**.
With linear accumulation the losses are the same (1500/50 → 500, 5000/200 → 4000), because
the reader does not yield control **at all** — the collector's speed is irrelevant here.
So both parts are needed: linear accumulation cures the root cause of the slowness and of the
event-loop stall, and the protection from §4 makes overflow impossible.

## 3. Why the two previous options were rejected

**The "everything has been read" signal.** The reader puts a special marker into the box, and the collector drains
up to it. Rejected: there are providers that send part of the text **after** the formal
"turn finished" response, and our code relies on this (it drains the box even after
receiving the response). The marker stops the collector exactly at the turn boundary, and the entire
tail behind it is **silently discarded** — measured: 50 chunks out of 50 are lost with a
"chunks after the response" stream shape, versus 0 without the marker. The cure turned out to be the same
disease.

**Naive backpressure.** The reader waits for free space — and nothing more. The first
measurement showed zero losses, but it was wrong for two independent reasons: (1)
the test rig resolved a "tie" between two events in favor of the queue, whereas
production code checks "turn finished" first, and (2) the rig still contained the marker from the rejected
option, i.e. a hybrid was being measured. After the fix — losses = the whole tail
(50 of 50, 200 of 200), and the counter showed: 4802 frames of 5001 read, i.e. the
tail was still in the buffer when the collector finalized. Conclusion: **backpressure
is safe only as long as it never triggers**, and any technique that makes the reader
interruptible must go together with a new end-of-stream signal.

## 4. Final solution

**Backpressure on the notification channel.** `session/update` is delivered via
`await put`: when the box is full, the reader **waits for space** and stops reading the pipe.
The pressure goes into the pipe buffer and further into the CLI itself, which blocks on its own
write. Nothing is lost and no timeout is needed — the reader is woken by the box itself when
the collector frees up space.

**Control frames bypass the blocking queue.** JSON-RPC responses and the agent's
requests to the client (including the permission request) are dispatched synchronously. This is not
an improvement but the difference between "works" and "deadlock": with the turn response on
the blocking path, 1500/50 passed, while 5000/200 and 20000/200 **hung** — the reader
waits for space, the collector waits for the end of the turn.

**End-of-stream signal: "queue is empty AND the reader is idle".** The reader
sets an "idle" event when it is waiting for new bytes and holds no
unprocessed frame, and clears it for as long as a frame is in its hands —
including while waiting for space in a full box. After the response resolves, the collector drains
until **both** conditions hold. "Box is empty" alone is not the signal:
once the reader became interruptible, an empty box can mean either "the stream
has ended" or "the next frame is in the reader's hands". Both conditions are events, so
no timeout or polling is needed. The notification count cannot be taken from the protocol: the ACP
response to a turn request carries only the stop reason (`end_turn`,
`max_tokens`, `refusal`, `cancelled`, `max_turn_requests`).

**Failure paths.** EOF, a reader crash and client closure are a two-source
wait (a frame from the queue versus the turn's future), with a loud `AcpClientError` error,
and the work already done is preserved on the state before any raise. When the reader dies,
the "idle" event is forcibly set, otherwise a second, silent
way to hang would appear.

### Measurements of the final solution

Capacity 1000, permission request in the middle of the stream, long tail after the response:

| total/tail | losses | max box | reader waited | order |
|---|---|---|---|---|
| 20/2 | 0 | 20 | 0 | yes |
| 900/200 | 0 | 900 | 0 | yes |
| 1500/50 | 0 | 1000 | 451 | yes |
| 5000/200 | 0 | 1000 | 3801 | yes |
| 20000/200 | 0 | 1000 | 18801 | yes |
| **50000/500** | **0** | **1000** | 48501 | yes |

Linear accumulation: 10,000 notifications — **1.97 s → 0.0044 s (448×)**, the curve is
flat up to 20,000; draining 8000 notifications — 1.44 s → 0.01 s.

Mutations (each mechanism was removed separately, failures classified by cause):
remove backpressure → 6 failures; remove the "reader is idle" check in the
production helper → 5 failures; never clear the idle flag → 6 failures;
restore the quadratic recomputation → exactly three performance tests go red.

### A limitation you need to know about

Backpressure reaches the CLI. A collector that stopped consuming entirely
will stall the CLI's writes instead of failing the turn. This is a deliberate trade-off: a
stall is better than silent loss. The existing turn-idle safeguard
is not reused by this solution and continues to work separately.

## 5. Explanations (a–e)

**(a) What happens during a burst and why the turn is lost** — see §1, step by step.

**(b) Why the "everything has been read" signal turned out to be harmful** — it stops
the collector at the turn boundary, while there are providers that send text after the response;
the tail was silently discarded (50 of 50 versus 0 without the marker). See §3, first paragraph.

**(c) Why backpressure in the first design was wrong** — §3, second paragraph:
the "no losses" table was obtained on a rig that (1) resolved a tie
in favor of the queue, whereas production code checks the future first, and (2) contained
the marker from the rejected option. After the fix — losses equal the whole tail,
`read_at_break` 4802/5001.

**(d) Two options for the end-of-stream signal.** "Queue is empty AND the reader is
idle" — works, no timeout needed (both conditions are events), the risk is in the exact
definition of "idle": a reader waiting for bytes and a reader stuck on
a full box look alike from outside, but mean the opposite — which is why the signal is cleared
for the whole time a frame is owned. "Check against the number in the response" — **impossible**:
the protocol provides no such number, the response has only the stop reason (verified by
reading the protocol, not assumed).

**(e) Recommendation and what is needed so the turn is not lost** — the first
option is implemented, the second is ruled out by the absence of data in the protocol. Six
properties need to be proven: no overflow on a burst larger than the box; the test does not go green from merely
speeding up the collector; not a single chunk is lost (measure **delivery**, not "did not
crash"); the tail after the response is long (≥50); a control on the same rig without the fix;
and sixth — **the tail is delivered precisely when the limiter triggered**
(`blocked > 0`), otherwise the test is green in a mode where the mechanism never engaged.

## 6. References

`acp_client.py`: `_read_loop`, `_dispatch_async`, `_put_notification`,
`reader_is_idle`/`wait_reader_idle`, `_on_eof`, `aclose`.
`chat_model.py`: `_consume_turn` (two-source wait and the response branch),
`_drain_until_stream_exhausted`, `_drain_pump`, `_preserve_turn_work`,
`_BoundedTextAcc`/`_append_bounded_text`.
Tests: `test_notification_backpressure.py`, `test_text_accumulation_linear.py`,
`test_acp_client_integration.py::TestSinkOverflow` (contract rewritten: the stream is
absorbed instead of killing the turn).
