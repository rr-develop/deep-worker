# FIX PLAN — blocking netref on the event loop (backlog item)

**Status:** plan only, NOT implemented.
**Date:** 2026-08-29. Written after the **third** failure (2026-07-10, 2026-08-27, 2026-08-29).
**Defect analysis:** a related internal incident analysis (not included here) and the corresponding internal backlog item.

This document does not answer the question "what is broken" (the incident
analysis does that). It answers **"which changes close this, and what proves
that it is closed"**.

---

## 0. Main conclusion, which changes the direction of the backlog item

The backlog item proposes passing a factory (`Callable[[], Awaitable]`) instead
of a value. **A factory is not enough, and this must be recorded before work
starts.**

A factory removes the computation from the argument position — the blocking
expression moves inside the task. But the task runs **on the same event loop**,
and `netref.__getattribute__` is blocking by nature:

```python
# rpyc_async/core/netref.py:199,204 — both branches are unconditionally synchronous
return syncreq(self, consts.HANDLE_GETATTR, name)
```

`syncreq` → `conn.sync_request` → `async_.wait` → `protocol.serve` → `stream.read`.
Inside the task this stack yields control to the loop no more than it does
outside. Result: `loop.create_task(lambda: root.get_last_response())`
**reproduces the same failure**, just one line lower. The only difference is that
`asyncio.wait_for` is now formally "attached" — but it cannot fire, because the
timer lives on the stalled loop.

**Consequence:** the fix must replace the request mechanism itself with a
non-blocking one, not move the place where it is called. The factory is needed,
but as part of solution #2, not instead of it.

The good news: the fork **already has** a non-blocking primitive, so nothing has
to be designed from scratch.

* `asyncreq(proxy, handler, *args)` (`netref.py:67`) → `conn.async_request(...)`
  → returns an `AsyncResult`.
* `AsyncResult.__await__` (`async_.py:175`) is implemented and yields control to
  the loop; it requires asyncio serving to be enabled on the connection. The web
  does **not call it explicitly** and should not: connections are brought up via
  `async_connect()` (`agent_service.py:1027`), which auto-enables serving and
  eagerly fetches `conn.root`. A separate `enable_asyncio_serving(...)` after
  `async_connect` is an outdated pattern, and the CLI has a test forbidding it
  (`test_interactive_uses_async_connect.py:77`). That is, the precondition for
  `AsyncResult` on web connections is **already met** and requires no changes.
* `AsyncResult` already has regression tests for cancellation / slot leaks
  (`tests/test_asyncresult_cancel_leak.py`), so this path is battle-tested, not
  dormant.

Exactly one thing is missing: **a public non-blocking getattr+call form for
netref**. That is what must be added.

---

## 1. Scope

Affected files (by grep on 2026-08-29):

| File | Call sites | Role in the fix |
|---|---|---|
| `rpyc_async/utils/helpers.py` | signature | step 2 |
| `rpyc_async/core/netref.py` | — | step 1 (new helper) |
| web service `app/services/agent_service.py` | 13 | step 3 |
| web service `app/routers/agents.py` | 1 | step 3 |
| web service `app/services/mcp_service.py` | 1 | step 3 |
| CLI `.../process/agent_process.py` | 2 | step 3 |
| CLI `.../rpyc_service/agent_service.py` | 1 | step 3 |

Plus the fork's tests: `tests/test_fire_and_forget*.py` (15 calls) —
they pin the OLD signature and must be migrated, not "left as is".

The two web trees (master and dev) are byte-for-byte identical (`diff -q`,
rechecked 2026-08-29) — the fix is made in dev and goes to master as a
fast-forward.

---

## 2. Steps

### Step 1 — non-blocking netref call (the core of the fix)

Add a public helper to `rpyc_async/core/netref.py` that **does not touch**
`__getattribute__`:

```python
def async_call(proxy, name, *args, **kwargs):
    """Resolve ``proxy.name`` and call it WITHOUT blocking the caller's loop.

    Both round-trips (HANDLE_GETATTR, then the call) go through
    ``asyncreq``/``AsyncResult``, so neither ever runs ``stream.read``
    on the event loop. Returns an awaitable.
    """
```

Requirement: **both** points — attribute resolution and the call itself — go
through `asyncreq`. If the attribute is resolved via `syncreq` and only the call
goes through `asyncreq`, the defect remains: the stacks from 2026-08-27 and
2026-08-29 showed the stall exactly at `HANDLE_GETATTR`, i.e. at the FIRST of the
two points.

Why a helper rather than a change to `netref.__getattribute__`:
`__getattribute__` is synchronous by Python's contract; an awaitable cannot be
returned from it without breaking all the other synchronous code in the fork.
The helper is the option with the smallest blast radius.

### Step 2 — close the trap in the `fire_and_forget_async` signature

`helpers.py:633` accepts `awaitable: Awaitable[T]` — a value. As long as the
signature stays like that, the trap reproduces in the next call site, no matter
how many existing ones we fix (over two incidents, rule #5 in the docstring
helped not once).

* accept `Callable[[], Awaitable[T]]`;
* **forbid** the old form rather than silently deprecating it: an `Awaitable` in
  the first-argument position raises `TypeError` with a message explaining
  exactly this mistake (ideally pointing to this document);
* the docstring example in `helpers.py` currently ITSELF demonstrates the
  defective form (`conn.root.long_running_task()`) — rewrite it, otherwise it
  remains a source of recurrence.

### Step 3 — migrate the 18 call sites

13 in `agent_service.py` (lines 1123, 1230, 1287, 1386, 1459, 1610, 1669, 1705,
1763, 1834, 1918, 1992, 2099) + 5 outside it (see the table). Each one is
mechanically switched to `async_call` from step 1 inside a factory from step 2.
Line 738 is a docstring mention, not a call.

### Step 4 — guard against recurrence

A rule in a docstring is not enforceable and failed twice. An AST guard is needed
in the tests: a netref access (`root.<attr>`, `conn.root.<attr>`) in the argument
position of any async wrapper = test failure.

Beware of test theatre: the guard must catch the **shape of the expression in
the AST**, not a variable name by string match. Compare the lesson from an
earlier internal note on turn-ending tools — an AST guard against a test that
checks itself.

---

## 3. How we prove it is fixed

Restarting the process is not a disposition (this is recorded in the backlog item
and confirmed three times). The acceptance criterion is a **reproducing test that
fails on the current code**.

The key to it is the trigger observed on 2026-08-29 and absent from the previous
two records. The failure was preceded (by 43 s) by a `CancelledError` from
`exposed_invoke_agent` on the AGENT side (`asyncio.shield(invoke_task)`,
`rpyc_service/agent_service.py:529`). That is, the stall is caused **not only by a
dead peer** (the 2026-07-10 hypothesis about a dropped tab), but by any live peer
that accepted a request and did not respond.

So the defect can be reproduced in a controlled way, without killing processes:

1. start a test agent;
2. cancel an in-flight task in it so that the request is left without a response;
3. hit `GET /api/agents/{name}/last-response`;
4. check that the web **responds to another, independent** request within budget.

On the current code, step 4 must fail. The test is bounded, ≤60 s, event-driven,
with a guaranteed `SIGKILL` in `finally` (otherwise it becomes a hang itself —
see the internal note on bounded integration tests).

Additionally — a cheap runtime detector found on 2026-08-29: a stalled process
has a **non-empty `Recv-Q`** on port 7887 (9 was observed) — the accept queue
builds up because the loop is not spinning. It is checked with one command,
without sudo:

```
ss -ltn | grep 7887
```

This is more visible and cheaper than `rss.log` with `state=R` (the detector from
the backlog item, which was built but never read by anyone).

---

## 4. Order and risks

The order is mandatory: **1 → 2 → 3 → 4**. Step 3 without step 1 is useless (see
§0); step 2 without step 3 breaks 18 places at once.

The blast radius of step 2 is the entire signature change in `rpyc_async`, which is
shared by all agents and the web. Verification is done on the target tree and
includes tests with subprocesses; `PYTHONPATH` as a way to "slip in" the fork is an
anti-pattern (see the internal note on verifying on the target tree).

Separately: step 2 changes the public signature of a helper in the fork — the
change goes through the review gate as a code feature, and reviewer sign-off is
required before commit.

---

## Amendment to step 2 (2026-08-29, after the owner's review)

Step 2 above proposed **changing the signature** of `fire_and_forget_async` and
forbidding the awaitable form. That was done on the first pass — and it was a
mistake, which the owner rightly rejected.

`rpyc-async` is a **public library on PyPI** (1.0.0). `fire_and_forget_async` is
its public API. Changing the signature breaks any external code that knows nothing
about deep_worker's problem and has no need to. An internal defect of one consumer
is no reason to break the contract for everyone.

**How to do it correctly (and how it was done):**

* `fire_and_forget_async` is **untouched**: signature, behavior and acceptance of
  an awaitable are preserved. The diff against 1.0 contains not a single changed
  line of code — only the docstring.
* The hazard is **documented** on the function itself: a `.. warning::` section
  explains that `timeout` does not cover the creation of the awaitable, why a
  netref in the argument position blocks the loop, and where to go instead.
* A **new** function `fire_and_forget_call` (`versionadded 1.1`) with a factory
  signature is added next to it. It is additive: old code keeps working, new code
  gets the safe form.
* deep_worker was migrated to `fire_and_forget_call` (all 18 call sites).

**Rule for the future:** everything in the `rpyc_async` library is published
externally. A defect found from the deep_worker side is fixed in the library by
**addition**, and a breaking change only by a separate explicit decision of the
owner, with a version and migration notes.
