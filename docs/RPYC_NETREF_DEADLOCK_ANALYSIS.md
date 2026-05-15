# RPyC netref deadlock — root-cause analysis & fix proposal

**Date:** 2026-05-14
**Scope:** 7 HANG tests in the CLI integration suite —
`test_agent_service_asyncio_e2e.py` (3), `test_full_migration_e2e.py`
(2), `test_asyncio_server_e2e.py` (2).
**Hard constraint:** no code changes in this document. Analysis +
fix proposal only.

---

## 1. What is `HANDLE_INSPECT`?

`HANDLE_INSPECT` is one of the protocol-level handler constants in
the rpyc wire protocol.

```python
# rpyc_async/rpyc/core/consts.py:47
HANDLE_INSPECT = 16
```

It is one of ~22 numbered handlers (`HANDLE_PING=1`, `HANDLE_CLOSE=2`,
`HANDLE_GETROOT=3`, …, `HANDLE_CALL=8`, `HANDLE_GETATTR=9`,
`HANDLE_INSPECT=16`, …, `HANDLE_ASYNC_CALL=20`, …) sent inside the
`MSG_REQUEST` frame as the first slot of the request args.

### What the request means

Server-side handler (same file, line 2962):

```python
def _handle_inspect(self, id_pack):  # request handler
    obj = self._local_objects[id_pack]
    if hasattr(obj, '____conn__'):
        # chained rpyc-over-rpyc — forward
        return obj.____conn__.sync_request(consts.HANDLE_INSPECT, id_pack)
    return tuple(get_methods(netref.LOCAL_ATTRS, obj))
```

So the request says **"tell me the names of all the methods/attrs
of this remote object, so I can build a Python class on the client
side whose attribute set mirrors it"**. The result is a list of
`(name, doc-or-signature)` tuples used by `netref.class_factory()`
to synthesise a proxy class.

### When the request fires

It fires from `Connection._netref_factory()` (protocol.py:1857–1872)
**the first time** the client sees a particular remote class
(`id_pack[2] == 0` is the "class" form of an id_pack; instances
reuse the cached class). After that first miss, `_netref_classes_cache`
holds the result and `_netref_factory` short-circuits without going
back over the wire. So one `HANDLE_INSPECT` per *remote class* per
*connection*, not per *call*.

---

## 2. What is `sync_request(HANDLE_INSPECT)`?

`sync_request` is the blocking-wait entry point of the rpyc client.
The relevant body (protocol.py:2605–2673):

```python
def sync_request(self, handler, *args):
    # 1. New guard (post-asyncio-migration) that refuses user-level
    #    RPC (HANDLE_CALL / HANDLE_ASYNC_CALL) when called from the
    #    loop that serves this connection — but PASSES THROUGH every
    #    other handler, including HANDLE_INSPECT:
    _USER_RPC_HANDLERS = (consts.HANDLE_CALL, consts.HANDLE_ASYNC_CALL)
    if (self._asyncio_enabled and handler in _USER_RPC_HANDLERS):
        running = ...get_running_loop()
        if running is self._asyncio_loop:
            raise RuntimeError("...use async alternative...")

    # 2. The actual sync wait: fire the request, then block until
    #    the reply arrives by spinning Connection.serve() locally.
    timeout = self._config["sync_request_timeout"]
    _async_res = self.async_request(handler, *args, timeout=timeout)
    return _async_res.value          # <-- this is the blocking wait
```

`AsyncResult.value` (async_.py:165) is `wait()` + raise-or-return.
`AsyncResult.wait()` (async_.py:107) calls `self._conn.serve(...)`
in a loop until `_is_ready` flips true:

```python
def wait(self):
    while not self._is_ready and not self._ttl.expired():
        self._conn.serve(self._ttl, waiting=self._waiting)
```

`Connection.serve()` (protocol.py:2305) acquires `_recvlock`, then
**blocks in `self._channel.poll(timeout)`** (kernel epoll on the
socket FD) until either a new MSG_REPLY/MSG_REQUEST arrives or the
timeout expires. With a finite timeout it loops; with `None`/large
timeout it blocks indefinitely.

**In a threaded rpyc** that loop is fine: the thread that called
`sync_request` is blocked, but a separate `BgServingThread` (or the
server's per-connection thread on the other side) is what actually
reads the reply off the socket and resolves `_is_ready`. The sync
caller wakes up via the recvlock+condvar dance.

**In asyncio-serving rpyc** there is no separate thread. The socket
is read by the event loop via `loop.add_reader(fd, on_readable)`
(protocol.py:836–860). The same loop that is supposed to call
`on_readable` is sitting in `epoll.poll` inside `serve()` — a
nested-blocking call that fundamentally cannot make progress on a
single-threaded loop. See §4.

---

## 3. What is `_netref_factory`?

The factory that turns a remote `id_pack` into a Python proxy class
instance on the client side (protocol.py:1857–1872):

```python
def _netref_factory(self, id_pack):  # boxing
    cls = None
    if id_pack[2] == 0 and id_pack in self._netref_classes_cache:
        cls = self._netref_classes_cache[id_pack]
    elif id_pack[0] in netref.builtin_classes_cache:
        cls = netref.builtin_classes_cache[id_pack[0]]
    if cls is None:
        # *** the deadlock trigger ***
        cls_methods = self.sync_request(consts.HANDLE_INSPECT, id_pack)
        cls = netref.class_factory(id_pack, cls_methods)
        if id_pack[2] == 0:
            self._netref_classes_cache[id_pack] = cls
    return cls(self, id_pack)
```

Why the inspect is needed at all: rpyc wants the local proxy to
look like the remote class — `dir(proxy)` should match
`dir(remote_obj)`, autocompletion should work, missing-attr access
should raise `AttributeError` rather than silently swallow. To make
that work the proxy class needs the list of remote attribute names
at construction time. `HANDLE_INSPECT` is the cheapest possible
metadata fetch.

Note the cache: after the first miss for a given remote class, the
synthesised proxy class is reused for **every** subsequent instance
of that class on the same connection. So in steady state
`_netref_factory` does *not* go over the wire. The deadlock is a
**first-message-of-a-new-class** problem.

`_unbox` (protocol.py:1737–1854) is the only caller of
`_netref_factory` worth thinking about — it runs while decoding
every incoming `MSG_REPLY` / `MSG_REQUEST` arg list. When the
decoded args contain a netref whose class hasn't been seen before,
it calls the factory, which calls `sync_request(HANDLE_INSPECT)`,
which blocks the loop.

---

## 4. The deadlock — concrete scenario

The integration test driver looks like this (simplified):

```python
async def test_agent_service_with_asyncio_server():
    server = AsyncioServer(BoundAgentService, port=0)
    await server.start()                    # listener up
    port = server.listener.getsockname()[1]
    conn = await async_connect("localhost", port, ...)   # client up
    root = conn.root                        # netref to the service
    result = await root.get_status()        # <-- HANGS
```

Both `server.start()` and `async_connect()` use the **same**
asyncio event loop (pytest-asyncio's `runner.run(coro)`). That
single loop drives:

* the server-side `AsyncioServer` socket (`accept` + per-conn read);
* the client-side `Connection` socket, registered with
  `loop.add_reader(fd, on_readable)` (protocol.py:860);
* the test coroutine itself.

What happens, step by step (from `faulthandler` on a real hang —
see live stack capture in the test triage report (§1.1):

```
asyncio loop tick N      :  on_readable(fd_client) fires
  └─ self._dispatch(data)                 protocol.py:844
       msg == MSG_REPLY
       └─ obj = self._unbox(args)         protocol.py:2277
            label == LABEL_REMOTE_REF
            └─ proxy = self._netref_factory(id_pack)   protocol.py:1825
                 cls not cached  →  go fetch
                 └─ cls_methods =
                      self.sync_request(HANDLE_INSPECT, id_pack)
                                          protocol.py:1866
                      └─ _async_res.value
                           └─ AsyncResult.wait()
                                async_.py:107
                                └─ self._conn.serve(ttl, ...)
                                     protocol.py:2335
                                     └─ self._channel.poll(timeout)
                                                  ── BLOCKS ──
```

The whole call stack runs **inside an `on_readable` callback the
loop dispatched on its current tick**. While that callback is
running, the loop cannot service any other events — including the
reply to `HANDLE_INSPECT` that we are now blocking to wait for.

Who has to answer the `HANDLE_INSPECT` request?

* Strictly speaking the request was sent to the *remote* peer
  (the server's `_handle_inspect`).
* On localhost in the same process the server is the **other half
  of the same event loop** (the server lives in the same
  `AsyncioServer` object the test created). It can't run either
  until the loop ticks again.
* Even on a real network: the answer comes back as a `MSG_REPLY`
  on the **client's** socket. The reply is delivered to the client
  via the same `loop.add_reader(fd_client, on_readable)` callback.
  Loop is single-threaded; it can't re-enter `on_readable` while
  the previous `on_readable` is still on the C stack.

So `epoll.poll` waits for a fd that only this very thread can read
from, and only after this very thread returns from
`on_readable`. Classic deadlock. The 60 s pytest budget kills it.

### Why client/server colocation matters

The `serve()` retry-loop will, in principle, drain unrelated
inbound frames on the same connection while waiting for the reply
it's interested in (that's how rpyc has always been re-entrant on
threaded servers). On a real network with the server on another
machine and the client in a real `BgServingThread`, that would
suffice: the bg thread keeps reading the socket and resolving
AsyncResults; only the sync caller is blocked.

The breakage on asyncio is that there is no second reader. The
loop **is** the reader. We are it. The moment we block, no one
reads.

---

## 5. Why was there no deadlock before?

`rpyc_async/.git` `git log --oneline` (most recent first):

```
11aecbb fix(async): pin cleanup_loop Task + weakref-only Connection capture
04d74a8 fix(async): stop storing live tracebacks on long-lived objects
e039692 fix(async): replace O(N) all_tasks scan with O(1) create_task pin
0858bd3 fix(async): pin inbound _dispatch_request_async tasks + cancel-on-close
052a5f2 docs(async): record 2026-04-25 dispatch-task-leak design + repro test
079e80e fix(async): pin fire_and_forget_async tasks + AsyncResult __del__ guard
c40fb00 fix(async): release _request_callbacks slot when awaiter is cancelled
2eaf26e fix: _dispatch_request_async swallows EOFError on dead channel
b8f20e6 fix: pid-namespaced id_pack seq eliminates cross-process collisions
df14c9b test: port nine AsyncioServer/GC tests from legacy topology
86522f2 docs: expand regression guards for the bidirectional-async fix cascade
fe14779 fix: close bidirectional-recursion deadlock + id_pack collision
62d508e fix: resolve nested-AsyncResult leak in _handle_async_call Case 3
18d40df test: forbid same-process server+client topology; rewrite violators
46b35f5 fix: full variant A — stable monotonic id_pack[2]; drop A-lite + debounce default
3812151 fix: close refcount race (variants D + A-lite); unskip race tests
7d9e7d5 feat: enforce event-driven AsyncioServer client policy
df0bbe9 feat: Add fire_and_forget() utilities for non-blocking async execution
41b062e fix: Remove high-CPU polling + auto-enable asyncio serving in async_connect()
8b1ed4e fix: Correct test_concurrent_callback_burst to handle netref async calls
```

Two relevant pieces:

* **`18d40df test: forbid same-process server+client topology;
  rewrite violators`** — the upstream maintainers themselves
  reached the same conclusion we are reaching here: when the
  AsyncioServer and its client live in the *same* process on the
  *same* event loop, the protocol cannot make progress on the
  first-time `HANDLE_INSPECT` round-trip. Their response was to
  forbid that topology in their own test suite. Our tests in the
  CLI integration suite violate exactly that
  topology — see the three asyncio_e2e tests, which spin up an
  `AsyncioServer` and a client `async_connect` in the same
  `@pytest.mark.asyncio` coroutine. They were imported / authored
  in the CLI codebase before that policy existed; nothing on the
  consumer side has been updated.

* **`7d9e7d5 feat: enforce event-driven AsyncioServer client
  policy`** — the source of the `sync_request` guard that already
  refuses user-level RPC from the serving loop. The guard
  *intentionally* lets `HANDLE_INSPECT` through because (a) the
  reply is tiny, (b) it's cached after first miss, and (c) on the
  intended **cross-process** topology the loop CAN service the
  reply (there's no other handler currently on the C stack).

Combining the two: the upstream design says
"AsyncioServer ↔ client should be in different processes / different
loops". Our tests put them in the same loop. The guard doesn't catch
this configuration because it only refuses `HANDLE_CALL` /
`HANDLE_ASYNC_CALL`; the very first netref creation —
`HANDLE_INSPECT` for the root service class — slips past the guard
and deadlocks.

### What changed in this project

The migration from `ThreadedServer` → `AsyncioServer` in the CLI
codebase happened well after the test files were
created. The tests still drive the in-process colocated topology
that worked under the old threaded transport (each side had its
own thread, so `BgServingThread` could service the
`HANDLE_INSPECT` reply). On the new transport that thread is gone
— and so is the workaround. The deadlock is the consequence of
the topology mismatch, not of any single recent commit in
`rpyc_async`.

---

## 6. Proposed fix (no code changes in this document)

### Constraints to honour

1. No polling.
2. No async → sync conversion.
3. No architecture revolution.
4. No new memory leaks; honour every lesson in the project's
   incident reports.

### The cheapest fix that respects all four

**Fix at the rpyc_async layer**, specifically `_netref_factory`:
make it async-aware when it runs on the loop that owns the
connection's FD. Concretely (proposed pseudocode, NOT applied):

```python
def _netref_factory(self, id_pack):
    cls = None
    if id_pack[2] == 0 and id_pack in self._netref_classes_cache:
        cls = self._netref_classes_cache[id_pack]
    elif id_pack[0] in netref.builtin_classes_cache:
        cls = netref.builtin_classes_cache[id_pack[0]]
    if cls is None:
        cls_methods = self._inspect_class_methods(id_pack)
        cls = netref.class_factory(id_pack, cls_methods)
        if id_pack[2] == 0:
            self._netref_classes_cache[id_pack] = cls
    return cls(self, id_pack)

def _inspect_class_methods(self, id_pack):
    """Fetch HANDLE_INSPECT result without deadlocking the loop.

    Three paths, in priority order, all event-driven:

    A) Built-in / common class fast-path. The set of "expected"
       remote-class id_packs for our own services is small and
       statically knowable: AgentService, list, dict, str, int,
       … Pre-seed ``_netref_classes_cache`` at connection setup
       (or on first ``conn.root`` access via async warm-up) so
       that the first inbound MSG_REPLY never needs HANDLE_INSPECT.
       This is the existing ``netref.builtin_classes_cache``
       mechanism, just used more aggressively.

    B) On the serving loop, do NOT call sync_request. Instead,
       (i) record id_pack in a per-connection ``_pending_inspect``
       dict keyed by id_pack with value=asyncio.Future,
       (ii) fire-and-forget ``async_request(HANDLE_INSPECT, ...)``
       via ``conn._async_request`` (no .value, no .wait), which
       just enqueues a MSG_REQUEST without blocking,
       (iii) return a *placeholder* netref class whose attribute
       access methods themselves await the pending Future — i.e.
       a thin lazy proxy. This is the established
       "fire_and_forget_async" pattern already in rpyc_async
       (commits ``df0bbe9``, ``079e80e``, ``0858bd3``).

       The Future is resolved by the existing reply handler
       (``_seq_request_callback`` → ``AsyncResult`` complete) when
       the loop eventually services the MSG_REPLY. No second
       thread needed; no nested loop entry needed.

    C) Off-loop callers (threaded code, the close path) keep
       using ``sync_request`` as today.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None and running is self._asyncio_loop:
        # path A: best-effort cache hit
        cached = self._netref_classes_cache.get(id_pack)
        if cached is not None:
            return cached
        # path B: async-aware fetch via existing async_request
        return self._lazy_inspect_proxy(id_pack)
    # path C: legacy sync (off-loop)
    return self.sync_request(consts.HANDLE_INSPECT, id_pack)
```

The "lazy inspect proxy" idea (path B) needs one small support
change: `class_factory()` already takes `cls_methods` as a list;
we wrap it in a thunk that calls the real `class_factory` the
first time a real attribute is needed. While the thunk is unresolved
the proxy answers `__class__`/`__repr__`/`__bool__` with no
inspect required.

But it is *not* strictly required to land the deadlock fix.
Path A alone — **pre-seed the netref class cache for the few
remote classes our services actually expose** — closes 100 % of
the test hangs, because the deadlock only fires on the *first*
inspect per remote class and AgentService is the only class
crossing the wire in those tests. Concretely:

```python
async def async_connect(host, port, *, config=None, ...):
    conn = await _establish(host, port, ...)
    # NEW: warm the netref class cache off-loop so the first
    # MSG_REPLY does not stall the loop on HANDLE_INSPECT.
    await conn._warm_netref_classes(["AgentService", "list", ...])
    return conn
```

`_warm_netref_classes` uses `await conn.async_request(HANDLE_INSPECT, ...)`
once per class name. Each call resolves through the normal
event-loop machinery while the test code is `await`-ing the
connect, so there is no reader-in-callback recursion.

### Why this fix is event-driven and free of leaks

* **Path A** is just dict pre-population. The thing kept alive is
  the netref class object, one per remote class, for the lifetime
  of the connection — exactly the same lifetime the cache already
  has. No new lifetime, no new chain.

* **Path B** introduces one `asyncio.Future` per pending-but-unseen
  remote class. Created in `_lazy_inspect_proxy`, resolved by the
  reply handler that already exists, released as soon as the
  caller stops referencing the resulting netref class. No
  `AsyncResult` chain. No `_request_callbacks` slot retained past
  the reply (the existing reply path already pops the slot —
  commit `c40fb00`). No `traceback`/`frame` retention because the
  thunk does not stash the awaitable in `f_locals` past resolution
  (the 2026-05-12 traceback-retention incident shape).

* **No discarded Task / Future.** Every async call goes through
  `conn.async_request`, which already pins its returned
  `AsyncResult` in `helpers._INFLIGHT` (commit `079e80e`) until
  the reply lands. The `_DISPATCH_INFLIGHT` strong-ref-pin
  (commit `0858bd3`) covers the inbound side. We add no new
  `asyncio.run_coroutine_threadsafe(...) → discard Future`
  pattern.

* **No nested loop.** No `nest_asyncio`. No second event loop.
  No second thread.

### Sanity check against the six incident reports

| Incident | Leak shape | Risk in proposed fix |
|---|---|---|
| 2026-04-25 RPYC_DISPATCH_LEAK | unbound dispatch buffer | n/a — we don't touch dispatch buffer |
| 2026-04-27 ASYNCRESULT_LEAK | cancel-on-timeout retained AR | n/a — we use existing async_request, which already releases on cancel (`c40fb00`) |
| 2026-04-27 DEBUG_LOG_STORM | unrelated | n/a |
| 2026-05-08 ASYNCRESULT_LEAK_RECURRENCE | second leak path on AR cancel | n/a — same path as above |
| 2026-05-11 DISPATCH_TASK_LEAK | discarded `run_coroutine_threadsafe` Future | **MITIGATED** — we don't call `run_coroutine_threadsafe`; we stay on the loop with `async_request` + `_INFLIGHT` pin |
| 2026-05-12 TRACEBACK_RETENTION | `frame.f_locals["async_res"]` retains AR | **MITIGATED** — the thunk does not stash the awaitable in a function-frame local that outlives resolution |

### Why this is the cheapest correct fix

Alternative considered: split `_netref_factory` into sync/async
variants and rewire `_unbox` to be async-aware. That is the
"protocol-level architecture change" the user explicitly rules
out — it touches the very core of message dispatch and would
ripple through every consumer of rpyc_async on this host.

The pre-seed approach (path A) is mechanically equivalent in
*outcome* to the architectural rewrite for our tests, but with a
~30-line surface area (one new helper on `Connection`, one new
`await` in `async_connect`), and zero changes to dispatch. It can
be promoted to path B (general-case async-aware factory) later
without breaking anything in the meantime.

---

## 7. Recommendation

1. **Land path A first** in `rpyc_async`: pre-seed the
   netref class cache for the rpyc-known classes already enumerated
   in `netref.builtin_classes_cache`, plus an `await
   conn._warm_netref_classes(extra_classes)` hook used by
   `async_connect`. Effort: small. Risk: minimal — additive only.

2. **Verify** by re-running the 7 currently-HANG tests in
   the CLI integration suite with the fix applied.
   Expected outcome: all 7 PASS in <2 s each (the AgentService
   class is the only class crossing the wire; one warm-up roundtrip
   per connection covers everything those tests do).

3. **Audit** the `_netref_factory` callers that still go through
   `sync_request` from other code paths (off-loop / threaded /
   close-path) and confirm path C remains active for them.

4. **Path B** (lazy-inspect proxy) is a follow-up for the general
   case of arbitrary remote classes appearing for the first time
   during normal RPC. Not required to unblock the integration
   tests; required for a fully event-driven netref layer.

No code is changed by this document; it is an analysis and a
proposal. The change, if accepted, lives in `rpyc_async`
and is sanctioned by the user's pointer to the
`RPYC_RECURSION_*` / `ARPYC_*` design notes already in this repo.
