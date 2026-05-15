# DESIGN: Root-cause fix for AgentService memory bloat + polling

**Status:** ✅ IMPLEMENTED (2026-04-23)
**Date:** 2026-04-23
**Scope:** `deep_worker_web/app/services/agent_service.py`,
`deep_worker_web/app/config.py`, `deep_worker_web/run.py`, RPyC
broadcast contract on the agent side (the agent process module `agent_process.py`).

## 0. Problem

The previous session captured a live snapshot:

* A single `uvicorn` worker inside deep_worker_web —
  held **14.95 GiB RSS** (45 % of host memory) + 210 MB swap after
  ~3.5 h uptime, while doing ~0 % CPU. The host was under load 34
  and 6.9 GB swap used.
* `/proc/<pid>/smaps_rollup`: almost all of it is
  `Private_Dirty / Anonymous` — Python heap + 37 large anon `mmap`
  segments (6.8 + 2.86 + 1.4 + 0.7 + 0.67 GiB…).
* Log stats over the same window: **7 739 broadcast events**,
  7 170 of which are `Updated status for <agent>: ready` — roughly
  one per 5 s per agent in registry. Plus **48** `WARNING: Failed
  to delete remote object … Possible memory leak on remote side`.

The process was idle, yet allocator-pinned to many GB. That is the
signature of **high-churn small-allocation workload** combined with
**long-lived holders** that keep the arena from being trimmed.

## 1. Root causes

### C1 — 5-second status polling loop

`AgentService.monitor_agents()` (`app/services/agent_service.py`
~lines 1374–1463) does:

```python
while not self._shutdown_event.is_set():
    await asyncio.sleep(5)                # wake every 5 s
    current = await loop.run_in_executor(None, registry.list_all)
    …
    for name in known & current:
        await self._update_agent_status(name, root)   # per-agent RPC
```

`_update_agent_status` fires a `get_status()` via
`fire_and_forget_async` with two closures (`on_success` /
`on_error`) that capture `self`, `agent_name`, `root` — i.e. each
tick allocates an `asyncio.Task`, a couple of coroutine frames,
one or two dicts, two JSON (de)serialisations in `_broadcast_message`
(`json.dumps(default=str)` → `json.loads`), plus whatever RPyC
netref proxies come back for status.

Multiplied by ~1 tick/s over multiple agents for hours: churn +
GC-held frames + fragmented glibc arenas → the 14 GiB we observed.

This **also violates the project's own rule**: _"polling is
forbidden, an event-driven architecture must be used"_ — so it has to
go on principle, not just for memory.

### C2 — `convert_netref` does a sync RPC from the async loop

`agent_service.py` ~389–402:

```python
def convert_netref(obj):
    if isinstance(obj, dict):
        return {k: convert_netref(v) for k, v in obj.items()}
    …
```

`obj` here is a **remote netref** living on an async RPyC
connection; `obj.items()` triggers `sync_request()` from the same
event loop that serves the connection, which RPyC itself detects
and reports:

```
Error dispatching ASYNC callback to subscriber …
sync_request() was called from the asyncio loop that serves this
connection for a user-level RPC (handler=7). This would block
the loop for the full remote round-trip and can deadlock.
```

Side effects:

* the callback never finishes, so the RPyC remote reference isn't
  `__del__`ed on the agent side — hence the **48 «memory leak on
  remote side»** warnings per session;
* every failed conversion allocates an exception + traceback frames
  held by asyncio until GC;
* the broadcast the UI was supposed to get is dropped.

### C3 — `debug=True` forces `uvicorn reload=True`

`run.py:53`:

```python
uvicorn.run("app.main:app", …, reload=settings.debug, …)
```

`debug=True` is the default in `config.py:25`, so production runs
on a reloader + `multiprocessing.spawn` worker. That spawn worker
is exactly the PID that bloated. We don't actually use the reload
feature in production (it's a dev convenience), but we pay for:

* an extra subprocess whose memory is harder to observe,
* WatchFiles scanning the tree,
* per-reload re-import cost if a `.py` is touched.

We want `reload` to be an **explicit** opt-in, orthogonal to
logging verbosity. (The user's rule "any creation of new threads
requires explicit approval" extends to extra worker processes for
the same reason.)

## 2. Fixes

### F1 — replace `monitor_agents` with inotify watcher

Kernel event-driven replacement, mirroring what
`AgentLifecycleService._wait_registered` already does successfully
(in `lifecycle.py`):

1. `inotify_init1(O_NONBLOCK)` on the directory holding
   `/tmp/deep_worker_registry.json`.
2. Watch `IN_CLOSE_WRITE | IN_MOVED_TO | IN_MODIFY | IN_CREATE |
   IN_DELETE` on that dir (registry is updated via atomic
   `write+rename`).
3. `loop.add_reader(inotify_fd, _on_registry_change)`.
4. `_on_registry_change` re-reads the file, diffs against the
   in-memory `self._agents`, and emits the usual
   `agent_discovered` / `agent_terminated` events — **no periodic
   sleep, no periodic broadcasts**.

**Removing the status poll entirely is the key win.** Status is
already pushed to subscribers by the agent itself (the agent does
`set_status()` internally when it changes state and, on subscribed
sessions, the web callback `MessageCallbackService.exposed_on_message`
receives it). The 5 s get_status sweep was belt-and-braces — and
it's what ticked 7 170 times in 3.5 h while contributing zero new
information 99 % of the time.

If inotify is ever unavailable (non-Linux, FUSE…), fall back to a
**large-period** safety tick (`call_later(60s)`) that just re-scans
the registry — no per-agent RPC.

### F2 — make the broadcast payload serialisable on the agent side

Two parts, both in the agent process module `agent_process.py` where
`notify_callback` is invoked, and the RPyC service that dispatches
subscribe messages:

* When the agent calls `_notify_subscribers(msg)`, it should first
  `json.dumps(msg, default=str)` on its side, so subscribers receive
  a plain `str` over RPyC (by-value, no netref).
* Web `MessageCallbackService.exposed_on_message(msg)` then just
  `json.loads(msg)` — no recursive traversal, no `obj.items()` on
  a netref.

Effect:

* **zero sync RPCs** on the callback path → the RuntimeError
  disappears;
* no leaked remote objects → the `Failed to delete remote object`
  warnings disappear;
* the callback path stops allocating N×len(msg) one-shot netref
  objects per message.

Backward compatibility: the web side can accept both shapes for a
transition period (if it gets a `dict`, it converts using the old
path; if it gets a `str`, it `json.loads`). This lets us roll out
the agent change and the web change independently.

### F3 — decouple `debug` from `reload`

1. Add `reload: bool = False` to `Settings` in `app/config.py`
   (separate from `debug`).
2. `run.py` passes `reload=settings.reload` (defaults to `False`),
   and `log_level` derives from `settings.debug`.
3. `.env` or env var `DEEP_WORKER_WEB_RELOAD=1` opts in
   explicitly for dev.

This removes the implicit WatchFiles + spawn-worker overhead from
every production start. Not a memory fix on its own, but it
collapses the reproducer surface and removes a silent thread-like
background (WatchFiles).

### F4 — minor clean-ups that fall out of F1/F2

* drop `_message_callbacks` traversal that happens for every
  agent_status event (since there will be no per-5s status events
  anymore);
* `_broadcast_message` no longer needs to re-run
  `json.dumps/loads` in its hot path — we can do one `json.dumps`
  at the edge (WS send). We will **not** change that in this PR
  to minimise blast radius; only remove the events that cause the
  churn.

## 3. What we will NOT do in this PR

* No change to `lifecycle.py` (already event-driven).
* No change to the WS dispatcher format or request_id contract.
* No change to `exposed_shutdown` semantics.
* No switch to jemalloc or `MALLOC_ARENA_MAX` tweaks. They would
  help glibc arenas behave, but they mask the real problem (we
  just stopped the churn). We leave them as a follow-up if needed.
* No rewrite of `convert_netref` on the web side: it is kept as a
  defensive fallback for older agents that haven't been upgraded
  yet.

## 4. TDD plan

Write failing tests first, then implement.

### 4.1 Unit

1. **test_monitor_is_inotify_not_polling**
   — new `tests/unit/test_agent_service_monitor.py`. Mock
   `inotify_init1` / `inotify_add_watch`, assert `loop.add_reader`
   is called with the inotify fd, assert `asyncio.sleep(5)` is
   **never** called during normal operation.
2. **test_monitor_picks_up_new_agent_on_inotify_event**
   — simulate an inotify event (callback directly invoked),
   verify `_agents` is updated and `agent_discovered` broadcast.
3. **test_monitor_picks_up_terminated_agent_on_inotify_event**
   — same for termination.
4. **test_convert_netref_accepts_json_string**
   — new `tests/unit/test_message_callback_json.py`. Feed a
   JSON string into `MessageCallbackService.exposed_on_message`,
   assert downstream `_broadcast_message` is called with the
   parsed dict and **no** `items()`/`keys()` access on netref
   proxies.
5. **test_message_callback_falls_back_to_dict**
   — the dict path still works (backwards compatibility).
6. **test_settings_reload_defaults_false**
   — new `tests/unit/test_config_reload.py`. `Settings().reload is
   False`; with `DEEP_WORKER_WEB_RELOAD=1` env, `.reload is True`.
   Plus a test that `settings.debug is True` does **not** imply
   `settings.reload`.

### 4.2 Integration

Keep all 22 existing integration tests green (live agent, WebSocket
events, lifecycle). Add:

7. **test_no_status_polling_on_idle_server**
   — new integration. Start server, spawn an agent, then do
   nothing for 10 s. Assert the log has **zero** occurrences of
   `Updated status for … :` that were not triggered by a real
   status change. (Under F1, all status updates must come from
   the subscribe channel or from inotify events.)
8. **test_broadcast_uses_json_string_path**
   — live agent, subscribe via WS, wait for one message,
   assert the web log does **not** contain `sync_request()` or
   `Failed to delete remote object`.

### 4.3 Regression baseline

Before touching anything, record:

* current broadcast count over 60 s of idle: `grep -c` the running
  server's log ⇒ expect ~12 (1 per agent per 5s × agents).
* current RSS of the worker over 5 min: expect slow monotonic
  creep.

After the fix:

* broadcast count on idle with N=3 agents, no user activity,
  60 s: **0** (we only emit on real events).
* RSS steady-state within ±5 MB over 5 min of idle.

## 5. Commit plan

One focused commit per fix, in order, so each is trivially
revertable:

1. `fix(config): split reload from debug; default reload=False` —
   F3, no behaviour change for tests.
2. `fix(agent-service): drop 5s status polling; inotify registry
   watcher` — F1 + associated unit tests.
3. `fix(agent): broadcast subscriber messages as JSON strings
   (backwards-compatible)` — F2 agent-side + unit tests on agent.
4. `fix(web): accept JSON-string subscriber messages (keep dict
   fallback)` — F2 web-side + unit tests.

If items 3 and 4 can land together atomically, fine — a single
`fix(subscribe): serialize on agent, parse on web` commit is
acceptable and in fact preferred (there's no meaningful state where
only one side is deployed in this repo).

## 6. Acceptance criteria

* All existing unit + integration tests still pass (178 + 21).
* New unit tests 1–6 pass.
* New integration tests 7–8 pass.
* Manual re-run of the original scenario (3 test_ agents running,
  lifecycle tests from the previous PR): broadcast rate on idle =
  **0/min** (was ~12/min); RSS growth over 5 min = **<10 MB** (was
  trivially hundreds of MB in the same window). Measurement via
  `ps -o rss -p <worker_pid>` at t0 and t0+5min.
* `grep -E "sync_request\\(\\) was called from the asyncio loop|
  Failed to delete remote object" /tmp/deep_worker_web.log` =
  **empty** for any run after the fix.
* No new threads. No `asyncio.to_thread`. No new long-running
  subprocesses.
