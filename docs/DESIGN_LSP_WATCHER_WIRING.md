# Design: wiring the change-detector into the live service (§13, commit 8)

Status: accepted (CODE+DESIGN SIGN OFF), commit 8.
Context: `docs/DESIGN_LSP_SHARED_SERVICE.md` §9 (invalidation rules),
§13 (`test_add_module_refreshes_index`), `docs/POLICY_NO_INPROCESS_HEAVY_MCP.md`.

## 1. Problem

`change_detector.py` is written, covered by unit tests, and **used by
nobody**. Verified:

```
grep -rn "change_detector|ChangeDetector" --include=*.py deep_worker_cli/
  → no imports outside the module itself
grep -rn "publish_session_event|REFRESH_STARTED" --include=*.py deep_worker_cli/
  → no production calls
```

That is, in the running service **nobody watches the file system**.
`ChangeDetector.notify()` is called only from tests. Consequence: a module
added to the project is never noticed by the service, and the model will
see a stale index until a manual `RefreshProject`.

Therefore the §13 item `test_add_module_refreshes_index` **cannot** be
closed with a single test: first we need the production piece, which does
not exist.

## 2. Task boundaries

In scope:

- a watcher that translates filesystem events into `ChangeDetector.notify`
  calls;
- starting/stopping the watcher together with the service;
- publishing `SESSION_EVENT_REFRESH_STARTED/FINISHED` to `AttachSession`
  subscribers and updating `last_refresh_*` for `ProjectStatus`;
- filling `active_providers` / `watched_paths` in `ProjectStatus`
  (currently hardcoded empty with a "commit 5" note);
- the §13 end-to-end test.

Out of scope (a separate stage): attaching a real Serena via
`set_tool_handler`. Here `refresh_action` is a call to the service's
`on_refresh` hook, not a command to the language server. The hook is the
single point where the next stage will plug in the real
`serena.refresh_project`.

## 3. Solution

A new module `deep_worker_cli/lsp/service/watcher.py` — inside the
allowlisted package `lsp.service*` (POLICY §8).

```
ProjectWatcher(root, detector)
    async def start()   # creates an asyncio task with watchfiles.awatch
    async def stop()    # ends awatch via stop_event and awaits the task
    @property running        # bool
    @property watched_paths  # list[Path] — for ProjectStatus
    @property active_providers  # list[str] — for ProjectStatus
```

`on_refresh` does not pass through the watcher: the detector holds the
hook passed into its constructor directly, and the server uses
`_trigger_refresh` as the single entry point from both the detector and
`RefreshProject`. This rules out duplicated decision policy.

Mechanics:

- `watchfiles.awatch(root, stop_event=…)` — an async generator,
  inotify inside. No polling: the task blocks on `__anext__` until a
  real kernel event.
- Each batch of events → `detector.notify(paths)`. Debounce and the
  structural comparison live in the detector and are **not duplicated**
  here: the watcher is deliberately dumb; all the policy is in the
  §9 table.
- Stopping via `stop_event` (anyio-`Event`-compatible), not via a
  mid-iteration cancellation: `awatch` can terminate itself, which is
  cleaner and leaves no dangling inotify fd.

Filtering. `awatch` pulls junk by default (`.git/objects`,
`__pycache__`, `node_modules`, `.venv`, `build`, `target`). We provide a
`watch_filter` that cuts these directories out — otherwise a single
`git status` on a large repository spawns thousands of events and we pay
for them with wakeups. This is not "economy for economy's sake": §9
explicitly requires frugal updating.

## 4. Where it plugs in

`LSPServiceServer` gets an optional watcher, because the server already
owns the lifecycle (`start`/`stop`) and event broadcasting to sessions.

- `ServerConfig.watch_enabled: bool = True` — disableable in unit tests
  where filesystem events are not needed.
- `LSPServiceServer.start()` brings up `ChangeDetector` + `ProjectWatcher`.
- `LSPServiceServer.stop()` stops them **before** stopping gRPC, so no
  task is left writing to closed session queues.
- The server's `on_refresh(reason)`:
  1. `publish_session_event(REFRESH_STARTED)`;
  2. call `self._refresh_backend()` — currently a no-op hook; in the
     next stage Serena will sit there;
  3. `note_refresh(reason)`;
  4. `publish_session_event(REFRESH_FINISHED)`.

The order matters: `note_refresh` sits between START and FINISH so that a
client that saw FINISHED is guaranteed to read the already-updated
`last_refresh_unix` through `ProjectStatus`.

## 5. What we deliberately do NOT do

- We do **not** recompute the snapshot in the watcher. Otherwise two
  places would make the refresh decision, and the §9 table would cease
  to be the single source of truth.
- We do **not** add a thread. The watcher is an asyncio task on the same
  loop (policy: no new threads without explicit agreement).
- We do **not** fall back to polling if inotify is unavailable.
  `watchfiles` supports `force_polling`, and that is exactly what the
  policy forbids. No inotify → the service logs and works without
  auto-refresh; the manual `RefreshProject` remains. Silently degrading
  into polling is not allowed.
- We do **not** store the snapshot in `state.json`. §9 mentions it, but
  `state.json` belongs to the registry (pid/socket) and is written by
  the supervisor from another process. The snapshot lives in the
  detector's memory; after a service restart it is honestly recomputed
  from disk. The divergence from §9 is recorded here deliberately —
  otherwise two processes write to one file without a lock.

## 6. TDD

RED before implementation:

- `test_watcher_feeds_detector_on_manifest_change` — a module created +
  a manifest edit → `on_refresh` is called exactly once.
- `test_watcher_ignores_edit_inside_existing_source` — an edit to the
  body of a `.py` → `on_refresh` is NOT called.
- `test_watcher_stops_cleanly` — after `stop()` events do not get through.
- `test_watcher_filters_noise_dirs` — a write into `.git/`/`__pycache__`
  does not wake the detector.
- `test_server_publishes_refresh_events_to_attached_session` —
  START/FINISH arrive on the `AttachSession` stream.
- `test_status_reports_active_providers_and_watched_paths`.

E2E (§13, on a real service process):

- `test_add_module_refreshes_index` — bring up the service, attach,
  add a module, wait for REFRESH_FINISHED **by event** (not by sleep),
  check that `last_refresh_unix` grows.
- `test_edit_inside_module_does_not_refresh` — the flip side; without it
  the first test passes even on "refresh at any sneeze".

Budget ≤60 s per file, a `reaper` with SIGTERM→SIGKILL, and a check for
zero orphaned processes.

## 7. Q1 — in-memory snapshot (decided at review)

The reviewer confirmed option **(a)**: the snapshot lives only in the
detector's memory; after a service restart it is recomputed from disk;
an optional `snapshot.json` is deferred until a proven need. The
amendment to §9 of the main design is the short line "the snapshot lives
in the detector's memory; state.json remains the registry's property".

## 8. Single-flight refresh (non-blocker #1 from the review)

`_trigger_refresh` schedules `_run_refresh` as a separate task, which
previously allowed two refresh cycles to run in parallel: `STARTED,
STARTED, FINISHED, FINISHED` on the stream and — more importantly — a
`refresh_project` race in the language server against itself once stage 2
connects it.

Fix: `LSPServiceServer._refresh_lock: asyncio.Lock`. `_run_refresh`
takes the lock for the whole cycle (STARTED → backend hook → note_refresh
→ FINISHED). Triggers flying in a row line up in a queue; each
STARTED/FINISHED pair is intact. The test
`test_refresh_is_single_flight` drives three triggers with a slow backend
hook, measures the peak parallelism (must be 1), and checks the strict
START/FINISH alternation of events on the stream. The MUT-F mutation
(remove `async with self._refresh_lock`) is caught by the test
precisely.
