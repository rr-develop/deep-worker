# DESIGN — Serena pidfd-monitor + auto-restart

Series A, commit 10. Continuation of
`DESIGN_LSP_SERENA_BACKEND_ATTACH.md` §7 ("Deferred to the next
commit").

Branch: `dev` (the development checkout) → ff into `master`
(the main repository) — per the general policy.

---

## §1. Goal

The service must **itself** recover from Serena's death, not only
declare `Health.alive=False` (already done in commit 9).

Boundaries:

- Auto-restart applies only to the **service's spawned backend**
  (the single `SerenaBackend` instance inside `LSPServiceServer`).
- Limit **≤3 restarts per 60 s sliding window** (design §7).
  After that the service stays with `alive=False`, tool RPCs answer
  `is_error=True`. Fail-loud, not an endless crash-loop.
- No implicit policy changes: **event-driven**, no polling;
  no new thread (only an asyncio task next to the existing ones).

What is NOT in scope:

- Auto-restart of other backends (there are none).
- Changes to the MCP protocol, the gRPC protocol, or session event
  formats. All existing tests remain valid.
- An external watchdog on top of the service. A service failure is a
  matter for the supervisor (the service's Popen).

## §2. Invariants

1. **`os.pidfd_open` is the only death-observation mechanism for
   Serena.** Not `wait()` (we have no `Process` object because of
   `stdio_client`), not `SIGCHLD` signals (they race with the event
   loop of the asyncio child watcher), not polling `/proc/<pid>`.
2. **Watcher — an asyncio task** registered via `loop.add_reader`
   on top of the pidfd file descriptor. Integrates cleanly with the
   current uvloop-free asyncio (see `_wait_pid_exit` in `supervisor.py`).
3. **Rate limit with a sliding window.** A list of restart timestamps;
   drop everything older than 60 s; length ≤3.
4. **One restart at a time.** An async `Lock` so parallel death events
   (coinciding with a manual `stop`) cannot create a race.
5. **`_mark_lost` on a dead instance is not undone** — the
   `SerenaBackend` object itself does not "come back to life". A restart
   creates a **new** `SerenaBackend` via `backend_factory()`, calls
   `attach_backend` (which re-installs `set_tool_handler`,
   `set_backend_refresh_handler` and `mark_backend_ready`). The previous
   dead object is released to the GC.
6. **Fail-loud telemetry.** Every restart (success and failure) is a
   log entry with the reason and the current counter.

## §3. How to obtain the pid for `pidfd_open`

**Result: option A (own transport) was chosen.** Option B (swapping the
private `_create_platform_compatible_process`) was reproduced live while
coding and turned out to be **non-viable** — see the "Why B was
rejected" subsection below. The reviewer accepted the rejection within
the same commit where the failure was first measured.

### Chosen implementation (option A)

`SerenaBackend` **no longer** uses
`mcp.client.stdio.stdio_client`. Instead, its own
transport `_OwnedStdioSession`, ~150 lines:

- Spawn Serena via `asyncio.create_subprocess_exec` (the pid is known
  right away via `proc.pid`, no `os.fork` in the supervisor).
- Own JSON-RPC framing by lines (which is exactly the MCP stdio spec) —
  two background asyncio tasks: reader (child.stdout → anyio memory
  stream) and writer (anyio memory stream → child.stdin).
- The ready anyio memory streams are passed to the ordinary
  `mcp.ClientSession`, which does the handshake as always.
- The pid is publicly available via `_OwnedStdioSession.pid`, and through
  it — `SerenaBackend.pid` (see §4).

Key property: `_OwnedStdioSession` is **not tied to the
cancel-scope of a specific owner task**. Enter/exit can be called
from different asyncio tasks without failing (see the B rejection
criterion).

**The only second layer of protection** — `SerenaBackend.start()`
spins up a dedicated `_run_session` task that holds all the
anyio contexts open inside one task. An external `stop()` only
sets `_stop_event` and awaits that task. Even if the internals of
mcp.ClientSession again want an anyio cancel scope, it is entered and
exited in the same task — guaranteed.

### Tests that pinned A

- `test_owned_session_reports_child_pid` — a direct transport test:
  a real child; `sess.pid` returns a valid OS pid.
- `test_backend_pid_reflects_real_child` — end-to-end through
  `SerenaBackend` with a JSON-RPC echo child (~50 lines).
- `test_backend_pid_is_none_when_fake_session_bypasses_transport` —
  a fake session (unit mocks) yields no pid → the supervisor honestly
  does not arm the monitor.

### Why B was rejected (after a live check)

- The swap approach was implemented and verified in isolation. It works
  as long as `stack.aclose()` (which tears down `stdio_client`) is called
  from **the same asyncio task** that entered it.
- In our service this is not the case:
  - `SerenaBackend.start()` is initiated from the service's main-loop
    task (inside `attach_backend`).
  - `SerenaBackend.stop()` is called **either** from the signal handler
    task (`server.stop → owner.stop`), **or** from the supervisor's
    `_on_death` task during a restart.
- Hence the live service crash:
  `RuntimeError: Attempted to exit cancel scope in a different task
  than it was entered in` — from the anyio cancel-scope invariant in
  `mcp.client.stdio.stdio_client.__aexit__`. The full stacktrace was
  captured from a live run and included in the code review of commit 10.
- Intermediate fix attempts (SIGTERM the child before aclose;
  suppress the `RuntimeError`) do not help: `stack.aclose()` is not
  atomic and cascades into `mcp.shared.session` — also an anyio task
  group on another task. We hit the **second** cancel-scope check and
  die again, this time on "asynchronous generator is already running".
- The only radical fix is to not let anyio own the transport
  (option A). Which is what was done.

### What happened to MUT-S

The mutation point MUT-S was "remove the guard on the missing
`_create_platform_compatible_process` → tests fail". In option A we
**do not touch** that private symbol at all, so MUT-S
loses its meaning. In its place — three unit tests of the pid (see
above) and an integ test on a real Serena death, each of which crashes
if the pid is not captured.

## §4. Classes and API

### `SerenaBackend` (changes)

```python
class SerenaBackend:
    ...
    def __init__(self, project_root, *, on_lost=None,
                 call_timeout_s=120.0):
        ...
        self._pid: int | None = None   # set in start()

    @property
    def pid(self) -> int | None:
        """Pid of the real Serena process; None before start() and after stop()."""
        return self._pid
```

`start()` uses the new CM `_capture_serena_process` to obtain the
pid. On failure the pid stays `None`.

### New `SerenaSupervisor`

Location: `deep_worker_cli/lsp/service/serena_supervisor.py`.
It lives next to the backend, not mixed with it — a single
responsibility.

```python
_MAX_RESTARTS: Final = 3
_WINDOW_S: Final = 60.0

class SerenaSupervisor:
    def __init__(
        self,
        server: "LSPServiceServer",
        backend_factory: Callable[[], SerenaBackend],
    ) -> None:
        self._server = server
        self._factory = backend_factory
        self._backend: SerenaBackend | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._restart_lock = asyncio.Lock()
        self._restarts: list[float] = []  # monotonic timestamps
        self._stopped = asyncio.Event()

    async def start(self) -> bool:
        """First attach. Returns the same as attach_backend."""

    async def stop(self) -> None:
        """Graceful stop. Shuts down both the monitor and the backend."""

    # ── internal ─────────────────────────────────────────
    async def _monitor_pidfd(self, pid: int) -> None:
        """Sleeps on an event via loop.add_reader(pidfd). On the event:
        removes the reader, closes the fd, calls _on_death (not the
        restart itself, so the reader callback is not blocked)."""

    async def _on_death(self) -> None:
        """The single entry point into restart: rate limit, then
        _restart_once. Holds _restart_lock. The algorithm is strictly:

            if self._stopped.is_set():
                return                       # graceful stop — no restart
            async with self._restart_lock:
                now = _now_fn()
                if not self._restart_allowed(now):
                    _log.error("serena: restart budget exhausted "
                               "(%d in %.0fs); backend stays not-ready",
                               len(self._restarts), _WINDOW_S)
                    return
                self._restarts.append(now)
                await self._restart_once()
        """

    async def _restart_once(self) -> bool:
        """Best-effort stop() of the old backend, then factory() → NEW
        SerenaBackend → server.attach_backend (this re-installs
        set_tool_handler / set_backend_refresh_handler /
        mark_backend_ready). On success starts a new pidfd monitor
        over the new pid. On failure: leaves alive=False, the counter is
        already incremented (attempt count, not success count)."""
```

### Rate limiting

```python
def _restart_allowed(self, now: float) -> bool:
    self._restarts = [t for t in self._restarts if now - t < _WINDOW_S]
    return len(self._restarts) < _MAX_RESTARTS
```

Computed atomically under `_restart_lock`; adds `now` when
allowed.

### Wiring in `service_cli.cmd_serve_impl`

Instead of the direct `await server.attach_backend(SerenaBackend(...))` —
```python
supervisor = SerenaSupervisor(
    server=server,
    backend_factory=lambda: SerenaBackend(project_root),
)
await supervisor.start()
# supervisor's stop is registered in the server shutdown-hook (A6 order)
```

### Rate-limit semantics

The counter counts **attempts**, not successes: even a failed restart
takes a slot. Otherwise a broken environment (a torn `serena` binary,
a broken pyrightconfig) could spin the factory forever.

Additionally — a restart is **not** triggered on cross-orthogonal
death sources:

- A **transport error** inside `call_tool`/`refresh` (commit 9)
  flips `alive=False` via `on_lost`, but does NOT trigger a restart.
  Auto-restart is driven **only by the pidfd event** (a real process
  exit from the kernel's perspective). If the session died but the
  process is alive — tool RPCs fail loud, no restart. "Hung-but-alive"
  is out of scope, see §9.

## §5. Wait mechanism — how exactly to wait for death

`os.pidfd_open(pid)` → fd. Register via
`loop.add_reader(fd, callback)`. The kernel makes the fd "readable"
when the process transitions to zombie/exit. The callback removes the
reader, closes the fd, sends an event to the queue; the async `_on_death`
picks it up.

Pseudo:
```python
loop = asyncio.get_running_loop()
fd = os.pidfd_open(pid)
event = asyncio.Event()

def _on_readable() -> None:
    try:
        loop.remove_reader(fd)
    finally:
        os.close(fd)
    event.set()

loop.add_reader(fd, _on_readable)
await event.wait()   # <— the only waiting point. Not polling.
```

Runs as `asyncio.create_task(self._monitor_pidfd(pid))` — the task lives
as long as the backend, until the event fires or `stop()` cancels it.

## §6. Shutdown order (update of A6 from commit 9)

The existing A6 of `LSPServiceServer.stop` does not change. Added:

- `SerenaSupervisor.stop()` is called by the server **instead of**
  `backend.stop()` (currently via `_backend.stop()` in `stop()`).
  Inside the supervisor: (1) set `_stopped`, (2) cancel the monitor
  without racing a new spawn, (3) call `backend.stop()`.

## §7. Tests

### Unit — `tests/unit_tests/lsp/test_serena_supervisor.py`

- `test_start_returns_true_when_backend_attaches`.
- `test_start_returns_false_when_first_attach_fails`.
- `test_stop_cancels_monitor_and_stops_backend`.
- `test_rate_limit_allows_three_and_blocks_fourth_in_window`
  (purely on `_restart_allowed`, without a real process).
- `test_rate_limit_window_expires_after_60s` (mock time via
  `now_fn` injection).

All without spawning Serena; the supervisor accepts `backend_factory`
and `pidfd_open_fn`/`add_reader_fn` via injection (see §8); the
supervisor's behavior is what is tested.

### Unit — `tests/unit_tests/lsp/test_serena_backend_pid.py`

- `test_start_captures_pid_via_swap` — we swap
  `_open_mcp_session`, but check that the public `.pid` is set;
  uses a fake process with a known pid (fork a sleeping child,
  clean up in finally).
- `test_swap_restored_on_exception` — if `session.initialize`
  raises, `_create_platform_compatible_process` is restored.

### Integ — `tests/integration_tests/lsp/test_serena_auto_restart.py`

(marker `integration`)

- `test_service_restarts_serena_on_kill` — bring up the service from
  this tree, wait for `Health.alive=True`, `kill -TERM` the real
  Serena pid, wait for `Health.alive` to become `True` again without
  manual actions.
- `test_service_gives_up_after_three_restarts_in_window` — four
  consecutive kills within 30 s → on the fourth, `Health.alive`
  stays `False`.

### Mutations

- **MUT-Q**: remove `list.append(now)` in `_restart_allowed` → the
  `test_rate_limit_allows_three_and_blocks_fourth_in_window` test fails.
- **MUT-R**: change `<` to `<=` (window off-by-one) →
  `test_rate_limit_window_expires_after_60s` fails.
- ~~**MUT-S**~~: dropped — option B (swap) was rejected; the private
  mcp SDK symbol is no longer touched (see §3).
- **MUT-T**: do not call `remove_reader/close` in the pidfd callback →
  the zombie fd is not released; a separate test `test_pidfd_closed_on_exit`.
- **MUT-U**: remove `mark_backend_ready` after a successful restart →
  the integ `test_service_restarts_serena_on_kill` fails (Health.alive
  stays False).
- **MUT-V**: remove the `_stopped` check in `_on_death` → a graceful
  `stop()` would start restarting the backend in a race with shutdown;
  the `test_stop_does_not_race_a_restart` test fails.
- **MUT-W**: in `_restart_once` call only `mark_backend_ready()`,
  without `set_tool_handler`/`set_backend_refresh_handler` (in other
  words, do not do the full `attach_backend`) → tool RPCs for the new
  backend still go to the old handler; the unit
  `test_restart_installs_handlers_on_new_backend` fails.

## §8. Injection points for testability

The supervisor accepts **optional** references:

- `pidfd_open_fn` / `add_reader_fn` — emulate death without a real
  `kill` in unit tests. By default — real syscalls.
- `now_fn` — source of monotonic time for the rate-limit tests.
  By default — `time.monotonic`.

### Pidfd callback hygiene

The reader callback runs in the event-loop context (not in a thread), but
the discipline is strict:

1. `remove_reader(fd)` + `os.close(fd)` — **exactly once** (guarded
   by a local flag, so a double fire is a no-op).
2. Do NOT `await` inside the reader callback (it is not an async
   function). The restart is launched via
   `asyncio.create_task(self._on_death())`;
   the callback itself is synchronous and returns control immediately.
3. The `_on_death` task is the only owner of the restart decision
   (via `_restart_lock`).

## §9. Limitations and open questions

- **A kernel without `pidfd_open`** (<5.3) — not supported. Our baseline
  is Linux 6.x; we do nothing.
- **A Serena that is hung but alive** (does not crash but does not
  answer) — out of scope; we catch it with the timeout on tool calls
  in commit 9; a "deadlock watchdog" is separate work.
- **Signal ordering on service shutdown** — `SerenaSupervisor.stop`
  is called before the server itself closes gRPC; otherwise tool RPCs
  could reach a killed backend. Pinned by a separate test in
  test_grpc_server (the A6 chain).

## §10. Delta boundaries

- NEW: `deep_worker_cli/lsp/service/serena_supervisor.py`.
- MOD: `deep_worker_cli/lsp/service/serena_backend.py` — `pid`
  property + `_capture_serena_process` CM.
- MOD: `deep_worker_cli/lsp/service/service_cli.py` — instead of a
  direct attach, a supervisor is created.
- MOD: `deep_worker_cli/lsp/service/grpc_server.py` — shutdown calls
  `supervisor.stop()` (if set) instead of `backend.stop()`. The reference
  to the supervisor goes through the new `attach_supervisor`/`set_backend_owner`.
- NEW: three test files (unit x2, integ x1).
- MOD: `docs/DESIGN_LSP_SERENA_BACKEND_ATTACH.md` — replace the §7
  "deferred" note with a reference to this document (see the reviewer's
  amendment №6).

Nothing in the bridge changes, the gRPC protocol does not change, and the
public `LSPServiceServer` API is extended by one method.
