# DESIGN: Fire-and-forget agent lifecycle (stop / restart / batch)

**Status:** ✅ IMPLEMENTED (2026-04-23)
**Date:** 2026-04-23
**Scope:** `deep_worker_web` backend routers/services + `deep_worker_cli` agent RPyC surface + frontend WebSocket dispatcher.

---

## 0. Motivation (the bug)

The `POST /api/agents/{name}/stop`, `.../restart` and `POST /api/agents/batch` endpoints in `app/routers/agents.py` were added in a single commit `e945c06` and **never** conformed to the fire-and-forget contract that the other endpoints rely on:

- `stop_agent_process` (`agents.py:570-625`) blocks waiting for the process to exit:
  `os.kill(pid, SIGTERM)` → 20 × `await asyncio.sleep(0.5)` with `/proc/{pid}` checks. That is **polling** + a **synchronous HTTP response** of up to 10 s.
- `batch_operation` (`agents.py:813-850`) — `for name in request.agent_names: await stop_agent_process(...)`: sequentially awaits each agent; the client response can hang for minutes.
- `restart_agent` (`agents.py:770-810`) — same, plus `_wait_for_agent_in_registry` (`agents.py:645-661`) with `await asyncio.sleep(0.5)` × 30.
- No `request_id`, no WS progress events, no SIGKILL escalation, no visible error to the user.
- The working `stop_task_manager` endpoint (`agents.py:321-405`) already uses `fire_and_forget_async`, but its WS events (`task_manager_stopped`, `task_manager_stop_error`) are **not handled in `main.js:657-707`** — they fall through to `default → console.warn`. The user never sees the result.

### User requirements (durable, must hold)

1. **Any** button press in the web UI must work through `fire_and_forget_async` (or a local `asyncio.create_task` for purely local steps).
2. Turning async code into sync is forbidden.
3. Polling is forbidden (including `asyncio.sleep` in a "wait for the process to disappear" loop).
4. Any new thread creation requires explicit user approval. Default is event-driven, in a single event loop.
5. Event-driven architecture: state changes on an event (signal, RPyC callback, WS event), not on a timer/poll.
6. Calls in loops must be non-blocking.
7. Errors must be visible to the user (WS notification + UI toast).
8. If SIGTERM doesn't work — escalate to SIGKILL, like `dwcli --stop` does (`deep_worker_cli/cli/client_command.py:171-312`).

---

## 1. Target architecture

### 1.1 Contract of each endpoint

A single template for any endpoint that starts a long-running operation:

1. The HTTP endpoint validates the input.
2. Generates `request_id = uuid4().hex`.
3. **Creates a background task via `fire_and_forget_async`** with `success_callback` and `error_callback` that broadcast WS events.
4. Immediately returns `202 Accepted` with body
   `{"success": true, "request_id": "<uuid>", "message": "<op> requested"}`.
5. Everything else goes over WebSocket: `*_started`, `*_progress` (for batch and multi-stage ops), `*_completed`, `*_failed`. Every event carries `request_id`, `agent_name` (or `agent_names` for batch), and an error structure on failure.

The HTTP endpoint does **not wait** for the result. It does not `await` domain logic longer than input validation and task scheduling. If it cannot schedule the task — `503` + a WS `*_failed` event (in case the client subscribed earlier).

### 1.2 Event-driven SIGTERM→SIGKILL without polling

Key idea: **SIGCHLD is an event**. The event loop listens for it via `loop.add_signal_handler(signal.SIGCHLD, ...)`. But the web server is not the agent's parent (the agent daemonizes via double-fork in `spawn_agent_process`, `agents.py:722` → `start_new_session=True`), so SIGCHLD never reaches the web server.

The solution has two paths, both event-driven:

**A. Preferred: the agent initiates shutdown itself (RPyC + fire-and-forget).**
Add an async method `exposed_shutdown(grace_seconds: float = 5.0, force: bool = False)` on the agent. It:
- asynchronously sends a WS `agent_shutdown_started` event to its subscribers (the web receives it via its RPyC callback);
- schedules `asyncio.create_task(self._graceful_exit(grace_seconds, force))` and returns from the RPC **immediately**;
- `_graceful_exit` in the same event loop: closes the RPyC server, waits for the current tool_call (with cancellation), then calls `loop.stop()` and `os._exit(0)`. With `force=True` — immediately the `os._exit(signal.SIGKILL)` equivalent (or send itself `SIGKILL` via `os.kill(os.getpid(), signal.SIGKILL)`).

The web layer calls this RPC via `fire_and_forget_async` exactly like `stop_task_manager`. The result is a WS event from the agent about the shutdown.

**B. Fallback: the agent does not respond to the RPC.**
The web layer sends `SIGTERM` externally and waits for the process to die via an **asynchronous watcher based on pidfd + `loop.add_reader`** — no polling:

```
import os, signal
pidfd = os.pidfd_open(pid)               # Linux ≥ 5.3 — ok
loop.add_reader(pidfd, _on_pid_exit, pidfd, request_id, agent_name)
os.kill(pid, signal.SIGTERM)
# if not dead within grace — loop.call_later(grace, _escalate_to_sigkill, ...)
```

`pidfd_open(pid)` opens a file descriptor that becomes readable exactly when the process dies. This is a real kernel event, not polling. `loop.call_later` schedules a one-shot escalation without a loop (exactly like `deep_worker_cli/process/agent_stop.py:91-112`, but without backoff polling).

Escalation stage: `_escalate_to_sigkill(pid, pidfd, request_id)` → `os.kill(pid, SIGKILL)`. We still wait on the `pidfd` `add_reader` (SIGKILL is guaranteed to close the fd — usually within milliseconds), plus a time cap via `loop.call_later(kill_hard_timeout, _report_zombie, ...)`.

**Zombie check** (like `client_command.py:274-312`) is also done once from `_on_pid_exit` via `open("/proc/{pid}/stat")`; if the PID is not found — "fully dead", if state is `Z` — "dead enough", otherwise — this is an impossible path → WS `*_failed` event with details.

If `pidfd_open` is not supported (kernel < 5.3), the fallback is a one-shot `loop.add_signal_handler(SIGCHLD, ...)` if the web itself spawned the process, or — as a last resort — a **single** `loop.call_later(delay, _check_once)` with exponential backoff (this is not a tight poll; each check costs one syscall and only runs if the previous event did not fire). But on kernels ≥ 5.3, pidfd is always available.

### 1.3 Event-driven SpawnAgent (restart)

`spawn_agent_process` (`agents.py:664-744`) currently does `subprocess.Popen` + `await _wait_for_agent_in_registry(timeout=15.0)`, which is polling `registry.lookup` with `asyncio.sleep(0.5)`.

Replace it with an **inotify watcher** on the `ProcessRegistry` directory (usually `/tmp/deep_worker_registry.json` — a single file). In the same event loop:

```
from asyncio.loop import add_reader
fd = inotify.init1(NONBLOCK)
inotify.add_watch(fd, registry_path, IN_CLOSE_WRITE | IN_MOVED_TO)
loop.add_reader(fd, _on_registry_change, fd, agent_name, request_id)
subprocess.Popen(...)                       # daemonized double-fork
loop.call_later(spawn_hard_timeout, _spawn_failed, ...)
```

When the registry is atomically rewritten by the agent after registration (json.dump via tmp+rename), an `IN_MOVED_TO` event arrives. `_on_registry_change` reads the file; if `agent_name` has appeared — broadcast `agent_spawn_completed` and `remove_reader`. Otherwise — ignore (it may be a change for another agent).

If inotify is unavailable (non-Linux), no separate event-based watcher via `asyncio.subprocess.Process.wait()` over a helper "probe" command is needed; the fallback stays the same: a single `call_later` for the hard timeout.

### 1.4 Batch — fan-out, not sequential

`POST /api/agents/batch`:
1. For each name — immediately places the job in the internal `JobRegistry` queue and does **not wait**.
2. Returns `{"request_id": "<batch-uuid>", "jobs": [{"agent_name": ..., "request_id": ...}, ...]}`.
3. Each per-agent operation is a separate `fire_and_forget_async` (SIGTERM watcher or RPyC shutdown). All start in parallel and each sends its own WS events with a per-job `request_id`.
4. Additionally — an optional aggregator: a subscriber to all per-job events that sends `batch_stop_progress` (`completed`/`total`) and `batch_stop_summary` at the end. The aggregator is a regular async task (not a thread) that lives while `batch.jobs_pending > 0`.

### 1.5 JobRegistry — single source of truth for request_id

The `LifecycleJobRegistry` service (inside `AgentService` or as a separate singleton):
- `start_job(op: str, agent_name: str | None, payload: dict) -> str` — issues a `request_id`, stores metadata, broadcasts `<op>_started`;
- `complete_job(request_id, result)` — `<op>_completed` + cleanup;
- `fail_job(request_id, exc)` — `<op>_failed` with `{"error": str(exc), "type": exc.__class__.__name__}`;
- `list_jobs()` — for diagnostics (not required now).

This gives a unified event format and a single audit point, and will also ease a future `GET /api/jobs/{request_id}` if needed (not in this change).

### 1.6 No new threads

Per the user's rule — we do not create threads. Everything runs in the main FastAPI/uvicorn asyncio loop. `asyncio.to_thread` is forbidden (it is still a thread pool, even if default). All I/O primitives we use (pidfd, inotify, signalfd, RPyC async) are kernel events via `loop.add_reader`/`loop.add_signal_handler`, without threads.

Existing threads (`LanceDBBackgroundEventLoop`, the `aiosqlite` worker, the `AnyIO` worker thread, the `asyncio` executor) — we leave them alone; they already exist and are not part of this change.

### 1.7 Status as a source of truth

Remove the gate "if `status=running` — refuse the stop" (`agents.py:594-606` and `agent_stop.py:72-78`). Reasons:

1. `session_metadata.json` is updated rarely and lags (an agent is currently stuck at `idle` for 6 hours during active work).
2. The user still wants to be able to kill a stuck agent; that is exactly the point of "stop".
3. The agent-side RPyC shutdown itself decides — to cancel the current ainvoke cleanly (path A), and if it can't — SIGTERM/SIGKILL comes from outside (path B).

To protect against accidentally killing useful work — if the user wants a soft stop, the UI sends `{"grace_seconds": N, "force": false}`; the agent may wait for the current tool call to finish up to `grace_seconds`. After the grace period expires — hard-kill. No 409s.

---

## 2. API surface (new)

All HTTP responses are `202 Accepted` with `{"success": true, "request_id": "<uuid>"}`; everything else goes over WS.

| Endpoint | Request | HTTP response | WS events |
|---|---|---|---|
| `POST /api/agents/{name}/stop` | `{"grace_seconds": 5.0, "force": false}` | `{"success":true,"request_id":"..."}` | `agent_stop_started`, `agent_stop_completed`, `agent_stop_failed`, `agent_terminated` |
| `POST /api/agents/{name}/restart` | `{"grace_seconds":5.0,"model":..., "provider":..., "clean":false}` | `{"success":true,"request_id":"..."}` | `agent_stop_started` → `agent_stop_completed` → `agent_spawn_started` → `agent_spawn_completed` / `agent_spawn_failed` |
| `POST /api/agents/batch` | `{"action":"stop"|"restart","agent_names":[...],"grace_seconds":...,"force":...}` | `{"success":true,"request_id":"<batch>","jobs":[{"agent_name":..,"request_id":..},..]}` | per-job events as above + `batch_started`, `batch_progress` (`{"completed":N,"total":M,"failed":K}`), `batch_summary` |
| `POST /api/agents/{name}/stop_task_manager` | `{"force":false}` | already fire-and-forget, keep as-is | `task_manager_stopped`, `task_manager_stop_error` |

All events include: `type`, `request_id`, `agent_name` (where applicable), `timestamp`, and for errors — `error`, `error_type`.

### 2.1 Event structure (unified)

```json
{
  "type": "agent_stop_started",
  "request_id": "6f2c...",
  "agent_name": "my-agent",
  "timestamp": "2026-04-23T18:40:00.123Z",
  "payload": { "grace_seconds": 5.0, "force": false, "pid": 12345 }
}
```

```json
{
  "type": "agent_stop_failed",
  "request_id": "6f2c...",
  "agent_name": "my-agent",
  "timestamp": "2026-04-23T18:40:03.456Z",
  "error": "Process 12345 in state 'D' after SIGKILL",
  "error_type": "RuntimeError",
  "payload": { "pid": 12345, "phase": "post_sigkill_verify" }
}
```

---

## 3. File-by-file changes (no code — design only)

### 3.1 `deep_worker_web/app/services/lifecycle.py` (NEW)
- Class `AgentLifecycleService` (injected via `Depends`; held at the `AgentService` level or alongside it).
- Methods (all async, all return `request_id` immediately):
  - `async def request_stop(agent_name, grace_seconds, force) -> str`
  - `async def request_restart(agent_name, overrides) -> str`
  - `async def request_batch(action, agent_names, params) -> BatchHandle`
- Internals:
  - `_start_shutdown_job(request_id, agent_name, grace, force)` — tries RPyC `root.shutdown(...)` via `fire_and_forget_async`; on `AttributeError`/`timeout` — falls back to `_pidfd_stop_job`.
  - `_pidfd_stop_job(request_id, pid, agent_name, grace, force)` — `pidfd_open` + `loop.add_reader` + `os.kill(SIGTERM)` + a single `loop.call_later(grace, _escalate)`; `_escalate` sends SIGKILL and schedules a second `call_later(kill_hard, _verify_zombie)`. Everything is event-driven.
  - `_on_pid_exit(pidfd, request_id, agent_name)` — remove_reader, close pidfd, zombie check (a single `open("/proc/{pid}/stat")`), `_registry.unregister`, `complete_job`, broadcast `agent_terminated`.
  - `_spawn_job(request_id, agent_name, cfg)` — `subprocess.Popen(..., start_new_session=True)` + an inotify watcher via `loop.add_reader` on the registry file + `call_later(spawn_hard_timeout, _spawn_failed)`.
  - `_batch_fanout(batch_id, action, names, params)` — for each name calls the corresponding `_start_shutdown_job` or `_start_restart_job`, keeps a `pending_set`, subscribes to per-job events via in-process pub-sub (the same `service._broadcast_message`) and sends `batch_progress`.
- `LifecycleJobRegistry` — a `request_id → JobState` dict with `start/complete/fail` methods that immediately call `service._broadcast_message(...)`.
- **Forbidden**: no function calls `asyncio.sleep` in a loop, creates threads, or uses `asyncio.to_thread`.

### 3.2 `deep_worker_web/app/routers/agents.py`
- `stop_agent`, `restart_agent`, `batch_operation` are rewritten in ~10 lines each:
  - validation → `request_id = await lifecycle.request_stop(...)` → return `{"success": true, "request_id": ...}`.
- `stop_agent_process`, `spawn_agent_process`, `_wait_for_agent_in_registry`, `load_launch_config` — move to `lifecycle.py` (or a separate `lifecycle_helpers.py`) as event-driven helpers without `asyncio.sleep` polling.
- Remove the 409 gate on `status=running`. If the client wants a soft stop — it passes `grace_seconds`, otherwise `force=true`.
- `stop_task_manager` — leave untouched (already correct); only verify its WS events now formally flow through `LifecycleJobRegistry` for unification.

### 3.3 `deep_worker_web/app/schemas/agent_schemas.py`
- `StopRequest` (new): `grace_seconds: float = 5.0`, `force: bool = False`.
- `RestartRequest` — extend with: `grace_seconds`, `force` (defaults `grace_seconds=5, force=False`).
- `BatchRequest` — add `grace_seconds`, `force`.
- Common `JobAcceptedResponse`: `success: bool`, `request_id: str`, `message: str`.
- `BatchAcceptedResponse`: + `jobs: list[JobAcceptedResponse]`.

### 3.4 `deep_worker_cli/deep_worker_cli/process/agent_process.py` + `rpyc_service.py`
- Add to the agent's RPyC service:
  - `async def exposed_shutdown(grace_seconds: float = 5.0, force: bool = False) -> dict`.
- Implementation: in the agent's same event loop:
  - if `force` or there is no active ainvoke — schedules an immediate `asyncio.create_task(self._do_exit(grace_seconds, force))` and returns `{"accepted": True, "pid": os.getpid()}`.
  - `_do_exit`: (a) cancel `self.current_worker_task` (already exists, `agent_process.py:1006`); (b) wait `asyncio.wait_for(current_worker_task, grace_seconds)` (a single await, not a poll); (c) close the RPyC server (`await self.rpyc_server.aclose()`); (d) `os._exit(0)` (or `os.kill(os.getpid(), SIGKILL)` when force).
- This makes path A feasible and event-driven for 99 % of cases — no external SIGTERM needed; the web simply waits for a WS event from the agent or from the registry watcher (`agent_terminated`).

### 3.5 `deep_worker_web/static/js/main.js`
- In the `handleWebSocketMessage` dispatcher (lines 657-707) add cases:
  - `agent_stop_started`, `agent_stop_completed`, `agent_stop_failed`
  - `agent_spawn_started`, `agent_spawn_completed`, `agent_spawn_failed`
  - `agent_shutdown_started` (from the agent)
  - `batch_started`, `batch_progress`, `batch_summary`
  - `task_manager_stopped`, `task_manager_stop_error` (these were missing!)
- Introduce a `pendingRequests[request_id] = { op, startedAt, buttonId, agentName }` table. `showNotification` on completion events; find the button via `buttonId` → reset state.
- The «⏹ Stop»/«🔄 Restart» buttons (index.html:30,53,55) — a single `trackButton(btn, 'Stopping…')` helper that records into `pendingRequests` and auto-resets on `*_completed` / `*_failed`.
- In `stopCurrentAgent` (main.js:1220-1233) — add `try/finally` and a visual state (currently there is none).
- Remove `alert('TaskManager stop requested')` in `stopTaskManager` — replace with `showNotification`.
- **All** error handlers must show `showNotification('...', 'error')` with text from `data.error` / `data.error_type` — to satisfy the "errors must be visible" requirement.

### 3.6 `deep_worker_web/app/services/agent_service.py`
- Clean up `convert_netref` (lines 368-378): make it async-friendly or require the agent to send already-serialized dicts. This is a separate task, but without it the agent's WS events (including the new `agent_shutdown_started`) will not reach the UI. Include it in the scope of this design, because otherwise fire-and-forget "stays silent" exactly as today.
- Add `broadcast_lifecycle_event(event: dict)` — a thin wrapper over `_broadcast_message` that guarantees the `type`, `request_id`, `agent_name`, `timestamp` fields.

---

## 4. Error paths (all visible to the user)

| Case | Path | WS event | UI |
|---|---|---|---|
| Agent not found | `request_stop` → fail_job | `agent_stop_failed` with `error_type=NotFound` | toast-error "Agent X not found" |
| RPyC shutdown failed (AttributeError / 2 s timeout) | fall back to pidfd+SIGTERM, log a warning | `agent_stop_started` with `payload.phase=external_signal` | toast-info "Graceful RPC failed, falling back to SIGTERM" |
| SIGTERM did not work within grace | SIGKILL | `agent_stop_started` with `payload.phase=escalated_sigkill` | toast-warning |
| After SIGKILL the process state != Z and it did not disappear | fail_job | `agent_stop_failed` with `error_type=KillVerifyFailed` + pid | toast-error with PID, advise manual `kill -9 PID` |
| Batch: some failed | per-job `_failed` + `batch_summary` with `failed_count` | batch_summary | summary toast-error "3/10 failed" + list |
| Spawn did not appear in the registry within hard-timeout | `_spawn_failed` + attach `err_log[-500:]` | `agent_spawn_failed` | toast-error with the first 500 characters of the log |
| Client disconnected before `*_completed` | state stays in `LifecycleJobRegistry`; on reconnect the web sends `pending_jobs_snapshot` | `pending_jobs_snapshot` | button state is restored |

---

## 5. Explicitly forbidden in this implementation

- ❌ `asyncio.sleep` in a loop waiting for process death or registry appearance.
- ❌ `await stop_agent_process(...)` inside an HTTP endpoint.
- ❌ Any `asyncio.to_thread` / `run_in_executor`.
- ❌ New `threading.Thread`s.
- ❌ Turning async into sync "for simplicity".
- ❌ The `status=running` gate from `session_metadata.json`.
- ❌ Silent `console.warn('Unknown message type')` on production events.
- ❌ HTTP responses `>100ms` for any of these endpoints (SLA).

---

## 6. Tests (TDD order)

### 6.1 Unit (no real process, mock + in-memory event loop)
1. `test_stop_returns_request_id_immediately` — HTTP returns in <100 ms, no `os.kill` is called in the main task; `fire_and_forget_async` is called with the expected `awaitable`.
2. `test_batch_fanout_is_parallel` — 5 names, all 5 job tasks start in a single loop tick; HTTP is answered before any `complete`.
3. `test_pidfd_watcher_no_polling` — mock `os.pidfd_open` and `loop.add_reader`; verify `asyncio.sleep` is **never** called on the happy path; escalation — exactly one `loop.call_later`.
4. `test_sigkill_escalation` — SIGTERM mocked to no-op; after grace we get SIGKILL; then — zombie check; the `agent_stop_completed` event with `payload.phase=escalated_sigkill`.
5. `test_sigkill_verify_failed` — state != Z → `agent_stop_failed` + message with PID.
6. `test_status_running_does_not_block_stop` — regression against the 409 gate.
7. `test_frontend_dispatches_new_events` — Jest/Playwright: a mocked WS sends `agent_stop_completed` → the toast appears, the button is reset.

### 6.2 Integration (real agent)
1. `test_stop_live_agent_via_rpyc_shutdown` — start an agent, call `/stop`, wait for `agent_stop_completed` over WS within <2 s.
2. `test_stop_hung_agent_force_kill` — start an agent, freeze its tool call (mock), `/stop {force:true}` → SIGKILL → the agent is gone from the registry within <1 s.
3. `test_batch_stop_5_agents_parallel` — 5 agents, `/batch {action:stop}`, HTTP within <100 ms, all 5 `agent_stop_completed` over WS within <3 s.
4. `test_restart_stop_then_spawn` — full cycle, two events `agent_stop_completed` + `agent_spawn_completed`; the agent appeared in the registry.
5. `test_websocket_loss_and_reconnect` — disconnect WS mid-operation; on reconnect — `pending_jobs_snapshot`.

### 6.3 Regression
- All existing integration tests `test_api_agents_integration.py`, `test_websocket_integration.py` must stay green (currently 21 passed).
- Add an explicit assertion that `/stop` returns within <100 ms (via `time.monotonic` in the test client).

---

## 7. Backward compatibility

- The HTTP response code changes from `200 OK` with the full result to `202 Accepted` with a `request_id`. This breaks any existing HTTP clients that read `response.message`. There are no such clients in the repository (see `main.js` — it already reads `data.message`, so the new format is compatible).
- There is no backward compatibility for the 409 "actively running" — this code is never returned again (that is a feature).
- `stop_task_manager` keeps working and the event format does not change — only new handlers are added in the UI.

---

## 8. Final principles-compliance checklist

| Principle | How it is ensured |
|---|---|
| No sync rewrites | all domain logic is `async def`, HTTP endpoints are `async def`, broadcast is `await service._broadcast_message` |
| No polling | `pidfd_open` + `loop.add_reader` for PID death; `inotify` + `loop.add_reader` for the registry; exactly one `loop.call_later` per grace and per hard timeout |
| No new threads | no `threading.Thread`, `asyncio.to_thread`, or `loop.run_in_executor`. Existing threads are neither added nor touched. |
| Event-driven | signals/pidfd/inotify/RPyC callbacks — all kernel/RPyC events |
| Non-blocking calls, especially in loops | batch fan-out via N parallel `fire_and_forget_async`, not `for await` |
| Errors visible | every error → WS `*_failed` event → UI toast with `error_type` and `error` |
| Stop like `dwcli --stop` | SIGTERM → grace → SIGKILL → zombie check, exactly like `client_command.py:171-312`, but event-driven |

---

## 9. Open questions (require explicit user decision)

1. **Default grace** — 5 s (like `dwcli`)? Or 2 s, or 0 (immediate SIGKILL)?
2. **Policy for `status=error`** — stop like a normal agent or require `force=true`?
3. **batch_progress frequency** — on every per-job event or throttled (every 250 ms)?
4. **pending_jobs_snapshot on WS reconnect** — needed now or in a later phase? (does not block the MVP.)
5. **A separate `GET /api/jobs/{request_id}` endpoint** to query job status over REST — needed, or WS only?
6. **Integrating the `convert_netref` async fix** — in the same PR or separately? (Without it the UI will not see the new events, so I propose the same PR.)

---

## 10. Rollout

One PR, one atomic refactor:

1. Add `lifecycle.py` + `LifecycleJobRegistry` + pidfd/inotify helpers.
2. Rewrite the `stop`/`restart`/`batch` endpoints + schemas.
3. Add `exposed_shutdown` on the agent.
4. Fix `convert_netref` (minimum: option B — `to_thread`, BUT since threads are forbidden — option C: serialize on the agent side into dict/str before publish; or option A: recursive async via `rpyc.async_`).
5. Update the frontend: new WS types, the `trackButton` helper, remove `alert`.
6. Tests: unit + integration + frontend.
7. Update `docs/API.md` and `docs/USER_GUIDE.md`.

Acceptance criteria: pressing any button in the UI results in an HTTP response within <100 ms, all visual changes and errors arrive via WS, `pytest tests/integration -m integration` is green, and the grep check `grep -RnE 'asyncio\.sleep|to_thread|threading\.Thread' app/` on the diff is empty (except for sleeps in tests).
