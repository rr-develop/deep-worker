# DESIGN: Event-driven agent status + TaskManager visibility

**Status:** ✅ IMPLEMENTED (2026-04-24)
**Author:** Claude (analysis + fix)
**Date:** 2026-04-24
**Scope:** `deep_worker_cli/session/session_manager.py`,
`deep_worker_cli/process/agent_process.py`,
`deep_worker_cli/rpyc_service/agent_service.py`,
`deep_worker_web/app/services/agent_service.py`,
`deep_worker_web/app/schemas/agent_schemas.py`,
`deep_worker_web/static/js/main.js`.

## 0. Motivation

Two concrete user-facing symptoms:

1. `deep_worker_web` shows `demo_agent` as `idle` **hours after** the
   agent has been actively `working`. The live agent, probed directly
   via RPyC, returns `status: working, task_manager_active: true` —
   but the UI is stuck on the first `idle/ready` value recorded at
   discovery time.
2. TaskManager control row in the UI (`⏹️ Stop TM`, TM on/off toggle,
   `🔄 Reset TM`) stays hidden even though the agent was started
   with `--task-manager` and is really running a TM loop (confirmed
   in `~/.deep_worker/logs/demo_agent.err.log`: lines like
   `[INVOKE] TaskManager iteration 1` prove TM has been evaluating).

Both symptoms have the same root cause.

## 1. Root cause

Commit `231c369` (fix: drop 5s status polling) removed the
`asyncio.sleep(5)` loop that used to refresh `AgentInfo.status` /
TM flags every 5 seconds. The design doc at the time claimed the
agent would push status changes through the existing
`subscribe_messages` channel. **That was wrong** — the agent never
did that. Concretely:

* `SessionManager.set_status(new)` (called on every state transition
  inside `agent_process.py` and `status_callback.py`) only persists
  to `~/.deep_worker/<name>/session_metadata.json`. No `notify_callback`
  invocation — subscribers are not informed.
* Only six event types are ever broadcast through
  `_notify_subscribers`: `user`, `assistant`, `tool_call`,
  `tool_result`, `error`, `agent_interrupted`. No status, no TM
  transitions.
* `_update_agent_status(name, root)` on the web side is called once
  per connect and then never again. TM fields (`task_manager_active`,
  `task_manager_disabled`) are delivered only inside that one
  broadcast, and only to WebSocket clients connected at that exact
  moment.
* `AgentInfo` schema (the envelope sent for `agent_discovered` and
  returned by `GET /api/agents`) does not carry any TM field. So
  anyone who reloads the page after the agent came up gets agents
  marked as plain `idle` with no TaskManager information at all.

End result: new UI clients see a stale snapshot forever. The poll was
the only thing papering over the absence of push, and removing it
exposed a pre-existing gap.

## 2. Goals / constraints

* **No polling.** No `asyncio.sleep(5)`-style loops anywhere on either
  side. No periodic ticks. Every UI state change must trace to a
  real event.
* **No new threads.** asyncio only, single event loop.
* **Honest data only** (established in the previous fix): if a field
  is not known, surface "N/A" / omit, never synthesize.
* **Visibility vs usability of buttons** (clarified by user):
  whenever the agent has a TaskManager configured, the TM control
  row must be **visible** in the UI. Individual buttons may be
  **disabled** when the current state makes them unusable (e.g. no
  TM job active → `⏹️ Stop TM` disabled; TM already disabled →
  `Reset TM` may stay disabled). When circumstances change, the
  button becomes enabled again automatically through the same event
  stream.
* **Backward-compatible**. Older clients (dwcli, tests that speak
  the existing `MessageCallbackService.on_message` contract) must
  not break when the agent starts emitting a new event type. The
  new event is simply filtered out by clients that don't recognise it.

## 3. Design

### 3.1 Agent emits `status_change` events through `_notify_subscribers`

Extend `SessionManager` with an optional async notifier:

```python
class SessionManager:
    def __init__(self, ..., notifier: Callable[[dict], Awaitable[None]] | None = None):
        ...
        self._notifier = notifier

    async def set_status(self, status):
        self.status = status
        self.last_activity = datetime.now()
        await self._persist()
        if self._notifier is not None:
            await self._notifier({
                "type": "status_change",
                "status": status.value,
                # TaskManager fields are set when the agent wires the
                # notifier; see §3.2.
            })
```

Same treatment for `set_error` / `clear_error`.

`AgentProcess` (in `agent_process.py`) wires the notifier at
construction time:

```python
async def _status_notifier(payload):
    # Enrich with TaskManager state at the moment of the push.
    if self.task_manager is not None:
        payload = {
            **payload,
            "task_manager_active": self.task_manager.state != "inactive",
            "task_manager_disabled": self.task_manager.disabled,
        }
    await self._notify_subscribers(payload)

self.session_manager = SessionManager(data_dir, launch_config, notifier=_status_notifier)
```

`_notify_subscribers` already exists on the RPyC service side and
already serialises the payload to a JSON string (prior fix).
Subscribers just receive one more `type` value.

### 3.2 Initial-state snapshot at subscribe time

Race condition fix: if a new client subscribes after the agent has
already set its current status, nothing would ever tell them.
`exposed_subscribe_messages` must push the snapshot immediately:

```python
async def exposed_subscribe_messages(self, callback):
    self._agent_process._message_subscribers.add(callback)
    # Immediately send the current status_change so this new
    # subscriber doesn't have to wait for the next transition.
    try:
        snapshot = self._build_status_snapshot()
        await callback.on_message(json.dumps(snapshot))
    except Exception as exc:
        logger.warning("initial status snapshot failed: %s", exc)
    return True
```

Where `_build_status_snapshot()` returns exactly the same envelope
the notifier pushes, filled from `session_manager.status` and the
live `task_manager` (or `configured: False` when no TM).

This is still event-driven — the event is "a subscriber arrived",
and the response is a one-shot push. No poll, no loop.

### 3.3 TaskManager state-change push

TaskManager already has internal transitions (active ↔ inactive,
enabled ↔ disabled). These will be wired through the same notifier
whenever TM calls `session_manager.set_status` — which it already
does on iteration start / completion. So TM updates ride on the
existing status push with no extra plumbing.

If in a later PR TaskManager needs a dedicated event (e.g. new
"evaluation" type for UI badges), the same notifier can carry it —
the envelope is open-ended by design.

### 3.4 Web side — translate `status_change` into the existing
`agent_status` broadcast

`MessageCallbackService.exposed_on_message` already json-decodes.
Add a branch:

```python
if plain_msg.get("type") == "status_change":
    broadcast = {
        "type": "agent_status",
        "agent_name": agent_name,
        "status": plain_msg.get("status"),
    }
    if "task_manager_active" in plain_msg:
        broadcast["task_manager_active"] = plain_msg["task_manager_active"]
    if "task_manager_disabled" in plain_msg:
        broadcast["task_manager_disabled"] = plain_msg["task_manager_disabled"]
    # Update in-memory AgentInfo mirror for REST callers.
    if agent_name in agent_service._agents:
        agent_service._agents[agent_name].status = agent_service._convert_status(
            plain_msg.get("status", "idle")
        )
    await agent_service._broadcast_message(broadcast)
    return
```

Because the agent now pushes on every `set_status`, the web cache
stays in sync without any poll. Reloading the page and re-subscribing
also works: the new `MessageCallbackService` subscription triggers
the initial-snapshot (§3.2), and the web receives the current state
immediately.

### 3.5 `AgentInfo` schema — carry TM fields so REST is also honest

Today `GET /api/agents` omits TM information entirely. That is the
reason a reloaded UI has to wait for a WS event to know the TM
state. The authoritative store is `AgentService._agents`, which is
already updated by `_update_agent_status` / the new status_change
handler. Extend the schema:

```python
class AgentInfo(BaseModel):
    ...
    has_task_manager: bool | None = Field(
        None, description="True if agent was launched with a TaskManager; None when unknown")
    task_manager_active: bool | None = Field(None)
    task_manager_disabled: bool | None = Field(None)
```

Populated in `_update_agent_status` when TM fields are present in
the status response, and in the new `status_change` handler.

`None` vs `False` distinction is important: `None` means "we have
never heard from this agent about TM" (honest unknown); `False`
means "agent explicitly reported no TM". The UI treats `None` and
`False` the same at first paint (hide the row), but flips to
visible the moment `True` / `False` arrives.

### 3.6 Frontend — TM row visibility + per-button disabled state

`updateTmControls(agent)` currently hides the whole row when
`has_task_manager` is falsy. New rules:

* Row **visible** iff `agent.has_task_manager === true`. The moment
  agent's `status_change` event flips it to true, the row shows up.
* Each button decides its own `disabled` state from current status:
  * `⏹️ Stop TM` disabled when `!task_manager_active` (no work to stop).
  * TM enable/disable checkbox: `checked = !task_manager_disabled`,
    always enabled while the row is visible.
  * `🔄 Reset TM` enabled when the row is visible (always safe to
    reset history).

Buttons become enabled/disabled automatically as the event stream
arrives — no explicit re-check loop.

## 4. Out of scope

* No changes to TaskManager behaviour itself.
* No new DwSession work.
* No change to `get_status` / `get_metadata` RPyC contracts (kept
  unchanged per prior user instruction).
* No lifecycle (stop/restart/batch) changes.

## 5. TDD plan

### 5.1 Agent (deep_worker_cli)

Unit tests in `tests/unit_tests/session/test_session_manager_notify.py`:

1. `set_status` with no notifier → just persists (no regression).
2. `set_status` with an async notifier → `notifier` called exactly
   once with `{"type": "status_change", "status": <value>}`.
3. `set_error` also fires the notifier with the error status.
4. Notifier failure does NOT prevent `set_status` from persisting
   (defensive).

Unit tests in `tests/rpyc_service/test_subscribe_initial_snapshot.py`:

5. `exposed_subscribe_messages` invokes `callback.on_message(...)`
   once with a JSON-serialised status_change snapshot **before**
   returning `True`.
6. Snapshot payload includes `status` and, if TM configured,
   `task_manager_active` / `task_manager_disabled`.
7. When TM is not configured, snapshot omits TM fields (no
   fabrication).

### 5.2 Web (deep_worker_web)

Unit tests in `tests/unit/test_status_change_handling.py`:

8. JSON payload `{"type": "status_change", "status": "working",
   "task_manager_active": true, "task_manager_disabled": false}`
   arriving through `MessageCallbackService.exposed_on_message`
   triggers a broadcast of `{"type": "agent_status", "agent_name":
   ..., "status": "working", "task_manager_active": true, ...}`.
9. Same path updates `AgentService._agents[name].status` and
   `.task_manager_active` in place.
10. A payload without TM fields leaves the in-memory TM fields
    untouched (respects honest-unknown).

### 5.3 Frontend (Playwright + node parse)

11. `tests/test_ui_tm_visible_on_status.py` — Playwright smoke:
    feed a mock WS `agent_status` with `task_manager_active: true`;
    `#tm-toggle-label` becomes `display != none` within 500 ms.
12. Extend existing `test_frontend_js_syntax.py` coverage — already
    in place, catches parse errors across every JS change.

## 6. Acceptance criteria

* All 11 new unit tests pass (3 agent + 3 rpyc_service + 3 web + 2
  frontend).
* Existing 190 web unit + 21 web integration + all agent
  rpyc/cli tests stay green.
* Manual verification on `demo_agent`:
  * `deep-worker --restart demo_agent` (user said this is OK).
  * Reload UI → `demo_agent` appears with real status (not stuck on
    `idle`) and the TaskManager control row is visible within a
    second, even though no one typed any message.
  * Run `dwcli --status demo_agent` — same data as the UI (both
    honest, both synced).
* `/tmp/deep_worker_web.log` shows no status-related broadcasts
  outside of real events (no periodic tick).
* `grep -nE "asyncio.sleep\\([1-9]" deep_worker_web/app`  empty.
* No new threads. No new timers beyond what already exists in
  lifecycle (`call_later` for SIGTERM grace).

## 7. Rollout

Single PR (two logical commits acceptable):

1. `feat(agent): emit status_change events + initial snapshot on
   subscribe`
2. `feat(web): translate status_change to agent_status; show TM
   row on presence; sync AgentInfo`

Both commits TDD-first. After the first, integration tests still
green (subscribe channel is superset of before). After the second,
UI truly reflects live state.
