# DESIGN: Show agent uptime in the web UI (event-driven; no polling)

## User requirement

> In the web UI, show each agent's current uptime (since launch / last
> restart) in the details/buttons panel.
> Polling is forbidden.

## What "uptime" means

An agent's uptime = `now() − agent.started_at`, where `started_at` is the moment
the agent's daemon process was started/restarted. It comes from the server
in the `AgentInfo.started_at` schema field (already present) and is taken from
`ProcessRegistry.list_all()` → `AgentProcessInfo.started_at`, where it is
written at registration time (after `daemonize()`).

Display format: `1d 03:14:22`, `03:14:22`, `07:31`, `42s` — compact,
like in dashboards. It refreshes in the UI **every second locally** (not over
the network).

## Architectural contract (important)

### What is NOT polling (and is allowed)

* A JS timer `setInterval(..., 1000)` in the browser that **re-renders the
  already received** `started_at` from the server relative to `Date.now()`. This
  is a local clock tick, not a network request. It makes no HTTP/WS calls.

### What IS polling (and is forbidden)

* Periodically fetching `/api/agents` or `/api/agents/{name}` to update
  uptime.
* Periodically calling `get_status` over RPyC.

### How the `started_at` value reaches the UI (event-driven)

Existing channels, nothing new needs to be invented:

1. **Initial load**: `GET /api/agents` → `AgentInfo[]` (already contains
   `started_at`).
2. **New agent**: `agent_discovered` WS event (already contains the `agent`
   field with `started_at`).
3. **Restart** (the agent stays in the registry, but `started_at` changes to a
   new value):
   - **Gap in the current code**: `AgentService._reconcile_from_registry()`
     updates `pid`/`rpyc_port` in `_agents[name]`, but **not** `started_at`
     and does **not** send a broadcast. After a restart the UI shows the OLD
     start time.
   - **Fix**: if for an already-known agent `proc_info.started_at !=
     info.started_at`, update it and broadcast a new
     `agent_started_at_changed` event with `{agent_name, started_at}`.

## Changes

### Backend (`deep_worker_web/app/services/agent_service.py`)

In `_reconcile_from_registry`, in the `known_names & current_names` block
(lines ~2247-2253), between updating `pid`/`rpyc_port` and reconnecting:

```python
if proc_info is not None and info is not None:
    info.pid = proc_info.pid
    info.rpyc_port = proc_info.rpyc_port
    if info.started_at != proc_info.started_at:
        info.started_at = proc_info.started_at
        await self._broadcast_message({
            "type": "agent_started_at_changed",
            "agent_name": name,
            "started_at": proc_info.started_at.isoformat(),
        })
```

This is the only backend change. All other channels already carry `started_at`.

### Frontend (`deep_worker_web/static/index.html` + `.../js/main.js`)

1. **HTML** (`index.html`): add an Uptime panel to `.agent-info-bar`:
   ```html
   <span class="status-label">Uptime:</span>
   <span id="agent-uptime" data-started-at="">-</span>
   ```

2. **JS** (**symbol names are prefixed with `Agent` to keep them separate from
   the existing `updateUptimeDisplay` for `#server-uptime` — that is a different
   feature**):
   - `formatAgentUptime(startedAtIso, nowMs)`: **pure** — takes an
     ISO time and the current `Date.now()`, returns a string like `1d 03:14:22`
     / `03:14:22` / `42s`. Easy to unit-test in node.
     * **Clamp**: when `delta < 0` (clock skew, started_at in the future) or
       `delta == 0` — return `"0s"`. No `NaN`/`-5s`.
   - `refreshAgentUptimeDisplay()`: reads `data-started-at` from the
     `#agent-uptime` span, formats it relative to `Date.now()`, updates
     `textContent`.
   - `startAgentUptimeTicker()` / `stopAgentUptimeTicker()`: the single
     `setInterval(refreshAgentUptimeDisplay, 1000)`.
     * **Uniqueness guarantee**: `startAgentUptimeTicker()`
       ALWAYS calls `stopAgentUptimeTicker()` first, so that switching
       `selectedAgent`/a repeated `agent_started_at_changed` does not
       spawn parallel intervals (a leak).
   - In `selectAgent(name)`: set `data-started-at =
     agent.started_at`, call `refreshAgentUptimeDisplay()` and
     `startAgentUptimeTicker()`.
   - In `handleAgentDiscovered(data)`: if the incoming agent is the current
     `selectedAgent`, update `data-started-at` and restart the ticker.
     **This covers the ungraceful restart path**: if reconcile managed to
     see the agent as absent between the death of the old one and the
     registration of the new pid, the event goes through `terminated → discovered` (rather than
     `agent_started_at_changed`). Both paths are NEEDED.
   - New handler `handleAgentStartedAtChanged(data)`: if
     `data.agent_name === selectedAgent`, update `data-started-at =
     data.started_at` and call `refreshAgentUptimeDisplay()` +
     `startAgentUptimeTicker()`. Always update
     `agents[name].started_at` in the in-memory map.
   - When the selected agent changes / `handleAgentTerminated` — the ticker
     stops via `stopAgentUptimeTicker()`.

## Tests (TDD)

### Unit backend

`deep_worker_web/tests/unit/test_agent_started_at_changed_broadcast.py`:

* Start `AgentService` with a mock `ProcessRegistry`, where one agent has
  `started_at = T0`. The first `_reconcile_from_registry` creates the entry
  (broadcast `agent_discovered`).
* Change `started_at → T1` (same agent, same pid — simulate that the
  agent was re-registered by a restart; a restart can happen on the same pid
  per our registration model — or on a new one; the key point is the timestamp
  changed).
* Expect a broadcast `agent_started_at_changed` with `agent_name=<name>` and
  `started_at=T1.isoformat()`. **And** — with a separate assertion (thread #3) —
  `service._agents[name].started_at == T1` (the in-memory value updated, not
  only the broadcast went out; catches a half-fix).

### Unit frontend (node.js, no browser)

`deep_worker_web/tests/unit/test_format_uptime.mjs`:

* `formatUptime(startedAt, now)` — a pure function, exported from
  `main.js`. Test several cases: 5s → `5s`, 65s → `01:05`,
  3665s → `01:01:05`, 90000s → `1d 01:00:00`. A node-ready module.

### Integration (real web + agent, no mocks)

`deep_worker_web/tests/integration/test_uptime_ui_e2e.py`:

* Start a real `AgentService` in the test, bring up a test agent
  via `agent_fixture`, wait for it in the registry.
* Connect a WS client to the web endpoint.
* From the first `agent_discovered` (or from the initial `list_agents`)
  read `started_at`; assert that it is ISO-dated and
  `now - started_at` is non-negative and below some reasonable
  upper threshold (the agent was started just now).
* Simulate a "restart" by directly updating
  `AgentProcessInfo.started_at` in the registry (a real agent restart
  cannot be done safely in tests on a live host — but patching the registry
  via `registry.register(info_with_new_started_at)` creates exactly
  the state that reconcile sees). Wait for the broadcast
  `agent_started_at_changed`. There are no mocks — only the real
  service + real registry + real WS.

## Scope / non-goals

* One backend change (the `started_at` diff in reconcile) + one WS event.
* One DOM element + one pure function + one local ticker.
* Privacy/security/architecture are not affected.
* Do **not** touch the existing SubscribeMessages channel.
* Do **not** add any new REST endpoints.
* Do **not** touch the existing server-uptime channel (`data.uptime` /
  `updateUptimeDisplay` / `#server-uptime`) — that is a separate feature about
  the uptime of the web process itself, not the agent.

## Documentation update

* `IMPLEMENTATION_STATUS.md` — a short note about the added feature.
* A comment at the new backend handler — "why event, not poll".
