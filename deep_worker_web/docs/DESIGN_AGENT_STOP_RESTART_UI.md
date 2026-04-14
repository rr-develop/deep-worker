# Design: Agent Selection, Stop & Restart in deep_worker_web

**Status: ✅ IMPLEMENTED**

## Overview

Add multi-select agent management to deep_worker_web UI, inspired by the
general-purpose web application's session management. Users can select one or more agents and stop or restart
them via toolbar buttons.

## Current State

### What exists
- Agent list in left panel (populated via WebSocket `agent_discovered`)
- Single-agent click → shows interaction panel on the right
- Stop TaskManager button (stops TM, not the agent itself)
- Agent status tracking via WebSocket events

### What's missing
- Multi-select checkboxes on agents
- Stop agent (terminate process) — neither API nor UI
- Restart agent (`--restart`) — neither API nor UI
- Batch operations (stop/restart multiple agents)

## Design

### 1. Backend API Endpoints

#### `POST /api/agents/{name}/stop`

Gracefully stop a running agent process.

**Request:**
```json
{}
```

**Behavior:**
1. Look up agent in `ProcessRegistry`
2. If not found → 404
3. Read status from `session_metadata.json`
4. If status is `"running"` (actively working) → 409 Conflict with error message
5. Send SIGTERM to process
6. Wait up to 10s for exit (poll `/proc/{pid}`)
7. Broadcast `agent_terminated` via WebSocket
8. Return 200

**Response (success):**
```json
{"success": true, "message": "Agent 'myagent' stopped"}
```

**Response (conflict — actively working):**
```json
{"success": false, "error": "Agent 'myagent' is actively running (status: running). Wait for it to finish or interrupt it first."}
```

Uses `deep_worker_cli.process.agent_stop.stop_agent()` — already implemented.

#### `POST /api/agents/{name}/restart`

Restart an agent using saved `launch_config` from `session_metadata.json`.

**Request (optional overrides):**
```json
{
  "model": "sonnet",
  "clean": false
}
```
If body is empty or `{}`, uses saved parameters as-is (like `--restart`).

**Behavior:**
1. Load `launch_config` from `~/.deep_worker/{name}/session_metadata.json`
2. If no `launch_config` → 404 with message
3. If agent is running + status `"running"` → 409 Conflict
4. If agent is running + idle/waiting/error → stop it first (SIGTERM + wait)
5. Merge overrides from request body into saved config
6. Spawn new agent process via subprocess:
   ```
   python3 -m deep_worker_cli.cli.agent_command {name} \
     --provider {provider} --model {model} --dir {working_dir} \
     [-sp] [-tm] [--memory-only] [--use-sqlite] [--clean]
   ```
7. Wait up to 15s for new agent to appear in registry
8. Broadcast `agent_discovered` via WebSocket
9. Return 200 with new agent info

**Response (success):**
```json
{
  "success": true,
  "message": "Agent 'myagent' restarted",
  "agent": {"name": "myagent", "pid": 12345, "status": "idle", ...}
}
```

**Why subprocess instead of calling `run_agent()` directly:**
The agent process must be a daemon (double-fork, separate process group).
The web server cannot host the agent in its own event loop — they must be
separate OS processes. The CLI `agent_command.py` already handles daemonization.

#### `POST /api/agents/batch`

Batch operations on multiple agents.

**Request:**
```json
{
  "action": "stop",
  "agent_names": ["agent1", "agent2", "agent3"]
}
```

Supported actions: `"stop"`, `"restart"`.

**Behavior:**
- Execute action on each agent sequentially
- Collect results per agent
- Return summary

**Response:**
```json
{
  "results": {
    "agent1": {"success": true, "message": "Stopped"},
    "agent2": {"success": false, "error": "Actively running"},
    "agent3": {"success": true, "message": "Stopped"}
  }
}
```

### 2. WebSocket Events (new)

| Event type | Direction | Fields | When |
|---|---|---|---|
| `agent_stop_requested` | Server → Client | `agent_name` | Stop initiated |
| `agent_restart_requested` | Server → Client | `agent_name` | Restart initiated |
| `agent_stop_failed` | Server → Client | `agent_name, error` | Stop failed (e.g. actively running) |
| `agent_restart_failed` | Server → Client | `agent_name, error` | Restart failed |

Existing events that will fire naturally:
- `agent_terminated` — when process exits after stop
- `agent_discovered` — when new process starts after restart

### 3. Frontend UI Changes

#### Agent list (left panel) — multi-select

```
┌─────────────────────────────────────┐
│ Agents                    [toolbar] │
│ ┌─────────────────────────────────┐ │
│ │ [☑] agent-one   ● idle         │ │
│ │     claude-code / opus          │ │
│ │     /path/to/project-a          │ │
│ ├─────────────────────────────────┤ │
│ │ [☐] agent-two   ● running      │ │
│ │     claude-code / opus          │ │
│ │     /path/to/project-b          │ │
│ ├─────────────────────────────────┤ │
│ │ [☑] agent-three ● idle         │ │
│ │     deepseek / deepseek-chat    │ │
│ │     /path/to/project-c          │ │
│ └─────────────────────────────────┘ │
│                                     │
│ [⏹ Stop Selected] [🔄 Restart]     │
│ Selected: 2 agents                  │
└─────────────────────────────────────┘
```

**Interaction model (inspired by the general-purpose web application):**

- **Checkbox** on each agent item — toggles selection
- **Click on agent name/row** (not checkbox) — opens agent interaction panel (existing behavior)
- **"Stop Selected"** button — calls `POST /api/agents/batch` with `action: "stop"`
- **"Restart Selected"** button — calls `POST /api/agents/batch` with `action: "restart"`
- **Selection counter** — "Selected: N agents" shown below buttons
- **Select All / Deselect All** — checkbox in the toolbar header

**Button states:**
- "Stop Selected" enabled when ≥1 agent selected
- "Restart Selected" enabled when ≥1 agent selected
- Both disabled when 0 selected
- During operation: button shows spinner, disabled until complete

#### Individual agent controls (right panel header)

Add stop/restart buttons to the existing agent header:

```
┌──────────────────────────────────────────────────┐
│ agent-one                                        │
│ claude-code / opus  ● idle                       │
│ [📜 Dialog] [⏹ Stop] [🔄 Restart] [⏹ Stop TM]  │
└──────────────────────────────────────────────────┘
```

- **⏹ Stop** — calls `POST /api/agents/{name}/stop`
- **🔄 Restart** — calls `POST /api/agents/{name}/restart`

Both show confirmation toast on success, error toast on failure.

### 4. Implementation Plan

#### Phase 1: Backend (3 files)

| File | Changes |
|------|---------|
| `app/routers/agents.py` | Add `POST /stop`, `POST /restart`, `POST /batch` endpoints |
| `app/schemas/agent_schemas.py` | Add `StopRequest`, `RestartRequest`, `BatchRequest`, `BatchResult` schemas |
| `app/services/agent_service.py` | Add `stop_agent()`, `restart_agent()` methods |

The service methods reuse:
- `deep_worker_cli.process.agent_stop.stop_agent()` for stopping
- `deep_worker_cli.cli.agent_command.load_launch_config()` for reading saved config
- `subprocess.Popen()` for spawning new daemon process

#### Phase 2: Frontend (2 files)

| File | Changes |
|------|---------|
| `static/index.html` | Add checkboxes to agent items, add toolbar buttons, add individual stop/restart buttons |
| `static/js/main.js` | Add selection state, batch API calls, individual stop/restart handlers, button state management, WebSocket event handlers for new events |
| `static/css/style.css` | Checkbox styling, button states, spinner animation |

### 5. Error Handling

| Scenario | Backend Response | Frontend Behavior |
|---|---|---|
| Agent not found | 404 | Toast: "Agent not found" |
| Agent actively running | 409 | Toast: "Agent is actively working. Wait or interrupt first." |
| No launch_config for restart | 404 | Toast: "No previous config. Start agent normally first." |
| Agent already stopped | 200 (idempotent) | Toast: "Agent already stopped" |
| SIGTERM timeout | 500 | Toast: "Agent didn't stop in time. Check manually." |
| Subprocess spawn failure | 500 | Toast: "Failed to restart agent. Check logs." |

### 6. Reused Patterns

| Web application pattern | Adapted for deep_worker_web |
|---|---|
| Session checkboxes + `closeSessions()` | Agent checkboxes + `stopAgents()` / `restartAgents()` |
| `POST /api/sessions/close` batch endpoint | `POST /api/agents/batch` |
| `session_stop` WebSocket event | `agent_terminated` (already exists) |
| HTMX auto-refresh | Keep WebSocket-based updates (already real-time) |
| Runner state indicator | Agent status dot (already exists, just ensure updates) |

### 7. Out of Scope

- Model/provider override UI for restart (restart uses saved config)
- Agent creation/launch from web UI (agents are started via CLI only)
- Agent log viewer in web UI
- Agent configuration editor in web UI
