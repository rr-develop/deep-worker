# Deep-Worker-Web API Documentation

## Base URL

```
http://127.0.0.1:7887
```

## Fire-and-Forget Architecture

**Important**: Most RPC operations in this API use a **fire-and-forget** pattern:

- **Immediate Response**: API endpoints return immediately without blocking
- **Background Execution**: Actual operations run in the background with timeouts
- **WebSocket Results**: Results are delivered via WebSocket callbacks
- **Error Handling**: Errors broadcast via WebSocket with detailed error types

**Affected Endpoints:**
- `POST /api/agents/{agent_name}/message` - Returns immediately, response via WebSocket
- `GET /api/agents/{agent_name}/history` - Returns immediately, history via WebSocket
- `GET /api/agents/{agent_name}/last-response` - Returns immediately, last response via WebSocket
- `GET /api/agents/{agent_name}/status` - Returns immediately, status via WebSocket
- `POST /api/agents/{agent_name}/stop_task_manager` - Returns immediately, result via WebSocket
- `POST /api/agents/{agent_name}/stop` - **202 Accepted**, progress via WebSocket (see Lifecycle)
- `POST /api/agents/{agent_name}/restart` - **202 Accepted**, progress via WebSocket (see Lifecycle)
- `POST /api/agents/batch` - **202 Accepted**, per-agent progress via WebSocket (see Lifecycle)

**Timeouts:**
- Send message: 60 seconds
- Get history: 15 seconds
- Get last response: 5 seconds
- Get status: 10 seconds
- Stop TaskManager: 5 seconds

### Lifecycle operations — fire-and-forget contract

Every lifecycle button in the UI (Stop / Restart / batch variants) is a
fire-and-forget call: the HTTP handler returns **`202 Accepted`** with a
`request_id` in **<100 ms**, and all progress flows through WebSocket
events correlated by that `request_id`.

Internals (event-driven, no polling, no threads):

- Process death is detected via `os.pidfd_open` + `loop.add_reader`
- SIGTERM → grace → SIGKILL escalation via one `loop.call_later`
- Registry appearance (spawn) is detected via inotify + `loop.add_reader`
- Batch is a parallel fan-out (`asyncio.create_task` per agent), not a
  serial `for await`

On-agent `exposed_shutdown(grace_seconds, force)` is tried first; if the
RPC succeeds, the agent self-exits after a 50 ms flush window. If it
isn't available (older agent) or the RPC fails, the web server falls
back to SIGTERM + pidfd watcher.

Event sequence (happy path):

```
HTTP  /api/agents/X/stop   → 202 {"request_id": "abc..."}
WS    agent_stop_started       {request_id, agent_name, payload.pid}
WS    agent_shutdown_requested (if RPyC path available)
WS    agent_stop_escalated     (if grace expired; payload.signal=SIGKILL)
WS    agent_terminated         {agent_name}
WS    agent_stop_completed     {request_id, payload.phase}
```

If stop fails (e.g. SIGKILL didn't actually kill), `agent_stop_failed`
is broadcast with `error`, `error_type`, and enough payload to debug
manually (`kill -9 <pid>`).

See **WebSocket API** section below for response message formats.

## REST API Endpoints

### List Agents

Get a list of all discovered deep-worker agents.

**Endpoint:** `GET /api/agents`

**Response:**
```json
[
  {
    "name": "project-a",
    "pid": 12345,
    "status": "idle",
    "provider": "claude",
    "model": "claude-sonnet-4",
    "rpyc_port": 18861,
    "working_dir": "/path/to/project",
    "started_at": "2024-01-01T12:00:00Z"
  }
]
```

### Get Agent Info

Get detailed information about a specific agent.

**Endpoint:** `GET /api/agents/{agent_name}`

**Response:**
```json
{
  "name": "project-a",
  "pid": 12345,
  "status": "idle",
  "provider": "claude",
  "model": "claude-sonnet-4",
  "rpyc_port": 18861,
  "working_dir": "/path/to/project",
  "started_at": "2024-01-01T12:00:00Z"
}
```

**Error Response (404):**
```json
{
  "detail": "Agent 'project-a' not found"
}
```

### Send Message to Agent

Send a message to an agent for processing.

**Endpoint:** `POST /api/agents/{agent_name}/message`

**Request Body:**
```json
{
  "message": "Hello, agent!"
}
```

**Response (Immediate - Fire-and-Forget):**
```json
{
  "success": true,
  "message": "Message sent to agent (response will arrive via WebSocket)",
  "agent_name": "project-a"
}
```

**Important:** This endpoint returns immediately. The actual agent response arrives via WebSocket:
- Success: `agent_response` message type
- Error: `agent_error` message type
- Timeout: 60 seconds (agent processing time)

### Get Conversation History

Retrieve the conversation history for an agent.

**Endpoint:** `GET /api/agents/{agent_name}/history`

**Response:**
```json
[
  {
    "role": "user",
    "content": "Hello",
    "timestamp": "2024-01-01T12:00:00Z",
    "metadata": null
  },
  {
    "role": "assistant",
    "content": "Hi! How can I help?",
    "timestamp": "2024-01-01T12:00:05Z",
    "metadata": null
  }
]
```

### Get Last Response

Get the last assistant response for an agent. More efficient than loading
full history when only the latest response is needed.

**Endpoint:** `GET /api/agents/{agent_name}/last-response`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_id` | string | auto-generated | Request correlation ID |

**Immediate Response:**
```json
{
  "success": true,
  "message": "Last response request sent (response will arrive via WebSocket)",
  "request_id": "abc-123"
}
```

**WebSocket Response** (type: `last_response_result`):
```json
{
  "type": "last_response_result",
  "agent_name": "project-a",
  "request_id": "abc-123",
  "success": true,
  "last_response": {
    "type": "AIMessage",
    "id": "msg-456",
    "content": "Here is the answer."
  }
}
```

If no assistant messages exist, `last_response` will be `null`.

- Timeout: 5 seconds

### Stop TaskManager

Stop the TaskManager running inside an agent.

**Endpoint:** `POST /api/agents/{agent_name}/stop_task_manager`

**Request Body:**
```json
{
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "TaskManager stop requested",
  "agent_name": "project-a",
  "force": false
}
```

### Stop Agent (fire-and-forget)

Stop the agent process. Returns **202 Accepted** immediately. Real
work runs in a background task; watch WebSocket events correlated
by `request_id` for progress.

**Endpoint:** `POST /api/agents/{agent_name}/stop`

**Request Body (all optional):**
```json
{
  "grace_seconds": 5.0,
  "force": false
}
```

- `grace_seconds` — wait this long after SIGTERM before escalating to
  SIGKILL. Default `5.0`.
- `force` — skip SIGTERM entirely, send SIGKILL immediately. Default
  `false`. Use when the agent is genuinely stuck.

**Response (202):**
```json
{
  "success": true,
  "request_id": "6f2c8b...",
  "message": "agent_stop scheduled; watch WebSocket for progress",
  "agent_name": "project-a"
}
```

Unknown agents do NOT return 404 — they return 202 and surface the
failure via an `agent_stop_failed` WS event with `error_type="LookupError"`.
This keeps the HTTP contract uniform so every UI button click gets
a trackable `request_id`.

### Restart Agent (fire-and-forget)

Restart the agent: stop then spawn using the saved launch_config
in `~/.deep_worker/{name}/session_metadata.json`. Returns **202**
instantly; watch WS events.

**Endpoint:** `POST /api/agents/{agent_name}/restart`

**Request Body (all optional):**
```json
{
  "model": null,
  "provider": null,
  "clean": false,
  "grace_seconds": 5.0,
  "force": false
}
```

- `model`, `provider`, `clean` — override saved launch config.
- `grace_seconds`, `force` — passed to the stop phase.

**Response (202):**
```json
{
  "success": true,
  "request_id": "6f2c8b...",
  "message": "agent_restart scheduled; watch WebSocket for progress",
  "agent_name": "project-a"
}
```

WS events flow: `agent_restart_started` → `agent_stop_*` sequence →
`agent_spawn_started` → `agent_spawn_completed` (or `agent_spawn_failed`
with a tail of `~/.deep_worker/logs/{name}.spawn.log` as payload).

### Batch Operations (fire-and-forget)

Fan out `stop` or `restart` across multiple agents in parallel.
Returns **202** with per-agent `request_id`s plus a batch-level
`request_id` for aggregate tracking.

**Endpoint:** `POST /api/agents/batch`

**Request Body:**
```json
{
  "action": "stop",
  "agent_names": ["project-a", "project-b", "project-c"],
  "grace_seconds": 5.0,
  "force": false,
  "model": null,
  "provider": null,
  "clean": false
}
```

- `action` — `"stop"` or `"restart"`.
- `agent_names` — non-empty list.
- `grace_seconds`, `force` — applied to every agent.
- `model`, `provider`, `clean` — only used when `action="restart"`.

**Response (202):**
```json
{
  "success": true,
  "request_id": "batch-abc...",
  "action": "stop",
  "jobs": [
    {
      "success": true,
      "request_id": "job-1...",
      "message": "stop scheduled",
      "agent_name": "project-a"
    },
    {"success": true, "request_id": "job-2...", "message": "stop scheduled", "agent_name": "project-b"}
  ]
}
```

Each per-agent job emits its own full `agent_stop_*` / `agent_restart_*`
event sequence. A `batch_progress` event is broadcast periodically with
`{completed, pending, total}`, and `batch_completed` fires once all
child jobs reach a terminal state.

### Get Detailed Status

Get detailed status information for an agent.

**Endpoint:** `GET /api/agents/{agent_name}/status`

**Response:**
```json
{
  "name": "project-a",
  "pid": 12345,
  "status": "running",
  "provider": "claude",
  "model": "claude-sonnet-4",
  "rpyc_port": 18861,
  "working_dir": "/path/to/project",
  "started_at": "2024-01-01T12:00:00Z",
  "current_thread_id": "thread-123",
  "message_count": 5,
  "last_activity": "2024-01-01T12:05:00Z",
  "task_manager_active": true
}
```

### Health Check

Check if the server is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "host": "127.0.0.1",
  "port": 7887
}
```

## WebSocket API

### Agent Events Stream

Real-time stream of agent events.

**Endpoint:** `WS /ws/agents`

**Protocol:** JSON messages

#### Client → Server Messages

**Ping:**
```json
{
  "type": "ping"
}
```

#### Server → Client Messages

**Pong Response:**
```json
{
  "type": "pong"
}
```

**Agent Message:**
```json
{
  "type": "agent_message",
  "agent_name": "project-a",
  "message": {
    "role": "assistant",
    "content": "Hello!",
    "timestamp": "2024-01-01T12:00:00Z",
    "metadata": null
  }
}
```

**Agent Status Change:**
```json
{
  "type": "agent_status",
  "agent_name": "project-a",
  "status": "running",
  "details": "Processing request"
}
```

**Agent Discovered:**
```json
{
  "type": "agent_discovered",
  "agent": {
    "name": "project-b",
    "pid": 12346,
    "status": "idle",
    "provider": "deepseek",
    "model": "deepseek-coder",
    "rpyc_port": 18862,
    "working_dir": "/path/to/project-b",
    "started_at": "2024-01-01T13:00:00Z"
  }
}
```

**Agent Terminated:**
```json
{
  "type": "agent_terminated",
  "request_id": "6f2c8b...",
  "agent_name": "project-a",
  "timestamp": "2026-04-23T18:40:03.456Z",
  "payload": {"pid": 12345, "phase": "pidfd"}
}
```

**Lifecycle — Agent Stop:**
```json
{"type": "agent_stop_started",   "request_id": "...", "agent_name": "project-a", "timestamp": "...", "payload": {"pid": 12345, "grace_seconds": 5.0, "force": false}}
{"type": "agent_shutdown_requested","agent_name": "project-a", "timestamp": "...", "payload": {"grace_seconds": 5.0, "force": false}}
{"type": "agent_stop_escalated", "request_id": "...", "agent_name": "project-a", "timestamp": "...", "payload": {"pid": 12345, "signal": "SIGKILL"}}
{"type": "agent_stop_completed", "request_id": "...", "agent_name": "project-a", "timestamp": "...", "payload": {"pid": 12345, "phase": "escalated_sigkill"}}
{"type": "agent_stop_failed",    "request_id": "...", "agent_name": "project-a", "timestamp": "...", "error": "Process 12345 still alive (state='D') after escalated_sigkill", "error_type": "RuntimeError", "payload": {"pid": 12345, "phase": "escalated_sigkill", "proc_state": "D"}}
```

`phase` values: `pidfd` (happy path — kernel event), `forced_sigkill`
(force=True), `escalated_sigkill` (grace expired), `already_dead`,
`no_pidfd` (fallback if pidfd_open unavailable).

**Lifecycle — Agent Restart / Spawn:**
```json
{"type": "agent_restart_started",  "request_id": "...", "agent_name": "project-a"}
{"type": "agent_spawn_started",    "request_id": "...", "agent_name": "project-a"}
{"type": "agent_spawn_completed",  "request_id": "...", "agent_name": "project-a", "payload": {"pid": 67890}}
{"type": "agent_spawn_failed",     "request_id": "...", "agent_name": "project-a", "error": "...", "payload": {"spawn_log_tail": "..."}}
{"type": "agent_restart_completed","request_id": "...", "agent_name": "project-a", "payload": {"spawn_request_id": "..."}}
```

**Lifecycle — Batch:**
```json
{"type": "batch_started",   "request_id": "batch-...", "payload": {"action": "stop", "agent_names": ["a","b","c"]}}
{"type": "batch_progress",  "request_id": "batch-...", "payload": {"action": "stop", "total": 3, "completed": 2, "pending": 1}}
{"type": "batch_completed", "request_id": "batch-...", "payload": {"action": "stop", "total": 3, "completed": 3, "failed": 0}}
```

**TaskManager Stop (fire-and-forget):**
```json
{"type": "task_manager_stopped", "agent_name": "project-a", "success": true, "force": false}
{"type": "task_manager_stop_error", "agent_name": "project-a", "success": false, "error": "..."}
```

**Agent Response (Fire-and-Forget Success):**
```json
{
  "type": "agent_response",
  "agent_name": "project-a",
  "success": true,
  "response": {
    "success": true,
    "response": "Agent's response content"
  }
}
```

**Agent Error (Fire-and-Forget Error):**
```json
{
  "type": "agent_error",
  "agent_name": "project-a",
  "success": false,
  "error": "Connection to agent lost",
  "error_type": "ConnectionError"
}
```

**Error Types:**
- `TimeoutError` - Agent did not respond within timeout (60s for messages)
- `ConnectionError` - Network/socket failure
- `EOFError` - Agent process died
- `RemoteError` - Exception in agent code

**Last Response Result (Fire-and-Forget):**
```json
{
  "type": "last_response_result",
  "agent_name": "project-a",
  "request_id": "uuid-1234",
  "success": true,
  "last_response": {
    "type": "AIMessage",
    "id": "msg-456",
    "content": "The answer is 7"
  }
}
```

**Last Response Error (Fire-and-Forget):**
```json
{
  "type": "last_response_error",
  "agent_name": "project-a",
  "request_id": "uuid-1234",
  "success": false,
  "error": "Failed to get last response"
}
```

**History Response (Fire-and-Forget):**
```json
{
  "type": "history_response",
  "agent_name": "project-a",
  "request_id": "uuid-1234",
  "success": true,
  "history": [...]
}
```

**Detailed Status Response (Fire-and-Forget):**
```json
{
  "type": "detailed_status_response",
  "agent_name": "project-a",
  "request_id": "uuid-5678",
  "success": true,
  "status": {...}
}
```

**TaskManager Stopped:**
```json
{
  "type": "task_manager_stopped",
  "agent_name": "project-a",
  "success": true,
  "force": false
}
```

**Subscription Success:**
```json
{
  "type": "subscription_success",
  "agent_name": "project-a"
}
```

## Data Models

### AgentStatus Enum

- `idle` - Agent is waiting for input
- `running` - Agent is actively processing
- `waiting_input` - Agent is waiting for user input
- `waiting_approval` - Agent is waiting for approval
- `error` - Agent encountered an error
- `shutdown` - Agent is shutting down

### MessageRole Enum

- `user` - Message from user
- `assistant` - Message from AI assistant
- `tool_call` - Agent requesting tool execution
- `tool_result` - Result from tool execution

## Error Responses

All endpoints may return error responses:

**400 Bad Request:**
```json
{
  "detail": "Invalid request format"
}
```

**404 Not Found:**
```json
{
  "detail": "Agent 'name' not found"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

No rate limiting is currently implemented. This is a local-only application.

## Authentication

No authentication is required. This application MUST only run on localhost (127.0.0.1:7887).

**SECURITY WARNING:** Never expose this API to external networks!
