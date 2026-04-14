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

**Timeouts:**
- Send message: 60 seconds
- Get history: 15 seconds
- Get last response: 5 seconds
- Get status: 10 seconds
- Stop TaskManager: 5 seconds

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
  "agent_name": "project-a"
}
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
