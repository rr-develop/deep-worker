# Deep-Worker-Web

Web UI for working with deep-worker agents.

## 🔒 CRITICAL SECURITY NOTICE

**⚠️ THIS APPLICATION MUST ONLY RUN ON 127.0.0.1:7887**

- **NEVER** expose this server to external networks
- **NEVER** change the host to 0.0.0.0 or any other address
- **NEVER** change the port from 7887
- This is a **LOCAL TRUSTED ENVIRONMENT ONLY** application
- No authentication or authorization is implemented
- Direct access to deep-worker agents with full capabilities

**Violating these requirements creates serious security vulnerabilities!**

## Overview

Deep-worker-web is a web UI for managing and interacting with deep-worker agents. Unlike the general-purpose web application which works with various session types (including DwSession), this application works directly with deep-worker agent processes via RPyC communication.

### Key Differences from the general-purpose web application

- **No DwSession abstraction**: Works directly with deep-worker agent processes
- **No session concept**: Manages running agent processes, not sessions
- **Specialized for deep-worker**: Designed exclusively for deep-worker agents
- **Direct RPyC communication**: Connects to agents via RPyC protocol

### Architecture

```
┌─────────────────────────────────────────────┐
│  deep-worker-web (127.0.0.1:7887)          │
│  ┌─────────────────────────────────────┐   │
│  │  FastAPI Web Server                 │   │
│  │  - Agent list UI                    │   │
│  │  - Agent status monitoring          │   │
│  │  - Message input/output             │   │
│  │  - Full dialog view                 │   │
│  │  - TaskManager controls             │   │
│  └─────────┬───────────────────────────┘   │
└────────────┼─────────────────────────────────┘
             │ RPyC
             ▼
┌──────────────────────────────────────────────┐
│  deep-worker agents (background processes)  │
│  ┌──────────────┐  ┌──────────────┐         │
│  │  Agent 1     │  │  Agent 2     │  ...    │
│  │  (RPyC port) │  │  (RPyC port) │         │
│  └──────────────┘  └──────────────┘         │
└──────────────────────────────────────────────┘
```

## Features

- **Agent Management**: View list of running deep-worker agents
- **Status Monitoring**: Real-time agent status (working, waiting for tool approval, waiting for user input, etc.)
- **Provider Display**: Show which AI provider each agent is using
- **Message Interface**: Input field for user messages and display area for agent responses
- **Full Dialog View**: Button to open complete conversation history in separate tab
- **TaskManager Control**: Stop button for agent's internal TaskManager
- **Non-blocking Architecture**: All RPC calls use fire-and-forget pattern for responsive UI
- **Real-time Notifications**: WebSocket-based error and success notifications

## Installation

```bash
cd deep_worker_web

# Install dependencies (NO virtual environment!)
pip install -r requirements.txt
```

**IMPORTANT**: Do NOT create a virtual environment! This project must use system-level Python packages to maintain compatibility with the other projects in the same checkout root.

## Running the Application

```bash
cd deep_worker_web
python run.py
```

The server will start on `http://127.0.0.1:7887`

**Security check**: The application will refuse to start if configured with any address other than `127.0.0.1` or any port other than `7887`.

## Type Checking

This project uses **pyright strict mode** with pre-commit hooks:

```bash
# Check types manually
python3 -m pyright app

# Install pre-commit hooks (required before first commit)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

**CRITICAL**: All Python code must pass pyright strict type checking. The pre-commit hook will prevent commits with type errors.

## Project Structure

```
deep_worker_web/
├── app/
│   ├── __init__.py
│   ├── config.py              # Application configuration with security validation
│   ├── main.py                # FastAPI application entry point
│   ├── routers/               # API endpoints
│   ├── services/              # Business logic (agent communication)
│   ├── models/                # Data models
│   ├── schemas/               # Pydantic schemas for API
│   ├── templates/             # Jinja2 HTML templates
│   └── utils/                 # Utility functions
├── static/
│   ├── css/                   # Stylesheets
│   └── js/                    # JavaScript
├── pyrightconfig.json         # Strict type checking configuration
├── .pre-commit-config.yaml    # Pre-commit hooks (pyright)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Development Principles

### Security First

- **127.0.0.1:7887 ONLY**: Hardcoded security requirement
- Server validates address/port on startup and refuses to run otherwise
- No authentication (relies on localhost isolation)

### Type Safety

- **Pyright strict mode**: All code must pass strict type checking
- **Pre-commit hooks**: Prevent commits with type errors
- **No exceptions**: Type errors must be fixed, not ignored

### No Virtual Environments

- All projects in this checkout root use system Python
- Virtual environments cause hard-to-debug compatibility issues
- A repository pre-commit hook prevents `.venv` creation

## Developer Documentation

For developers working with fire-and-forget patterns:

- **[DEVELOPER_EXAMPLES.md](docs/DEVELOPER_EXAMPLES.md)** - Code examples and patterns
- **[FIRE_AND_FORGET_MIGRATION_DESIGN.md](docs/FIRE_AND_FORGET_MIGRATION_DESIGN.md)** - Full design document
- **[API.md](docs/API.md)** - API reference with WebSocket message types
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - User guide with non-blocking architecture details

## Contributing

Before committing:

1. Ensure code passes pyright: `python3 -m pyright app`
2. Install pre-commit hooks: `pre-commit install`
3. All commits will be automatically checked for type errors
4. Fix any type errors - do NOT weaken type checking rules

## Technical Notes

### RPyC Communication

This application uses RPyC to communicate with deep-worker agents. The RPyC implementation must be imported from a local `rpyc_async` source checkout (not pip installed) to get `AsyncioServer` support for true bidirectional async communication.

### Fire-and-Forget Architecture

All RPC calls use the fire-and-forget pattern from `rpyc.utils.helpers.fire_and_forget_async()`:

- **Non-blocking**: RPC calls return immediately without waiting for agent response
- **Timeout protection**: Local timeouts prevent UI hanging if agent becomes unresponsive
- **Callback-driven**: Results delivered via WebSocket callbacks
- **Error visibility**: All RPC errors broadcast to UI via WebSocket notifications

**Migrated RPC calls:**
1. `invoke_agent()` - 60s timeout (agent processing can be slow)
2. `subscribe_messages()` - 10s timeout
3. `get_status()` (connection) - 5s timeout
4. `get_history()` - 15s timeout (large history)
5. `get_status()` (detailed) - 10s timeout
6. `unsubscribe_messages()` - 2s timeout (cleanup)
7. `stop_task_manager()` - 5s timeout

See `docs/FIRE_AND_FORGET_MIGRATION_DESIGN.md` for implementation details.

### Agent Discovery

Agents are discovered via ProcessRegistry (`/tmp/deep_worker_registry.json`). Each agent registers itself with:
- Agent name
- PID
- RPyC port
- Working directory
- Status

### Message Flow

Following deep-worker architecture principles:
1. User sends message via web UI
2. Web UI sends message to agent via RPyC
3. Agent broadcasts message to ALL subscribers (including web UI)
4. Agent processes message
5. Agent broadcasts response to ALL subscribers
6. Web UI displays response

**CRITICAL**: The web UI must NOT display messages locally before sending to agent. All messages must come from agent via RPyC callbacks to maintain single source of truth.

## License

See main repository license.
