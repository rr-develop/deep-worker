# Deep-Worker-Web: Initial setup complete

## ✅ Completed tasks

### 1. Project structure created

```
deep_worker_web/
├── app/
│   ├── __init__.py              # Application package
│   ├── config.py                # Configuration with security validation
│   ├── routers/                 # API endpoints (future)
│   ├── services/                # Business logic (future)
│   ├── models/                  # Data models (future)
│   ├── schemas/                 # Pydantic schemas (future)
│   ├── templates/               # Jinja2 templates (future)
│   └── utils/                   # Utilities (future)
├── static/
│   ├── css/                     # Styles (future)
│   └── js/                      # JavaScript (future)
├── .git/                        # Git repository
├── .gitignore                   # Ignored files
├── .pre-commit-config.yaml      # Pre-commit hooks (pyright strict)
├── pyrightconfig.json           # Pyright strict mode configuration
├── .pyrightconfig_strict_policy.md  # Type safety policy
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── run.py                       # Application entry point
```

### 2. Pyright Strict Mode configured

- ✅ `pyrightconfig.json` created with strict mode settings
- ✅ `.pre-commit-config.yaml` configured with mandatory type checking
- ✅ `.pyrightconfig_strict_policy.md` describes the type safety policy
- ✅ The check works: `python3 -m pyright app` → 0 errors

### 3. Security configured

- ✅ `app/config.py` with a mandatory `127.0.0.1:7887` check
- ✅ The server will refuse to start on other addresses/ports
- ✅ The check works: importing `settings` succeeds

### 4. Git repository initialized

- ✅ `git init` executed
- ✅ `.gitignore` created (forbids venv, __pycache__, etc.)
- ✅ Pre-commit hooks configured (although `pre-commit` is not installed globally)

### 5. Documentation created

- ✅ `README.md` with a detailed project description
- ✅ Operating principles described
- ✅ Differences from the reference web application explained
- ✅ Installation and startup instructions

## 🔧 Next steps

You can now start developing the functionality:

### Priority 1: Core functionality

1. **Create `app/main.py`** - a FastAPI application with lifespan
   - Similar to the reference web application's `app/main.py`
   - Security validation at startup
   - Event loop initialization for WebSocket

2. **Create `app/services/agent_service.py`** - working with deep-worker agents
   - Connecting to agents via RPyC
   - Getting the list of agents from ProcessRegistry
   - Subscribing to agent messages
   - Sending commands to agents

3. **Create `app/schemas/agent_schemas.py`** - Pydantic models
   - AgentInfo (name, status, provider, port)
   - AgentStatus (working, waiting_for_tool, waiting_for_user)
   - MessageSchema (role, content, timestamp)

4. **Create `app/routers/agents.py`** - API endpoints
   - `GET /api/agents` - list of agents
   - `GET /api/agents/{name}` - agent information
   - `POST /api/agents/{name}/message` - send a message
   - `GET /api/agents/{name}/history` - conversation history
   - `POST /api/agents/{name}/interrupt` - stop TaskManager

5. **Create `app/routers/websocket.py`** - WebSocket for real-time updates
   - Subscribing to agent events
   - Broadcasting messages to clients
   - Handling disconnects

### Priority 2: UI

1. **Create `app/templates/index.html`** - the main page
   - List of agents with statuses
   - Form for sending messages
   - Area for displaying responses
   - "Full Dialog" button
   - "Stop TaskManager" button

2. **Create `static/css/style.css`** - styles
   - Adapt from the reference web application
   - Simplify (no sessions)

3. **Create `static/js/main.js`** - JavaScript
   - WebSocket connection
   - Updating the agent list
   - Displaying messages
   - Handling button events

### Priority 3: Testing

1. **Create `tests/`** - unit and integration tests
   - Configuration tests (security validation)
   - agent_service tests
   - API endpoint tests

## 📋 Development requirements

### Python packages (already in requirements.txt)

```bash
pip install -r requirements.txt
```

Main dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` + `pydantic-settings` - Validation and settings
- `jinja2` - Templates
- `websockets` - WebSocket support
- `pyright` - Type checking

### RPyC Async (IMPORTANT!)

Do **NOT** install it via pip! Use a source import from the local RPyC source checkout.

In the code, use:
```python
import sys
from pathlib import Path

# Add rpyc_async to path
rpyc_async_path = Path(RPYC_ASYNC_SOURCE_DIR)
if str(rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(rpyc_async_path))

from rpyc.utils.async_server import AsyncioServer
```

### Deep-worker CLI

To interact with agents you will need:
```python
import sys
sys.path.insert(0, DEEP_WORKER_CLI_DIR)

from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.types import AgentStatus
```

## 🔒 Critically important principles

### 1. Security

**ALWAYS** verify:
- The server runs only on `127.0.0.1:7887`
- Never use `0.0.0.0`
- The validation in `config.py` must not be bypassed

### 2. Type Safety

**ALWAYS** use:
- Type annotations for all functions
- The `python3 -m pyright app` check before committing
- The rules in `pyrightconfig.json` must not be relaxed

### 3. Deep-Worker architecture

**ALWAYS** remember:
- The agent is the single source of truth for messages
- All messages go through the agent (do not render them locally!)
- Subscribe to events via RPyC callbacks
- Follow the principles from the CLI's `ARCHITECTURE.md`

## 🚀 Running

Once `app/main.py` has been created:

```bash
cd deep_worker_web
python run.py
```

Open a browser at: `http://127.0.0.1:7887`

## 📚 Reference projects

While developing, study:

1. **The reference web application** - a similar architecture
   - `app/main.py` - lifespan, security
   - `app/services/session_service.py` - working with sessions
   - `app/routers/websocket.py` - WebSocket broadcast
   - `app/templates/` - UI examples

2. **Deep-Worker CLI**
   - `process/registry.py` - ProcessRegistry
   - `client/agent_client.py` - AgentClient
   - `rpyc_service/agent_service.py` - RPyC service
   - `ARCHITECTURE.md` - architecture principles

## ✅ Ready for development!

The project structure has been created, the settings are done, the documentation is ready.
You can start writing code by following the steps described above.

Good luck! 🎉
