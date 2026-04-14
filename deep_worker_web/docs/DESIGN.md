# Deep-Worker-Web: Architecture Design Document

**Version**: 1.0  
**Date**: 2024-04-02  
**Status**: Draft for Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architectural Principles](#3-architectural-principles)
4. [Core Architecture](#4-core-architecture)
5. [Component Design](#5-component-design)
6. [Communication Protocol](#6-communication-protocol)
7. [Data Models](#7-data-models)
8. [UI Design](#8-ui-design)
9. [Security Architecture](#9-security-architecture)
10. [Testing Strategy](#10-testing-strategy)
11. [Implementation Phases](#11-implementation-phases)
12. [Open Questions](#12-open-questions)

---

## 1. Executive Summary

### Purpose

Deep-Worker-Web is a specialized web UI for managing and interacting with deep-worker agents running on localhost. It provides real-time monitoring, message exchange, and control capabilities for deep-worker agent processes.

### Key Differentiators from the reference web application

| Aspect | Reference web application | Deep-Worker-Web |
|--------|--------|-----------------|
| **Session Model** | Manages multiple session types (CCO, CSDK, Persistent, LangChain, DwSession) | No session abstraction - works directly with agent processes |
| **Communication** | Through session abstractions (DwSession wraps agent) | Direct RPyC connection to agent processes |
| **Scope** | General-purpose session manager | Specialized for deep-worker agents only |
| **Task Management** | External TaskManager | Internal TaskManager inside each agent |
| **Discovery** | CCO daemon for session discovery | ProcessRegistry for agent discovery |

### Design Goals

1. **Direct Agent Communication**: No intermediate abstractions, work directly with agent processes via RPyC
2. **Real-time Updates**: WebSocket-based live status and message streaming
3. **Type Safety**: 100% pyright strict compliance with pre-commit enforcement
4. **Security First**: Hardcoded localhost:7887 binding with startup validation
5. **Async Native**: Single event loop, no threads, pure asyncio with AsyncioServer
6. **TDD Approach**: Test-driven development with comprehensive coverage
7. **Zero Dependencies on External Packages**: Work with local source checkouts directly

---

## 2. System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (http://127.0.0.1:7887)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Agent List   │  │ Message I/O  │  │ Full Dialog  │          │
│  │ (Status)     │  │ (Input/Out)  │  │ (New Tab)    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          │ HTTP REST        │ WebSocket        │ HTTP REST
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  deep-worker-web (FastAPI @ 127.0.0.1:7887)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Main Event Loop (asyncio)                               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │ AgentService│  │ WebSocket  │  │ API Routes │         │   │
│  │  │ (RPyC Mgr) │  │ Broadcaster│  │ (REST)     │         │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘         │   │
│  └────────┼───────────────┼────────────────┼────────────────┘   │
│           │               │                │                     │
│  ┌────────▼───────────────▼────────────────▼────────────────┐   │
│  │  ProcessRegistry (/tmp/deep_worker_registry.json)        │   │
│  │  - Agent discovery                                        │   │
│  │  - PID, RPyC port, status tracking                       │   │
│  └───────────────────────────────────────────────────────────┘   │
└────────────┬────────────────────────────────────────────────────┘
             │ RPyC (AsyncioServer)
             │ - No threads, pure asyncio
             │ - Bidirectional callbacks
             │ - Message subscriptions
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Deep-Worker Agents (background processes)                      │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Agent "project-a"│  │ Agent "project-b"│  ...               │
│  │ PID: 12345       │  │ PID: 12346       │                    │
│  │ Port: 18861      │  │ Port: 18862      │                    │
│  │ Provider: claude │  │ Provider: deepseek│                   │
│  │ Status: running  │  │ Status: idle     │                    │
│  │                  │  │                  │                    │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                   │
│  │ │ RPyC Service │ │  │ │ RPyC Service │ │                   │
│  │ │ (AgentService)│ │  │ │ (AgentService)│ │                   │
│  │ └──────────────┘ │  │ └──────────────┘ │                   │
│  │                  │  │                  │                    │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                   │
│  │ │ TaskManager  │ │  │ │ TaskManager  │ │                   │
│  │ │ (Internal)   │ │  │ │ (Internal)   │ │                   │
│  │ └──────────────┘ │  │ └──────────────┘ │                   │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Web Server (FastAPI)**: HTTP/WebSocket server on 127.0.0.1:7887
2. **AgentService**: Manages RPyC connections to agent processes
3. **ProcessRegistry Client**: Discovers and tracks running agents
4. **WebSocket Broadcaster**: Real-time updates to browser clients
5. **Deep-Worker Agents**: Independent processes with RPyC servers

---

## 3. Architectural Principles

### P1: Direct Agent Communication

**NO session abstraction layer**. Unlike the reference web application which uses DwSession to wrap agents, deep-worker-web connects directly to agent processes via RPyC.

```python
# ❌ WRONG (reference web application pattern - uses DwSession wrapper)
session = DwSession(agent_name="project-a")
await session.invoke("Hello")

# ✅ CORRECT (deep-worker-web pattern - direct RPyC)
client = AgentClient("project-a")  # Direct RPyC connection
await client.connect()
await client.invoke_agent("Hello")
```

**Rationale**: Eliminates abstraction overhead, provides direct access to agent capabilities, simplifies architecture.

### P2: Single Event Loop Architecture

**ALL async operations run in ONE event loop**. No threads, no multiple loops, no run_coroutine_threadsafe.

```python
# ✅ CORRECT: Single event loop for everything
async def main():
    loop = asyncio.get_running_loop()
    
    # FastAPI runs in this loop
    # RPyC AsyncioServer uses this loop
    # WebSocket broadcasting uses this loop
    # All agent communication uses this loop
```

**Rationale**: Simplifies concurrency, prevents race conditions, eliminates threading bugs, enables true async/await.

### P3: Agent as Single Source of Truth

Following deep-worker architecture principles (see the CLI's `ARCHITECTURE.md`):

**ALL messages flow through agent and are broadcast to ALL subscribers**.

```python
# ❌ WRONG: Display message locally before sending to agent
await websocket.send_json({"type": "user", "content": message})
await agent.invoke(message)

# ✅ CORRECT: Send to agent, let agent broadcast back
await agent.invoke(message)
# Message will arrive via RPyC callback subscription
# and THEN be broadcast to WebSocket clients
```

**Rationale**: Ensures consistency across all clients, prevents message duplication, maintains single source of truth.

### P4: Security by Design

**127.0.0.1:7887 ONLY**. Hardcoded, validated at startup, no exceptions.

```python
class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 7887
    
    def validate_security(self) -> bool:
        if self.host != "127.0.0.1":
            raise ValueError("SECURITY ERROR: Must be 127.0.0.1 only!")
        if self.port != 7887:
            raise ValueError("SECURITY ERROR: Must be 7887 only!")
        return True

# Validated on import and at startup
settings = Settings()
settings.validate_security()
```

**Rationale**: Prevents accidental exposure to external networks, no authentication needed (localhost isolation), explicit fail-fast on misconfiguration.

### P5: Type Safety Everywhere

**100% pyright strict compliance**. No exceptions, enforced by pre-commit hooks.

```python
# ✅ All functions must have type annotations
async def get_agent_status(agent_name: str) -> AgentStatusInfo:
    ...

# ✅ All variables have inferred or explicit types
agents: list[AgentInfo] = []

# ❌ No untyped code allowed
def process_data(data):  # Fails pyright strict
    return data.get("value")  # Unknown types
```

**Rationale**: Catches bugs at development time, prevents runtime type errors, improves IDE support, enforces code quality.

### P6: No Virtual Environments

**Work directly with local source checkouts**. No pip install for local projects.

```python
# ✅ CORRECT: Import from source
import sys
sys.path.insert(0, RPYC_ASYNC_SOURCE_DIR)
from rpyc.utils.async_server import AsyncioServer

sys.path.insert(0, DEEP_WORKER_CLI_DIR)
from deep_worker_cli.process.registry import ProcessRegistry

# ❌ WRONG: Never pip install local projects
# pip install rpyc  # Uses pip version, not rpyc_async!
```

**Rationale**: Ensures consistency with the other local checkouts, uses latest source code, avoids version conflicts.

### P7: Test-Driven Development

**Write tests first, then implementation**. All features must have tests.

```python
# 1. Write test first
async def test_agent_discovery():
    service = AgentService()
    agents = await service.list_agents()
    assert len(agents) > 0
    assert agents[0].name == "test-agent"

# 2. Then implement to make test pass
class AgentService:
    async def list_agents(self) -> list[AgentInfo]:
        ...
```

**Rationale**: Ensures testability, catches regressions, documents expected behavior, improves design.

---

## 4. Core Architecture

### 4.1 Application Lifecycle

```python
# app/main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    logger.critical(f"SECURITY: Bound to {settings.host}:{settings.port} ONLY")
    
    # 1. Get main event loop
    loop = asyncio.get_running_loop()
    
    # 2. Set loop reference for WebSocket broadcasting
    websocket.set_main_event_loop(loop)
    
    # 3. Initialize AgentService (discovers agents from ProcessRegistry)
    agent_service = get_agent_service()
    await agent_service.initialize()
    
    # 4. Start background tasks (agent status monitoring)
    asyncio.create_task(agent_service.monitor_agents())
    
    yield
    
    # === SHUTDOWN ===
    await agent_service.shutdown()
    logger.info("Shutdown complete")
```

### 4.2 Request Flow: List Agents

```
Browser                    FastAPI                  AgentService               ProcessRegistry
  │                          │                          │                           │
  │──GET /api/agents────────>│                          │                           │
  │                          │                          │                           │
  │                          │──list_agents()──────────>│                           │
  │                          │                          │                           │
  │                          │                          │──list_all()──────────────>│
  │                          │                          │                           │
  │                          │                          │<──agents[]────────────────│
  │                          │                          │   (from registry JSON)    │
  │                          │                          │                           │
  │                          │<──AgentInfo[]────────────│                           │
  │                          │   (enriched with status) │                           │
  │                          │                          │                           │
  │<─JSON: [{name, status,   │                          │                           │
  │         provider, ...}]──│                          │                           │
```

### 4.3 Request Flow: Send Message to Agent

```
Browser                    FastAPI                  AgentService               Agent Process
  │                          │                          │                           │
  │──POST /api/agents/       │                          │                           │
  │    project-a/message─────>│                          │                           │
  │  {"message": "Hello"}    │                          │                           │
  │                          │                          │                           │
  │                          │──send_message()─────────>│                           │
  │                          │                          │                           │
  │                          │                          │──RPyC: invoke_agent()────>│
  │                          │                          │   (AsyncioServer)         │
  │                          │                          │                           │
  │                          │                          │<──RPyC callback───────────│
  │                          │                          │   {"type": "user",        │
  │                          │                          │    "content": "Hello"}    │
  │                          │                          │                           │
  │                          │                          │──broadcast_to_websockets()│
  │<═════════════════════════WebSocket═══════════════════════════════════════════════│
  │  {"type": "user",        │                          │                           │
  │   "content": "Hello"}    │                          │                           │
  │                          │                          │                           │
  │                          │                          │<──RPyC callback───────────│
  │                          │                          │   {"type": "assistant",   │
  │                          │                          │    "content": "Hi!"}      │
  │                          │                          │                           │
  │                          │                          │──broadcast_to_websockets()│
  │<═════════════════════════WebSocket═══════════════════════════════════════════════│
  │  {"type": "assistant",   │                          │                           │
  │   "content": "Hi!"}      │                          │                           │
```

**CRITICAL**: Notice that the user message is NOT broadcast by FastAPI! It comes back via RPyC callback from the agent, following the "Agent is Single Source of Truth" principle.

### 4.4 WebSocket Event Flow

```
Browser                    WebSocket Router          ConnectionManager          AgentService
  │                          │                          │                           │
  │──Connect /ws/agents─────>│                          │                           │
  │                          │                          │                           │
  │                          │──connect(websocket)─────>│                           │
  │                          │                          │ (add to active_connections)
  │<─Connected───────────────│                          │                           │
  │                          │                          │                           │
  │                          │                          │                           │
  │                          │                    [Agent event occurs]               │
  │                          │                          │                           │
  │                          │                          │<──on_message_callback─────│
  │                          │                          │   {"type": "assistant",   │
  │                          │                          │    "agent": "project-a",  │
  │                          │                          │    "content": "..."}      │
  │                          │                          │                           │
  │                          │                          │──broadcast()              │
  │<═════════════════════════════════════════════════════│   (to all connections)   │
  │  JSON event              │                          │                           │
```

---

## 5. Component Design

### 5.1 AgentService (app/services/agent_service.py)

**Purpose**: Central service for managing connections to deep-worker agents via RPyC.

**Key Responsibilities**:
- Discover agents from ProcessRegistry
- Establish RPyC connections using AsyncioServer
- Subscribe to agent message callbacks
- Broadcast agent events to WebSocket clients
- Manage agent lifecycle

**Interface**:

```python
class AgentService:
    """Service for managing deep-worker agent connections."""
    
    def __init__(self) -> None:
        """Initialize AgentService."""
        self._registry: ProcessRegistry = ProcessRegistry()
        self._connections: dict[str, AgentConnection] = {}
        self._message_callbacks: list[Callable[[dict[str, Any]], Awaitable[None]]] = []
    
    async def initialize(self) -> None:
        """Initialize service (discover agents, establish connections)."""
        ...
    
    async def list_agents(self) -> list[AgentInfo]:
        """List all running agents with their status.
        
        Returns:
            List of AgentInfo with name, status, provider, etc.
        """
        ...
    
    async def get_agent_status(self, agent_name: str) -> AgentStatusInfo:
        """Get detailed status for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentStatusInfo with detailed status
            
        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        ...
    
    async def send_message(
        self,
        agent_name: str,
        message: str
    ) -> dict[str, Any]:
        """Send message to agent.
        
        Args:
            agent_name: Target agent name
            message: User message
            
        Returns:
            Response from agent
            
        Raises:
            AgentNotFoundError: If agent doesn't exist
            RPCError: If communication fails
        """
        ...
    
    async def get_history(
        self,
        agent_name: str,
        limit: int = 50,
        offset: int = 0
    ) -> list[MessageInfo]:
        """Get conversation history from agent.
        
        Args:
            agent_name: Target agent name
            limit: Max messages to return
            offset: Offset for pagination
            
        Returns:
            List of MessageInfo
        """
        ...
    
    async def stop_task_manager(self, agent_name: str) -> bool:
        """Stop agent's internal TaskManager.
        
        Args:
            agent_name: Target agent name
            
        Returns:
            True if stopped successfully
        """
        ...
    
    def register_message_callback(
        self,
        callback: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register callback for agent messages.
        
        Args:
            callback: Async function to call on messages
        """
        ...
    
    async def monitor_agents(self) -> None:
        """Background task to monitor agent status changes.
        
        Runs continuously, checking ProcessRegistry for new/dead agents.
        """
        ...
    
    async def shutdown(self) -> None:
        """Shutdown service (close all connections)."""
        ...
```

**Implementation Notes**:

```python
class AgentConnection:
    """Represents RPyC connection to a single agent."""
    
    def __init__(self, agent_name: str, rpyc_port: int):
        self.agent_name = agent_name
        self.rpyc_port = rpyc_port
        self._client: AgentClient | None = None
        self._callback_handler: Any = None  # RPyC callback object
    
    async def connect(self) -> None:
        """Establish RPyC connection using AsyncioServer."""
        # Import from source (not pip!)
        import sys
        sys.path.insert(0, RPYC_ASYNC_SOURCE_DIR)
        sys.path.insert(0, DEEP_WORKER_CLI_DIR)
        
        from deep_worker_cli.client.agent_client import AgentClient
        
        self._client = AgentClient(self.agent_name)
        await self._client.connect()
        
        # Subscribe to messages
        self._callback_handler = MessageCallbackHandler(self._on_message)
        await self._client._root.subscribe_messages(self._callback_handler)
    
    async def _on_message(self, msg: dict[str, Any]) -> None:
        """Handle message from agent (RPyC callback)."""
        # Add agent name to message
        msg["agent_name"] = self.agent_name
        
        # Broadcast to all registered callbacks
        for callback in self._service._message_callbacks:
            await callback(msg)
```

### 5.2 WebSocket Management (app/routers/websocket.py)

**Purpose**: Handle WebSocket connections and broadcast agent events to browser clients.

```python
# app/routers/websocket.py

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove disconnected WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()

# Global reference to main event loop
main_event_loop: asyncio.AbstractEventLoop | None = None


def set_main_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Set main event loop reference for broadcasting."""
    global main_event_loop
    main_event_loop = loop


@router.websocket("/ws/agents")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent updates."""
    await manager.connect(websocket)
    
    try:
        # Keep connection alive
        while True:
            # Receive messages from client (e.g., ping/pong)
            data = await websocket.receive_json()
            
            # Handle client messages if needed
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


async def broadcast_agent_message(msg: dict[str, Any]) -> None:
    """Callback for AgentService to broadcast messages.
    
    Called when agent sends a message (user, assistant, tool_call, etc.).
    """
    await manager.broadcast(msg)
```

### 5.3 API Routes (app/routers/agents.py)

**Purpose**: REST API endpoints for agent management.

```python
# app/routers/agents.py

from fastapi import APIRouter, HTTPException, Depends
from app.services.agent_service import AgentService, get_agent_service
from app.schemas.agent_schemas import (
    AgentInfo,
    AgentStatusInfo,
    SendMessageRequest,
    SendMessageResponse,
    HistoryResponse,
)

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/", response_model=list[AgentInfo])
async def list_agents(
    service: AgentService = Depends(get_agent_service)
) -> list[AgentInfo]:
    """List all running deep-worker agents."""
    return await service.list_agents()


@router.get("/{agent_name}", response_model=AgentStatusInfo)
async def get_agent_status(
    agent_name: str,
    service: AgentService = Depends(get_agent_service)
) -> AgentStatusInfo:
    """Get detailed status for a specific agent."""
    try:
        return await service.get_agent_status(agent_name)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")


@router.post("/{agent_name}/message", response_model=SendMessageResponse)
async def send_message(
    agent_name: str,
    request: SendMessageRequest,
    service: AgentService = Depends(get_agent_service)
) -> SendMessageResponse:
    """Send message to agent.
    
    IMPORTANT: Response may not be immediate. Client should subscribe
    to WebSocket for real-time updates.
    """
    try:
        result = await service.send_message(agent_name, request.message)
        return SendMessageResponse(
            success=result.get("success", False),
            message="Message sent successfully"
        )
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_name}/history", response_model=HistoryResponse)
async def get_history(
    agent_name: str,
    limit: int = 50,
    offset: int = 0,
    service: AgentService = Depends(get_agent_service)
) -> HistoryResponse:
    """Get conversation history from agent."""
    try:
        messages = await service.get_history(agent_name, limit, offset)
        return HistoryResponse(messages=messages)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")


@router.post("/{agent_name}/interrupt")
async def interrupt_task_manager(
    agent_name: str,
    service: AgentService = Depends(get_agent_service)
) -> dict[str, bool]:
    """Stop agent's internal TaskManager."""
    try:
        success = await service.stop_task_manager(agent_name)
        return {"success": success}
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
```

---

## 6. Communication Protocol

### 6.1 RPyC Communication with AsyncioServer

**Import Strategy**:

```python
# CRITICAL: Import rpyc from source, not pip
import sys
from pathlib import Path

# Add rpyc_async to path FIRST
rpyc_async_path = Path(RPYC_ASYNC_SOURCE_DIR)
if str(rpyc_async_path) not in sys.path:
    sys.path.insert(0, str(rpyc_async_path))

# Now import rpyc (will use AsyncioServer-enabled version)
import rpyc
from rpyc.utils.async_server import AsyncioServer

# Add deep-worker CLI to path
dw_cli_source_path = Path(DEEP_WORKER_CLI_DIR)
if str(dw_cli_source_path) not in sys.path:
    sys.path.insert(0, str(dw_cli_source_path))

from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.process.registry import ProcessRegistry
```

**Connection Pattern**:

```python
async def connect_to_agent(agent_name: str, rpyc_port: int) -> rpyc.Connection:
    """Connect to agent via RPyC using AsyncioServer.
    
    CRITICAL: Must enable asyncio serving BEFORE accessing conn.root!
    """
    # 1. Establish connection (SECURITY: localhost only)
    conn = rpyc.connect(
        "127.0.0.1",
        rpyc_port,
        config={"allow_public_attrs": True}
    )
    
    # 2. Enable asyncio serving (BEFORE accessing root!)
    loop = asyncio.get_running_loop()
    conn.enable_asyncio_serving(loop=loop)
    
    # 3. Now safe to get root
    root = conn.root
    
    return conn, root
```

**Message Subscription Pattern**:

```python
class MessageCallbackHandler:
    """RPyC callback handler for agent messages."""
    
    def __init__(self, callback: Callable[[dict[str, Any]], Awaitable[None]]):
        self._callback = callback
    
    async def on_message(self, msg: dict[str, Any]) -> None:
        """Called by agent when new message arrives."""
        await self._callback(msg)


async def subscribe_to_agent_messages(root: Any) -> None:
    """Subscribe to agent messages via RPyC callback."""
    handler = MessageCallbackHandler(on_agent_message)
    
    # Subscribe (agent will call handler.on_message for each message)
    await root.subscribe_messages(handler)
```

### 6.2 WebSocket Protocol

**Message Format**: All WebSocket messages use JSON with `type` field.

#### Client → Server Messages

```typescript
// Ping (keep-alive)
{
  "type": "ping"
}

// Pong response from server
{
  "type": "pong"
}
```

#### Server → Client Messages

```typescript
// Agent message (user, assistant, tool_call, tool_result)
{
  "type": "agent_message",
  "agent_name": "project-a",
  "message": {
    "role": "user" | "assistant" | "tool_call" | "tool_result",
    "content": string,
    "timestamp": string (ISO 8601)
  }
}

// Agent status change
{
  "type": "agent_status",
  "agent_name": "project-a",
  "status": "idle" | "running" | "waiting_input" | "error",
  "details": string | null
}

// Agent discovered (new agent started)
{
  "type": "agent_discovered",
  "agent": {
    "name": string,
    "pid": number,
    "provider": string,
    "status": string
  }
}

// Agent terminated
{
  "type": "agent_terminated",
  "agent_name": string
}
```

---

## 7. Data Models

### 7.1 Pydantic Schemas (app/schemas/agent_schemas.py)

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Status of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    WAITING_APPROVAL = "waiting_approval"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentInfo(BaseModel):
    """Basic information about an agent."""
    name: str = Field(..., description="Agent name")
    pid: int = Field(..., description="Process ID")
    status: AgentStatus = Field(..., description="Current status")
    provider: str = Field(..., description="AI provider (claude, deepseek, etc.)")
    model: str = Field(..., description="Model name")
    rpyc_port: int = Field(..., description="RPyC server port")
    working_dir: str = Field(..., description="Working directory")
    started_at: datetime = Field(..., description="Start time")


class AgentStatusInfo(BaseModel):
    """Detailed status information for an agent."""
    name: str
    pid: int
    status: AgentStatus
    provider: str
    model: str
    rpyc_port: int
    working_dir: str
    started_at: datetime
    current_thread_id: str | None = Field(None, description="Current conversation thread")
    message_count: int = Field(0, description="Total messages in current thread")
    last_activity: datetime | None = Field(None, description="Last activity timestamp")
    task_manager_active: bool = Field(False, description="Is TaskManager running")


class MessageRole(str, Enum):
    """Message role."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class MessageInfo(BaseModel):
    """Message in conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime
    thread_id: str | None = None


class SendMessageRequest(BaseModel):
    """Request to send message to agent."""
    message: str = Field(..., min_length=1, description="User message")


class SendMessageResponse(BaseModel):
    """Response from sending message."""
    success: bool
    message: str = Field(..., description="Status message")


class HistoryResponse(BaseModel):
    """Response with conversation history."""
    messages: list[MessageInfo]
```

### 7.2 Internal Models (app/models/agent.py)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AgentConnection:
    """Internal representation of agent connection."""
    agent_name: str
    rpyc_port: int
    client: Any  # AgentClient instance
    callback_handler: Any  # RPyC callback handler
    connected_at: datetime
    last_activity: datetime
```

---

## 8. UI Design

### 8.1 Main Page Layout

```
┌────────────────────────────────────────────────────────────────┐
│ Deep-Worker-Web                                    [127.0.0.1] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ┌─────────────────┐  ┌────────────────────────────────────┐   │
│ │ Agents          │  │ Message Interface                  │   │
│ │                 │  │                                    │   │
│ │ ☑ project-a     │  │ Selected: project-a               │   │
│ │   ● Running     │  │                                    │   │
│ │   Claude Sonnet │  │ ┌────────────────────────────────┐ │   │
│ │   Port: 18861   │  │ │ Last Response:                 │ │   │
│ │                 │  │ │                                │ │   │
│ │ ☐ project-b     │  │ │ [Agent response appears here]  │ │   │
│ │   ○ Idle        │  │ │                                │ │   │
│ │   DeepSeek R1   │  │ │                                │ │   │
│ │   Port: 18862   │  │ └────────────────────────────────┘ │   │
│ │                 │  │                                    │   │
│ │ ☐ project-c     │  │ [Full Dialog] [Stop TaskManager]  │   │
│ │   ⏸ Waiting     │  │                                    │   │
│ │   Claude Opus   │  │ ┌────────────────────────────────┐ │   │
│ │   Port: 18863   │  │ │ Your message:                  │ │   │
│ │                 │  │ │                                │ │   │
│ │                 │  │ │ [Type message here...]         │ │   │
│ │                 │  │ │                                │ │   │
│ │                 │  │ └────────────────────────────────┘ │   │
│ │                 │  │                                    │   │
│ │ [Refresh]       │  │ [Send]                             │   │
│ └─────────────────┘  └────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 Agent List Item

Each agent in the list shows:

```html
<div class="agent-item" data-agent-name="project-a">
    <input type="radio" name="selected-agent" value="project-a">
    
    <div class="agent-info">
        <div class="agent-name">project-a</div>
        
        <div class="agent-status">
            <span class="status-indicator status-running">●</span>
            Running
        </div>
        
        <div class="agent-provider">
            <span class="provider-badge provider-claude">Claude</span>
            Sonnet 4.5
        </div>
        
        <div class="agent-meta">
            PID: 12345 | Port: 18861
        </div>
    </div>
</div>
```

**Status Indicators**:
- `status-running` (green): Agent is executing
- `status-idle` (gray): Agent is idle
- `status-waiting` (yellow): Waiting for input or approval
- `status-error` (red): Error state

### 8.3 Message Interface

```html
<div class="message-interface">
    <!-- Selected agent header -->
    <div class="selected-agent-header">
        <h3>Agent: <span id="selected-agent-name">project-a</span></h3>
        <span id="selected-agent-status" class="status-badge">Running</span>
    </div>
    
    <!-- Last response panel -->
    <div class="last-response-panel">
        <div class="panel-header">
            <h4>Last Response:</h4>
            <div class="panel-actions">
                <button onclick="copyResponse()">📋 Copy</button>
                <button onclick="openFullDialog()">Full Dialog</button>
                <button onclick="stopTaskManager()">⏹ Stop TaskManager</button>
            </div>
        </div>
        
        <div id="last-response-content" class="response-content">
            [Agent response will appear here]
        </div>
    </div>
    
    <!-- Message input -->
    <div class="message-input-panel">
        <label for="message-input">Your message:</label>
        <textarea id="message-input" 
                  placeholder="Type your message here..."
                  rows="5"></textarea>
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
</div>
```

### 8.4 Full Dialog Page (New Tab)

Opened when user clicks "Full Dialog" button.

URL: `/agents/{agent_name}/dialog`

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dialog: {agent_name}</title>
</head>
<body>
    <div class="dialog-container">
        <div class="dialog-header">
            <h2>Full Dialog: <span>{agent_name}</span></h2>
            <div class="dialog-meta">
                Thread: <span id="thread-id">default</span> |
                Messages: <span id="message-count">0</span>
            </div>
        </div>
        
        <div id="dialog-messages" class="dialog-messages">
            <!-- Messages rendered here -->
            <div class="message message-user">
                <div class="message-header">
                    <span class="message-role">User</span>
                    <span class="message-time">14:32:15</span>
                </div>
                <div class="message-content">
                    Hello, can you help me?
                </div>
            </div>
            
            <div class="message message-assistant">
                <div class="message-header">
                    <span class="message-role">Assistant</span>
                    <span class="message-time">14:32:18</span>
                </div>
                <div class="message-content">
                    Of course! I'd be happy to help.
                </div>
            </div>
            
            <!-- More messages... -->
        </div>
        
        <div class="dialog-actions">
            <button onclick="refreshDialog()">🔄 Refresh</button>
            <button onclick="exportDialog()">💾 Export</button>
            <button onclick="window.close()">✕ Close</button>
        </div>
    </div>
</body>
</html>
```

### 8.5 JavaScript (static/js/main.js)

```javascript
// WebSocket connection
let ws = null;
let selectedAgent = null;

function connectWebSocket() {
    ws = new WebSocket('ws://127.0.0.1:7887/ws/agents');
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        // Send ping every 30 seconds
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'agent_message':
            handleAgentMessage(data);
            break;
        case 'agent_status':
            updateAgentStatus(data);
            break;
        case 'agent_discovered':
            addAgentToList(data.agent);
            break;
        case 'agent_terminated':
            removeAgentFromList(data.agent_name);
            break;
        case 'pong':
            // Ping/pong for keep-alive
            break;
    }
}

function handleAgentMessage(data) {
    const { agent_name, message } = data;
    
    // Only update if this is the selected agent
    if (agent_name !== selectedAgent) {
        return;
    }
    
    // Display message based on role
    if (message.role === 'user') {
        displayUserMessage(message);
    } else if (message.role === 'assistant') {
        displayAssistantMessage(message);
    }
}

function displayAssistantMessage(message) {
    const contentEl = document.getElementById('last-response-content');
    contentEl.textContent = message.content;
    
    // Show the response panel
    document.querySelector('.last-response-panel').style.display = 'block';
}

async function sendMessage() {
    if (!selectedAgent) {
        alert('Please select an agent first');
        return;
    }
    
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message');
        return;
    }
    
    // Send message to agent
    const response = await fetch(`/api/agents/${selectedAgent}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    if (response.ok) {
        // Clear input
        messageInput.value = '';
        
        // Response will arrive via WebSocket
        // Show loading indicator
        const contentEl = document.getElementById('last-response-content');
        contentEl.textContent = 'Waiting for response...';
    } else {
        alert('Failed to send message');
    }
}

async function stopTaskManager() {
    if (!selectedAgent) {
        return;
    }
    
    if (!confirm('Stop TaskManager for this agent?')) {
        return;
    }
    
    const response = await fetch(`/api/agents/${selectedAgent}/interrupt`, {
        method: 'POST'
    });
    
    if (response.ok) {
        alert('TaskManager stopped');
    } else {
        alert('Failed to stop TaskManager');
    }
}

function openFullDialog() {
    if (!selectedAgent) {
        return;
    }
    
    window.open(`/agents/${selectedAgent}/dialog`, '_blank');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadAgents();
});

async function loadAgents() {
    const response = await fetch('/api/agents');
    const agents = await response.json();
    
    const listEl = document.getElementById('agents-list');
    listEl.innerHTML = '';
    
    for (const agent of agents) {
        addAgentToList(agent);
    }
}
```

---

## 9. Security Architecture

### 9.1 Localhost-Only Binding

**Requirement**: Server MUST ONLY bind to 127.0.0.1:7887.

**Implementation**:

```python
# app/config.py
class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 7887
    
    class Config:
        # NO env overrides allowed for security settings
        env_file = None
    
    def validate_security(self) -> bool:
        """Validate security requirements."""
        if self.host != "127.0.0.1":
            raise ValueError(
                f"SECURITY ERROR: Host must be 127.0.0.1 only! Current: {self.host}\n"
                "This application MUST NEVER be exposed to external networks.\n"
                "Running on 0.0.0.0 or any other address is STRICTLY PROHIBITED."
            )
        
        if self.port != 7887:
            raise ValueError(
                f"SECURITY ERROR: Port must be 7887 only! Current: {self.port}\n"
                "This is a security requirement to prevent accidental exposure."
            )
        
        return True

# Validate on module import
settings = Settings()
settings.validate_security()
```

```python
# run.py
def main() -> None:
    """Start the server with security validation."""
    # Double-check security
    if settings.host != "127.0.0.1":
        print(f"❌ CRITICAL SECURITY ERROR: Host must be 127.0.0.1!")
        print(f"   Current: {settings.host}")
        sys.exit(1)
    
    if settings.port != 7887:
        print(f"❌ CRITICAL SECURITY ERROR: Port must be 7887!")
        print(f"   Current: {settings.port}")
        sys.exit(1)
    
    print(f"🔒 SECURITY: Server bound to {settings.host}:{settings.port} ONLY")
    print("⚠️  This application MUST NOT be exposed to external networks!")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
```

### 9.2 RPyC Security

**Requirement**: RPyC connections MUST ONLY connect to localhost.

```python
# Always use 127.0.0.1 for RPyC connections
conn = rpyc.connect(
    "127.0.0.1",  # NEVER use other addresses
    rpyc_port,
    config={"allow_public_attrs": True}
)
```

### 9.3 No Authentication

**Design Decision**: No authentication/authorization because:
1. Server only binds to localhost (no external access)
2. Relies on OS-level localhost isolation
3. Simplifies architecture
4. Consistent with deep-worker agent design

**Caveat**: This application is ONLY suitable for trusted local environments. Do NOT expose to networks.

---

## 10. Testing Strategy

### 10.1 TDD Workflow

**Approach**: Write tests BEFORE implementation.

```python
# 1. Write test first
async def test_list_agents_returns_running_agents():
    """Test that list_agents returns all running agents."""
    # Setup: Start test agents
    agent1 = await start_test_agent("test-agent-1")
    agent2 = await start_test_agent("test-agent-2")
    
    # Test
    service = AgentService()
    await service.initialize()
    agents = await service.list_agents()
    
    # Assert
    assert len(agents) == 2
    assert any(a.name == "test-agent-1" for a in agents)
    assert any(a.name == "test-agent-2" for a in agents)
    
    # Cleanup
    await stop_test_agent(agent1)
    await stop_test_agent(agent2)

# 2. Run test (fails because not implemented)
# 3. Implement feature to make test pass
# 4. Refactor if needed
# 5. Test passes!
```

### 10.2 Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── unit/
│   ├── test_config.py             # Config validation
│   ├── test_agent_service.py      # AgentService unit tests
│   └── test_schemas.py            # Pydantic schema validation
├── integration/
│   ├── test_api_endpoints.py      # API integration tests
│   ├── test_websocket.py          # WebSocket tests
│   └── test_agent_communication.py # RPyC communication tests
└── fixtures/
    ├── test_agents.py             # Test agent fixtures
    └── mock_registry.py           # Mock ProcessRegistry
```

### 10.3 Key Test Cases

**Unit Tests**:

```python
# tests/unit/test_config.py
def test_config_rejects_non_localhost():
    """Config must reject non-localhost addresses."""
    with pytest.raises(ValueError, match="SECURITY ERROR"):
        Settings(host="0.0.0.0")

def test_config_rejects_wrong_port():
    """Config must reject non-7887 ports."""
    with pytest.raises(ValueError, match="SECURITY ERROR"):
        Settings(port=8000)

# tests/unit/test_agent_service.py
async def test_agent_service_discovers_agents():
    """AgentService discovers agents from registry."""
    # Mock ProcessRegistry
    mock_registry = MockRegistry([
        AgentProcessInfo(name="test-1", pid=123, ...)
    ])
    
    service = AgentService(registry=mock_registry)
    await service.initialize()
    
    agents = await service.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "test-1"
```

**Integration Tests**:

```python
# tests/integration/test_api_endpoints.py
@pytest.mark.asyncio
async def test_list_agents_endpoint(test_client, test_agent):
    """Test /api/agents endpoint returns agent list."""
    response = await test_client.get("/api/agents")
    
    assert response.status_code == 200
    agents = response.json()
    assert len(agents) > 0
    assert agents[0]["name"] == test_agent.name

@pytest.mark.asyncio
async def test_send_message_endpoint(test_client, test_agent):
    """Test sending message to agent."""
    response = await test_client.post(
        f"/api/agents/{test_agent.name}/message",
        json={"message": "Hello"}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True

# tests/integration/test_websocket.py
@pytest.mark.asyncio
async def test_websocket_receives_agent_messages(test_client, test_agent):
    """Test WebSocket receives messages from agent."""
    async with test_client.websocket_connect("/ws/agents") as ws:
        # Send message to agent
        await test_client.post(
            f"/api/agents/{test_agent.name}/message",
            json={"message": "Hello"}
        )
        
        # Should receive user message via WebSocket
        msg1 = await ws.receive_json()
        assert msg1["type"] == "agent_message"
        assert msg1["message"]["role"] == "user"
        
        # Should receive assistant response
        msg2 = await ws.receive_json()
        assert msg2["type"] == "agent_message"
        assert msg2["message"]["role"] == "assistant"
```

### 10.4 Test Fixtures

```python
# tests/conftest.py

@pytest.fixture
async def test_agent():
    """Start a test deep-worker agent."""
    agent_name = f"test-agent-{uuid.uuid4()}"
    
    # Start agent process
    agent = await start_test_agent(
        name=agent_name,
        provider="deepseek",
        model="deepseek-chat"
    )
    
    # Wait for agent to be ready
    await wait_for_agent_ready(agent_name)
    
    yield agent
    
    # Cleanup
    await stop_test_agent(agent)


@pytest.fixture
async def test_client(test_agent):
    """FastAPI test client."""
    from httpx import AsyncClient
    from app.main import app
    
    async with AsyncClient(app=app, base_url="http://127.0.0.1:7887") as client:
        yield client


@pytest.fixture
def mock_registry():
    """Mock ProcessRegistry for unit tests."""
    return MockRegistry()
```

---

## 11. Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Basic infrastructure working.

**Tasks**:
1. ✅ Project structure created
2. ✅ Config with security validation
3. ✅ Pyright strict setup
4. ✅ Pre-commit hooks
5. Implement `AgentService` skeleton
6. Implement `ProcessRegistry` client
7. Write unit tests for config and schemas
8. Setup test fixtures

**Deliverables**:
- AgentService can discover agents from ProcessRegistry
- Unit tests pass
- Pyright strict passes

### Phase 2: RPyC Communication (Week 2)

**Goal**: Establish RPyC connections to agents.

**Tasks**:
1. Implement RPyC connection logic with AsyncioServer
2. Implement message subscription
3. Test RPyC communication with real agents
4. Handle connection errors and retries
5. Write integration tests for RPyC

**Deliverables**:
- Can connect to agents via RPyC
- Can subscribe to agent messages
- Can send messages to agents
- Integration tests pass

### Phase 3: REST API (Week 3)

**Goal**: API endpoints working.

**Tasks**:
1. Implement `/api/agents` (list agents)
2. Implement `/api/agents/{name}` (get status)
3. Implement `/api/agents/{name}/message` (send message)
4. Implement `/api/agents/{name}/history` (get history)
5. Implement `/api/agents/{name}/interrupt` (stop TaskManager)
6. Write API tests

**Deliverables**:
- All API endpoints working
- API tests pass
- Postman/curl manual testing successful

### Phase 4: WebSocket (Week 4)

**Goal**: Real-time updates working.

**Tasks**:
1. Implement WebSocket endpoint
2. Implement ConnectionManager
3. Bridge AgentService callbacks to WebSocket
4. Handle WebSocket lifecycle (connect/disconnect/reconnect)
5. Write WebSocket tests

**Deliverables**:
- WebSocket connection stable
- Real-time updates working
- WebSocket tests pass

### Phase 5: UI Implementation (Week 5-6)

**Goal**: Complete web interface.

**Tasks**:
1. Create base HTML template
2. Implement agent list view
3. Implement message interface
4. Implement full dialog page (new tab)
5. Add CSS styling
6. Add JavaScript for interactivity
7. Test UI manually

**Deliverables**:
- Full UI working
- Can send messages and see responses
- Full dialog page works
- UI is responsive and user-friendly

### Phase 6: Polish & Documentation (Week 7)

**Goal**: Production-ready.

**Tasks**:
1. Add error handling and user feedback
2. Improve UI/UX
3. Write user documentation
4. Write deployment guide
5. Code review and refactoring
6. Performance testing

**Deliverables**:
- All features working smoothly
- Documentation complete
- Ready for production use

---

## 12. Open Questions

### Q1: Agent Provider Detection

**Question**: How to reliably detect which AI provider an agent is using?

**Options**:
- A: Parse agent config file
- B: Query agent via RPyC (expose provider in status)
- C: Store in ProcessRegistry

**DECISION**: Option B (query via RPyC) - User choice: "not sure - pick one yourself"

Agent knows its own provider and should expose it in status response. This provides accurate, real-time information.

**Implementation**:
```python
# Agent should expose in get_status()
{
    "success": True,
    "session_info": {
        "status": "idle",
        "provider": "claude",  # ← Expose this
        "model": "claude-sonnet-4"  # ← And this
    }
}
```

### Q2: Multi-Agent Message Display

**Question**: When multiple agents are running, how to handle messages from non-selected agents?

**Options**:
- A: Only display messages from selected agent
- B: Show notifications for messages from other agents
- C: Show all messages in a combined view

**DECISION**: Option A - Only display messages from selected agent (User choice)

Messages from non-selected agents will be filtered out in the WebSocket message handler. This keeps the UI clean and focused on the current agent.

### Q3: Agent Restart Handling

**Question**: What happens if an agent restarts while web UI is connected?

**Options**:
- A: Auto-reconnect when agent comes back
- B: Show error, require manual reconnect
- C: Remove from list until manually refreshed

**DECISION**: Option A - Auto-reconnect when agent comes back (User choice)

Background task in AgentService monitors ProcessRegistry and automatically reconnects to restarted agents, providing seamless experience.

### Q4: History Pagination

**Question**: For agents with long conversation history, how to handle pagination?

**Options**:
- A: Load all messages upfront (could be slow)
- B: Implement pagination with offset/limit
- C: Infinite scroll with lazy loading

**DECISION**: Option A - Load all messages upfront, but only on button click (User choice)

Full dialog view loads complete message history when user clicks "Full Dialog" button. Main view shows only user requests and final responses (similar to the reference web application's approach).

---

## Appendix A: File Structure

```
deep_worker_web/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app with lifespan
│   ├── config.py                  # Settings with security validation
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── agents.py              # /api/agents endpoints
│   │   ├── websocket.py           # /ws/agents WebSocket
│   │   └── pages.py               # HTML page routes
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── agent_service.py       # Core agent management service
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── agent.py               # Internal data models
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── agent_schemas.py       # Pydantic schemas for API
│   │
│   ├── templates/
│   │   ├── base.html              # Base template
│   │   ├── index.html             # Main page
│   │   └── dialog.html            # Full dialog page
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py      # Logging setup
│       └── exceptions.py          # Custom exceptions
│
├── static/
│   ├── css/
│   │   └── style.css              # Main stylesheet
│   └── js/
│       └── main.js                # Main JavaScript
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_agent_service.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_websocket.py
│   │   └── test_agent_communication.py
│   └── fixtures/
│       ├── test_agents.py
│       └── mock_registry.py
│
├── docs/
│   ├── DESIGN.md                  # This file
│   ├── API.md                     # API documentation
│   └── USER_GUIDE.md              # User guide
│
├── .gitignore
├── .pre-commit-config.yaml
├── pyrightconfig.json
├── requirements.txt
├── run.py
└── README.md
```

---

## Appendix B: Dependencies

```txt
# requirements.txt

# Web framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# WebSocket support
websockets>=12.0

# Templating
jinja2>=3.1.2

# Settings management
pydantic>=2.5.0
pydantic-settings>=2.1.0

# HTTP client (for tests)
httpx>=0.25.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Type checking (development)
pyright>=1.1.338

# Pre-commit (development)
pre-commit>=3.5.0
```

**CRITICAL**: Do NOT install rpyc or deep-worker CLI via pip! Use source imports only.

---

## Appendix C: Security Checklist

- [ ] Server ONLY binds to 127.0.0.1:7887 (validated at startup)
- [ ] Config validation prevents other addresses/ports
- [ ] RPyC connections ONLY to 127.0.0.1
- [ ] No external network exposure
- [ ] No authentication (relies on localhost isolation)
- [ ] All user inputs validated via Pydantic
- [ ] No eval() or exec() anywhere
- [ ] No shell injection vulnerabilities
- [ ] WebSocket connections authenticated (by localhost binding)
- [ ] Error messages don't leak sensitive information

---

**End of Design Document**
