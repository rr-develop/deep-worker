# Migration Plan: Deep Worker → RPyC AsyncioServer

> **✅ MIGRATION COMPLETE (April 2026).** The `arpyc` library and its directory
> have been fully removed from the repository.
> All code now uses the local `rpyc_async` source checkout with `AsyncioServer` directly.
> All references to `arpyc`, `RpycAsyncSession`, `ThreadedServer`, and
> `BgServingThread` below are **HISTORICAL** — they describe what was replaced.
> This document is kept for historical reference only. DO NOT follow any
> instructions below that reference arpyc.

**Project:** deep_worker + dwcli
**Target:** Full migration from classic RPyC/arpyc to native AsyncioServer
**Date:** 2026-03-22
**Status:** ✅ COMPLETED

---

## Executive Summary

This document outlines the complete migration plan to transition the `deep_worker` project from classic RPyC with arpyc wrappers to the native AsyncioServer implementation from the local `rpyc_async` source checkout. The migration will enable true bidirectional asynchronous communication and eliminate the performance bottlenecks and architectural limitations of the current approach.

### Key Objectives

1. **Replace ThreadedServer with AsyncioServer** - Enable persistent event loops for true bidirectional async
2. **Eliminate arpyc wrapper dependency** - Use native rpyc_async directly from source
3. **Maintain full functionality** - All existing features must work identically
4. **Direct source integration** - Use rpyc_async from its local source checkout without pip installation
5. **Enable immediate testing** - Any fixes in rpyc_async can be tested immediately in deep_worker

### Expected Benefits

| Aspect | Current | After Migration | Improvement |
|--------|---------|-----------------|-------------|
| **Bidirectional Async** | ❌ Fails (deadlock with ThreadedServer) | ✅ Works (persistent event loops) | Functional |
| **Performance (I/O)** | Baseline | 65x-156x faster | 65x-156x |
| **Latency** | ~301ms (BgServingThread) | ~1.93ms (event-driven) | 156x faster |
| **Memory/Connection** | ~8MB (thread) | ~10KB (coroutine) | 800x less |
| **Max Connections** | ~1,000 | ~10,000+ | 10x scale |
| **Architecture** | 2 threads (Main + BgServing) | 1 thread (event loop) | Simpler |

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Target Architecture Design](#2-target-architecture-design)
3. [Migration Strategy](#3-migration-strategy)
4. [Source Integration Setup](#4-source-integration-setup)
5. [Component Migration Steps](#5-component-migration-steps)
6. [Testing Strategy](#6-testing-strategy)
7. [Rollout Plan](#7-rollout-plan)
8. [Risk Assessment & Mitigation](#8-risk-assessment--mitigation)
9. [Success Criteria](#9-success-criteria)
10. [Appendix: File Mapping](#10-appendix-file-mapping)

---

## 1. Current Architecture Analysis

### 1.1 Process Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CURRENT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────┐          │
│  │  Agent Process (deep-worker)                │          │
│  │  ─────────────────────────────────────────  │          │
│  │  Main Thread:                               │          │
│  │    - asyncio event loop (agent execution)   │          │
│  │    - LangGraph/LangChain                    │          │
│  │    - SessionManager                         │          │
│  │                                             │          │
│  │  RPyC Thread (daemon):                      │          │
│  │    - rpyc.ThreadedServer                    │          │
│  │    - AgentService (exposed_* methods)       │          │
│  │    - Handles: invoke_agent, get_history,    │          │
│  │      subscribe_messages, etc.               │          │
│  │    ❌ Creates TEMPORARY event loops          │          │
│  │       (asyncio.run per async request)       │          │
│  │    ❌ DEADLOCKS on bidirectional async       │          │
│  │                                             │          │
│  │  TaskManager Thread (optional, daemon):     │          │
│  │    - Evaluates subordinate responses        │          │
│  │    - Own asyncio event loop                 │          │
│  └────────────┬────────────────────────────────┘          │
│               │                                             │
│               │ TCP/RPyC (ThreadedServer)                  │
│               │                                             │
│  ┌────────────▼────────────────────────────────┐          │
│  │  Client Process (dwcli)                     │          │
│  │  ─────────────────────────────────────────  │          │
│  │  Main Thread:                               │          │
│  │    - asyncio event loop (Textual TUI)       │          │
│  │    - RpycAsyncSession (connection)          │          │
│  │    - InteractiveApp (UI)                    │          │
│  │                                             │          │
│  │  BgServingThread (daemon):                  │          │
│  │    - Handles reverse calls (server→client)  │          │
│  │    - Polls with sleep(0.1) delays           │          │
│  │    ⚠️ ~301ms latency (polling overhead)      │          │
│  │    - Callbacks execute in BgServing thread  │          │
│  │    - Uses queue + call_from_thread()        │          │
│  └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components & RPyC/arpyc Usage

#### Server Side (Agent Process)

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py:196-221`

```python
def _start_rpyc_server(self) -> None:
    """Start RPyC server - BLOCKS in daemon thread."""
    from rpyc import ThreadedServer  # ← Classic RPyC
    from rpyc.utils.helpers import classpartial

    extended_safe_attrs = set(rpyc.core.protocol.DEFAULT_CONFIG["safe_attrs"])
    extended_safe_attrs.update(["__aiter__", "__anext__"])

    protocol_config = {
        "allow_public_attrs": True,
        "allow_pickle": True,
        "safe_attrs": extended_safe_attrs,
    }

    service_cls = classpartial(AgentService, self)

    # PROBLEM: ThreadedServer creates threads per connection
    # and uses temporary event loops for async requests
    self.rpyc_server = ThreadedServer(
        service_cls,
        hostname="127.0.0.1",
        port=self.rpyc_port,
        protocol_config=protocol_config,
    )

    self.rpyc_server.start()  # Blocking call - runs in daemon thread
```

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/rpyc_service/agent_service.py:38-268`

```python
class AgentService(rpyc.Service):
    """RPyC service for agent interaction."""

    def exposed_invoke_agent(self, message: str, stream: bool) -> dict[str, Any]:
        """Sync method → bridges to async agent."""
        # PROBLEM: Bridges async code via run_coroutine_threadsafe
        # Creates temporary event loop via asyncio.run internally
        future = asyncio.run_coroutine_threadsafe(
            self._agent_process._handle_invoke(...),
            self._event_loop
        )
        return future.result(timeout=300.0)

    def _notify_subscribers(self, message: dict[str, Any]):
        """Broadcast to all clients - synchronous calls."""
        for subscriber in subscribers:
            subscriber.on_message(message)  # Reverse call
            # PROBLEM: If on_message is async, this fails with ThreadedServer
```

#### Client Side (dwcli)

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/cli/interactive.py:339-360`

```python
async def _connect_via_rpyc(self) -> None:
    """Connect to agent via RPyC."""
    from arpyc import RpycAsyncSession  # ← arpyc wrapper (BgServingThread)

    config = {
        "allow_public_attrs": True,
        "allow_pickle": True,
        "sync_request_timeout": 300.0,
    }

    # PROBLEM: RpycAsyncSession uses BgServingThread with polling delays
    self.rpyc_session = RpycAsyncSession(
        "127.0.0.1",
        self.rpyc_port,
        config=config
    )

    await self.rpyc_session.connect()

    # Subscribe to messages
    callback_obj = type("MessageCallback", (), {
        "on_message": lambda self, msg: handle_message(msg)
    })()

    await self.rpyc_session.async_call(
        self.rpyc_session.root.subscribe_messages,
        callback_obj
    )
```

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/client/agent_client.py:100-150`

```python
class AgentClient:
    """Client SDK wrapping RPyC connection."""

    def __init__(self, rpyc_session: RpycAsyncSession):
        self.rpyc_session = rpyc_session

    async def invoke_agent(self, message: str, stream: bool = False):
        """Invoke agent - fire-and-forget in streaming mode."""
        # PROBLEM: Uses arpyc wrapper's async_call
        return await self.rpyc_session.async_call(
            self.rpyc_session.root.invoke_agent,
            message,
            stream
        )
```

### 1.3 Current Dependencies

**arpyc Wrapper Location:** the `arpyc/` directory at the repository root

**Files:**
- `arpyc/arpyc/session.py` - RpycAsyncSession (BgServingThread approach)
- `arpyc/arpyc/session_native.py` - RpycAsyncSessionNative (event-driven, 156x faster)
- `arpyc/arpyc/helpers.py` - await_rpyc utility
- `arpyc/arpyc/proxy.py` - Proxy helpers
- `arpyc/arpyc/exceptions.py` - Custom exceptions

**Status:** arpyc is a LOCAL wrapper around classic RPyC 5.0, not the new rpyc_async.

### 1.4 Identified Limitations

#### Problem 1: ThreadedServer Bidirectional Async Deadlock

**Root Cause:**
```python
# In rpyc.core.protocol.py (classic RPyC 5.0)
def _dispatch_request(self, seq, handler, args):
    if needs_async:
        import asyncio
        # PROBLEM: Creates TEMPORARY event loop per request
        asyncio.run(self._dispatch_request_async(seq, args))
        # Loop exits after request completes
```

**Consequence:**
- Agent calls client async callback
- Client needs to respond
- **DEADLOCK:** Agent's temporary loop already exited, cannot receive response
- Timeout after ~30 seconds

#### Problem 2: BgServingThread Polling Delays

**Issue:** `arpyc.session.RpycAsyncSession` uses `BgServingThread` with `sleep(0.1)`:

```python
# In rpyc.utils.helpers.BgServingThread
def run(self):
    while not self.stopped:
        self.conn.serve(0.1)  # Process for 0.1s
        time.sleep(0.1)       # Sleep for 0.1s
        # 100ms delay PER CYCLE = high latency
```

**Consequence:**
- Average latency: ~301ms for recursive calls
- Callbacks execute in separate thread (requires thread-safe queue)

#### Problem 3: arpyc is Not rpyc_async

**Current State:**
- `arpyc/` is a wrapper around **classic RPyC 5.0** (uses ThreadedServer internally)
- `rpyc_async/` is **new RPyC 5.1** with AsyncioServer implementation
- These are **completely different** codebases

**Consequence:**
- Cannot simply import from arpyc and get AsyncioServer benefits
- Need full replacement of classic RPyC with rpyc_async

---

## 2. Target Architecture Design

### 2.1 Target Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   TARGET ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────┐          │
│  │  Agent Process (deep-worker)                │          │
│  │  ─────────────────────────────────────────  │          │
│  │  Main Thread / Event Loop:                  │          │
│  │    ┌──────────────────────────────────────┐ │          │
│  │    │ AsyncioServer (rpyc_async)           │ │          │
│  │    │ ───────────────────────────────────  │ │          │
│  │    │ - Native asyncio integration         │ │          │
│  │    │ - loop.add_reader() for I/O          │ │          │
│  │    │ - Coroutines per connection          │ │          │
│  │    │ - PERSISTENT event loop              │ │          │
│  │    │ ✅ Bidirectional async WORKS          │ │          │
│  │    └──────────────────────────────────────┘ │          │
│  │                                             │          │
│  │    ┌──────────────────────────────────────┐ │          │
│  │    │ AgentService (rpyc.Service)          │ │          │
│  │    │ ───────────────────────────────────  │ │          │
│  │    │ - All methods are async              │ │          │
│  │    │ - exposed_invoke_agent (async)       │ │          │
│  │    │ - exposed_get_history (async)        │ │          │
│  │    │ - exposed_subscribe_messages (async) │ │          │
│  │    │ - No thread bridging needed!         │ │          │
│  │    └──────────────────────────────────────┘ │          │
│  │                                             │          │
│  │    ┌──────────────────────────────────────┐ │          │
│  │    │ Agent Core (LangGraph/LangChain)     │ │          │
│  │    │ ───────────────────────────────────  │ │          │
│  │    │ - Same async implementation          │ │          │
│  │    │ - No changes needed                  │ │          │
│  │    └──────────────────────────────────────┘ │          │
│  │                                             │          │
│  │  TaskManager (optional, same thread):      │          │
│  │    - asyncio.create_task() for evaluation  │          │
│  │    - All in shared event loop              │          │
│  └────────────┬────────────────────────────────┘          │
│               │                                             │
│               │ TCP/RPyC (AsyncioServer)                   │
│               │ - Persistent event loops                   │
│               │ - Event-driven I/O                         │
│               │                                             │
│  ┌────────────▼────────────────────────────────┐          │
│  │  Client Process (dwcli)                     │          │
│  │  ─────────────────────────────────────────  │          │
│  │  Main Thread / Event Loop:                  │          │
│  │    ┌──────────────────────────────────────┐ │          │
│  │    │ rpyc.connect() with asyncio          │ │          │
│  │    │ ───────────────────────────────────  │ │          │
│  │    │ - conn.enable_asyncio_serving()      │ │          │
│  │    │ - loop.add_reader() integration      │ │          │
│  │    │ - Event-driven callbacks             │ │          │
│  │    │ - NO BgServingThread                 │ │          │
│  │    │ ✅ ~1.93ms latency (156x faster!)     │ │          │
│  │    └──────────────────────────────────────┘ │          │
│  │                                             │          │
│  │    ┌──────────────────────────────────────┐ │          │
│  │    │ InteractiveApp (Textual TUI)         │ │          │
│  │    │ ───────────────────────────────────  │ │          │
│  │    │ - Same UI implementation             │ │          │
│  │    │ - Callbacks in event loop            │ │          │
│  │    │ - Can use asyncio.create_task()!     │ │          │
│  │    └──────────────────────────────────────┘ │          │
│  └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Source Integration Pattern

**Requirement:** Use rpyc_async directly from source for immediate testability.

**Directory Structure:**
```
<checkout root>/
├── <main project>/                # Main project
│   ├── deep_worker/
│   │   └── deep_worker_cli/
│   │       ├── pyproject.toml     # Add rpyc_async as path dependency
│   │       └── deep_worker_cli/
│   │           ├── process/
│   │           │   └── agent_process.py  # Use AsyncioServer
│   │           ├── rpyc_service/
│   │           │   └── agent_service.py  # All async methods
│   │           └── cli/
│   │               └── interactive.py     # Use native rpyc_async
│   └── arpyc/                     # DELETE after migration
│
└── rpyc_async/                    # Source of truth
    ├── rpyc/
    │   ├── core/
    │   │   ├── protocol.py         # enable_asyncio_serving()
    │   │   ├── async_.py           # AsyncResult with __await__
    │   │   └── async_handlers.py   # _handle_async_call
    │   └── utils/
    │       └── async_server.py     # AsyncioServer
    └── examples/
        └── bidirectional_async/     # Reference implementations
```

**Integration Method:** Python path dependency in `pyproject.toml`:

```toml
[tool.poetry.dependencies]
# Use local rpyc_async from source (NOT pip!)
rpyc = { path = "../../../../rpyc_async/", develop = true }
```

**Benefits:**
1. ✅ Any fixes in the local `rpyc_async` checkout immediately available
2. ✅ Can modify rpyc_async and test in deep_worker instantly
3. ✅ No pip install needed
4. ✅ Shared development workflow
5. ✅ Source control for both projects

### 2.3 API Changes

#### Server Side API Changes

**Before (classic RPyC + ThreadedServer):**
```python
from rpyc import ThreadedServer
import rpyc

class AgentService(rpyc.Service):
    def exposed_invoke_agent(self, message: str, stream: bool):
        """Sync method - bridges to async."""
        future = asyncio.run_coroutine_threadsafe(
            self._agent_process._handle_invoke(...),
            self._event_loop
        )
        return future.result(timeout=300.0)

# Startup
service_cls = classpartial(AgentService, agent_process)
server = ThreadedServer(service_cls, port=port)
server.start()  # Blocking
```

**After (rpyc_async + AsyncioServer):**
```python
from rpyc.utils.async_server import AsyncioServer
import rpyc

class AgentService(rpyc.Service):
    async def exposed_invoke_agent(self, message: str, stream: bool):
        """Native async method - no bridging needed!"""
        return await self._agent_process._handle_invoke(
            message=message,
            stream=stream,
            notify_callback=self._notify_subscribers
        )

# Startup (in async context)
async def start():
    service_cls = classpartial(AgentService, agent_process)
    server = AsyncioServer(service_cls, port=port)
    await server.serve_forever()  # Async
```

#### Client Side API Changes

**Before (arpyc wrapper):**
```python
from arpyc import RpycAsyncSession

async def connect():
    session = RpycAsyncSession("localhost", port)
    await session.connect()

    # Uses async_call wrapper
    result = await session.async_call(
        session.root.invoke_agent,
        message,
        stream
    )

    await session.disconnect()
```

**After (native rpyc_async):**
```python
import rpyc

async def connect():
    # Standard rpyc.connect
    conn = rpyc.connect("localhost", port, config=config)

    # Enable asyncio serving (persistent event loop)
    conn.enable_asyncio_serving()

    try:
        # Direct async call - no wrapper!
        result = await conn.root.invoke_agent(message, stream)
    finally:
        conn.disable_asyncio_serving()
        conn.close()
```

**Key Difference:** No custom wrapper needed! Native RPyC 5.1 supports async directly.

---

## 3. Migration Strategy

### 3.1 Migration Phases

#### Phase 1: Source Integration Setup (1-2 hours)
- Setup path dependency from deep_worker to rpyc_async
- Verify import structure
- Run basic import tests

#### Phase 2: Server Migration (4-6 hours)
- Replace ThreadedServer with AsyncioServer
- Convert AgentService methods to async
- Update agent_process.py startup flow
- Test server functionality

#### Phase 3: Client Migration (4-6 hours)
- Replace RpycAsyncSession with native rpyc.connect
- Update enable_asyncio_serving integration
- Adapt callback handling for event loop
- Test client functionality

#### Phase 4: Integration Testing (6-8 hours)
- End-to-end streaming tests
- Bidirectional async tests
- Performance benchmarking
- Stress testing (multiple clients)

#### Phase 5: Cleanup & Documentation (2-3 hours)
- Remove arpyc dependency
- Update documentation
- Create migration notes
- Code review

**Total Estimated Time:** 17-25 hours

### 3.2 Feature Parity Matrix

| Feature | Current | After Migration | Status |
|---------|---------|-----------------|--------|
| **Agent Invocation** | ✅ Works | ✅ Works | Must maintain |
| **Streaming Responses** | ✅ Works | ✅ Works (faster!) | Must maintain |
| **Message History** | ✅ Works | ✅ Works | Must maintain |
| **Message Subscription** | ✅ Works | ✅ Works (event-driven!) | Must maintain |
| **Bidirectional Callbacks** | ⚠️ Limited (can deadlock) | ✅ Full support | Enhanced |
| **Multiple Clients** | ✅ Works | ✅ Works (more scalable) | Enhanced |
| **TaskManager Integration** | ✅ Works | ✅ Works (same thread!) | Simplified |
| **HITL Approvals** | ✅ Works | ✅ Works | Must maintain |
| **Slash Commands** | ✅ Works | ✅ Works | Must maintain |
| **Agent Interruption** | ✅ Works | ✅ Works | Must maintain |
| **Graceful Shutdown** | ✅ Works | ✅ Works | Must maintain |

### 3.3 Backward Compatibility

**Policy:** Maintain 100% external API compatibility.

**Internal Changes Allowed:**
- Server implementation (ThreadedServer → AsyncioServer)
- Connection management (BgServingThread → event loop)
- Service method signatures (sync → async internally)

**External API Must Not Change:**
- CLI commands (`deep-worker`, `dwcli`)
- Configuration files
- Registry format
- Message formats
- Exposed RPC methods (names, parameters, return types)

---

## 4. Source Integration Setup

### 4.1 Dependency Configuration

**File:** `deep_worker/deep_worker_cli/pyproject.toml`

**Current:**
```toml
[tool.poetry.dependencies]
python = "^3.11"
rpyc = "^5.0.0"  # Classic RPyC from PyPI
langchain = "^0.1.0"
# ...
```

**After Migration:**
```toml
[tool.poetry.dependencies]
python = "^3.11"

# Use local rpyc_async from source (develop mode)
rpyc = { path = "../../../../rpyc_async/", develop = true }

langchain = "^0.1.0"
# ...

[tool.poetry.dev-dependencies]
# Development dependencies remain the same
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
# ...
```

**Note:** `develop = true` creates a symlink, so changes in rpyc_async are immediately reflected.

### 4.2 Import Path Verification

**Test Script:** `deep_worker/deep_worker_cli/scripts/verify_rpyc_async_integration.py`

```python
"""Verify rpyc_async source integration."""

import sys
from pathlib import Path

def verify_rpyc_async_integration():
    """Verify that rpyc is imported from correct source."""
    import rpyc
    from rpyc.utils.async_server import AsyncioServer

    # Check that AsyncioServer is available
    assert hasattr(rpyc.utils, 'async_server'), \
        "AsyncioServer module not found - wrong rpyc version?"

    # Check rpyc location
    rpyc_path = Path(rpyc.__file__).resolve()
    expected_path = Path(__file__).resolve().parents[5] / "rpyc_async" / "rpyc"

    assert expected_path in rpyc_path.parents, \
        f"rpyc imported from wrong location!\n" \
        f"Expected: {expected_path}\n" \
        f"Actual: {rpyc_path}"

    # Check protocol version
    from rpyc.core import consts
    assert consts.PROTOCOL_VERSION >= (5, 1), \
        f"Wrong protocol version: {consts.PROTOCOL_VERSION}"

    print("✅ rpyc_async integration verified!")
    print(f"   rpyc location: {rpyc_path}")
    print(f"   Protocol version: {consts.PROTOCOL_VERSION}")

    return True

if __name__ == "__main__":
    try:
        verify_rpyc_async_integration()
        sys.exit(0)
    except AssertionError as e:
        print(f"❌ Integration verification failed: {e}")
        sys.exit(1)
```

### 4.3 Environment Setup Steps

**Step 1:** Install rpyc_async in development mode

```bash
cd deep_worker/deep_worker_cli/

# Update pyproject.toml (as shown above)

# Install dependencies
poetry install

# Verify integration
poetry run python scripts/verify_rpyc_async_integration.py
```

**Step 2:** Verify AsyncioServer is available

```bash
poetry run python -c "from rpyc.utils.async_server import AsyncioServer; print('✅ AsyncioServer imported successfully')"
```

**Step 3:** Run basic connection test

```python
# test_basic_asyncio_server.py
import asyncio
import rpyc
from rpyc.utils.async_server import AsyncioServer

class TestService(rpyc.Service):
    async def exposed_hello(self):
        return "Hello from AsyncioServer!"

async def test_basic():
    # Start server
    server = AsyncioServer(TestService, port=0)
    await server.start()
    port = server.listener.getsockname()[1]

    # Connect client
    conn = rpyc.connect("localhost", port)
    conn.enable_asyncio_serving()

    try:
        result = await conn.root.hello()
        print(f"✅ Basic test passed: {result}")
    finally:
        conn.disable_asyncio_serving()
        conn.close()
        await server.close()

if __name__ == "__main__":
    asyncio.run(test_basic())
```

---

## 5. Component Migration Steps

### 5.1 Server Side Migration

#### 5.1.1 AgentProcess - Replace ThreadedServer with AsyncioServer

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/process/agent_process.py`

**Changes Required:**

**BEFORE:**
```python
def _start_rpyc_server(self) -> None:
    """Start RPyC server in daemon thread."""
    from rpyc import ThreadedServer
    from rpyc.utils.helpers import classpartial

    protocol_config = { ... }
    service_cls = classpartial(AgentService, self)

    self.rpyc_server = ThreadedServer(
        service_cls,
        hostname="127.0.0.1",
        port=self.rpyc_port,
        protocol_config=protocol_config,
    )

    # Start in daemon thread
    def run_server():
        self.rpyc_server.start()  # Blocking

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

async def start(self):
    """Agent startup flow."""
    # ... initialization ...
    self._start_rpyc_server()  # Starts thread
    await self._register()

    # Event loop
    while self.running:
        await asyncio.sleep(1.0)
```

**AFTER:**
```python
async def _start_rpyc_server(self) -> None:
    """Start RPyC AsyncioServer - native asyncio integration."""
    from rpyc.utils.async_server import AsyncioServer
    from rpyc.utils.helpers import classpartial

    protocol_config = {
        "allow_public_attrs": True,
        "allow_pickle": True,
        "safe_attrs": extended_safe_attrs,
    }

    service_cls = classpartial(AgentService, self)

    # Create AsyncioServer
    self.rpyc_server = AsyncioServer(
        service_cls,
        hostname="127.0.0.1",
        port=self.rpyc_port,
        protocol_config=protocol_config,
    )

    # Start server (non-blocking)
    await self.rpyc_server.start()

    # Get actual port (if port was 0)
    if self.rpyc_port == 0:
        self.rpyc_port = self.rpyc_server.listener.getsockname()[1]

    logger.info(f"AsyncioServer started on port {self.rpyc_port}")

async def start(self):
    """Agent startup flow - fully async."""
    # ... initialization ...

    # Start AsyncioServer (same thread, same event loop!)
    await self._start_rpyc_server()

    await self._register()

    # Event loop - server runs in background via accept loop
    while self.running:
        await asyncio.sleep(1.0)

async def stop(self):
    """Graceful shutdown."""
    self.running = False

    if self.rpyc_server:
        await self.rpyc_server.close()

    # ... rest of cleanup ...
```

**Key Changes:**
1. ✅ `_start_rpyc_server()` is now async
2. ✅ No daemon thread - runs in main event loop
3. ✅ `await server.start()` instead of `server.start()` (blocking)
4. ✅ `await server.close()` in cleanup

#### 5.1.2 AgentService - Convert Methods to Async

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/rpyc_service/agent_service.py`

**Changes Required:**

**BEFORE:**
```python
class AgentService(rpyc.Service):
    def __init__(self, agent_process: "AgentProcess"):
        self._agent_process = agent_process
        self._event_loop = agent_process._event_loop
        self._message_subscribers: set = set()
        self._subscribers_lock = threading.Lock()

    def exposed_invoke_agent(self, message: str, stream: bool) -> dict[str, Any]:
        """Sync method - bridges to async."""
        command = {"message": message, "stream": stream}

        # Bridge to async via run_coroutine_threadsafe
        future = asyncio.run_coroutine_threadsafe(
            self._agent_process._handle_invoke(
                command,
                notify_callback=self._notify_subscribers
            ),
            self._event_loop
        )

        return future.result(timeout=300.0)

    def exposed_subscribe_messages(self, callback) -> bool:
        """Sync method - register callback."""
        with self._subscribers_lock:
            self._message_subscribers.add(callback)
        return True

    def _notify_subscribers(self, message: dict[str, Any]):
        """Broadcast - synchronous calls."""
        with self._subscribers_lock:
            subscribers = list(self._message_subscribers)

        for subscriber in subscribers:
            try:
                subscriber.on_message(message)  # Sync call
            except Exception as e:
                logger.error(f"Subscriber notification failed: {e}")
```

**AFTER:**
```python
class AgentService(rpyc.Service):
    def __init__(self, agent_process: "AgentProcess"):
        self._agent_process = agent_process
        # NO event_loop reference needed - we're in the loop!
        self._message_subscribers: set = set()
        self._subscribers_lock = asyncio.Lock()  # Async lock!

    async def exposed_invoke_agent(self, message: str, stream: bool) -> dict[str, Any]:
        """Native async method - no bridging needed!"""
        command = {"message": message, "stream": stream}

        # Direct async call - we're already in the event loop!
        return await self._agent_process._handle_invoke(
            command,
            notify_callback=self._notify_subscribers
        )

    async def exposed_subscribe_messages(self, callback) -> bool:
        """Async method - register callback."""
        async with self._subscribers_lock:
            self._message_subscribers.add(callback)
        return True

    async def _notify_subscribers(self, message: dict[str, Any]):
        """Broadcast - async calls."""
        async with self._subscribers_lock:
            subscribers = list(self._message_subscribers)

        # Notify all subscribers concurrently
        tasks = []
        for subscriber in subscribers:
            # Check if subscriber has async on_message
            if hasattr(subscriber, 'on_message'):
                method = subscriber.on_message
                if inspect.iscoroutinefunction(method):
                    # Async callback - await it
                    tasks.append(asyncio.create_task(method(message)))
                else:
                    # Sync callback - call directly (or run in executor)
                    try:
                        method(message)
                    except Exception as e:
                        logger.error(f"Subscriber notification failed: {e}")

        # Wait for all async callbacks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Async subscriber failed: {result}")
```

**All Other Exposed Methods:**

Apply same pattern to:
- `exposed_get_history()` → `async def exposed_get_history()`
- `exposed_get_status()` → `async def exposed_get_status()`
- `exposed_unsubscribe_messages()` → `async def exposed_unsubscribe_messages()`
- `exposed_handle_command()` → `async def exposed_handle_command()`
- `exposed_respond_to_approval()` → `async def exposed_respond_to_approval()`
- `exposed_interrupt_agent()` → `async def exposed_interrupt_agent()`
- `exposed_shutdown()` → `async def exposed_shutdown()`
- `exposed_get_metadata()` → `async def exposed_get_metadata()`

**Key Changes:**
1. ✅ All `exposed_*` methods are now `async def`
2. ✅ No thread bridging via `run_coroutine_threadsafe`
3. ✅ Use `asyncio.Lock()` instead of `threading.Lock()`
4. ✅ Support async callbacks in `_notify_subscribers`
5. ✅ Use `asyncio.create_task()` for concurrent notifications

### 5.2 Client Side Migration

#### 5.2.1 InteractiveApp - Replace RpycAsyncSession

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/cli/interactive.py`

**Changes Required:**

**BEFORE:**
```python
from arpyc import RpycAsyncSession

class InteractiveApp(App):
    def __init__(self, agent_name: str, ...):
        self.rpyc_session: RpycAsyncSession | None = None
        # ...

    async def _connect_via_rpyc(self) -> None:
        """Connect using arpyc wrapper."""
        config = {
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 300.0,
        }

        # Create arpyc session
        self.rpyc_session = RpycAsyncSession(
            "127.0.0.1",
            self.rpyc_port,
            config=config
        )

        # Connect
        await self.rpyc_session.connect()

        # Subscribe to messages
        callback_obj = type("MessageCallback", (), {
            "on_message": lambda self, msg: handle_message(msg)
        })()

        await self.rpyc_session.async_call(
            self.rpyc_session.root.subscribe_messages,
            callback_obj
        )

    async def _load_history(self) -> None:
        """Load conversation history."""
        response = await self.rpyc_session.async_call(
            self.rpyc_session.root.get_history,
            limit=50,
            offset=0
        )
        # ... process response ...
```

**AFTER:**
```python
import rpyc
import asyncio

class InteractiveApp(App):
    def __init__(self, agent_name: str, ...):
        self.rpyc_conn = None  # Native rpyc connection
        # ...

    async def _connect_via_rpyc(self) -> None:
        """Connect using native rpyc_async."""
        config = {
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 300.0,
        }

        # Create native RPyC connection
        self.rpyc_conn = rpyc.connect(
            "127.0.0.1",
            self.rpyc_port,
            config=config
        )

        # Enable asyncio serving (CRITICAL!)
        self.rpyc_conn.enable_asyncio_serving()

        # Subscribe to messages - direct async call!
        callback_obj = type("MessageCallback", (), {
            "on_message": lambda self, msg: asyncio.create_task(handle_message(msg))
        })()

        # No wrapper needed - direct async call
        await self.rpyc_conn.root.subscribe_messages(callback_obj)

    async def _load_history(self) -> None:
        """Load conversation history."""
        # Direct async call - no wrapper
        response = await self.rpyc_conn.root.get_history(
            limit=50,
            offset=0
        )
        # ... process response ...

    async def _disconnect(self) -> None:
        """Cleanup connection."""
        if self.rpyc_conn:
            self.rpyc_conn.disable_asyncio_serving()
            self.rpyc_conn.close()
            self.rpyc_conn = None
```

**Key Changes:**
1. ✅ Replace `RpycAsyncSession` with `rpyc.connect()`
2. ✅ Call `conn.enable_asyncio_serving()` after connection
3. ✅ Direct `await conn.root.method()` - no `async_call` wrapper
4. ✅ Use `asyncio.create_task()` in callbacks (now in event loop!)
5. ✅ Call `disable_asyncio_serving()` before closing

#### 5.2.2 AgentClient - Update SDK

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/client/agent_client.py`

**BEFORE:**
```python
from arpyc import RpycAsyncSession

class AgentClient:
    def __init__(self, rpyc_session: RpycAsyncSession):
        self.rpyc_session = rpyc_session

    async def invoke_agent(self, message: str, stream: bool = False):
        """Invoke agent via arpyc wrapper."""
        return await self.rpyc_session.async_call(
            self.rpyc_session.root.invoke_agent,
            message,
            stream
        )

    async def get_history(self, limit: int = 50, offset: int = 0):
        """Get history via arpyc wrapper."""
        return await self.rpyc_session.async_call(
            self.rpyc_session.root.get_history,
            limit,
            offset
        )
```

**AFTER:**
```python
import rpyc

class AgentClient:
    def __init__(self, rpyc_conn):
        """Initialize with native rpyc connection."""
        self.rpyc_conn = rpyc_conn

    async def invoke_agent(self, message: str, stream: bool = False):
        """Invoke agent - native async call."""
        return await self.rpyc_conn.root.invoke_agent(message, stream)

    async def get_history(self, limit: int = 50, offset: int = 0):
        """Get history - native async call."""
        return await self.rpyc_conn.root.get_history(limit, offset)

    async def subscribe_messages(self, callback):
        """Subscribe to messages - native async call."""
        return await self.rpyc_conn.root.subscribe_messages(callback)

    async def handle_command(self, command: str):
        """Handle slash command - native async call."""
        return await self.rpyc_conn.root.handle_command(command)
```

**Key Changes:**
1. ✅ Accept `rpyc_conn` instead of `RpycAsyncSession`
2. ✅ Direct `await conn.root.method()` calls
3. ✅ No wrapper logic needed

#### 5.2.3 Callback Handling Updates

**Issue:** With native rpyc_async, callbacks execute in the event loop (not BgServingThread).

**BEFORE (thread-based callbacks):**
```python
def handle_message(msg: dict[str, Any]):
    """Callback - runs in BgServingThread."""
    # Thread-safe queue
    msg_queue.put(msg)

def _worker_thread():
    """Worker processes queue."""
    while True:
        msg = msg_queue.get()
        # Bridge to Textual event loop
        self.call_from_thread(self._handle_realtime_message, msg)
```

**AFTER (event loop callbacks):**
```python
async def handle_message(msg: dict[str, Any]):
    """Callback - runs in EVENT LOOP! (can use asyncio)"""
    # Direct call - we're already in the loop!
    await self._handle_realtime_message(msg)

# OR: Create task if you don't want to block callback
def handle_message(msg: dict[str, Any]):
    """Callback - runs in EVENT LOOP."""
    # Create task to handle message asynchronously
    asyncio.create_task(self._handle_realtime_message(msg))
```

**Key Changes:**
1. ✅ No thread queue needed
2. ✅ No `call_from_thread()` needed
3. ✅ Can use `asyncio.create_task()` directly
4. ✅ Callbacks can be async

### 5.3 TaskManager Integration

**Current Implementation:** TaskManager runs in separate thread with own event loop.

**Target Implementation:** TaskManager runs as coroutine in shared event loop.

**File:** `deep_worker/deep_worker_cli/deep_worker_cli/task_manager/task_manager.py`

**BEFORE:**
```python
def _start_manager_thread(self):
    """Start manager in separate thread."""
    def run_manager():
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._manager_loop())

    thread = threading.Thread(target=run_manager, daemon=True)
    thread.start()
```

**AFTER:**
```python
async def start(self):
    """Start manager as coroutine in shared event loop."""
    # Create manager loop task
    self._manager_task = asyncio.create_task(self._manager_loop())

async def stop(self):
    """Stop manager task."""
    if self._manager_task:
        self._manager_task.cancel()
        try:
            await self._manager_task
        except asyncio.CancelledError:
            pass
```

**Key Changes:**
1. ✅ No separate thread
2. ✅ Use `asyncio.create_task()` instead of `Thread()`
3. ✅ All in shared event loop
4. ✅ Queues remain `asyncio.Queue` (no change)

---

## 6. Testing Strategy

### 6.1 Test Phases

#### Phase 1: Unit Tests (Per-Component)

**Server Tests:**
```python
# test_async_server_migration.py

import pytest
import asyncio
from rpyc.utils.async_server import AsyncioServer
from deep_worker_cli.rpyc_service.agent_service import AgentService

@pytest.mark.asyncio
async def test_asyncio_server_starts():
    """Test that AsyncioServer starts successfully."""
    server = AsyncioServer(AgentService, port=0)
    await server.start()

    assert server.listener is not None
    port = server.listener.getsockname()[1]
    assert port > 0

    await server.close()

@pytest.mark.asyncio
async def test_exposed_methods_are_async():
    """Test that all exposed methods are async."""
    import inspect

    service = AgentService(mock_agent_process)

    for name in dir(service):
        if name.startswith("exposed_"):
            method = getattr(service, name)
            assert inspect.iscoroutinefunction(method), \
                f"{name} must be async"
```

**Client Tests:**
```python
# test_native_rpyc_client.py

import pytest
import asyncio
import rpyc

@pytest.mark.asyncio
async def test_native_connection():
    """Test native rpyc connection with asyncio."""
    # Start test server
    server = start_test_server()

    # Connect client
    conn = rpyc.connect("localhost", server.port)
    conn.enable_asyncio_serving()

    try:
        result = await conn.root.test_method()
        assert result == "test"
    finally:
        conn.disable_asyncio_serving()
        conn.close()

@pytest.mark.asyncio
async def test_async_callbacks():
    """Test async callbacks work correctly."""
    results = []

    async def async_callback(value):
        await asyncio.sleep(0.01)
        results.append(value)
        return value * 2

    conn = rpyc.connect("localhost", server.port)
    conn.enable_asyncio_serving()

    try:
        # Server should be able to call our async callback
        result = await conn.root.call_with_callback(async_callback, 5)
        assert result == 10
        assert results == [5]
    finally:
        conn.disable_asyncio_serving()
        conn.close()
```

#### Phase 2: Integration Tests

**End-to-End Flow:**
```python
# test_e2e_migration.py

import pytest
import asyncio

@pytest.mark.asyncio
async def test_full_agent_interaction():
    """Test complete agent lifecycle with AsyncioServer."""
    # 1. Start agent with AsyncioServer
    agent = AgentProcess(name="test-agent")
    await agent.start()

    # 2. Connect client with native rpyc
    conn = rpyc.connect("localhost", agent.rpyc_port)
    conn.enable_asyncio_serving()

    try:
        # 3. Invoke agent
        response = await conn.root.invoke_agent(
            "What is 2+2?",
            stream=False
        )
        assert response["status"] == "success"

        # 4. Get history
        history = await conn.root.get_history(limit=10)
        assert len(history["messages"]) > 0
    finally:
        conn.disable_asyncio_serving()
        conn.close()
        await agent.stop()

@pytest.mark.asyncio
async def test_streaming_with_callbacks():
    """Test streaming mode with real-time callbacks."""
    agent = AgentProcess(name="test-agent")
    await agent.start()

    messages_received = []

    async def on_message(msg):
        messages_received.append(msg)

    conn = rpyc.connect("localhost", agent.rpyc_port)
    conn.enable_asyncio_serving()

    try:
        # Subscribe
        callback = type("Callback", (), {"on_message": on_message})()
        await conn.root.subscribe_messages(callback)

        # Invoke with streaming
        await conn.root.invoke_agent("Tell me a joke", stream=True)

        # Wait for messages
        await asyncio.sleep(2.0)

        # Should have received multiple chunks
        assert len(messages_received) > 1
        assert any(msg["type"] == "assistant_chunk" for msg in messages_received)
    finally:
        conn.disable_asyncio_serving()
        conn.close()
        await agent.stop()
```

#### Phase 3: Bidirectional Async Tests

**Deep Recursion Test:**
```python
# test_bidirectional_async.py

@pytest.mark.asyncio
async def test_recursive_callbacks():
    """Test deep recursive bidirectional async calls."""

    class RecursiveService(rpyc.Service):
        async def exposed_recursive_call(self, callback, depth):
            if depth == 0:
                return "base case"

            # Call client callback (async)
            result = await callback(depth)

            # Client calls back to us
            # This should NOT deadlock!
            return f"{result} -> depth {depth}"

    server = AsyncioServer(RecursiveService, port=0)
    await server.start()

    conn = rpyc.connect("localhost", server.port)
    conn.enable_asyncio_serving()

    try:
        async def client_callback(depth):
            # Client calls back to server
            return await conn.root.recursive_call(client_callback, depth - 1)

        # Start recursion (depth=5)
        result = await conn.root.recursive_call(client_callback, 5)

        # Should complete without deadlock!
        assert "base case" in result
    finally:
        conn.disable_asyncio_serving()
        conn.close()
        await server.close()
```

#### Phase 4: Performance Benchmarks

**Latency Benchmark:**
```python
# test_performance_migration.py

import pytest
import time

@pytest.mark.asyncio
async def test_latency_improvement():
    """Verify latency improvement vs old implementation."""

    # Setup AsyncioServer
    server = AsyncioServer(TestService, port=0)
    await server.start()

    conn = rpyc.connect("localhost", server.port)
    conn.enable_asyncio_serving()

    try:
        # Measure latency for 100 calls
        latencies = []
        for _ in range(100):
            start = time.time()
            await conn.root.simple_call()
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)

        # Should be < 10ms (vs ~301ms with BgServingThread)
        assert avg_latency < 0.01, \
            f"Latency too high: {avg_latency*1000:.2f}ms"

        print(f"✅ Average latency: {avg_latency*1000:.2f}ms")
    finally:
        conn.disable_asyncio_serving()
        conn.close()
        await server.close()
```

**Throughput Benchmark:**
```python
@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent request handling."""
    server = AsyncioServer(TestService, port=0)
    await server.start()

    async def make_request(i):
        conn = rpyc.connect("localhost", server.port)
        conn.enable_asyncio_serving()
        try:
            return await conn.root.slow_operation(i)
        finally:
            conn.disable_asyncio_serving()
            conn.close()

    # 100 concurrent requests
    start = time.time()
    results = await asyncio.gather(*[make_request(i) for i in range(100)])
    elapsed = time.time() - start

    # Should complete in ~1-2 seconds (vs ~10+ with ThreadedServer)
    assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"
    print(f"✅ 100 concurrent requests in {elapsed:.2f}s")

    await server.close()
```

### 6.2 Test Execution Plan

**Stage 1: Source Integration Verification**
```bash
cd deep_worker/deep_worker_cli/

# Verify rpyc_async imports
poetry run python scripts/verify_rpyc_async_integration.py

# Run import tests
poetry run pytest tests/test_imports.py -v
```

**Stage 2: Server Component Tests**
```bash
# Test AsyncioServer integration
poetry run pytest tests/unit_tests/process/test_agent_process_async_server.py -v

# Test AgentService async methods
poetry run pytest tests/unit_tests/rpyc_service/test_agent_service_async.py -v
```

**Stage 3: Client Component Tests**
```bash
# Test native rpyc client
poetry run pytest tests/unit_tests/cli/test_interactive_native_rpyc.py -v

# Test AgentClient
poetry run pytest tests/unit_tests/client/test_agent_client_native.py -v
```

**Stage 4: Integration Tests**
```bash
# End-to-end tests
poetry run pytest tests/integration_ai/test_e2e_async_server.py -v

# Bidirectional async tests
poetry run pytest tests/integration_ai/test_bidirectional_async.py -v

# Streaming tests
poetry run pytest tests/integration_ai/test_streaming_async_server.py -v
```

**Stage 5: Performance Benchmarks**
```bash
# Performance comparison
poetry run pytest tests/integration_ai/test_performance_migration.py -v --benchmark
```

**Stage 6: Regression Tests**
```bash
# Run ALL existing tests to ensure no regressions
poetry run pytest tests/ -v --cov=deep_worker_cli --cov-report=html
```

---

## 7. Rollout Plan

### 7.1 Development Workflow

**Phase 1: Setup (Day 1)**
- Configure source integration
- Verify imports
- Create migration branch

**Phase 2: Server Migration (Days 2-3)**
- Migrate AgentProcess
- Migrate AgentService
- Unit tests

**Phase 3: Client Migration (Days 3-4)**
- Migrate InteractiveApp
- Migrate AgentClient
- Unit tests

**Phase 4: Integration Testing (Days 4-5)**
- End-to-end tests
- Performance benchmarks
- Stress tests

**Phase 5: Cleanup (Day 5)**
- Remove arpyc dependency
- Documentation updates
- Code review

### 7.2 Rollback Strategy

**Rollback Trigger Conditions:**
1. Any critical functionality breaks
2. Performance degrades
3. Stability issues in production
4. Critical bugs that can't be fixed quickly

**Rollback Process:**
```bash
# Revert to previous commit
git checkout <previous-commit>

# Or: Revert pyproject.toml to use classic RPyC
[tool.poetry.dependencies]
rpyc = "^5.0.0"  # Classic RPyC from PyPI

# Rebuild
poetry install
```

**Rollback Verification:**
```bash
# Run smoke tests
poetry run pytest tests/integration_ai/test_smoke.py -v

# Verify agent starts
deep-worker test-agent

# Verify client connects
dwcli test-agent
```

### 7.3 Feature Flags (Optional)

**For gradual rollout, implement feature flag:**

```python
# config.py
USE_ASYNCIO_SERVER = os.getenv("DEEPWORKER_USE_ASYNCIO_SERVER", "true").lower() == "true"

# agent_process.py
async def _start_rpyc_server(self):
    if USE_ASYNCIO_SERVER:
        from rpyc.utils.async_server import AsyncioServer
        # ... AsyncioServer implementation ...
    else:
        from rpyc import ThreadedServer
        # ... ThreadedServer implementation (fallback) ...
```

**Enable/Disable:**
```bash
# Use AsyncioServer (default after migration)
export DEEPWORKER_USE_ASYNCIO_SERVER=true

# Fallback to ThreadedServer if needed
export DEEPWORKER_USE_ASYNCIO_SERVER=false
```

---

## 8. Risk Assessment & Mitigation

### 8.1 Identified Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Breaking existing functionality** | Medium | High | Comprehensive test suite, feature parity matrix |
| **Performance regression** | Low | High | Performance benchmarks before/after |
| **Incompatibility with rpyc_async** | Low | High | Early integration testing, verify examples work |
| **Callback handling issues** | Medium | Medium | Extensive bidirectional tests, async callback tests |
| **TaskManager integration problems** | Low | Medium | Isolated testing, gradual integration |
| **Windows compatibility** | High | Low | Document Linux/macOS only (add_reader limitation) |
| **Production stability** | Medium | High | Gradual rollout, feature flags, rollback plan |

### 8.2 Mitigation Details

#### Risk: Breaking Existing Functionality

**Mitigation:**
1. Create comprehensive test coverage BEFORE migration
2. Use feature parity matrix to track all features
3. Run existing integration tests after each change
4. Manual testing of critical flows

**Verification:**
```bash
# Before migration - establish baseline
poetry run pytest tests/ --cov-report=html

# After migration - verify no regressions
poetry run pytest tests/ --cov-report=html --compare-baseline
```

#### Risk: Performance Regression

**Mitigation:**
1. Establish performance baselines with current implementation
2. Run benchmarks after each phase
3. Compare latency, throughput, memory usage
4. Accept only if performance improves or is neutral

**Benchmark Script:**
```python
# benchmark_migration.py

def benchmark_current():
    """Benchmark current implementation (ThreadedServer + arpyc)."""
    # ... measure latency, throughput, memory ...

def benchmark_migrated():
    """Benchmark migrated implementation (AsyncioServer)."""
    # ... measure latency, throughput, memory ...

def compare():
    current = benchmark_current()
    migrated = benchmark_migrated()

    print(f"Latency: {current.latency:.2f}ms → {migrated.latency:.2f}ms")
    print(f"Throughput: {current.throughput:.2f} req/s → {migrated.throughput:.2f} req/s")
    print(f"Memory: {current.memory:.2f}MB → {migrated.memory:.2f}MB")
```

#### Risk: Callback Handling Issues

**Mitigation:**
1. Test both sync and async callbacks
2. Test callback exceptions
3. Test callback timeouts
4. Test multiple subscribers

**Test Cases:**
```python
@pytest.mark.asyncio
async def test_sync_callback():
    """Test sync callback still works."""
    def sync_callback(msg):
        results.append(msg)
    # ...

@pytest.mark.asyncio
async def test_async_callback():
    """Test async callback works."""
    async def async_callback(msg):
        await asyncio.sleep(0.01)
        results.append(msg)
    # ...

@pytest.mark.asyncio
async def test_callback_exception():
    """Test callback exception handling."""
    def failing_callback(msg):
        raise ValueError("Test error")
    # Should not crash server
```

---

## 9. Success Criteria

### 9.1 Functional Requirements

✅ **All features work identically:**
- Agent invocation (streaming and non-streaming)
- Message subscription and real-time updates
- Conversation history retrieval
- Slash commands (/clear, /remember, /tokens)
- HITL approvals
- Agent interruption
- Graceful shutdown
- Multiple concurrent clients
- TaskManager integration

✅ **New capability enabled:**
- Bidirectional async calls work without deadlock
- Deep recursion (depth > 10) works reliably

### 9.2 Performance Requirements

✅ **Performance improves or is neutral:**
- Latency: < 10ms (vs ~301ms current)
- Throughput: >= 2x improvement for I/O-bound operations
- Memory: <= current usage
- CPU: <= current usage

### 9.3 Quality Requirements

✅ **Code quality maintained:**
- All tests pass
- Code coverage >= current coverage
- No new linter warnings
- Documentation updated

✅ **Stability:**
- No crashes in 1-hour stress test
- No memory leaks
- Graceful error handling

### 9.4 Acceptance Checklist

**Before declaring migration complete, verify:**

- [ ] Source integration setup complete (`poetry install` works)
- [ ] AsyncioServer starts successfully
- [ ] All AgentService methods are async
- [ ] Client connects with native rpyc
- [ ] `enable_asyncio_serving()` is called on client
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Bidirectional async test passes (depth=10)
- [ ] Performance benchmarks meet requirements
- [ ] Stress test passes (100 concurrent clients, 1 hour)
- [ ] All existing features work identically
- [ ] No regressions in test suite
- [ ] Documentation updated
- [ ] Code review completed
- [ ] arpyc dependency removed
- [ ] Manual testing completed

---

## 10. Appendix: File Mapping

### 10.1 Files Requiring Changes

**Server Side:**

| File | Type | Changes |
|------|------|---------|
| `deep_worker_cli/process/agent_process.py` | Major | Replace ThreadedServer with AsyncioServer, make `_start_rpyc_server()` async |
| `deep_worker_cli/rpyc_service/agent_service.py` | Major | Convert all `exposed_*` methods to async, use `asyncio.Lock`, async callbacks |
| `deep_worker_cli/task_manager/task_manager.py` | Minor | Use `asyncio.create_task()` instead of thread, shared event loop |
| `deep_worker_cli/pyproject.toml` | Major | Change rpyc dependency to path dependency |

**Client Side:**

| File | Type | Changes |
|------|------|---------|
| `deep_worker_cli/cli/interactive.py` | Major | Replace `RpycAsyncSession` with `rpyc.connect()`, call `enable_asyncio_serving()` |
| `deep_worker_cli/client/agent_client.py` | Moderate | Update API to use native rpyc connection, remove `async_call` wrapper |

**Tests:**

| File | Type | Changes |
|------|------|---------|
| `tests/unit_tests/process/test_agent_process.py` | Moderate | Update to test AsyncioServer |
| `tests/unit_tests/rpyc_service/test_agent_service.py` | Moderate | Update to test async methods |
| `tests/unit_tests/cli/test_interactive.py` | Moderate | Update to test native rpyc |
| `tests/integration_ai/test_interactive_rpyc.py` | Major | Full rewrite for AsyncioServer |
| Add: `tests/integration_ai/test_bidirectional_async.py` | New | Test bidirectional async |
| Add: `tests/integration_ai/test_performance_migration.py` | New | Benchmark migration |

**Documentation:**

| File | Type | Changes |
|------|------|---------|
| `ARCHITECTURE.md` | Moderate | Update architecture diagrams, remove BgServingThread references |
| `README.md` | Minor | Update dependencies section |
| Add: `MIGRATION_NOTES.md` | New | Document migration for users |

### 10.2 Files to Remove

After successful migration and verification:

| File/Directory | Reason |
|----------------|--------|
| the `arpyc/` directory at the repository root | No longer needed - replaced by rpyc_async |
| `ARPYC_ASYNCIO_NATIVE_RESEARCH.md` | Historical - can archive |
| `ARPYC_ASYNC_RECURSION_GUIDE.md` | Historical - can archive |

**Note:** Archive documents to `docs/historical/` instead of deleting.

### 10.3 Reference Examples from rpyc_async

**Study these examples during migration:**

| Example File | Purpose |
|--------------|---------|
| `examples/bidirectional_async/01_simple_bidirectional.py` (in rpyc_async) | Basic bidirectional pattern |
| `examples/bidirectional_async/02_recursive_callbacks.py` (in rpyc_async) | Recursive bidirectional pattern |
| `examples/bidirectional_async/asyncio_server_success.py` (in rpyc_async) | Full AsyncioServer implementation |
| `docs/ASYNCIO_SERVER_MIGRATION.md` (in rpyc_async) | Official migration guide |
| `docs/API_REFERENCE.md` (in rpyc_async) | API reference |

---

## Conclusion

This migration plan provides a comprehensive roadmap for transitioning the deep_worker project from classic RPyC with arpyc wrappers to the native AsyncioServer implementation from rpyc_async. The migration will:

1. ✅ Enable true bidirectional asynchronous communication
2. ✅ Improve performance by 65x-156x for I/O-bound operations
3. ✅ Reduce latency from ~301ms to ~1.93ms
4. ✅ Simplify architecture (single thread instead of multiple threads)
5. ✅ Allow immediate testing of rpyc_async fixes
6. ✅ Eliminate technical debt (arpyc wrapper)

**Estimated Timeline:** 17-25 hours over 5 days

**Risk Level:** Medium (comprehensive testing and rollback plan mitigate risks)

**Expected Outcome:** Fully functional system with significantly improved performance and architecture.

---

**Document Status:** ✅ Ready for Review
**Next Steps:** Review with team, get approval, begin Phase 1 (Source Integration Setup)
**Last Updated:** 2026-03-22
