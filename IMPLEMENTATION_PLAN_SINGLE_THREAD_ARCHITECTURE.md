# Implementation Plan for a Single-Thread, Single-Event-Loop Architecture

## Creation date
2026-03-30

## Goal
Bring the project architecture in line with the following principles:
- One execution thread
- One event loop
- Interaction between the agent and the TaskManager in the main event loop
- RPyC via AsyncioServer in the main event loop without additional threads
- TUI in a separate CLI client process with its own single thread and event loop
- Maximum alignment with the architecture of the upstream reference project

## Current state (problems)

### 1. Multiple event loops
- **Main loop**: `asyncio.run()` in `agent_launcher.py:47`
- **ManagerThread loop**: `asyncio.new_event_loop()` in `manager_thread.py:57-58`

### 2. Multiple threads
- Main thread (agent + RPyC)
- **ManagerThread** for the TaskManager (`manager_thread.py`)
- RPyC background thread (from AsyncioServer)
- Textual worker thread (in TUI mode, `interactive.py:408`)

### 3. Blocking operations
- `msg_queue.get()` without a timeout in `interactive.py:400`
- `msg_queue.put()` from the RPyC BgServingThread in `interactive.py:394`
- `manager_thread.join(timeout=5.0)` in `task_manager.py:254`

## Architecture of the upstream reference project

### Key characteristics
```
<upstream reference CLI package>/
├── main.py           # asyncio.run() - the single entry point
├── app.py            # Textual App in the same event loop
├── agent.py          # The agent runs in the main loop
└── sessions.py       # Checkpointer - async operations
```

- **One thread**: Everything in the main thread
- **One event loop**: `asyncio.run()` in main.py
- **Textual integration**: Runs in the same event loop via Textual.run_workers
- **No TaskManager**: This functionality is absent in the upstream project

## Migration plan

### Phase 1: TaskManager refactoring (CRITICAL)

#### 1.1. Remove ManagerThread
**Files to change:**
- `task_manager/manager_thread.py` - DELETE or mark as deprecated
- `task_manager/task_manager.py`

**Changes in task_manager.py:**
```python
# BEFORE:
class TaskManager:
    def __init__(self):
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.manager_thread = ManagerThread(...)
        self.manager_thread.start()

# SHOULD BE:
class TaskManager:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._evaluation_task: asyncio.Task | None = None
        # No threads, no separate queues
```

#### 1.2. Rewrite the evaluation logic as an async task
**Instead of a separate thread:**
```python
async def start(self) -> None:
    """Start the background evaluation task in the current event loop."""
    if self._evaluation_task is None or self._evaluation_task.done():
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

async def _evaluation_loop(self) -> None:
    """Background async task for periodic evaluation."""
    while self._running:
        try:
            await asyncio.sleep(self.evaluation_interval)
            await self._evaluate_context()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Evaluation error: {e}")

async def stop(self) -> None:
    """Stop the background task."""
    self._running = False
    if self._evaluation_task:
        self._evaluation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._evaluation_task
```

**Advantages:**
- No inter-thread communication
- No blocking on Queue.get/put
- Natural integration with the main event loop
- Simple cancellation via `task.cancel()`

#### 1.3. Integration with AgentProcess
**File:** `process/agent_process.py`

```python
async def start(self) -> None:
    """Start the agent process."""
    # ... existing code ...

    # If the TaskManager is enabled
    if self.task_manager:
        await self.task_manager.start()  # Start an async task, not a thread

    # ... rpyc server start ...

    try:
        await self._main_loop()
    finally:
        if self.task_manager:
            await self.task_manager.stop()  # Correct task shutdown
```

### Phase 2: Verifying the RPyC AsyncioServer

#### 2.1. Make sure AsyncioServer is used
**File:** `process/agent_process.py:606-624`

```python
# CHECK:
from rpyc_async.server import AsyncioServer  # From the async RPyC library

server = AsyncioServer(
    service=agent_service,
    hostname=self.hostname,
    port=self.port,
    # IMPORTANT: do not use ThreadedServer
)

# Start in the main event loop
await server.start()
```

#### 2.2. Verify the absence of BgServingThread
**Files to check:**
- `arpyc/session.py` - there must be no `threading.Thread`
- All callbacks must run via `asyncio.create_task()` or `await`

### Phase 3: Full separation of the TUI and the agent (ALREADY IMPLEMENTED)

#### 3.1. Confirm the client-server architecture
**Current structure (correct):**
```
┌─────────────────────┐         RPyC/AsyncIO         ┌──────────────────────┐
│  CLI client         │◄────────────────────────────►│  agent (server)      │
│                     │                               │                      │
│ - Separate process  │                               │ - Separate process   │
│ - One event loop    │                               │ - One event loop     │
│ - Textual TUI       │                               │ - LangGraph agent    │
│ - interactive.py    │                               │ - agent_process.py   │
└─────────────────────┘                               └──────────────────────┘
```

#### 3.2. Remove the built-in TUI from the agent process
**File:** `cli/interactive.py:408`

```python
# REMOVE the worker thread:
# worker = threading.Thread(target=_worker_thread, daemon=True)
# worker.start()

# REPLACE with RPyC callbacks in the event loop:
async def handle_message_async(msg: dict[str, Any]) -> None:
    """Handle messages asynchronously in the main loop."""
    # Direct dispatch to Textual via call_from_thread is not needed
    # if the TUI runs in a separate process
    pass
```

**Note:** If the TUI has already been fully moved into the CLI client, then `interactive.py` in the agent process should be minimal or absent.

### Phase 4: Synchronization with the upstream reference project

#### 4.1. Project structure comparison
**Upstream:**
```
<upstream reference CLI package>/
├── main.py          # Entry point with asyncio.run()
├── app.py           # Textual App
├── agent.py         # LangGraph agent creation
├── sessions.py      # Checkpointer operations
└── tools.py         # Agent tools
```

**Current project:**
```
<CLI package>/
├── main.py                     # CLI entry
├── cli/
│   ├── agent_command.py        # Agent startup command
│   ├── client_command.py       # Client startup command
│   └── interactive.py          # TUI (should be in the client only)
├── process/
│   ├── agent_launcher.py       # Agent process startup
│   └── agent_process.py        # Main agent process
├── task_manager/
│   ├── task_manager.py         # TaskManager (refactoring needed)
│   └── manager_thread.py       # ManagerThread (DELETE)
└── rpyc_service/
    └── agent_service.py        # RPyC service
```

#### 4.2. Architecture alignment
**Differences to keep:**
- RPyC client-server architecture (new feature)
- TaskManager functionality (new feature, but to be rewritten without threads)
- Separation of agent/client commands (new feature)

**Things to bring into line:**
- Remove multiple threads
- Remove multiple event loops
- Use asyncio.create_task() for background tasks
- Use async/await for all IO operations

#### 4.3. Adopt patterns from the upstream project
**From the upstream reference `app.py`:**
```python
# Pattern for running Textual in the same event loop:
class DeepAgentsApp(App):
    async def on_mount(self) -> None:
        # Background tasks via Textual workers
        self.run_worker(self._background_task(), exclusive=False)

    async def _background_task(self) -> None:
        # Asynchronous background work in the same loop
        while True:
            await asyncio.sleep(1)
            # Processing...
```

**Apply to the TaskManager:**
```python
class TaskManager:
    async def start_background_evaluation(self) -> None:
        """Start background evaluation as an async task."""
        self._task = asyncio.create_task(self._evaluation_loop())

    async def _evaluation_loop(self) -> None:
        """Periodic context evaluation."""
        while self._running:
            await asyncio.sleep(self.interval)
            await self._evaluate()
```

### Phase 5: Testing and validation

#### 5.1. Unit tests
**Verify the absence of:**
- `threading.Thread` in production code
- `asyncio.new_event_loop()` outside of tests
- Blocking calls without a timeout

**Files to check:**
```bash
# Find threading usage
grep -r "threading\." <CLI package>/ --include="*.py" | grep -v test | grep -v "#"

# Find event loop creation
grep -r "new_event_loop\|get_event_loop" <CLI package>/ --include="*.py" | grep -v test

# Find blocking operations
grep -r "\.get()\|\.put()" <CLI package>/ --include="*.py" | grep -v test
```

#### 5.2. Integration tests
**Tests to run:**
- `tests/integration_ai/test_taskmanager_threads.py` - rewrite for the async task
- `tests/integration_ai/test_rpyc_async_recursion.py` - verify operation without threads
- `tests/integration_ai/test_agent_rpyc_metadata.py` - verify the RPyC AsyncioServer

#### 5.3. Regression testing
**Scenarios:**
1. Starting the agent with the TaskManager
2. Connecting a client over RPyC
3. Executing tool calls
4. Stopping the agent (graceful shutdown)
5. Interrupting the agent (Ctrl+C)

## Detailed per-file changes

### 1. `task_manager/task_manager.py`

#### Remove:
```python
# lines 75-76
self.task_queue: asyncio.Queue = asyncio.Queue()
self.result_queue: asyncio.Queue = asyncio.Queue()

# lines 84-90
self.manager_thread = ManagerThread(
    agent_name=agent_name,
    task_queue=self.task_queue,
    result_queue=self.result_queue,
    # ...
)

# lines 106-115
async def start(self) -> None:
    self.manager_thread.start()
    await asyncio.sleep(0.1)  # BLOCKING!

# lines 247-254
async def stop(self) -> None:
    self.manager_thread.join(timeout=5.0)  # BLOCKING!
```

#### Add:
```python
def __init__(self, ...):
    self._evaluation_task: asyncio.Task | None = None
    self._running = False
    # The remaining fields without a queue

async def start(self) -> None:
    """Start the background evaluation task."""
    if self._evaluation_task is None or self._evaluation_task.done():
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info(f"TaskManager started for agent {self.agent_name}")

async def _evaluation_loop(self) -> None:
    """Background context evaluation loop."""
    while self._running:
        try:
            await asyncio.sleep(self.evaluation_interval)
            if self._should_evaluate():
                await self._evaluate_context()
        except asyncio.CancelledError:
            logger.info("Evaluation loop cancelled")
            break
        except Exception as e:
            logger.error(f"Evaluation error: {e}", exc_info=True)

async def stop(self) -> None:
    """Stop the background task."""
    self._running = False
    if self._evaluation_task and not self._evaluation_task.done():
        self._evaluation_task.cancel()
        try:
            await asyncio.wait_for(self._evaluation_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        finally:
            self._evaluation_task = None
    logger.info(f"TaskManager stopped for agent {self.agent_name}")
```

### 2. `task_manager/manager_thread.py`

#### Action: DELETE THE FILE or add a deprecation notice
```python
"""
DEPRECATED: This file is deprecated and will be removed.

TaskManager now runs in the main event loop using asyncio.create_task()
instead of a separate thread with its own event loop.

See: IMPLEMENTATION_PLAN_SINGLE_THREAD_ARCHITECTURE.md
"""
import warnings

warnings.warn(
    "manager_thread.py is deprecated. TaskManager now runs in main event loop.",
    DeprecationWarning,
    stacklevel=2,
)
```

### 3. `process/agent_process.py`

#### Change the start() method:
```python
async def start(self) -> None:
    """Start the agent process."""
    logger.info(f"Starting agent process for {self.agent_name}")

    # Component initialization
    await self._initialize()

    # Start the TaskManager in the same event loop
    if self.task_manager:
        await self.task_manager.start()
        logger.info("TaskManager background task started")

    # Start the RPyC AsyncioServer (without threads)
    if self.enable_rpyc:
        await self._start_rpyc_server()
        logger.info(f"RPyC AsyncioServer started on {self.hostname}:{self.port}")

    # Main agent loop
    try:
        await self._main_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await self._shutdown()

async def _shutdown(self) -> None:
    """Graceful shutdown."""
    logger.info("Shutting down agent process")

    # Stop the TaskManager
    if self.task_manager:
        await self.task_manager.stop()
        logger.info("TaskManager stopped")

    # Stop the RPyC server
    if self.rpyc_server:
        await self.rpyc_server.close()
        logger.info("RPyC server stopped")

    # Cleanup of other resources
    await self._cleanup()
```

### 4. `cli/interactive.py` (if used in the agent process)

#### Remove the worker thread (lines 400-420):
```python
# REMOVE:
def _worker_thread():
    while True:
        msg = msg_queue.get()  # BLOCKING CALL
        # ...

worker = threading.Thread(target=_worker_thread, daemon=True)
worker.start()
```

#### If the TUI lives entirely in the CLI client - delete the file from the agent process

### 5. `process/agent_launcher.py`

#### Current code (correct):
```python
# Line 47
asyncio.run(agent_process.start())  # The single entry point into the event loop
```

**DO NOT CHANGE** - this is the correct architecture, matching the upstream reference project.

## Checklist

### Before starting the migration:
- [ ] Create the branch `refactor/single-thread-architecture`
- [ ] Make a full backup of the current state
- [ ] Run all existing tests and record the results
- [ ] Create baseline performance metrics

### Phase 1: TaskManager
- [ ] Remove the use of `ManagerThread` from `task_manager.py`
- [ ] Rewrite `_evaluation_loop` as an async task
- [ ] Replace `task_queue`/`result_queue` with direct async calls
- [ ] Update the `start()` and `stop()` methods
- [ ] Mark `manager_thread.py` as deprecated
- [ ] Update the `test_taskmanager_threads.py` tests
- [ ] Run the TaskManager tests

### Phase 2: RPyC
- [ ] Verify the use of `AsyncioServer` from the async RPyC library
- [ ] Make sure `BgServingThread` is absent
- [ ] Check all callbacks for async/await
- [ ] Run the `test_rpyc_async_recursion.py` tests

### Phase 3: TUI
- [ ] Confirm full separation of the CLI client and agent processes
- [ ] Remove worker threads from the agent process
- [ ] Verify that `interactive.py` is used only in the CLI client
- [ ] Run the client-server integration tests

### Phase 4: Synchronization with the upstream reference project
- [ ] Compare the structure of main.py
- [ ] Compare the structure of app.py
- [ ] Adopt async patterns for background tasks
- [ ] Update the documentation

### Phase 5: Testing
- [ ] Run all unit tests
- [ ] Run all integration tests
- [ ] Perform manual testing of all scenarios
- [ ] Verify the absence of threading via grep
- [ ] Verify the absence of new_event_loop via grep
- [ ] Compare performance metrics against the baseline

### Finalization:
- [ ] Code review
- [ ] Update the ARCHITECTURE documentation
- [ ] Update the README with the new architecture
- [ ] Merge into the main branch

## Risks and mitigation

### Risk 1: Breaking existing TaskManager functionality
**Mitigation:**
- Keep the current version in a deprecated file
- Create a feature flag to switch between versions
- Run parallel testing of both implementations

### Risk 2: Problems with RPyC callbacks
**Mitigation:**
- Thoroughly test all RPyC scenarios
- Use the existing tests from the async RPyC library
- Add logging to debug callback execution

### Risk 3: Deadlocks during shutdown
**Mitigation:**
- Use `asyncio.wait_for()` with a timeout for all shutdown operations
- Add handling of `asyncio.CancelledError`
- Test Ctrl+C and graceful shutdown

### Risk 4: TUI regression
**Mitigation:**
- The TUI is already in a separate process - minimal risk
- Client-server integration tests should cover all scenarios

## Success metrics

### Technical metrics:
- [ ] 0 uses of `threading.Thread` in production code
- [ ] 1 use of `asyncio.run()` (only in agent_launcher.py)
- [ ] 0 uses of `asyncio.new_event_loop()` in production code
- [ ] 0 blocking calls without a timeout
- [ ] All tests green
- [ ] No performance regression (±5%)

### Functional metrics:
- [ ] The TaskManager works correctly
- [ ] The RPyC client-server works without problems
- [ ] The TUI is responsive and does not hang
- [ ] Graceful shutdown works correctly
- [ ] Ctrl+C is handled correctly

## Next steps after the rollout

1. **Documentation:**
   - Update ARCHITECTURE.md with the new scheme
   - Add examples of async patterns
   - Document the differences from the upstream reference project

2. **Optimization:**
   - Profiling of async operations
   - Optimization of awaitable chains
   - Reducing the number of context switches

3. **Monitoring:**
   - Add event loop lag metrics
   - Monitoring of pending tasks
   - Tracking memory leaks in async tasks

## References

### Project documents:
- `ARPYC_ASYNC_RECURSION_GUIDE.md` - Guide to async RPyC
- `RPYC_RECURSION_FINAL_REPORT.md` - Solution to the recursion problems
- `MIGRATION_ASYNCIO_SERVER.md` - Migration to AsyncioServer

### Upstream reference project:
- The upstream reference `main.py` - reference main entry point
- The upstream reference `app.py` - reference Textual integration

### RPyC async:
- `arpyc/session.py` - AsyncioServer implementation
- `arpyc/server.py` - Server without threads

## Conclusion

This migration will bring the project architecture in line with the single-thread and single-event-loop principles, which will:

1. **Simplify the code** - no inter-thread synchronization
2. **Increase reliability** - no race conditions or deadlocks
3. **Improve debugging** - one thread = a linear traceback
4. **Move closer to the upstream reference project** - architectural uniformity
5. **Preserve the new features** - TaskManager, RPyC, CLI client

Main changes:
- TaskManager: thread → async task
- Communication: Queue → direct async calls
- TUI: already in a separate process ✓
- RPyC: AsyncioServer without threads ✓

The migration is safe provided the plan is followed and all tests are executed.
