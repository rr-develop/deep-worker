# Design: Checkpointer Behavior Fix

## Problem Statement

Currently, `AgentProcess` does not pass a `checkpointer` parameter to `create_cli_agent()`, causing all agents to fall back to `InMemorySaver()` (in-memory only storage). This means:

1. Agent conversation history is NOT persisted to SQLite database
2. `deep-worker --list` does not show running agents
3. `dwcli --history <agent>` shows no history for running agents

**Expected Behavior:**
- Default (normal runs): Agents write to SQLite database using `AsyncSqliteSaver`
- Tests: Agents write to memory only using `InMemorySaver` by default
- CLI flags: Explicit options to override default behavior
  - `--memory-only`: Force in-memory storage (even for normal runs)
  - `--use-sqlite`: Force SQLite storage (even for tests)

## Current Code Analysis

### Location of Bug
**File:** `deep_worker_cli/process/agent_process.py`
**Lines:** 253-262

```python
self.agent, self.backend = create_cli_agent(
    model=model,
    assistant_id=self.name,
    tools=web_tools,
    auto_approve=self.skip_permissions,
    enable_memory=True,
    enable_skills=True,
    enable_shell=True,
    project_memory_dir=Path(self.config.working_dir),
    # checkpointer parameter is MISSING here
)
```

### Fallback Behavior
**File:** `deep_worker_cli/agent.py`
**Line:** 689

```python
final_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
```

When no checkpointer is provided, it defaults to `InMemorySaver()`.

## Design Solution

### 1. Default Behavior: SQLite Persistence

#### 1.1 Modify `agent_process.py`

**Location:** `AgentProcess.start()` method (around line 253)

**Changes:**
1. Import `get_checkpointer` from sessions module
2. Create checkpointer before calling `create_cli_agent()`
3. Pass checkpointer to `create_cli_agent()`
4. Store checkpointer instance for proper lifecycle management

**Pseudocode:**
```python
async def start(self):
    # ... existing code ...

    # Determine checkpointer based on flags and environment
    # Priority order:
    # 1. Explicit CLI flags (--use-sqlite or --memory-only) override everything
    # 2. Test environment detection (auto use InMemorySaver)
    # 3. Default: use SQLite for normal runs

    if self.use_sqlite:
        # Explicit --use-sqlite flag: ALWAYS use SQLite (even in tests)
        from deep_worker_cli.sessions import get_checkpointer
        self._checkpointer_ctx = get_checkpointer()
        checkpointer = await self._checkpointer_ctx.__aenter__()
    elif self.memory_only:
        # Explicit --memory-only flag: ALWAYS use memory (even in normal runs)
        checkpointer = None  # Will fall back to InMemorySaver
    elif is_running_tests():
        # Test environment: default to memory (unless --use-sqlite specified)
        checkpointer = None  # Will fall back to InMemorySaver
    else:
        # Normal operation: use SQLite
        from deep_worker_cli.sessions import get_checkpointer
        self._checkpointer_ctx = get_checkpointer()
        checkpointer = await self._checkpointer_ctx.__aenter__()

    self.agent, self.backend = create_cli_agent(
        model=model,
        assistant_id=self.name,
        tools=web_tools,
        auto_approve=self.skip_permissions,
        enable_memory=True,
        enable_skills=True,
        enable_shell=True,
        project_memory_dir=Path(self.config.working_dir),
        checkpointer=checkpointer,  # ADD THIS
    )

    # ... existing code ...
```

#### 1.2 Checkpointer Lifecycle Management

**Challenge:** `get_checkpointer()` returns an async context manager (`AsyncSqliteSaver`)

**Options:**

**Option A: Store context manager, manage lifecycle manually**
```python
self._checkpointer_ctx = get_checkpointer()
self.checkpointer = await self._checkpointer_ctx.__aenter__()

# Later in cleanup/shutdown:
await self._checkpointer_ctx.__aexit__(None, None, None)
```

**Option B: Keep reference, rely on agent cleanup**
```python
# If agent.py already manages checkpointer lifecycle properly
checkpointer = await get_checkpointer().__aenter__()
# Pass to create_cli_agent
# Agent should handle closing
```

**Recommendation:** Option A - explicit lifecycle management in `AgentProcess`
- Add `__aenter__` call in `start()`
- Add `__aexit__` call in cleanup/shutdown logic
- Ensures proper database connection closing

### 2. Test Detection: InMemorySaver for Tests

**Goal:** Automatically detect when running under pytest and use `InMemorySaver`

#### 2.1 Detection Mechanism

**Method 1: Check for pytest module**
```python
import sys

def is_running_tests() -> bool:
    """Detect if code is running under pytest."""
    return "pytest" in sys.modules
```

**Method 2: Environment variable**
```python
import os

def is_running_tests() -> bool:
    """Detect if code is running under pytest."""
    return os.environ.get("PYTEST_CURRENT_TEST") is not None
```

**Method 3: Explicit test marker**
```python
import os

def is_running_tests() -> bool:
    """Detect if code is running in test mode."""
    return os.environ.get("DEEP_WORKER_TEST_MODE") == "1"
```

**Recommendation:** Combination approach with production override
```python
def is_running_tests() -> bool:
    """Detect if running in test environment.

    Checks multiple indicators:
    1. DEEP_WORKER_FORCE_PRODUCTION=1 - override, force production mode (return False)
    2. DEEP_WORKER_TEST_MODE=1 - explicit test mode (return True)
    3. PYTEST_CURRENT_TEST - pytest sets this automatically
    4. pytest in sys.modules - pytest is loaded

    Returns:
        True if running in test environment
    """
    import os
    import sys

    # Allow explicit override to force production mode in tests
    if os.environ.get("DEEP_WORKER_FORCE_PRODUCTION") == "1":
        return False

    # Explicit test mode
    if os.environ.get("DEEP_WORKER_TEST_MODE") == "1":
        return True

    # Pytest environment variable
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True

    # Pytest module loaded
    if "pytest" in sys.modules:
        return True

    return False
```

**Usage in Test Fixtures:**
```python
# Test fixture that needs SQLite persistence
@pytest.fixture
def with_sqlite_storage(monkeypatch):
    """Force SQLite storage even in test environment."""
    monkeypatch.setenv("DEEP_WORKER_FORCE_PRODUCTION", "1")
    yield
    # Cleanup happens automatically via monkeypatch
```

#### 2.2 Integration into AgentProcess

**Decision Logic:**

```python
# Priority order for checkpointer selection:
# 1. --use-sqlite flag: Use SQLite (overrides test detection)
# 2. --memory-only flag: Use memory (overrides test detection)
# 3. Test environment: Use memory (default for tests)
# 4. Normal run: Use SQLite (default for production)

if self.use_sqlite:
    # Explicit flag: force SQLite even in tests
    checkpointer = await get_checkpointer().__aenter__()
elif self.memory_only:
    # Explicit flag: force memory even in production
    checkpointer = None
elif is_running_tests():
    # Auto-detected test: default to memory
    checkpointer = None
else:
    # Normal production run: use SQLite
    checkpointer = await get_checkpointer().__aenter__()
```

### 3. New CLI Options: --memory-only and --use-sqlite

**Goal:** Allow explicit control over storage mode, overriding defaults

#### 3.1 Add Arguments to Parser

**File:** `deep_worker_cli/cli/agent_command.py`
**Function:** `create_parser()`

```python
# Mutually exclusive group for storage mode
storage_group = parser.add_mutually_exclusive_group()

storage_group.add_argument(
    "--memory-only",
    action="store_true",
    dest="memory_only",
    help="Use in-memory storage only (no persistence to database)",
)

storage_group.add_argument(
    "--use-sqlite",
    action="store_true",
    dest="use_sqlite",
    help="Force SQLite persistence (even in test environment)",
)
```

**Note:** These flags are mutually exclusive - only one can be specified.

#### 3.2 Pass to AgentProcess

**File:** `deep_worker_cli/cli/agent_command.py`
**Function:** `run_agent()`

```python
# Create and start agent
agent = AgentProcess(
    name=args.name,
    config=config,
    skip_permissions=args.skip_permissions,
    enable_task_manager=args.task_manager,
    memory_only=args.memory_only,    # ADD THIS
    use_sqlite=args.use_sqlite,      # ADD THIS
)
```

#### 3.3 Add Fields to AgentProcess

**File:** `deep_worker_cli/process/agent_process.py`
**Class:** `AgentProcess`

```python
def __init__(
    self,
    name: str,
    config: AgentConfig,
    skip_permissions: bool = False,
    enable_task_manager: bool = False,
    memory_only: bool = False,   # ADD THIS
    use_sqlite: bool = False,    # ADD THIS
):
    # ... existing fields ...

    # Validation: both flags cannot be True simultaneously
    if memory_only and use_sqlite:
        raise ValueError("Cannot specify both --memory-only and --use-sqlite")

    self.memory_only = memory_only
    self.use_sqlite = use_sqlite
    self._checkpointer_ctx = None  # For lifecycle management
```

### 4. Cleanup and Shutdown

**Goal:** Properly close database connections when agent stops

**File:** `deep_worker_cli/process/agent_process.py`

**Add cleanup method:**
```python
async def cleanup(self):
    """Clean up resources, including checkpointer."""
    if self._checkpointer_ctx is not None:
        try:
            await self._checkpointer_ctx.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error closing checkpointer: {e}", file=sys.stderr)
        finally:
            self._checkpointer_ctx = None
```

**Call in shutdown:**
```python
# In run_agent() or wherever shutdown happens
try:
    await agent.start()
    while agent.running:
        await asyncio.sleep(1)
finally:
    await agent.cleanup()
```

## Implementation Checklist

### Phase 1: Core Functionality
- [ ] Add `is_running_tests()` utility function to appropriate module
- [ ] Add `memory_only` and `use_sqlite` parameters to `AgentProcess.__init__()`
- [ ] Add validation to prevent both flags being True
- [ ] Add `_checkpointer_ctx` field to `AgentProcess`
- [ ] Modify `AgentProcess.start()` to create and pass checkpointer with priority logic
- [ ] Add `AgentProcess.cleanup()` method for proper shutdown
- [ ] Update `run_agent()` to call cleanup on exit

### Phase 2: CLI Options
- [ ] Add mutually exclusive group to parser in `agent_command.py`
- [ ] Add `--memory-only` argument to storage group
- [ ] Add `--use-sqlite` argument to storage group
- [ ] Pass both flags to `AgentProcess` constructor
- [ ] Update help text and documentation

### Phase 3: Testing
- [ ] Verify existing tests still pass (should auto-detect test mode and use memory)
- [ ] Test `--use-sqlite` flag in test environment (should force SQLite)
- [ ] Verify `deep-worker agent_name` creates SQLite entries (normal run)
- [ ] Verify `deep-worker --memory-only agent_name` does NOT create SQLite entries
- [ ] Verify `deep-worker --list` shows running agents (when using SQLite)
- [ ] Verify `dwcli --history agent_name` shows conversation history
- [ ] Verify cleanup properly closes database connections
- [ ] Test mutual exclusivity: `--memory-only --use-sqlite` should fail
- [ ] Test with TaskManager enabled (should create two threads: `agent_name` and `agent_name_tm`)
- [ ] Verify test fixtures can override behavior with explicit flags

### Phase 4: Documentation
- [ ] Update CLI help text for both storage options
- [ ] Update README with `--memory-only` and `--use-sqlite` options
- [ ] Document test detection mechanism for test writers
- [ ] Document how to use `--use-sqlite` in test fixtures for integration tests
- [ ] Add examples of both flags in documentation

## Edge Cases and Considerations

### 1. TaskManager Threads
TaskManager creates its own thread with ID `{agent_name}_tm`. It should:
- Use the SAME checkpointer instance as main agent
- OR create its own checkpointer with same database path

**Recommendation:** Pass checkpointer to TaskManager constructor
```python
# In agent_process.py
if self.enable_task_manager:
    self.task_manager_thread = TaskManagerThread(
        agent_name=self.name,
        # ... other params ...
        checkpointer=checkpointer,  # Share same checkpointer
    )
```

### 2. Concurrent Access to SQLite
Multiple threads accessing same SQLite database:
- `AsyncSqliteSaver` handles this with connection pooling
- Each thread should have its own connection from pool
- Main agent thread: `thread_id = agent_name`
- TaskManager thread: `thread_id = agent_name_tm`

### 3. Database Path Consistency
Ensure `get_checkpointer()` always returns same database path:
- Currently uses `~/.deepagents/sessions.db`
- Should be consistent across all agent instances
- No need for per-agent databases (thread_id provides isolation)

### 4. Migration from InMemorySaver
Agents currently running with InMemorySaver:
- Will lose in-memory history when stopped
- After fix, new runs will persist to database
- No migration needed (clean slate)

### 5. Test Isolation and Fixtures
Tests should NOT interfere with each other:
- `InMemorySaver` provides natural isolation (each test gets own instance)
- No database cleanup needed between tests with default (memory) mode
- Existing test fixtures remain valid

**Test Fixtures with SQLite:**
For integration tests that need to verify database persistence, test fixtures can pass explicit flags:

```python
@pytest.fixture
async def agent_with_sqlite(tmp_path):
    """Create agent that writes to SQLite (for integration tests)."""
    # Option 1: Pass use_sqlite=True to AgentProcess
    agent = AgentProcess(
        name="test_agent",
        config=config,
        use_sqlite=True,  # Override test detection
    )
    yield agent
    await agent.cleanup()

    # Option 2: Set environment variable to disable test detection
    # This allows testing as if in production environment
    import os
    os.environ["DEEP_WORKER_FORCE_PRODUCTION"] = "1"
    agent = AgentProcess(name="test_agent", config=config)
    # ... will use SQLite because test detection is bypassed
```

**Alternative: Environment Variable Control**
Add `DEEP_WORKER_FORCE_PRODUCTION` environment variable that disables test detection:

```python
def is_running_tests() -> bool:
    """Detect if running in test environment."""
    # Allow explicit override to force production mode in tests
    if os.environ.get("DEEP_WORKER_FORCE_PRODUCTION") == "1":
        return False  # Pretend we're NOT in test mode

    # ... rest of detection logic ...
```

This allows test fixtures to control behavior via environment variable instead of passing flags.

## Questions for Review

1. **Checkpointer Lifecycle:** Should we manage lifecycle explicitly in `AgentProcess`, or rely on `agent.py` to handle it?
   - **Recommendation:** Explicit management in `AgentProcess` for clarity

2. **Test Detection:** Which method(s) should we use for detecting test environment?
   - **Recommendation:** Multi-method approach (env var + pytest detection)

3. **TaskManager Checkpointer:** Should TaskManager share checkpointer or create its own?
   - **Recommendation:** Share same instance (same database, different thread_id)

4. **Error Handling:** What should happen if database connection fails?
   - **Recommendation:** Log error, fall back to `InMemorySaver`, continue running

5. **CLI Naming:** Is `--memory-only` and `--use-sqlite` clear enough?
   - **Recommendation:** Both are clear and explicit

6. **Test Environment Override:** Should we use CLI flag (`--use-sqlite`) or environment variable (`DEEP_WORKER_FORCE_PRODUCTION`) in test fixtures?
   - **Recommendation:** Support both approaches:
     - CLI flag: Explicit, visible in command
     - Env var: Cleaner for pytest fixtures with `monkeypatch`

## Files to Modify

1. **`deep_worker_cli/process/agent_process.py`**
   - Add `memory_only` and `use_sqlite` parameters
   - Add validation for mutual exclusivity
   - Add `_checkpointer_ctx` field
   - Modify `start()` to implement priority logic for checkpointer selection
   - Add `cleanup()` method

2. **`deep_worker_cli/cli/agent_command.py`**
   - Add mutually exclusive group for storage options
   - Add `--memory-only` argument to group
   - Add `--use-sqlite` argument to group
   - Pass both flags to `AgentProcess` constructor
   - Call `cleanup()` on exit

3. **`deep_worker_cli/sessions.py`** or new **`deep_worker_cli/utils.py`**
   - Add `is_running_tests()` utility function
   - Support `DEEP_WORKER_FORCE_PRODUCTION` environment variable
   - Support `DEEP_WORKER_TEST_MODE` environment variable
   - Check `PYTEST_CURRENT_TEST` environment variable
   - Check `pytest` in `sys.modules`

4. **`deep_worker_cli/task_manager/manager_thread.py`** (optional)
   - Accept `checkpointer` parameter in constructor
   - Pass to `create_task_manager_agent()`

5. **Test fixtures** (for integration tests requiring persistence)
   - Add fixture using `monkeypatch.setenv("DEEP_WORKER_FORCE_PRODUCTION", "1")`
   - OR create `AgentProcess` with `use_sqlite=True` parameter

## Success Criteria

After implementation, the following should work:

```bash
# 1. Normal run: persist to database (default)
deep-worker my_agent

# List all threads (should show my_agent)
deep-worker --list

# View history (should show conversation)
dwcli --history my_agent

# 2. Explicit memory-only mode
deep-worker --memory-only temp_agent

# List threads (should NOT show temp_agent)
deep-worker --list

# 3. Test environment: automatic memory mode
pytest tests/

# All tests should pass without modification
# Tests automatically use InMemorySaver

# 4. Force SQLite in test environment
pytest tests/integration/test_with_persistence.py

# Test uses fixture with DEEP_WORKER_FORCE_PRODUCTION=1
# OR uses --use-sqlite flag when creating AgentProcess
# Should write to SQLite database even in test

# 5. Mutual exclusivity check
deep-worker --memory-only --use-sqlite my_agent
# Should fail with error: "Cannot specify both flags"

# 6. Force SQLite (explicit override)
# Useful in test environments where you want to verify persistence
deep-worker --use-sqlite test_agent
```

## Priority Logic Summary

```
Decision Tree for Checkpointer Selection:

1. Is --use-sqlite flag set?
   YES → Use SQLite (even in tests)
   NO  → Go to 2

2. Is --memory-only flag set?
   YES → Use InMemorySaver (even in production)
   NO  → Go to 3

3. Is DEEP_WORKER_FORCE_PRODUCTION=1 set?
   YES → Use SQLite (override test detection)
   NO  → Go to 4

4. Is this a test environment?
   (Check: DEEP_WORKER_TEST_MODE=1, PYTEST_CURRENT_TEST, or 'pytest' in sys.modules)
   YES → Use InMemorySaver
   NO  → Go to 5

5. Default: Use SQLite (normal production run)
```

This priority ensures:
- Explicit flags always override automatic detection
- Tests default to memory (fast, isolated)
- Test fixtures can force SQLite for integration tests
- Production always uses SQLite unless explicitly disabled

## Timeline Estimate

- **Phase 1 (Core):** 2-3 hours
- **Phase 2 (CLI):** 30 minutes
- **Phase 3 (Testing):** 1-2 hours
- **Phase 4 (Docs):** 30 minutes

**Total:** 4-6 hours of development time
