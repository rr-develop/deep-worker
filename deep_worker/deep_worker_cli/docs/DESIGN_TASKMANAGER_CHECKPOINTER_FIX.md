# Design: TaskManager Checkpointer and Thread ID Fix

**Date:** 2026-03-29
**Status:** Design Review

## 📋 Problem

### Current state

TaskManager threads are not displayed in `deep-worker --list` because of two critical problems:

1. **The wrong checkpointer:**
   - The TaskManager manager_agent uses `InMemorySaver` instead of `AsyncSqliteSaver`
   - The history of the `d1_tm` thread is NOT saved to the SQLite database
   - The `--list` command cannot find the thread, because it reads only from the database

2. **The wrong thread_id:**
   - `{agent_name}_tm` (an underscore) is used
   - It should be `{agent_name}:tm` (a colon)
   - Reason: to avoid conflicts with agents that have the `_tm` suffix in their name

### Requirements

1. The TaskManager MUST use the same checkpointer type as the main agent:
   - If the main agent → InMemorySaver, then the TaskManager → InMemorySaver
   - If the main agent → AsyncSqliteSaver, then the TaskManager → AsyncSqliteSaver

2. The thread ID for the TaskManager MUST be `{agent_name}:tm` (with a colon)

3. The config for the manager_agent MUST contain metadata:
   - `agent_name`: `{agent_name}:tm`
   - `assistant_id`: `{agent_name}:tm`
   - `updated_at`: an ISO timestamp

## 🔍 Analysis of the current implementation

### The checkpointer creation chain

```
agent_process.py:start()
  ↓
  self._checkpointer_ctx = sessions.get_checkpointer(agent_name)
  ↓
  checkpointer = await self._checkpointer_ctx.__aenter__()
  ↓
  agent, backend = create_cli_agent(checkpointer=checkpointer, ...)
  ↓
  [PROBLEM] The TaskManager is created WITHOUT being passed a checkpointer
  ↓
  TaskManager(agent_name, config, session_manager)
  ↓
  ManagerThread creates the agent WITHOUT a checkpointer
  ↓
  create_cli_agent() → falls back to InMemorySaver
```

### Thread ID usage

**manager_thread.py:39:**
```python
super().__init__(name=f"{agent_name}:tm", daemon=True)  # ✅ Correct for the thread name
```

**manager_thread.py:186:**
```python
{"configurable": {"thread_id": f"{self.agent_name}_tm"}}  # ❌ Wrong for the thread_id
```

**The mismatch:** The thread name is `d1:tm`, but the thread_id is `d1_tm`!

## 🎯 Design of the solution

### Component 1: Passing the checkpointer to the TaskManager

#### 1.1. Changes in `agent_process.py`

**Location:** `process/agent_process.py:320-324`

**BEFORE:**
```python
self.task_manager = TaskManager(
    agent_name=self.name,
    config=self.config,
    session_manager=self.session_manager
)
```

**AFTER:**
```python
# Pass the actual checkpointer instance (not context manager)
# If agent uses InMemorySaver, TaskManager will too
# If agent uses AsyncSqliteSaver, TaskManager will too
task_manager_checkpointer = checkpointer  # Same as main agent

self.task_manager = TaskManager(
    agent_name=self.name,
    config=self.config,
    session_manager=self.session_manager,
    checkpointer=task_manager_checkpointer,  # NEW
)
```

**Important:** What is passed is the `checkpointer` (the result of `__aenter__`), NOT `_checkpointer_ctx`!

#### 1.2. Changes in `task_manager.py`

**Location:** `task_manager/task_manager.py:47-63`

**BEFORE:**
```python
def __init__(
    self,
    agent_name: str,
    config: 'AgentConfig',
    session_manager: 'SessionManager'
):
```

**AFTER:**
```python
def __init__(
    self,
    agent_name: str,
    config: 'AgentConfig',
    session_manager: 'SessionManager',
    checkpointer: BaseCheckpointSaver | None = None,  # NEW
):
    """
    Initialize TaskManager.

    Args:
        agent_name: Name of the subordinate agent (for thread naming)
        config: Agent configuration for creating manager agent
        session_manager: For accessing conversation history
        checkpointer: Checkpointer for manager agent persistence.
                     MUST be same type as main agent:
                     - InMemorySaver if main agent uses InMemorySaver
                     - AsyncSqliteSaver if main agent uses AsyncSqliteSaver
                     If None, defaults to InMemorySaver (test mode)
    """
    self.agent_name = agent_name
    self.config = config
    self.session_manager = session_manager
    self.checkpointer = checkpointer  # NEW - store for ManagerThread
```

**Add the import:**
```python
from langgraph.checkpoint.base import BaseCheckpointSaver
```

#### 1.3. Changes in `manager_thread.py`

**Location:** `task_manager/manager_thread.py:96-105`

**BEFORE:**
```python
self.manager_agent, self.manager_backend = create_cli_agent(
    model=model,
    assistant_id=f"{self.agent_name}_tm",
    system_prompt=TASK_MANAGER_SYSTEM_PROMPT,
    auto_approve=True,
    enable_memory=False,
    enable_skills=False,
    enable_shell=False,
    project_memory_dir=project_memory_dir,
)
```

**AFTER:**
```python
self.manager_agent, self.manager_backend = create_cli_agent(
    model=model,
    assistant_id=f"{self.agent_name}:tm",  # ✅ Changed to colon
    system_prompt=TASK_MANAGER_SYSTEM_PROMPT,
    auto_approve=True,
    enable_memory=False,
    enable_skills=False,
    enable_shell=False,
    project_memory_dir=project_memory_dir,
    checkpointer=self.task_manager.checkpointer,  # ✅ NEW - use same as main agent
)
```

### Component 2: Fixing the thread_id from an underscore to a colon

**Location:** `task_manager/manager_thread.py:186`

**BEFORE:**
```python
result = await self.manager_agent.ainvoke(
    {"messages": [{"role": "user", "content": prompt}]},
    {"configurable": {"thread_id": f"{self.agent_name}_tm"}}
)
```

**AFTER:**
```python
# CRITICAL: Metadata field meanings:
# - thread_id: Unique identifier for this conversation thread (agent_name:tm)
# - assistant_id: Technical ID for the assistant/agent instance (agent_name:tm)
# - agent_name: The MAIN agent name (WITHOUT :tm suffix) - identifies which agent this thread belongs to
# - updated_at: ISO 8601 UTC timestamp of last update
#
# This ensures TaskManager threads are associated with the parent agent in --list output
result = await self.manager_agent.ainvoke(
    {"messages": [{"role": "user", "content": prompt}]},
    {
        "configurable": {"thread_id": f"{self.agent_name}:tm"},
        "metadata": {
            "assistant_id": f"{self.agent_name}:tm",
            "agent_name": self.agent_name,  # Parent agent name (without :tm)
            "updated_at": datetime.now(UTC).isoformat(),
        }
    }
)
```

**Add the import:**
```python
from datetime import UTC, datetime
```

### Component 3: Updating sessions.py to support the colon

**A potential problem:** A colon in a thread_id may be an invalid character.

#### Validity check

Check whether SQLite and LangGraph support a colon in a thread_id:

1. **SQLite:** A colon is allowed in VARCHAR/TEXT fields ✅
2. **LangGraph RunnableConfig:** the thread_id is just a string, any characters are allowed ✅
3. **AsyncSqliteSaver:** Uses the thread_id as a string key ✅

**Conclusion:** A colon in a thread_id is VALID and safe.

#### A possible problem with CLI arguments

If `deep-worker --list --agent "agent:tm"` is used in a shell, the colon may need escaping:
```bash
deep-worker --list --agent "agent:tm"  # ✅ Works with quotes
deep-worker --list --agent agent:tm    # ⚠️ May need escaping in some shells
```

**Solution:** Document the use of quotes for names containing a colon.

## 📦 Affected files

### 1. `process/agent_process.py`
- **Line ~320:** Pass the `checkpointer` to `TaskManager()`
- **Imports:** All the necessary imports are already there

### 2. `task_manager/task_manager.py`
- **Lines 47-63:** Add the `checkpointer` parameter to `__init__`
- **Line 63:** Store `self.checkpointer = checkpointer`
- **Imports:** Add `from langgraph.checkpoint.base import BaseCheckpointSaver`

### 3. `task_manager/manager_thread.py`
- **Line 98:** Change `assistant_id` from `_tm` to `:tm`
- **Line 105:** Add `checkpointer=self.task_manager.checkpointer`
- **Line 186:** Change the `thread_id` from `_tm` to `:tm` and add metadata
- **Imports:** Add `from datetime import UTC, datetime`

### 4. `sessions.py`
- **No changes:** a thread_id with a colon is already supported

## 🧪 Testing plan

### Unit tests

**1. Test: TaskManager receives correct checkpointer type**

```python
def test_taskmanager_receives_checkpointer(agent_config):
    """Test TaskManager gets same checkpointer as main agent."""
    process = AgentProcess(
        name="test_agent",
        config=agent_config,
        enable_task_manager=True,
        use_sqlite=True,  # Force SQLite
    )

    # Mock checkpointer
    mock_checkpointer = MagicMock()

    with patch("deep_worker_cli.sessions.get_checkpointer"):
        with patch("deep_worker_cli.agent.create_cli_agent") as mock_create:
            mock_create.return_value = (MagicMock(), MagicMock())

            await process.start()

            # Verify TaskManager received checkpointer
            assert process.task_manager is not None
            assert process.task_manager.checkpointer is mock_checkpointer
```

**2. Test: Manager agent uses colon in thread_id**

```python
def test_manager_thread_uses_colon_in_thread_id():
    """Test manager thread uses 'agent:tm' not 'agent_tm'."""
    # Create mock manager agent call
    # Verify thread_id is "test_agent:tm"
    # Verify metadata contains "test_agent:tm"
```

### Integration tests

**1. Test: TaskManager thread appears in --list**

```python
@pytest.mark.asyncio
async def test_taskmanager_thread_in_list():
    """Test TaskManager thread appears in deep-worker --list."""
    agent_name = "test_tm_agent"

    # Start agent with TaskManager and SQLite
    start_agent(agent_name, extra_args=["--use-sqlite", "-tm"])

    # Send message to trigger TaskManager evaluation
    send_message(agent_name, "Create file test.txt")

    # Query --list
    result = subprocess.run(
        ["deep-worker", "--list", "--format", "json"],
        capture_output=True
    )

    threads = json.loads(result.stdout)["threads"]
    thread_ids = [t["thread_id"] for t in threads]

    # Verify both main agent and TaskManager threads exist
    assert agent_name in thread_ids
    assert f"{agent_name}:tm" in thread_ids  # ✅ With colon

    # Verify TaskManager thread has metadata
    tm_thread = [t for t in threads if t["thread_id"] == f"{agent_name}:tm"][0]
    assert tm_thread["agent_name"] == f"{agent_name}:tm"
    assert tm_thread["updated_at"] is not None
```

**2. Test: TaskManager uses InMemorySaver in test mode**

```python
def test_taskmanager_uses_memory_in_tests():
    """Test TaskManager uses InMemorySaver in test environment."""
    # Start agent WITHOUT --use-sqlite (test mode)
    # Verify TaskManager thread does NOT appear in --list
    # Verify TaskManager still functions correctly
```

## ⚠️ Potential problems

### 1. The colon in the thread_id

**Problem:** Some systems may interpret the colon as a special character.

**Solution:**
- The colon is valid for SQLite, LangGraph and Python strings ✅
- Document the use of quotes in the CLI: `--agent "agent:tm"`
- Alternative: Use a hyphen `agent-tm` instead of a colon (less preferable)

**Status:** ✅ FEASIBLE with documentation

### 2. Migrating the existing thread_ids

**Problem:** Existing threads with `agent_tm` will not be renamed to `agent:tm` automatically.

**Solution:**
- New threads will use `agent:tm`
- Old threads with `agent_tm` will stay in the database (they do not conflict)
- Optionally: Create a migration script for renaming

**Status:** ✅ FEASIBLE (backward compatible)

### 3. Checkpointer lifecycle

**Problem:** The main agent owns `_checkpointer_ctx` and closes it in `cleanup()`.

**Question:** Should the TaskManager close its checkpointer separately?

**Analysis:**
```python
# agent_process.py:192
async def cleanup(self):
    if self._checkpointer_ctx:
        await self._checkpointer_ctx.__aexit__(None, None, None)
```

- If we pass the `checkpointer` (the result of `__aenter__`), then the TaskManager must NOT close it
- The lifecycle is managed by the main agent through `_checkpointer_ctx`
- The TaskManager simply uses the already-opened checkpointer

**Solution:** The TaskManager does NOT close the checkpointer (the main agent manages the lifecycle).

**Status:** ✅ FEASIBLE (the correct design)

### 4. Name conflicts with the colon

**Problem:** An agent named `agent:tm` may conflict with the TaskManager thread of the agent `agent`.

**Analysis:**
- This is a VERY rare edge case
- It requires explicitly creating an agent whose name contains `:tm`
- Document it as a limitation: "Do not use the `:tm` suffix in agent names"

**Solution:** Add validation in `agent_command.py`:
```python
if ":tm" in agent_name:
    print("Error: Agent name cannot contain ':tm' suffix (reserved for TaskManager)")
    return 1
```

**Status:** ✅ FEASIBLE with validation

## 📝 Documentation

### README.md updates

Add a section about TaskManager threads:

```markdown
## TaskManager Threads

When using `--task-manager` (`-tm`), two threads are created:

1. **Main agent thread:** `{agent_name}` - Executes tasks
2. **Manager thread:** `{agent_name}:tm` - Evaluates task completion

Both threads share the same checkpointer type:
- In production: AsyncSqliteSaver (persisted to a per-agent threads database in the user data directory)
- In tests: InMemorySaver (memory-only, not persisted)

View both threads:
```bash
deep-worker --list --agent "my_agent"
# Output (note - both threads show "My Agent" as the agent name):
# Thread ID         Agent Name   Last Updated
# ============================================
# my_agent         my_agent     Feb 29, 3:45pm
# my_agent:tm      my_agent     Feb 29, 3:46pm
#
# IMPORTANT: The "Agent Name" column shows the main agent (my_agent),
# NOT the thread name. This is correct, because the TaskManager is part
# of the same agent, it just uses a separate thread for its work.
```

**Note:** Use quotes when filtering by TaskManager thread names containing colons.
```

## 🎯 Implementation plan

### Phase 1: Core changes
1. ✅ Add the `checkpointer` parameter to `TaskManager.__init__`
2. ✅ Pass the checkpointer to `TaskManager()` from `agent_process.py`
3. ✅ Use the checkpointer in `create_cli_agent()` in `manager_thread.py`

### Phase 2: Thread ID fix
4. ✅ Change `assistant_id` from `_tm` to `:tm`
5. ✅ Change the `thread_id` config from `_tm` to `:tm`
6. ✅ Add metadata to the RunnableConfig

### Phase 3: Validation
7. ✅ Add agent name validation (forbid the `:tm` suffix)
8. ✅ Add unit tests
9. ✅ Add integration tests

### Phase 4: Documentation
10. ✅ Update README.md
11. ✅ Update the docstrings

## ✅ Final status

### Feasibility

**ALL the requirements are FEASIBLE:**

1. ✅ **The TaskManager uses the same checkpointer as the main agent** - FEASIBLE
   - Passing the checkpointer through parameters
   - The lifecycle is managed by the main agent

2. ✅ **A thread ID with a colon `agent:tm`** - FEASIBLE
   - The colon is valid for SQLite and LangGraph
   - It requires documentation for CLI usage

3. ✅ **Metadata in the config** - FEASIBLE
   - Analogous to the recent fix for the main agent
   - Add `agent_name`, `assistant_id`, `updated_at`

### Potential limitations

1. ⚠️ **Backward compatibility:** Old `agent_tm` threads will stay in the database
   - **Impact:** Minimal, they do not conflict with the new ones
   - **Solution:** An optional migration

2. ⚠️ **CLI usage:** Quotes are required for names containing a colon
   - **Impact:** Documentation
   - **Solution:** `--agent "agent:tm"`

3. ⚠️ **Edge case:** A conflict if an agent is deliberately named `agent:tm`
   - **Impact:** A very rare case
   - **Solution:** Agent name validation

## 🔗 Related documents

- Commit e2723b1: Fix the metadata for the main agents
- `docs/DESIGN_CHECKPOINTER_FIX.md`
- `task_manager/README.md` (if it exists)

---

**Generated:** 2026-03-29
**Review Status:** Ready for implementation
**Estimated Effort:** 2-3 hours implementation + 1-2 hours testing
