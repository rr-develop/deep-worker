# Test Agent Lifecycle Management

## Critical Principle

**Tests MUST always test the current code, never old agent processes.**

When running integration tests with deep-worker agents, it is critical that tests run against freshly started agents with the latest code changes. Testing against old running agents leads to:

- False test failures (bug already fixed, but old agent still has it)
- False test passes (bug introduced, but old agent doesn't have it)
- Hours of debugging non-existent problems
- Attempts to fix code that is already correct

## Default Behavior: Always Kill Old Processes

**Default behavior for test fixtures:**

1. **Kill any existing agent process** with the same name
2. **Wait for process to actually die** (not just send kill signal!)
3. **Verify process removed from registry**
4. **Verify process no longer exists in OS**
5. Only then start new agent process
6. Wait for new agent to register

**This is the ONLY safe default behavior.**

## Implementation in Test Fixtures

### Required Steps When Starting Test Agent

```python
from deep_worker_cli.process.registry import ProcessRegistry
import os
import signal
import time

registry = ProcessRegistry()
agent_name = "test_agent_name"

# 1. Kill any existing agent with this name
existing = registry.lookup(agent_name)
if existing:
    logger.info(f"Killing existing agent PID {existing.pid}")
    try:
        os.kill(existing.pid, signal.SIGKILL)
    except ProcessLookupError:
        logger.info(f"Process {existing.pid} already dead")

    # 2. CRITICAL: Wait for process to actually die
    kill_timeout = 5.0
    kill_start = time.time()
    process_dead = False

    while time.time() - kill_start < kill_timeout:
        # Check if still in registry
        if not registry.lookup(agent_name):
            process_dead = True
            logger.info(f"✓ Process confirmed dead and removed from registry")
            break

        # Also check if process exists in OS
        try:
            os.kill(existing.pid, 0)  # signal 0 checks if process exists
            logger.debug(f"Process {existing.pid} still exists, waiting...")
        except ProcessLookupError:
            logger.info(f"✓ Process no longer exists in OS")
            time.sleep(0.2)  # Give registry time to update
            if not registry.lookup(agent_name):
                process_dead = True
                logger.info(f"✓ Confirmed removal from registry")
                break

        time.sleep(0.1)

    # 3. CRITICAL: Fail if process didn't die
    if not process_dead:
        raise RuntimeError(
            f"Old agent process {existing.pid} did not die within {kill_timeout}s. "
            f"Cannot safely start new agent - would test old code!"
        )

# 4. Now safe to start fresh agent
# ... start agent code ...
```

### Why Both Registry and OS Checks?

We check both because:

1. **Registry check**: Fast, but may lag behind reality
2. **OS check** (`os.kill(pid, 0)`): Authoritative, confirms process truly dead
3. **Final registry check**: Ensures registry caught up with reality

This belt-and-suspenders approach prevents race conditions.

## What NOT To Do

### ❌ WRONG: Just sleep and hope

```python
os.kill(existing.pid, signal.SIGKILL)
time.sleep(0.5)  # ❌ Hope process died, don't check!
# Start new agent - DANGER: old process might still be running
```

**Problem**: 0.5s may not be enough. Process might still be alive. Race condition.

### ❌ WRONG: Ignore errors silently

```python
try:
    os.kill(existing.pid, signal.SIGKILL)
    time.sleep(0.5)
except ProcessLookupError:
    pass  # ❌ Ignore and assume it's fine
```

**Problem**: Don't know if process actually died. Registry might still have stale entry.

### ❌ WRONG: Skip killing old process

```python
# Just start new agent, don't check for old one
agent_process = subprocess.Popen(["deep-worker", agent_name])
```

**Problem**: If old agent running, new agent may fail to bind port, or worse - test might connect to old agent!

## Use Existing Fixtures

Two fixtures implement the correct lifecycle:

### 1. `clean_agent` fixture

**File**: `tests/integration_ai/clean_agent_fixture.py`

**Usage**:
```python
def test_something(clean_agent):
    agent_process = clean_agent("my_test_agent", working_dir="/tmp")
    # Agent is guaranteed fresh with current code
```

### 2. `slow_tool_agent` fixture

**File**: `tests/integration_ai/test_slow_tool_display.py`

**Usage**:
```python
def test_slow_tool(slow_tool_agent):
    agent_name = slow_tool_agent
    # Agent "test_slow_tool_display" is guaranteed fresh
```

Both fixtures implement the correct kill-verify-start lifecycle.

## Optional Flag to Reuse Existing Agent

For debugging and faster iteration, both `clean_agent` and `slow_tool_agent` fixtures support the `@pytest.mark.reuse_agent` marker.

### Usage with `clean_agent` fixture

```python
@pytest.mark.reuse_agent
def test_something(clean_agent):
    agent_process = clean_agent("test_my_agent")
    # Will reuse existing agent if running, won't kill it
    # Great for debugging - keeps agent state between test runs
```

### Usage with `slow_tool_agent` fixture

```python
@pytest.mark.reuse_agent
async def test_slow_tool(slow_tool_agent):
    agent_name = slow_tool_agent
    # Will reuse existing agent "test_slow_tool_display" if running
    # Won't kill or restart the agent
    # Connect manually with: dwcli test_slow_tool_display
```

### When to Use Reuse Mode

**Good use cases:**
- Manual debugging - keep agent running between test runs
- Inspecting agent state after test failure
- Faster iteration - skip agent startup time
- Connecting to agent manually with `dwcli` during debugging

**Example debugging workflow:**
```bash
# 1. Run test with reuse_agent marker
pytest tests/integration_ai/test_slow_tool_display.py::test_slow_tool -k reuse_agent

# 2. Test fails, agent still running
# 3. Connect manually to inspect
dwcli test_slow_tool_display

# 4. Run test again (reuses same agent)
pytest tests/integration_ai/test_slow_tool_display.py::test_slow_tool -k reuse_agent

# 5. When done debugging, kill agent manually
pkill -f "deep-worker test_slow_tool_display"
```

### Important Notes

1. **Default is still safe**: Without `@pytest.mark.reuse_agent`, fixtures always kill old processes
2. **Reuse does NOT clean history**: Agent keeps all previous conversation history
3. **Manual cleanup required**: You must kill reused agents manually when done debugging
4. **Use only for debugging**: Never use in CI/CD pipelines

**This is NOT the default!** Default must always be safe (kill old, start fresh).

## Summary

1. **Default behavior**: Always kill old agent processes before starting test
2. **Always verify** process actually died (don't just send signal and hope)
3. **Check both** registry and OS to avoid race conditions
4. **Fail loudly** if old process won't die
5. Only start new agent when 100% certain old one is gone

This prevents the nightmare scenario of debugging for hours because tests ran against old code.
