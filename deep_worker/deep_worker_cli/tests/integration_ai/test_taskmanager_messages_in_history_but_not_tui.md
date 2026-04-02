# Test Report: test_taskmanager_messages_in_history_but_not_tui

## Test Information
- **Test Name:** `test_taskmanager_messages_in_history_but_not_tui`
- **Test File:** `deep_worker/deep_worker_cli/tests/integration_ai/test_taskmanager_tui.py`
- **Date:** 2026-03-16
- **Status:** ❌ **FAILED**

## Test Execution Result

**Execution Time:** 9.41s
**Exit Code:** FAILED

## Error Output

```
Failed: Expected 'TaskManager:>' to appear in 'dwcli --history' output, but it was not found.
The TaskManager may not have sent any corrective prompts, or the checkpointer is not storing them.

History output:
Agent 'test_taskmanager_tui' may not be running.
Use 'deep-worker-cli --list' to see running agents
```

## Problem Description

### Issue: Agent Not Running

The test failed because the agent `test_taskmanager_tui` is **not running** when the test tries to query its history.

**Test Flow:**
1. Test calls `dwcli -p <prompt> test_taskmanager_tui` in headless mode
2. Test calls `dwcli --history test_taskmanager_tui --limit 50` to check history
3. **FAILURE:** Agent is not running, history command returns error

### Root Cause

**Primary Cause:** Test fixture issue or test setup problem

The test relies on the `tm_agent` fixture which should:
1. Start a clean agent with TaskManager enabled
2. Keep the agent running for the test duration
3. Clean up after test completes

However, the agent is **not running** when the test executes, which indicates:
- Fixture failed to start agent, OR
- Agent crashed/stopped before test started, OR
- Test is running in wrong order (dependent on previous test)

### Evidence

**Error message:** `"Agent 'test_taskmanager_tui' may not be running."`

This is the standard error message from dwcli when trying to access a non-running agent.

**No history output** - Cannot verify whether TaskManager messages exist in history because agent never ran.

## Degree of Certainty

**Certainty Level:** 🟡 **MEDIUM (60%)**

**What IS certain:**
- Agent is not running when test executes
- Test cannot complete without running agent
- Either fixture failed OR agent crashed

**What is NOT certain:**
- Whether fixture `tm_agent` is correctly implemented
- Whether agent crashes on startup with TaskManager mode
- Whether this test depends on previous test (`test_taskmanager_messages_visible_in_tui`)
- Whether TaskManager mode itself has bugs preventing agent startup

## Impact Assessment

**Severity:** 🟡 **MEDIUM** (Test infrastructure issue, not core functionality bug)

**User Impact:**
- No direct impact - test failure, not production bug
- However, indicates potential issues with:
  - Test fixtures
  - Agent lifecycle management
  - TaskManager mode stability

**Test Impact:**
- Cannot verify TaskManager message visibility
- Cannot verify history vs TUI consistency
- Test suite incomplete without this validation

## Recommendation

### Investigation Required

1. **Check `tm_agent` fixture implementation:**
   - File: `test_taskmanager_tui.py`
   - Look for `@pytest.fixture` named `tm_agent`
   - Verify it starts agent correctly
   - Check if it waits for agent to be ready

2. **Check agent logs:**
   ```bash
   ls -la ~/.deep_worker/logs/test_taskmanager_tui.*
   cat ~/.deep_worker/logs/test_taskmanager_tui.out.log
   cat ~/.deep_worker/logs/test_taskmanager_tui.err.log
   ```

3. **Test manual agent startup:**
   ```bash
   deep-worker test_taskmanager_tui --dir /tmp --task-manager --skip-permissions
   # Wait for agent to start
   dwcli --list
   # Should show test_taskmanager_tui
   ```

4. **Check test dependencies:**
   - Does this test depend on `test_taskmanager_messages_visible_in_tui` completing first?
   - Should they share the same agent instance?
   - Or should each test have independent agent?

### Possible Fixes

**Option 1: Fix fixture to ensure agent starts**
```python
@pytest.fixture
def tm_agent():
    agent_name = "test_taskmanager_tui"
    # Start agent
    process = subprocess.Popen([
        "deep-worker", agent_name,
        "--dir", "/tmp",
        "--task-manager",
        "--skip-permissions"
    ])

    # CRITICAL: Wait for agent to actually start
    time.sleep(5)  # Or better: poll until agent appears in registry

    # Verify agent is running
    result = subprocess.run(["dwcli", "--list"], capture_output=True, text=True)
    assert agent_name in result.stdout, f"Agent {agent_name} failed to start"

    yield agent_name

    # Cleanup
    process.terminate()
    process.wait(timeout=10)
```

**Option 2: Use existing fixture infrastructure**

Check if there's a `clean_agent` fixture with `--task-manager` support:
```python
@pytest.fixture
def tm_agent(clean_agent):
    return clean_agent("test_taskmanager_tui", task_manager=True)
```

**Option 3: Combine both tests into single test**

Since both tests use same agent, combine them:
```python
def test_taskmanager_tui_and_history(tm_agent):
    # 1. Check TUI shows TaskManager messages
    # 2. Check history shows TaskManager messages
    # 3. Verify consistency
```

### What NOT to do

- ❌ Don't skip this test - it validates important functionality
- ❌ Don't ignore agent startup failure - it may indicate real bug
- ❌ Don't assume agent is running without verification

## Related Issues

1. **Previous test failure** - `test_taskmanager_messages_visible_in_tui` also failed
   - That test DID have agent running
   - But had different issue (input area invisible)
   - This suggests tests may not share agent instance correctly

2. **Potential TaskManager bugs:**
   - If agent crashes ONLY with `--task-manager` flag
   - That would indicate TaskManager mode has initialization bugs
   - Need to verify agent can start with TaskManager enabled

## Notes

- This test is testing architecture principle: history and TUI should be consistent
- Cannot validate this principle without running agent
- Fixture implementation is critical for test reliability
- May need to review entire test file's fixture strategy

## Test Verdict

**Test Result:** ❌ FAILED (infrastructure issue)

**Cannot verify intended functionality** because agent not running.

**Next Steps:**
1. Fix agent startup in fixture
2. Verify TaskManager mode doesn't crash agent
3. Re-run test after fixture fix
4. Check if TaskManager messages actually appear in history/TUI
