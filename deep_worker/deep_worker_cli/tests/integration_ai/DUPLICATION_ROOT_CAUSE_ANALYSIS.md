# TUI Message Duplication - Root Cause Analysis

## Executive Summary

Message duplication in dwcli TUI has **TWO distinct causes**:

1. **Primary Cause**: Agent process persistence with InMemorySaver (accumulates history)
2. **Secondary Issue**: RPyC callback threading problem (doesn't display realtime messages)

## Problem 1: History Accumulation (Primary Cause)

### Root Cause

`InMemorySaver` stores checkpoint **in agent process memory**. Agent processes don't restart between test runs, so history **accumulates infinitely**.

### Evidence

With dirty agent (multiple test runs):
```
Messages in checkpoint: 58 (from previous runs!)
Test sends: 1 query
UI shows: 4+ duplicate pairs (limited by screen size)
```

With clean agent (fresh start):
```
Messages in checkpoint: 0
Test sends: 1 query
UI shows: 1 query + 1 response (NO DUPLICATION!)
```

### Impact

- **Severe**: Each test run adds messages to checkpoint
- **Cumulative**: Problem gets worse with each run
- **Flaky Tests**: Results depend on execution history

### Solution

Use `clean_agent` fixture to restart agent with clean state:

```python
def test_dwcli_query_visibility(clean_agent):
    agent_process = clean_agent("test_dwcli_query_visibility")
    # Agent now has empty InMemorySaver
```

See: `clean_agent_fixture.py` and `CLEAN_AGENT_USAGE.md`

---

## Problem 2: RPyC Callback Threading (Secondary)

### Root Cause

RPyC callbacks execute in **main thread** (not RPyC thread), causing:

```python
RuntimeError: The `call_from_thread` method must run in a different thread from the app
```

### Evidence

From logs:
```
[CALLBACK_HANDLER] handle_message called in RPyC thread
[CALLBACK_HANDLER] Calling self.call_from_thread...
[CALLBACK_HANDLER] ERROR: The `call_from_thread` method must run in a different thread from the app
```

### Why It Happens

`AgentService._notify_subscribers()` calls callback **synchronously** in the thread that called `exposed_invoke_agent()`, which is the **main asyncio thread**.

### Impact

- **Medium**: Realtime messages not displayed via callback
- **Workaround exists**: Messages still appear via invoke response
- **Not visible to users**: Functionally works, just not via intended path

### Current State

Messages ARE displayed correctly because:
1. `exposed_invoke_agent()` returns full response with content
2. dwcli receives response and displays it
3. Callback failure is silent - doesn't break functionality

### Why deepagents Doesn't Have This

deepagents creates **new process per session** with fresh `InMemorySaver`, so:
- No history accumulation
- Each run starts clean
- No duplication ever occurs

---

## Test Results

### Before Fix (Dirty Agent)

```
▎ > What is 2+2?
  4

▎ > What is 2+2?
  4

▎ > What is 2+2?
  4

▎ > What is 2+2?
  4
```

Messages in checkpoint: 58

### After Fix (Clean Agent)

```
▎ > What is 2+2?
  2+2 = 4

  Connected to agent 'test_dwcli_query_visibility'
```

Messages in checkpoint: 0
**NO DUPLICATION!**

---

## Recommendations

### For test_dwcli_query_visibility

**Required**: Use `clean_agent` fixture

```python
def test_dwcli_query_visibility(clean_agent):
    \"\"\"Test query visibility with CLEAN agent (no history).\"\"\"
    agent_process = clean_agent("test_dwcli_query_visibility", working_dir="/tmp")
    # Now guaranteed no duplicates
```

### For Other Tests

Choose based on test requirements:

**Use clean_agent when**:
- Testing initial state / first message
- Verifying empty history behavior
- Ensuring reproducible results
- Benchmarking

**Don't use clean_agent when**:
- Testing history loading/pagination
- Testing scroll with many messages
- Testing conversation continuity
- Testing history-dependent features

### For RPyC Callback Issue

**Low priority** - Functionally works via response path. If fixing:

1. Make `_notify_subscribers()` run in dedicated thread
2. Or use `asyncio.create_task()` instead of `call_from_thread()`
3. Or make callback truly async

---

## Files Created

1. `clean_agent_fixture.py` - Pytest fixture for clean agents
2. `CLEAN_AGENT_USAGE.md` - Usage guide with examples
3. `DUPLICATION_ROOT_CAUSE_ANALYSIS.md` - This document

## Files Modified (Logging Only)

- `interactive.py` - Added debug logging (can be removed)
- `agent_service.py` - Added debug logging (can be removed)

**Note**: All debug logging can be removed after confirming fix works.
