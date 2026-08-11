# Summary: Tool Call Duplication Bug

## What was discovered

**Bug:** After a query with a tool call, subsequent queries show OLD tool calls in the TUI.

**Reproduction:**
- Query 1: "What is your current working directory?" → shows `execute("pwd")` ✅
- Query 2: "2+2=" → shows `execute("pwd")` again ❌ (BUG!)

## Testing

### ✅ A test was created for the deepseek provider

**File:** `tests/integration_ai/test_agent_tool_call_duplication_deepseek.py`

**Result:** ❌ FAILED - the bug reproduces!

### ✅ A test was created for the claude-code provider

**File:** `tests/integration_ai/test_agent_tool_call_duplication_claude_code.py`

**Result:** ❌ FAILED - the bug reproduces!

### Conclusion: a COMMON BUG affecting ALL providers

This is NOT specific to claude-code. It is a bug in the agent.

## Root cause

**Commit:** `024dcbc` - "fix: Make tool calls visible in non-streaming mode (TUI display)"

**Date:** Wed Mar 11 00:19:48 2026 (yesterday)

**What was added:** Code to broadcast tool call notifications in non-streaming mode

**Problem:** The code broadcasts **THE ENTIRE history** of messages, not just the new ones:

```python
# agent_process.py, lines ~933-978
if notify_callback is not None:
    for msg in messages:  # ← "messages" contains the ENTIRE thread history!
        if msg_tool_calls:
            notify_callback({"type": "tool_call", ...})  # ← Broadcasts old tool calls!
```

**Why:**
- The LangGraph Checkpointer stores the full history
- `result['messages']` = all messages since the start of the session
- The code iterates over ALL of them and broadcasts, including old tool calls

**Why there is no bug in streaming mode:**
- `astream()` returns only NEW events
- It does not return the entire history

## Solution

### Location

**File:** `process/agent_process.py`

**Class:** `AgentProcess`

### Changes

**1. Add to `__init__`:**

```python
self._last_broadcasted_count: dict[str, int] = {}
```

**2. Change the notification loop (lines ~933-978):**

```python
if notify_callback is not None:
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    # Broadcast only the NEW messages
    last_broadcasted = self._last_broadcasted_count.get(thread_id, 0)
    new_messages = messages[last_broadcasted:]

    for msg in new_messages:  # ← Only the new ones, not the entire history!
        # ... existing broadcast logic ...

    # Update the counter
    self._last_broadcasted_count[thread_id] = len(messages)
```

### Analogy

This is **EXACTLY THE SAME LOGIC** as in `ClaudeCodeChatModel.ainvoke()`:

```python
# ClaudeCodeChatModel
sent_count = self._client_manager.get_sent_count(thread_id)
new_messages = messages[sent_count:]
# ... send only new_messages ...
self._client_manager.update_sent_count(thread_id, len(messages))
```

## Documentation

**Detailed analysis:** `docs/analysis_tool_duplication_bug_updated.md`

**Includes:**
- Terminal snapshots for both providers
- Git history analysis
- A detailed explanation of the root cause
- A full technical specification of the solution
- The execution flow before and after the fix

## Important

### ✅ What is correct:

- ClaudeCodeChatModel works correctly
- The message mirroring architecture is correct
- Streaming mode works correctly

### ❌ What is incorrect:

- ONLY the notification loop in non-streaming mode
- ONLY as of commit 024dcbc (yesterday)

### 🎯 The fix:

- ✅ Is not provider-specific
- ✅ Improves the agent as a whole
- ✅ Is simple and understandable
- ✅ Is analogous to the existing pattern (sent_count)
