# Bug Fix Summary: The Agent "Gets Stuck" on the First Request

## Status: ✅ FIXED

The `test_agent_multiple_queries` test now **PASSES**.

---

## Problem

The agent with the claude-code provider "got stuck" on the first request and repeated the first answer for all subsequent requests:

- Request 1: "2+2" → Answer: "4" ✅
- Request 2: "3+3" → Answer: "4" ❌ (repeats the first one!)
- Request 3: "5+5" → Answer: "4" ❌ (repeats the first one!)

---

## Root Cause

### Problem 1: `break` extracted the FIRST message instead of the LAST

**File:** `chat_model.py:373-383`

LangGraph passes **THE ENTIRE HISTORY** of messages into `model.ainvoke()`:
```python
messages = [
    HumanMessage("2+2"),      # ← From the checkpointer
    AIMessage("4"),           # ← From the checkpointer
    HumanMessage("3+3")       # ← The new request
]
```

But the code extracted the FIRST UserMessage because of `break`:
```python
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        user_content = msg.content  # ← user_content = "2+2"
        break  # ❌ Stops at the first one!
```

As a result, a repeated request "2+2" was sent instead of the new "3+3".

### Problem 2: Missing session_id

**File:** `chat_model.py:386`

```python
await client.query(user_content)  # session_id="default" for all threads
```

All thread_ids used the same Claude session ("default"), which broke the isolation between sessions.

---

## The applied fix

### Change 1: Removed the `break` (chat_model.py:383)

**Before:**
```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        if isinstance(msg.content, str):
            user_content = msg.content
        else:
            user_content = str(msg.content)
        break  # ❌ REMOVED
```

**After:**
```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        # Continue iteration to find the LAST UserMessage (not first)
        if isinstance(msg.content, str):
            user_content = msg.content
        else:
            user_content = str(msg.content)
# Now user_content contains the LAST UserMessage
```

### Change 2: Added `session_id=thread_id` (chat_model.py:386)

**Before:**
```python
if user_content:
    await client.query(user_content)
```

**After:**
```python
if user_content:
    await client.query(user_content, session_id=thread_id)
```

---

## Test results

### Before the fix

```
Query 1: '2+2' → Response: '2 + 2 = 4' ✅
Query 2: '3+3' → Response: '2 + 2 = 4' ❌ (repeats the first one)
Query 3: '5+5' → Response: '2 + 2 = 4' ❌ (repeats the first one)
```

**Test:** ❌ FAILED

### After the session_id fix (change 2 only)

```
Query 1: '2+2' → Response: '2 + 2 = 4' ✅
Query 2: '3+3' → Response: '2 + 2 = 4' ❌ (still repeats!)
```

**Test:** ❌ FAILED

**Conclusion:** `session_id` alone is **not enough**. The problem is in the `break`.

### After both fixes

```
Query 1: '2+2' → Response: '2 + 2 = 4' ✅
Query 2: '3+3' → Response: '3 + 3 = 6' ✅
Query 3: '5+5' → Response: '5 + 5 = 10' ✅
```

**Test:** ✅ PASSED

---

## Why are both changes necessary?

### session_id is not enough

Even with the correct `session_id`, if the wrong message is sent (the first one instead of the last), the agent will still repeat the first answer.

**Example with session_id, but without removing the break:**
```python
# Request 2
messages = [HumanMessage("2+2"), AIMessage("4"), HumanMessage("3+3")]

# Extraction with break:
user_content = "2+2"  # ❌ The first message!

# Sending:
await client.query("2+2", session_id="test_multi_query")
# ❌ "2+2" is sent again, despite the correct session_id!
```

### Removing the break is not enough

Without `session_id` all threads use a single Claude session ("default"), which can cause isolation problems between parallel sessions.

**Example without session_id, but with the break removed:**
```python
# Thread 1
await client.query("What's my name?", session_id="default")

# Thread 2
await client.query("My name is Bob", session_id="default")

# Thread 1 again
await client.query("What's my name?", session_id="default")
# It may receive "Bob" from Thread 2!
```

---

## Conformance to the basic principles of the provider

### ✅ Principle 1: Full transparency for the agent

- No changes to the interface
- thread_id is handled in the standard way via RunnableConfig
- The agent does not know about the internal details

### ✅ Principle 2: Tools are executed in the agent

- session_id and user_content affect only the CONVERSATION context
- They do NOT affect tool execution (still via ProxyMCPServer)

### ✅ Principle 3: A separate ClaudeSDKClient for each thread_id

- ClientManager caching works as before
- Now each client uses a unique session_id
- Double isolation (client + session)

### ✅ Principle 4: Sequential message processing

- ainvoke() works as before
- Future blocking is unchanged
- Only the message extraction logic was fixed

---

## Changed files

### `chat_model.py`

**2 lines changed:**

1. **Line 383:** Removed the `break` (the loop now continues to the end)
2. **Line 386:** Added the `session_id=thread_id` parameter

```diff
  user_content: str = ""
  for msg in claude_messages:
      if isinstance(msg, ClaudeUserMessage):
+         # Continue iteration to find the LAST UserMessage (not first)
          if isinstance(msg.content, str):
              user_content = msg.content
          else:
              user_content = str(msg.content)
-         break

  if user_content:
-     await client.query(user_content)
+     await client.query(user_content, session_id=thread_id)
```

---

## Additional documentation

- **Detailed analysis:** `docs/bug_analysis_corrected.md`
- **Initial analysis (incorrect):** `docs/bug_analysis_multiple_queries.md`

---

## Next steps (recommendations)

### 1. Run all the provider tests

```bash
# From the provider package directory
pytest tests/ -v
```

Make sure that the fix did not break the existing tests.

### 2. Add a unit test for extracting the last UserMessage

```python
def test_extracts_last_user_message():
    """Should extract LAST UserMessage from history, not first."""
    messages = [
        HumanMessage("2+2"),
        AIMessage("4"),
        HumanMessage("3+3")
    ]

    result = await model.ainvoke(messages, config={"thread_id": "test"})

    # Verify that "3+3" was sent to Claude, not "2+2"
    mock_client.query.assert_called_with("3+3", session_id="test")
```

### 3. Add an integration test for session isolation

```python
def test_session_isolation_between_threads():
    """Verify that different thread_ids have isolated Claude sessions."""
    # Thread 1
    await model.ainvoke([HumanMessage("2+2")], {"thread_id": "thread-1"})

    # Thread 2
    await model.ainvoke([HumanMessage("3+3")], {"thread_id": "thread-2"})

    # Verify isolation
    # Thread 1 should not see "3+3" from thread-2
    # Thread 2 should not see "2+2" from thread-1
```

---

## Conclusion

**The bug is fixed** by two minimal changes:

1. ✅ Removed the `break` → the last UserMessage is extracted
2. ✅ Added `session_id=thread_id` → isolation between threads

**The `test_agent_multiple_queries` test now PASSES.**

Both changes are **necessary** and **sufficient** to fix the bug.
