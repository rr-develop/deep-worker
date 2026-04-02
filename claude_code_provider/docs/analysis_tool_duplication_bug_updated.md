# Bug Analysis: Duplicated Tool Calls in the TUI (UPDATED)

## Test results

### ✅ The test confirmed: this is a COMMON BUG affecting ALL providers

**Test of the deepseek provider:**
```bash
cd /tmp && pytest test_agent_tool_call_duplication_deepseek.py -v -s
```

**Result:** ❌ FAILED

**Terminal snapshot (deepseek):**
```
> What is your current working directory?
⏺ execute("pwd")    ← Query 1: correct
⎿ /tmp

> 2+2=
⏺ execute("pwd")    ← Query 2: BUG! Duplicated tool call!
⎿ /tmp
4
```

**Conclusion:**
- ❌ The bug reproduces for the deepseek provider
- ❌ The bug reproduces for the claude-code provider
- ✅ This is a COMMON bug in the agent, NOT provider-specific

## History: When the bug appeared

### Git History Analysis

```bash
git log --oneline -20 -- process/agent_process.py
```

**Key commit:**
```
024dcbc fix: Make tool calls visible in non-streaming mode (TUI display)
```

**Date:** Wed Mar 11 00:19:48 2026 +0100 (yesterday!)

**Commit message (excerpts):**
> CRITICAL BUG FIX: Tool execution was completely invisible to users in non-streaming mode.
>
> ROOT CAUSE:
> In agent_process.py lines 931-1033, the non-streaming iteration loop only sent
> the final assistant response notification. It NEVER sent tool_call or tool_result
> notifications, unlike streaming mode which properly sends all message types.
>
> FIX:
> 1. Added notification loop (lines 946-991) that iterates through ALL messages
>    in each iteration and sends appropriate notifications:
>    - tool_call: {"type": "tool_call", "name": "execute", "args": {...}}
>    - tool_result: {"type": "tool_result", "content": "..."}

### What changed in commit 024dcbc

**File:** `process/agent_process.py`

**Code added (lines +949 through +991):**

```python
# CRITICAL FIX: Notify ALL messages in this iteration
# This ensures tool_call and tool_result messages are visible to users
# Similar to streaming mode behavior in stream_processor.py
if notify_callback is not None:
    for msg in messages:  # ← PROBLEM: "messages" contains the ENTIRE history!
        # Extract message data
        if isinstance(msg, dict):
            msg_role = msg.get("role", "")
            msg_content = msg.get("content", "")
            msg_tool_calls = msg.get("tool_calls")
        elif hasattr(msg, "type"):
            msg_role = msg.type
            msg_content = getattr(msg, "content", "")
            msg_tool_calls = getattr(msg, "tool_calls", None)
        else:
            continue

        # Send tool_call notifications
        if msg_tool_calls:
            for tool_call in msg_tool_calls:
                # ... notify tool_call ...

        # Send tool_result notifications
        if msg_role == "tool":
            # ... notify tool_result ...
```

## Root cause of the bug

### The problem in commit 024dcbc

**THE INTENT (correct):**
- Make tool calls visible in non-streaming mode
- Broadcast tool_call and tool_result notifications
- Similar to the behavior of streaming mode

**THE IMPLEMENTATION (INCORRECT):**
```python
for msg in messages:  # ← "messages" = the ENTIRE thread history!
    if msg_tool_calls:
        notify_callback({"type": "tool_call", ...})
```

**Why this is a bug:**

1. **The LangGraph Checkpointer** stores the full history in the state
2. `result['messages']` contains **ALL messages** since the start of the session
3. The code iterates over **ALL** messages and broadcasts them

**What happens:**

**Query 1:**
- `messages` = [SystemMessage, HumanMessage("pwd?"), AIMessage(tool_calls), ToolMessage]
- The code broadcasts the AIMessage with the tool_call → the TUI shows `execute("pwd")` ✅

**Query 2:**
- `messages` = [SystemMessage, HumanMessage("pwd?"), AIMessage(tool_calls), ToolMessage, AIMessage("answer"), HumanMessage("2+2="), AIMessage("4")]
- The code again finds AIMessage(tool_calls) at index [2] and broadcasts it! ❌
- The TUI shows `execute("pwd")` again, even though this is an OLD tool call

### Why the bug did not show up earlier

**BEFORE commit 024dcbc:**
- Non-streaming mode did NOT broadcast tool calls at all
- Tool calls were **invisible** in the TUI
- But there was NO duplication, because nothing was being sent

**AFTER commit 024dcbc:**
- Non-streaming mode started broadcasting tool calls
- But it broadcasts **THE ENTIRE history**, not just the new messages
- Duplication appeared

**Why streaming mode does not have this bug:**

Streaming mode uses `graph.astream()` (lines 771-777):

```python
async for chunk in self.agent.astream(...):
    # chunk contains ONLY new events
    await process_message_chunk(
        message=message_obj,
        metadata=metadata,
        notify_callback=notify_callback,
        tool_call_buffer=tool_call_buffer,
    )
```

`astream()` returns **only new events/chunks**, not the entire history!

## The correct solution

### Requirements

1. ✅ Tool calls must be visible in the TUI (the goal of commit 024dcbc)
2. ✅ Broadcast only NEW messages, not the entire history
3. ✅ Work identically for ALL providers (transparency)
4. ✅ Do not violate the agent's architecture

### Solution: Tracking last_broadcasted_count

**Similar to `sent_count` in ClaudeCodeChatModel!**

**Concept:**
- Keep a per-thread counter of already sent notifications
- Broadcast only new messages (from index `last_broadcasted`)
- Update the counter after broadcasting

### Technical specification

**File:** `process/agent_process.py`

**Class:** `AgentProcess`

#### 1. Add state tracking in __init__

```python
class AgentProcess:
    def __init__(
        self,
        name: str,
        config: AgentConfig,
        session_manager: SessionManager | None = None,
        enable_task_manager: bool = False,
        skip_permissions: bool = False,
    ):
        # ... existing code ...

        # Track how many messages we've broadcasted per thread
        # This prevents re-broadcasting old messages from LangGraph checkpointer
        # Similar to sent_count pattern in ClaudeCodeChatModel
        self._last_broadcasted_count: dict[str, int] = {}
```

#### 2. Change the notification loop (lines ~933-978)

**BEFORE (BUG):**
```python
if notify_callback is not None:
    for msg in messages:  # ← All messages (the entire history)
        # ... broadcast tool_calls ...
```

**SHOULD BE (FIX):**
```python
if notify_callback is not None:
    # Extract thread_id from config
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    # Determine which messages are NEW (not yet broadcasted)
    last_broadcasted = self._last_broadcasted_count.get(thread_id, 0)
    new_messages = messages[last_broadcasted:]

    print(f"[INVOKE] Iteration {iteration}: Broadcasting {len(new_messages)} new messages (total={len(messages)}, last_broadcasted={last_broadcasted})", file=sys.stderr)

    # Broadcast ONLY new messages
    for msg in new_messages:
        # Extract message data
        if isinstance(msg, dict):
            msg_role = msg.get("role", "")
            msg_content = msg.get("content", "")
            msg_tool_calls = msg.get("tool_calls")
        elif hasattr(msg, "type"):
            msg_role = msg.type
            msg_content = getattr(msg, "content", "")
            msg_tool_calls = getattr(msg, "tool_calls", None)
        else:
            continue

        # Send tool_call notifications
        if msg_tool_calls:
            for tool_call in msg_tool_calls:
                if isinstance(tool_call, dict):
                    notify_callback({
                        "type": "tool_call",
                        "name": tool_call.get("name", ""),
                        "args": tool_call.get("args", {}),
                        "id": tool_call.get("id", ""),
                    })
                else:
                    # LangChain ToolCall object
                    notify_callback({
                        "type": "tool_call",
                        "name": getattr(tool_call, "name", ""),
                        "args": getattr(tool_call, "args", {}),
                        "id": getattr(tool_call, "id", ""),
                    })

        # Send tool_result notifications
        if msg_role == "tool":
            notify_callback({
                "type": "tool_result",
                "content": msg_content,
                "tool_call_id": (
                    msg.get("tool_call_id", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "tool_call_id", "")
                ),
            })

    # Update last_broadcasted_count AFTER broadcasting
    # This ensures we don't re-broadcast these messages in future iterations
    self._last_broadcasted_count[thread_id] = len(messages)
```

### Execution flow after the fix

#### Query 1: "What is your current working directory?"

**Iteration 1:**
- `messages` = [SystemMessage, HumanMessage] (2 messages)
- `last_broadcasted` = 0
- `new_messages` = messages[0:] = all 2 messages
- Broadcast: (system messages are usually not broadcast)
- `last_broadcasted_count["default"]` = 2

**Iteration 2 (after tool execution):**
- `messages` = [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage] (4 messages)
- `last_broadcasted` = 2
- `new_messages` = messages[2:] = [AIMessage(tool_calls), ToolMessage]
- Broadcast: tool_call("execute", "pwd"), tool_result("/tmp") ✅
- `last_broadcasted_count["default"]` = 4

#### Query 2: "2+2="

**Iteration 1:**
- `messages` = [SystemMessage, ..., AIMessage("answer"), HumanMessage("2+2=")] (6 messages)
- `last_broadcasted` = 4
- `new_messages` = messages[4:] = [AIMessage("answer"), HumanMessage("2+2=")]
- Broadcast: only the new messages, WITHOUT the old tool_calls ✅
- `last_broadcasted_count["default"]` = 6

**Iteration 2:**
- `messages` = [..., HumanMessage("2+2="), AIMessage("4")] (7 messages)
- `last_broadcasted` = 6
- `new_messages` = messages[6:] = [AIMessage("4")]
- Broadcast: only the final answer "4" ✅
- NO old tool_calls! ✅

## Important notes

### 1. Not provider-specific

✅ The fix works identically for **ALL** providers
✅ It does not violate the provider transparency principle
✅ It improves the correctness of the agent as a whole

### 2. Analogy with ClaudeCodeChatModel

ClaudeCodeChatModel already implements this pattern:

```python
# ClaudeCodeChatModel.ainvoke()
sent_count = self._client_manager.get_sent_count(thread_id)
new_messages = messages[sent_count:]  # Only the new ones!

for msg in new_messages:
    # Send to Claude...

self._client_manager.update_sent_count(thread_id, len(messages))
```

We are simply applying the same logic to the broadcast notifications!

### 3. Streaming mode already works correctly

Streaming mode does NOT have this bug, because `astream()` returns only new events.

Non-streaming mode must imitate this behavior.

### 4. Cleanup of old sessions (optional)

Cleanup for `_last_broadcasted_count` can be added:

```python
# In AgentProcess
async def cleanup_session(self, thread_id: str):
    """Clean up state for a finished session."""
    self._last_broadcasted_count.pop(thread_id, None)
```

But this is not critical - the dictionary will grow slowly (one key per session).

## Conclusions

### ✅ What was confirmed:

1. **The bug is common to ALL providers** - deepseek and claude-code both reproduce the bug
2. **The bug appeared yesterday** - commit 024dcbc (Wed Mar 11 00:19:48 2026)
3. **Cause: an incorrect implementation** - broadcasting the entire history instead of only the new messages
4. **ClaudeCodeChatModel works correctly** - it is NOT the source of the bug

### ❌ What was wrong in my first analysis:

1. ❌ Claimed that ClaudeCodeChatModel was correct (this is true)
2. ❌ But did NOT check whether the bug reproduces for other providers
3. ❌ Assumed it was a common bug, but did NOT test it

### ✅ The correct solution:

**File:** `process/agent_process.py`

1. Add `self._last_broadcasted_count: dict[str, int] = {}` to `__init__`
2. Change the notification loop (lines ~933-978):
   - Compute `new_messages = messages[last_broadcasted:]`
   - Broadcast only `new_messages`
   - Update `_last_broadcasted_count[thread_id] = len(messages)`

### 🎯 Expected result:

After the fix both tests should pass:
- ✅ `test_agent_tool_call_duplication_deepseek.py` - PASS
- ✅ `test_agent_tool_call_duplication_claude_code.py` - PASS

Tool calls will be visible (the goal of commit 024dcbc), but WITHOUT duplication!
