# Bug Analysis: Duplicated Tool Calls in the TUI

## Problem description

When using the claude-code provider with the agent in non-streaming mode:

1. **Query 1**: "What is your current working directory?" → the TUI correctly shows `execute("pwd")` and the result `/tmp`
2. **Query 2**: "2+2=" → the TUI **INCORRECTLY** shows `execute("pwd")` from Query 1 again, even though NO tool should be called

## Reproduction

Test: `tests/integration_ai/test_agent_tool_call_duplication_claude_code.py`

**Test result**: FAILED

```
> What is your current working directory?
⏺ execute("pwd")    ← Correct
⎿ /tmp

> 2+2=
⏺ execute("pwd")    ← BUG! The tool from the first request!
⎿ /tmp
2 + 2 = 4
```

## Architecture analysis (per design_communication.txt)

### Key principle: Message Mirroring

From `docs/design_communication.txt` (lines 329-336):

> ClaudeSDKClient (when used as an AI provider) **mirrors** the messages that exist in LangChain (in the thread). When the Claude model calls a tool, it must do so through an in-process MCP server... This tool call must be passed to the agent and handled there EXACTLY THE SAME WAY as a tool call when working with any other provider.

### Transparency requirement (line 332):

> For the agent, working with the AI provider being designed on top of the Claude SDK must be absolutely transparent and look EXACTLY THE SAME as working with any other AI provider.

### Message-by-message processing (line 336):

> In the agent (LangGraph) messages are processed **one at a time**. Accordingly, this must be ensured when working with ClaudeSDKClient.

## Execution logs

### Query 1: "What is your current working directory?"

```
[2026-03-12T02:49:18.527574] ========== START AINVOKE ==========
thread_id=default, sent_count=0, len(messages)=2

messages[0]: SystemMessage - You are an AI assistant...
messages[1]: HumanMessage - What is your current working directory?

new_messages[0]: SystemMessage - You are an AI assistant...
new_messages[1]: HumanMessage - What is your current working directory?

Sending HumanMessage to Claude:
  content='What is your current working directory?'
  session_id='default'

Received TextBlock: 'I'll check the current working directory for you.'
Tool 'execute' called with args={'command': 'pwd'}, call_id=execute_1_124269584162880
Detected pending tool call!
Returning AIMessage with 1 tool_calls: ['execute']
```

**LangGraph then executes the tool and calls ainvoke() again:**

```
[2026-03-12T02:49:21.097630] ========== START AINVOKE ==========
thread_id=default, sent_count=2, len(messages)=4

messages[0]: SystemMessage - You are an AI assistant...
messages[1]: HumanMessage - What is your current working directory?
messages[2]: AIMessage - (empty, with tool_calls)
messages[3]: ToolMessage - /tmp [Command succeeded with exit code 0]

new_messages[0]: AIMessage - (empty, with tool_calls)
new_messages[1]: ToolMessage - /tmp [Command succeeded with exit code 0]

tool_resolved=True
Calling client.receive_response()...
Tool call_id=execute_1_124269584162880 resolved, returning result
Received TextBlock: 'The current working directory is `/tmp`.'
Received ResultMessage
Returning AIMessage with content: 'The current working directory is `/tmp`.'
```

**Result of Query 1: 4 messages in the thread:**
1. SystemMessage
2. HumanMessage - "What is your current working directory?"
3. AIMessage - (with tool_calls)
4. ToolMessage - "/tmp"

### Query 2: "2+2="

```
[2026-03-12T02:49:26.094022] ========== START AINVOKE ==========
thread_id=default, sent_count=4, len(messages)=6

messages[0]: SystemMessage - You are an AI assistant...
messages[1]: HumanMessage - What is your current working directory?
messages[2]: AIMessage - (with tool_calls)
messages[3]: ToolMessage - /tmp [Command succeeded with exit code 0]
messages[4]: AIMessage - The current working directory is `/tmp`.
messages[5]: HumanMessage - 2+2=

new_messages[0]: AIMessage - The current working directory is `/tmp`.
new_messages[1]: HumanMessage - 2+2=

Sending HumanMessage to Claude:
  content='2+2='
  session_id='default'

Received TextBlock: '2 + 2 = 4'
Received ResultMessage
Returning AIMessage with content: '2 + 2 = 4'
```

**CRITICAL OBSERVATION:**

ClaudeCodeChatModel correctly sends **ONLY the new messages** (new_messages) to Claude:
- AIMessage - "The current working directory is `/tmp`." (the final response of Query 1)
- HumanMessage - "2+2="

Claude does NOT call any tools for Query 2 (correct!).

## Root cause of the bug

**The problem is NOT in ClaudeCodeChatModel!**

ClaudeCodeChatModel works correctly:
- ✅ Mirrors the messages from the LangChain thread
- ✅ Sends only new messages to Claude (sent_count tracking)
- ✅ Claude does NOT call tools for Query 2

**The problem is in the agent's agent_process.py (non-streaming mode)!**

### File: `process/agent_process.py`

**Lines 933-978** (non-streaming mode with notify_callback):

```python
if notify_callback is not None:
    for msg in messages:  # ← BUG! Iterates over ALL messages from result
        # Extract message data
        if isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("role", "assistant")
            tool_calls = msg.get("tool_calls", [])
        else:
            # LangChain BaseMessage
            content = getattr(msg, "content", "")
            role = getattr(msg, "role", "assistant")
            tool_calls = getattr(msg, "tool_calls", [])

        # Broadcast tool calls
        if tool_calls:
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    notify_callback({
                        "type": "tool_call",
                        "name": tool_call.get("name", ""),
                        "args": tool_call.get("args", {}),
                    })
                else:
                    # ToolCall object
                    notify_callback({
                        "type": "tool_call",
                        "name": getattr(tool_call, "name", ""),
                        "args": getattr(tool_call, "args", {}),
                    })
```

**What happens:**

1. **Query 1**: `result['messages']` contains 4 messages. The code iterates over all 4, including `messages[2]` (the AIMessage with tool_calls). Tool call broadcast → the TUI shows `execute("pwd")` ✅ CORRECT

2. **Query 2**: `result['messages']` contains **6 messages** (the entire history!):
   ```
   messages[0]: SystemMessage
   messages[1]: HumanMessage - "What is your current working directory?"
   messages[2]: AIMessage - (with tool_calls execute("pwd"))  ← OLD!
   messages[3]: ToolMessage - "/tmp"
   messages[4]: AIMessage - "The current working directory is `/tmp`."
   messages[5]: HumanMessage - "2+2="
   messages[6]: AIMessage - "2 + 2 = 4"
   ```

   The code iterates over ALL 7 messages, including `messages[2]` with the **OLD** tool_call!

   Tool call broadcast → the TUI shows `execute("pwd")` again ❌ BUG!

## Why this happens

**The LangGraph Checkpointer** stores the full message history in the state.

When `agent.ainvoke()` returns a result, it includes:
- `result['messages']` = **THE ENTIRE message history** (from the start of the session up to the current moment)

In **streaming mode** (lines 771-777) this is NOT a problem, because `graph.astream()` is used, which returns **ONLY new events/chunks**.

In **non-streaming mode** (lines 933-978) the code receives the full `result` with the entire history and broadcasts ALL messages, including the old ones.

## Solution

### FORBIDDEN (violates the provider transparency principle):

❌ Changing anything in the agent specifically for the claude-code provider
❌ Adding hacks for an individual provider into the agent's code

### ALLOWED:

✅ Fix the logic in `agent_process.py` (non-streaming mode) so that it broadcasts **ONLY NEW** messages, the way streaming mode does

### Proposed fix

**File:** `process/agent_process.py`

**Lines 927-978** (the non-streaming block with notify_callback):

```python
# BEFORE (BUG):
if notify_callback is not None:
    for msg in messages:  # ← Iterates over ALL messages (the entire history)
        # ... broadcast tool calls ...

# SHOULD BE (FIX):
if notify_callback is not None:
    # Determine the index of the first NEW message
    # (the messages we have NOT broadcast yet)

    # Option 1: Track last_notified_count (similar to sent_count in ClaudeCodeChatModel)
    # Option 2: Broadcast only the last AIMessage (the final response)
    # Option 3: Broadcast only the messages after the last HumanMessage in this iteration

    # RECOMMENDATION: Option 1 - the most reliable
    # Keep a per-thread counter of already sent notifications
    # and broadcast only the new messages

    # Example (pseudocode):
    last_notified = self._last_notified_count.get(thread_id, 0)
    new_messages_to_notify = messages[last_notified:]

    for msg in new_messages_to_notify:
        # ... broadcast tool calls ...

    self._last_notified_count[thread_id] = len(messages)
```

### Additional considerations

1. **Streaming mode works correctly**, because `graph.astream()` returns **only new events**. This confirms that the broadcast logic in streaming mode is correct.

2. **Non-streaming mode must imitate the behavior of streaming mode** - broadcast only new messages.

3. **ClaudeCodeChatModel already implements this pattern** (sent_count tracking), which proves its correctness.

## Conclusions

### ✅ What works correctly:

1. **ClaudeCodeChatModel** - correctly mirrors messages, sends only new ones
2. **The message mirroring architecture** - works as intended
3. **Streaming mode in agent_process.py** - broadcasts only new events
4. **The Claude Code SDK** - does NOT call tools for Query 2 (correct)

### ❌ What does NOT work:

1. **Non-streaming mode in agent_process.py** - broadcasts ALL messages from `result['messages']`, including the old ones

### 🔧 What needs to be fixed:

**ONLY** `process/agent_process.py`

Lines 933-978: Add tracking of already sent notifications (similar to `sent_count` in ClaudeCodeChatModel).

### ⚠️ Important:

- ❌ Do NOT change the code specifically for the claude-code provider
- ✅ The fix must work identically for **ALL** providers
- ✅ The fix improves the correctness of the agent as a whole
- ✅ It follows the "agent as single source of truth" principle - broadcast only new data

## Technical specification of the fix

### Location:

`process/agent_process.py`

### Class:

`AgentProcess`

### Add:

```python
class AgentProcess:
    def __init__(...):
        # ... existing code ...
        self._last_notified_count: dict[str, int] = {}  # thread_id -> count
```

### Change (lines 933-978):

```python
# Inside the non-streaming mode block with notify_callback
if notify_callback is not None:
    # Get thread_id from config
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    # Determine which messages are new (not yet notified)
    last_notified = self._last_notified_count.get(thread_id, 0)
    new_messages = messages[last_notified:]

    # Broadcast only NEW messages
    for msg in new_messages:
        # ... existing broadcast logic for tool_calls, tool_results, content ...

    # Update last notified count
    self._last_notified_count[thread_id] = len(messages)
```

### Alternative approach (simpler):

If we do not want to add state to AgentProcess, we can broadcast only the last AIMessage:

```python
if notify_callback is not None:
    # In non-streaming mode, only broadcast the FINAL response
    # (last AIMessage in result)
    final_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or (isinstance(msg, dict) and msg.get("role") == "assistant"):
            final_ai_message = msg
            break

    if final_ai_message:
        # Broadcast only the final message
        content = getattr(final_ai_message, "content", "") or final_ai_message.get("content", "")
        notify_callback({
            "type": "assistant",
            "role": "assistant",
            "content": content,
        })
```

But this approach **will NOT broadcast tool calls**, which is also incorrect.

**RECOMMENDATION:** Use the first approach with `_last_notified_count` for full correctness.
