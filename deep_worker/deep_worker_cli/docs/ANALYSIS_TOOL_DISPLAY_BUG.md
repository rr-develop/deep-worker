# Analysis: Why tool calls are not displayed in the dwcli TUI

**Date**: 2026-03-14
**Status**: ROOT CAUSE FOUND
**Priority**: CRITICAL

## Executive Summary

Tool calls are **NOT** displayed in the dwcli TUI, even though:
- ✅ tool_call_chunk items arrive from LangGraph
- ✅ Streaming works
- ✅ notify_callback is invoked
- ❌ BUT the **wrong data** is passed: `name=None, id=None`

**Root cause**: After the first `tool_call_chunk`, the subsequent chunks lose `id` and `name`, which leads to incorrect buffering and to sending an incomplete tool call to the TUI.

---

## Detailed Analysis

### 1. Testing

**Tests:**
- `test_slow_tool_display.py` (dwcli) - ❌ FAILS
- `test_deepagents_slow_tool_display.py` (the reference) - ✅ Shows tool call (fails on approval popup)

**The dwcli result:**
- The TUI shows an empty screen for 10+ seconds
- The tool call is NOT displayed even after completion
- Only the text response appears after the tool has run

**The deepagents result:**
- After 1 second it shows "⠋ Thinking... (1s)"
- The tool call appears: `▎ ⏺ execute("sleep 10")`
- **The tool call is VISIBLE in the TUI!**

---

### 2. Investigation of the dwcli Logs

#### A. Streaming works (`/tmp/dwcli_astream_debug.log`)

```
[1773437362.893] CHUNK #20 received:
  Type: <class 'tuple'>
  Stream mode: messages
  MESSAGE chunk:
    Message type: <class 'langchain_core.messages.ai.AIMessageChunk'>
    Calling process_message_chunk...
    ✓ process_message_chunk completed
```

**Conclusion**: The chunks DO ARRIVE and are processed.

#### B. Tool Call Chunks Do Arrive (`/tmp/dwcli_stream_debug.log`)

**The first chunk:**
```
[1773432493.679] [Thread: MainThread] TOOL_CALL_CHUNK:
  id=call_00_32ci7xxOkmldxh4PQq8rsnQX  ← THE ID IS PRESENT
  name=execute                         ← THE NAME IS PRESENT
  args=''
```

**The subsequent chunks:**
```
[1773432493.681] [Thread: MainThread] TOOL_CALL_CHUNK:
  id=None      ← THE ID IS GONE!
  name=None    ← THE NAME IS GONE!
  args='{'

[1773432493.684] [Thread: MainThread] TOOL_CALL_CHUNK:
  id=None
  name=None
  args='"'

[1773432493.686] [Thread: MainThread] TOOL_CALL_CHUNK:
  id=None
  name=None
  args='command'
```

**The final notify_callback invocation:**
```
[1773432493.705] [Thread: MainThread] TOOL_CALL COMPLETE - calling notify_callback:
  name=None      ← WRONG!
  id=None        ← WRONG!
  args={'command': 'sleep 10'}  ← Only the args are correct
[1773432494.272] notify_callback returned
```

**Conclusion**:
- The first chunk carries the `id` and the `name`
- The subsequent chunks carry only the `args`
- The final tool_call is sent with `name=None, id=None`

---

### 3. The Root Cause in the Code

**File**: `process/stream_processor.py:86-151`

**The problematic code:**
```python
# Line 87
elif block_type == "tool_call_chunk":
    call_id = block.get("id")       # ← None after the first chunk
    index = block.get("index", 0)

    # Line 106
    buffer_key = f"{call_id}:{index}"  # ← It becomes "None:0"!

    if buffer_key not in tool_call_buffer:
        tool_call_buffer[buffer_key] = {
            "id": call_id,           # ← None
            "name": block.get("name"), # ← None
            "args": "",
        }
```

**What happens:**

1. **The first chunk**:
   - `id=call_00_xxx, name=execute, args=''`
   - `buffer_key = "call_00_xxx:0"`
   - An entry is created in the buffer

2. **The second chunk**:
   - `id=None, name=None, args='{'`
   - `buffer_key = "None:0"`  ← **A NEW KEY!**
   - A **SECOND** entry is created with `id=None, name=None`

3. **The remaining chunks**:
   - All the args accumulate in the buffer under the key `"None:0"`
   - The first entry with the correct ID stays untouched

4. **When the args are fully assembled**:
   - The JSON parses successfully
   - The tool_call from the `"None:0"` buffer is sent
   - **It is sent with `name=None, id=None`!**

---

### 4. Why does it work in deepagents?

**Hypotheses:**

1. **Deepagents uses a different buffering method**
   - It may store the `id` and the `name` from the first chunk
   - And update only the `args` in the subsequent chunks

2. **Deepagents receives different chunks from LangGraph**
   - It may be that all the chunks contain the `id` and the `name`
   - Or that only the first chunk matters

3. **Deepagents processes the tool_call synchronously in the main thread**
   - Textual requires updates from the main event loop
   - dwcli triggers updates from the RPyC thread

**What is needed:**
- Study `deepagents_cli/textual_adapter.py`
- Compare the tool_call_chunk handling method
- Check the threading model

---

### 5. Additional Problems

#### A. Character-by-character Streaming

Every character of the args arrives as a separate chunk:
```
args='{'
args='"'
args='command'
args='"'
args=': '
args='"'
args='sleep'
args=' '
args='10'
```

**Problem**: It is inefficient and creates many tiny updates.

**Deepagents**: It may use batching or collect several characters before sending.

#### B. Threading vs Async

**dwcli**: `_notify_subscribers` calls `subscriber.on_message()` **synchronously** from the RPyC thread.

```python
# agent_service.py:373
for subscriber in subscribers:
    try:
        subscriber.on_message(message)  # ← A synchronous call from the RPyC thread!
    except Exception as e:
        logger.exception(f"Error notifying subscriber {subscriber}: {e}")
```

**Problem**: Textual (the TUI framework) may require updates only from the main event loop.

**Deepagents**: It probably uses:
- `asyncio.run_coroutine_threadsafe()` to hand things over safely to the main loop
- Or `app.call_from_thread()` (a Textual API)
- Or an async notify mechanism

#### C. The Absence of Async in Critical Places

**The dwcli stream_processor**:
```python
async def process_message_chunk(...):
    # ... async function
    notify_callback({  # ← But the callback is SYNCHRONOUS!
        "type": "tool_call",
        "name": ...,
        "args": ...,
    })
```

**It should be**:
```python
await notify_callback({  # ← Async!
    "type": "tool_call",
    ...
})
```

Or use `asyncio.create_task()` for a non-blocking dispatch.

---

### 6. What Needs To Be Fixed

#### CRITICAL (blocks the functionality):

1. **Fix the tool_call_chunk buffering**
   - Store the `id` and the `name` from the first chunk
   - Use the first `id` as the key for all the subsequent chunks
   - Do not overwrite the `id` and the `name` when they are `None`

#### HIGH PRIORITY (architectural problems):

2. **Fix the threading model for TUI updates**
   - Use `app.call_from_thread()` or an equivalent
   - Or `asyncio.run_coroutine_threadsafe()` for an async dispatch
   - Make sure all the TUI updates come from the main event loop

3. **Make notify_callback async**
   - Change the signature to `async def notify_callback(...)`
   - Use `await` when calling it
   - Allow a non-blocking message dispatch

#### MEDIUM PRIORITY (optimization):

4. **Optimize the character-by-character streaming**
   - Batch several characters before sending
   - Or use debouncing
   - Reduce the number of TUI updates

---

### 7. Investigation Plan (Next Steps)

**Do not change the code! Analysis only:**

1. ✅ Study the dwcli logs
2. ✅ Run the deepagents test
3. ⏳ Compare `deepagents_cli/textual_adapter.py` with `deep_worker_cli/app.py`
4. ⏳ Find out how deepagents handles tool_call chunks
5. ⏳ Check the threading model in deepagents
6. ⏳ Find where deepagents uses async/await for TUI updates
7. ⏳ Document the architectural differences

---

### 8. Conclusions

**The problem is NOT that:**
- ❌ The chunks do not arrive (they do arrive)
- ❌ Streaming does not work (it works)
- ❌ notify_callback is not invoked (it is invoked)

**The problem IS that:**
- ✅ The tool call chunks lose the `id` and the `name` after the first chunk
- ✅ The buffering uses the wrong key (`"None:0"`)
- ✅ The TUI receives an incomplete tool_call (`name=None, id=None`)
- ✅ The updates may be coming from the wrong thread

**This explains:**
- Why tool calls are not displayed AT ALL (the data is wrong)
- Why it works in deepagents (correct buffering and threading)
- Why the streaming is slow (character by character, no batching)

---

## Appendices

### A. Key Files

**dwcli:**
- `process/stream_processor.py:86-151` - tool_call_chunk buffering (THE BUG)
- `rpyc_service/agent_service.py:346-378` - _notify_subscribers (threading issue)
- `process/agent_process.py:797-803` - the agent.astream() call

**deepagents (for comparison):**
- the upstream reference project's `textual_adapter.py` - TUI adapter (reference)

### B. Log Files

**dwcli:**
- `/tmp/dwcli_stream_debug.log` - tool_call_chunk details
- `/tmp/dwcli_astream_debug.log` - Streaming chunks
- `/tmp/dwcli_process_message_chunk_CALLED.log` - Message processing

**deepagents:**
- `/tmp/deepagents_test_output.log` - Test output (shows the tool call in the TUI)

### C. Tests

- `tests/integration_ai/test_slow_tool_display.py` - dwcli test (FAILS)
- `tests/integration_ai/test_deepagents_slow_tool_display.py` - Reference test (shows tool call)

---

**Next step**: Study the deepagents textual_adapter.py and document HOW IT SHOULD WORK.
