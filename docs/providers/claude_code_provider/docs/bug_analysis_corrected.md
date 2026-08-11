# Bug Analysis: The Agent "Gets Stuck" on the First Request (Corrected)

## Bug description

The `test_agent_multiple_queries` test fails:
- Request 1: "2+2" → Answer: "4" ✅
- Request 2: "3+3" → Answer: "4" ❌ (repeats the first one!)
- Request 3: "5+5" → Answer: "4" or "6" ❌ (repeats the previous ones!)

---

## Root Cause Analysis (Corrected)

### Key insight: what is passed into ainvoke()?

**Analysis of the agent code (agent_process.py:836-839):**
```python
result = await self.agent.ainvoke(
    {"messages": [{"role": "user", "content": current_message}]},
    config
)
```

**Critical:** Each request passes **ONLY the new message**, NOT the entire history!

**How LangGraph works with the checkpointer:**

1. **Request 1:** `agent.ainvoke({"messages": [HumanMessage("2+2")]}, config={"thread_id": "test"})`
   - The checkpointer loads the state for thread_id="test" (empty)
   - LangGraph appends HumanMessage("2+2") to the empty state
   - **Model.ainvoke() receives:** `[HumanMessage("2+2")]`
   - The model answers: AIMessage("4")
   - The checkpointer saves: `[HumanMessage("2+2"), AIMessage("4")]`

2. **Request 2:** `agent.ainvoke({"messages": [HumanMessage("3+3")]}, config={"thread_id": "test"})`
   - The checkpointer loads the state: `[HumanMessage("2+2"), AIMessage("4")]`
   - LangGraph appends HumanMessage("3+3")
   - **Model.ainvoke() SHOULD receive:** `[HumanMessage("2+2"), AIMessage("4"), HumanMessage("3+3")]`
   - The model should answer: AIMessage("6") taking the context into account
   - The checkpointer saves: `[..., HumanMessage("3+3"), AIMessage("6")]`

**Conclusion:** LangGraph **ALWAYS** passes **THE ENTIRE HISTORY** of messages from the state into model.ainvoke()!

---

### Problem 1: Missing session_id

**File:** `chat_model.py:386`

```python
if user_content:
    await client.query(user_content)  # ❌ session_id="default"
```

**Consequences:**

1. All requests go into a single Claude session ("default")
2. ClaudeSDKClient **does NOT send** the previous messages (it depends on the session in the CLI)
3. The Claude Code CLI stores its own history for session_id="default"

**BUT:** This still does not fully explain the bug, because:
- The Claude CLI session already contains the context
- Why does Claude repeat "4"?

---

### Problem 2: ClaudeCodeChatModel does NOT send the entire history!

**CRITICAL ERROR in chat_model.py:355-367:**

```python
# Convert messages to Claude format (skip AIMessage with tool_calls)
claude_messages: list[...] = []
for msg in messages:
    # Skip AIMessage with tool_calls (these were already sent to Claude)
    if isinstance(msg, AIMessage) and msg.tool_calls:
        continue  # ❌ PROBLEM!
    # Skip ToolMessage (already processed above)
    if isinstance(msg, ToolMessage):
        continue  # ❌ PROBLEM!

    claude_msg = self._convert_to_claude(msg)
    if claude_msg is not None:
        claude_messages.append(claude_msg)
```

**What happens:**

### Request 1: "2+2"

**Input messages into ainvoke():**
```python
[HumanMessage("2+2")]
```

**After filtering (lines 355-367):**
```python
claude_messages = [ClaudeUserMessage("2+2")]
```

**Extraction of user_content (lines 373-383):**
```python
user_content = "2+2"
```

**Sending to Claude (lines 385-386):**
```python
await client.query("2+2")  # session_id="default"
```

**Claude answers:** "2 + 2 = 4"

**Result:**
```python
return AIMessage(content="2 + 2 = 4")
```

**Claude session state after request 1:**
```
Session "default":
- User: "2+2"
- Assistant: "2 + 2 = 4"
```

---

### Request 2: "3+3"

**Input messages into ainvoke() (from LangGraph!):**
```python
[
    HumanMessage("2+2"),           # ← From the checkpointer
    AIMessage("2 + 2 = 4"),        # ← From the checkpointer
    HumanMessage("3+3")            # ← The new message
]
```

**❗ LangGraph PASSES THE ENTIRE HISTORY!**

**After filtering (lines 355-367):**
```python
# AIMessage("2 + 2 = 4") does NOT have tool_calls, so it SHOULD be converted
claude_messages = [
    ClaudeUserMessage("2+2"),        # ✅ Converted
    ClaudeAssistantMessage("4"),     # ✅ Converted (if NOT tool_calls)
    ClaudeUserMessage("3+3")         # ✅ Converted
]
```

**But look at lines 373-383:**

```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        user_content = msg.content  # ❌ OVERWRITES!
        # ...
        break  # ❌ STOPS at the first UserMessage!
```

**❌ BUG:** `break` stops at the FIRST UserMessage!

**Extraction of user_content:**
```python
user_content = "2+2"  # ❌ The FIRST message, NOT the last!
```

**Sending to Claude (lines 385-386):**
```python
await client.query("2+2")  # ❌ Sends the FIRST request again!
```

**Claude session state BEFORE this request:**
```
Session "default":
- User: "2+2"
- Assistant: "2 + 2 = 4"
```

**Claude receives AGAIN:** "2+2"

**Claude may decide:**
- "I have already answered this"
- Returns the cached answer: "2 + 2 = 4"

**Result:**
```python
return AIMessage(content="2 + 2 = 4")  # ❌ Repeats the first answer!
```

---

## Two critical problems

### Problem 1: `break` at the first UserMessage

**File:** `chat_model.py:373-383`

```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        user_content = msg.content
        # ...
        break  # ❌ STOPS AT THE FIRST ONE!
```

**Consequences:**
- The FIRST UserMessage is extracted rather than the LAST one
- A repeated request "2+2" is sent instead of the new "3+3"

### Problem 2: Missing session_id

**File:** `chat_model.py:386`

```python
await client.query(user_content)  # session_id="default"
```

**Consequences:**
- All threads use a single Claude session
- There is no isolation between thread_ids

---

## Proposed fix

### Change 1: Extract the LAST UserMessage

**File:** `chat_model.py:373-383`

**Before:**
```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        user_content = msg.content
        # ...
        break  # ❌ The FIRST message
```

**After:**
```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        # Continue iterating in order to find the LAST UserMessage
        if isinstance(msg.content, str):
            user_content = msg.content
        else:
            user_content = str(msg.content)
# We do not use break - we continue to the end
```

### Change 2: Pass session_id

**File:** `chat_model.py:386`

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

## Additional question: is it necessary to send the ENTIRE history to Claude?

### Current behavior

ClaudeCodeChatModel converts ALL messages, but sends only the last user_content:

```python
# Converts everything:
claude_messages = [
    ClaudeUserMessage("2+2"),
    ClaudeAssistantMessage("4"),
    ClaudeUserMessage("3+3")
]

# But sends only:
await client.query("3+3")  # ❌ Without context!
```

### Question: does the Claude CLI store the history?

**Yes!** The Claude CLI stores the history for each session_id:

```python
# client.query("message", session_id="my-session")
#
# The Claude CLI stores:
# Session "my-session": [all previous messages]
# The new request is appended to the existing history
```

**BUT:** This only works if:
1. The **SAME session_id** is used
2. The Claude CLI is **NOT restarted** between requests
3. ClaudeSDKClient stays **connected**

### The problem with the current approach

**ClaudeCodeChatModel relies on the Claude CLI to store the context:**

1. Request 1: `client.query("2+2")` → the Claude CLI saves it into the session
2. Request 2: `client.query("3+3")` → the Claude CLI appends to the same session
3. **It is assumed** that the Claude CLI remembers the context

**Risks:**
- What if the Claude CLI loses the session?
- What if ClaudeSDKClient is recreated?
- What if fork_session is used?

### A more reliable approach (optional)

**Send the entire history explicitly:**

Instead of:
```python
await client.query("3+3", session_id=thread_id)
```

Send:
```python
# Build a full representation of the history
messages_for_claude = [
    {"role": "user", "content": "2+2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "3+3"}
]
# Send as an AsyncIterable or construct the request
```

**BUT:** The Claude SDK `client.query()` accepts only:
- `str` - a single message
- `AsyncIterable[dict]` - a stream of messages

**Problem:** Sending the history requires using streaming mode, which complicates the code.

---

## Recommended solution (minimal)

### Fix 1: Remove the `break` (extract the last UserMessage)

**File:** `chat_model.py:373-383`

```python
user_content: str = ""
for msg in claude_messages:
    if isinstance(msg, ClaudeUserMessage):
        # Continue in order to find the LAST UserMessage
        if isinstance(msg.content, str):
            user_content = msg.content
        else:
            user_content = str(msg.content)
# Now user_content contains the LAST message
```

### Fix 2: Add session_id

**File:** `chat_model.py:386`

```python
if user_content:
    await client.query(user_content, session_id=thread_id)
```

---

## Why is this the right solution?

### ✅ Fix 1: Remove the break

**Problem:**
- `break` stops at the FIRST UserMessage
- When several UserMessages are present in the history, the wrong one is extracted

**Solution:**
- Continue iterating to the end
- The last assignment will be the LAST UserMessage

**Expected behavior after the fix:**

Request 2:
```python
messages = [
    HumanMessage("2+2"),      # ← user_content = "2+2"
    AIMessage("4"),
    HumanMessage("3+3")       # ← user_content = "3+3" (the last one!)
]

# After the loop:
user_content = "3+3"  # ✅ Correct!
await client.query("3+3", session_id=thread_id)
```

### ✅ Fix 2: Add session_id

**Problem:**
- All threads use session_id="default"
- There is no isolation

**Solution:**
- thread_id → session_id
- Each thread = a separate Claude session

---

## Limitations and risks

### Risk 1: Dependence on the Claude CLI for storing the history

**The current approach:**
- ClaudeCodeChatModel sends only the last message
- It relies on the Claude CLI to store the entire history

**What can go wrong:**
- The Claude CLI loses the session (crash, restart)
- ClaudeSDKClient reconnects (a new session)
- ClientManager recreates the client

**Mitigation:**
- ClientManager caches the client by thread_id (minimizes recreation)
- session_id is bound to thread_id (a stable identifier)
- The checkpointer in LangGraph stores the full history (a backup copy)

### Risk 2: The Claude CLI may not support a long context

**If the history is very long:**
- The Claude CLI may truncate the history
- It may apply summarization

**Solution:**
- That is fine! The Claude CLI manages the context window
- The LangGraph checkpointer stores the full history

---

## Alternative solutions (not recommended for the minimal fix)

### Alternative 1: Send the entire history explicitly

**Idea:**
```python
# Send all messages via an AsyncIterable
async def history_stream():
    for msg in claude_messages:
        yield convert_to_claude_format(msg)

await client.query(history_stream(), session_id=thread_id)
```

**Why not now:**
- It complicates the code
- The Claude SDK requires streaming mode
- ClientManager would need more changes
- **The current approach with session_id works!**

### Alternative 2: Create a new client for each request

**Why this is bad:**
- It violates the caching principle in ClientManager
- Overhead on connect/disconnect
- The session is lost between requests

---

## Testing the fix

### Test 1: Sequential requests (the bug)

```python
config = {"configurable": {"thread_id": "test_multi_query"}}

# Request 1
result1 = await model.ainvoke([HumanMessage("2+2")], config)
assert "4" in result1.content  # ✅

# Request 2 (with the history from LangGraph)
result2 = await model.ainvoke([
    HumanMessage("2+2"),
    AIMessage("4"),
    HumanMessage("3+3")  # ← The new request
], config)
assert "6" in result2.content  # ✅ Should be "6", NOT "4"!
```

### Test 2: Session isolation

```python
config1 = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}

await model.ainvoke([HumanMessage("2+2")], config1)  # → "4"
await model.ainvoke([HumanMessage("3+3")], config2)  # → "6"

# Check that thread-2 does not see "4" from thread-1
result = await model.ainvoke([HumanMessage("what's the answer?")], config2)
assert "6" in result.content or "3+3" in result.content
assert "4" not in result.content or "2+2" in result.content  # May be mentioned, but not as the answer
```

---

## Conclusion

### The problem

1. **`break` in the extraction of user_content** → the FIRST message is extracted rather than the LAST
2. **Missing session_id** → all threads share a single Claude session

### The minimal fix

1. **Remove the `break`** (line 383)
2. **Add `session_id=thread_id`** (line 386)

### Why is this sufficient?

✅ It fixes the repeated-answer bug
✅ It provides isolation between threads
✅ It relies on the Claude CLI for context management (which is fine!)
✅ Minimal code changes
✅ It conforms to the basic principles of the provider

### Next steps

After applying the fix:
1. Run `test_agent_multiple_queries` → it should pass
2. Add a unit test for extracting the last UserMessage
3. Add an integration test for session isolation
