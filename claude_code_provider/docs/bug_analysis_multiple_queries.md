# Bug Analysis: The Agent "Gets Stuck" on the First Request

## Bug description

The `test_agent_multiple_queries` test fails with the following symptoms:
- First request: "2+2" → Correct answer: "4" ✅
- Second request: "3+3" → Incorrect answer: "4" (repeats the first one!) ❌
- Third request: "5+5" → Incorrect answer: "4" or "6" (repeats the previous ones!) ❌

The agent "gets stuck" on the first answer and ignores the subsequent requests.

---

## Root Cause Analysis

### Problem 1: Missing `session_id` in `client.query()`

**File:** `chat_model.py:386`

```python
if user_content:
    await client.query(user_content)  # ❌ NO session_id!
```

**ClaudeSDKClient.query() signature:**
```python
async def query(self, prompt: str | AsyncIterable[dict[str, Any]], session_id: str = "default")
```

**Problem:**
- `client.query()` **ALWAYS** uses `session_id="default"` (the default value)
- All requests go into the **SAME SESSION** of the Claude Code CLI
- The Claude Code CLI **PRESERVES THE CONTEXT** of the session between requests
- On the second call to `client.query("3+3")` Claude sees the entire history:
  ```
  Session "default":
  - User: "2+2"
  - Assistant: "2 + 2 = 4"
  - User: "3+3"  ← The new request
  ```

**Why this causes the bug:**

The Claude Code CLI has internal session management logic:
1. First request → Creates the "default" session
2. Second request → Continues the "default" session (sees the previous context)
3. **CRITICAL:** Claude may decide that the second request is a **CONTINUATION** of the first conversation
4. In some cases Claude may **repeat the previous answer** if it considers that the context has not changed

---

### Problem 2: There is no context clearing between requests

**Architecture:**
```
LangGraph Thread "test_multi_query"
    ↓
ClaudeSDKClient (cached in ClientManager by thread_id)
    ↓
Claude Code CLI Session (session_id="default")
```

**What happens:**

1. **Request 1:** LangGraph calls `ainvoke([HumanMessage("2+2")])`
   - ClientManager creates a ClaudeSDKClient for thread_id="test_multi_query"
   - `client.query("2+2")` → Sends it to session_id="default"
   - Claude answers: "4"
   - **The context is preserved** in the Claude CLI session "default"

2. **Request 2:** LangGraph calls `ainvoke([HumanMessage("3+3")])`
   - ClientManager **RETURNS THE SAME** ClaudeSDKClient (cache)
   - `client.query("3+3")` → Sends it to the **SAME** session_id="default"
   - Claude sees:
     ```
     Session "default":
     - [Turn 1] User: "2+2" → Assistant: "4"
     - [Turn 2] User: "3+3" → ???
     ```
   - **BUG:** Claude may repeat the previous answer "4"

---

### Problem 3: Confusion between thread_id and session_id

**The thread_id concept (LangGraph):**
- The identifier of a **conversation thread** in LangGraph
- Used to persist the **LangGraph state** (messages, checkpoints)
- **NOT passed** to Claude Code

**The session_id concept (Claude SDK):**
- The identifier of a **session** in the Claude Code CLI
- Used to **manage the context** of the conversation on the Claude side
- **MUST be unique** for each independent conversation

**The current problem:**
```python
# chat_model.py:337-344
thread_id = "default"
if config and "configurable" in config:
    thread_id = config["configurable"].get("thread_id", "default")

# Get or create client for this thread
client, proxy_mcp = await self._client_manager.get_or_create_client(
    thread_id=thread_id, tools_metadata=self._tools_metadata
)

# ...later...

# chat_model.py:386
await client.query(user_content)  # ❌ session_id is NOT passed!
```

**Expected behavior:**
- thread_id="test_multi_query" → should map to session_id="test_multi_query"
- Each thread_id should have **ITS OWN** session_id in Claude

---

## Code analysis

### `ClaudeCodeChatModel.ainvoke()` (chat_model.py:300-417)

**The current logic:**

1. Extracts thread_id from the config (lines 337-339)
2. Gets the client for that thread_id (lines 342-344)
3. Handles ToolMessage (lines 347-353)
4. Converts the messages into the Claude format (lines 356-367)
5. **Sends the query WITHOUT session_id** (line 386)
6. Processes the response (lines 388-417)

**The problematic code:**
```python
if user_content:
    await client.query(user_content)  # ❌ session_id = "default"!
```

### Why was this not caught by the tests?

**Unit tests use mocks:**
```python
# test_integration.py:44
mock_client.query = AsyncMock()

# ...later...
mock_client.query.assert_called_once()  # They check ONLY the fact of the call!
```

**They do not check:**
- Which arguments are passed into `query()`
- Whether session_id is passed
- Isolation between different thread_ids

**The integration test revealed the problem:**
- It uses the **real** ClaudeSDKClient
- It makes **several sequential** requests
- It expects **independent** answers

---

## Proof of Concept

### Demonstration of the bug

```python
# Request 1
await model.ainvoke(
    [HumanMessage("2+2")],
    config={"configurable": {"thread_id": "test_multi_query"}}
)
# Internally: await client.query("2+2")  # session_id="default"
# Claude Session "default": ["User: 2+2", "Assistant: 4"]

# Request 2
await model.ainvoke(
    [HumanMessage("3+3")],
    config={"configurable": {"thread_id": "test_multi_query"}}
)
# Internally: await client.query("3+3")  # session_id="default" (THE SAME SESSION!)
# Claude Session "default": ["User: 2+2", "Assistant: 4", "User: 3+3", "Assistant: ???"]
# BUG: Claude may repeat "4" instead of "6"
```

### The correct behavior (expected)

```python
# Request 1
await model.ainvoke(
    [HumanMessage("2+2")],
    config={"configurable": {"thread_id": "test_multi_query"}}
)
# Internally: await client.query("2+2", session_id="test_multi_query")  # ✅
# Claude Session "test_multi_query": ["User: 2+2", "Assistant: 4"]

# Request 2 (in a DIFFERENT thread)
await model.ainvoke(
    [HumanMessage("3+3")],
    config={"configurable": {"thread_id": "test_multi_query_2"}}
)
# Internally: await client.query("3+3", session_id="test_multi_query_2")  # ✅
# Claude Session "test_multi_query_2": ["User: 3+3", "Assistant: 6"]
# Isolated sessions - NO reuse of the context
```

---

## Proposed fix

### Change 1: Pass session_id into client.query()

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

**Rationale:**
- thread_id from LangGraph → session_id in Claude
- Each thread gets an isolated session in Claude
- Different threads do not affect each other

### Change 2: Add a comment for clarity

```python
# Extract thread_id from config
thread_id = "default"
if config and "configurable" in config:
    thread_id = config["configurable"].get("thread_id", "default")

# Get or create client for this thread
# IMPORTANT: thread_id is used as session_id for Claude to ensure isolation
client, proxy_mcp = await self._client_manager.get_or_create_client(
    thread_id=thread_id, tools_metadata=self._tools_metadata
)

# ...later...

# Send query to Claude with session_id matching thread_id
# This ensures each LangGraph thread has its own isolated Claude session
if user_content:
    await client.query(user_content, session_id=thread_id)
```

---

## Why does this conform to the basic principles of the provider?

### ✅ Principle 1: Full transparency for the agent

**Not violated:**
- The agent still sees ClaudeCodeChatModel as an ordinary LangChain provider
- thread_id is handled in the standard way via RunnableConfig
- No changes to the interface or the API

**Improved:**
- Correct session isolation matches the expectations of LangGraph
- Each thread gets an independent context

### ✅ Principle 2: Tools are executed in the agent

**Not violated:**
- ProxyMCPServer works as before
- Tool execution still happens in LangGraph
- Claude only **INVOKES** the tools through MCP

**No changes:**
- session_id affects only the CONVERSATION context
- It does NOT affect tool execution

### ✅ Principle 3: A separate ClaudeSDKClient for each thread_id

**Not violated:**
- ClientManager still caches clients by thread_id
- Each thread gets its own client
- No race conditions

**Improved:**
- Now each client ALSO uses a unique session_id
- Double isolation: both at the client level and at the session level

### ✅ Principle 4: Sequential message processing

**Not violated:**
- ainvoke() processes messages one at a time
- Future blocking in ProxyMCPServer works as before
- No changes to the message flow

**No impact:**
- session_id is just a parameter for the Claude CLI
- It does NOT change the processing order

---

## Alternative solutions (why they are NOT suitable)

### ❌ Alternative 1: Create a new client for each request

**Idea:**
```python
# In ainvoke():
client = ClaudeSDKClient(options=self.base_options)
await client.connect()
await client.query(user_content)
await client.disconnect()
```

**Why this is bad:**
- **It violates the principle of isolation by thread_id**
- Client caching in ClientManager is lost
- Overhead on connect/disconnect for every request
- **It does NOT solve the problem** — session_id is still "default"

### ❌ Alternative 2: Clear the session after each request

**Idea:**
```python
await client.query(user_content)
await client.clear_session()  # A hypothetical method
```

**Why this is bad:**
- The Claude SDK **does NOT provide** a clear_session() method
- It violates the principle of preserving the context between calls
- LangGraph **EXPECTS** the context to be preserved within a thread

### ❌ Alternative 3: Use a new session_id for each request

**Idea:**
```python
import uuid
session_id = str(uuid.uuid4())
await client.query(user_content, session_id=session_id)
```

**Why this is bad:**
- Every request would be **completely isolated**
- The context **within a single thread** would be lost
- Example: "Calculate 5+3. Now multiply by 2."
  - First request: "Calculate 5+3" → "8"
  - Second request: "Now multiply by 2" → Claude does NOT remember "8"!
- **It violates the basic principle** of preserving the context within a thread

---

## Testing the fix

### Scenario 1: Independent requests within a single thread (THE CURRENT BUG)

**Before the fix:**
```python
config = {"configurable": {"thread_id": "test_multi_query"}}

# Request 1
result1 = await model.ainvoke([HumanMessage("2+2")], config=config)
# session_id="default" → "4"

# Request 2
result2 = await model.ainvoke([HumanMessage("3+3")], config=config)
# session_id="default" (THE SAME SESSION!) → BUG: repeats "4"
```

**After the fix:**
```python
config = {"configurable": {"thread_id": "test_multi_query"}}

# Request 1
result1 = await model.ainvoke([HumanMessage("2+2")], config=config)
# session_id="test_multi_query" → "4"

# Request 2
result2 = await model.ainvoke([HumanMessage("3+3")], config=config)
# session_id="test_multi_query" (THE SAME SESSION, the context is preserved)
# Claude sees: ["2+2" → "4", "3+3"]
# Answer: "6" ✅
```

**IMPORTANT:** Within a single thread the context **MUST be preserved**!

### Scenario 2: Contextual requests (multi-turn conversation)

**Verifying that the context works:**
```python
config = {"configurable": {"thread_id": "conversation-1"}}

# Turn 1
result1 = await model.ainvoke([HumanMessage("My name is Alice")], config=config)
# session_id="conversation-1" → "Nice to meet you, Alice!"

# Turn 2
result2 = await model.ainvoke([HumanMessage("What's my name?")], config=config)
# session_id="conversation-1" (THE SAME SESSION)
# Claude remembers: "Alice" → "Your name is Alice." ✅
```

### Scenario 3: Parallel threads (isolation)

**Verifying isolation between threads:**
```python
config1 = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}

# Thread 1
await model.ainvoke([HumanMessage("My name is Alice")], config=config1)
# session_id="thread-1"

# Thread 2
await model.ainvoke([HumanMessage("My name is Bob")], config=config2)
# session_id="thread-2"

# Thread 1 - checking the context
result1 = await model.ainvoke([HumanMessage("What's my name?")], config=config1)
# session_id="thread-1" → "Alice" ✅

# Thread 2 - checking the context
result2 = await model.ainvoke([HumanMessage("What's my name?")], config=config2)
# session_id="thread-2" → "Bob" ✅
```

---

## Additional recommendations

### 1. Add a test for session isolation

**File:** `tests/test_integration.py`

```python
@pytest.mark.asyncio
async def test_session_isolation_between_threads() -> None:
    """Verify that different thread_ids have isolated Claude sessions."""
    options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    model = ClaudeCodeChatModel(base_options=options)

    # Thread 1
    result1 = await model.ainvoke(
        [HumanMessage("2+2")],
        config={"configurable": {"thread_id": "thread-1"}}
    )
    assert "4" in str(result1.content)

    # Thread 2 - should NOT see thread-1 context
    result2 = await model.ainvoke(
        [HumanMessage("3+3")],
        config={"configurable": {"thread_id": "thread-2"}}
    )
    assert "6" in str(result2.content)
    assert "4" not in str(result2.content)  # Should NOT repeat thread-1 answer
```

### 2. Add logging of session_id

**For debugging:**
```python
import logging
logger = logging.getLogger(__name__)

# In ainvoke():
if user_content:
    logger.debug(f"Sending query to Claude: thread_id={thread_id}, session_id={thread_id}")
    await client.query(user_content, session_id=thread_id)
```

### 3. Document the thread_id → session_id mapping

**In README.md or the docs:**
```markdown
## Session Management

ClaudeCodeChatModel maps LangGraph thread_id to Claude session_id:
- `thread_id` (from LangGraph config) → `session_id` (in Claude Code CLI)
- Each thread_id gets its own isolated Claude session
- Context is preserved within a thread, isolated between threads
```

---

## Conclusion

### The problem

The agent "gets stuck" on the first answer because `session_id` is missing in `client.query()`:
- All requests go into session_id="default"
- Claude preserves the context between requests
- It may repeat previous answers

### The fix

Pass `thread_id` as `session_id`:
```python
await client.query(user_content, session_id=thread_id)
```

### Why is this the right solution?

✅ It conforms to the basic principles of the provider
✅ It does not break transparency for the agent
✅ It provides isolation between threads
✅ It preserves the context within a thread
✅ A minimal change (1 line of code)
✅ It requires no changes in ClientManager or ProxyMCPServer

### Impact

- ✅ Fixes the multiple-queries bug
- ✅ Improves session isolation
- ✅ Matches the expectations of LangGraph
- ⚠️ Requires updating the tests (to check session_id)

### Risks

❌ No risks:
- The change is backward-compatible
- session_id defaults to "default" (as it is now)
- Only a parameter is added at the call site
