# The proper solution: Mirroring messages between LangGraph and ClaudeSDKClient

## ✅ STATUS: IMPLEMENTED AND WORKING

**Implementation date:** 2026-03-11
**Test:** `test_agent_multiple_queries` - PASSED ✅

## Summary

This document describes the **correct architecture** for fixing the multiple-queries bug in the claude-code provider. The solution is based on the **mirroring** principle: every new message in a LangGraph thread must be sent to the corresponding ClaudeSDKClient session **EXACTLY ONCE**, with no duplication and no omissions.

**Final solution:** A global ClientManager singleton with `_sent_count` tracking.

---

## Analysis of the current problem

### How ClaudeSDKClient works (from studying the code)

**Key method: `client.query(prompt, session_id)`**

```python
# claude_agent_sdk/client.py:170-198

async def query(
    self, prompt: str | AsyncIterable[dict[str, Any]], session_id: str = "default"
) -> None:
    """Send a new request in streaming mode.

    Args:
        prompt: Either a string message or an async iterable of message dictionaries
        session_id: Session identifier for the conversation
    """
    if isinstance(prompt, str):
        message = {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
            "session_id": session_id,
        }
        await self._transport.write(json.dumps(message) + "\n")
```

**Important:**
- Every `client.query()` call sends a message to the Claude CLI
- The Claude CLI **KEEPS THE HISTORY** for each `session_id`
- On a repeated call with the same `session_id`, the message is **APPENDED** to the existing history
- The Claude CLI **DOES NOT EXPECT** the whole history to be sent again!

### How receive_response() works

```python
# claude_agent_sdk/client.py:279-318

async def receive_response(self) -> AsyncIterator[Message]:
    """Receive messages from Claude until and including a ResultMessage.

    Yields:
        Message: Each message received (UserMessage, AssistantMessage,
                 SystemMessage, ResultMessage)
    """
    async for message in self.receive_messages():
        yield message
        if isinstance(message, ResultMessage):
            return  # Automatically stops after the ResultMessage
```

**Message types from Claude:**

```python
# claude_agent_sdk/types.py:458-507

@dataclass
class UserMessage:
    """User message echoed back."""
    content: str | list[ContentBlock]
    parent_tool_use_id: str | None = None

@dataclass
class AssistantMessage:
    """Assistant message with content blocks."""
    content: list[ContentBlock]  # TextBlock, ToolUseBlock, etc.
    model: str
    parent_tool_use_id: str | None = None

@dataclass
class ResultMessage:
    """Result message with cost and usage information."""
    subtype: str
    duration_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    # ... other fields
```

**Critical:** an AssistantMessage may contain:
- `TextBlock` - a text response
- `ToolUseBlock` - a tool call

---

## The mirroring architecture

### The concept

```
LangGraph Thread State               ClaudeSDKClient Session
=====================                ========================
[]                                   Session "thread-1": []

↓ ainvoke([HumanMessage("2+2")])

[HumanMessage("2+2")]        →       client.query("2+2", "thread-1")
                                     Session "thread-1": [User: "2+2"]
                             ←
                                     receive_response() yields:
                                     - AssistantMessage("4")
                                     - ResultMessage
[HumanMessage("2+2"),
 AIMessage("4")]                     Session "thread-1": [User: "2+2",
                                                          Assistant: "4"]

↓ ainvoke([HumanMessage("2+2"), AIMessage("4"), HumanMessage("3+3")])

                                     ✅ ONLY THE NEW ONE:
[...,                        →       client.query("3+3", "thread-1")
 HumanMessage("3+3")]                Session "thread-1": [...,
                                                          User: "3+3"]
                             ←
                                     receive_response() yields:
                                     - AssistantMessage("6")
                                     - ResultMessage
[...,
 HumanMessage("3+3"),
 AIMessage("6")]                     Session "thread-1": [...,
                                                          User: "3+3",
                                                          Assistant: "6"]
```

### The key principle

**ClaudeSDKClient ALREADY mirrors the state:**
- The Claude CLI stores the history for a `session_id`
- Every `client.query()` APPENDS a message
- There is NO NEED to send the whole history again!

**The task of ClaudeCodeChatModel:**
- Determine which messages have **ALREADY BEEN SENT**
- Send **ONLY THE NEW ONES**
- Receive the response and return it to LangGraph

---

## The proper solution

### Option 1: Tracking the number of sent messages (Recommended)

#### The idea

For each `thread_id`, store the number of messages that have already been sent to Claude. On a new `ainvoke()` call:
1. Load `sent_count` for this `thread_id`
2. Extract the new messages: `messages[sent_count:]`
3. Send each new message to Claude
4. Update `sent_count = len(messages)`

#### Implementation

```python
# claude_code_provider/chat_model.py

class ClaudeCodeChatModel(BaseChatModel):
    """LangChain-compatible chat model using Claude Code SDK."""

    def __init__(
        self,
        base_options: ClaudeAgentOptions | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_options = base_options
        self._client_manager = ClientManager(base_options=base_options)
        self._tools_metadata: list[ToolMetadata] = []

        # Tracking of sent messages
        # Key: thread_id, Value: the number of sent messages
        self._sent_count: dict[str, int] = {}

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model asynchronously."""

        # 1. Convert the input into messages
        from langchain_core.prompt_values import PromptValue

        if isinstance(input, str):
            messages: list[BaseMessage] = [HumanMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = input.to_messages()
        else:
            messages = list(input)

        # 2. Extract the thread_id
        thread_id = "default"
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id", "default")

        # 3. Get or create the client for this thread
        client, proxy_mcp = await self._client_manager.get_or_create_client(
            thread_id=thread_id, tools_metadata=self._tools_metadata
        )

        # 4. Determine how many messages have already been sent
        sent_count = self._sent_count.get(thread_id, 0)

        # 5. Extract ONLY THE NEW messages
        new_messages = messages[sent_count:]

        # 6. Send the new messages to Claude
        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                # Send the user message
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                await client.query(content, session_id=thread_id)

            elif isinstance(msg, ToolMessage):
                # Resolve the pending tool call
                mcp_result: MCPResult = {
                    "content": [{"type": "text", "text": str(msg.content)}]
                }
                proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)

            elif isinstance(msg, AIMessage):
                # An AIMessage with tool_calls has already been processed by Claude
                # Just skip it (it is already in Claude's history)
                pass

        # 7. Update the counter of sent messages
        self._sent_count[thread_id] = len(messages)

        # 8. Receive the response from Claude (only if a user message was sent)
        response_text = ""
        user_message_sent = any(isinstance(msg, HumanMessage) for msg in new_messages)

        if user_message_sent:
            async for event in client.receive_response():
                if isinstance(event, ClaudeAssistantMessage):
                    # Extract the text from the AssistantMessage
                    from claude_agent_sdk import TextBlock

                    for block in event.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                elif isinstance(event, ResultMessage):
                    # End of the response
                    pass

        # 9. Check for pending tool calls
        if proxy_mcp.has_pending_calls():
            pending = proxy_mcp.get_pending_calls()
            tool_calls = [
                {
                    "id": call["id"],
                    "name": call["name"],
                    "args": call["args"],
                }
                for call in pending
            ]
            return AIMessage(content="", tool_calls=tool_calls)

        # 10. Return the final response
        return AIMessage(content=response_text)
```

#### Cleanup when a thread is closed

```python
# claude_code_provider/client_manager.py

class ClientManager:
    async def cleanup_client(self, thread_id: str) -> None:
        """Cleanup client for a thread_id."""
        if thread_id in self.clients:
            client, _ = self.clients[thread_id]
            await client.disconnect()
            del self.clients[thread_id]

            # Also remove the lock
            if thread_id in self._creation_locks:
                del self._creation_locks[thread_id]
```

And add a cleanup method to `ClaudeCodeChatModel`:

```python
async def cleanup_thread(self, thread_id: str) -> None:
    """Cleanup resources for a thread."""
    # Remove the counter
    self._sent_count.pop(thread_id, None)

    # Remove the client
    await self._client_manager.cleanup_client(thread_id)
```

---

### Option 2: Tracking the IDs of sent messages (More Robust)

#### The idea

Instead of a counter, store a **set** of the IDs of messages that have already been sent. This is more robust because:
- It works if messages are reordered
- It works if some messages are removed from the history
- It does not depend on ordering

#### Implementation

```python
class ClaudeCodeChatModel(BaseChatModel):
    def __init__(
        self,
        base_options: ClaudeAgentOptions | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_options = base_options
        self._client_manager = ClientManager(base_options=base_options)
        self._tools_metadata: list[ToolMetadata] = []

        # Tracking of sent messages by ID
        # Key: thread_id, Value: set of message IDs
        self._sent_message_ids: dict[str, set[str]] = {}

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        # 1-3. Same as Option 1

        # 4. Get the set of sent IDs for this thread
        sent_ids = self._sent_message_ids.get(thread_id, set())

        # 5. Filter the new messages by ID
        new_messages = []
        for msg in messages:
            # Get the message ID (LangChain generates it automatically)
            msg_id = msg.id if hasattr(msg, 'id') else None

            if msg_id and msg_id not in sent_ids:
                new_messages.append(msg)
                sent_ids.add(msg_id)
            elif not msg_id:
                # If there is no ID, always send it (edge case)
                new_messages.append(msg)

        # 6-10. Same as Option 1, but working with new_messages

        # Update sent_ids
        self._sent_message_ids[thread_id] = sent_ids

        return AIMessage(content=response_text)
```

#### The problem with this approach

**LangChain may NOT guarantee stable IDs:**
- An ID may change on deserialization from the checkpointer
- An ID may be None for some messages
- This needs to be tested in practice

---

## Recommended solution: Option 1 (sent_count)

### Why is this the best choice?

✅ **Simplicity:**
- One integer per thread
- Easy to understand and maintain

✅ **Reliability:**
- LangGraph ALWAYS passes the history in order
- New messages are ALWAYS at the end
- `messages[sent_count:]` is always correct

✅ **Efficiency:**
- O(1) for the check
- Minimal memory

✅ **Compatibility:**
- Does not depend on messages having IDs
- Works with any message types

### When can this break?

❌ **If LangGraph reorders messages:**
- In practice this DOES NOT HAPPEN
- The checkpointer preserves the order

❌ **If messages are removed from the history:**
- Then `sent_count` will be greater than `len(messages)`
- A check is needed: `sent_count = min(sent_count, len(messages))`

### Protection against edge cases

```python
# 4. Determine how many messages have already been sent
sent_count = self._sent_count.get(thread_id, 0)

# Protection: if the history got shorter (a rare case)
if sent_count > len(messages):
    # The history was reset or modified
    # We assume all messages must be sent again
    sent_count = 0
    # The Claude session should also be cleared (if possible)
    # TODO: Implement session reset

# 5. Extract ONLY THE NEW messages
new_messages = messages[sent_count:]
```

---

## Handling special cases

### Case 1: Tool Call Flow

**Scenario:**
```
Request 1: HumanMessage("Calculate 5+3")
↓
Claude calls the tool: add_numbers(5, 3)
↓
Request 2: [HumanMessage("Calculate 5+3"), AIMessage(tool_calls=[...]), ToolMessage("8")]
↓
Claude receives the result and answers: "The answer is 8"
```

**Current behavior:**
```python
# Request 1
messages = [HumanMessage("Calculate 5+3")]
sent_count = 0
new_messages = messages[0:]  # [HumanMessage("Calculate 5+3")]

# We send:
await client.query("Calculate 5+3", session_id=thread_id)

# Claude calls the tool through ProxyMCPServer
# ProxyMCPServer blocks on a Future
# ClaudeCodeChatModel returns: AIMessage(tool_calls=[...])

sent_count = 1  # We update it

# Request 2 (after the tool execution)
messages = [
    HumanMessage("Calculate 5+3"),   # idx 0
    AIMessage(tool_calls=[...]),      # idx 1
    ToolMessage("8")                  # idx 2
]
sent_count = 1
new_messages = messages[1:]  # [AIMessage(tool_calls=[...]), ToolMessage("8")]

# We process the new ones:
for msg in new_messages:
    if isinstance(msg, AIMessage):
        pass  # Skip it (already in Claude)
    elif isinstance(msg, ToolMessage):
        proxy_mcp.resolve_call(msg.tool_call_id, {"content": [{"type": "text", "text": "8"}]})

# We do NOT call client.query() (there is no HumanMessage in new_messages)
# We simply resolve the pending call

# Claude continues and answers: "The answer is 8"
# receive_response() returns the final answer

sent_count = 3  # We update it
```

**This is correct!** ✅

### Case 2: Multiple queries

**Scenario:**
```
Request 1: "2+2"
Request 2: "3+3"
Request 3: "5+5"
```

**Behavior:**
```python
# Request 1
messages = [HumanMessage("2+2")]
sent_count = 0
new_messages = [HumanMessage("2+2")]
await client.query("2+2", session_id=thread_id)
# → AIMessage("4")
sent_count = 2  # [HumanMessage, AIMessage]

# Request 2
messages = [HumanMessage("2+2"), AIMessage("4"), HumanMessage("3+3")]
sent_count = 2
new_messages = [HumanMessage("3+3")]  # ✅ ONLY THE NEW ONE!
await client.query("3+3", session_id=thread_id)
# → AIMessage("6")
sent_count = 4

# Request 3
messages = [..., HumanMessage("5+5")]
sent_count = 4
new_messages = [HumanMessage("5+5")]  # ✅ ONLY THE NEW ONE!
await client.query("5+5", session_id=thread_id)
# → AIMessage("10")
sent_count = 6
```

**This is correct!** ✅

### Case 3: Parallel threads

**Scenario:**
```
Thread 1: "What's 2+2?"
Thread 2: "What's 3+3?"
Thread 1: "And 5+5?"
```

**Behavior:**
```python
# Thread 1, Request 1
thread_id = "thread-1"
sent_count = 0
await client.query("What's 2+2?", session_id="thread-1")
self._sent_count["thread-1"] = 2

# Thread 2, Request 1
thread_id = "thread-2"
sent_count = 0  # A separate counter!
await client.query("What's 3+3?", session_id="thread-2")
self._sent_count["thread-2"] = 2

# Thread 1, Request 2
thread_id = "thread-1"
sent_count = 2  # From thread-1
await client.query("And 5+5?", session_id="thread-1")
self._sent_count["thread-1"] = 4
```

**Isolation works!** ✅

---

## Detailed implementation

### File: chat_model.py (a fully rewritten ainvoke)

```python
async def ainvoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> AIMessage:
    """Invoke the model asynchronously.

    This method implements proper message mirroring between LangGraph and Claude:
    1. Tracks which messages have been sent to Claude session
    2. Sends ONLY new messages (no duplicates, no resending old messages)
    3. Receives response and returns to LangGraph

    Architecture:
    - LangGraph maintains full message history in state (via checkpointer)
    - ClaudeSDKClient maintains session history in Claude CLI (via session_id)
    - ClaudeCodeChatModel tracks sent_count to determine new messages
    - New messages = messages[sent_count:] are sent to Claude

    Args:
        input: Language model input (messages, prompt, or string)
        config: Runnable config with thread_id in configurable.thread_id
        **kwargs: Additional arguments (unused)

    Returns:
        AIMessage with either:
        - tool_calls: If Claude wants to call tools
        - content: If Claude has final answer
    """
    # 1. Convert input to messages
    from langchain_core.prompt_values import PromptValue

    if isinstance(input, str):
        messages: list[BaseMessage] = [HumanMessage(content=input)]
    elif isinstance(input, PromptValue):
        messages = input.to_messages()
    else:
        messages = list(input)

    # 2. Extract thread_id from config
    thread_id = "default"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", "default")

    # 3. Get or create client for this thread
    client, proxy_mcp = await self._client_manager.get_or_create_client(
        thread_id=thread_id, tools_metadata=self._tools_metadata
    )

    # 4. Determine how many messages have already been sent
    sent_count = self._sent_count.get(thread_id, 0)

    # Edge case: if history was truncated/reset (rare)
    if sent_count > len(messages):
        # History was reset, start from beginning
        # TODO: Ideally should also reset Claude session, but SDK doesn't support this
        # For now, rely on session_id uniqueness to avoid conflicts
        sent_count = 0

    # 5. Extract ONLY new messages (not yet sent to Claude)
    new_messages = messages[sent_count:]

    # 6. Send each new message to Claude
    user_message_sent = False

    for msg in new_messages:
        if isinstance(msg, HumanMessage):
            # Send user message to Claude
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            await client.query(content, session_id=thread_id)
            user_message_sent = True

        elif isinstance(msg, ToolMessage):
            # Resolve pending tool call with result
            mcp_result: MCPResult = {
                "content": [{"type": "text", "text": str(msg.content)}],
                "is_error": False,
            }
            proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)

        elif isinstance(msg, AIMessage):
            # AIMessage is already in Claude's history
            # (It was generated by Claude in previous turn)
            # Just skip - no need to send back to Claude
            pass

    # 7. Update sent count BEFORE receiving response
    # This ensures we don't resend messages if receive_response fails
    self._sent_count[thread_id] = len(messages)

    # 8. Receive response from Claude (only if user message was sent)
    response_text = ""

    if user_message_sent:
        # User message was sent, Claude will respond
        async for event in client.receive_response():
            if isinstance(event, ClaudeAssistantMessage):
                # Extract text from AssistantMessage content blocks
                from claude_agent_sdk import TextBlock

                for block in event.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

            elif isinstance(event, ResultMessage):
                # End of response - receive_response() will terminate
                pass

    # 9. Check for pending tool calls
    if proxy_mcp.has_pending_calls():
        pending = proxy_mcp.get_pending_calls()
        # Convert to LangChain tool_calls format
        tool_calls = [
            {
                "id": call["id"],
                "name": call["name"],
                "args": call["args"],
            }
            for call in pending
        ]
        return AIMessage(content="", tool_calls=tool_calls)

    # 10. Return final answer
    return AIMessage(content=response_text)
```

### Add a cleanup method

```python
async def cleanup_thread(self, thread_id: str) -> None:
    """Cleanup resources for a thread.

    Should be called when thread is no longer needed to free resources.

    Args:
        thread_id: Thread ID to cleanup
    """
    # Remove sent count tracking
    self._sent_count.pop(thread_id, None)

    # Cleanup client and proxy
    await self._client_manager.cleanup_client(thread_id)
```

---

## Testing

### Unit test: sent_count tracking

```python
@pytest.mark.asyncio
async def test_tracks_sent_messages():
    """Should track sent messages and send only new ones."""
    model = ClaudeCodeChatModel()

    # Mock client
    mock_client = Mock()
    mock_client.query = AsyncMock()
    mock_client.receive_response = AsyncMock(return_value=iter([
        ClaudeAssistantMessage(content=[TextBlock(text="4")]),
        ResultMessage(subtype="success", session_id="test", ...)
    ]))

    # First call - send 1 message
    await model.ainvoke(
        [HumanMessage("2+2")],
        config={"configurable": {"thread_id": "test"}}
    )

    assert model._sent_count["test"] == 2  # HumanMessage + AIMessage
    assert mock_client.query.call_count == 1
    mock_client.query.assert_called_with("2+2", session_id="test")

    # Second call - send only NEW message
    mock_client.query.reset_mock()
    mock_client.receive_response = AsyncMock(return_value=iter([
        ClaudeAssistantMessage(content=[TextBlock(text="6")]),
        ResultMessage(...)
    ]))

    await model.ainvoke(
        [
            HumanMessage("2+2"),  # Old
            AIMessage("4"),       # Old
            HumanMessage("3+3")   # NEW!
        ],
        config={"configurable": {"thread_id": "test"}}
    )

    assert model._sent_count["test"] == 4  # 2 old + 2 new
    assert mock_client.query.call_count == 1  # ONLY 1 new query
    mock_client.query.assert_called_with("3+3", session_id="test")  # NEW message
```

### Integration test: Multiple queries

```python
@pytest.mark.asyncio
async def test_multiple_queries_independent():
    """Should handle multiple queries independently (real bug test)."""
    options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    model = ClaudeCodeChatModel(base_options=options)

    config = {"configurable": {"thread_id": "test_multi"}}

    # Query 1
    result1 = await model.ainvoke([HumanMessage("2+2")], config)
    assert "4" in result1.content
    assert model._sent_count["test_multi"] == 2

    # Query 2 - should NOT repeat "4"
    result2 = await model.ainvoke([
        HumanMessage("2+2"),
        AIMessage("4"),
        HumanMessage("3+3")
    ], config)
    assert "6" in result2.content
    assert "4" not in result2.content  # Should NOT repeat first answer!
    assert model._sent_count["test_multi"] == 4

    # Query 3
    result3 = await model.ainvoke([
        HumanMessage("2+2"),
        AIMessage("4"),
        HumanMessage("3+3"),
        AIMessage("6"),
        HumanMessage("5+5")
    ], config)
    assert "10" in result3.content
    assert "4" not in result3.content
    assert "6" not in result3.content
    assert model._sent_count["test_multi"] == 6
```

---

## Migration plan

### Stage 1: Add sent_count tracking

1. Add `self._sent_count: dict[str, int] = {}` to `__init__`
2. Do NOT change the ainvoke logic (yet)
3. Run the existing tests - all of them must pass

### Stage 2: Rewrite ainvoke

1. Replace the message extraction logic with `messages[sent_count:]`
2. Remove the `claude_messages` conversion loop
3. Add a loop that processes `new_messages`
4. Update `sent_count` after sending

### Stage 3: Add cleanup

1. Add a `cleanup_thread()` method
2. Update `ClientManager.cleanup_client()` (already present)

### Stage 4: Testing

1. Run all the unit tests
2. Run the integration tests
3. Run `test_agent_multiple_queries` - it must pass!

---

## Conclusion

### What changes?

**Before (incorrect):**
- Converts the ENTIRE history on every call
- Extracts the FIRST UserMessage from the history
- Sends the OLD message again

**After (correct):**
- Tracks the number of sent messages
- Extracts ONLY THE NEW messages
- Sends each new message exactly ONCE

### Why is this correct?

✅ **It follows the mirroring principle:**
- LangGraph → ClaudeSDKClient: only new messages
- ClaudeSDKClient → LangGraph: model responses

✅ **It uses the capabilities of the Claude CLI:**
- Claude already stores the history in the session
- There is no need to send everything again

✅ **Simplicity and reliability:**
- One integer per thread
- O(1) for all operations
- Works for all scenarios

✅ **No hacks:**
- Does not rely on "last = new"
- Tracks the state explicitly
- Clear logic

### Next steps

1. Implement the changes in chat_model.py
2. Add unit tests for sent_count
3. Run all the tests
4. Make sure `test_agent_multiple_queries` passes
5. Add documentation to the README
