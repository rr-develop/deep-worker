# Design: restoring the ProxyMCPServer proxy architecture

## Problem

Commits `a8cbdcd` and `01872d7` (Apr 9) replaced the proxy architecture with direct execution.
Tools are now executed INSIDE the MCP handler → LangGraph does not see tool calls
in `result.messages` → dwcli cannot display tool calls → agent_process broadcasts
only the final response.

## Goal

Restore the proxy: tool calls go through LangGraph, are visible in `result.messages`,
and are broadcast to dwcli. Event-driven, no polling, no extra threads,
everything in a single event loop.

## How the Claude SDK works (findings from research)

When Claude calls an MCP tool:

1. **The Claude CLI sends `AssistantMessage(ToolUseBlock)` IMMEDIATELY** — BEFORE the tool
   starts executing. This is a regular message; it travels through the message stream
   (`_message_send` queue) and is available in `receive_response()`.

2. **The Claude CLI sends `control_request(subtype="mcp_message")`** — a request to
   execute the MCP tool. The SDK calls `_spawn_control_request_handler()` →
   `asyncio.create_task()` → the handler runs in a **separate asyncio task**.

3. **The handler returns a result** → the SDK sends a `control_response` back to the CLI.

4. **The Claude CLI receives the tool result**, continues generation, and sends further
   messages and a `ResultMessage`.

**Key point**: step 2 (handler) runs in a task separate from step 1 (message stream).
`receive_response()` CAN receive `AssistantMessage(ToolUseBlock)` and continue
processing WITHOUT waiting for the tool handler to finish.

## Solution architecture

### Principle: receive_response exits on ToolUseBlock

```
ainvoke() calls receive_response()
    ↓
AssistantMessage(TextBlock) → accumulate text
AssistantMessage(ToolUseBlock) → record tool call
    ↓
Got a ToolUseBlock → EXIT receive_response (break/return)
    ↓ (do NOT wait for ResultMessage)
ainvoke() checks: there is a ToolUseBlock → collect pending_calls
    ↓
Return AIMessage(tool_calls=[...]) to LangGraph
    ↓
LangGraph executes the tool via ToolNode → ToolMessage in state
    ↓
LangGraph calls ainvoke() again with the ToolMessage
    ↓
ainvoke() calls resolve_call(tool_call_id, result)
    → the Future is unblocked
    → the MCP handler returns the result
    → control_response goes to the Claude CLI
    ↓
ainvoke() does NOT call client.query() — the result already went out via MCP
ainvoke() calls receive_response() again
    ↓
The Claude CLI continues generation → new AssistantMessage → ResultMessage
    ↓
If there are more tool calls → repeat the cycle
If it is the final response → return AIMessage(content=text)
```

### Message flow (event-driven, no polling)

```
                    ainvoke()                LangGraph               MCP Handler
                       │                        │                       │
client.query(msg) ─────┤                        │                       │
                       │                        │                       │
receive_response()  ───┤                        │                       │
  │                    │                        │                       │
  ├─ TextBlock ────────┤ (accumulate text)      │                       │
  ├─ ToolUseBlock ─────┤ (record tool call)     │                       │
  │  BREAK! ───────────┤                        │         ┌─────────────┤
  │                    │                        │         │ handler()   │
  │                    │                        │         │ await future│
  │                    │                        │         │ (blocked)   │
  │  has_pending=True ─┤                        │         │             │
  │                    │                        │         │             │
  │  return AIMessage( │                        │         │             │
  │    tool_calls)  ───┼────────────────────────┤         │             │
  │                    │                  ToolNode:       │             │
  │                    │                  execute(tool)   │             │
  │                    │                        │         │             │
  │                    │              ToolMessage│         │             │
  │                    │◄───────────────────────┤         │             │
  │                    │                        │         │             │
  │ resolve_call() ────┼────────────────────────┼─────────┤             │
  │                    │                        │         │ future done │
  │                    │                        │         │ return res  │
  │                    │                        │         └─────────────┤
  │                    │                        │                       │
  │                    │                        │    control_response───┤──→ Claude CLI
  │                    │                        │                       │
receive_response()  ───┤                        │                       │
  │                    │                        │                       │
  ├─ TextBlock ────────┤ (accumulate text)      │                       │
  ├─ ResultMessage ────┤ BREAK                  │                       │
  │                    │                        │                       │
  │  return AIMessage( │                        │                       │
  │    content=text)───┼────────────────────────┤                       │
                       │                  (graph continues)             │
```

### Change 1: ProxyMCPServer — restore the Future-based proxy

**File:** `claude_code_provider/proxy_mcp_server.py`

Remove `direct_handler`. All tool handlers use the Future-based proxy:

```python
def _create_tool_handler(self, tool_name: str) -> Any:
    """Create async handler that blocks on Future.
    
    ⚠️ ARCHITECTURAL RULE: the handler MUST block on the Future!
    This ensures tool calls pass through the LangGraph ToolNode,
    which is required for:
    1. Visibility of tool calls in result.messages (for broadcast to dwcli)
    2. Correct checkpointer behavior (saving tool history)
    3. Compatibility with interrupt_on (HITL approval)
    
    DO NOT REPLACE with direct execution! Executing directly in the handler
    makes tools invisible to LangGraph and dwcli.
    """
    async def handler(args: dict[str, Any]) -> MCPResult:
        self._call_counter += 1
        call_id = f"{tool_name}_{self._call_counter}_{id(args)}"
        
        future: asyncio.Future[MCPResult] = asyncio.Future()
        self.pending_calls[call_id] = {
            "future": future,
            "name": tool_name,
            "args": args,
        }
        
        # Block on the Future — resolve_call() unblocks it
        # once LangGraph has executed the tool and returned the result
        result = await future
        return result
    
    return handler
```

Remove `tool_executors`, `_make_runtime_executor`, `_make_tool_call_executor`,
and the `__agent_name__` injection — all of this belongs to direct execution, which is being deleted.

Keep `working_dir` and `agent_name` — they are needed for other purposes
(agent_name for subagent thread_id).

### Change 2: ainvoke() — exit receive_response on ToolUseBlock

**File:** `claude_code_provider/chat_model.py`

```python
async for event in client.receive_response():
    if isinstance(event, AssistantMessage):
        for block in event.content:
            if isinstance(block, TextBlock):
                response_text += block.text
            elif isinstance(block, ToolUseBlock):
                tool_use_blocks.append(block)
        
        # If there is a ToolUseBlock — EXIT IMMEDIATELY
        # Do NOT wait for ResultMessage — the Claude CLI is waiting for the tool result
        if tool_use_blocks:
            break
    
    elif isinstance(event, ResultMessage):
        break
```

### Change 3: ainvoke() — resolve_call for ToolMessage

**File:** `claude_code_provider/chat_model.py`

When a `ToolMessage` is received from LangGraph — call `resolve_call()` instead of `client.query()`:

```python
elif isinstance(msg, ToolMessage):
    # Resolve pending tool call — unblocks the MCP handler Future
    tool_call_id = msg.tool_call_id
    mcp_result: MCPResult = {
        "content": [{"type": "text", "text": str(msg.content)}],
    }
    proxy_mcp.resolve_call(tool_call_id, mcp_result)
    tool_resolved = True
```

### Change 4: ainvoke() — building pending_calls from ToolUseBlock

Instead of checking `proxy_mcp.has_pending_calls()` (which depends on a race condition
between the spawned task and receive_response), build tool_calls directly from
`tool_use_blocks`:

```python
if tool_use_blocks:
    tool_calls = [
        {
            "id": block.id,
            "name": f"mcp__deepworker__{block.name}" if not block.name.startswith("mcp__") else block.name,
            "args": block.input,
        }
        for block in tool_use_blocks
    ]
    return AIMessage(content=response_text, tool_calls=tool_calls)
```

BUT: `resolve_call` needs the `call_id` from `pending_calls`, not from `ToolUseBlock.id`.
A mapping is needed: `ToolUseBlock.id` → `pending_call.id`. Or use `ToolUseBlock.id`
directly as the pending_calls key (since the MCP handler in the spawned task receives the same id).

**Recommended approach**: the MCP handler uses `tool_use_id` from the control_request
as the `call_id`. Then `ToolUseBlock.id` == `call_id` and no mapping is needed.

Needs verification: does the Claude SDK use the same ID in `ToolUseBlock` and in
`control_request` → `_handle_sdk_mcp_request`? If so, `block.id` can be used
as the key for `resolve_call`.

**Alternative approach**: wait for pending_calls to appear (they are created in the spawned
task from the control_request) and use their ID. There is a small delay (< 1ms) between
receiving the ToolUseBlock in the message stream and the pending_call appearing in the spawned task.
`await asyncio.sleep(0)` — yielding control to the event loop — should be enough.

```python
if tool_use_blocks:
    # Yield control so the spawned task can process the control_request
    await asyncio.sleep(0)
    
    if proxy_mcp.has_pending_calls():
        pending = proxy_mcp.get_pending_calls()
        tool_calls = [...]
        return AIMessage(content=response_text, tool_calls=tool_calls)
```

### Change 5: Remove the direct execution infrastructure

**Files:** `chat_model.py`, `proxy_mcp_server.py`, `client_manager.py`

Remove:
- the `_tool_executors` dict
- `_make_runtime_executor()`
- `_make_tool_call_executor()`
- `_has_injected_tool_call_id()`
- the `tool_executors` parameter in `ProxyMCPServer.__init__`
- the `tool_executors` parameter in `ClientManager.get_or_create_client`
- `direct_handler` in `ProxyMCPServer._create_tool_handler`
- the `__agent_name__` injection in `direct_handler`

Tools are invoked by the LangGraph ToolNode, which uses `BaseTool.ainvoke()`.
It does not need wrappers — ToolNode injects `ToolRuntime` itself.

### Change 6: bind_tools — metadata only, no executors

**File:** `chat_model.py`

```python
def bind_tools(self, tools, **kwargs):
    tools_metadata = []
    for tool_item in tools:
        if isinstance(tool_item, BaseTool):
            # Extract metadata only — tools execute via LangGraph ToolNode
            metadata = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": schema,
            }
            tools_metadata.append(metadata)
    
    new_model = ClaudeCodeChatModel(base_options=self._base_options)
    new_model._tools_metadata = tools_metadata
    return new_model
```

### Change 7: Mapping tool_call_id between LangGraph and MCP

The LangGraph ToolNode uses `tool_call.id` from `AIMessage.tool_calls`. This ID
must match the `call_id` in ProxyMCPServer pending_calls so that
`resolve_call(tool_call_id, result)` works.

**Problem**: the `pending_calls` key is built in the handler as
`f"{tool_name}_{counter}_{id(args)}"`, whereas `tool_call.id` in AIMessage is taken
from ToolUseBlock.id (Claude format: `toolu_xxxx`).

**Solution**: use `ToolUseBlock.id` as the pending_calls key:

```python
# In the handler (when the control_request arrives):
# call_id must match ToolUseBlock.id
# But the handler does not know ToolUseBlock.id...
```

This is a problem. The handler receives args via the MCP protocol, but does not receive
`tool_use_id` directly. It needs to be checked whether the Claude SDK passes `tool_use_id`
to the MCP handler.

**Alternative solution**: in `ainvoke()`, when building `AIMessage(tool_calls)` from
pending_calls, use `pending_call["id"]` as `tool_call.id`. Then LangGraph
will return `ToolMessage(tool_call_id=pending_call["id"])`, and `resolve_call` will find it by that key.

```python
# In ainvoke:
tool_calls = [
    {
        "id": call["id"],  # pending_call["id"] = "{tool_name}_{counter}_{...}"
        "name": call["name"],
        "args": call["args"],
    }
    for call in proxy_mcp.get_pending_calls()
]
return AIMessage(content=response_text, tool_calls=tool_calls)

# Later, when the ToolMessage is received:
proxy_mcp.resolve_call(msg.tool_call_id, mcp_result)
# tool_call_id == pending_call["id"] → match
```

This is a working approach — it is exactly how the original implementation in `bcc7583` worked.

### Change 8: Update the tests

- Remove tests for direct_handler, _make_runtime_executor, etc.
- Restore/update tests for the Future-based proxy
- Update the `test_fallback_to_future_without_executors` test — this is now the main path
- Add a test: ToolUseBlock → pending_calls → resolve_call → result

### Implementation order

1. Restore the Future-based handler in ProxyMCPServer (remove direct execution)
2. Update ainvoke: exit on ToolUseBlock, resolve_call for ToolMessage
3. Remove the tool_executors infrastructure from bind_tools, ClientManager
4. Update the tests
5. E2E test: tool calls are visible in result.messages
6. Integration test: dwcli displays tool calls

### What is preserved

- `mandatory thread_id` in ainvoke (from the previous design)
- `agent_name` in ProxyMCPServer (for subagent thread_id)
- `working_dir` in ProxyMCPServer
- Subagent thread isolation (thread_id in subagents.py)
- TaskManager on the same provider (task_manager.py)

## Risks

1. **Timing between ToolUseBlock and pending_calls**: the ToolUseBlock arrives via
   the message stream, while the pending_call is created via the control_request in a spawned task.
   There may be a race condition if the ToolUseBlock arrives before the control_request.
   `await asyncio.sleep(0)` should solve this.

2. **Multiple tool calls in a single turn**: Claude may call several tools
   at once. All ToolUseBlocks arrive in one AssistantMessage. All pending_calls
   must be resolved before calling receive_response again.

3. **Tool call ID mapping**: it must be confirmed that pending_call.id is used consistently
   as tool_call.id throughout the whole LangGraph cycle.
