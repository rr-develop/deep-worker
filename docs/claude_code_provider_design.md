# Design document: Claude Code SDK as an AI provider for deep-worker

## Summary

This document describes the architecture for integrating Claude Code (via ClaudeSDKClient) as an AI provider for deep-worker, providing full transparency at the LangGraph level.

## Key research findings

### 1. The MCP mechanism in the Claude SDK

The Claude SDK supports in-process MCP servers through the control protocol:

1. The Claude Code CLI sends a `control_request` with subtype `"mcp_message"` (query.py:274-289)
2. The SDK routes the request to the registered MCP server (query.py:357-484)
3. The MCP server invokes the handler for `CallToolRequest` (query.py:437-469)
4. The handler **blocks** (`await handler(call_request)`) until a result is received
5. The result is returned to the CLI through control_response

**Critical point**: The handler is asynchronous and can block for an indefinite amount of time while waiting for external events.

### 2. Session management

- **LangGraph/deep-worker**: uses `thread_id` in `RunnableConfig`
- **Claude SDK**: uses `session_id` when calling `query()`
- Each thread_id must map to a separate ClaudeSDKClient instance

### 3. Message handling

- LangGraph processes messages sequentially
- ClaudeSDKClient works asynchronously and can return multiple messages
- Messages must be aggregated up to the `ResultMessage`

---

## Solution architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                       deep-worker                            │
│                      (LangGraph)                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ invoke(messages, config={thread_id})
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              ClaudeCodeChatModel                             │
│          (LangChain BaseChatModel wrapper)                   │
│                                                               │
│  • Conversion of LangChain ↔ Claude messages                │
│  • ClientManager management                                  │
│  • Tool call handling through ProxyMCPServer                │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ get_or_create_client(thread_id)
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   ClientManager                              │
│                                                               │
│  • ClaudeSDKClient cache keyed by thread_id                 │
│  • Creation of new clients                                   │
│  • Lifecycle management                                      │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ thread_id → ClaudeSDKClient + ProxyMCPServer
               ▼
┌─────────────────────────────────────────────────────────────┐
│              ClaudeSDKClient Instance                        │
│                                                               │
│  mcp_servers = {                                             │
│    "deepworker": ProxyMCPServer                             │
│  }                                                           │
│  disallowed_tools = [all built-in]                          │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ MCP control_request (tools/call)
               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ProxyMCPServer                               │
│             (in-process MCP server)                          │
│                                                               │
│  • Registers tools from deep-worker                         │
│  • Does NOT execute tools                                    │
│  • Blocks on a Future until a result arrives                │
│  • Returns the result to Claude                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed component specification

### 1. ProxyMCPServer

**Purpose**: An in-process MCP server that delegates tool execution to deep-worker.

**Key characteristics**:
- Created with `create_sdk_mcp_server()`
- Registers tools that are NOT executed inside the MCP server
- Uses `asyncio.Future` to block until a result arrives from the outside

**Pseudocode**:

```python
import asyncio
from typing import Any, Dict
from claude_agent_sdk import create_sdk_mcp_server, tool

class ProxyMCPServer:
    """Proxy MCP server for delegating tools to deep-worker."""

    def __init__(self, tools_metadata: List[Dict[str, Any]]):
        """
        Args:
            tools_metadata: Tool metadata from deep-worker
                [{"name": "tool1", "description": "...", "input_schema": {...}}, ...]
        """
        self.pending_calls: Dict[str, asyncio.Future] = {}
        self.tool_handlers = []

        # Create a handler for each tool
        for tool_meta in tools_metadata:
            handler = self._create_tool_handler(tool_meta["name"])

            # Decorate as an MCP tool
            decorated = tool(
                tool_meta["name"],
                tool_meta["description"],
                tool_meta["input_schema"]
            )(handler)

            self.tool_handlers.append(decorated)

        # Create the MCP server
        self.mcp_server_config = create_sdk_mcp_server(
            name="deepworker",
            version="1.0.0",
            tools=self.tool_handlers
        )

    def _create_tool_handler(self, tool_name: str):
        """Create an async handler for a tool."""
        async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
            # Create a Future for this call
            call_id = f"{tool_name}_{id(args)}"
            future = asyncio.Future()
            self.pending_calls[call_id] = {
                'future': future,
                'name': tool_name,
                'args': args
            }

            # WE BLOCK here until a result arrives from the outside
            result = await future

            # Clean up
            del self.pending_calls[call_id]

            return result

        return handler

    def has_pending_calls(self) -> bool:
        """Check whether there are pending calls."""
        return len(self.pending_calls) > 0

    def get_pending_calls(self) -> List[Dict[str, Any]]:
        """Get the list of pending calls."""
        return [
            {
                'id': call_id,
                'name': info['name'],
                'args': info['args']
            }
            for call_id, info in self.pending_calls.items()
        ]

    def resolve_call(self, call_id: str, result: Any):
        """Resolve a pending call with a result."""
        if call_id in self.pending_calls:
            future = self.pending_calls[call_id]['future']

            # Convert the result into MCP format
            mcp_result = {
                "content": [{"type": "text", "text": str(result)}]
            }

            future.set_result(mcp_result)
```

**Critical point**: The handler blocks on `await future`, which suspends execution of the Claude Code CLI until deep-worker executes the tool and calls `resolve_call()`.

---

### 2. ClientManager

**Purpose**: Management of ClaudeSDKClient instances for different thread_id values.

**Pseudocode**:

```python
from typing import Dict, Optional
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

class ClientManager:
    """Manages Claude clients for different thread_id values."""

    def __init__(self, base_options: ClaudeAgentOptions):
        """
        Args:
            base_options: Base options for all clients
        """
        self.base_options = base_options
        self.clients: Dict[str, tuple[ClaudeSDKClient, ProxyMCPServer]] = {}

    async def get_or_create_client(
        self,
        thread_id: str,
        tools_metadata: List[Dict[str, Any]]
    ) -> tuple[ClaudeSDKClient, ProxyMCPServer]:
        """Get or create a client for a thread_id."""

        if thread_id in self.clients:
            return self.clients[thread_id]

        # Create a ProxyMCPServer with the tools
        proxy_mcp = ProxyMCPServer(tools_metadata)

        # Create options with the MCP server
        options = ClaudeAgentOptions(
            **self.base_options.__dict__,
            mcp_servers={
                "deepworker": proxy_mcp.mcp_server_config
            },
            # Block ALL built-in tools
            disallowed_tools=[
                "Read", "Write", "Edit", "MultiEdit",
                "Bash", "Glob", "Grep", "Task",
                "TodoWrite", "Skill", "NotebookEdit",
                "WebFetch", "WebSearch", "BashOutput", "KillShell",
                "AskUserQuestion", "SlashCommand", "LS"
            ]
        )

        # Create and connect the client
        client = ClaudeSDKClient(options=options)
        await client.connect()

        # Cache it
        self.clients[thread_id] = (client, proxy_mcp)

        return client, proxy_mcp

    async def cleanup_client(self, thread_id: str):
        """Clean up a client after the session ends."""
        if thread_id in self.clients:
            client, _ = self.clients[thread_id]
            await client.disconnect()
            del self.clients[thread_id]
```

---

### 3. ClaudeCodeChatModel

**Purpose**: A LangChain BaseChatModel wrapper around ClaudeSDKClient.

**Interface**:
- Inherits from `BaseChatModel`
- Implements `ainvoke()` for asynchronous calls
- Converts LangChain messages ↔ Claude messages
- Handles tool calls through ProxyMCPServer

**Pseudocode**:

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from typing import List, Any, Dict

class ClaudeCodeChatModel(BaseChatModel):
    """LangChain wrapper for Claude Code through the SDK."""

    def __init__(self, base_options: ClaudeAgentOptions):
        super().__init__()
        self.client_manager = ClientManager(base_options)
        self.tools_metadata: List[Dict[str, Any]] = []

    def bind_tools(self, tools: List[BaseTool]) -> "ClaudeCodeChatModel":
        """Bind tools to the model (LangChain API)."""
        # Extract tool metadata
        self.tools_metadata = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.args_schema.model_json_schema()
                    if hasattr(tool, 'args_schema') else {}
            }
            for tool in tools
        ]
        return self

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        config: Optional[RunnableConfig] = None
    ) -> AIMessage:
        """Asynchronous model invocation."""

        # Extract thread_id from config
        thread_id = config.get("configurable", {}).get("thread_id", "default") \
            if config else "default"

        # Get the client for this thread
        client, proxy_mcp = await self.client_manager.get_or_create_client(
            thread_id,
            self.tools_metadata
        )

        # Convert LangChain messages into Claude format
        claude_messages = self._convert_to_claude(messages)

        # Send the messages to Claude
        await client.query(claude_messages, session_id=thread_id)

        # Collect the response
        response_messages = []
        async for msg in client.receive_response():
            response_messages.append(msg)

        # Check whether there are pending tool calls in the MCP server
        if proxy_mcp.has_pending_calls():
            pending = proxy_mcp.get_pending_calls()[0]  # Take the first one

            # Return an AIMessage with tool_calls
            return AIMessage(
                content="",  # Claude may return text before the tool call
                tool_calls=[{
                    'id': pending['id'],
                    'name': pending['name'],
                    'args': pending['args']
                }]
            )

        # No tool calls - return a regular response
        return self._convert_from_claude(response_messages)

    def _convert_to_claude(self, messages: List[BaseMessage]) -> Any:
        """Convert LangChain messages into Claude format."""
        claude_msgs = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                claude_msgs.append({
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": msg.content
                    }
                })

            elif isinstance(msg, AIMessage):
                # If there are tool_calls, do not send them - they already happened
                if not msg.tool_calls:
                    claude_msgs.append({
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": msg.content}]
                        }
                    })

            elif isinstance(msg, ToolMessage):
                # Resolve the pending call in the MCP server
                # Find the matching call_id by tool_call_id
                call_id = msg.tool_call_id  # This is the id from tool_calls

                # Convert the result into MCP format
                result = {
                    "content": [{"type": "text", "text": str(msg.content)}]
                }

                # Resolve the Future in ProxyMCPServer
                # IMPORTANT: this unblocks the MCP handler,
                # which will return the result to Claude
                # There is NO need to send a message via client.query()!

        return claude_msgs

    def _convert_from_claude(
        self,
        messages: List[Message]
    ) -> AIMessage:
        """Convert Claude messages into an AIMessage."""
        # Aggregate all text blocks
        text_parts = []

        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

        return AIMessage(content="\n".join(text_parts))
```

---

## Critical point: handling ToolMessage

When deep-worker executes a tool and creates a `ToolMessage`, the next `ainvoke()` call must:

1. **NOT** send a new message via `client.query()`
2. **Instead**, resolve the corresponding Future in ProxyMCPServer
3. The Future unblocks → the MCP handler returns the result → Claude receives the result
4. Claude continues generation and returns the next response

**Implementation**:

```python
# In ClaudeCodeChatModel.ainvoke()

# Check whether there is a ToolMessage among the messages
tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

if tool_messages:
    # This is a continuation after tool execution

    for tool_msg in tool_messages:
        call_id = tool_msg.tool_call_id

        # Resolve the Future in ProxyMCPServer
        proxy_mcp.resolve_call(call_id, {
            "content": [{"type": "text", "text": str(tool_msg.content)}]
        })

    # Claude has now received the results and will continue generation
    # Wait for the next response
    response_messages = []
    async for msg in client.receive_response():
        response_messages.append(msg)

    # Process the response (it may be another tool call or the final answer)
    if proxy_mcp.has_pending_calls():
        # Another tool call
        pending = proxy_mcp.get_pending_calls()[0]
        return AIMessage(
            content="",
            tool_calls=[{
                'id': pending['id'],
                'name': pending['name'],
                'args': pending['args']
            }]
        )
    else:
        # Final answer
        return self._convert_from_claude(response_messages)
```

---

## Execution flow

### Scenario 1: A simple request without tools

```
1. LangGraph → ainvoke([HumanMessage("Hello")])
2. ClaudeCodeChatModel:
   - Get the client for the thread_id
   - Send via client.query()
3. Claude generates a response (text, no tool calls)
4. ClaudeCodeChatModel:
   - Collect the AssistantMessage
   - Convert it into an AIMessage
   - Return it to LangGraph
```

### Scenario 2: A request with a tool call

```
1. LangGraph → ainvoke([HumanMessage("What files are in /tmp?")])

2. ClaudeCodeChatModel:
   - Get the client for the thread_id
   - Send via client.query()

3. Claude decides to call the "ls" tool
   - The CLI sends an MCP control_request
   - ProxyMCPServer.handler("ls") is invoked
   - The handler creates a Future and BLOCKS

4. ClaudeCodeChatModel:
   - Sees the pending call in proxy_mcp
   - Returns AIMessage(tool_calls=[{id, name:"ls", args}])

5. LangGraph:
   - Sees tool_calls in the AIMessage
   - Executes the "ls" tool through deep-worker
   - Creates a ToolMessage with the result
   - Calls ainvoke([HumanMessage, AIMessage, ToolMessage])

6. ClaudeCodeChatModel:
   - Sees the ToolMessage
   - Calls proxy_mcp.resolve_call(call_id, result)
   - The Future in the handler is resolved
   - The handler returns the result to the CLI
   - The CLI passes the result to Claude

7. Claude:
   - Receives the tool result
   - Generates the final answer
   - "The files in /tmp are: file1, file2, ..."

8. ClaudeCodeChatModel:
   - Collects the response
   - Returns AIMessage(content="The files...")

9. LangGraph receives the final answer
```

---

## Session management

### Creating a client for a new thread

```python
# LangGraph calls with a new thread_id
config = {"configurable": {"thread_id": "session-123"}}
response = await model.ainvoke([HumanMessage("Hello")], config=config)

# ClaudeCodeChatModel:
# - ClientManager creates a new ClaudeSDKClient
# - Creates a new ProxyMCPServer
# - Caches the (client, proxy) pair for "session-123"
```

### Continuing an existing session

```python
# LangGraph calls with the same thread_id
config = {"configurable": {"thread_id": "session-123"}}
response = await model.ainvoke([new_message], config=config)

# ClaudeCodeChatModel:
# - ClientManager returns the existing client from the cache
# - History is preserved in the Claude Code CLI
```

### Cleaning up finished sessions

```python
# After the session ends
await client_manager.cleanup_client("session-123")

# Or periodic cleanup of inactive ones
async def cleanup_inactive(max_idle_time):
    for thread_id, (client, _) in list(client_manager.clients.items()):
        if is_inactive(thread_id, max_idle_time):
            await client_manager.cleanup_client(thread_id)
```

---

## Limitations and edge cases

### 1. Multiple tool calls in a single response

Claude may return several tool calls at once. ProxyMCPServer supports this through the `pending_calls` dictionary.

**Solution**:
```python
if proxy_mcp.has_pending_calls():
    pending_list = proxy_mcp.get_pending_calls()

    # Return an AIMessage with all tool_calls
    return AIMessage(
        content="",
        tool_calls=[
            {'id': p['id'], 'name': p['name'], 'args': p['args']}
            for p in pending_list
        ]
    )
```

### 2. Errors during tool execution

When deep-worker returns an error through a ToolMessage:

```python
# ToolMessage with is_error=True
tool_msg = ToolMessage(
    content="Error: file not found",
    tool_call_id=call_id,
    is_error=True
)

# ProxyMCPServer must return the MCP error format
proxy_mcp.resolve_call(call_id, {
    "content": [{"type": "text", "text": "Error: file not found"}],
    "is_error": True
})
```

### 3. Timeout for tool execution

If deep-worker "hangs" and does not return a result:

```python
# In the ProxyMCPServer handler
async def handler(args):
    future = asyncio.Future()
    # ...

    try:
        # Timeout of 60 seconds
        result = await asyncio.wait_for(future, timeout=60.0)
        return result
    except asyncio.TimeoutError:
        return {
            "content": [{"type": "text", "text": "Tool execution timeout"}],
            "is_error": True
        }
```

### 4. Parallel threads

ClientManager supports parallel threads automatically through separate ClaudeSDKClient instances.

---

## Advantages of this approach

1. **Full transparency**: deep-worker knows nothing about Claude Code - it is just a regular AI provider
2. **Tool isolation**: All tools are executed in deep-worker, not in the Claude CLI
3. **LangChain compatibility**: The standard BaseChatModel interface
4. **Session support**: Each thread_id has its own context
5. **Claude Code pricing model**: We use Claude Code rather than the Anthropic API

---

## Next implementation steps

1. Implement `ProxyMCPServer` with Future-based blocking
2. Implement `ClientManager` with client caching
3. Implement `ClaudeCodeChatModel` with message conversion
4. Tests:
   - A simple request without tools
   - A request with one tool call
   - A request with multiple tool calls
   - Error handling
   - Parallel threads
5. Integration into deep-worker
6. Optimization (caching, cleanup, timeouts)

---

## Conclusion

This architecture is **fully implementable** and satisfies all requirements:

✅ ClaudeSDKClient handles messages from deep-worker
✅ Tools are executed in deep-worker, not in the Claude Code CLI
✅ Full transparency for deep-worker (a standard LangChain provider)
✅ Support for parallel threads through separate instances
✅ Sequential message processing (one at a time)
✅ Mirroring of LangChain message history into Claude sessions

The key innovation: using `asyncio.Future` to block MCP handlers, which makes it possible to suspend Claude until results are received from deep-worker.

---

## Reference information

### Claude SDK files to study

- The Claude Agent SDK client module - `ClaudeSDKClient`
- The SDK's internal query module - control protocol and MCP routing
- The SDK's MCP calculator example - an example of an SDK MCP server
- The SDK's MCP integration tests

### deep-worker files to study

- The agent graph module - `create_deep_agent`
- The middleware package - middleware for tools
- The backend protocol module - backend protocol
