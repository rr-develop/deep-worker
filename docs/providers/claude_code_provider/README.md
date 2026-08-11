# Claude Code Provider for LangChain

LangChain-compatible AI provider that uses [Claude Code SDK](https://github.com/anthropics/claude-code) as the backend, enabling deep-worker agents to leverage Claude's capabilities while maintaining full transparency at the LangGraph level.

## Features

- **🔗 LangChain Integration**: Drop-in replacement for any LangChain `BaseChatModel`
- **🛠️ Tool Delegation**: All tools execute in deep-worker, not in Claude CLI
- **🔒 Session Isolation**: Each thread_id gets its own isolated ClaudeSDKClient instance
- **⚡ Async Support**: Full async/await support with proper streaming
- **✅ Type-Safe**: Strict typing with pyright compliance
- **🧪 Well-Tested**: 44 tests with 100% pass rate

## Architecture

The provider consists of three main components:

1. **ClaudeCodeChatModel**: LangChain `BaseChatModel` wrapper around ClaudeSDKClient
2. **ProxyMCPServer**: In-process MCP server that delegates tool execution to deep-worker
3. **ClientManager**: Manages ClaudeSDKClient instances for different thread_ids

### How It Works

```
LangGraph Thread
     ↓
ClaudeCodeChatModel.ainvoke()
     ↓
ClientManager.get_or_create_client(thread_id)
     ↓
ClaudeSDKClient + ProxyMCPServer
     ↓
Claude API
     ↓
Tool Call? → ProxyMCPServer (blocks on asyncio.Future)
     ↓
ClaudeCodeChatModel returns AIMessage with tool_calls
     ↓
LangGraph executes tool in deep-worker
     ↓
ToolMessage → resolve_call() unblocks Future
     ↓
Claude continues → Final Answer
```

## Installation

```bash
# Install from local directory
pip install -e /path/to/claude_code_provider

# Or add to requirements.txt
-e /path/to/claude_code_provider
```

**Dependencies:**
- `langchain-core>=0.3.0`
- `claude-agent-sdk` (from a local source checkout)
- `pydantic>=2.0`

## Quick Start

### Basic Usage

```python
from claude_agent_sdk import ClaudeAgentOptions
from claude_code_provider import ClaudeCodeChatModel
from langchain_core.messages import HumanMessage

# Create the model
model = ClaudeCodeChatModel(
    base_options=ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929"
    )
)

# Simple conversation
response = await model.ainvoke(
    [HumanMessage(content="Hello! How are you?")],
    config={"configurable": {"thread_id": "session-1"}}
)

print(response.content)
```

### With Tools

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# Bind tools to model
model = ClaudeCodeChatModel(base_options=None)
model = model.bind_tools([add_numbers, multiply_numbers])

# Invoke with tools
response = await model.ainvoke(
    [HumanMessage(content="What is (5 + 3) * 2?")],
    config={"configurable": {"thread_id": "session-2"}}
)

# If Claude wants to call tools, response.tool_calls will be populated
if response.tool_calls:
    print(f"Claude wants to call: {response.tool_calls}")
else:
    print(f"Final answer: {response.content}")
```

### With LangGraph

```python
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search through documents."""
    # Your search logic here
    return f"Found results for: {query}"

# Create model with tools
model = ClaudeCodeChatModel(base_options=None)

# Create agent graph
agent = create_react_agent(
    model=model,
    tools=[search_documents],
    state_modifier="You are a helpful research assistant."
)

# Run agent
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Find information about quantum computing")]},
    config={"configurable": {"thread_id": "research-1"}}
)

print(result["messages"][-1].content)
```

## Configuration

### Thread ID

The `thread_id` is used to isolate sessions. Each unique `thread_id` gets its own ClaudeSDKClient instance:

```python
# Session 1
await model.ainvoke(
    messages,
    config={"configurable": {"thread_id": "user-123"}}
)

# Session 2 (separate client)
await model.ainvoke(
    messages,
    config={"configurable": {"thread_id": "user-456"}}
)

# Default thread_id
await model.ainvoke(messages)  # Uses thread_id="default"
```

### Base Options

Pass ClaudeAgentOptions to configure the underlying ClaudeSDKClient:

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-sonnet-4-5-20250929",
    # Add other options as needed
)

model = ClaudeCodeChatModel(base_options=options)
```

### Disallowed Tools

By default, all Claude Code built-in tools are disabled to ensure tools execute in deep-worker:

```python
BUILT_IN_TOOLS = [
    "Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep",
    "Task", "TodoWrite", "Skill", "NotebookEdit", "WebFetch",
    "WebSearch", "BashOutput", "KillShell", "AskUserQuestion",
    "SlashCommand", "LS"
]
```

## API Reference

### ClaudeCodeChatModel

```python
class ClaudeCodeChatModel(BaseChatModel):
    def __init__(
        self,
        base_options: ClaudeAgentOptions | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with optional base options."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> "ClaudeCodeChatModel":
        """Bind tools to model (returns new instance)."""

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke model asynchronously."""
```

### ClientManager

```python
class ClientManager:
    async def get_or_create_client(
        self, thread_id: str, tools_metadata: list[ToolMetadata]
    ) -> tuple[ClaudeSDKClient, ProxyMCPServer]:
        """Get or create client for thread_id."""

    async def cleanup_client(self, thread_id: str) -> None:
        """Cleanup client for thread_id."""
```

### ProxyMCPServer

```python
class ProxyMCPServer:
    def has_pending_calls(self) -> bool:
        """Check if there are pending tool calls."""

    def get_pending_calls(self) -> list[PendingCall]:
        """Get list of pending tool calls."""

    async def wait_for_pending_calls(
        self, *, tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> list[PendingCall]:
        """Wait for a matching pending call without polling."""

    async def wait_for_matching_pending_calls(
        self, *,
        expected_calls: Sequence[tuple[str, dict[str, Any]]],
        timeout: float = 30.0,
    ) -> list[PendingCall]:
        """Wait until all matching pending calls exist without polling."""

    def resolve_call(self, call_id: str, result: MCPResult) -> None:
        """Resolve pending tool call with result."""
```

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_chat_model.py -v

# Run with coverage
python3 -m pytest tests/ --cov=claude_code_provider --cov-report=html

# Type checking
python3 -m pyright .
```

## Development

### Project Structure

```
claude_code_provider/
├── __init__.py              # Package entry point
├── chat_model.py            # ClaudeCodeChatModel implementation
├── client_manager.py        # ClientManager implementation
├── proxy_mcp_server.py      # ProxyMCPServer implementation
├── types.py                 # Type definitions
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_chat_model.py   # ChatModel tests (12 tests)
│   ├── test_client_manager.py  # ClientManager tests (11 tests)
│   ├── test_proxy_mcp_server.py  # ProxyMCPServer tests (15 tests)
│   └── test_integration.py  # Integration tests (6 tests)
├── pyproject.toml           # Package config
├── pyrightconfig.json       # Type checking config
└── README.md                # This file
```

### Type Checking

This package uses strict typing with pyright:

```json
{
  "typeCheckingMode": "strict",
  "extraPaths": [
    "/path/to/claude_sdk/src"
  ]
}
```

### Pre-commit Hook

The pre-commit hook automatically runs pyright on changes:

```bash
# Manually run pre-commit
.githooks/pre-commit
```

## Design Decisions

### Why ProxyMCPServer Blocks on asyncio.Future?

This is the key innovation that enables transparent tool delegation:

1. Claude calls tool through MCP
2. ProxyMCPServer handler creates `asyncio.Future` and **blocks**
3. `ClaudeCodeChatModel` sees proxy `ToolUseBlock` values and awaits
   `wait_for_matching_pending_calls(...)`; the MCP handlers wake that
   waiter as each pending call is registered. A single AssistantMessage
   with multiple proxy tool uses returns all matching `tool_calls`.
   There is no sleep loop or repeated polling.
4. `ClaudeCodeChatModel` returns `AIMessage` with `tool_calls`
5. LangGraph executes tool in deep-worker
6. `ToolMessage` triggers `resolve_call()` which resolves the Future
7. ProxyMCPServer unblocks and returns result to Claude
8. Claude continues with final answer

This design maintains full transparency - LangGraph controls all tool execution while Claude thinks it's calling tools directly through MCP.

### Why Disable All Built-In Tools?

All Claude Code built-in tools are disabled (`disallowed_tools`) to ensure:
- **Transparency**: Deep-worker sees all tool calls
- **Control**: LangGraph decides which tools to execute
- **Consistency**: Same tool implementation across different AI providers

### Why Separate ClientManager?

Each `thread_id` needs its own isolated ClaudeSDKClient to prevent:
- **Session mixing**: Different conversations sharing state
- **Race conditions**: Parallel requests interfering with each other
- **Memory leaks**: Unlimited client creation

ClientManager provides thread-safe caching and cleanup.

## Troubleshooting

### "ModuleNotFoundError: No module named 'claude_agent_sdk'"

Make sure `claude_agent_sdk` is imported from source, not pip:

```python
# In client_manager.py and proxy_mcp_server.py
_claude_sdk_path = Path(__file__).parent.parent.parent / "claude_sdk" / "src"
if _claude_sdk_path.exists() and str(_claude_sdk_path) not in sys.path:
    sys.path.insert(0, str(_claude_sdk_path))
```

### "Pydantic validation error"

ClaudeCodeChatModel uses `object.__setattr__` to bypass Pydantic field validation. This is intentional to avoid Pydantic conflicts with LangChain's BaseChatModel.

### "Tool execution timeout"

Default timeout is 60 seconds. Increase it when creating ProxyMCPServer:

```python
proxy = ProxyMCPServer(tools_metadata, timeout=120.0)
```

## License

Same as Claude Code SDK.

## Contributing

This is part of the deep-worker project. See main repository for contribution guidelines.

## Credits

- Built on [Claude Code SDK](https://github.com/anthropics/claude-code)
- Uses [LangChain](https://github.com/langchain-ai/langchain) for agent framework
- Part of the [DeepAgents](https://github.com/yourusername/deep-worker) ecosystem
