# Implementation plan for the Claude Code Provider for deep-worker

## Base design document
`docs/claude_code_provider_design.md`

## Project structure

```
<repository root>/
├── claude_code_provider/
│   ├── __init__.py              # Public API
│   ├── py.typed                 # Typed package marker
│   ├── proxy_mcp_server.py      # ProxyMCPServer
│   ├── client_manager.py        # ClientManager
│   ├── chat_model.py            # ClaudeCodeChatModel
│   ├── types.py                 # Types and protocols
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_proxy_mcp_server.py
│   │   ├── test_client_manager.py
│   │   ├── test_chat_model.py
│   │   ├── test_integration.py
│   │   └── conftest.py          # Pytest fixtures
│   ├── pyrightconfig.json       # pyright strict settings
│   └── pyproject.toml           # Dependencies and settings
```

## Implementation stages (TDD approach)

### Stage 1: Infrastructure (Steps 1-3)

#### Step 1: Create the package structure
- [ ] Create the `claude_code_provider/` directory
- [ ] Create `__init__.py` with the public API
- [ ] Create `py.typed` for PEP 561
- [ ] Create the `tests/` directory

#### Step 2: Set up the configuration
- [ ] Create `pyproject.toml` with the dependencies:
  - claude-agent-sdk (from a local checkout of the Claude Agent SDK)
  - langchain-core
  - pytest, pytest-asyncio, pytest-mock
- [ ] Create `pyrightconfig.json` with strict settings
- [ ] Create `conftest.py` with basic fixtures

#### Step 3: Update the pre-commit hook
- [ ] Add a `claude_code_provider/` check to the pre-commit hook
- [ ] Verify that the hook works

### Stage 2: ProxyMCPServer (Steps 4-6)

#### Step 4: Types for ProxyMCPServer
- [ ] Create `types.py` with:
  - `ToolMetadata` TypedDict
  - `PendingCall` TypedDict
  - `MCPResult` TypedDict

#### Step 5: Tests for ProxyMCPServer (TDD)
File: `tests/test_proxy_mcp_server.py`

**Test 1: Initialization**
```python
def test_proxy_mcp_server_init():
    """ProxyMCPServer must initialize with tool metadata."""
    tools_metadata = [
        {"name": "test_tool", "description": "Test", "input_schema": {"arg": str}}
    ]
    proxy = ProxyMCPServer(tools_metadata)
    assert proxy.mcp_server_config is not None
    assert proxy.mcp_server_config["type"] == "sdk"
```

**Test 2: Creating handlers**
```python
@pytest.mark.asyncio
async def test_tool_handler_blocks_on_future():
    """The handler must block on the Future until a result is received."""
    # Create a ProxyMCPServer
    # Invoke the handler in the background
    # Check that the handler has not finished
    # Resolve the Future
    # Check that the handler returned a result
```

**Test 3: Pending calls**
```python
def test_get_pending_calls():
    """get_pending_calls must return the list of pending calls."""
```

**Test 4: Resolve call**
```python
@pytest.mark.asyncio
async def test_resolve_call():
    """resolve_call must resolve the Future and return the result."""
```

**Test 5: Timeout**
```python
@pytest.mark.asyncio
async def test_handler_timeout():
    """The handler must return an error on timeout."""
```

#### Step 6: ProxyMCPServer implementation
- [ ] Implement `ProxyMCPServer.__init__`
- [ ] Implement `_create_tool_handler`
- [ ] Implement `has_pending_calls`
- [ ] Implement `get_pending_calls`
- [ ] Implement `resolve_call`
- [ ] Add a timeout to the handler
- [ ] All tests must pass
- [ ] Check pyright strict: 0 errors

### Stage 3: ClientManager (Steps 7-9)

#### Step 7: Tests for ClientManager (TDD)
File: `tests/test_client_manager.py`

**Test 1: Initialization**
```python
def test_client_manager_init():
    """ClientManager must initialize with base options."""
```

**Test 2: Client creation**
```python
@pytest.mark.asyncio
async def test_get_or_create_client_new():
    """Must create a new client for a new thread_id."""
```

**Test 3: Client caching**
```python
@pytest.mark.asyncio
async def test_get_or_create_client_cached():
    """Must return the cached client for an existing thread_id."""
```

**Test 4: Parallel threads**
```python
@pytest.mark.asyncio
async def test_multiple_threads():
    """Must support several parallel threads."""
```

**Test 5: Cleanup**
```python
@pytest.mark.asyncio
async def test_cleanup_client():
    """Must clean up a client correctly."""
```

#### Step 8: ClientManager implementation
- [ ] Implement `ClientManager.__init__`
- [ ] Implement `get_or_create_client`
- [ ] Implement `cleanup_client`
- [ ] All tests must pass
- [ ] Check pyright strict: 0 errors

#### Step 9: Mock ClaudeSDKClient for tests
- [ ] Create `tests/conftest.py` with a mock client
- [ ] The mock must simulate connect/disconnect
- [ ] The mock must simulate query/receive_response

### Stage 4: ClaudeCodeChatModel (Steps 10-12)

#### Step 10: Tests for ClaudeCodeChatModel (TDD)
File: `tests/test_chat_model.py`

**Test 1: Initialization**
```python
def test_chat_model_init():
    """ClaudeCodeChatModel must initialize with options."""
```

**Test 2: bind_tools**
```python
def test_bind_tools():
    """bind_tools must extract tool metadata."""
```

**Test 3: A simple request (without tools)**
```python
@pytest.mark.asyncio
async def test_ainvoke_simple():
    """ainvoke must handle a simple request without tools."""
```

**Test 4: A request with a tool call**
```python
@pytest.mark.asyncio
async def test_ainvoke_with_tool_call():
    """ainvoke must return an AIMessage with tool_calls."""
```

**Test 5: ToolMessage handling**
```python
@pytest.mark.asyncio
async def test_ainvoke_with_tool_message():
    """ainvoke must resolve the Future when a ToolMessage is received."""
```

**Test 6: Multiple tool calls**
```python
@pytest.mark.asyncio
async def test_ainvoke_multiple_tool_calls():
    """ainvoke must handle several tool calls."""
```

**Test 7: Message conversion**
```python
def test_convert_to_claude():
    """_convert_to_claude must convert messages correctly."""
```

**Test 8: Conversion from Claude**
```python
def test_convert_from_claude():
    """_convert_from_claude must convert responses correctly."""
```

**Test 9: Sessions (thread_id)**
```python
@pytest.mark.asyncio
async def test_thread_isolation():
    """Different thread_id values must use different clients."""
```

#### Step 11: ClaudeCodeChatModel implementation
- [ ] Implement `ClaudeCodeChatModel.__init__`
- [ ] Implement `bind_tools`
- [ ] Implement `ainvoke` (basic logic)
- [ ] Implement `_convert_to_claude`
- [ ] Implement `_convert_from_claude`
- [ ] Implement tool call handling
- [ ] Implement ToolMessage handling
- [ ] All tests must pass
- [ ] Check pyright strict: 0 errors

#### Step 12: Edge case handling
- [ ] Errors during tool execution (is_error)
- [ ] Timeout for tool execution
- [ ] Handling of empty responses
- [ ] Handling of ThinkingBlock
- [ ] Tests for all edge cases

### Stage 5: Integration (Steps 13-15)

#### Step 13: Integration tests
File: `tests/test_integration.py`

**Test 1: End-to-end without tools**
```python
@pytest.mark.asyncio
async def test_e2e_simple_conversation():
    """Full cycle: request → response (without tools)."""
```

**Test 2: End-to-end with a tool**
```python
@pytest.mark.asyncio
async def test_e2e_with_tool():
    """Full cycle: request → tool call → result → response."""
```

**Test 3: Parallel sessions**
```python
@pytest.mark.asyncio
async def test_e2e_parallel_sessions():
    """Several parallel sessions must be isolated."""
```

#### Step 14: Public API
- [ ] Export the classes in `__init__.py`:
  - `ClaudeCodeChatModel`
  - `ProxyMCPServer`
  - `ClientManager`
- [ ] Add docstrings
- [ ] Add usage examples

#### Step 15: Documentation
- [ ] Create `claude_code_provider/README.md` with:
  - A description of the package
  - Usage examples
  - Requirements
- [ ] Add type hints everywhere
- [ ] Verify that all public APIs are documented

### Stage 6: Finalization (Steps 16-17)

#### Step 16: Quality checks
- [ ] Run all tests: `pytest claude_code_provider/tests/`
- [ ] Check coverage: `pytest --cov=claude_code_provider`
- [ ] Run pyright: `pyright claude_code_provider/`
- [ ] Check the pre-commit hook
- [ ] Fix all errors and warnings

#### Step 17: Integration with deep-worker
- [ ] Create a usage example in deep-worker
- [ ] Test with a real ClaudeSDKClient
- [ ] Make sure it works transparently

## Order of execution

1. **Stage 1** (infrastructure) - in full
2. **Stage 2** (ProxyMCPServer):
   - Step 4: Types
   - Step 5: Tests (write ALL of them)
   - Step 6: Implementation (make the tests pass one by one)
3. **Stage 3** (ClientManager):
   - Step 7: Tests (write ALL of them)
   - Step 8: Implementation
   - Step 9: Mocks
4. **Stage 4** (ClaudeCodeChatModel):
   - Step 10: Tests (write ALL of them)
   - Step 11: Implementation
   - Step 12: Edge cases
5. **Stage 5** (integration):
   - Step 13: Integration tests
   - Step 14: API
   - Step 15: Documentation
6. **Stage 6** (finalization):
   - Step 16: Quality checks
   - Step 17: Integration with deep-worker

## Success criteria

- [ ] All tests pass
- [ ] Pyright strict: 0 errors, 0 warnings
- [ ] The pre-commit hook works
- [ ] Test coverage > 90%
- [ ] All public APIs are documented
- [ ] Works with a real ClaudeSDKClient
- [ ] Transparent integration with deep-worker

## Dependencies

### External
- claude-agent-sdk (from a local checkout of the Claude Agent SDK)
- langchain-core
- pytest
- pytest-asyncio
- pytest-mock

### Internal
- The path to the Claude Agent SDK checkout in pyright's extraPaths

## Progress reports

Progress will be reported after each stage is completed:
- After Stage 1: Infrastructure is ready
- After Stage 2: ProxyMCPServer is implemented and tested
- After Stage 3: ClientManager is implemented and tested
- After Stage 4: ClaudeCodeChatModel is implemented and tested
- After Stage 5: Integration is complete
- After Stage 6: The project is ready for use
