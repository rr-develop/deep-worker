

# AI-Powered Integration Testing Infrastructure

Comprehensive testing infrastructure for `deep_worker_cli` using AI-powered validation through claude_sdk meta-queries.

## Table of Contents

- [Overview](#overview)
- [Critical Testing Principles](#critical-testing-principles)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Cost Management](#cost-management)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

This infrastructure solves the challenge of testing non-deterministic AI agent responses. Instead of brittle string matching, we use **meta-queries** - asking one AI to validate another AI's response.

### Key Innovations

1. **Meta-Query Validation**: AI validates AI responses
2. **Smart Agent Lifecycle**: Automatic cleanup with debug preservation
3. **Type Safety**: Strict Pydantic models throughout
4. **Cost Optimization**: Haiku-first with fallback to better models

### Why AI Validation?

```python
# Traditional approach (FRAGILE)
assert "4" in response  # Breaks if agent says "four" or "2+2=4"

# AI-powered approach (ROBUST)
result = await check_reply("What is 2+2?", response)
assert result.is_valid  # Works for any correct answer
```

## Critical Testing Principles

### 1. Skip Permissions by Default

**⚠️ CRITICAL: All tests MUST use `skip_permissions=True` (default)!**

Integration tests invoke tools (read, write, execute, etc.) and would hang forever
waiting for approval prompts that can't be provided in CI/CD environments.

```python
# ✓ CORRECT - uses default skip_permissions=True
with agent_fixture('test_my_agent'):
    ...

# ✓ CORRECT - explicitly set to True
with agent_fixture('test_my_agent', skip_permissions=True):
    ...

# ✗ WRONG - will hang forever on tool approvals!
with agent_fixture('test_my_agent', skip_permissions=False):
    ...  # Test will NEVER complete
```

**Exception:** Only set `skip_permissions=False` for tests that specifically test
the approval/interrupt workflow (HITL - human-in-the-loop). These tests are rare
and must be marked accordingly.

### 2. Agent Session Isolation

**NEVER use production agents in tests!**

```python
# ✗ WRONG - affects production
with agent_fixture('my_production_agent'):
    ...

# ✓ CORRECT - isolated test agent
with agent_fixture('test_my_agent'):
    ...
```

All test agent names MUST start with `test_` prefix. This is enforced by validation.

### 2. Agent Naming Convention

```
Pattern: test_<purpose>_<feature>

Examples:
- test_agent_basic      → Basic functionality
- test_agent_streaming  → Streaming responses
- test_agent_tools      → Tool usage
- test_agent_memory     → Memory/state
```

### 3. Agent Lifecycle Management

The `agent_fixture` handles three scenarios intelligently:

**a) Fixture creates agent + test PASSES:**
```
→ Terminate agent
→ Clean up socket files
→ Remove from registry
```

**b) Fixture creates agent + test FAILS:**
```
→ PRESERVE agent for debugging
→ Keep socket for inspection
→ Log connection command
```

**c) Pre-existing test agent:**
```
→ Reuse existing agent
→ NEVER terminate
→ Verify correct type
```

## Quick Start

### Installation

```bash
# Ensure claude_sdk is available
export CLAUDE_SDK_PATH="/path/to/claude_sdk"

# Install dependencies
uv pip install -e .
```

### Basic Test Example

```python
import pytest
from tests.integration_ai import agent_fixture, check_reply

@pytest.fixture
def agent(request):
    '''Provide test agent.'''
    with agent_fixture('test_agent_basic', working_dir='/tmp') as agent:
        yield agent

@pytest.mark.asyncio
async def test_simple_query(agent):
    '''Test agent answers simple query.'''
    from deep_worker_cli.client.agent_client import AgentClient

    # Connect to agent
    client = AgentClient(agent.agent_name)
    await client.connect()

    # Send query
    response = await client.invoke_agent("What is 2+2?")

    # Validate with AI
    result = await check_reply(
        query_text="What is 2+2?",
        reply_text=response['response']['content']
    )

    await client.disconnect()

    assert result.is_valid, f"Invalid response: {result.comment}"
```

### Running Tests

```bash
# Run all AI integration tests
pytest tests/integration_ai/ -v

# Run with logging
pytest tests/integration_ai/ -v -s --log-cli-level=INFO

# Run specific test
pytest tests/integration_ai/test_basic.py::test_simple_query -v

# Run with cost tracking
pytest tests/integration_ai/ -v --log-cli-level=INFO | grep "cost:"
```

## Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                  Integration AI Tests                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐           ┌────────────────┐          │
│  │  Test Files  │──────────>│ agent_fixture  │          │
│  │  test_*.py   │           │ - Start agent  │          │
│  └──────────────┘           │ - Lifecycle    │          │
│         │                    │ - Cleanup      │          │
│         │                    └────────────────┘          │
│         │                            │                   │
│         v                            v                   │
│  ┌──────────────┐           ┌────────────────┐          │
│  │ check_reply()│<──────────│  claude_sdk    │          │
│  │ - Meta-query │           │  - query()     │          │
│  │ - Validate   │           │  - Options     │          │
│  └──────────────┘           └────────────────┘          │
│         │                            │                   │
│         v                            v                   │
│  ┌──────────────┐           ┌────────────────┐          │
│  │ Pydantic     │           │  Deep Worker   │          │
│  │ Models       │           │  - Process     │          │
│  │ - Validation │           │  - Client      │          │
│  └──────────────┘           │  - IPC         │          │
│                              └────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

### Data Flow: Response Validation

```
1. Test → Agent
   └─> Send query via IPC

2. Agent → Test
   └─> Return response (non-deterministic)

3. Test → check_reply()
   ├─> Build meta-query prompt
   ├─> Send to Claude (haiku)
   └─> Get validation result

4. check_reply() → Claude API
   ├─> Structured prompt prevents injection
   ├─> Request JSON response
   └─> Parse validation

5. Test
   └─> Assert is_valid or fail with comment
```

## API Reference

### agent_fixture()

```python
@contextmanager
def agent_fixture(
    agent_name: str,
    working_dir: Optional[Path] = None,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    skip_permissions: bool = True,
) -> Generator[AgentFixtureInfo, None, None]:
```

Context manager for test agent lifecycle.

**Parameters:**
- `agent_name`: Agent name (must start with 'test_')
- `working_dir`: Working directory (default: /tmp)
- `model`: AI model to use
- `provider`: AI provider
- `skip_permissions`: Skip prompts (default: True)
  - ⚠️ **CRITICAL**: This MUST be True for all tests that invoke tools!
  - Only set to False for tests that specifically test the approval workflow (HITL)
  - Changing this default will break 95% of integration tests

**Yields:**
- `AgentFixtureInfo`: Agent details

**Raises:**
- `ValueError`: If name doesn't start with 'test_'
- `RuntimeError`: If agent fails to start

### check_reply()

```python
async def check_reply(
    query_text: str,
    reply_text: str,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    strict_mode: bool = False,
    additional_context: Optional[str] = None,
) -> CheckReplyResult:
```

Validate reply using AI meta-query.

**Parameters:**
- `query_text`: Original query
- `reply_text`: Reply to validate
- `model`: Claude model (haiku/sonnet/opus)
- `strict_mode`: Use strict validation
- `additional_context`: Extra context

**Returns:**
- `CheckReplyResult`: Validation outcome

**Raises:**
- `ValidationError`: If validation fails
- `ImportError`: If claude_sdk unavailable

### validate_test_reply()

```python
async def validate_test_reply(
    query_text: str,
    reply_text: str,
    test_name: str = "reply_validation",
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    strict_mode: bool = False,
) -> bool:
```

Convenience function that raises AssertionError if invalid.

**Usage:**
```python
await validate_test_reply("What is 2+2?", response)
# Passes if valid, fails test if not
```

## Examples

### Example 1: Simple Arithmetic

```python
@pytest.mark.asyncio
async def test_arithmetic(agent):
    '''Test agent can do arithmetic.'''
    client = AgentClient(agent.agent_name)
    await client.connect()

    response = await client.invoke_agent("Calculate 15 * 7")

    await client.disconnect()

    result = await check_reply(
        query_text="Calculate 15 * 7",
        reply_text=response['response']['content'],
        model="haiku"
    )

    assert result.is_valid, f"Failed: {result.comment}"
```

### Example 2: Streaming Response

```python
@pytest.mark.asyncio
async def test_streaming(agent):
    '''Test streaming responses.'''
    client = AgentClient(agent.agent_name)
    await client.connect()

    chunks = []
    # Note: Streaming implementation depends on your agent
    response = await client.invoke_agent("Count from 1 to 5", stream=False)

    await client.disconnect()

    content = response['response']['content']
    result = await check_reply("Count from 1 to 5", content)

    assert result.is_valid
```

### Example 3: Batch Validation

```python
from tests.integration_ai import batch_check_replies

@pytest.mark.asyncio
async def test_multiple_queries(agent):
    '''Test multiple queries in batch.'''
    client = AgentClient(agent.agent_name)
    await client.connect()

    queries = [
        "What is 2+2?",
        "What is 3*3?",
        "What is 10-5?"
    ]

    responses = []
    for query in queries:
        resp = await client.invoke_agent(query)
        responses.append(resp['response']['content'])

    await client.disconnect()

    pairs = list(zip(queries, responses))
    results = await batch_check_replies(pairs, model="haiku")

    assert all(r.is_valid for r in results)
```

## Cost Management

### Model Costs

| Model  | Input (per 1M tokens) | Output (per 1M tokens) | Use Case |
|--------|-----------------------|------------------------|----------|
| Haiku  | $0.25                 | $1.25                  | Default, fast validation |
| Sonnet | $3.00                 | $15.00                 | Complex validation |
| Opus   | $15.00                | $75.00                 | Critical validation |

### Estimated Costs

```python
# 100 tests with haiku (default)
# ~$0.10 USD

# 100 tests with sonnet
# ~$0.50 USD

# 100 tests with opus
# ~$2.50 USD
```

### Cost Optimization Tips

1. **Use haiku by default** - Good for most validations
2. **Batch validations** - Reduces overhead
3. **Cache results** - Don't re-validate same responses
4. **Monitor costs** - Check logs for `cost:` entries

```python
# Monitor costs in tests
result = await check_reply(query, reply, model="haiku")
logger.info(f"Validation cost: ${result.cost_usd:.4f}")
```

## Troubleshooting

### Agent Won't Start

**Problem:** `RuntimeError: Agent failed to start within 10s`

**Solutions:**
1. Check socket doesn't exist: `ls /tmp/deep_worker_test_*.sock`
2. Check no stale process: `ps aux | grep deep-worker`
3. Increase timeout in agent_fixture.py
4. Check logs: `cat /tmp/deep_worker_test_agent.log`

### Agent Exists But Can't Connect

**Problem:** Agent registered but connection fails

**Solutions:**
```bash
# Check if process is alive
ps aux | grep test_agent

# Check socket permissions
ls -l /tmp/deep_worker_test_agent.sock

# Try manual connection
dwcli test_agent

# Clean up registry
rm /tmp/deep_worker_registry.json
```

### Validation Always Fails

**Problem:** `check_reply()` returns `is_valid=False`

**Solutions:**
1. Check response format:
   ```python
   logger.debug(f"Response: {response}")
   ```

2. Try stricter model:
   ```python
   result = await check_reply(query, reply, model="sonnet")
   ```

3. Add context:
   ```python
   result = await check_reply(
       query, reply,
       additional_context="Agent should provide numerical answer"
   )
   ```

### claude_sdk Not Found

**Problem:** `ImportError: claude_sdk not available`

**Solution:**
```bash
# Check claude_sdk location
ls -la $CLAUDE_SDK_PATH/src/claude_agent_sdk/

# Add to PYTHONPATH
export PYTHONPATH="$CLAUDE_SDK_PATH/src:$PYTHONPATH"

# Or update check_reply.py CLAUDE_SDK_PATH
```

### Test Cleanup Issues

**Problem:** Test agents not cleaned up

**Manual cleanup:**
```bash
# List all test agents
deep-worker --list | grep test_

# Terminate specific agent
deep-worker --terminate test_agent_name

# Force kill if needed
ps aux | grep test_agent | awk '{print $2}' | xargs kill -9

# Clean sockets
rm /tmp/deep_worker_test_*.sock

# Clean registry
rm /tmp/deep_worker_registry.json
```

## Best Practices

### 1. Agent Naming

```python
# ✓ GOOD
test_agent_basic
test_agent_streaming
test_agent_memory

# ✗ BAD
my_agent  # No test_ prefix
test-agent-basic  # Use underscores, not hyphens
```

### 2. Validation Model Selection

```python
# ✓ GOOD - haiku for simple validation
await check_reply(query, reply, model="haiku")

# ✓ GOOD - sonnet for complex validation
await check_reply(query, reply, model="sonnet", strict_mode=True)

# ✗ BAD - opus unnecessarily expensive
await check_reply("What is 2+2?", reply, model="opus")
```

### 3. Error Handling

```python
# ✓ GOOD - clear error messages
result = await check_reply(query, reply)
assert result.is_valid, (
    f"Agent failed to answer query correctly.\n"
    f"Query: {query}\n"
    f"Reply: {reply}\n"
    f"Reason: {result.comment}"
)

# ✗ BAD - unclear failures
assert result.is_valid
```

### 4. Agent Cleanup

```python
# ✓ GOOD - let fixture handle it
with agent_fixture('test_agent') as agent:
    # ... test code ...
# Agent cleaned up automatically

# ✗ BAD - manual management
agent = start_agent('test_agent')
try:
    # ... test code ...
finally:
    terminate_agent(agent)  # Error-prone
```

### 5. Test Isolation

```python
# ✓ GOOD - each test gets fresh agent
@pytest.fixture
def agent():
    with agent_fixture('test_agent_unique') as ag:
        yield ag

# ✗ BAD - sharing agent between tests
@pytest.fixture(scope="module")  # DON'T DO THIS
def agent():
    ...
```

## Related Documentation

- [CHECK_REPLY_GUIDE.md](./CHECK_REPLY_GUIDE.md) - Detailed validation guide
- [models.py](./models.py) - Pydantic models reference
- [IPC_PROTOCOL.md](../../docs/IPC_PROTOCOL.md) - IPC protocol spec

## Contributing

When adding new infrastructure:

1. **Add type hints** - Full typing required
2. **Add docstrings** - Document all public APIs
3. **Add tests** - Test infrastructure components
4. **Update docs** - Keep README current
5. **Run type check** - `pyright tests/integration_ai/`

## License

Same as deep_worker_cli project.
