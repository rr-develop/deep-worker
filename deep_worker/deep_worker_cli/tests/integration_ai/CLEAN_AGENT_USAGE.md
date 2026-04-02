# Clean Agent Fixture Usage Guide

## Problem

Deep-worker agents use `InMemorySaver` for checkpoint storage, which accumulates conversation history in the agent process memory. When running multiple tests against the same agent, history from previous test runs persists and can cause:

1. **Test result pollution** - Previous test messages appear in current test
2. **Duplicate messages** - History loads old + new messages
3. **Flaky tests** - Tests pass/fail depending on execution order

## Solution

Use the `clean_agent` fixture to ensure agent has clean state before test execution.

### ⚠️ CRITICAL: skip_permissions=True by default

The `clean_agent` fixture ALWAYS uses `--skip-permissions` by default to prevent tests
from hanging on approval prompts. This is essential for:

- **CI/CD environments**: No human to approve tool calls
- **Automated testing**: Tests must run without interaction
- **Tool invocation tests**: Most tests invoke tools (read, write, execute)

**Only set `skip_permissions=False` for tests that specifically test the approval workflow.**

## Usage Examples

### Example 1: Basic Clean Agent

```python
import pytest
from tests.integration_ai.clean_agent_fixture import clean_agent

def test_dwcli_query_visibility(clean_agent):
    \"\"\"Test with guaranteed clean agent state.\"\"\"
    # Set up clean agent
    agent_process = clean_agent("test_dwcli_query_visibility", working_dir="/tmp")

    # Now agent has no history
    # Run your test...

    # Cleanup happens automatically after test
```

### Example 2: Using Agent Name from Test Function

```python
def test_dwcli_fresh_start(clean_agent):
    \"\"\"Test that requires clean agent.\"\"\"
    agent_name = "test_dwcli_fresh_start"
    agent_process = clean_agent(agent_name)

    # Test logic here - agent guaranteed to have no history
    session = DwcliTerminalSession(agent_name=agent_name)
    session.start()
    # ...
```

### Example 3: Parametrized Test for Both Clean and Dirty

```python
import pytest
from tests.integration_ai.clean_agent_fixture import agent_state

@pytest.mark.parametrize("should_clean", [True, False])
def test_history_handling(clean_agent, should_clean):
    \"\"\"Test works with both clean and preserved history.\"\"\"
    agent_name = "test_history_handling"

    if should_clean:
        # Use clean agent
        agent_process = clean_agent(agent_name)
    else:
        # Use existing agent (may have history)
        # Just connect normally
        pass

    # Test logic...
```

### Example 4: Module-Level Setup

For tests that need clean agent for entire test module:

```python
import pytest

@pytest.fixture(scope="module")
def module_clean_agent(clean_agent):
    \"\"\"Clean agent for entire module.\"\"\"
    return clean_agent("test_module_agent")

def test_first(module_clean_agent):
    # Uses module-level clean agent
    ...

def test_second(module_clean_agent):
    # Same clean agent, but now has history from test_first
    ...
```

## When to Use Clean Agent

### ✅ Use Clean Agent When:

- Testing initial connection/first message behavior
- Verifying empty history display
- Testing that UI shows correct state with no messages
- Benchmarking first-message performance
- Ensuring reproducible test results
- **Any test that invokes tools** (uses skip_permissions=True by default)

### ❌ Don't Use Clean Agent When:

- Testing history loading/pagination
- Testing scroll behavior with many messages
- Testing conversation continuity
- Performance testing with large history
- Testing history-dependent features (e.g., context retention)

## Implementation Details

The `clean_agent` fixture performs these steps:

1. **Kill existing agent** - Terminates running agent process if exists
2. **Clean directories** - Removes `~/.deep_worker/{agent_name}/` data
3. **Remove logs** - Deletes `~/.deep_worker/logs/{agent_name}.*`
4. **Start fresh** - Launches new agent with `--skip-permissions` (default: True)
   - ⚠️ **CRITICAL**: `--skip-permissions` is ALWAYS used by default
   - This prevents tests from hanging on approval prompts
   - Only disable for tests that specifically test the approval workflow
5. **Wait for registration** - Ensures agent is ready before test
6. **Cleanup** - Terminates agent after test completion

## Troubleshooting

### Agent Not Starting

If agent fails to start:
- Check if port is already in use
- Verify `deep-worker` command is in PATH
- Check agent logs in `~/.deep_worker/logs/`

### History Still Present

If history persists despite using clean_agent:
- Verify fixture is called before running test logic
- Check that agent name matches between fixture and test
- Ensure no other test is using same agent name concurrently

### Timeout During Registration

If agent registration times out:
- Increase timeout in `clean_agent_fixture.py` (default: 30s)
- Check system resources (CPU, memory)
- Verify no deadlocks in agent initialization

## Migration Guide

### Before (Flaky Tests):

```python
def test_dwcli_query_visibility():
    # Agent may have history from previous runs
    agent_name = "test_dwcli_query_visibility"

    # Start agent (may already be running)
    agent_process = subprocess.Popen([...])

    # Test may see old messages...
```

### After (Reliable Tests):

```python
def test_dwcli_query_visibility(clean_agent):
    # Agent guaranteed clean
    agent_process = clean_agent("test_dwcli_query_visibility")

    # Test sees only new messages
```

## See Also

- `conftest.py` - Pytest configuration
- `agent_fixture.py` - General agent fixtures
- `test_terminal_emulator.py` - Example usage
