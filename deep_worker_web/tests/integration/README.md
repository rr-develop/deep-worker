# Integration Tests for Deep-Worker-Web

This directory contains integration tests that verify deep-worker-web functionality with real deep-worker agents.

## Overview

Integration tests use real deep-worker agent processes (via multiprocessing) to test:
- AgentService RPyC connections
- REST API endpoints with real agents
- WebSocket broadcasting and message callbacks

## Running Integration Tests

```bash
# Run all integration tests
python3 -m pytest tests/integration/ -v -m integration

# Run specific test file
python3 -m pytest tests/integration/test_agent_service_integration.py -v -m integration

# Run without integration tests (unit tests only)
python3 -m pytest tests/unit/ -v
```

## Test Structure

### `agent_fixture.py`
Provides `agent_fixture()` context manager for creating real deep-worker agents:
- Uses multiprocessing with 'spawn' start method (avoids fork deadlocks)
- Auto-assigns RPyC ports (no port conflicts)
- Smart cleanup: preserves failed test agents for debugging
- Based on `deep_worker_cli/tests/integration_ai/agent_fixture.py`

Example:
```python
with agent_fixture() as agent_info:
    # agent_info.name, agent_info.pid, agent_info.rpyc_port
    # Test with real agent
    pass
```

### Test Files

1. **`test_agent_service_integration.py`** (6 tests)
   - Test RPyC connections to real agents
   - Test message callbacks
   - Test multiple agent discovery
   - Test reconnection after disconnect

2. **`test_api_agents_integration.py`** (8 tests)
   - Test REST API endpoints with real agents
   - GET /api/agents, GET /api/agents/{name}
   - POST /api/agents/{name}/message
   - GET /api/agents/{name}/history
   - POST /api/agents/{name}/stop_task_manager
   - GET /api/agents/{name}/status

3. **`test_websocket_integration.py`** (7 tests, 1 skipped)
   - Test WebSocket connections
   - Test ping/pong
   - Test message broadcasting
   - Test callback registration/unregistration
   - **Note**: Multiple concurrent WebSocket connections are skipped due to TestClient limitations

## Important Notes

### Agent Fixture Configuration
- **skip_permissions=True**: Required for tests (avoids hanging on approval prompts)
- **rpyc_port=True**: Auto-assigns ports (prevents port conflicts)
- **model/provider**: Default is deepseek-chat/deepseek (no API key needed for basic tests)

### API Key Requirements
Most integration tests don't require API keys because they test infrastructure (connections, routing) rather than AI responses. Tests that send actual messages to agents may require API keys configured in `~/.deepagents/.env`.

### Test Performance
Integration tests are slower than unit tests because they:
- Start real agent processes (~1-2 seconds per agent)
- Establish RPyC connections
- Wait for agent initialization

Typical runtime: **2-4 minutes** for all 21 integration tests (6 + 8 + 7).

### Debugging Failed Tests
When a test fails, the agent data directory is preserved:
```
Preserving failed test agent data: /tmp/deep_worker_test_test_agent_XXX_YYY
```

You can inspect:
- Agent logs
- Working directory
- PID files

### Known Limitations
- **Multiple WebSocket connections**: TestClient doesn't support multiple concurrent WebSocket connections well, so `test_websocket_multiple_connections` is skipped
- **Agent startup time**: Each agent takes ~1-2 seconds to start
- **Port availability**: Tests use auto-assigned ports but require free port range

## Adding New Integration Tests

1. Import the fixture:
```python
from tests.integration.agent_fixture import agent_fixture
```

2. Create a test with real agent:
```python
@pytest.mark.integration
class TestMyFeature:
    def test_something(self) -> None:
        with agent_fixture() as agent_info:
            # Test with real agent
            assert agent_info.rpyc_port > 0
```

3. Run your test:
```bash
python3 -m pytest tests/integration/test_my_feature.py -v -m integration
```

## Troubleshooting

### Test hangs
- Check if agent process started (look for `[AGENT_INIT]` logs)
- Verify RPyC port is available
- Check ~/.deepagents/.env for API key (if test requires AI)

### Connection refused
- Agent may not have started in time (increase timeout in fixture)
- Port may be in use (fixture auto-assigns ports to avoid this)

### WebSocket disconnect
- Check endpoint path is `/ws/agents` (not `/ws`)
- Verify agent_service is in app.state
- TestClient has limitations with WebSocket testing

## Test Results Summary

Current status (all passing):
- ✅ **6/6** AgentService integration tests
- ✅ **8/8** API integration tests
- ✅ **6/7** WebSocket integration tests (1 skipped)

**Total: 20/21 passing** (1 skipped due to TestClient limitations)
