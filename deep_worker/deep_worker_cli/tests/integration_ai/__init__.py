"""AI-Powered Integration Testing Infrastructure for deep_worker_cli.

This module provides infrastructure for testing deep_worker agents using
AI-powered validation through claude_sdk meta-queries.

CRITICAL TESTING PRINCIPLES
===========================

1. AGENT SESSION ISOLATION
--------------------------
   - NEVER use production agent sessions in tests
   - ALWAYS use 'test_' prefix for test agent names
   - Example: test_agent_basic, test_agent_streaming

   WHY: Production agents may have important state, long-running tasks,
        or configurations that tests should not modify or terminate.

2. AGENT NAMING CONVENTION
---------------------------
   Pattern: test_<purpose>_<feature>

   Examples:
   - test_agent_basic      → Basic agent functionality
   - test_agent_streaming  → Streaming responses
   - test_agent_tools      → Tool usage
   - test_agent_memory     → Memory/state management

   WHY: Predictable names make cleanup easier and prevent accidents.

3. AGENT LIFECYCLE MANAGEMENT
------------------------------
   The agent_fixture handles three scenarios:

   a) Test creates agent + test PASSES:
      → Terminate and cleanup agent
      → Remove socket files
      → Clean temporary files

   b) Test creates agent + test FAILS:
      → PRESERVE agent for debugging
      → Keep socket for inspection
      → Log agent state

   c) Pre-existing test agent:
      → Reuse existing agent
      → NEVER terminate (user may be debugging)
      → Verify agent type matches expected

   WHY: Failed tests need preserved state for debugging, but passing
        tests should clean up to avoid resource leaks.

4. AI VALIDATION STRATEGY
--------------------------
   Use meta-queries to validate non-deterministic AI responses:

   Traditional (FRAGILE):
       assert "Hello" in response  # Breaks with variations

   AI-Powered (ROBUST):
       is_valid = await check_reply(query, response)
       assert is_valid  # Works with any correct format

   WHY: AI responses are inherently non-deterministic. String matching
        is brittle and maintenance-heavy.

5. COST OPTIMIZATION
--------------------
   - Default to 'haiku' model for validation (~$0.001 per check)
   - Use 'sonnet' only when haiku fails (~$0.003 per check)
   - Use 'opus' only for complex validation (~$0.015 per check)

   Example costs for 100 tests:
   - All haiku: ~$0.10 USD
   - Mixed haiku/sonnet: ~$0.20 USD
   - All opus: ~$1.50 USD

   WHY: Keep CI costs reasonable while maintaining test quality.

6. TYPE SAFETY
--------------
   All infrastructure has strict type checking:
   - Pydantic models for validation
   - Full type annotations
   - pyright strict mode
   - mypy compatibility

   WHY: Catch errors at development time, not in CI.

ARCHITECTURE
============

┌─────────────────────────────────────────────────────────┐
│                  Integration AI Tests                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌───────────────┐           ┌──────────────────┐      │
│  │  Test Files   │──────────>│ agent_fixture    │      │
│  │  test_*.py    │           │ - Start agent    │      │
│  └───────────────┘           │ - Verify type    │      │
│         │                     │ - Lifecycle mgmt │      │
│         │                     └──────────────────┘      │
│         │                              │                │
│         v                              v                │
│  ┌───────────────┐           ┌──────────────────┐      │
│  │ check_reply() │<──────────│  claude_sdk      │      │
│  │ - Meta-queries│           │  - query()       │      │
│  │ - Validation  │           │  - Options       │      │
│  └───────────────┘           └──────────────────┘      │
│         │                              │                │
│         v                              v                │
│  ┌───────────────┐           ┌──────────────────┐      │
│  │ Pydantic      │           │  Deep Worker     │      │
│  │ Models        │           │  - AgentProcess  │      │
│  │ - Type safety │           │  - AgentClient   │      │
│  └───────────────┘           │  - IPC Protocol  │      │
│                               └──────────────────┘      │
└─────────────────────────────────────────────────────────┘

USAGE EXAMPLES
==============

Basic Test with Agent Fixture:
-------------------------------

    @pytest.fixture
    def agent(request):
        '''Provide test agent with automatic cleanup.'''
        with agent_fixture(
            agent_name='test_agent_basic',
            working_dir='/tmp'
        ) as agent:
            yield agent

    @pytest.mark.asyncio
    async def test_agent_response(agent):
        '''Test agent responds correctly to query.'''
        query = "What is 2+2?"

        # Send query to agent
        client = AgentClient(agent.agent_name)
        await client.connect()
        response = await client.invoke_agent(query)
        await client.disconnect()

        # Validate response using AI
        is_valid, comment = await check_reply(query, response['response']['content'])
        assert is_valid, f"Invalid response: {comment}"

Streaming Response Test:
-------------------------

    @pytest.mark.asyncio
    async def test_streaming_response(agent):
        '''Test agent streaming responses.'''
        query = "Count from 1 to 5"

        client = AgentClient(agent.agent_name)
        await client.connect()

        chunks = []
        async for chunk in client.invoke_agent_stream(query):
            chunks.append(chunk)

        await client.disconnect()

        # Validate complete response
        full_response = ''.join(chunks)
        is_valid, _ = await check_reply(query, full_response)
        assert is_valid

RUNNING TESTS
=============

Run all AI integration tests:
    pytest tests/integration_ai/ -v

Run with logging:
    pytest tests/integration_ai/ -v -s --log-cli-level=DEBUG

Run specific test:
    pytest tests/integration_ai/test_agent_basic.py::test_simple_query -v

Run with cost tracking:
    pytest tests/integration_ai/ -v --log-cli-level=INFO

DEBUGGING FAILED TESTS
======================

When a test fails, the agent is preserved:

1. List running agents:
   deep-worker --list

2. Check agent status:
   deep-worker --status test_agent_basic

3. Connect to agent for inspection:
   dwcli test_agent_basic

4. View agent logs:
   cat /tmp/deep_worker_test_agent_basic.log

5. Terminate agent after debugging:
   deep-worker --terminate test_agent_basic

SECURITY CONSIDERATIONS
=======================

1. Test agents use /tmp for isolation
2. Socket permissions are user-only (600)
3. No production data in test agents
4. Claude API key required for validation
5. Validation uses bypassPermissions mode

"""

from .agent_fixture import agent_fixture, verify_agent_running
from .check_reply import batch_check_replies, check_reply, validate_test_reply
from .models import (
    AgentFixtureInfo,
    CheckReplyResult,
    MetaQueryRequest,
    ValidationError,
)

__all__ = [
    "agent_fixture",
    "verify_agent_running",
    "check_reply",
    "validate_test_reply",
    "batch_check_replies",
    "AgentFixtureInfo",
    "CheckReplyResult",
    "MetaQueryRequest",
    "ValidationError",
]
