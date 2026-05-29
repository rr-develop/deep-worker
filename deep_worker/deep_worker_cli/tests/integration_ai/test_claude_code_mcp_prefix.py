"""HONEST end-to-end regression test for the ``mcp__deepworker__`` prefix bug.

This test spawns a REAL ``claude-code`` agent via :func:`agent_fixture`
(no mocks, no patches). It then sends a single user message that
deliberately requires the agent to invoke MULTIPLE proxy MCP tools
(``ls`` + ``execute``) so that Claude emits several
``mcp__deepworker__*`` ``ToolUseBlock`` entries in one turn. This is the
exact production scenario that froze ``test_dev``.

Regression context
==================

Commit ``87c80b8 "Remove Claude Code provider polling"`` introduced a
membership check in :mod:`claude_code_provider.chat_model`::

    proxy_tool_names = {tool["name"] for tool in self._tools_metadata}
    ...
    if block.name in proxy_tool_names:

``ToolMetadata['name']`` stores the bare LangChain tool name (e.g.
``"ls"``), but Claude Agent SDK emits ``ToolUseBlock.name`` PREFIXED
with the MCP server name (e.g. ``"mcp__deepworker__ls"``). The
membership check therefore always returned False, every proxy tool was
misclassified as a built-in Claude tool, the MCP handler tasks stayed
blocked on their Futures, and the agent froze on the next user message
once RPyC inbound quarantine fired (see commit ``b813e4d``).

BEFORE the fix this test hangs/times out (the agent never returns a
final response when multiple MCP tools are required).
AFTER the fix the agent answers normally within the timeout.

Safety
======

All operations the agent is asked to perform are SAFE and READ-ONLY at
the filesystem-content level:

- ``ls /tmp``        — directory listing, no mutation
- ``execute echo …`` — pure stdout, no side effects, no network
- ``read_file <fixture>`` — reads a temp fixture this test created

No network, no writes outside the test's own temporary fixture, no
package installation, no destructive shell commands.

⚠️ CRITICAL: PORT USAGE IN INTEGRATION TESTS ⚠️
================================================
These tests use ``agent_fixture`` which creates REAL agents with RPyC
servers. NEVER pass fixed ``rpyc_port`` values — always rely on
automatic port assignment.

Run with::

    pytest deep_worker/deep_worker_cli/tests/integration_ai/test_claude_code_mcp_prefix.py -v -s
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient

from . import agent_fixture


# Per-test-run safe fixture file under /tmp.
_FIXTURE_DIR = Path("/tmp") / f"dw_mcp_prefix_fixture_{uuid.uuid4().hex[:8]}"
_FIXTURE_FILE = _FIXTURE_DIR / "marker.txt"
_FIXTURE_TOKEN = "MCP_PREFIX_FIX_OK_TOKEN_4242"


@pytest.fixture(scope="module")
def safe_fixture_dir() -> Path:
    """Create a private /tmp subdir with one marker file the agent can read.

    The contents include a unique token the agent is asked to echo back,
    proving that the ``read_file`` MCP tool actually ran and returned
    its result to the LangGraph loop (i.e. the prefix bug is fixed).
    """
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _FIXTURE_FILE.write_text(
        f"This is a safe test fixture.\nTOKEN={_FIXTURE_TOKEN}\n",
        encoding="utf-8",
    )
    try:
        yield _FIXTURE_DIR
    finally:
        # Best-effort cleanup; never fail the test on cleanup errors.
        try:
            if _FIXTURE_FILE.exists():
                _FIXTURE_FILE.unlink()
            if _FIXTURE_DIR.exists():
                _FIXTURE_DIR.rmdir()
        except OSError:
            pass


@pytest.fixture
def claude_code_mcp_agent(safe_fixture_dir: Path):
    """Spin up a REAL claude-code agent — no mocks.

    Working dir is the safe fixture dir so the agent's relative-path
    operations (if any) stay scoped to a throw-away location.
    """
    with agent_fixture(
        agent_name="test_claude_code_mcp_prefix",
        working_dir=safe_fixture_dir,
        model="sonnet",
        provider="claude-code",
        skip_permissions=True,
        api_key=None,
    ) as agent:
        yield agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_mcp_tools_in_one_turn_do_not_hang(
    claude_code_mcp_agent,
    safe_fixture_dir: Path,
) -> None:
    """Regression: agent must NOT hang when one turn needs many MCP tools.

    This is the honest e2e reproducer of the ``test_dev`` freeze. The
    user asks the agent to perform three tool operations in one go —
    a directory listing, a shell echo, and a file read. Claude will
    emit prefixed ``mcp__deepworker__*`` ``ToolUseBlock`` entries.
    Before the fix, the LangGraph loop never saw those tool_calls and
    ``receive_response()`` blocked forever; after the fix the agent
    completes its turn and replies with the fixture token.
    """
    client = AgentClient(claude_code_mcp_agent.agent_name)
    await client.connect()

    try:
        # Construct a prompt that strongly biases the agent toward
        # invoking several proxy MCP tools in a single assistant turn.
        # We name the exact tools and the exact paths to minimise
        # plan-and-confirm overhead and to maximise the chance of a
        # batched tool-use message.
        prompt = (
            "Please perform ALL THREE of the following actions in this "
            "single turn, then summarise the results in one short reply:\n"
            "1. Use the `ls` tool on the path `/tmp` (just to list it).\n"
            "2. Use the `execute` tool to run the command "
            "`echo hello-from-mcp-prefix-test` and report stdout.\n"
            f"3. Use the `read_file` tool to read the file "
            f"`{_FIXTURE_FILE}` and quote the TOKEN= line verbatim in "
            "your final reply.\n"
            "Do not ask for confirmation; just do all three and answer."
        )

        # Generous timeout so a slow LLM still passes when the fix is in.
        # Before the fix the agent freezes and this WILL time out.
        response = await asyncio.wait_for(
            client.invoke_agent(prompt, stream=False),
            timeout=180.0,
        )

        content = response.get("response", {}).get("content", "")
        assert content, "Agent must return a non-empty reply (no freeze)."

        # Strong proof that the read_file MCP tool result actually flowed
        # back through LangGraph into the assistant's final answer.
        assert _FIXTURE_TOKEN in content, (
            "Agent reply must contain the fixture TOKEN it was asked "
            "to quote, which means the read_file MCP tool's result was "
            "delivered back through the LangGraph loop. If this fails, "
            "the prefix bug is still present and tool_calls were lost.\n"
            f"Got reply: {content[:500]!r}"
        )

        # Soft proof that the echo also executed (helps diagnose
        # regressions where only some MCP tools are surfaced).
        assert "hello-from-mcp-prefix-test" in content, (
            "Agent reply should include the echo output, proving the "
            "execute MCP tool ran and its result reached LangGraph.\n"
            f"Got reply: {content[:500]!r}"
        )

    finally:
        await client.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_followup_turn_after_mcp_tools_still_works(
    claude_code_mcp_agent,
) -> None:
    """Regression: after a multi-MCP-tool turn, the NEXT turn must work.

    The original ``test_dev`` symptom was that the agent looked fine
    for the first reply but then froze on the next user message —
    because some MCP handlers from the previous turn were still
    blocked on un-resolved Futures, leaving Claude's session in a
    half-state. After the fix, a follow-up question must complete.
    """
    client = AgentClient(claude_code_mcp_agent.agent_name)
    await client.connect()

    try:
        # Turn 1: trigger multiple MCP tools (same shape as the
        # previous test, kept self-contained so the two tests can run
        # in any order or in isolation).
        prompt_1 = (
            "Use the `ls` tool on `/tmp` AND the `execute` tool to run "
            "`echo turn1-ok`. Then briefly say done."
        )
        await asyncio.wait_for(
            client.invoke_agent(prompt_1, stream=False),
            timeout=180.0,
        )

        # Turn 2: a trivial follow-up. Before the fix this hangs
        # forever because the previous turn left MCP handler tasks
        # blocked and Claude's session out of sync.
        prompt_2 = "What is 7 + 5? Just the number."
        response_2 = await asyncio.wait_for(
            client.invoke_agent(prompt_2, stream=False),
            timeout=120.0,
        )
        content_2 = response_2.get("response", {}).get("content", "")
        assert content_2, "Follow-up turn must not hang and must reply."
        assert "12" in content_2, (
            "Follow-up arithmetic answer should contain '12'. "
            f"Got: {content_2[:200]!r}"
        )

    finally:
        await client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
