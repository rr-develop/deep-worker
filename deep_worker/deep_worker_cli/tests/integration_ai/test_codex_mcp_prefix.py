"""HONEST end-to-end integration tests for the codex provider.

These tests spawn a REAL ``codex`` agent via :func:`agent_fixture`
(no mocks, no patches). They exercise the same scenarios used for
``claude_code_sdk_provider`` in
``test_claude_code_mcp_prefix.py``, so we can compare behaviour
across providers and catch any provider-specific MCP-bridge
regressions analogous to the ``mcp__deepworker__`` / sequential
dispatch bugs fixed for ``claude_code_sdk_provider``.

Architectural difference
========================
``codex_provider`` exposes deep-worker tools via an **in-agent-process
HTTP MCP server** (``codex_provider/proxy_mcp_server.py``), not via
the Claude Agent SDK's ``create_sdk_mcp_server``. Each tool call from
Codex arrives as an independent ``ItemStartedEvent(McpToolCallItem)``
on the JSONL stream, and ``CodexChatModel.ainvoke()`` already does
``wait_for_pending_calls(tool_name=…, arguments=…)`` for that single
item before returning to LangGraph. So the seq/parallel-dispatch
question is settled at the Codex protocol layer: one tool item per
event. These tests verify that this contract holds end-to-end against
the real ``codex`` binary, and that no analogous freeze can occur.

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

    pytest deep_worker/deep_worker_cli/tests/integration_ai/test_codex_mcp_prefix.py -v -s -m integration
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from deep_worker_cli.client.agent_client import AgentClient

from . import agent_fixture


# Per-test-run safe fixture file under /tmp.
_FIXTURE_DIR = Path("/tmp") / f"dw_codex_mcp_fixture_{uuid.uuid4().hex[:8]}"
_FIXTURE_FILE = _FIXTURE_DIR / "marker.txt"
_FIXTURE_TOKEN = "CODEX_MCP_FIX_OK_TOKEN_4242"


@pytest.fixture(scope="module")
def safe_fixture_dir() -> Path:
    """Create a private /tmp subdir with one marker file the agent can read.

    The contents include a unique token the agent is asked to echo back,
    proving that the ``read_file`` MCP tool actually ran and returned
    its result to the LangGraph loop.
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
def codex_mcp_agent(safe_fixture_dir: Path):
    """Spin up a REAL codex agent — no mocks.

    Codex CLI uses its own cached login at ``~/.codex/auth.json``, so
    no API key is required. The working_dir is the safe fixture dir
    so the agent's relative-path operations (if any) stay scoped to
    a throw-away location.
    """
    with agent_fixture(
        agent_name="test_codex_mcp_prefix",
        working_dir=safe_fixture_dir,
        # Codex provider does not need an explicit model name; passing
        # None makes CodexExecRunner skip the ``--model`` argv flag, so
        # codex CLI uses its default ChatGPT-bound model.
        # (Passing "unknown"/"gpt-5"/etc. makes the OpenAI backend
        # return ``invalid_request_error`` for ChatGPT-account logins.)
        model=None,
        provider="codex",
        skip_permissions=True,
        api_key=None,
    ) as agent:
        yield agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_mcp_tools_in_one_turn_do_not_hang(
    codex_mcp_agent,
    safe_fixture_dir: Path,
) -> None:
    """Regression-shape: agent must NOT hang when one turn needs many MCP tools.

    Mirrors ``test_claude_code_mcp_prefix.py::test_multiple_mcp_tools_in_one_turn_do_not_hang``
    against the codex provider. The user asks the agent to perform three
    tool operations in one go — a directory listing, a shell echo, and a
    file read. The Codex JSONL stream emits one ``McpToolCallItem`` per
    tool, each handled by ``CodexChatModel.ainvoke`` event-driven via
    ``wait_for_pending_calls``. The test fails (timeout) if any of the
    tool calls is lost or the bridge dead-locks.
    """
    client = AgentClient(codex_mcp_agent.agent_name)
    await client.connect()

    try:
        prompt = (
            "Please perform ALL THREE of the following actions in this "
            "single turn, then summarise the results in one short reply:\n"
            "1. Use the `ls` tool on the path `/tmp` (just to list it).\n"
            "2. Use the `execute` tool to run the command "
            "`echo hello-from-codex-mcp-test` and report stdout.\n"
            f"3. Use the `read_file` tool to read the file "
            f"`{_FIXTURE_FILE}` and quote the TOKEN= line verbatim in "
            "your final reply.\n"
            "Do not ask for confirmation; just do all three and answer."
        )

        # Generous timeout so a slow LLM still passes.
        response = await asyncio.wait_for(
            client.invoke_agent(prompt, stream=False),
            timeout=300.0,
        )

        content = response.get("response", {}).get("content", "")
        assert content, "Agent must return a non-empty reply (no freeze)."

        # Strong proof that the read_file MCP tool result actually flowed
        # back through LangGraph into the assistant's final answer.
        assert _FIXTURE_TOKEN in content, (
            "Agent reply must contain the fixture TOKEN it was asked "
            "to quote, which means the read_file MCP tool's result was "
            "delivered back through the LangGraph loop. If this fails, "
            "the MCP bridge dropped a tool result.\n"
            f"Got reply: {content[:500]!r}"
        )

        # Soft proof that the echo also executed (helps diagnose
        # regressions where only some MCP tools are surfaced).
        assert "hello-from-codex-mcp-test" in content, (
            "Agent reply should include the echo output, proving the "
            "execute MCP tool ran and its result reached LangGraph.\n"
            f"Got reply: {content[:500]!r}"
        )

    finally:
        await client.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_followup_turn_after_mcp_tools_still_works(
    codex_mcp_agent,
) -> None:
    """Regression-shape: after a multi-MCP-tool turn, the NEXT turn must work.

    Mirrors the claude_code test of the same name. The original
    ``test_dev`` symptom on claude_code was that the agent looked fine
    for the first reply but then froze on the next user message. We
    assert the same end-to-end behaviour on codex: turn 1 batched MCP
    tools, turn 2 a trivial follow-up arithmetic question. Both must
    complete.
    """
    client = AgentClient(codex_mcp_agent.agent_name)
    await client.connect()

    try:
        # Turn 1: trigger multiple MCP tools.
        prompt_1 = (
            "Use the `ls` tool on `/tmp` AND the `execute` tool to run "
            "`echo turn1-codex-ok`. Then briefly say done."
        )
        await asyncio.wait_for(
            client.invoke_agent(prompt_1, stream=False),
            timeout=300.0,
        )

        # Turn 2: a trivial follow-up.
        prompt_2 = "What is 7 + 5? Just the number."
        response_2 = await asyncio.wait_for(
            client.invoke_agent(prompt_2, stream=False),
            timeout=180.0,
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
