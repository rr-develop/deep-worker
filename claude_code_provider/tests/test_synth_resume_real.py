"""REAL, no-mock integration test for Approach B (synthesized-transcript resume).

This is the test that SHOULD have existed when the restore feature claimed
"E2: synthesized transcript resumes cleanly ✅". It does NOT mock the CLI: it
synthesizes a transcript with ``session_synthesis.synthesize_session_file``,
then resumes it with a real ``ClaudeSDKClient`` and asserts the model recalls a
secret codeword that exists ONLY in the synthesized history.

Failure modes it catches (all observed on a live agent, 2026-05-25):
  * CLI hangs at ``initialize`` (``Control request timeout: initialize``) when
    the synthesized assistant entries are missing fields the CLI needs to load
    them (id/model/type/usage/stop_reason).
  * ``No conversation found`` slug mismatch.

Skipped when the Claude CLI / credentials are unavailable, so it never breaks
a machine that can't reach the real CLI — but on a logged-in host it is the
ground truth for "does synthesized resume actually work".
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import uuid as _uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from claude_code_provider.session_synthesis import synthesize_session_file

_CLAUDE = shutil.which("claude")
_CREDS = (Path.home() / ".claude" / ".credentials.json").exists()

# ``real_cli`` opts these tests OUT of conftest's autouse HOME-isolation
# fixture: the CLI subprocess reads the REAL ~/.claude, so the synthesized file
# (written via Path.home()) must land in that same real home — not a tmp dir.
pytestmark = [
    pytest.mark.real_cli,
    pytest.mark.skipif(
        _CLAUDE is None or not _CREDS,
        reason="real Claude CLI + credentials required (no-mock integration test)",
    ),
]


async def _resume_and_ask(session_id: str, cwd: str, question: str) -> str:
    """Resume ``session_id`` with a real ClaudeSDKClient and return the reply
    text for ``question``. Raises on connect/initialize failure (which is
    exactly what we want the test to catch)."""
    # Import inside the test so collection never needs the SDK present.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "claude_agent_sdk"))
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    client = ClaudeSDKClient(
        options=ClaudeAgentOptions(
            resume=session_id,
            fork_session=True,
            cwd=cwd,
            permission_mode="bypassPermissions",
        )
    )
    # 75s > the SDK's 60s initialize timeout, so a hang surfaces as the real
    # "Control request timeout: initialize" rather than our own asyncio cancel.
    await asyncio.wait_for(client.connect(), timeout=80)
    reply = ""
    try:
        await client.query(question)
        async for event in client.receive_response():
            for block in getattr(event, "content", []) or []:
                reply += getattr(block, "text", "") or ""
            if type(event).__name__ == "ResultMessage":
                break
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
    return reply


@pytest.mark.asyncio
async def test_synthesized_transcript_resumes_and_recalls() -> None:
    """Synthesize a transcript carrying a unique secret, resume it for real,
    and assert the model recalls the secret. This is the end-to-end proof that
    Approach B (synthesis) actually works against the live CLI — not just that
    the .jsonl has the right fields."""
    # A token that exists ONLY in the synthesized history — if the model knows
    # it after resume, the synthesized transcript was genuinely loaded as
    # context (not re-derived, not hallucinated).
    secret = f"QUOKKA_{_uuid.uuid4().hex[:8].upper()}"
    # Use a real, stable working dir so the project-dir slug is well-formed.
    cwd = "/tmp"

    messages: list[BaseMessage] = [
        HumanMessage(content=f"Please remember this exact codeword: {secret}"),
        AIMessage(content=f"Got it. The codeword is {secret}. I will remember it."),
    ]
    session_id = synthesize_session_file(messages, cwd=cwd)
    assert session_id is not None, "synthesis produced no file"

    reply = await _resume_and_ask(
        session_id,
        cwd,
        "What exact codeword did I ask you to remember earlier? "
        "Reply with only the codeword.",
    )

    assert secret in reply, (
        "synthesized transcript did not resume as usable history — the model "
        f"did not recall the secret {secret!r}. Got reply: {reply!r}"
    )


@pytest.mark.asyncio
async def test_synthesized_transcript_with_tool_calls_resumes() -> None:
    """The realistic shape: history containing tool_use + tool_result (what a
    real agent's history looks like). Resuming this must also connect and
    recall a fact from a tool result."""
    from langchain_core.messages import ToolMessage

    secret_port = f"{_uuid.uuid4().int % 9000 + 1000}"
    cwd = "/tmp"
    messages: list[BaseMessage] = [
        HumanMessage(content="Read the server config and tell me the port."),
        AIMessage(
            content="Reading the config now.",
            tool_calls=[{
                "id": "toolu_cfg1",
                "name": "mcp__deepworker__read_file",
                "args": {"path": "/etc/server.conf"},
            }],
        ),
        ToolMessage(content=f"listen_port={secret_port}", tool_call_id="toolu_cfg1"),
        AIMessage(content=f"The server listens on port {secret_port}."),
    ]
    session_id = synthesize_session_file(messages, cwd=cwd)
    assert session_id is not None

    reply = await _resume_and_ask(
        session_id,
        cwd,
        "What port did the server config say it listens on? Reply with only the number.",
    )
    assert secret_port in reply, (
        "synthesized transcript with tool_use/tool_result did not resume as "
        f"usable history — model did not recall port {secret_port}. Got: {reply!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
