"""REAL, no-mock integration test for Approach B (synthesized-transcript resume).

This is the test that SHOULD have existed when the restore feature claimed
"E2: synthesized transcript resumes cleanly ✅". It does NOT mock the CLI: it
synthesizes a transcript with ``session_synthesis.synthesize_session_file``,
then resumes it with a real ``ClaudeSDKClient`` and asserts the model recalls a
secret codeword that exists ONLY in the synthesized history.

Failure modes it catches (all observed 2026-05-25):
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

from deep_worker.providers.claude_code_provider.session_synthesis import synthesize_session_file

_CLAUDE = shutil.which("claude")
_CREDS = (Path.home() / ".claude" / ".credentials.json").exists()

# ``real_cli`` opts these tests OUT of conftest's autouse HOME-isolation
# fixture: the CLI subprocess reads the REAL user-level Claude config dir, so
# the synthesized file (written via Path.home()) must land in that same real
# home — not a tmp dir.
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "claude_sdk" / "src"))
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
    cwd = "/tmp/project"

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
    cwd = "/tmp/project"
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


# ===========================================================================
# IMAGE BLOCKS IN A SYNTHESIZED TRANSCRIPT (no-mock, real CLI)
#
# The authoritative proof for
# docs/DESIGN_FIX_IMAGE_BLOCKS_IN_SYNTHESIZED_TRANSCRIPT.md (test #24/#25).
#
# The defect (production, observed 2026-08-04): read_image returns
# OpenAI-style ``image_url`` blocks and session_synthesis wrote them into the
# transcript verbatim, so every resume replayed an invalid block:
#   400 messages.N.content.0.tool_result.content.1: Input tag 'image_url'
#   found using 'type' does not match any of the expected tags: ...
# which poisoned the session permanently.
#
# ⚠️ Why this test lives HERE and not in the unit file: a unit test can only
# prove the .jsonl has the right SHAPE. Only a real resume proves the API
# ACCEPTS it. This module's own history is the reason — a manual "E2 works ✅"
# claim once stood in for a real test and was wrong.
# ===========================================================================

_IMAGE_FACT_COLOR = "purple"


def _known_png_data_url() -> tuple[str, str]:
    """A PNG whose content cannot be guessed from context.

    Returns (data_url, codeword_drawn_in_the_image).
    """
    import base64
    import io

    from PIL import Image, ImageDraw

    codeword = f"ZEBRA{_uuid.uuid4().int % 9000 + 1000}"
    img = Image.new("RGB", (480, 240), (96, 32, 160))  # purple
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
        )
    except Exception:  # noqa: BLE001 — bitmap default font is fine
        font = None
    draw.text((40, 90), codeword, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", codeword


def _image_history(data_url: str) -> list[BaseMessage]:
    """History whose ToolMessage carries read_image's LITERAL output shape."""
    from langchain_core.messages import ToolMessage

    return [
        HumanMessage(content="Look at this screenshot and remember it."),
        AIMessage(
            content="Reading the image.",
            tool_calls=[{
                "id": "toolu_img1",
                "name": "mcp__deepworker__read_image",
                "args": {"file_path": "/tmp/shot.png"},
            }],
        ),
        ToolMessage(
            # EXACTLY what deep_worker_cli.tools.read_image returns.
            content=[
                {"type": "text", "text": "Image: shot.png (480x240), image/png"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
            tool_call_id="toolu_img1",
        ),
    ]


@pytest.mark.asyncio
async def test_synthesized_transcript_with_image_resumes_and_model_sees_it() -> None:
    """#24 — authoritative proof of the fix, end to end, no mocks.

    Asserts BOTH that the API accepts the synthesized transcript (no 400) and
    that the image was genuinely delivered — the model reads a codeword that
    exists ONLY inside the picture, so a dropped/degraded image cannot pass.
    """
    data_url, codeword = _known_png_data_url()
    cwd = "/tmp/project"

    session_id = synthesize_session_file(_image_history(data_url), cwd=cwd)
    assert session_id is not None, "synthesis produced no file"

    # Shape guard first: cheap, and pinpoints the failure if the API rejects.
    from deep_worker.providers.claude_code_provider.session_synthesis import session_project_dir

    transcript = session_project_dir(cwd) / f"{session_id}.jsonl"
    raw = transcript.read_text()
    assert "image_url" not in raw, (
        "OpenAI-style image_url leaked into the synthesized transcript — this "
        "is the exact payload that produced the production 400"
    )

    reply = await asyncio.wait_for(
        _resume_and_ask(
            session_id,
            cwd,
            "In the image from our earlier conversation, what codeword is "
            "written, and what is the background colour? Reply with just the "
            "codeword and the colour.",
        ),
        timeout=180,
    )

    # An API 400 surfaces as an exception from _resume_and_ask; reaching here
    # already proves the transcript was accepted. Now prove the model SAW it.
    assert codeword in reply.upper(), (
        f"model did not read the codeword {codeword!r} from the restored "
        f"image — the image block did not survive synthesis. Reply: {reply!r}"
    )
    assert _IMAGE_FACT_COLOR in reply.lower(), (
        f"model did not report the image's background colour. Reply: {reply!r}"
    )


@pytest.mark.asyncio
async def test_verbatim_image_url_is_rejected_by_the_api() -> None:
    """#25 — mutation RED: proves the test above covers the real defect.

    Writes the PRE-FIX transcript (verbatim ``image_url``) and asserts the API
    rejects it with the verbatim production error. If this ever stops failing,
    the fix is no longer load-bearing and #24 proves nothing.
    """
    import json

    from deep_worker.providers.claude_code_provider.session_synthesis import session_project_dir

    data_url, _ = _known_png_data_url()
    cwd = "/tmp/project"

    session_id = synthesize_session_file(_image_history(data_url), cwd=cwd)
    assert session_id is not None
    transcript = session_project_dir(cwd) / f"{session_id}.jsonl"

    # Re-poison the transcript exactly the way the pre-fix code did.
    lines: list[str] = []
    for line in transcript.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block["content"] = [
                        {"type": "text", "text": "Image: shot.png"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
        lines.append(json.dumps(entry))
    transcript.write_text("\n".join(lines) + "\n")

    # ⚠️ The SDK does NOT raise on an API 400 here — verified against the real
    # CLI: the error arrives as assistant TEXT and as ``ResultMessage.result``
    # with ``is_error=True``. Asserting ``pytest.raises`` would make this test
    # fail for the wrong reason and mask the defect it exists to pin.
    reply = await asyncio.wait_for(
        _resume_and_ask(session_id, cwd, "What did you see?"),
        timeout=180,
    )

    assert "image_url" in reply, (
        "expected the verbatim production 400 about tag 'image_url'; the "
        f"mutation did not reproduce the defect. Got: {reply!r}"
    )
    assert "does not match any of the expected tags" in reply, (
        f"error text did not match the production failure. Got: {reply!r}"
    )
