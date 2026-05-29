"""Synthesize a Claude CLI session transcript (``.jsonl``) from a LangGraph
message history.

Approach B of ``docs/DESIGN_HISTORY_RESTORE_ON_RESTART.md``: when there is no
existing Claude session to resume (notably cross-provider migration — the
agent ran under deepseek/codex and we now continue it under claude-code), we
build a valid transcript file from the LangGraph history. The Claude CLI then
``resume``s it and treats it as prior context — WITHOUT re-executing any tool.

⚠️ Two things the CLI REQUIRES for a synthesized transcript to actually resume
(both verified by a REAL no-mock integration test,
``tests/test_synth_resume_real.py`` — NOT a manual spot-check; an earlier
"E2 ✅" claim here was false and synth-resume failed 100% in production,
observed 2026-05-25):

  1. The transcript MUST begin with the ``queue-operation`` (enqueue →
     dequeue) meta entries the CLI writes itself. A file starting straight
     with a ``user`` entry is rejected: "No conversation found with session
     ID" — even in the correct project-dir slug.
  2. Each ASSISTANT entry's ``message`` MUST carry the full CLI shape
     (``type:"message"``, ``id``, ``model``, ``usage``, ``stop_reason``,
     ``stop_sequence``), not just ``{role, content}``. Otherwise the CLI hangs
     loading it and ``initialize`` times out.

Transcript schema captured from a real ``~/.claude/projects/<slug>/<uuid>.jsonl``
(CLI 2.1.150). This mirrors a CLI-INTERNAL file format that is NOT a public
contract and may drift across CLI versions. The caller MUST fall back to the
normal (empty-session) path if a resume of the synthesized file fails — never
crash. See the design doc.
"""

from __future__ import annotations

import json
import re
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# CLI version we validated the schema against. Persisted into each entry's
# ``version`` field; also a breadcrumb for drift detection.
_SYNTH_CLI_VERSION = "2.1.150"


def _now_z() -> str:
    """ISO-8601 timestamp with a trailing ``Z`` (the CLI transcript format)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Mirror of the Claude CLI / SDK project-dir slug rule
# (claude_agent_sdk._internal.sessions._sanitize_path): every non-alphanumeric
# character becomes ``-``. We re-derive it here rather than importing the
# SDK-private helper (pyright-strict forbids private cross-module use). For
# paths longer than 200 chars the SDK appends a hash suffix — our agent cwds
# (short working dirs) are far shorter, so the simple rule matches;
# a length guard documents the boundary. Keep in sync if the SDK rule changes.
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")
_MAX_SLUG_LEN = 200


def _sanitize_cwd(cwd: str) -> str:
    slug = _SANITIZE_RE.sub("-", cwd)
    if len(slug) > _MAX_SLUG_LEN:
        # Beyond this the CLI uses a hash suffix we don't replicate; resume
        # would miss. Callers should avoid such long cwds (none of ours hit
        # it). Truncate defensively; the tail-match still gates correctness.
        slug = slug[:_MAX_SLUG_LEN]
    return slug


def session_project_dir(cwd: str) -> Path:
    """The ``~/.claude/projects/<sanitized-cwd>`` directory the CLI uses for a
    given working directory — where a synthesized file must land so
    ``--resume`` finds it."""
    return Path.home() / ".claude" / "projects" / _sanitize_cwd(cwd)


def _tool_result_content(content: Any) -> Any:
    """Shape a ToolMessage's content for a ``tool_result`` block.

    Claude accepts either a string or a list of content blocks. LangGraph
    ToolMessages are usually strings; pass strings through, and stringify
    anything exotic so the transcript stays valid JSON the CLI can parse.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content
    return str(content)


def synthesize_session_file(
    messages: list[BaseMessage],
    cwd: str,
    *,
    git_branch: str = "HEAD",
    model: str = "claude-opus-4-7",
) -> str | None:
    """Write a Claude transcript ``.jsonl`` for ``messages`` and return its
    new session id (a fresh uuid4), or ``None`` if there is nothing
    restorable to write.

    Args:
        messages: the LangGraph history (HumanMessage / AIMessage /
            ToolMessage / SystemMessage). SystemMessages are skipped (they are
            forwarded via ``ClaudeAgentOptions.system_prompt``, not as
            transcript entries).
        cwd: the working directory the resumed ``ClaudeSDKClient`` will run
            with — determines the project-dir slug. MUST match, or the CLI
            won't find the file on resume.
        git_branch: value for the entry ``gitBranch`` field (cosmetic).
        model: model id stamped into each synthesized assistant entry's
            ``message.model``. The CLI reads this (and ``id``/``type``/
            ``usage``/``stop_reason``) when LOADING the transcript on resume;
            a missing/empty value makes it hang at ``initialize`` (observed
            2026-05-25). Cosmetic for the conversation itself — any valid
            Claude model id works — but it MUST be present.

    Returns:
        The synthesized session uuid, or ``None`` when no user/assistant/tool
        message is present (nothing to restore).

    ⚠️ The assistant entry below mirrors the FULL ``message`` shape a real CLI
    transcript writes (``type``/``id``/``model``/``role``/``content``/
    ``stop_reason``/``stop_sequence``/``usage``). The CLI parses these on
    resume; ``{role, content}`` alone is NOT loadable and the resume hangs.
    Reference captured from a real ~/.claude/projects/.../<uuid>.jsonl,
    CLI 2.1.150 (see tests/test_session_synthesis.py +
    tests/test_synth_resume_real.py).
    """
    session_id = str(_uuid.uuid4())
    entries: list[dict[str, Any]] = []
    parent: str | None = None

    def _base() -> dict[str, Any]:
        return {
            "isSidechain": False,
            "cwd": cwd,
            "sessionId": session_id,
            "version": _SYNTH_CLI_VERSION,
            "gitBranch": git_branch,
            "userType": "external",
            "entrypoint": "sdk-py",
            "timestamp": _now_z(),
        }

    def _add(entry: dict[str, Any]) -> None:
        nonlocal parent
        u = str(_uuid.uuid4())
        entry["uuid"] = u
        entry["parentUuid"] = parent
        entries.append(entry)
        parent = u

    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Forwarded via system_prompt; never a transcript entry.
            continue

        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            _add({
                **_base(),
                "type": "user",
                "promptId": str(_uuid.uuid4()),
                "permissionMode": "bypassPermissions",
                "message": {"role": "user", "content": content},
            })

        elif isinstance(msg, AIMessage):
            blocks: list[dict[str, Any]] = []
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in (msg.tool_calls or []):
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "input": tc.get("args", {}),
                })
            # An assistant turn with neither text nor tool_use carries no
            # information — skip it (an empty content list is also invalid for
            # the API on a later real turn).
            if not blocks:
                continue
            # ``stop_reason`` mirrors a real turn: ``tool_use`` when the turn
            # ends on a tool call, else ``end_turn``.
            has_tool_use = any(b.get("type") == "tool_use" for b in blocks)
            _add({
                **_base(),
                "type": "assistant",
                "requestId": "req_synth",
                # Full assistant ``message`` schema — see the function
                # docstring. The CLI needs type/id/model/usage/stop_reason to
                # load this entry on resume.
                "message": {
                    "id": f"msg_{_uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": blocks,
                    "stop_reason": "tool_use" if has_tool_use else "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            })

        elif isinstance(msg, ToolMessage):
            _add({
                **_base(),
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": _tool_result_content(msg.content),
                    }],
                },
            })

    if not entries:
        return None

    # ⚠️ The CLI's ``--resume`` only RECOGNIZES a session whose transcript
    # begins with the ``queue-operation`` meta entries it writes itself
    # (enqueue → dequeue). A file that starts straight with a ``user`` entry is
    # rejected with "No conversation found with session ID" — even though it
    # sits in the right project-dir slug and the SDK's get_session_messages can
    # read it. Verified live (CLI 2.1.150): prepending these two lines is what
    # makes a synthesized transcript actually resumable (observed 2026-05-25;
    # tests/test_synth_resume_real.py). These two lines are NOT
    # part of the parentUuid chain (they carry no uuid/parentUuid), matching
    # the real on-disk layout.
    first_user_content = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            first_user_content = (
                m.content if isinstance(m.content, str) else str(m.content)
            )
            break
    ts = _now_z()
    prefix = [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": ts,
            "sessionId": session_id,
            "content": first_user_content,
        },
        {
            "type": "queue-operation",
            "operation": "dequeue",
            "timestamp": ts,
            "sessionId": session_id,
        },
    ]

    proj = session_project_dir(cwd)
    proj.mkdir(parents=True, exist_ok=True)
    sf = proj / f"{session_id}.jsonl"
    sf.write_text("\n".join(json.dumps(e) for e in prefix + entries) + "\n")
    return session_id
