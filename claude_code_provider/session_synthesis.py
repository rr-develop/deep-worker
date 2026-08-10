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

import base64
import binascii
import io
import json
import re
import uuid as _uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

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


# ===========================================================================
# IMAGE CONTENT BLOCKS
#
# See docs/DESIGN_FIX_IMAGE_BLOCKS_IN_SYNTHESIZED_TRANSCRIPT.md.
#
# ⚠️ POLICY — FAIL CLOSED. This module writes a transcript that is replayed
# VERBATIM to the Anthropic Messages API on every ``--resume``. A single
# invalid block poisons the session permanently (production incident
# 2026-08-04: a leaked OpenAI-style ``image_url`` block
# made even ``2+2`` fail with a 400). Therefore an ``image`` block is emitted
# ONLY when it is PROVABLY valid; anything doubtful degrades to a diagnostic
# text block. Never raise: history restore is an optimization and must never
# break a turn.
#
# ⚠️ DO NOT "simplify" this by reusing ``chat_model._build_mcp_content``. That
# builds the MCP shape (``data``/``mimeType``) for the LIVE tool path. The
# transcript needs the Anthropic Messages API shape (``source``); ``mimeType``
# is not a valid key there and would 400 again.
# ===========================================================================

# Anthropic Vision docs: "The maximum size per image is 10 MB (base64-encoded)
# when using the Claude API directly."
# https://platform.claude.com/docs/en/build-with-claude/vision
#
# ⚠️ This bounds the BASE64-ENCODED payload, NOT the decoded bytes. Capping
# decoded bytes at 10e6 would allow ~13.33 MB encoded — over the limit, which
# is exactly the failure this module exists to prevent. The name says
# ``BASE64`` on purpose; a decoded-sounding name is how that confusion recurs.
# Decimal MB (not MiB): 10_000_000 < 10_485_760, so we are conservative under
# either reading of the docs.
_MAX_BASE64_IMAGE_BYTES = 10_000_000

# Only these are accepted by the Messages API (Vision docs, "Supported
# formats"). Anything else — SVG, BMP, application/octet-stream — is rejected.
_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)

# Anthropic max dimension is 8000x8000 px. Note 8000*8000 = 64e6 px is BELOW
# Pillow's DecompressionBombWarning threshold (MAX_IMAGE_PIXELS = 89_478_485),
# so this check alone already rejects the whole "warn but don't raise" band.
_MAX_IMAGE_DIMENSION = 8000

# Magic-number prefixes, to catch a payload whose declared media_type lies.
#
# ⚠️ WebP is NOT a simple prefix: the RIFF container is shared with WAVE, AVI
# and others, so a bare ``b"RIFF"`` check would accept ``RIFF....WAVE`` as an
# image. The real signature is ``RIFF`` at [0:4] AND ``WEBP`` at [8:12] (the
# 4 bytes in between are the chunk size). Handled by ``_magic_ok`` below.
_IMAGE_MAGIC: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


def _magic_ok(media_type: str, decoded: bytes) -> bool:
    """True iff ``decoded``'s magic number matches the declared media type."""
    if media_type == "image/webp":
        # RIFF container + explicit WEBP form type — see note above.
        return (
            len(decoded) >= 12
            and decoded[:4] == b"RIFF"
            and decoded[8:12] == b"WEBP"
        )
    prefixes = _IMAGE_MAGIC.get(media_type, ())
    if not prefixes:
        return False
    return any(decoded.startswith(p) for p in prefixes)

# API-supported ``tool_result`` block types. Exactly the set enumerated by the
# real 400 message. Non-image ones keep their PRE-EXISTING pass-through
# contract (``type`` checked only) — see design §5.0: this module validates
# IMAGE blocks, and does not claim to schema-validate the others.
_PASSTHROUGH_BLOCK_TYPES = frozenset(
    {"text", "document", "search_result", "tool_reference"}
)

# ``data:<media-type>[;<param>=<value>]*;base64,<data>``. The media type may
# carry parameters (e.g. ``;charset=utf-8``) before the ``;base64`` marker, so
# everything up to the LAST ``;base64,`` is the media-type field and is
# normalized by ``_normalize_media_type`` (which strips parameters).
_DATA_URL_RE = re.compile(r"^data:(?P<media>[^,]*?);base64,(?P<data>.*)$", re.S)


def _reject(reason: str, media_type: str, size: int) -> dict[str, Any]:
    """A diagnostic text block for an image we refuse to emit.

    ⚠️ NEVER interpolate the payload (or any prefix of it) — that would put
    image bytes back into the transcript as text and defeat the size guard.
    """
    return {
        "type": "text",
        "text": (
            f"[Image omitted from restored history: {reason}; "
            f"media_type={media_type or 'unknown'}, {size} bytes]"
        ),
    }


def _normalize_media_type(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.split(";")[0].strip().lower()


def _anthropic_image_block(media_type: str, data: Any) -> dict[str, Any]:
    """Build a validated Anthropic image block, or a diagnostic text block.

    Validation order is normative (design §5.1): the ENCODED-size check runs
    FIRST, before any decoding, so an oversized payload never reaches
    ``b64decode`` or Pillow.
    """
    if not isinstance(data, str) or not data:
        return _reject("empty image data", media_type, 0)

    if media_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
        return _reject("unsupported media type", media_type, len(data))

    # (a) EXACT encoded-size check — this IS the documented Anthropic limit,
    #     and it also bounds allocation: nothing oversized gets decoded.
    try:
        encoded = data.encode("ascii")
    except UnicodeEncodeError:
        return _reject("invalid base64", media_type, len(data))
    if len(encoded) > _MAX_BASE64_IMAGE_BYTES:
        return _reject("too large", media_type, len(encoded))

    # (b) strict decode
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return _reject("invalid base64", media_type, len(encoded))
    if not decoded:
        return _reject("empty image data", media_type, 0)

    # (c) magic-number agreement — a declared png that is really something
    #     else must not be emitted.
    if not _magic_ok(media_type, decoded):
        return _reject("media type mismatch", media_type, len(decoded))

    # (d) dimensions, header-only. Pillow only WARNS between MAX_IMAGE_PIXELS
    #     and 2x that, so the warning is promoted to an error here; our own
    #     8000x8000 rule is the authoritative guard. Belt and braces.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(decoded)) as img:
                width, height = img.size
    except Exception:  # noqa: BLE001 — any failure ⇒ reject, never raise
        return _reject("unreadable image", media_type, len(decoded))

    if width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION:
        return _reject("dimensions too large", media_type, len(decoded))

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def _transcript_block(block: Any) -> dict[str, Any]:
    """Map one LangChain content block to a transcript-safe block."""
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)}

    block_type = block.get("type")

    if block_type in _PASSTHROUGH_BLOCK_TYPES:
        return block

    if block_type == "image_url":
        raw = block.get("image_url")
        url = raw.get("url", "") if isinstance(raw, dict) else raw
        if not isinstance(url, str):
            return _reject("invalid base64", "", 0)
        match = _DATA_URL_RE.match(url)
        if match is None:
            # A remote URL is NOT converted to an Anthropic url-source: the
            # transcript is replayed later, so that would introduce a
            # fetch-at-replay-time failure mode and an egress concern.
            return {"type": "text", "text": f"[Image: {url}]"}
        return _anthropic_image_block(
            _normalize_media_type(match.group("media")), match.group("data")
        )

    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict):
            # Already-Anthropic. Re-validate rather than pass through:
            # unconditional idempotence would PRESERVE a poisoned block.
            if source.get("type") != "base64":
                return _reject("unsupported media type", "", 0)
            return _anthropic_image_block(
                _normalize_media_type(source.get("media_type")), source.get("data")
            )
        # MCP shape ({"data": ..., "mimeType": ...}) — normalize it.
        return _anthropic_image_block(
            _normalize_media_type(block.get("mimeType")), block.get("data")
        )

    return {"type": "text", "text": str(block)}


def _tool_result_content(content: Any) -> Any:
    """Shape a ToolMessage's content for a ``tool_result`` block.

    Claude accepts either a string or a list of content blocks. Strings — the
    overwhelmingly common case — pass through untouched. Lists are converted
    block-by-block so that multimodal tool results (``read_image``) become
    valid Anthropic content instead of poisoning the session.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = [_transcript_block(b) for b in content]
        # An empty content list is invalid for the API; degrade to "".
        return blocks or ""
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
