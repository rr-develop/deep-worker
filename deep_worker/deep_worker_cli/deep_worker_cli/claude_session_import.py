"""Import a Claude CLI session transcript (``.jsonl``) into a LangGraph history.

Moved out of ``claude_code_sdk_provider`` (DESIGN_IMPORT_CLAUDE_SESSION.md):
parsing a Claude Code CLI transcript into ``list[BaseMessage]`` has no
dependency on ``claude_agent_sdk`` or on any provider's runtime — it is a
pure reader of the CLI's on-disk ``.jsonl`` format, useful for ``--import-claude``
under ANY provider (not just claude-code-sdk/claude-code). It lived inside
``claude_code_sdk_provider`` only because of where it was first written, which
meant the whole flag broke in any build that excludes that provider
(FilterGit rule D8.1) even though nothing here actually needs it.

Given a real Claude Code session file — produced by Claude run the ordinary
way, not by deep_worker — parse it into ``list[BaseMessage]`` so it can become
a deep_worker agent's conversation history (the agent's *memory*; NOT
replayed).

Mapping (mirror of the writer in
``deep_worker.providers.claude_code_sdk_provider.session_synthesis``;
superset-tolerant of real transcript fields):

  ``type:"user"``, ``message.content`` = str           -> ``HumanMessage``
  ``type:"assistant"`` with text/tool_use blocks        -> ``AIMessage`` (+ tool_calls)
  ``type:"user"`` with ``tool_result`` block            -> ``ToolMessage``

Whitelist: ONLY ``type in {"user","assistant"}`` entries with a ``message``
object are imported. Every other type (``system``, ``attachment``,
``last-prompt``, ``permission-mode``, ``file-history-snapshot``,
``queue-operation``, ``ai-title``, unknown future types) and any malformed line
is ignored — never crash. Entries are walked in ``parentUuid`` chain order.
Sub-agent (``Task``) turns (``isSidechain: true``, normally in separate
``subagents/*.jsonl`` files) are skipped.

⚠️ Reads a CLI-internal file format that is not a public contract and may drift
across CLI versions; the whitelist + best-effort parsing keep this robust.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

# Mirrors ``claude_code_sdk_provider.session_synthesis``'s sanitize rule
# exactly (same CLI-internal slug format) — duplicated rather than imported
# so this module stays free of any dependency on that (excluded) package.
# Keep in sync if the CLI's slug rule ever changes.
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")
_MAX_SLUG_LEN = 200


def _sanitize_cwd(cwd: str) -> str:
    slug = _SANITIZE_RE.sub("-", cwd)
    if len(slug) > _MAX_SLUG_LEN:
        slug = slug[:_MAX_SLUG_LEN]
    return slug


def session_project_dir(cwd: str) -> Path:
    """The ``~/.claude/projects/<sanitized-cwd>`` directory the CLI uses for a
    given working directory — where ``--resume`` looks for a session file."""
    return Path.home() / ".claude" / "projects" / _sanitize_cwd(cwd)


def _placeholder_tool_result() -> str:
    return "<no result captured>"


def _block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else None


def _join_text_blocks(content: Any) -> str:
    """Collect text from a content list (or pass a string through)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p)


def _ordered_entries(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order entries by the ``parentUuid`` linked list.

    Falls back to file order if the chain is broken/missing (no single root, a
    cycle, or dangling parents) so a malformed transcript still imports.
    """
    by_uuid: dict[str, dict[str, Any]] = {}
    for e in raw_entries:
        u = e.get("uuid")
        if isinstance(u, str):
            by_uuid[u] = e

    # children[parent_uuid] -> list of child entries (None parent = roots)
    roots: list[dict[str, Any]] = []
    children: dict[str, list[dict[str, Any]]] = {}
    for e in raw_entries:
        parent = e.get("parentUuid")
        if parent is None:
            roots.append(e)
        elif isinstance(parent, str):
            children.setdefault(parent, []).append(e)

    if len(roots) != 1:
        return raw_entries  # ambiguous chain — trust file order

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: dict[str, Any] | None = roots[0]
    while cursor is not None:
        u = cursor.get("uuid")
        if not isinstance(u, str) or u in seen:
            break  # cycle / missing uuid — stop and fall back
        seen.add(u)
        ordered.append(cursor)
        kids = children.get(u, [])
        cursor = kids[0] if len(kids) == 1 else None

    # Only trust the walk if it covered every entry; otherwise file order.
    return ordered if len(ordered) == len(raw_entries) else raw_entries


def parse_claude_transcript(path: str | Path) -> list[BaseMessage]:
    """Parse a Claude session ``.jsonl`` into a LangGraph message history.

    Returns an empty list if the file has no importable user/assistant turns.
    Best-effort: malformed lines and non-conversation entry types are skipped.
    """
    p = Path(path)
    raw_lines = p.read_text(errors="replace").splitlines()

    entries: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            entries.append(obj)

    messages: list[BaseMessage] = []
    # Track tool_use ids we emitted so we can detect which ones never got a
    # matching tool_result (→ placeholder ToolMessage at the end).
    open_tool_ids: list[str] = []
    resolved_tool_ids: set[str] = set()

    for entry in _ordered_entries(entries):
        if entry.get("isSidechain") is True:
            continue
        etype = entry.get("type")
        if etype not in ("user", "assistant"):
            continue  # whitelist
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")

        if etype == "user":
            # tool_result (content is a list with a tool_result block) vs a
            # plain user turn.
            if isinstance(content, list) and any(
                _block_type(b) == "tool_result" for b in content
            ):
                for block in content:
                    if _block_type(block) != "tool_result":
                        continue
                    tool_use_id = str(block.get("tool_use_id", ""))
                    result = block.get("content", "")
                    if not isinstance(result, (str, list)):
                        result = str(result)
                    status = "error" if block.get("is_error") is True else None
                    tm_kwargs: dict[str, Any] = {
                        "content": result,
                        "tool_call_id": tool_use_id,
                    }
                    if status is not None:
                        tm_kwargs["status"] = status
                    messages.append(ToolMessage(**tm_kwargs))
                    resolved_tool_ids.add(tool_use_id)
            else:
                messages.append(HumanMessage(content=_join_text_blocks(content)))

        else:  # assistant
            text = _join_text_blocks(content)
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, list):
                for block in content:
                    if _block_type(block) == "tool_use":
                        tid = str(block.get("id", ""))
                        tool_calls.append({
                            "id": tid,
                            "name": str(block.get("name", "")),
                            "args": block.get("input", {}) or {},
                        })
                        open_tool_ids.append(tid)
            if not text and not tool_calls:
                continue  # empty assistant turn carries nothing
            messages.append(
                AIMessage(content=text, tool_calls=tool_calls)
                if tool_calls
                else AIMessage(content=text)
            )

    # Well-formedness: any tool_use without a matching tool_result gets a
    # placeholder ToolMessage so providers that reject dangling tool calls
    # (Anthropic et al.) accept the imported history on the next real turn.
    for tid in open_tool_ids:
        if tid and tid not in resolved_tool_ids:
            messages.append(
                ToolMessage(content=_placeholder_tool_result(), tool_call_id=tid)
            )

    return messages


def resolve_transcript_path(uuid_or_path: str, cwd: str | None) -> Path:
    """Resolve ``--import-claude`` argument to a transcript file.

    - If it points at an existing file (or ends with ``.jsonl`` / contains a
      path separator), treat it as a path.
    - Otherwise treat it as a bare session uuid: prefer
      ``session_project_dir(cwd)/<uuid>.jsonl`` when ``cwd`` is known, else glob
      ``~/.claude/projects/*/<uuid>.jsonl``.

    Raises:
        FileNotFoundError: nothing matches.
        ValueError: a bare uuid matches more than one project dir (ambiguous).
    """
    candidate = Path(uuid_or_path)
    looks_like_path = (
        candidate.exists()
        or uuid_or_path.endswith(".jsonl")
        or "/" in uuid_or_path
        or os_sep_in(uuid_or_path)
    )
    if looks_like_path:
        if not candidate.exists():
            raise FileNotFoundError(
                f"Claude transcript file not found: {uuid_or_path}"
            )
        return candidate

    uuid = uuid_or_path
    # Preferred: the agent's own project dir.
    if cwd:
        preferred = session_project_dir(cwd) / f"{uuid}.jsonl"
        if preferred.exists():
            return preferred

    projects_root = Path.home() / ".claude" / "projects"
    matches = sorted(projects_root.glob(f"*/{uuid}.jsonl")) if projects_root.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"No Claude session '{uuid}.jsonl' found under {projects_root} "
            f"(looked in the agent project dir and across all projects)."
        )
    if len(matches) > 1:
        listing = "\n  ".join(str(m) for m in matches)
        raise ValueError(
            f"Ambiguous session uuid '{uuid}' — found in multiple project "
            f"dirs:\n  {listing}\nPass an explicit file path instead."
        )
    return matches[0]


def os_sep_in(s: str) -> bool:
    """True if ``s`` contains an OS path separator (``\\`` on Windows)."""
    import os  # noqa: PLC0415

    return os.sep in s or (os.altsep is not None and os.altsep in s)
