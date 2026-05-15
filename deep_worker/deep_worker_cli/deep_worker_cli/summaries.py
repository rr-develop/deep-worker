"""On-disk inventory and pruning for offloaded conversation-history files.

Design: ``deep_worker/docs/DESIGN_SUMMARIZATION_PERSISTENT_STORAGE.md``.

Layout this module operates on:

    <data-root>/<agent_name>/summaries/<thread_id>.md

Each file is the append-only log written by
``deepagents.middleware.summarization.SummarizationMiddleware``.
One section per summarization event::

    ## Summarized at <iso-8601-ts>

    <body: evicted messages as langchain get_buffer_string output>

This module is pure synchronous filesystem I/O over ``pathlib`` — no RPyC,
no asyncio, no threads, no polling. It is a thin read/edit layer on top of
a tiny append-only markdown-section format.

Public API:

- :func:`get_summaries_dir`  — per-agent directory path (does not create).
- :func:`iter_summary_files` — inventory, with parsed per-file metadata.
- :func:`parse_summary_headers` — list the ``## Summarized at <ts>`` lines.
- :func:`prune_summaries_before` — delete/rewrite sections older than a
  given timestamp. Used by ``deep-worker --remove-before``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Matches exactly the header the middleware writes:
#   "## Summarized at 2026-04-22T15:40:18.122420+00:00"
_HEADER_RE = re.compile(r"^## Summarized at (\S+)\s*$", re.MULTILINE)


def get_summaries_dir(agent_name: str) -> Path:
    """Return the per-agent summaries directory path.

    This is a pure function: it does NOT create the directory. Callers that
    need the directory to exist must call ``.mkdir(parents=True, exist_ok=True)``
    themselves (the middleware wiring in ``deep_worker_cli/agent.py`` does so
    once at agent startup).
    """
    return Path("~/.deep_worker").expanduser() / agent_name / "summaries"


def parse_summary_headers(text: str) -> list[str]:
    """Return the list of ISO-8601 timestamps from ``## Summarized at ...`` lines.

    Order is source order (== chronological, since the middleware appends).
    """
    return _HEADER_RE.findall(text)


@dataclass(frozen=True)
class SummaryFileInfo:
    """Metadata for one offloaded summary file.

    Attributes:
        agent: Agent name (directory name under the per-user data directory).
        thread: Thread id (filename stem — ``<agent>`` for main thread,
            ``<agent>:tm`` for the TaskManager thread, or a UUID for
            subagent threads).
        path: Absolute filesystem path of the file.
        events: Number of ``## Summarized at`` headers inside.
        latest_ts: Last header's timestamp string, or ``None`` for a
            header-less / corrupted file.
        size: File size in bytes.
    """
    agent: str
    thread: str
    path: Path
    events: int
    latest_ts: str | None
    size: int


def _deep_worker_root() -> Path:
    """Parent directory of all per-agent data directories."""
    return Path("~/.deep_worker").expanduser()


def iter_summary_files(agent_name: str | None = None) -> Iterator[SummaryFileInfo]:
    """Walk the per-agent summaries directories and yield :class:`SummaryFileInfo`.

    Args:
        agent_name: If given, yield only summaries for this agent. If
            ``None``, yield summaries for all agents.

    Behavior:
        - Missing root or missing per-agent dir → empty iterator (no error).
        - Non-``.md`` files in ``summaries/`` are silently skipped.
        - Files with no recognizable ``## Summarized at`` header still
          yield a record with ``events=0`` and ``latest_ts=None`` so the
          user can see and clean up corrupted files.
    """
    root = _deep_worker_root()
    if not root.is_dir():
        return

    agent_dirs: list[Path]
    if agent_name is None:
        agent_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        candidate = root / agent_name
        agent_dirs = [candidate] if candidate.is_dir() else []

    for agent_dir in agent_dirs:
        summaries_dir = agent_dir / "summaries"
        if not summaries_dir.is_dir():
            continue
        for f in sorted(summaries_dir.glob("*.md")):
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except OSError:
                continue
            headers = parse_summary_headers(text)
            yield SummaryFileInfo(
                agent=agent_dir.name,
                thread=f.stem,
                path=f,
                events=len(headers),
                latest_ts=headers[-1] if headers else None,
                size=f.stat().st_size,
            )


def _parse_ts(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp the middleware emits.

    Returns ``None`` on malformed input. Always returns a timezone-aware
    datetime (assumes UTC if the input lacks a tz suffix — the middleware
    always writes ``+00:00``, so this is just defensive).
    """
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split file text into ``(timestamp, section_text)`` tuples.

    ``section_text`` includes the ``## Summarized at ...`` header line AND
    everything until the next header (or EOF). Non-header preamble is
    silently dropped. Order is preserved.
    """
    # Find all header start offsets
    starts = [(m.start(), m.group(1)) for m in _HEADER_RE.finditer(text)]
    if not starts:
        return []
    sections: list[tuple[str, str]] = []
    for i, (start, ts) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections.append((ts, text[start:end]))
    return sections


def prune_summaries_before(
    cutoff: datetime,
    *,
    dry_run: bool = False,
    agent_name: str | None = None,
) -> dict[str, int]:
    """Delete or rewrite summary files so no section older than ``cutoff`` remains.

    For each summary file:

    - If ALL sections are older than ``cutoff`` → delete the file.
    - If SOME older and SOME newer → atomically rewrite the file keeping
      only the newer sections (write to ``<path>.tmp`` then ``os.replace``).
    - If ALL newer → leave the file untouched.

    Files with zero parseable headers are left alone (operator attention).

    Args:
        cutoff: Timestamp — sections with header timestamp STRICTLY older than
            this are eligible for removal.
        dry_run: If ``True``, compute what would be done but do not write or
            delete anything.
        agent_name: Optional filter — restrict to one agent.

    Returns:
        Summary dict with counters:

        - ``files_scanned`` — total files considered.
        - ``files_deleted`` — files removed entirely.
        - ``files_rewritten`` — files rewritten to drop old sections.
        - ``sections_deleted`` — how many sections total were evicted by
          rewrite (does NOT include sections from deleted-whole files; use
          ``sections_to_drop`` for the "would drop" projection).
        - ``sections_to_drop`` — planning counter: how many old sections
          exist in total (whether the file will end up deleted or rewritten).
          Useful for ``dry_run`` previews.
    """
    counters = {
        "files_scanned": 0,
        "files_deleted": 0,
        "files_rewritten": 0,
        "sections_deleted": 0,
        "sections_to_drop": 0,
    }

    for info in iter_summary_files(agent_name=agent_name):
        counters["files_scanned"] += 1
        try:
            text = info.path.read_text(encoding="utf-8")
        except OSError:
            continue
        sections = _split_sections(text)
        if not sections:
            # Nothing we can safely remove — leave for operator.
            continue

        kept: list[str] = []
        dropped = 0
        for ts, body in sections:
            dt = _parse_ts(ts)
            if dt is not None and dt < cutoff:
                dropped += 1
            else:
                kept.append(body)

        counters["sections_to_drop"] += dropped
        if dropped == 0:
            continue  # all sections newer; untouched

        if not kept:
            # Whole file obsolete.
            if not dry_run:
                try:
                    info.path.unlink()
                except OSError:
                    continue
            counters["files_deleted"] += 1
        else:
            # Rewrite, keeping only new sections, atomically.
            if not dry_run:
                new_text = "".join(kept)
                tmp = info.path.with_suffix(info.path.suffix + ".tmp")
                tmp.write_text(new_text, encoding="utf-8")
                os.replace(tmp, info.path)
            counters["files_rewritten"] += 1
            counters["sections_deleted"] += dropped

    return counters
