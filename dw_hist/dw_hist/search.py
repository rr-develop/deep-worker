"""Output formatting for dw-hist search.

The text format mirrors ``history_memory_tool._run_history_search`` byte-shape;
the JSON format is a flat, scripting-friendly projection of the search results.

A small Protocol describes the duck-typed result objects we accept so this
module can be unit-tested without importing the heavy ``semantic_search`` deps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class HistoryResultLike(Protocol):
    content: str
    score: float
    thread_id: str
    line_range: str
    timestamp: str
    message_types: str
    file_path: str


def _flags_segment(r: object) -> str:
    """`" | Flags: A,B"` for a per-message result, else `""`.

    Back-compat: the chunked ``-q`` result objects have no ``flags`` attribute,
    so their text/JSON output stays byte-identical. Per-message ``-qm`` results
    carry ``flag_names`` (a decoded name list); we render it when present.
    """
    names = getattr(r, "flag_names", None)
    if names:
        return f" | Flags: {','.join(names)}"
    return ""


def format_text(*, query: str, results: Sequence[HistoryResultLike]) -> str:
    """Mirror of ``history_memory_tool`` text formatting + score precision.

    For per-message (``-qm``) results an extra ``| Flags: …`` segment is appended
    to the metadata line; chunked (``-q``) results are unchanged.
    """
    if not results:
        return (
            f"History Memory: '{query}'\n\n"
            "No matching results found in conversation history."
        )

    parts: list[str] = [f"History Memory: '{query}'\n"]
    for idx, r in enumerate(results, 1):
        lines_info = f"Lines: {r.line_range}" if r.line_range else "Position: unknown"
        time_info = f" | When: {r.timestamp}" if r.timestamp else ""
        type_info = f" | Types: {r.message_types}" if r.message_types else ""
        flags_info = _flags_segment(r)
        parts.append(
            f"--- Result {idx} (score={r.score:.3f}) ---\n"
            f"Thread: {r.thread_id}\n"
            f"{lines_info}{time_info}{type_info}{flags_info}\n\n"
            f"{r.content}\n"
        )
    return "\n".join(parts)


def format_json(
    *, agent: str, query: str, results: Sequence[HistoryResultLike]
) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    for r in results:
        rec: dict[str, Any] = {
            "content": r.content,
            "score": float(r.score),
            "thread_id": r.thread_id,
            "line_range": r.line_range,
            "timestamp": r.timestamp,
            "message_types": r.message_types,
            "file_path": r.file_path,
        }
        # Per-message (-qm) extras, only when present (back-compat for -q).
        flags = getattr(r, "flags", None)
        if flags is not None:
            rec["flags"] = int(flags)
            rec["flag_names"] = list(getattr(r, "flag_names", []))
        out.append(rec)
    return {"agent": agent, "query": query, "results": out}
