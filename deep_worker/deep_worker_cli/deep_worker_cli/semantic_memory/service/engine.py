"""The heavy half of the service: llama-index, LanceDB, torch.

This module is the ONLY place in the tree allowed to import the semantic
search provider. Everything above it — the supervisor, the registry, the
client, the tools — must stay importable without paying the 7.13 s /
884 MB cost that motivated the whole service (design §2).

Two rules hold every function here together:

**Nothing at import time.** The provider imports live inside the call
bodies. Importing this module must stay free so tests can exercise the
dispatch logic without a model, and so ``grpc_server`` can construct its
servicer before the first request decides which engine is needed.

**Nothing on an event loop.** Every function below is synchronous and
blocking on purpose. LanceDB bridges to its own background event loop
through a blocking ``run_coroutine_threadsafe(...).result()``; called on
a loop thread it holds the GIL while waiting, LanceDB's loop starves on
that GIL, and the process deadlocks — the live agent_c hang of
2026-06-15. ``grpc_server`` therefore hands every call here to a
dedicated executor (design §4.2, blocker B2). Do not add ``async def``
to this module: the blocking signature is what forces callers to
offload, and an awaitable one would silently invite the deadlock back.

The rendered strings returned by the search functions are the tool
output verbatim. Their format is a compatibility invariant (design §12):
agents and their tests already depend on it, so it is reproduced here
exactly as the in-process tools produced it, not re-designed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Matched to the in-process tools this replaces.
_MIN_TOP_K = 1
_MAX_TOP_K = 20


class ProviderUnavailable(RuntimeError):
    """The semantic-search extra is not installed in the service venv.

    Distinct from "this tenant has no index": that is a routine answer
    the caller renders as a hint, while this means the service itself
    cannot serve anyone and needs an install.
    """


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def _resolve_provider() -> None:
    """Put the semantic-search provider on ``sys.path`` if it isn't.

    Source checkouts keep the provider outside the CLI package; installed
    deployments ship it as a wheel. ``provider_paths`` already
    distinguishes the two, so defer to it rather than guessing — no
    ``sys.path`` surgery here.
    """
    from deep_worker_cli.provider_paths import ensure_provider_importable

    resolution = ensure_provider_importable("deep_worker.providers.semantic_search")
    if resolution.unavailable:
        raise ProviderUnavailable(
            resolution.remedy or "deep-worker[semantic-search] is not installed"
        )


def embedding_model_name() -> str:
    """Name of the configured embedding model, or "" if unresolvable.

    Read from the provider's own config rather than restated here: a
    hardcoded copy would drift the moment the provider changes models and
    would report a model the service is not actually running.
    """
    try:
        _resolve_provider()
        from deep_worker.providers.semantic_search.config import (  # noqa: PLC0415
            IndexConfig,
        )

        return str(IndexConfig().embedding_model)
    except Exception:  # noqa: BLE001 — Status must answer even when broken
        return ""


#: Roots scanned one level deep for project indexes. Deliberately a short
#: fixed list, not a filesystem walk: the point is to name targets the
#: agent can actually use, and an unbounded walk would be slow, would
#: surface unrelated checkouts, and would still miss anything outside it.
#: Override for tests and for hosts that keep repositories elsewhere.
PROJECT_SEARCH_ROOTS_ENV = "DEEP_WORKER_PROJECT_ROOTS"
_DEFAULT_PROJECT_ROOTS = ("/repos",)


def project_search_roots() -> list[Path]:
    """Roots to scan for project indexes, honouring the env override."""
    import os  # noqa: PLC0415

    raw = os.environ.get(PROJECT_SEARCH_ROOTS_ENV, "")
    roots = [part for part in raw.split(os.pathsep) if part] or list(
        _DEFAULT_PROJECT_ROOTS
    )
    return [Path(root) for root in roots]


def list_tenants(project: bool) -> list[str]:
    """Targets that currently hold an index of the requested family.

    Only indexed targets are returned: an un-indexed directory is not a
    usable target, and offering one would hand the agent a suggestion
    guaranteed to fail.

    Discovering existing indexes is not the same thing as creating them:
    §16 A6 rules out auto-indexing repositories, and this does none —
    it reports what is already on disk. Without it the "name a target
    explicitly" hint (§5.1) would have no names in it, which is what
    makes the extended addressing usable at all.

    Project roots are scanned one level deep; agents live in a single
    known directory and enumerate exactly.
    """
    if project:
        found: list[str] = []
        for root in project_search_roots():
            try:
                entries = sorted(p for p in root.iterdir() if p.is_dir())
            except OSError:
                continue
            found.extend(str(repo) for repo in entries if project_index_exists(repo))
        return found

    from deep_worker_cli.storage_paths import deep_worker_home  # noqa: PLC0415

    agents_dir = deep_worker_home() / "agents"
    try:
        candidates = sorted(p for p in agents_dir.iterdir() if p.is_dir())
    except OSError:
        return []

    return [
        agent_dir.name
        for agent_dir in candidates
        if history_index_exists(agent_dir / "summaries")
    ]


def load_model() -> None:
    """Import the provider and warm the embedding model.

    Called once at startup so the first real search does not pay the
    ~7.1 s import. Safe to call repeatedly; the provider caches.
    """
    _resolve_provider()
    from deep_worker.providers.semantic_search.retriever import (  # noqa: PLC0415
        search as _search,
    )

    del _search


# ---------------------------------------------------------------------------
# Index presence
# ---------------------------------------------------------------------------


def project_index_exists(repo_path: Path) -> bool:
    """True iff *repo_path* holds a complete project index.

    All three components are required. A bare ``.vector_index/``
    directory is NOT enough: ``RepoIndex.exists()`` checks only the
    directory and would call a half-written index usable, so the strict
    three-component check is the one the service commits to (design
    §11.3).
    """
    index_dir = repo_path / ".vector_index"
    return all(
        (index_dir / component).exists()
        for component in ("lancedb", "docstore/docstore.json", "file_hashes.json")
    )


def history_index_exists(summaries_dir: Path) -> bool:
    """True iff *summaries_dir* holds a history index."""
    index_dir = summaries_dir / ".history_index"
    return all(
        (index_dir / component).exists()
        for component in ("lancedb", "docstore/docstore.json", "line_hashes.json")
    )


def history_pending_lines(summaries_dir: Path) -> int:
    """Count indexable records not yet in the history index.

    Reads only the sidecar counts and the ``.jsonl`` record totals — no
    model, no LanceDB — so ``IndexStatus`` stays inside its 5 s deadline
    (design §4.3).

    Counting goes through the indexer's own ``count_jsonl_lines``, never a
    local reimplementation. JSONL separates records on ``\\n`` alone, while
    ``str.splitlines()`` also breaks on ``\\r``, ``\\v``, ``U+2028`` and
    ``U+2029`` — characters that appear legitimately inside JSON string
    values. A second counter that disagreed with the indexer would make
    ``complete`` report work as pending forever, which is the endless
    reindex loop that motivated the indexer's own rule
    (``history_indexer.py``, ``JSONL_RECORD_SEPARATOR``).
    """
    import json  # noqa: PLC0415

    _resolve_provider()
    from deep_worker.providers.semantic_search.history_indexer import (  # noqa: PLC0415
        count_jsonl_lines,
    )

    counts_path = summaries_dir / ".history_index" / "line_hashes.json"
    try:
        indexed: dict[str, int] = json.loads(counts_path.read_text("utf-8"))
    except (OSError, ValueError):
        indexed = {}

    try:
        jsonl_files = sorted(summaries_dir.glob("*.jsonl"))
    except OSError:
        return 0

    pending = 0
    for path in jsonl_files:
        total = count_jsonl_lines(path)
        # Keyed by the absolute path, exactly as the indexer writes it
        # (``line_counts[str(jsonl_path)]``). Keying by ``path.name``
        # instead matches nothing, so every already-indexed record reads
        # as pending and ``complete`` is permanently False.
        pending += max(0, total - int(indexed.get(str(path), 0)))
    return pending


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_project(query: str, repo_path: Path, top_k: int) -> str:
    """Semantic search over a repository. Blocking; run on an executor.

    Returns the rendered tool output, including the error and empty-result
    renderings — the caller passes it through untouched.
    """
    if not _MIN_TOP_K <= top_k <= _MAX_TOP_K:
        return f"Error: top_k must be between {_MIN_TOP_K} and {_MAX_TOP_K}"

    _resolve_provider()
    from deep_worker.providers.semantic_search.exceptions import (  # noqa: PLC0415
        IndexNotFoundError,
    )
    from deep_worker.providers.semantic_search.retriever import search  # noqa: PLC0415

    try:
        results = search(query, repo_path=repo_path, top_k=top_k)
    except IndexNotFoundError:
        return (
            f"Error: Project memory not available.\n"
            f"The directory '{repo_path}' does not have a semantic search index.\n"
            f"This usually means the index was not created or is corrupted."
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the agent as text
        return f"Error: Search failed with exception: {exc}"

    if not results:
        return (
            f"Project Memory: No results found for '{query}'\n\n"
            f"The semantic index may not contain information about this topic.\n"
            f"Try rephrasing your query or use other tools like grep or read_file."
        )

    lines: list[str] = [f"Project Memory: '{query}'\n"]
    for i, result in enumerate(results, 1):
        lang = f" [{result.language}]" if result.language else ""
        lines.append(
            f"--- Result {i} (score={result.score:.3f}){lang} ---\n"
            f"File: {result.file_path}\n"
            f"Type: {result.node_type}\n"
            f"\n{result.content}\n"
        )
    return "\n".join(lines)


def search_history(query: str, summaries_dir: Path, top_k: int) -> str:
    """Semantic search over an agent's history. Blocking; run on an executor."""
    if not _MIN_TOP_K <= top_k <= _MAX_TOP_K:
        return f"Error: top_k must be between {_MIN_TOP_K} and {_MAX_TOP_K}"

    _resolve_provider()
    from deep_worker.providers.semantic_search.history_retriever import (  # noqa: PLC0415
        search_history as _search_history,
    )

    try:
        results = _search_history(query=query, summaries_dir=str(summaries_dir), top_k=top_k)
    except FileNotFoundError as exc:
        return (
            f"History not indexed: {exc}\n"
            "Run the agent with --history-memory flag and allow a "
            "summarization cycle to complete before searching history."
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the agent as text
        return f"Search error: {type(exc).__name__}: {exc}"

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
        parts.append(
            f"--- Result {idx} (score={r.score:.3f}) ---\n"
            f"Thread: {r.thread_id}\n"
            f"{lines_info}{time_info}{type_info}\n\n"
            f"{r.content}\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_history(
    summaries_dir: Path, clear_first: bool
) -> tuple[str, int, int, int, int]:
    """(Re)index an agent's history. Blocking; run on an executor.

    Returns rendered text and exact index statistics. Incremental by
    line counts, so re-running after a dropped call costs only the new
    lines — which is what makes the fire-and-forget summarization hook
    safe to lose (design §8.1, blocker B3).
    """
    _resolve_provider()
    from deep_worker.providers.semantic_search.history_indexer import (  # noqa: PLC0415
        clear_history_index,
        index_all_history_files,
    )

    if clear_first:
        clear_history_index(summaries_dir)

    stats: Any = index_all_history_files(summaries_dir)
    files_indexed = int(getattr(stats, "files_processed", 0))
    chunks_added = int(getattr(stats, "chunks_indexed", 0))
    lines_read = int(getattr(stats, "lines_read", 0))
    errors = int(getattr(stats, "errors", 0))

    rendered = (
        f"Indexed {files_indexed} file(s), {chunks_added} chunk(s) "
        f"from '{summaries_dir}'."
    )
    if errors:
        rendered += f" {errors} file(s) failed."
    return rendered, files_indexed, chunks_added, lines_read, errors
