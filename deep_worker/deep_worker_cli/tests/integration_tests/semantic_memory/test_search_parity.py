"""The service must answer exactly what the tools used to answer (§12).

Moving the engine into another process is only safe if the text coming
back is unchanged: these strings go into an agent's context, and the tool
output format is a signed invariant. So this file does not check that the
service returns "something reasonable" — it checks byte equality.

The reference is the formatting code the tools carried *before* slice 6,
transcribed below from commit ``aae5ef1``
(``deepagents/tools/project_memory.py:68-88`` and
``deep_worker_cli/history_memory_tool.py:84-100``). Transcribing it is
the point: importing today's renderer from the service would compare the
engine against itself and pass no matter how the format drifted. These
literals are the contract, and they must be edited only when the owner
decides the tool output changes.

Real indexes, no fixtures: this repository's own ``.vector_index`` and a
live agent's ``.history_index``. If neither exists the test skips rather
than inventing one, because building an index here would mean loading the
heavy stack into the test process — the exact thing this feature exists
to avoid.

``allow_live_home``: both tests READ the operator's home — the embedding
model cache under ``~/.cache/llama_index`` and the agents' history
indexes. Neither writes there, and neither can: the only mutating RPC is
IndexHistory, which nothing here calls. Without the opt-out the sandbox
HOME points the model loader at an empty cache, so the *reference* call
fails while the service (which keeps the real HOME) succeeds — turning a
parity test into an environment test.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from deep_worker_cli.semantic_memory.client import SemanticMemoryClient

from .conftest import BUDGET_S, ServiceRig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.allow_live_home,
    pytest.mark.timeout(BUDGET_S),
]

#: <tree>/deep_worker/deep_worker_cli/tests/integration_tests/semantic_memory
_TREE = Path(__file__).resolve().parents[5]

_TOP_K = 3


def render_project_pre_slice6(query: str, results: list[Any]) -> str:
    """Verbatim from ``tools/project_memory.py`` at aae5ef1 (pre-service)."""
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


def render_history_pre_slice6(query: str, results: list[Any]) -> str:
    """Verbatim from ``history_memory_tool.py`` at aae5ef1 (pre-service)."""
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


def _project_index_or_skip() -> Path:
    if not (_TREE / ".vector_index").is_dir():
        pytest.skip(f"no project index at {_TREE / '.vector_index'}")
    return _TREE


def _history_dir_or_skip() -> Path:
    from deep_worker_cli.storage_paths import deep_worker_home

    agents_dir = deep_worker_home() / "agents"
    if not agents_dir.is_dir():
        pytest.skip(f"no agents directory at {agents_dir}")

    for summaries in sorted(agents_dir.glob("*/summaries")):
        if (summaries / ".history_index").is_dir():
            return summaries
    pytest.skip("no agent has a .history_index yet")


@pytest.fixture
def parity_rig() -> Iterator[ServiceRig]:
    """A throwaway service for the parity comparisons.

    Not the shared ``service_rig`` fixture: these tests also load the
    engine in-process for the reference answer, so keeping their registry
    root separate makes a stuck service easy to attribute.
    """
    rig = ServiceRig(
        base_dir=Path(tempfile.mkdtemp(prefix="dw_semantic_parity_")) / "sm"
    )
    try:
        yield rig
    finally:
        rig.reap()


def test_project_search_matches_pre_service_rendering(
    parity_rig: ServiceRig,
) -> None:
    """SearchProject over the wire == what the old in-process tool printed."""
    repo_path = _project_index_or_skip()
    query = "semantic memory service"

    from deep_worker.providers.semantic_search.retriever import search

    expected = render_project_pre_slice6(query, search(query, repo_path, _TOP_K))

    parity_rig.ensure_started()
    client = SemanticMemoryClient(supervisor=parity_rig.supervisor)
    served = asyncio.run(client.search_project(query, repo_path, _TOP_K))

    assert served, "the service returned nothing for an indexed repo"
    assert "--- Result 1 " in served, "expected at least one hit to compare"
    assert served == expected


def test_history_search_matches_pre_service_rendering(
    parity_rig: ServiceRig,
) -> None:
    """SearchHistory over the wire == what the old in-process tool printed."""
    summaries_dir = _history_dir_or_skip()
    query = "what did we decide"

    from deep_worker.providers.semantic_search.history_retriever import (
        search_history,
    )

    expected = render_history_pre_slice6(
        query, search_history(query, summaries_dir, _TOP_K)
    )

    parity_rig.ensure_started()
    client = SemanticMemoryClient(supervisor=parity_rig.supervisor)
    served = asyncio.run(client.search_history(query, summaries_dir, _TOP_K))

    assert served, "the service returned nothing for an indexed agent"
    assert "--- Result 1 " in served, "expected at least one hit to compare"
    assert served == expected
