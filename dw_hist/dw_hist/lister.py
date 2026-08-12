"""Agent-with-history-index discovery.

Read-only scan of ``~/.deep_worker/<agent>/summaries/.history_index/lancedb/``.
"""

from __future__ import annotations

from pathlib import Path

from deep_worker_cli.storage_paths import deep_worker_home


def _has_index(summaries_dir: Path) -> bool:
    return (summaries_dir / ".history_index" / "lancedb").is_dir()


def discover_agents(home: Path | None = None) -> list[str]:
    """Return sorted list of agent names whose summaries directory contains a
    built history semantic index.

    Args:
        home: ``$HOME``-like base. Defaults to ``Path.home()``. Override for
            tests.

    Returns:
        Sorted list of agent names. Empty list when ``~/.deep_worker/``
        does not exist or contains no agents with an index.
    """
    base = deep_worker_home(home)
    if not base.is_dir():
        return []
    out: list[str] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        summaries = entry / "summaries"
        if not summaries.is_dir():
            continue
        if _has_index(summaries):
            out.append(entry.name)
    out.sort()
    return out


def agent_summaries_dir(agent_name: str, home: Path | None = None) -> Path:
    """Return ``deep_worker_home(home)/<agent>/summaries`` (pure path; does
    NOT check existence)."""
    return deep_worker_home(home) / agent_name / "summaries"


def agent_root(agent_name: str, home: Path | None = None) -> Path:
    """Return ``deep_worker_home(home)/<agent>`` (pure path; does NOT
    check)."""
    return deep_worker_home(home) / agent_name
