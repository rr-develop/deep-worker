"""dw-hist CLI entry point.

Exit codes (see docs/DESIGN_DW_HIST_INITIAL.md §2.3):
  0  success / --list with zero agents
  2  usage error (conflicting flags, invalid --top-k, unknown agent)
  3  agent has no history index, or index disappeared mid-search
  4  search runtime error
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Protocol

from dw_hist import lister, search
from dw_hist.search import HistoryResultLike


class SearchFunc(Protocol):
    """Signature of ``semantic_search.history_retriever.search_history`` (the
    subset dw-hist relies on). Stubbable in tests."""

    def __call__(
        self, *, query: str, summaries_dir: Any, top_k: int
    ) -> list[HistoryResultLike]: ...

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_INDEX = 3
EXIT_RUNTIME = 4


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dw-hist",
        description=(
            "Host-level CLI for exploring deep-worker agent conversation "
            "history. Read-only consumer of each agent's semantic history "
            "index (see docs/DESIGN_DW_HIST_INITIAL.md)."
        ),
    )
    p.add_argument(
        "agent_name",
        nargs="?",
        help="Agent to operate on (omit when using --list).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_agents",
        help="List all agents that have a built history semantic index.",
    )
    p.add_argument(
        "-q", "--query",
        help="Natural-language search query (required with <agent_name>).",
    )
    p.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (1..20, default 5).",
    )
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return p


def _load_search_history() -> SearchFunc:
    """Lazy import; only called when an actual search runs."""
    from semantic_search.history_retriever import search_history  # noqa: PLC0415

    return search_history  # type: ignore[return-value]


def main(
    argv: list[str] | None = None,
    *,
    home: Path | None = None,
    search_func: SearchFunc | None = None,
) -> int:
    """Entry point. Returns the exit code; does NOT call ``sys.exit``.

    Args:
        argv: argument vector (excludes program name). Defaults to sys.argv[1:].
        home: $HOME override (used by tests).
        search_func: stubbable ``search_history``; lazy-loaded when None and a
            search is requested.
    """
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # ---- Argument combination validation (RC-A) ----
    if args.list_agents and args.agent_name is not None:
        print(
            "error: cannot combine --list with an agent name.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.agent_name is None and not args.list_agents:
        if args.query is not None:
            print(
                "error: -q/--query requires an agent name.",
                file=sys.stderr,
            )
            return EXIT_USAGE
        # No agent and no --list — print usage and exit usage-error.
        parser.print_usage(sys.stderr)
        print("error: provide an agent name or use --list.", file=sys.stderr)
        return EXIT_USAGE
    if args.agent_name is not None and args.query is None:
        print(
            f"error: agent '{args.agent_name}' requires -q/--query.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # ---- Dispatch ----
    if args.list_agents:
        return _cmd_list(home=home, fmt=args.format)
    return _cmd_search(
        agent=args.agent_name,
        query=args.query,
        top_k=args.top_k,
        fmt=args.format,
        home=home,
        search_func=search_func,
    )


def _cmd_list(*, home: Path | None, fmt: str) -> int:
    agents = lister.discover_agents(home=home)
    if fmt == "json":
        print(json.dumps({
            "agents": [
                {
                    "name": name,
                    "summaries_dir": str(lister.agent_summaries_dir(name, home=home)),
                }
                for name in agents
            ]
        }))
    else:
        for name in agents:
            print(name)
    return EXIT_OK


def _cmd_search(
    *,
    agent: str,
    query: str,
    top_k: int,
    fmt: str,
    home: Path | None,
    search_func: SearchFunc | None,
) -> int:
    # RC-C: validate top_k BEFORE any heavy import / search call.
    if not (1 <= top_k <= 20):
        print(
            f"error: --top-k must be between 1 and 20 (got {top_k}).",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # RC-B: unknown agent → exit 2 (check agent ROOT, not summaries).
    root = lister.agent_root(agent, home=home)
    if not root.is_dir():
        print(f"error: unknown agent: '{agent}' (no {root})", file=sys.stderr)
        return EXIT_USAGE

    summaries_dir = lister.agent_summaries_dir(agent, home=home)
    # Pre-check index presence using our own helper (matches what
    # discover_agents would consider a valid index, and avoids importing
    # semantic_search for the negative case so unit tests stay fast).
    if not (summaries_dir / ".history_index" / "lancedb").is_dir():
        print(
            f"error: No history index for agent '{agent}'. "
            "Start the agent with --history-memory and let a "
            "summarization cycle complete before searching.",
            file=sys.stderr,
        )
        return EXIT_NO_INDEX

    # Lazy-load real search_history only if no stub was injected.
    sf = search_func or _load_search_history()

    # RC-E: exception mapping.
    # Some dependencies (e.g. llama_index) emit benign banners on stdout the
    # first time the embedder/index is loaded (notably
    # "LLM is explicitly disabled. Using MockLLM."). That noise must not
    # corrupt our own stdout output — especially --format json. Redirect any
    # stdout produced *by the search call* to stderr.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results = sf(query=query, summaries_dir=summaries_dir, top_k=top_k)
    except FileNotFoundError as exc:
        # Index removed between our pre-check and the call, or
        # search_history's own pre-check failed for some reason.
        print(
            f"error: No history index for agent '{agent}': {exc}",
            file=sys.stderr,
        )
        return EXIT_NO_INDEX
    except Exception as exc:  # noqa: BLE001 — intentional broad catch at CLI boundary
        print(f"error: Search error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    if fmt == "json":
        print(json.dumps(
            search.format_json(agent=agent, query=query, results=results),
            ensure_ascii=False,
        ))
    else:
        print(search.format_text(query=query, results=results))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
