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
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Protocol

from dw_hist import lister, search
from dw_hist.search import HistoryResultLike


def _force_offline_quiet_env() -> None:
    """Force HF / transformers / tokenizers into strictly-offline + quiet mode.

    Must run BEFORE any import of transformers / huggingface_hub / llama_index
    so the env vars take effect at module-load time. The embedding model is
    already on disk (semantic_search already passes ``local_files_only=True``);
    we make absolutely sure no HF Hub request is attempted and that progress
    bars / advisory warnings stay silent unless the operator asked for them.

    The user can opt back into all of this by passing ``--verbose`` (we set
    ``DW_HIST_VERBOSE=1`` before re-importing; this function then becomes a
    no-op).
    """
    if os.environ.get("DW_HIST_VERBOSE") == "1":
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # tqdm progress bars
    os.environ.setdefault("TQDM_DISABLE", "1")


def _silence_loggers_and_warnings() -> None:
    """Silence noisy third-party loggers after import."""
    if os.environ.get("DW_HIST_VERBOSE") == "1":
        return
    for name in (
        "transformers",
        "transformers.modeling_utils",
        "huggingface_hub",
        "llama_index",
        "llama_index.core",
        "sentence_transformers",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    # Best-effort: transformers.utils.logging may already be imported by the
    # time _ensure_embed_model runs.
    try:  # pragma: no cover - depends on optional dep being installed
        from transformers.utils import logging as _tlog  # noqa: PLC0415

        _tlog.set_verbosity_error()
        _tlog.disable_progress_bar()
    except Exception:
        pass


class SearchFunc(Protocol):
    """Signature of ``semantic_search.history_retriever.search_history`` (the
    subset dw-hist relies on). Stubbable in tests."""

    def __call__(
        self, *, query: str, summaries_dir: Any, top_k: int
    ) -> list[HistoryResultLike]: ...


class SearchMessagesFunc(Protocol):
    """Signature of ``semantic_search.message_search.search_messages`` (the
    subset dw-hist relies on). Stubbable in tests.

    The isotropy-transform kwargs (``transform``, ``whitening_method``,
    ``whitening_eps``, ``whitening_var``, ``whitening_dims``, ``stats_scope``,
    ``stats_out``) are always forwarded by the CLI. Test stubs must accept them
    (e.g. via ``**kwargs``) — a Protocol default does NOT make a concrete stub
    tolerant of extra kwargs.
    """

    def __call__(
        self, *,
        query: str,
        summaries_dir: Any,
        top_k: int,
        flags_any: int = 0,
        transform: str = "raw",
        whitening_method: str = "zca",
        whitening_eps: float = 1e-3,
        whitening_var: float = 0.90,
        whitening_dims: int | None = None,
        stats_scope: str = "corpus",
        stats_out: dict[str, Any] | None = None,
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
        help=(
            "Natural-language search over the chunked history index "
            "(mutually exclusive with -qm; one is required with <agent_name>)."
        ),
    )
    p.add_argument(
        "-qm", "--query-messages",
        dest="query_messages",
        help=(
            "Search over the per-message embedding store: ranks each single "
            "message by the query's projection (dot product / cosine) onto its "
            "embedding. Shows the matched flags. Mutually exclusive with -q."
        ),
    )
    p.add_argument(
        "-f", "--flag",
        dest="flags",
        action="append",
        metavar="FLAG",
        help=(
            "Restrict -qm scoring to messages with at least one of these flags "
            "(repeatable, logical OR). Case-insensitive MessageFlag name "
            "(USER_ROLE, TASK_MANAGER, PLAIN_USER, AGENT, AGENT_TO_AGENT, "
            "TOOL_RESULT, TOOL_CALL, SYSTEM, SUMMARIZATION, AGENT_FINAL). "
            "Only valid with -qm."
        ),
    )
    # Isotropy transforms (see docs/DESIGN_DW_HIST_QM_ISOTROPY.md). Only for -qm.
    p.add_argument(
        "-c", "--centering",
        action="store_true",
        dest="centering",
        help=(
            "Only with -qm. Rank in the mean-centered space (subtract corpus μ, "
            "renormalise, cosine). Mitigates e5's anisotropic-cone bias for "
            "diffuse queries. Mutually exclusive with -w."
        ),
    )
    p.add_argument(
        "-w", "--whitening",
        action="store_true",
        dest="whitening",
        help=(
            "Only with -qm. Rank in the whitened space "
            "((x-μ)·W, renormalise, cosine). Mutually exclusive with -c."
        ),
    )
    p.add_argument(
        "--whitening-method",
        choices=("pca", "zca"),
        default="zca",
        dest="whitening_method",
        help=(
            "Only with -w. Whitening variant (default zca — symmetric W, keeps "
            "the original axes). PCA and ZCA yield identical cosine rankings "
            "(cosine is invariant to the orthogonal factor that distinguishes "
            "them); the choice is kept for future non-cosine experiments."
        ),
    )
    p.add_argument(
        "--whitening-eps",
        type=float,
        default=1e-3,
        dest="whitening_eps",
        help=(
            "Only with -w. Regularisation ε added to the kept eigenvalue block "
            "(λ_r + ε) for numerical safety. Must be > 0 (default 1e-3)."
        ),
    )
    # Truncation controls (§3.6). default=None is a SENTINEL meaning "not passed"
    # so validation can tell an explicit value from the numeric default (0.90),
    # which lives in search_messages.
    p.add_argument(
        "--whitening-var", "-wv",
        type=float,
        default=None,
        dest="whitening_var",
        help=(
            "Only with -w. Keep the top principal axes covering this fraction of "
            "total variance (0 < F <= 1), dropping the noisy low-variance tail. "
            "Default 0.90. Use 1.0 for full whitening (keep all axes). Mutually "
            "exclusive with --whitening-dims."
        ),
    )
    p.add_argument(
        "--whitening-dims", "-wd",
        type=int,
        default=None,
        dest="whitening_dims",
        help=(
            "Only with -w. Keep exactly this many top axes (>= 1), overriding "
            "--whitening-var. Clamped to the usable rank. Mutually exclusive "
            "with --whitening-var. Use the embedding dim for full whitening."
        ),
    )
    p.add_argument(
        "--stats-scope",
        choices=("corpus", "filtered"),
        default="corpus",
        dest="stats_scope",
        help=(
            "Only with -c or -w. Rows used to estimate μ/Σ: 'corpus' "
            "(default, all N store rows — most stable) or 'filtered' (only "
            "the -f candidate rows — isotropise relative to the filter)."
        ),
    )
    p.add_argument(
        "--score-stats",
        action="store_true",
        dest="score_stats",
        help=(
            "Only with -qm. Also print a one-line score distribution "
            "(min/median/max/range over all M candidates, not just the top-k) "
            "to stderr for diagnostics."
        ),
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
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Show third-party library noise on stderr "
            "(HF download warnings, BertModel load report, tqdm bars, etc.). "
            "Off by default — the search runs strictly offline against the "
            "already-cached embedding model and stderr is silenced."
        ),
    )
    return p


def _load_search_history() -> SearchFunc:
    """Lazy import; only called when an actual search runs."""
    from deep_worker.providers.semantic_search.history_retriever import search_history  # noqa: PLC0415

    return search_history  # type: ignore[return-value]


def _load_search_messages() -> SearchMessagesFunc:
    """Lazy import; only called when an actual -qm search runs."""
    from deep_worker.providers.semantic_search.message_search import search_messages  # noqa: PLC0415

    return search_messages  # type: ignore[return-value]


def main(
    argv: list[str] | None = None,
    *,
    home: Path | None = None,
    search_func: SearchFunc | None = None,
    search_messages_func: SearchMessagesFunc | None = None,
) -> int:
    """Entry point. Returns the exit code; does NOT call ``sys.exit``.

    Args:
        argv: argument vector (excludes program name). Defaults to sys.argv[1:].
        home: $HOME override (used by tests).
        search_func: stubbable ``search_history``; lazy-loaded when None and a
            chunked (-q) search is requested.
        search_messages_func: stubbable ``search_messages``; lazy-loaded when
            None and a per-message (-qm) search is requested.
    """
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # ---- Verbose-mode opt-in BEFORE we touch any HF env vars. ----
    if getattr(args, "verbose", False):
        os.environ["DW_HIST_VERBOSE"] = "1"

    has_q = args.query is not None
    has_qm = args.query_messages is not None
    has_flags = bool(args.flags)
    has_c = bool(args.centering)
    has_w = bool(args.whitening)
    # argparse gives the default whenever the flag is absent; we detect
    # "user provided this flag" by checking argv rather than the value so a
    # user-supplied default (e.g. --whitening-method zca) is still an error
    # without -w.
    argv_str = argv if argv is not None else sys.argv[1:]
    provided_wm = any(a.startswith("--whitening-method") for a in argv_str)
    provided_we = any(a.startswith("--whitening-eps") for a in argv_str)
    provided_scope = any(a.startswith("--stats-scope") for a in argv_str)
    # -wv/-wd carry a None sentinel default, so "provided" == "not None" — this
    # also transparently covers the -wv/-wd aliases without argv string matching.
    provided_wv = args.whitening_var is not None
    provided_wd = args.whitening_dims is not None

    # ---- Argument combination validation (RC-A) ----
    if args.list_agents and args.agent_name is not None:
        print(
            "error: cannot combine --list with an agent name.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if has_q and has_qm:
        print(
            "error: -q/--query and -qm/--query-messages are mutually exclusive.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if has_flags and not has_qm:
        print(
            "error: -f/--flag is only valid with -qm/--query-messages.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    # Isotropy validation (see docs/DESIGN_DW_HIST_QM_ISOTROPY.md §4).
    if has_c and has_w:
        print(
            "error: -c/--centering and -w/--whitening are mutually exclusive.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if (has_c or has_w) and not has_qm:
        print(
            "error: -c/--centering and -w/--whitening are only valid with "
            "-qm/--query-messages.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if provided_scope and not (has_c or has_w):
        print(
            "error: --stats-scope only applies with -c/--centering or "
            "-w/--whitening.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if (provided_wm or provided_we or provided_wv or provided_wd) and not has_w:
        print(
            "error: --whitening-method / --whitening-eps / --whitening-var / "
            "--whitening-dims require -w/--whitening.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.whitening_eps <= 0.0:
        print(
            f"error: --whitening-eps must be > 0 (got {args.whitening_eps}).",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if provided_wv and provided_wd:
        print(
            "error: --whitening-var and --whitening-dims are mutually "
            "exclusive.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if provided_wv and not (0.0 < args.whitening_var <= 1.0):
        print(
            f"error: --whitening-var must be in (0, 1] (got "
            f"{args.whitening_var}).",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if provided_wd and args.whitening_dims < 1:
        print(
            f"error: --whitening-dims must be >= 1 (got {args.whitening_dims}).",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.score_stats and not has_qm:
        print(
            "error: --score-stats only applies with -qm/--query-messages.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.agent_name is None and not args.list_agents:
        if has_q or has_qm:
            print(
                "error: -q/-qm requires an agent name.",
                file=sys.stderr,
            )
            return EXIT_USAGE
        # No agent and no --list — print usage and exit usage-error.
        parser.print_usage(sys.stderr)
        print("error: provide an agent name or use --list.", file=sys.stderr)
        return EXIT_USAGE
    if args.agent_name is not None and not (has_q or has_qm):
        print(
            f"error: agent '{args.agent_name}' requires -q/--query or "
            "-qm/--query-messages.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # ---- Dispatch ----
    if args.list_agents:
        return _cmd_list(home=home, fmt=args.format)
    if has_qm:
        if has_c:
            transform = "centering"
        elif has_w:
            transform = "whitening"
        else:
            transform = "raw"
        return _cmd_search_messages(
            agent=args.agent_name,
            query=args.query_messages,
            flag_names=args.flags or [],
            top_k=args.top_k,
            fmt=args.format,
            home=home,
            search_messages_func=search_messages_func,
            transform=transform,
            whitening_method=args.whitening_method,
            whitening_eps=args.whitening_eps,
            whitening_var=args.whitening_var,
            whitening_dims=args.whitening_dims,
            stats_scope=args.stats_scope,
            score_stats=args.score_stats,
        )
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

    # RC-G/H: force strictly-offline + quiet env BEFORE the heavy import so
    # transformers / huggingface_hub / tqdm / llama_index pick up the silent
    # offline configuration when they initialise. ``--verbose`` opts out.
    _force_offline_quiet_env()

    # Lazy-load real search_history only if no stub was injected.
    sf = search_func or _load_search_history()

    # Silence loggers/warnings after import (some are only registered on
    # import). No-op under --verbose.
    _silence_loggers_and_warnings()

    # RC-E: exception mapping.
    # Third-party deps emit noise on BOTH streams the first time the embedder
    # is loaded:
    #   stdout: llama_index's "LLM is explicitly disabled. Using MockLLM."
    #   stderr: transformers/tqdm progress bar ("Loading weights: ..."),
    #           HF unauth warning, and the BertModel LOAD REPORT.
    # All of it is benign on a cached, strictly-offline model — but it
    # corrupts our --format json output and confuses operators who asked for
    # a search, not a load report. Redirect both streams to /dev/null around
    # the search call (unless --verbose). dw-hist's own error/result text
    # is printed AFTER the context manager closes.
    try:
        if os.environ.get("DW_HIST_VERBOSE") == "1":
            results = sf(query=query, summaries_dir=summaries_dir, top_k=top_k)
        else:
            with open(os.devnull, "w", encoding="utf-8") as devnull, \
                    contextlib.redirect_stdout(devnull), \
                    contextlib.redirect_stderr(devnull):
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


def _resolve_flags(names: list[str]) -> int:
    """Resolve case-insensitive MessageFlag names to a combined OR bitmask.

    Raises ``KeyError`` for an unknown name (caller → usage error). Uses
    semantic_search's single source of truth for the name→bit mapping.
    """
    from deep_worker.providers.semantic_search.message_embedding_store import flag_from_name  # noqa: PLC0415

    mask = 0
    for n in names:
        mask |= flag_from_name(n)
    return mask


def _valid_flag_names() -> str:
    from deep_worker.providers.semantic_search.message_embedding_store import FLAG_NAMES  # noqa: PLC0415

    return ", ".join(FLAG_NAMES)


def _cmd_search_messages(
    *,
    agent: str,
    query: str,
    flag_names: list[str],
    top_k: int,
    fmt: str,
    home: Path | None,
    search_messages_func: SearchMessagesFunc | None,
    transform: str = "raw",
    whitening_method: str = "zca",
    whitening_eps: float = 1e-3,
    whitening_var: float | None = None,
    whitening_dims: int | None = None,
    stats_scope: str = "corpus",
    score_stats: bool = False,
) -> int:
    # RC-2: reject empty / whitespace-only query BEFORE any heavy work — an empty
    # query embeds to a zero vector, making every score 0.0 (meaningless hits).
    if not query.strip():
        print(
            "error: -qm/--query-messages requires a non-empty query.",
            file=sys.stderr,
        )
        return EXIT_USAGE

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

    # Resolve flag names → bitmask; unknown name → usage error.
    try:
        flags_any = _resolve_flags(flag_names)
    except KeyError as exc:
        print(
            f"error: unknown flag {exc}. Valid flags: {_valid_flag_names()}.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    summaries_dir = lister.agent_summaries_dir(agent, home=home)
    # Pre-check the per-message embedding STORE (distinct from the chunked
    # history index used by -q): both the embeddings and the flag metadata must
    # exist (flags are needed for -f and for the per-result Flags column).
    store = summaries_dir / ".message_embeddings"
    if not (
        (store / "embeddings.npy").is_file()
        and (store / "flags.npy").is_file()
        and (store / "manifest.json").is_file()
    ):
        print(
            f"error: No per-message embedding store for agent '{agent}'. "
            "Build it with "
            "semantic_search.message_embedding_store.build_message_embeddings(...) "
            "and build_message_metadata(...).",
            file=sys.stderr,
        )
        return EXIT_NO_INDEX

    # RC-G/H: strictly-offline + quiet env BEFORE the heavy import.
    _force_offline_quiet_env()
    smf = search_messages_func or _load_search_messages()
    _silence_loggers_and_warnings()

    # `stats_out` is filled by search_messages. Needed for --score-stats AND for
    # the whitening kept-axis notice (RC-5, shown even without --score-stats), so
    # allocate the bucket whenever either applies.
    want_stats = score_stats or transform == "whitening"
    stats_bucket: dict[str, Any] | None = {} if want_stats else None
    # The numeric default for whitening_var (0.90) lives in search_messages; only
    # forward an explicit value. dims forwards through as-is (None == not set).
    call_kwargs: dict[str, Any] = dict(
        query=query, summaries_dir=summaries_dir,
        top_k=top_k, flags_any=flags_any,
        transform=transform,
        whitening_method=whitening_method,
        whitening_eps=whitening_eps,
        whitening_dims=whitening_dims,
        stats_scope=stats_scope,
        stats_out=stats_bucket,
    )
    if whitening_var is not None:
        call_kwargs["whitening_var"] = whitening_var
    # RC-E: redirect third-party load noise to /dev/null (unless --verbose) and
    # map exceptions to exit codes (FileNotFoundError → no store; else runtime).
    try:
        if os.environ.get("DW_HIST_VERBOSE") == "1":
            results = smf(**call_kwargs)
        else:
            with open(os.devnull, "w", encoding="utf-8") as devnull, \
                    contextlib.redirect_stdout(devnull), \
                    contextlib.redirect_stderr(devnull):
                results = smf(**call_kwargs)
    except FileNotFoundError as exc:
        print(
            f"error: No per-message embedding store for agent '{agent}': {exc}",
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
    # Kept-axis notice (RC-5): whenever whitening actually truncated, report the
    # ACTUAL number of axes kept — on stderr, independently of --score-stats, so
    # -wv/-wd alone show their effect and JSON stdout stays parseable.
    if stats_bucket is not None:
        kept = stats_bucket.get("whitening_dims_kept")
        if isinstance(kept, int):
            total = stats_bucket.get("whitening_dim", kept)
            requested = stats_bucket.get("whitening_dims_requested")
            var_target = stats_bucket.get("whitening_var")
            if isinstance(requested, int):
                # -wd path. Note a clamp only when the ask differs from reality.
                extra = (
                    f"requested {requested}"
                    if requested == kept
                    else f"requested {requested}, clamped to usable rank"
                )
            else:
                target = var_target if isinstance(var_target, float) else 0.90
                extra = f"variance target {target:.2f}"
            print(
                f"whitening: kept {kept} of {total} axes ({extra})",
                file=sys.stderr,
            )

    # --score-stats: one-line diagnostic on stderr AFTER the results, so JSON on
    # stdout stays parseable. Only when explicitly requested.
    if score_stats and stats_bucket is not None and stats_bucket.get("m"):
        print(
            f"score stats ({stats_bucket['transform']}, M={stats_bucket['m']}): "
            f"min={stats_bucket['min']:.3f} "
            f"median={stats_bucket['median']:.3f} "
            f"max={stats_bucket['max']:.3f} "
            f"range={stats_bucket['range']:.3f}",
            file=sys.stderr,
        )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
