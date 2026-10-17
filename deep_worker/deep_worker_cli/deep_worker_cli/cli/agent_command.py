"""CLI command to start a deep-worker agent process."""

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.tool_selection import (
    merge_increments,
    parse_increment_spec,
    native_tool_caveat,
    parse_tool_selection,
    parse_tool_selection_spec,
)
from deep_worker_cli.client.agent_client import AgentClient
from deep_worker_cli.middleware.tool_completion import (
    _canonical_turn_ending_spec_str,
    _parse_turn_ending_tool_specs,
)
from deep_worker_cli.process.agent_process import AgentProcess
from deep_worker_cli.process.daemon import DAEMON_UMASK as DAEMON_UMASK
from deep_worker_cli.process.registry import ProcessRegistry
from deep_worker_cli.runtime.availability_tracker import AvailabilityTracker
from deep_worker_cli.runtime.provider_chain import (
    SUPPORTED_PROVIDERS,
    all_provider_names,
)
from deep_worker_cli.storage_paths import agent_root, deep_worker_home


# Claude-session import (DESIGN_IMPORT_CLAUDE_SESSION.md). The provider lives at
# <repo-root>/claude_code_sdk_provider/ and is NOT importable until the repo root is
# on sys.path — exactly the bootstrap ``tools.py`` does for web_search. A plain
# top-level ``from deep_worker.providers.claude_code_sdk_provider...`` import here breaks EVERY invocation
# of ``deep-worker`` (even ``--help``), because the CLI module is imported
# before any sys.path setup runs. Add the repo root first, THEN import, so the
# names exist as module globals (tests patch
# ``agent_command.parse_claude_transcript`` / ``resolve_transcript_path``).
_repo_root = Path(__file__).resolve().parents[4]  # .../<repo>/  (cli→cli_pkg→pkg→deep_worker→repo)
if (_repo_root / "claude_code_sdk_provider").exists() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


_SDK_PROVIDER = "deep_worker.providers.claude_code_sdk_provider"


class _ParseClaudeTranscript(Protocol):
    def __call__(self, path: str | Path) -> list[Any]: ...


class _ResolveTranscriptPath(Protocol):
    def __call__(self, uuid_or_path: str, cwd: str | None) -> Path: ...


def _load_claude_session_imports() -> tuple[
    _ParseClaudeTranscript | None,
    _ResolveTranscriptPath | None,
]:
    """Best-effort import of the claude-code-sdk transcript importer.

    ``claude_code_sdk_provider`` is a monorepo-only package (NOT vendored into the
    PyPI wheel). When it is genuinely absent, degrade gracefully: keep this
    module importable (``--help`` etc.) and let ``--import-claude`` report a
    clear error at use time. Only the missing-provider case
    (``ModuleNotFoundError`` on ``claude_code_sdk_provider`` itself) is swallowed;
    any other import defect of an installed/source provider — including a
    missing/renamed ``claude_code_sdk_provider.session_import`` submodule — is
    re-raised so it is not masked as "package absent".

    Returns:
        ``(parse_claude_transcript, resolve_transcript_path)``, or
        ``(None, None)`` when ``claude_code_sdk_provider`` is not installed.

    Raises:
        ModuleNotFoundError: an import defect OTHER than the missing
            ``claude_code_sdk_provider`` package itself.
    """
    try:
        from deep_worker.providers.claude_code_sdk_provider.session_import import (  # noqa: PLC0415
            parse_claude_transcript,
            resolve_transcript_path,
        )
    except ModuleNotFoundError as exc:
        # ``exc.name`` is the dotted name of the module that is missing: the
        # provider package itself, or a parent package when the source root
        # is not on sys.path. A missing submodule of a present provider is
        # still a defect and is re-raised.
        name = exc.name or ""
        if name != "claude_code_sdk_provider" and not (
            name == _SDK_PROVIDER or _SDK_PROVIDER.startswith(name + ".")
        ):
            raise
        return None, None
    return parse_claude_transcript, resolve_transcript_path


# Module globals, patched by tests (``agent_command.parse_claude_transcript``
# etc.). None when the monorepo-only provider is absent.
parse_claude_transcript, resolve_transcript_path = _load_claude_session_imports()


# ---------------------------------------------------------------------------
# Restart helpers
# ---------------------------------------------------------------------------

# Argparse defaults for detecting explicit CLI overrides.
# Must match the defaults in create_parser().
_PARSER_DEFAULTS = {
    "provider": "deepseek",
    "model": None,
    "dir": None,
    "skip_permissions": False,
    "task_manager": False,
    "clean": False,
    "memory_only": False,
    "use_sqlite": False,
    "history_memory": False,
    "auto_approve_send_message": False,
    "dev_tree": False,
    "web_search_policy": "auto",
    "preferred_provider": None,
    "no_update_preferred": False,
    "failover_chain": None,
    "fork": None,
    "fork_from": None,
    "context_limit": None,
    "hot_cache_ttl": None,
    "summary_keep": None,
}


_CONTEXT_LIMIT_RE = re.compile(r"^(\d+)([kKmM])?$")

# Sentinel returned by ``parse_context_limit`` for the literal "default"
# token. Surfaces to ``merge_restart_config`` as an explicit "operator
# said clear this" signal — distinct from the bare-restart case where
# ``args.context_limit is None`` simply because no ``-l`` was passed.
# ``build_launch_config`` converts the sentinel to ``None`` for the
# runtime path via :func:`_resolved_context_limit`. See
# ``test_restart_preserves_context_limit.py`` for the contract.
_CONTEXT_LIMIT_CLEAR_SENTINEL = "__context_limit_cleared__"


def _resolved_context_limit(value):
    """Convert the CLI value to what the runtime wants.

    * sentinel → ``None`` (operator explicitly cleared the limit).
    * ``None`` → ``None`` (bare restart, no ``-l`` on the line).
    * ``int`` → unchanged.

    Used by :func:`build_launch_config` so the sentinel never reaches
    the runtime — it only carries meaning along the CLI dataflow.
    """
    if value == _CONTEXT_LIMIT_CLEAR_SENTINEL:
        return None
    return value


def parse_context_limit(value: str) -> int | str:
    """Parse ``--context_limit`` / ``-l`` into token count or a clear sentinel.

    ``default`` (any case) returns the clear sentinel
    (see ``_CONTEXT_LIMIT_CLEAR_SENTINEL``). ``k``/``K`` = ×1_000,
    ``M``/``m`` = ×1_000_000. Floats, empty strings, unknown suffixes,
    zero and negatives are rejected so the stored value is reversible
    from the CLI token.
    """
    text = value.strip()
    if text.lower() == "default":
        return _CONTEXT_LIMIT_CLEAR_SENTINEL
    if not text:
        raise argparse.ArgumentTypeError(
            "context_limit must be a positive integer with optional k/M suffix, or 'default'"
        )
    match = _CONTEXT_LIMIT_RE.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid context_limit {value!r}: expected a positive integer "
            f"with optional k/M suffix, or 'default'"
        )
    number = int(match.group(1))
    if number == 0:
        raise argparse.ArgumentTypeError("context_limit must be greater than 0")
    suffix = match.group(2)
    if suffix in ("k", "K"):
        number *= 1_000
    elif suffix in ("m", "M"):
        number *= 1_000_000
    return number


def parse_hot_cache_ttl(value: str) -> tuple[str, int]:
    """Parse ``--hot-cache-ttl <sec>`` or ``--hot-cache-ttl <provider>=<sec>``.

    Bare seconds apply to every provider (key ``*``). ``0`` is always-cold.
    """
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError(
            "hot-cache-ttl must be <sec> or <provider>=<sec>"
        )
    if "=" in text:
        provider, _, rest = text.partition("=")
        provider = provider.strip().lower()
        rest = rest.strip()
    else:
        provider = "*"
        rest = text
    if not provider:
        raise argparse.ArgumentTypeError("hot-cache-ttl provider must be non-empty")
    if not rest.isdigit():
        raise argparse.ArgumentTypeError(
            f"invalid hot-cache-ttl {value!r}: expected a non-negative integer "
            f"or <provider>=<sec>"
        )
    return provider, int(rest)


def parse_summary_keep(value: str) -> tuple[str, int | float]:
    """Parse ``--summary-keep <spec>`` into a ``keep`` retention policy.

    Accepted forms::

        0        -> ("tokens", 0)      no tail at all, summary only
        10000    -> ("tokens", 10000)  token CEILING
        10k      -> ("tokens", 10000)  token CEILING
        10%      -> ("fraction", 0.10) share of the effective window
        0.10     -> ("fraction", 0.10) share of the effective window
        20m      -> ("messages", 20)   last N messages

    A token value is a ceiling applied against the default share (see
    ``resolve_summary_keep``); a fraction or message count replaces the
    default outright. ``0`` is exact — it is never min()-ed.
    """
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError(
            "summary-keep must be 0, <tokens>, <N>k, <N>%, 0.<N> or <N>m"
        )

    if text.endswith("%"):
        body = text[:-1].strip()
        try:
            percent = float(body)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid summary-keep {value!r}: expected a number before '%'"
            ) from None
        if not 0 < percent <= 100:
            raise argparse.ArgumentTypeError(
                f"invalid summary-keep {value!r}: percentage must be in (0, 100]"
            )
        return "fraction", percent / 100.0

    lowered = text.lower()
    if lowered.endswith("m"):
        body = lowered[:-1].strip()
        if not body.isdigit():
            raise argparse.ArgumentTypeError(
                f"invalid summary-keep {value!r}: expected <N>m for a message count"
            )
        count = int(body)
        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"invalid summary-keep {value!r}: message count must be > 0 "
                f"(use 0 for an empty tail)"
            )
        return "messages", count

    if lowered.endswith("k"):
        body = lowered[:-1].strip()
        if not body.isdigit():
            raise argparse.ArgumentTypeError(
                f"invalid summary-keep {value!r}: expected <N>k for a token ceiling"
            )
        return "tokens", int(body) * 1_000

    if lowered.isdigit():
        return "tokens", int(lowered)

    # Bare decimal in (0, 1] is a fraction: 0.10 == 10%.
    try:
        fraction = float(lowered)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid summary-keep {value!r}: expected 0, <tokens>, <N>k, "
            f"<N>%, 0.<N> or <N>m"
        ) from None
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError(
            f"invalid summary-keep {value!r}: fractional values must be in (0, 1]"
        )
    return "fraction", fraction


def resolve_summary_keep(spec, default_keep):
    """Combine an operator ``--summary-keep`` spec with the computed default.

    A token spec is a CEILING — it may only lower the default, never raise it
    (operator requirement: "no more than 10k" is an upper bound). An explicit
    ``("tokens", 0)`` is exact and is never min()-ed. Fraction and message
    specs replace the default outright.

    A token ceiling against a *fraction* default cannot be compared here (the
    window is unknown until the model resolves), so the ceiling is carried
    through as-is.
    """
    if spec is None:
        return default_keep
    kind, value = spec
    if kind != "tokens":
        return spec
    if value == 0:
        return "tokens", 0
    default_kind, default_value = default_keep
    if default_kind == "tokens":
        return "tokens", min(int(value), int(default_value))
    return "tokens", int(value)


def _normalize_summary_keep(value):
    """CLI tuple or saved list/tuple → ``(kind, value)`` or ``None``.

    Anything that is not a recognised spec is treated as "not configured"
    rather than raising. ``merge_restart_config`` reads this off an
    ``argparse.Namespace`` that callers (and a number of existing tests)
    build as a ``MagicMock``, where a missing attribute yields a mock rather
    than the ``getattr`` default — raising here would turn an unrelated
    restart into a hard failure. A genuinely malformed spec is rejected at
    parse time by ``parse_summary_keep`` (argparse ``type=``), which is the
    boundary that faces the operator.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return parse_summary_keep(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        kind = str(value[0]).strip().lower()
        try:
            if kind == "fraction":
                return "fraction", float(value[1])
            if kind in {"tokens", "messages"}:
                return kind, int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _normalize_hot_cache_ttl(value) -> dict[str, int]:
    """CLI list-of-pairs or saved dict → ``{provider: seconds}``."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(key).strip().lower(): int(sec) for key, sec in value.items()}
    out: dict[str, int] = {}
    for item in value:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out[str(item[0]).strip().lower()] = int(item[1])
        elif isinstance(item, str):
            key, sec = parse_hot_cache_ttl(item)
            out[key] = sec
        else:
            raise TypeError(f"invalid hot_cache_ttl entry: {item!r}")
    return out


# ---------------------------------------------------------------------------
# Persisted-boolean-options registry
# ---------------------------------------------------------------------------
#
# Single source of truth for the booleans that survive an agent restart.
#
# Both restart paths (CLI ``deep-worker --restart`` and web
# ``_spawn_subprocess``) must consult THIS list rather than maintain their own
# parallel hard-coded sequence of ``if config.get(...)`` lines — that opt-in
# pattern is what caused the 2026-06-05 ``-m`` regression and is structurally
# unable to detect "you added an option but forgot to preserve it".
#
# Coverage rule (enforced by ``tests/unit_tests/cli/test_boolean_flags_registry.py``):
#   Every boolean key in _PARSER_DEFAULTS MUST be either in _BOOLEAN_FLAGS
#   (preserved) or in _NON_PRESERVED_BOOLEAN_OPTIONS (explicitly excluded).
#
# Format: (dest, cli_flag_for_true, cli_flag_for_false_or_None)
#   - cli_flag_for_true: the form a respawn emits when the persisted value
#     is True (e.g. "-m").
#   - cli_flag_for_false: the --no-* counterpart, for CLI-side disabling at
#     restart time (consumed by create_parser() / _explicit_cli_flags()).
#
# See docs/DESIGN_RESTART_OPTION_PRESERVATION.md.
_BOOLEAN_FLAGS: tuple[tuple[str, str, str | None], ...] = (
    ("skip_permissions",          "-sp",           "--no-skip-permissions"),
    ("task_manager",              "-tm",           "--no-tm"),
    ("auto_approve_send_message", "-asm",          "--no-auto-approve-send-message"),
    ("memory_only",               "--memory-only", "--no-memory-only"),
    ("history_memory",            "-m",            "--no-m"),
    ("use_sqlite",                "--use-sqlite",  "--no-use-sqlite"),
    ("dev_tree",                  "--dev",         "--no-dev"),
)

# Booleans that are intentionally NOT preserved across restart. Each entry
# must have a written justification (one-shot intent, explicit per-launch
# action, etc.). Adding to this set is a documented decision, not a default.
_NON_PRESERVED_BOOLEAN_OPTIONS: frozenset[str] = frozenset({
    "clean",                # Explicit per-launch action; never a persistent mode.
    "no_update_preferred",  # One-shot failover override; not a persistent mode.
})


def normalize_model_arg(args: argparse.Namespace) -> None:
    """Treat an explicit ``--model unknown`` as 'no model specified'.

    ``"unknown"`` was never a real model name — it was the sentinel an
    older build wrote when a provider had no default (codex). That
    string leaked into ``cli_args`` / ``session_metadata.json`` and got
    replayed verbatim as a real ``--model unknown`` on relaunch
    (notably by the web restart path, which copies the saved model into
    the spawn command line). Codex then rejected it with HTTP 400.

    Normalising it to ``None`` at the single CLI chokepoint — right
    after ``parse_args`` — covers every launch path (direct,
    ``--restart``, web-spawned) so the provider default kicks back in.
    Case-insensitive and defensive against a namespace with no
    ``model`` attribute.

    Mutates ``args`` in place; returns nothing.
    """
    model = getattr(args, "model", None)
    if isinstance(model, str) and model.strip().lower() == "unknown":
        args.model = None


def apply_preferred_provider_override(args: argparse.Namespace) -> None:
    """Make ``-pp``/``--preferred-provider`` select the ACTIVE provider.

    ``--preferred-provider`` is documented to override both the long-term
    preference and the active provider. That override lives in
    ``merge_restart_config()``, which only runs when there is a saved
    config to merge — so on a FIRST launch it never ran and
    ``args.provider`` stayed at the parser default. ``deep-worker foo -pp
    codex`` therefore started on deepseek and persisted deepseek's default
    model, which the web UI then reported as the agent's model (operator
    report 2026-09-01).

    Applied right after ``parse_args`` for the same reason as
    ``normalize_model_arg``: ``args`` is repaired once, before any consumer
    (``AgentConfig``, the tool-selection preflight, ``build_launch_config``)
    reads it, so they cannot disagree about which provider is active.

    Two guards keep this confined to a genuine first launch:

    * ``--restart`` is excluded outright. Under a temporary failover
      ``provider`` and ``preferred_provider`` legitimately differ, and
      ``merge_restart_config`` has already decided which wins; re-applying
      the override would drag the agent back onto its preferred provider
      and undo the failover
      (``test_restart_preserves_preferred_model_under_failover``).
    * An explicit ``--provider`` wins, since that flag *is* the
      active-provider selector. Presence is read from ``argv`` rather than
      inferred from "the value equals the default", because ``deepseek``
      is both the default and a provider an operator may name outright.

    Mutates ``args`` in place; returns nothing.
    """
    if getattr(args, "restart", False):
        return
    preferred = getattr(args, "preferred_provider", None)
    if not preferred:
        return
    if "provider" in _explicit_cli_flags():
        return
    args.provider = preferred


def _get_agent_data_dir(name: str) -> Path:
    """Return the data directory for an agent.

    Post-v6 we prefer the canonical ``agents/<name>/`` layout, but for
    agents that haven't lazily migrated yet the launch_config loader
    also checks the legacy ``~/.deep_worker/<name>/`` location via
    ``read_agent_file`` (DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §4.5).
    """
    canonical = agent_root(name, create=False)
    if (canonical / "session_metadata.json").exists():
        return canonical
    legacy = Path.home() / ".deep_worker" / name
    if (legacy / "session_metadata.json").exists():
        return legacy
    return canonical  # caller will see "no metadata" anyway


def load_launch_config(name: str) -> dict | None:
    """Load launch_config from session_metadata.json.

    Returns:
        The launch_config dict, or None if not available.
    """
    data_dir = _get_agent_data_dir(name)
    metadata_file = data_dir / "session_metadata.json"

    if not metadata_file.exists():
        return None

    try:
        data = json.loads(metadata_file.read_text())
        return data.get("launch_config")
    except (json.JSONDecodeError, OSError):
        return None


def _explicit_cli_flags(argv: list[str] | None = None) -> set[str]:
    """Return the set of dest-names that were EXPLICITLY passed on the CLI.

    Override detection cannot rely on "value differs from the parser
    default": the parser default for ``--provider`` is ``"deepseek"``, so a
    user who runs ``--restart a --provider deepseek`` produces a value that
    equals the default and would be mistaken for "flag not passed" — the
    saved provider would silently win (observed in a past incident).

    To detect presence reliably we re-parse ``argv`` with a throwaway parser
    whose options default to ``argparse.SUPPRESS``. With SUPPRESS, a dest only
    appears in the namespace when the flag was actually given, so the set of
    present attributes IS the set of explicit overrides — independent of the
    value. ``argv`` defaults to ``sys.argv[1:]`` (the real invocation).
    """
    if argv is None:
        argv = sys.argv[1:]

    detector = argparse.ArgumentParser(add_help=False)
    detector.add_argument("name", nargs="?", default=argparse.SUPPRESS)
    detector.add_argument("--provider", default=argparse.SUPPRESS)
    detector.add_argument("--model", default=argparse.SUPPRESS)
    detector.add_argument("--dir", default=argparse.SUPPRESS)
    detector.add_argument("-sp", "--skip-permissions", dest="skip_permissions",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("-tm", "--task-manager", dest="task_manager",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("-asm", "--auto-approve-send-message",
                          dest="auto_approve_send_message",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("--clean", action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("--memory-only", dest="memory_only",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("-m", "--history-memory", dest="history_memory",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("--use-sqlite", dest="use_sqlite",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("--dev", dest="dev_tree",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("-pp", "--preferred-provider", dest="preferred_provider",
                          default=argparse.SUPPRESS)
    detector.add_argument("--no-update-preferred", dest="no_update_preferred",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("--failover-chain", dest="failover_chain",
                          default=argparse.SUPPRESS)
    detector.add_argument("--fork", dest="fork",
                          default=argparse.SUPPRESS)
    detector.add_argument("--from", dest="fork_from",
                          default=argparse.SUPPRESS)
    detector.add_argument("--turn-ending-tool", dest="turn_ending_tools",
                          action="append", default=argparse.SUPPRESS)
    detector.add_argument("--tools", dest="tools", default=argparse.SUPPRESS)
    detector.add_argument("-l", "--context_limit", dest="context_limit",
                          default=argparse.SUPPRESS)
    detector.add_argument("--hot-cache-ttl", dest="hot_cache_ttl",
                          action="append", default=argparse.SUPPRESS)
    detector.add_argument("--summary-keep", dest="summary_keep",
                          default=argparse.SUPPRESS)
    # Exact ``-ls`` so prefix matching cannot swallow it as ``-l`` + ``s``.
    detector.add_argument("-ls", "--list", dest="list_threads",
                          action="store_true", default=argparse.SUPPRESS)

    # --no-* counterparts (store_const with const=False, default=SUPPRESS).
    # These extend the real parser's --no-* flags so that passing e.g.
    # --no-skip-permissions puts "skip_permissions" into the explicit set.
    detector.add_argument("--no-skip-permissions", dest="skip_permissions",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    detector.add_argument("--no-task-manager", "--no-tm", dest="task_manager",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    detector.add_argument("--no-auto-approve-send-message",
                          dest="auto_approve_send_message",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    detector.add_argument("--no-memory-only", dest="memory_only",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    detector.add_argument("--no-history-memory", "--no-m", dest="history_memory",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    detector.add_argument("--no-use-sqlite", dest="use_sqlite",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    detector.add_argument("--no-dev", dest="dev_tree",
                          action="store_const", const=False, default=argparse.SUPPRESS)
    # Tri-state web-search policy — either explicit flag puts
    # "web_search_policy" into the explicit-override set.
    detector.add_argument("--enable-web-search", dest="web_search_policy",
                          action="store_const", const="enabled",
                          default=argparse.SUPPRESS)
    detector.add_argument("--disable-web-search", dest="web_search_policy",
                          action="store_const", const="disabled",
                          default=argparse.SUPPRESS)

    # parse_known_args: ignore every other flag the real parser owns.
    ns, _ = detector.parse_known_args(argv)
    return set(vars(ns).keys()) - {"name"}


def _warn_model_reset(
    reason: str,
    provider: str | None,
    dropped: str | None,
    replacement: str | None,
) -> None:
    """Tell the operator, exactly once, that a model was dropped.

    A silent substitution is the whole defect: the operator ran
    ``--model fable`` and got ``sonnet`` with nothing on screen. Any transition
    that DISCARDS a model value must pass through here.

    Emission is a single ``print`` to stderr. There is deliberately no
    accompanying ``logger.warning``: in a normal CLI invocation the root logger
    already has a stderr StreamHandler, so doing both prints the same sentence
    twice. Note that ``merge_restart_config`` runs in the foreground restart
    invoker, *before* the daemon installs its stderr redirection — so this
    reaches the operator's terminal, but is not thereby guaranteed to land in
    the restarted agent's ``.err.log``.

    Args:
        reason: Machine-greppable cause. One of ``unknown-alias`` (the
            provider does not recognise the token), ``stale-provider`` (the
            saved model belonged to a different provider),
            ``provider-declared`` (operator named a provider without a model),
            ``preferred-reset-after-invalid-explicit`` (a *valid* preferred
            model cleared because the explicit replacement was invalid).
        provider: The provider the model was being applied to.
        dropped: The value being discarded. Falsy ⇒ nothing happened, no output.
        replacement: What will actually be used instead, as resolved by
            :func:`get_default_model_name`. ``None`` means the provider pins
            no default and its CLI will choose.
    """
    if not dropped:
        return
    repl_txt = (
        "provider default (resolved by the provider CLI)"
        if replacement is None
        else repr(replacement)
    )
    print(
        f"[deep-worker] WARNING: model {dropped!r} was reset "
        f"(reason={reason}, provider={provider!r}); "
        f"using {repl_txt} instead.",
        file=sys.stderr,
        flush=True,
    )


def _emit_tool_selection_preflight(provider: str, spec: list[str]) -> None:
    """Disclose un-suppressable provider natives on the TERMINAL.

    A restricted run is a measurement, and the user has to learn that the
    provider's own built-ins survived the restriction while they can still
    read it — i.e. BEFORE ``daemonize()`` sends every later byte to a log
    file nobody opens. The logger warning inside ``create_cli_agent``
    remains, but it lands post-detach and so cannot be the only disclosure.

    Silent by construction when nothing is restricted, so the default
    launch prints nothing at all.
    """
    caveat = native_tool_caveat(provider, parse_tool_selection(spec))
    if caveat:
        print(f"WARNING: {caveat}", file=sys.stderr, flush=True)


def _preflight_spec(value) -> list[str]:
    """Normalise a ``--tools`` value (CLI string or saved list) to a spec."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def resolve_tool_spec(args: argparse.Namespace) -> list[str]:
    """The tool spec this launch actually runs with.

    ``--tools`` states the selection outright; ``--tools-add`` and
    ``--tools-remove`` layer onto it. Both must reach the AgentConfig the
    process is BUILT from and the launch_config it is RESTARTED from, or
    the two disagree — which is not academic: resolving the increments for
    ``launch_config`` alone let ``deep-worker probe --tools-rm 'lsp_*'``
    reach ``idle`` with a pattern matching no tool, because the unknown-name
    check runs against the AgentConfig spec and never saw the rule
    (observed on a test probe). One owner for the rule,
    per POLICIES §14.

    On ``--restart`` the increments are already folded into ``args.tools``
    by ``merge_restart_config``, which then clears the two flags — so this
    is a no-op there rather than a second application.
    """
    return merge_increments(
        parse_tool_selection_spec(getattr(args, "tools", None)),
        parse_increment_spec(
            getattr(args, "tools_add", None),
            getattr(args, "tools_remove", None),
        ),
    )


def _emit_effective_tool_selection_preflight(args: argparse.Namespace) -> None:
    """Disclose the effective selection before the process daemonizes."""
    config_path = getattr(args, "config", None)
    if config_path:
        config = AgentConfig.from_file(config_path)
        _emit_tool_selection_preflight(config.provider, config.tools)
        return
    # The increments are part of the effective selection, so a preflight
    # that read only ``--tools`` would understate what the agent runs with.
    _emit_tool_selection_preflight(args.provider, resolve_tool_spec(args))


def merge_restart_config(
    saved: dict,
    args: argparse.Namespace,
    explicit: set[str] | None = None,
) -> argparse.Namespace:
    """Merge saved launch_config with CLI args.

    Saved values are used as defaults. A CLI flag overrides the saved value
    when it was EXPLICITLY passed on the command line (see ``explicit``) OR
    when its value differs from the parser default. Relying on "differs from
    default" alone is unsafe for ``--provider``, whose default is a real
    provider name (``"deepseek"``): an explicit ``--provider deepseek`` would
    otherwise be indistinguishable from "not passed" and silently dropped
    (observed in a past incident).

    When the provider is overridden but ``--model`` is NOT explicitly given,
    the saved model is NOT inherited (it belongs to the OLD provider and would
    boot the new provider on an incompatible model). It is reset to ``None`` so
    ``build_launch_config`` resolves the new provider's default.

    ``--clean`` is NEVER inherited from saved config.

    Args:
        saved: The launch_config dict from session_metadata.json.
        args: Parsed CLI arguments (may contain overrides).
        explicit: Set of dest-names explicitly passed on the CLI. When None,
            it is derived from ``sys.argv`` via ``_explicit_cli_flags()``.

    Returns:
        New argparse.Namespace with effective values.
    """
    if explicit is None:
        explicit = _explicit_cli_flags()

    def _is_override(dest: str, value, default) -> bool:
        """A flag overrides saved config if it was explicitly passed,
        or (for backward compatibility) if its value differs from default."""
        return dest in explicit or value != default

    effective = argparse.Namespace(**vars(args))

    # Apply saved values as defaults
    effective.provider = saved.get("provider", _PARSER_DEFAULTS["provider"])
    effective.model = saved.get("model", _PARSER_DEFAULTS["model"])
    # Heal legacy poison: older builds persisted an unspecified model as
    # the literal "unknown" (observed in a past incident). Treat that
    # as "no model specified" so the provider default kicks back in on
    # restart instead of forwarding the bad string to the API.
    if effective.model == "unknown":
        effective.model = None
    effective.dir = Path(saved["working_dir"]) if saved.get("working_dir") else _PARSER_DEFAULTS["dir"]
    effective.skip_permissions = saved.get("skip_permissions", False)
    effective.task_manager = saved.get("task_manager", False)
    effective.memory_only = saved.get("memory_only", False)
    effective.history_memory = saved.get("history_memory", False)
    effective.use_sqlite = saved.get("use_sqlite", False)
    effective.auto_approve_send_message = saved.get(
        "auto_approve_send_message", False
    )
    effective.dev_tree = saved.get("dev_tree", False)
    effective.web_search_policy = saved.get(
        "web_search_policy", _PARSER_DEFAULTS["web_search_policy"]
    )
    # Failover chain name. MUST be restored from the saved config: every
    # automatic failover restarts the agent, so a chain that reverted to
    # "default" here would silently undo the operator's choice on the first
    # usage limit — the failure mode that wiped preferred_model in 2026-08-14.
    effective.failover_chain = saved.get("failover_chain") or "default"
    effective.clean = False  # NEVER inherit --clean

    # External MCP CLI overrides — restore from saved launch_config so a
    # restart reuses them (DESIGN_EXTERNAL_MCP_TOOLS.md §3.7).
    effective.mcp_config_paths = list(saved.get("mcp_config_paths") or [])
    effective.mcp_server_inline = list(saved.get("mcp_server_inline") or [])

    # Turn-ending tool declarations — same restore semantics as the MCP
    # lists above (see DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md §3.3).
    effective.turn_ending_tools = list(saved.get("turn_ending_tools") or [])
    # Tool selection — restored as the raw spec in the CLI's own
    # comma-separated form, so a restricted agent restarts restricted.
    _saved_tools = list(saved.get("tools") or [])
    effective.tools = ",".join(_saved_tools) if _saved_tools else None
    effective.hot_cache_ttl = _normalize_hot_cache_ttl(saved.get("hot_cache_ttl"))
    effective.summary_keep = _normalize_summary_keep(saved.get("summary_keep"))

    # Override with explicitly provided CLI args
    provider_overridden = _is_override(
        "provider", args.provider, _PARSER_DEFAULTS["provider"]
    )
    if provider_overridden:
        effective.provider = args.provider

    model_explicit = _is_override("model", args.model, _PARSER_DEFAULTS["model"])
    if model_explicit:
        effective.model = args.model
    # NOTE: the explicit-provider-declaration model reset used to live here and
    # in the pp-block below, each guarded by ``effective.provider != saved``.
    # Both are superseded by a single consolidated gate placed AFTER the pp-block
    # finalizes ``effective.provider`` — see ``provider_decl_unpinned`` below.
    if _is_override("dir", args.dir, _PARSER_DEFAULTS["dir"]):
        effective.dir = args.dir
    if _is_override("skip_permissions", args.skip_permissions,
                    _PARSER_DEFAULTS["skip_permissions"]):
        effective.skip_permissions = args.skip_permissions
    if _is_override("task_manager", args.task_manager,
                    _PARSER_DEFAULTS["task_manager"]):
        effective.task_manager = args.task_manager
    if _is_override("clean", args.clean, _PARSER_DEFAULTS["clean"]):
        effective.clean = args.clean
    if _is_override("memory_only", args.memory_only,
                    _PARSER_DEFAULTS["memory_only"]):
        effective.memory_only = args.memory_only
    if _is_override("history_memory", getattr(args, "history_memory", False),
                    _PARSER_DEFAULTS["history_memory"]):
        effective.history_memory = args.history_memory
    if _is_override("use_sqlite", args.use_sqlite,
                    _PARSER_DEFAULTS["use_sqlite"]):
        effective.use_sqlite = args.use_sqlite
    if _is_override(
        "auto_approve_send_message",
        getattr(args, "auto_approve_send_message",
                _PARSER_DEFAULTS["auto_approve_send_message"]),
        _PARSER_DEFAULTS["auto_approve_send_message"],
    ):
        effective.auto_approve_send_message = args.auto_approve_send_message
    if _is_override(
        "dev_tree",
        getattr(args, "dev_tree", _PARSER_DEFAULTS["dev_tree"]),
        _PARSER_DEFAULTS["dev_tree"],
    ):
        effective.dev_tree = args.dev_tree
    if _is_override(
        "web_search_policy",
        getattr(args, "web_search_policy", _PARSER_DEFAULTS["web_search_policy"]),
        _PARSER_DEFAULTS["web_search_policy"],
    ):
        effective.web_search_policy = args.web_search_policy
    if _is_override(
        "failover_chain",
        getattr(args, "failover_chain", _PARSER_DEFAULTS["failover_chain"]),
        _PARSER_DEFAULTS["failover_chain"],
    ):
        effective.failover_chain = args.failover_chain
    mcp_config_paths = list(getattr(args, "mcp_config_paths", None) or [])
    mcp_server_inline = list(getattr(args, "mcp_server_inline", None) or [])
    if _is_override("mcp_config_paths", mcp_config_paths, []):
        effective.mcp_config_paths = mcp_config_paths
    if _is_override("mcp_server_inline", mcp_server_inline, []):
        effective.mcp_server_inline = mcp_server_inline
    # Passing --turn-ending-tool on --restart REPLACES the saved list
    # wholesale (no set-add semantics); passing nothing INHERITS it.
    turn_ending_tools = list(getattr(args, "turn_ending_tools", None) or [])
    if _is_override("turn_ending_tools", turn_ending_tools, []):
        effective.turn_ending_tools = turn_ending_tools
    # Passing --tools on --restart REPLACES the saved selection; an
    # explicit empty string CLEARS it (the only way back to an
    # unrestricted agent). Detection must be presence-based: by value,
    # ``--tools ''`` is indistinguishable from an absent flag.
    if "tools" in explicit:
        effective.tools = getattr(args, "tools", None)
    # ``--tools-add``/``--tools-remove`` are INCREMENTAL: they layer onto
    # whatever selection is in force — the saved one, or the fresh
    # ``--tools`` above when both are given — instead of restating it. That
    # is the whole point of the flags: adding one tool to a restricted agent
    # must not require listing the tools it already has.
    _increments = parse_increment_spec(
        getattr(args, "tools_add", None), getattr(args, "tools_remove", None)
    )
    if _increments:
        _base = parse_tool_selection_spec(effective.tools)
        effective.tools = ",".join(merge_increments(_base, _increments))
    # Consumed above: the merged result now lives in ``effective.tools``.
    # ``build_launch_config`` reads these two off the same namespace, so
    # leaving them set would apply the increment a second time — and
    # leaving them UNSET here would let them fall through to the bare CLI
    # namespace, the exact dataflow that wiped mcp_config_paths and
    # turn_ending_tools in a past incident.
    effective.tools_add = None
    effective.tools_remove = None

    # context_limit needs a special "explicit clear" path because the
    # argparse default is also ``None`` — bare ``--restart`` looks
    # identical to ``--restart -l default`` post-parse, but the
    # operator's intent differs ("inherit" vs "clear back to default").
    # ``parse_context_limit`` returns the sentinel
    # ``_CONTEXT_LIMIT_CLEAR_SENTINEL`` for the literal ``"default"``
    # token (see ``parse_context_limit``); if we see the sentinel on
    # the override branch, the operator explicitly asked to clear the
    # saved value. On all non-sentinel overrides the value wins, and
    # with no override at all the saved value is inherited. See
    # ``test_restart_preserves_context_limit.py``.
    saved_cl = saved.get("context_limit")
    args_cl = getattr(args, "context_limit", _PARSER_DEFAULTS["context_limit"])
    if args_cl == _CONTEXT_LIMIT_CLEAR_SENTINEL:
        effective.context_limit = None
    elif _is_override(
        "context_limit",
        args_cl,
        _PARSER_DEFAULTS["context_limit"],
    ):
        effective.context_limit = args_cl
    else:
        effective.context_limit = saved_cl

    args_ttl = getattr(args, "hot_cache_ttl", _PARSER_DEFAULTS["hot_cache_ttl"])
    if _is_override(
        "hot_cache_ttl",
        args_ttl,
        _PARSER_DEFAULTS["hot_cache_ttl"],
    ):
        effective.hot_cache_ttl = _normalize_hot_cache_ttl(args_ttl)

    args_keep = getattr(args, "summary_keep", _PARSER_DEFAULTS["summary_keep"])
    if _is_override(
        "summary_keep",
        args_keep,
        _PARSER_DEFAULTS["summary_keep"],
    ):
        effective.summary_keep = _normalize_summary_keep(args_keep)

    # ------------------------------------------------------------------
    # preferred_provider logic
    # ------------------------------------------------------------------
    # Backfill: agents created before preferred_provider existed get
    # preferred_provider = their current provider (first-run defaulting).
    # This runs once and is sticky thereafter.
    saved_pp = saved.get("preferred_provider")
    if saved_pp:
        effective.preferred_provider = saved_pp
    else:
        effective.preferred_provider = saved.get("provider", _PARSER_DEFAULTS["provider"])

    # --preferred-provider X: override BOTH preferred_provider AND provider.
    # The operator is explicitly setting a new long-term preference, so
    # the active provider follows.
    pp_explicit = _is_override(
        "preferred_provider",
        getattr(args, "preferred_provider", _PARSER_DEFAULTS["preferred_provider"]),
        _PARSER_DEFAULTS["preferred_provider"],
    )
    if pp_explicit:
        effective.preferred_provider = args.preferred_provider
        effective.provider = args.preferred_provider

    # ------------------------------------------------------------------
    # Consolidated model reset (operator rule, incident 2026-06-29)
    # ------------------------------------------------------------------
    # When the operator explicitly declares a provider (via ``-pp`` OR
    # ``--provider``) but gives no ``--model``, the saved model is taken to be
    # stale and MUST be reset to None so ``build_launch_config()`` resolves the
    # declared provider's own default. The old code only fired this reset when
    # the provider STRING changed (``effective.provider != saved.provider``),
    # which left an already-poisoned same-provider state (``-pp codex`` over
    # saved codex + cross-provider model) untouched. The categorical operator
    # rule — "no explicit --model ⇒ provider default" — supersedes that gate,
    # and same-provider re-declaration is the only way an operator can heal
    # a poisoned config without supplying a specific model.
    # ``--no-update-preferred`` is the internal failover/restore-on-idle path;
    # the operator did not type this provider switch — TaskManager did — and
    # the whole point of failover return is to RESTORE the operator's saved
    # ``preferred_model``. So we exclude that case from the consolidated reset
    # gate: only an OPERATOR-driven explicit provider declaration is treated
    # as "use that provider's default".
    no_update_preferred = bool(getattr(args, "no_update_preferred", False))
    provider_decl_unpinned = (
        (pp_explicit or provider_overridden)
        and not model_explicit
        and not no_update_preferred
    )
    from deep_worker_cli.config import (
        get_default_model_name,
        model_can_belong_to_provider,
    )

    # Why a model was dropped, or None if nothing was dropped. Consumed by the
    # preferred_model cascade below, which cannot re-derive the intent from
    # state that has already been mutated.
    model_reset_cause: str | None = None

    # T1: operator declared a provider without --model.
    switched_unpinned = (
        provider_overridden
        and effective.provider != saved.get("provider")
        and not model_explicit
    )
    if provider_decl_unpinned or switched_unpinned:
        if effective.model:
            _warn_model_reset(
                "provider-declared",
                effective.provider,
                effective.model,
                get_default_model_name(effective.provider),
            )
            model_reset_cause = "provider-declared"
        effective.model = None

    # T2: the model belongs to another provider, or the provider does not
    # recognise the token.
    saved_model_provider = saved.get("model_provider")
    provider_mismatch = bool(
        isinstance(saved_model_provider, str)
        and saved_model_provider
        and saved_model_provider != effective.provider
    )
    if model_explicit:
        # ``saved_model_provider`` describes the PREVIOUS token and says
        # nothing about the one the operator just typed. Attributing it would
        # report a typo'd --model as a provider mismatch and send the operator
        # debugging the wrong thing.
        provider_mismatch = False
    if effective.model and (
        provider_mismatch
        or not model_can_belong_to_provider(effective.provider, effective.model)
    ):
        reason = "stale-provider" if provider_mismatch else "unknown-alias"
        _warn_model_reset(
            reason,
            effective.provider,
            effective.model,
            get_default_model_name(effective.provider),
        )
        model_reset_cause = reason
        effective.model = None

    # --no-update-preferred: keep preferred_provider intact even when
    # provider is overridden. Used by automatic failover so that a
    # temporary provider switch does not overwrite the operator's
    # long-term preference.
    if getattr(args, "no_update_preferred", False):
        # Restore preferred to saved (undo any provider-change side effect)
        effective.preferred_provider = saved.get("preferred_provider") or saved.get(
            "provider", _PARSER_DEFAULTS["provider"]
        )

    # preferred_model: operator's model on preferred_provider (for restore-on-idle).
    saved_pp = effective.preferred_provider
    saved_provider = saved.get("provider", _PARSER_DEFAULTS["provider"])
    # When ``-pp`` moved the preferred provider to a new one, any saved
    # preferred_model belonged to the OLD preferred provider and is stale —
    # do not carry it over (incident 2026-06-29). The new preferred provider
    # resolves its own default unless an explicit --model was given.
    pp_switched = pp_explicit and args.preferred_provider != saved.get(
        "preferred_provider", saved_provider
    )
    if provider_decl_unpinned:
        # RC-1 (review): on ANY explicit provider declaration without
        # --model, preferred_model must track the (None) reset of model.
        # Otherwise a stale ``preferred_model`` survives same-provider ``-pp``
        # and would re-pin the foreign model later via the restore-on-idle /
        # failover path. ``effective.model`` was just reset to None above.
        effective.preferred_model = effective.model
    elif (
        model_explicit
        and model_reset_cause == "unknown-alias"
        and effective.provider == effective.preferred_provider
    ):
        # The operator asked to REPLACE the model on their preferred provider
        # and the new token was invalid. Falling through to the branches below
        # would restore ``saved["preferred_model"]`` — silently resurrecting
        # the very preference they were replacing. Collapse both to the
        # provider default instead.
        #
        # The provider == preferred_provider guard matters: when a temporary
        # failover has moved us onto a different provider, a model dropped
        # here says nothing about the preference belonging to the preferred
        # provider, which must survive untouched (design C3).
        saved_preferred = saved.get("preferred_model")
        if saved_preferred:
            # NOT ``unknown-alias``: the preferred model itself may be
            # perfectly valid (e.g. ``opus``). It is cleared because the
            # explicit replacement failed, and the message must say so.
            _warn_model_reset(
                "preferred-reset-after-invalid-explicit",
                effective.preferred_provider,
                saved_preferred,
                get_default_model_name(effective.preferred_provider),
            )
        effective.preferred_model = None
    elif pp_switched and not model_explicit:
        effective.preferred_model = effective.model
    elif saved.get("preferred_model") and model_can_belong_to_provider(
        effective.preferred_provider, saved.get("preferred_model")
    ):
        effective.preferred_model = saved.get("preferred_model")
    elif (
        saved_pp
        and saved_provider == saved_pp
        and saved.get("model")
        and model_can_belong_to_provider(saved_pp, saved.get("model"))
    ):
        effective.preferred_model = saved.get("model")
    else:
        effective.preferred_model = None

    if getattr(args, "no_update_preferred", False) and provider_overridden:
        target = effective.provider
        if (
            target == effective.preferred_provider
            and effective.preferred_model
            and model_can_belong_to_provider(target, effective.preferred_model)
        ):
            # Restore-on-idle / failover return: keep opus (etc.), not provider default.
            if not model_explicit:
                effective.model = effective.preferred_model

    if effective.provider == effective.preferred_provider and effective.model:
        effective.preferred_model = effective.model

    return effective


def build_launch_config(args: argparse.Namespace) -> dict:
    """Build a launch_config dict from effective argparse namespace.

    Args:
        args: Namespace with effective (merged) values.

    Returns:
        Dict suitable for storage in session_metadata.json.
    """
    from deep_worker_cli.config import get_default_model_name

    provider = args.provider
    preferred_provider = getattr(
        args, "preferred_provider", _PARSER_DEFAULTS["preferred_provider"]
    )

    model = args.model
    if model is None:
        # Resolve the provider default. For providers with no pinned
        # default (codex), this stays None and MUST be persisted as
        # None — the codex CLI then picks its own current default. Do
        # NOT coerce to "unknown": that string would be read back on
        # restart and forwarded to CodexChatModel, which the API
        # rejects with HTTP 400 (observed in a past incident).
        model = get_default_model_name(provider)

    working_dir = args.dir
    if working_dir is None:
        original_cwd = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
        working_dir = Path(original_cwd) if original_cwd else Path.cwd()

    # dev_tree may come from either the parsed flag or, after the bash
    # launcher's re-exec (which strips --dev from argv), from the
    # DEEP_WORKER_DEV_TREE=1 env var. Honor either source. Only the
    # exact string "1" counts as truthy — '0', 'false', '' do not.
    #
    # THIRD source (fix 2026-06-18): the code actually running. When an
    # agent is launched DIRECTLY from the dev checkout (e.g.
    # ``python3 -m deep_worker_cli.cli.agent_command <name> --provider grok``
    # without going through the bash launcher / without ``--dev``), neither
    # of the above is set, so ``dev_tree`` would persist as False even though
    # ``tree_root`` (==``_repo_root``) IS the dev tree. A web-UI restart then
    # reconstructs the spawn on the PROD tree (``_spawn_subprocess`` keys off
    # ``dev_tree``), where a dev-only provider like ``grok`` is unknown to the
    # argparse ``--provider`` choices → the agent dies at startup and never
    # comes back (observed in a past incident). Deriving dev_tree from the
    # real code tree keeps the persisted flag consistent with reality, so the
    # flag never lies about which tree must be used on restart. The canonical
    # dev tree path mirrors the bash launcher's DEV_TREE (override honored for
    # tests).
    _dev_tree_root = Path(
        os.environ.get(
            "DEEP_WORKER_DEV_TREE_ROOT_OVERRIDE", "/nonexistent/dev-tree"
        )
    )
    _running_from_dev_tree = _repo_root.resolve() == _dev_tree_root.resolve()
    dev_tree = (
        getattr(args, "dev_tree", False)
        or (os.environ.get("DEEP_WORKER_DEV_TREE") == "1")
        or _running_from_dev_tree
    )

    return {
        "provider": provider,
        "model": model,
        "model_provider": provider if model else None,
        "model_source": "configured" if model else "provider_default",
        "working_dir": str(working_dir),
        "skip_permissions": args.skip_permissions,
        "task_manager": args.task_manager,
        "memory_only": args.memory_only,
        "history_memory": getattr(args, "history_memory", False),
        "use_sqlite": args.use_sqlite,
        "clean": args.clean,
        "auto_approve_send_message": getattr(
            args, "auto_approve_send_message", False
        ),
        "dev_tree": dev_tree,
        "web_search_policy": getattr(args, "web_search_policy", "auto"),
        "preferred_provider": preferred_provider or provider,
        "preferred_model": (
            model
            if provider == (preferred_provider or provider)
            else getattr(args, "preferred_model", None)
        ),
        # External MCP CLI overrides — persisted so that --restart preserves them.
        "mcp_config_paths": list(getattr(args, "mcp_config_paths", None) or []),
        "mcp_server_inline": list(getattr(args, "mcp_server_inline", None) or []),
        # Turn-ending tools are persisted as the RAW CLI strings, not as
        # parsed declarations: the format is stable, readable in
        # session_metadata.json, and re-parsed through the same code path
        # on every start, so first launch and --restart cannot diverge.
        "turn_ending_tools": list(getattr(args, "turn_ending_tools", None) or []),
        # Tool selection, persisted for the same reason and in the same
        # form: the raw spec, re-parsed on every start, so a restricted
        # agent comes back restricted after --restart.
        "tools": resolve_tool_spec(args),
        "context_limit": _resolved_context_limit(getattr(
            args, "context_limit", _PARSER_DEFAULTS["context_limit"]
        )),
        "hot_cache_ttl": _normalize_hot_cache_ttl(getattr(
            args, "hot_cache_ttl", _PARSER_DEFAULTS["hot_cache_ttl"]
        )),
        "summary_keep": _normalize_summary_keep(getattr(
            args, "summary_keep", _PARSER_DEFAULTS["summary_keep"]
        )),
        # Name of this agent's failover chain in
        # ~/.deep_worker/failover_chains.json. Absent/empty reads as
        # "default", so every pre-existing agent keeps today's behaviour
        # without a migration pass.
        "failover_chain": (
            getattr(args, "failover_chain", None)
            or _PARSER_DEFAULTS["failover_chain"]
            or "default"
        ),
        "storage_version": 6,
        "tree_root": str(_repo_root),
        "started_at": datetime.now().isoformat(),
        "pid": os.getpid(),
        "cli_args": " ".join(sys.argv[1:]),
    }


async def restart_agent(args: argparse.Namespace) -> int:
    """Handle --restart command.

    Loads saved launch_config, merges with CLI overrides,
    stops running agent if needed, then launches.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    from deep_worker_cli.process.agent_stop import stop_agent
    from deep_worker_cli.process.registry import ProcessRegistry

    name = args.name
    if not name:
        print("Error: Agent name is required with --restart", file=sys.stderr)
        return 1

    # 1. Load saved config
    saved = load_launch_config(name)
    if saved is None:
        print(
            f"Error: Cannot restart agent '{name}' — no previous launch configuration found.\n"
            f"Start the agent normally first:  deep-worker {name} --provider ... --model ...",
            file=sys.stderr,
        )
        return 1

    # 2. Merge with CLI overrides
    effective = merge_restart_config(saved, args)

    # 3. Stop running agent if needed.
    #
    # `force=True` makes stop_agent escalate SIGTERM -> SIGKILL instead of
    # giving up. Before that, a shutdown slower than the budget (in a past
    # incident: a 644 MB threads.db flushing on close) raised TimeoutError,
    # which this function turned into `return 1` — so step 5 never ran. The
    # old process died moments later anyway and nothing replaced it, leaving
    # the agent DOWN. That is strictly worse than not running --restart.
    registry = ProcessRegistry()
    existing = registry.lookup(name)
    if existing:
        try:
            data_dir = _get_agent_data_dir(name)
            await stop_agent(
                name,
                registry,
                data_dir=data_dir,
                force=True,
                # Set only by the ``restart_self`` tool. Waives the
                # "actively running" guard for a self-initiated restart —
                # the agent is always mid-turn when it calls the tool, so
                # without this the restart is refused and the tool has to
                # fall back to killing its own process, which destroys the
                # in-process MCP server and makes the CLI fabricate a
                # "user rejected" tool result. See
                # DESIGN_RESTART_SELF_FALSE_DENIAL.md.
                #
                # The flag is only a claim; the token is what authorizes it.
                # stop_agent verifies the token against the one the agent
                # minted in its own private dir and consumes it, so an
                # external --self-requested cannot bypass the guard.
                self_requested=getattr(args, "self_requested", False),
                self_requested_token=getattr(
                    args, "self_requested_token", None
                ),
            )
        except RuntimeError as e:
            # status=running — the agent is mid-turn. Unchanged contract.
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except TimeoutError as e:
            # Survived SIGKILL (uninterruptible I/O). Deliberately abort
            # rather than continue to step 5: a second process on the same
            # data directory means two writers on one threads.db and a
            # registry entry that describes neither. A DOWN agent is
            # recoverable; a corrupted checkpoint database is not.
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # 4. Handle --clean flag (if explicitly passed)
    if effective.clean:
        from deep_worker_cli.sessions import clean_agent_history
        try:
            await clean_agent_history(name)
        except Exception as e:
            print(f"Error cleaning agent history: {e}", file=sys.stderr)
            return 1

    # 5. Delegate to run_agent with the FULL merged namespace.
    #
    # ⚠ DO NOT reintroduce a field-by-field copy-back of `effective` into
    # `args`. That is exactly the pattern that silently wiped SIX
    # persisted inputs of build_launch_config on every --restart
    # (observed in a past incident, see
    # docs/DESIGN_RESTART_PRESERVES_LAUNCH_CONFIG.md):
    #   mcp_config_paths, mcp_server_inline, turn_ending_tools → []
    #   web_search_policy                                      → "auto"
    #   dev_tree, preferred_model  → masked (build re-derives them from
    #       _repo_root / model respectively), so the loss surfaces only
    #       in the failover shape provider != preferred_provider.
    # Every persisted field a future contributor adds needs another line
    # here, and one omission silently regresses to the parser default.
    #
    # `merge_restart_config` already produced a Namespace that is a strict
    # superset of `args` (it starts from `argparse.Namespace(**vars(args))`
    # and only overrides). `run_agent` and `build_launch_config` read the
    # SAME attribute names, so passing `effective` directly gives every
    # existing AND future persisted field the correct value with no
    # per-field code. Two tests pin this against a future "let me restore
    # clarity" refactor: `test_no_field_by_field_copy_back_in_restart_agent`
    # (no copy-back reappears) and
    # `test_every_build_launch_config_input_is_handled_by_merge_restart_config`
    # (every args attribute build_launch_config reads is deliberately
    # assigned on `effective`).
    effective.clean = False  # NEVER inherit --clean (already documented above)
    return await run_agent(effective)


async def do_fork(args: argparse.Namespace) -> int:
    """Handle --fork command.

    Forks (snapshot-copies) an existing agent's data directory and starts
    the new agent with its own history.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 success, 1 error).
    """
    from deep_worker_cli.fork_agent import fork_agent_data as _fork_data

    new_name = args.name
    old_name = args.fork

    if not new_name:
        print("Error: Agent name is required with --fork", file=sys.stderr)
        return 1

    if not old_name:
        print("Error: --fork requires a source agent name", file=sys.stderr)
        return 1

    # --fork and --clean are incompatible
    if getattr(args, "clean", False):
        print(
            "Error: --clean and --fork are incompatible. "
            "A fork creates a new agent with its own data; "
            "cleaning the source is a separate operation.",
            file=sys.stderr,
        )
        return 1

    from_msg = getattr(args, "fork_from", None)
    if from_msg:
        print(f"Forking with --from {from_msg} (truncate at message_id).")

    # Resolve paths for pre-checks
    src_dir = _get_agent_data_dir(old_name)
    dst_dir = _get_agent_data_dir(new_name)

    if not src_dir.is_dir():
        print(
            f"Error: Source agent '{old_name}' not found at {src_dir}.\n"
            f"Start the source agent first:  deep-worker {old_name} --provider ... --model ...",
            file=sys.stderr,
        )
        return 1

    if dst_dir.exists():
        print(
            f"Error: Agent '{new_name}' already exists at {dst_dir}.\n"
            f"Choose a different name or remove it first.",
            file=sys.stderr,
        )
        return 1

    # Load saved launch_config from source
    saved = load_launch_config(old_name)
    if saved is None:
        print(
            f"Error: Source agent '{old_name}' has no launch configuration.\n"
            f"Start the source agent first:  deep-worker {old_name} --provider ... --model ...",
            file=sys.stderr,
        )
        return 1

    # Merge CLI overrides (same logic as --restart)
    effective = merge_restart_config(saved, args)

    # Build a complete launch_config via build_launch_config() (unlike
    # run_agent()'s incomplete inline dict which misses dev_tree and
    # preferred_provider — see DESIGN_FORK_AGENT.md ISSUE 2).
    launch_config = build_launch_config(effective)
    launch_config["name"] = new_name

    # Fork the data
    try:
        _fork_data(old_name, new_name, message_id=from_msg)
    except Exception as e:
        # Partial copy cleanup
        shutil.rmtree(dst_dir, ignore_errors=True)
        print(f"Error during fork: {e}", file=sys.stderr)
        return 1

    # Write the merged launch_config to the new agent's metadata.
    # We overwrite what fork_agent_data wrote (which copied the old
    # metadata) with the fully merged config.
    meta_path = dst_dir / "session_metadata.json"
    launch_config["cli_args"] = " ".join(sys.argv[1:])
    meta_data = {
        "status": "idle",
        "tasks_completed": 0,
        "message_count": 0,
        "error_message": None,
        "last_activity": "",
        "storage_version": launch_config.get("storage_version", 5),
        "launch_config": launch_config,
    }
    meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False))
    os.chmod(meta_path, 0o600)

    print(f"Forked agent '{new_name}' from '{old_name}'.")

    # Apply effective args and delegate to run_agent
    args.provider = effective.provider
    args.model = effective.model
    args.dir = effective.dir
    args.skip_permissions = effective.skip_permissions
    args.task_manager = effective.task_manager
    args.memory_only = effective.memory_only
    args.history_memory = effective.history_memory
    args.use_sqlite = effective.use_sqlite
    args.auto_approve_send_message = effective.auto_approve_send_message
    args.clean = False
    args.preferred_provider = getattr(effective, "preferred_provider", args.provider)

    return await run_agent(args)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for deep-worker command.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="deep-worker",
        description="Start a deep-worker agent process",
        # Subcommands are not registered through add_subparsers (the positional
        # ``name`` with nargs='?' would make that a breaking change — see
        # _dispatch_subcommand), so we list them in the epilog:
        # without this, the user would not find the command in --help.
        epilog=(
            "subcommands (run '<subcommand> --help' for details):\n"
            "  lsp enable|disable|status\n"
            "                        Manage optional LSP code navigation "
            "(Serena MCP)\n"
            "  mcp list|add|remove   Manage external MCP servers\n"
            "  skills ...            Manage agent skills\n"
            "  threads list|delete   Manage conversation threads\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "name",
        nargs="?",
        help="Unique name for this agent (not required with --list)",
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Working directory for the agent (default: current directory)",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: provider-specific default)",
    )

    parser.add_argument(
        "--provider",
        default="deepseek",
        # Built-ins PLUS anything declared in ~/.deep_worker/providers.json,
        # so a config-added API provider is selectable with no code change.
        choices=list(all_provider_names()),
        help="AI provider to use (default: deepseek)",
    )

    parser.add_argument(
        "-sp",
        "--skip-permissions",
        action="store_true",
        dest="skip_permissions",
        help="Skip permissions (auto-approve dangerous operations)",
    )
    parser.add_argument(
        "--no-skip-permissions",
        dest="skip_permissions",
        action="store_const",
        const=False,
        help="Disable skip-permissions (useful with --restart to override a saved True value)",
    )

    parser.add_argument(
        "-tm",
        "--task-manager",
        action="store_true",
        dest="task_manager",
        help="Enable TaskManager for AI-powered task evaluation and correction",
    )
    parser.add_argument(
        "--no-task-manager", "--no-tm",
        dest="task_manager",
        action="store_const",
        const=False,
        help="Disable TaskManager (useful with --restart to override a saved True value)",
    )

    parser.add_argument(
        "-asm",
        "--auto-approve-send-message",
        action="store_true",
        dest="auto_approve_send_message",
        help=(
            "Auto-approve calls to the send_message_to_agent "
            "inter-agent messaging tool, while keeping HITL approvals "
            "active for all other side-effect tools (execute, edit_file, "
            "write_file, web_search, fetch_url, task). Has no effect "
            "when -sp / --skip-permissions is also set (everything is "
            "already auto-approved in that case)."
        ),
    )
    parser.add_argument(
        "--no-auto-approve-send-message",
        dest="auto_approve_send_message",
        action="store_const",
        const=False,
        help="Disable auto-approve-send-message (useful with --restart to override a saved True value)",
    )

    # Tri-state universal web-search policy.
    # Mutual exclusion is enforced by a shared dest + mutually exclusive group:
    # passing both flags is an argparse error.
    web_search_group = parser.add_mutually_exclusive_group()
    web_search_group.add_argument(
        "--enable-web-search",
        dest="web_search_policy",
        action="store_const",
        const="enabled",
        default="auto",
        help=(
            "Force-enable the universal web_search delegation tool. Fails "
            "loudly with a remedy when the Claude Code SDK fork is missing."
        ),
    )
    web_search_group.add_argument(
        "--disable-web-search",
        dest="web_search_policy",
        action="store_const",
        const="disabled",
        default="auto",
        help=(
            "Disable the universal web_search delegation tool. Never "
            "resolves/imports the claude provider or SDK."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean all history and state before starting (agent must not be running)",
    )

    parser.add_argument(
        "--import-claude",
        metavar="UUID_OR_PATH",
        dest="import_claude",
        default=None,
        help=(
            "Import a Claude Code CLI session transcript (a real .jsonl, given "
            "as a session uuid or a file path) as this agent's conversation "
            "history, written to the agent's SQLite DB, then start the agent. "
            "Works for any --provider. Combine with --clean to replace existing "
            "history. See docs/DESIGN_IMPORT_CLAUDE_SESSION.md."
        ),
    )

    parser.add_argument(
        "--import-only",
        action="store_true",
        dest="import_only",
        help=(
            "With --import-claude: import the history and EXIT without starting "
            "the agent (validate / prepare only)."
        ),
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        dest="dev_tree",
        help=(
            "Run the agent using the development code tree instead of the "
            "production code tree. The flag is normally consumed by the "
            "bash launcher (deep-worker) BEFORE Python starts; it is also "
            "declared here so argparse does not reject it inside the dev "
            "tree after the re-exec, and so it can be persisted to "
            "launch_config for --restart auto-detection. See "
            "docs/DESIGN_DEV_FLAG.md."
        ),
    )
    parser.add_argument(
        "--no-dev",
        dest="dev_tree",
        action="store_const",
        const=False,
        help="Disable dev-tree mode (useful with --restart to override a saved True value)",
    )

    parser.add_argument(
        "-l",
        "--context_limit",
        dest="context_limit",
        type=parse_context_limit,
        default=None,
        metavar="VALUE",
        help="Effective context window in tokens (e.g. 200000, 200k, 1M). "
             "Summarization trigger and keep are computed from this window. "
             "Pass 'default' to clear a previously saved override.",
    )

    parser.add_argument(
        "--hot-cache-ttl",
        dest="hot_cache_ttl",
        type=parse_hot_cache_ttl,
        action="append",
        default=None,
        metavar="SPEC",
        help="Hot-cache TTL in seconds for summarization "
             "(DESIGN_SUMMARIZATION_HOT_COLD §4). Repeatable. "
             "Forms: <sec> (all providers) or <provider>=<sec>. "
             "0 = always cold. Survives --restart.",
    )

    parser.add_argument(
        "--summary-keep",
        dest="summary_keep",
        type=parse_summary_keep,
        default=None,
        metavar="SPEC",
        help="How much raw conversation tail to keep after summarization. "
             "Forms: 0 (no tail, summary only), <N>/<N>k (token CEILING, "
             "lowers but never raises the default), <N>%% or 0.<N> "
             "(share of the window, replaces the default), <N>m (message "
             "count). Default: 10%% of --context_limit. Survives --restart.",
    )

    parser.add_argument(
        "--restart",
        "-r",
        action="store_true",
        help="Restart agent with parameters from previous launch. "
             "Additional options override saved values. "
             "Use --no-* flags (e.g. --no-tm, --no-skip-permissions) to "
             "disable a previously saved boolean flag. "
             "--clean is NOT inherited; pass it explicitly if needed.",
    )

    # Internal: set by the ``restart_self`` tool when an agent restarts
    # itself. The flag alone grants nothing — it must be accompanied by
    # --self-requested-token, a one-time capability the agent minted in its
    # own private data dir. See process/self_restart_token.py.
    parser.add_argument(
        "--self-requested",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--self-requested-token",
        dest="self_requested_token",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--fork",
        metavar="SOURCE_NAME",
        default=None,
        help="Fork (clone) an existing agent's data directory. "
             "Creates a new agent with the source's launch config, "
             "checkpoint history, summaries, and working data. "
             "Optional --from <message_id> truncates history at that point. "
             "Use --provider/--model/--dir/-sp/-tm/-m etc. to override "
             "the source's saved launch config.",
    )

    parser.add_argument(
        "--from",
        metavar="MESSAGE_ID",
        dest="fork_from",
        default=None,
        help="When used with --fork, truncate the new agent's history "
             "at this message_id. Messages after this ID are dropped. "
             "If the message_id is not found, the command fails with "
             "a non-zero exit code.",
    )

    parser.add_argument(
        "-pp",
        "--preferred-provider",
        default=None,
        dest="preferred_provider",
        help=(
            "Set the preferred provider for this agent (overrides both "
            "preferred_provider AND active provider). When omitted on a "
            "restart, preferred_provider is inherited from the saved "
            "launch config. Automatic failover always uses "
            "--no-update-preferred to preserve your preference."
        ),
    )

    parser.add_argument(
        "--no-update-preferred",
        action="store_true",
        dest="no_update_preferred",
        help=(
            "When restarting with --provider, do NOT update the saved "
            "preferred_provider. Intended for automatic failover so the "
            "operator's original preference survives a temporary provider "
            "switch. Also see --preferred-provider."
        ),
    )

    parser.add_argument(
        "--failover-chain",
        metavar="NAME",
        default=None,
        dest="failover_chain",
        help=(
            "Name of the failover chain this agent follows when a provider "
            "hits a usage limit. Chains live in "
            "~/.deep_worker/failover_chains.json; 'default' is used when "
            "omitted. See --list-failover-chains."
        ),
    )

    parser.add_argument(
        "--list-failover-chains",
        action="store_true",
        dest="list_failover_chains",
        help=(
            "Print the configured failover chains and exit. Shows each step "
            "and flags any that names an unknown or uncredentialed provider."
        ),
    )

    parser.add_argument(
        "--migrate-storage",
        metavar="NAME",
        dest="migrate_storage",
        help="Migrate a stopped agent's storage to the current per-agent layout",
    )

    parser.add_argument(
        "--to-version",
        type=int,
        default=3,
        dest="to_version",
        help="Target storage version for --migrate-storage (default: 3)",
    )

    parser.add_argument(
        "--list", "-ls",
        action="store_true",
        dest="list_threads",
        help="List all conversation threads in database",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for --list (default: text)",
    )

    parser.add_argument(
        "--with-counts",
        action="store_true",
        dest="with_counts",
        help="Include message counts (slow, for --list only)",
    )

    parser.add_argument(
        "--agent",
        metavar="NAME",
        help="Filter by agent name (for --list only)",
    )

    parser.add_argument(
        "--sort",
        choices=["thread_id", "agent", "updated"],
        default="thread_id",
        help="Sort field for --list (default: thread_id)",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order for --list",
    )

    parser.add_argument(
        "-rm",
        "--remove-thread",
        metavar="THREAD_ID",
        dest="remove_thread",
        help="Remove a specific thread from the database",
    )

    parser.add_argument(
        "--history",
        metavar="THREAD_ID",
        dest="show_history",
        help="Show message history for a specific thread (works offline)",
    )

    parser.add_argument(
        "--remove-before",
        metavar="DATE",
        dest="remove_before",
        help=(
            "Remove conversation data older than DATE (ISO 8601 or relative "
            "like 30d, 1y). Prunes checkpoint threads AND offloaded summary "
            "sections under ~/.deep_worker/<agent>/summaries/ across ALL "
            "agents in the current $HOME (destructive and global — NEVER "
            "run in ad-hoc integration tests without an isolated HOME). "
            "If DATE resolves to a future timestamp, the command refuses "
            "by default and requires --force (a future cutoff means "
            "\"delete everything\"). Always pair with --dry-run first."
        ),
    )

    parser.add_argument(
        "--info",
        action="store_true",
        dest="info",
        help="Show launch configuration and runtime status of an agent (read-only, offline).",
    )

    parser.add_argument(
        "--limits-info",
        action="store_true",
        dest="limits_info",
        help="Show persisted provider limit/reset information for --provider (read-only).",
    )

    parser.add_argument(
        "--list-summaries",
        action="store_true",
        dest="list_summaries",
        help=(
            "List offloaded conversation-history files from "
            "~/.deep_worker/<agent>/summaries/. Use --agent NAME to filter, "
            "--format json for machine-readable output."
        ),
    )

    parser.add_argument(
        "--reindex-history",
        action="store_true",
        dest="reindex_history",
        help="Rebuild this agent's history-memory index and exit.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would be deleted without actually deleting (for -rm and --remove-before)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        dest="force",
        help=(
            "Required to confirm --remove-before when the cutoff is in "
            "the future (which would delete ALL data). Harmless for past "
            "cutoffs. See DESIGN_REMOVE_BEFORE_FUTURE_CUTOFF_GUARD.md."
        ),
    )

    # Mutually exclusive group for storage mode
    storage_group = parser.add_mutually_exclusive_group()

    storage_group.add_argument(
        "--memory-only",
        action="store_true",
        dest="memory_only",
        help="Use in-memory storage only (no persistence to database)",
    )

    storage_group.add_argument(
        "--use-sqlite",
        action="store_true",
        dest="use_sqlite",
        help="Force SQLite persistence (even in test environment)",
    )

    # --no-* counterparts live OUTSIDE the mutually exclusive group
    # (they are not exclusive — they just set the flag to False).
    parser.add_argument(
        "--no-memory-only",
        dest="memory_only",
        action="store_const",
        const=False,
        help="Disable memory-only mode (useful with --restart to override a saved True value)",
    )
    parser.add_argument(
        "--no-use-sqlite",
        dest="use_sqlite",
        action="store_const",
        const=False,
        help="Disable sqlite (useful with --restart to override a saved True value)",
    )

    parser.add_argument(
        "-m", "--history-memory",
        action="store_true",
        dest="history_memory",
        help="Enable semantic search over summarised conversation history "
             "(indexes past messages for recall with the history_memory tool)",
    )
    parser.add_argument(
        "--no-history-memory", "--no-m",
        dest="history_memory",
        action="store_const",
        const=False,
        help="Disable history-memory (useful with --restart to override a saved True value)",
    )

    # External MCP — see docs/DESIGN_EXTERNAL_MCP_TOOLS.md.
    parser.add_argument(
        "--mcp-config",
        dest="mcp_config_paths",
        action="append",
        default=[],
        help=(
            "Path to an additional MCP config file (same JSON shape as "
            "~/.deep_worker/mcp.json). Repeatable; later files override earlier."
        ),
    )
    parser.add_argument(
        "--mcp-server",
        dest="mcp_server_inline",
        action="append",
        default=[],
        help=(
            "Inline MCP server, formatted '<name>={...json...}'. Repeatable. "
            "Overrides any same-named entry from config files."
        ),
    )
    # Turn-ending tools — see docs/DESIGN_CLI_TURN_ENDING_TOOL_FLAG.md.
    parser.add_argument(
        "--turn-ending-tool",
        dest="turn_ending_tools",
        action="append",
        default=[],
        type=_canonical_turn_ending_spec_str,
        metavar="NAME[:FIELD]",
        help=(
            "Declare a tool whose successful result ends the agent's turn. "
            "Repeatable. NAME is the GRAPH-side tool name — for an external "
            "MCP server it is the mangled '<server>__<tool>' form, e.g. "
            "'repo_graph__submit_fragments'. With ':FIELD' the turn ends "
            "only when that top-level field of the result is exactly True; "
            "without it, when 'success' is truthy."
        ),
    )
    parser.add_argument(
        "--tools",
        metavar="SPEC",
        default=None,
        help=(
            "Restrict which tools the agent may call (see "
            "docs/DESIGN_TOOL_SELECTION.md). Omit for every tool — the "
            "default, unchanged. 'none' (or '-*') denies all; "
            "'none,read_file,grep' denies all but those; '-execute' removes "
            "only those. A spec starting with '-' needs the '=' form, e.g. "
            "--tools='-execute', because argparse reads a leading dash as an "
            "option. Applies to "
            "deep-worker tools; on the ACP providers (claude-code, grok) "
            "and partly on codex the CLI's own native tools stay reachable "
            "and a warning says so at startup."
        ),
    )
    parser.add_argument(
        "--tools-add",
        dest="tools_add",
        metavar="NAMES",
        action="append",
        default=None,
        help=(
            "Add tools on top of the CURRENT selection instead of restating "
            "it. Comma-separated and repeatable; '*' and '?' are wildcards, "
            "e.g. --tools-add 'serena__*'. On --restart this layers over the "
            "saved selection, so a restricted agent can gain one tool "
            "without listing the rest."
        ),
    )
    parser.add_argument(
        "--tools-remove",
        "--tools-rm",
        dest="tools_remove",
        metavar="NAMES",
        action="append",
        default=None,
        help=(
            "Remove tools from the CURRENT selection. Same form as "
            "--tools-add. When both name the same tool the more specific "
            "one wins, so \"--tools-rm 'serena__*' --tools-add "
            "serena__find_symbol\" drops the family but keeps that one; "
            "equally specific opposites are an error rather than a "
            "silent winner."
        ),
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help=(
            "Summarize the agent's active context now. With --query or "
            "--query-file, that text is used as a one-shot summary prompt."
        ),
    )

    # Startup query options — see docs/DESIGN_STARTUP_QUERY_OPTIONS.md.
    # Two independent mutually-exclusive pairs. --sum-query* offloads the
    # current thread history to <agent>.jsonl and starts from a summary-shape
    # message; -q/--query* is a plain starting query. Both pairs may be
    # combined (sum-query + query → a single turn with two HumanMessage).
    # Neither is persisted to launch_config (one-shot per invocation).
    sum_query_group = parser.add_mutually_exclusive_group()
    sum_query_group.add_argument(
        "--sum-query",
        dest="sum_query",
        default=None,
        metavar="TEXT",
        help=(
            "Starting query treated as a pseudo-summary: the current thread "
            "history is offloaded to ~/.deep_worker/<agent>/summaries/"
            "<agent>.jsonl, the active context is wiped, and this text becomes "
            "a summary-shape starting message. One-shot; not persisted."
        ),
    )
    sum_query_group.add_argument(
        "--sum-query-file",
        dest="sum_query_file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Same as --sum-query but the text is read from a UTF-8 file.",
    )

    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "-q",
        "--query",
        dest="query",
        default=None,
        metavar="TEXT",
        help=(
            "Plain starting query — the agent begins executing it immediately "
            "after startup, with no history offload. One-shot; not persisted."
        ),
    )
    query_group.add_argument(
        "--query-file",
        dest="query_file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Same as --query but the text is read from a UTF-8 file.",
    )

    return parser


def resolve_startup_queries(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    """Resolve the startup sum-query and plain query from parsed args.

    Reads the ``--sum-query`` / ``--sum-query-file`` and ``-q/--query`` /
    ``--query-file`` options into two resolved strings. File variants are
    read as UTF-8. Empty / whitespace-only content (inline OR file) is a
    hard error, matching the fail-loud goal (§2.6).

    The argparse mutual-exclusion groups already reject giving both members
    of a pair, so this helper only has to handle file I/O and emptiness.

    Args:
        args: Parsed CLI namespace.

    Returns:
        ``(sum_query, query)`` — each is the resolved text or ``None`` when
        that option was not supplied.

    Raises:
        SystemExit: (exit code 2) when a file is unreadable or resolved
            content is empty / whitespace-only. Raised via
            ``argparse``-style ``parser.error`` semantics — but since we do
            not hold the parser here, we raise ``SystemExit(2)`` directly
            after writing to stderr, matching the pre-daemonize contract.
    """

    def _read_file(path: Path, opt: str) -> str:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"deep-worker: {opt}: cannot read {path}: {e}\n")
            raise SystemExit(2) from e
        return content

    def _resolve(text: str | None, file: Path | None, opt: str) -> str | None:
        if text is not None:
            value = text
        elif file is not None:
            value = _read_file(file, opt)
        else:
            return None
        if not value.strip():
            sys.stderr.write(
                f"deep-worker: {opt}: query text is empty or whitespace-only\n"
            )
            raise SystemExit(2)
        return value

    if getattr(args, "summarize", False) and (
        getattr(args, "sum_query", None) is not None
        or getattr(args, "sum_query_file", None) is not None
    ):
        sys.stderr.write(
            "deep-worker: --summarize cannot be combined with --sum-query "
            "or --sum-query-file\n"
        )
        raise SystemExit(2)

    sum_query = _resolve(
        getattr(args, "sum_query", None),
        getattr(args, "sum_query_file", None),
        "--sum-query",
    )
    query = _resolve(
        getattr(args, "query", None),
        getattr(args, "query_file", None),
        "--query",
    )
    return sum_query, query


# ⚠️ DO NOT re-add a "seed the claude session_uuid hint" step here.
#
# An earlier version seeded ``claude_session_state.json`` with the imported
# transcript's uuid so the claude-code-sdk provider would RESUME that real file.
# That broke tool availability (observed in a past incident): the imported transcript is
# an ORDINARY ``claude`` session created OUTSIDE deep_worker. Resuming it makes
# the CLI restore THAT session's tool/permission configuration, which has no
# ``mcp__deepworker__*`` tools. deep_worker only sets ``disallowed_tools`` (it
# blocks the native Bash/Read/Edit) and never sets ``allowed_tools`` — it relies
# on a FRESH session granting the deepworker MCP tools by default. On a resumed
# external session that default does not apply, so the agent ends up with ~1
# tool instead of its full set.
#
# Correct behaviour: import writes ONLY the SQLite history (provider-agnostic).
# Under claude-code-sdk, with no hint, the provider starts a fresh session (full MCP
# tools) and carries the imported memory the normal way — synthesizing OUR own
# transcript from the LangGraph history (session_synthesis), exactly like a
# cross-provider continuation. See DESIGN_IMPORT_CLAUDE_SESSION.md §6.


async def handle_import_claude(args: argparse.Namespace) -> int:
    """Import a Claude CLI session transcript into the agent's SQLite history.

    Resolve ``--import-claude`` → parse → write to the agent's per-agent DB.
    This is provider-agnostic: it writes ONLY the message history (the agent's
    memory). It deliberately does NOT touch claude-code-sdk session state — see the
    note above on why seeding a resume hint loses the agent's tools.
    Returns 0 on success, non-zero on a user-facing error. Errors are printed
    to stderr (this runs pre-daemonize so the user sees them).
    """
    from deep_worker_cli import sessions  # noqa: PLC0415

    if parse_claude_transcript is None or resolve_transcript_path is None:
        print(
            "Error: --import-claude requires the claude_code_sdk_provider package, "
            "which is only available in the source checkout (not in the PyPI "
            "wheel).",
            file=sys.stderr,
        )
        return 2

    uuid_or_path = args.import_claude
    cwd = str(args.dir.resolve()) if getattr(args, "dir", None) else None

    # 1. Resolve the transcript file.
    try:
        transcript_path = resolve_transcript_path(uuid_or_path, cwd=cwd)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # 2. Parse to a LangGraph message history.
    try:
        messages = parse_claude_transcript(transcript_path)
    except Exception as exc:  # noqa: BLE001 — surface a clean error, no trace
        print(
            f"Error: failed to parse Claude transcript {transcript_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not messages:
        print(
            f"Error: no importable conversation turns found in "
            f"{transcript_path} (no user/assistant messages).",
            file=sys.stderr,
        )
        return 1

    # 3/4. Write into the agent's SQLite DB (replace=clean). This is the ONLY
    # side effect — provider-agnostic. We do NOT seed any claude-code-sdk session
    # hint (see the note above ``handle_import_claude``).
    replace = bool(getattr(args, "clean", False))
    try:
        n = await sessions.import_history(
            args.name, messages, replace=replace
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Imported {n} message(s) from {transcript_path} into agent "
        f"'{args.name}'."
    )
    return 0


async def migrate_storage_command(args: argparse.Namespace) -> int:
    from deep_worker_cli.process.registry import ProcessRegistry
    from deep_worker_cli.storage_paths import (
        CURRENT_STORAGE_VERSION,
        migrate_agent_storage_to_v4,
    )

    agent_name = args.migrate_storage
    if not agent_name:
        print("Error: --migrate-storage requires an agent name", file=sys.stderr)
        return 2
    if args.to_version != CURRENT_STORAGE_VERSION:
        print(
            f"Error: unsupported target storage version {args.to_version}; "
            f"only {CURRENT_STORAGE_VERSION} is supported",
            file=sys.stderr,
        )
        return 2
    if ProcessRegistry().lookup(agent_name):
        print(
            f"Error: Agent '{agent_name}' is running — stop it before migrating storage.",
            file=sys.stderr,
        )
        return 1
    try:
        result = migrate_agent_storage_to_v4(agent_name)
    except Exception as exc:
        print(f"Error migrating storage for '{agent_name}': {exc}", file=sys.stderr)
        return 1

    print(f"Migrated storage for '{agent_name}' to version {CURRENT_STORAGE_VERSION}.")
    print(f"  Agent root: {result['agent_root']}")
    print(f"  Threads DB: {result['threads_db']}")
    return 0


async def _send_startup_query_to_running_agent(
    agent_name: str,
    sum_query: str | None,
    query: str | None,
) -> int:
    """Send a startup query to an ALREADY-RUNNING agent via RPyC (§4.5).

    Instead of spawning a new process, dispatch ``sum_query`` / ``query`` to
    the live agent's extended ``exposed_invoke_agent`` so it runs the same
    offload→wipe→invoke sequence in-process. Fire-and-forget from the CLI's
    perspective — the agent streams to its subscribers as usual.

    Args:
        agent_name: Name of the running agent.
        sum_query: Pseudo-summary query text, or ``None``.
        query: Plain query text, or ``None``.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    from deep_worker_cli.client.agent_client import AgentClient

    client = AgentClient(agent_name)
    try:
        await client.connect()
        response = await client._send_command(
            {
                "command": "invoke",
                "message": "",
                "stream": True,
                "sum_query": sum_query,
                "query": query,
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error: failed to send startup query to '{agent_name}': {e}", file=sys.stderr)
        return 1
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    if not response.get("success"):
        print(
            f"Error: agent '{agent_name}' rejected startup query: "
            f"{response.get('error', 'unknown error')}",
            file=sys.stderr,
        )
        return 1
    print(f"Startup query sent to running agent '{agent_name}'.")
    return 0


async def _send_summarize_to_running_agent(
    agent_name: str,
    prompt: str | None,
) -> int:
    from deep_worker_cli.client.agent_client import AgentClient

    client = AgentClient(agent_name)
    try:
        await client.connect()
        response = await client._send_command(
            {"command": "summarize", "prompt": prompt}
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Error: failed to summarize '{agent_name}': {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass
    if not response.get("success"):
        print(
            f"Error: agent '{agent_name}' rejected summarization: "
            f"{response.get('error', 'unknown error')}",
            file=sys.stderr,
        )
        return 1
    print(
        f"Agent '{agent_name}' context "
        f"{'summarized' if response.get('summarized') else 'did not require replacement'}."
    )
    return 0


async def run_agent(args: argparse.Namespace) -> int:
    """Run the agent process.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Determine working directory
        # Priority: --dir arg > DEEP_WORKER_ORIGINAL_CWD env > current dir
        working_dir = args.dir
        if working_dir is None:
            # Get from environment variable set by bash wrapper
            original_cwd = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
            if original_cwd:
                working_dir = Path(original_cwd)
            else:
                working_dir = Path.cwd()

        # Load or create config
        if args.config:
            config = AgentConfig.from_file(args.config)
        else:
            config = AgentConfig(
                model=args.model,
                working_dir=working_dir,
                provider=args.provider,
                web_search_policy=getattr(args, "web_search_policy", "auto"),
                tools=resolve_tool_spec(args),
            )

        # Build launch_config for persistent storage via the single source
        # of truth, build_launch_config(). The previous hand-rolled inline
        # dict here was MISSING ``dev_tree`` and ``preferred_provider`` — the
        # same wart the fork path already avoided (see line ~593). A missing
        # ``dev_tree`` is exactly what left an agent un-restartable from the
        # web UI in a past incident: the metadata never recorded that the agent
        # ran on the dev tree, so ``_spawn_subprocess`` rebuilt the launch on
        # the prod tree where the ``grok`` provider is unknown to argparse.
        # build_launch_config also auto-detects dev_tree from the running
        # code tree, so a direct dev-checkout launch (no ``--dev``) is now
        # persisted correctly. ``args`` is the effective namespace (merged on
        # restart by merge_restart_config), which is what build_launch_config
        # expects.
        launch_config = build_launch_config(args)

        # One-shot startup queries (never persisted to launch_config). NIT-2:
        # avoid a second physical file read (TOCTOU). main() already resolved
        # them PRE-daemonize and stashed the results on args; reuse those.
        # Fall back to resolving here only for direct run_agent() callers/tests
        # that never went through main().
        if getattr(args, "_resolved_startup_queries", None) is not None:
            startup_sum_query, startup_query = args._resolved_startup_queries
        else:
            startup_sum_query, startup_query = resolve_startup_queries(args)

        # Turn-ending tool specs: validate the FULL list here, before any
        # AgentProcess exists. Per-value grammar was already enforced by
        # argparse's ``type=`` handler on first launch, but cross-entry
        # collisions (a name declared twice, or a CLI declaration of the
        # built-in ``send_message_to_agent``) cannot be seen one value at a
        # time — and on --restart the values come from
        # session_metadata.json, which argparse never touched.
        #
        # Validating here keeps the failure OUT of AgentProcess.start()'s
        # init ``try``, which would otherwise swallow it into an ERROR-mode
        # agent that stays registered. The parsed result is intentionally
        # discarded: AgentProcess.__init__ re-parses the raw strings, so
        # each entry point enforces the contract independently.
        turn_ending_tool_specs = list(
            getattr(args, "turn_ending_tools", None) or []
        )
        _parse_turn_ending_tool_specs(turn_ending_tool_specs)

        # Create and start agent
        agent = AgentProcess(
            name=args.name,
            config=config,
            skip_permissions=args.skip_permissions,
            enable_task_manager=args.task_manager,
            memory_only=args.memory_only,
            use_sqlite=args.use_sqlite,
            launch_config=launch_config,
            auto_approve_send_message=getattr(
                args, "auto_approve_send_message", False
            ),
            mcp_config_paths=[
                Path(p) for p in (getattr(args, "mcp_config_paths", None) or [])
            ],
            mcp_server_inline=list(
                getattr(args, "mcp_server_inline", None) or []
            ),
            turn_ending_tool_specs=turn_ending_tool_specs,
            startup_query=startup_query,
            startup_sum_query=startup_sum_query,
            startup_summarize=bool(getattr(args, "summarize", False)),
            startup_summary_prompt=(
                startup_query if getattr(args, "summarize", False) else None
            ),
        )

        try:
            await agent.start()

            # Keep running until shutdown signal - EVENT DRIVEN, NO POLLING
            # Wait for shutdown event instead of polling agent.running
            await agent.shutdown_event.wait()
            
            return 0

        finally:
            # Always cleanup checkpointer on exit
            await agent.cleanup()

    except KeyboardInterrupt:
        print("\nShutdown requested")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _detach_from_session_scope(agent_name: str | None = None) -> None:
    from deep_worker_cli.process.daemon import _detach_from_session_scope as detach

    detach(agent_name)


# File-mode mask applied by ``daemonize()`` right after ``setsid()``.
#
# The historical daemon recipe (Stevens' *Advanced Programming in the UNIX
# Environment*) uses ``umask(0)`` so a daemon writing files with *explicit*
# mode bits does not inherit the caller's umask. That is the WRONG choice
# for us: our daemon runs an LLM agent that creates user-visible files and
# directories WITHOUT explicit mode (via ``execute``/``write_file`` and via
# plenty of internal ``Path.mkdir(parents=True, exist_ok=True)`` sites), so
# the daemon must behave like a normal interactive user shell.
#
# Value: ``0o022`` — the system default from ``/etc/login.defs`` (``UMASK
# 022``). Effect: files created without explicit mode become ``0o644``,
# directories ``0o755``. Private storage paths remain private: they use
# explicit ``chmod 0o600/0o700`` in ``storage_paths.py``
# (``ensure_private_file`` / ``ensure_private_dir``) and are unaffected by
# this mask. See ``docs/DESIGN_FIX_DAEMONIZE_UMASK.md``.


def _reap_empty_cgroup_leaves() -> None:
    from deep_worker_cli.process.daemon import _reap_empty_cgroup_leaves as reap

    reap()


def daemonize(name: str, log_dir: Path | None = None) -> None:
    from deep_worker_cli.process.daemon import daemonize as daemonize_process

    daemonize_process(
        name, log_dir, detach=_detach_from_session_scope, reap=_reap_empty_cgroup_leaves
    )


async def remove_thread_command(thread_id: str, dry_run: bool) -> int:
    """Handle -rm/--remove-thread command.

    Args:
        thread_id: Thread ID to remove
        dry_run: If True, show what would be deleted without deleting

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli.sessions import remove_thread
    from deep_worker_cli.config import console

    try:
        result = await remove_thread(thread_id, dry_run=dry_run)

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        if dry_run:
            console.print(f"[yellow]Dry run - would delete thread '{thread_id}':[/yellow]")
            console.print(f"  - Checkpoints: {result['checkpoints_found']}")
            if result.get("writes_found", 0) > 0:
                console.print(f"  - Writes: {result['writes_found']}")
            console.print("\n[dim]Run without --dry-run to actually delete[/dim]")
        else:
            console.print(f"[green]✓ Removed thread '{thread_id}':[/green]")
            console.print(f"  - Checkpoints deleted: {result['checkpoints_deleted']}")
            if result.get("writes_deleted", 0) > 0:
                console.print(f"  - Writes deleted: {result['writes_deleted']}")

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


async def show_history_command(thread_id: str, format: str) -> int:
    """Handle --history command.

    Args:
        thread_id: Thread ID to show history for
        format: Output format ("text" or "json")

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli.sessions import get_thread_history
    from deep_worker_cli.config import console

    try:
        result = await get_thread_history(thread_id, format=format)

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        if format == "json":
            print(result["json_output"])
        else:
            # Display header similar to dwcli --history
            console.print(f"\nConversation History for '{thread_id}':")
            console.print(f"Showing {len(result['messages'])} messages ({len(result['checkpoints'])} checkpoints)")
            console.print("=" * 60)

            # Display messages with sender labels, timestamps, and separators
            for i, msg in enumerate(result['messages'], 1):
                # Handle both dict and LangChain Message objects
                if isinstance(msg, dict):
                    msg_type = msg.get('type', 'unknown')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp')
                elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                    # LangChain Message object
                    msg_type = msg.type if hasattr(msg, 'type') else type(msg).__name__
                    content = msg.content
                    timestamp = getattr(msg, 'timestamp', None)
                else:
                    msg_type = type(msg).__name__
                    content = str(msg)
                    timestamp = None

                # Map message types to display labels (matching dwcli --history format)
                if msg_type in ("human", "HumanMessage"):
                    type_label = "USER"
                    separator = "-" * 60
                elif msg_type in ("ai", "AIMessage"):
                    type_label = "ASSISTANT"
                    separator = "=" * 60
                elif msg_type in ("tool", "ToolMessage"):
                    type_label = "TOOL"
                    separator = "." * 60
                elif msg_type in ("system", "SystemMessage"):
                    type_label = "SYSTEM"
                    separator = "-" * 60
                else:
                    type_label = msg_type.upper()
                    separator = "-" * 60

                # Format timestamp for display with millisecond precision
                # This precision is critical for distinguishing message order when
                # comparing events across different agent threads. Database stores
                # timestamps with microsecond precision (e.g., "2026-03-29T22:40:17.822305+00:00")
                # and we preserve milliseconds in display for accurate event sequencing.
                timestamp_str = ""
                if timestamp:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp)
                        # Format: "Mar 29, 10:40:17.822pm" (includes seconds + milliseconds)
                        ms = int(dt.microsecond / 1000)
                        timestamp_str = f" - {dt.strftime('%b %d, %-I:%M:%S')}.{ms:03d}{dt.strftime('%p').lower()}"
                    except (ValueError, AttributeError):
                        pass

                console.print(f"\n[{type_label}]{timestamp_str}")
                console.print(content)

                # Don't print separator after last message
                if i < len(result['messages']):
                    console.print(separator)

            console.print("\n" + "=" * 60)

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


async def remove_before_command(
    date_str: str, dry_run: bool, force: bool = False
) -> int:
    """Handle --remove-before command.

    Prunes BOTH checkpoint threads in per-agent SQLite DBs AND offloaded
    summary sections under ``~/.deep_worker/<agent>/summaries/`` across
    ALL agents in the current ``$HOME``.

    Safety guard (DESIGN_REMOVE_BEFORE_FUTURE_CUTOFF_GUARD.md): when the
    parsed cutoff is in the future — semantically "delete everything" —
    the command refuses with exit code ``2`` unless ``force=True``. Even
    ``dry_run=True`` does NOT bypass the guard, because a future dry-run
    is still a strong signal of an operator bug.

    Args:
        date_str: Date string (ISO 8601 or relative).
        dry_run: If True, show what would be deleted without deleting.
        force: Bypass the future-cutoff guard. Required when ``cutoff > now``.

    Returns:
        Exit code:

        - ``0`` success (or nothing to do).
        - ``1`` real error (invalid date, I/O failure).
        - ``2`` future-cutoff guard fired; no data touched.
    """
    from datetime import datetime, timezone

    from deep_worker_cli.sessions import remove_threads_before, parse_date
    from deep_worker_cli.summaries import prune_summaries_before
    from deep_worker_cli.config import console

    # Parse the cutoff date ONCE here so the summaries side uses the same
    # timestamp as the DB side (and so we fail fast on malformed input).
    try:
        cutoff_dt = parse_date(date_str)
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid date format: {e}")
        return 1

    # ----- Future-cutoff safety guard -----
    # A future cutoff matches every existing thread's updated_at (all of
    # which are by definition <= now), which is equivalent to "nuke
    # everything." This is almost always an operator mistake. Block
    # unless --force was passed explicitly. Fires even in dry-run, since
    # a dry-run of "would delete everything" is itself the smoking-gun
    # evidence of the bug we are trying to catch.
    now = datetime.now(timezone.utc)
    if cutoff_dt > now and not force:
        console.print(
            "[red]Refusing --remove-before: parsed cutoff is in the future.[/red]"
        )
        console.print(
            f"  parsed cutoff: [bold]{cutoff_dt.isoformat()}[/bold]"
        )
        console.print(
            f"  current time:  {now.isoformat()}"
        )
        console.print(
            "  A future cutoff matches every existing thread and would "
            "delete ALL data under ~/.deep_worker/ and "
            "~/.deep_worker/*/summaries/."
        )
        console.print(
            "  If you truly intend this, re-run with [bold]--force[/bold]."
        )
        console.print(
            "  (Even --dry-run requires --force for future cutoffs; "
            "the dry-run of \"delete everything\" is itself likely a bug.)"
        )
        return 2

    try:
        # --- 1) DB threads ---
        result = await remove_threads_before(date_str, dry_run=dry_run)
        if not result["success"]:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        cutoff_date = result["cutoff_date"]

        # --- 2) Summaries ---
        # Pure-sync pathlib I/O; no new threads, no async needed, no polling.
        summary_counts = prune_summaries_before(cutoff_dt, dry_run=dry_run)

        if dry_run:
            console.print(f"[yellow]Dry run - would delete data before {cutoff_date}:[/yellow]")
            console.print(f"  - Threads: {result['threads_found']}")
            console.print(f"  - Checkpoints: {result['checkpoints_found']}")
            if result.get("writes_found", 0) > 0:
                console.print(f"  - Writes: {result['writes_found']}")
            console.print(
                f"  - Summary sections: {summary_counts['sections_to_drop']} "
                f"across {summary_counts['files_scanned']} file(s)"
            )
            console.print("\n[dim]Run without --dry-run to actually delete[/dim]")
        else:
            console.print(f"[green]✓ Removed data before {cutoff_date}:[/green]")
            console.print(f"  - Threads deleted: {result['threads_deleted']}")
            console.print(f"  - Checkpoints deleted: {result['checkpoints_deleted']}")
            if result.get("writes_deleted", 0) > 0:
                console.print(f"  - Writes deleted: {result['writes_deleted']}")
            console.print(
                f"  - Summary files deleted: {summary_counts['files_deleted']}; "
                f"rewritten: {summary_counts['files_rewritten']} "
                f"({summary_counts['sections_deleted']} sections dropped)"
            )

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1



async def list_summaries_command(agent_name: str | None, fmt: str) -> int:
    """Handle --list-summaries command.

    Walks ``~/.deep_worker/<agent>/summaries/*.md`` and prints one row per
    offloaded thread. Offline, read-only, no RPyC, no running-agent
    dependency.

    Args:
        agent_name: Filter to a single agent, or ``None`` for all agents.
        fmt: ``"text"`` (default human table) or ``"json"``.

    Returns:
        Exit code (always 0 unless something truly exceptional happens).
    """
    import json as _json
    from deep_worker_cli.summaries import iter_summary_files
    from deep_worker_cli.config import console

    rows = list(iter_summary_files(agent_name=agent_name))

    if fmt == "json":
        payload = [
            {
                "agent": r.agent,
                "thread": r.thread,
                "path": str(r.path),
                "events": r.events,
                "latest_ts": r.latest_ts,
                "size": r.size,
            }
            for r in rows
        ]
        print(_json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    # Text format
    if not rows:
        scope = f" for agent '{agent_name}'" if agent_name else ""
        console.print(f"[yellow]No summaries found{scope}.[/yellow]")
        return 0

    title = (
        f"Offloaded summaries for '{agent_name}' ({len(rows)} file(s))"
        if agent_name
        else f"Offloaded summaries ({len(rows)} file(s))"
    )
    console.print(f"\n{title}")
    console.print("=" * 78)
    # Header
    console.print(
        f"{'Agent':<18} {'Thread':<28} {'Events':>6}  "
        f"{'Latest':<26} {'Size':>8}"
    )
    console.print("-" * 78)
    for r in rows:
        latest = r.latest_ts or "(no headers)"
        if len(latest) > 26:
            latest = latest[:25] + "…"
        size_h = _human_size(r.size)
        console.print(
            f"{r.agent:<18.18} {r.thread:<28.28} {r.events:>6}  "
            f"{latest:<26.26} {size_h:>8}"
        )
    console.print("=" * 78)
    console.print("[dim]Files live under ~/.deep_worker/<agent>/summaries/.[/dim]")
    return 0


def _human_size(n: int) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            if unit == "B":
                return f"{n}{unit}"
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n}GB"


def show_failover_chains() -> int:
    """Print the configured failover chains and flag unusable steps.

    Read-only. Diagnoses the two ways a chain can disappoint at failover
    time: a step naming a provider that does not exist, and one whose API
    key env var is unset (a restart onto it could not authenticate).
    """
    from deep_worker_cli.runtime.failover_chains import (
        FailoverChainConfigError,
        chains_path,
        load_chains,
    )
    from deep_worker_cli.runtime.provider_chain import all_provider_names

    path = chains_path()
    try:
        chains = load_chains()
    except FailoverChainConfigError as exc:
        print(f"config error: {exc}")
        return 1

    known = set(all_provider_names())
    try:
        from deep_worker_cli.runtime.provider_registry import load_provider_specs

        specs = {spec.name: spec for spec in load_provider_specs()}
    except Exception:  # noqa: BLE001 — listing must survive a bad registry
        specs = {}

    print(f"config: {path}")
    for name in sorted(chains):
        chain = chains[name]
        print(f"\nchain: {name}")
        if chain.description:
            print(f"  description: {chain.description}")
        for i, step in enumerate(chain.steps):
            line = f"  {i}. {step.provider}"
            if step.model:
                line += f" (model={step.model})"
            notes: list[str] = []
            if step.provider not in known:
                notes.append("UNKNOWN PROVIDER")
            spec = specs.get(step.provider)
            if spec is not None and not spec.has_credentials():
                notes.append(f"no credentials (${spec.api_key_env} unset)")
            if notes:
                line += "  <- " + "; ".join(notes)
            print(line)
    return 0


def show_provider_limits_info(provider: str) -> int:
    """Display persisted provider limit/reset information."""
    info = AvailabilityTracker().limit_info(provider)
    print(f"provider: {provider}")
    if info is None:
        print("status: no_limit_record")
        return 0

    raw_reset = info.get("available_after_utc")
    status = "limited"
    if isinstance(raw_reset, str):
        try:
            reset = datetime.fromisoformat(raw_reset)
            if reset <= datetime.now(reset.tzinfo):
                status = "available"
        except ValueError:
            status = "unknown"
    elif raw_reset is None:
        status = "unknown"
    print(f"status: {status}")
    if raw_reset:
        print(f"available_after_utc: {raw_reset}")
    source = info.get("available_after_source")
    if source:
        print(f"source: {source}")
    note = info.get("available_after_note")
    if note:
        print(f"note: {note}")
    print(f"last_failure_at_utc: {info.get('last_failure_at_utc', '')}")
    print(f"last_failure_agent: {info.get('last_failure_agent', '')}")
    marker = info.get("last_failure_marker")
    if marker:
        print(f"last_failure_marker: {marker}")
    return 0


def show_agent_info(name: str) -> int:
    """Display launch configuration and runtime status of an agent.

    Reads session_metadata.json and queries the process registry to produce
    a formatted summary. This command is read-only and does not start or
    restart the agent.

    Returns:
        0 on success, 1 if no metadata found.
    """
    import subprocess

    from deep_worker_cli.process.registry import ProcessRegistry

    data_dir = _get_agent_data_dir(name)
    metadata_file = data_dir / "session_metadata.json"

    if not metadata_file.exists():
        print(
            f"Error: No session metadata found for agent '{name}'.\n"
            f"Agent has never been started, or its data directory "
            f"({data_dir}) was removed.",
            file=sys.stderr,
        )
        return 1

    try:
        metadata = json.loads(metadata_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Error: Cannot read session metadata for agent '{name}': {exc}",
            file=sys.stderr,
        )
        return 1

    launch_config = metadata.get("launch_config") or {}
    status = metadata.get("status", "unknown")
    last_activity = metadata.get("last_activity", "")

    # Resolve provider name (handle preferred_provider fallback)
    provider = str(launch_config.get("provider", "unknown"))
    preferred = str(launch_config.get("preferred_provider", ""))
    if preferred:
        provider = f"{provider} (preferred: {preferred})"

    model = str(launch_config.get("model", "unknown"))
    working_dir = str(launch_config.get("working_dir", ""))
    tree_root = str(launch_config.get("tree_root", ""))
    dev_tree = launch_config.get("dev_tree", False)

    # Git commit info
    commit = "unknown"
    if tree_root:
        try:
            sha = subprocess.run(
                ["git", "-C", tree_root, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if sha:
                describe = subprocess.run(
                    ["git", "-C", tree_root, "describe", "--always", "--dirty"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                commit = describe if describe else sha
        except Exception:
            commit = "unknown"

    tree_root_str = tree_root
    if dev_tree:
        tree_root_str = f"{tree_root} (dev_tree=true)"

    # Process registry lookup
    process_info = ProcessRegistry().lookup(name)
    pid = launch_config.get("pid", "?")
    running_str = ""
    if process_info is not None:
        pid = str(process_info.pid)
        running_str = " (running)"
    elif os.path.isdir(f"/proc/{pid}"):
        running_str = " (running)"
    else:
        running_str = " (NOT RUNNING)"

    started_at = str(launch_config.get("started_at", ""))
    cli_args = str(launch_config.get("cli_args", ""))

    flags = {
        "skip_permissions": launch_config.get("skip_permissions", False),
        "task_manager": launch_config.get("task_manager", False),
        "history_memory": launch_config.get("history_memory", False),
        "memory_only": launch_config.get("memory_only", False),
        "use_sqlite": launch_config.get("use_sqlite", False),
        "auto_approve_send_message": launch_config.get("auto_approve_send_message", False),
        "clean": launch_config.get("clean", False),
        "dev_tree": dev_tree,
    }

    print(f"agent: {name}")
    print(f"provider: {provider}")
    print(f"model: {model}")
    print(f"working_dir: {working_dir}")
    print(f"tree_root: {tree_root_str}")
    print(f"commit: {commit}")
    print(f"started_at: {started_at}")
    print(f"pid: {pid}{running_str}")
    # Context-limit display (the operator must
    # see the enforced limit at a glance). Shows ``None`` for
    # default behaviour and the number for an explicit ``-l``. See
    # ``docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md`` §3.7.
    _cl_display = launch_config.get("context_limit")
    if _cl_display is None or _cl_display == "":
        print("context_limit: (default — provider/model cap applies)")
    else:
        print(f"context_limit: {_cl_display:,} tokens (-l {(_cl_display // 1000) if _cl_display >= 1000 else _cl_display}{'k' if _cl_display >= 1000 else ''}, operator-set)")
    _ttl_display = launch_config.get("hot_cache_ttl") or {}
    if _ttl_display:
        parts = [f"{k}={v}s" for k, v in sorted(_ttl_display.items())]
        print("hot_cache_ttl: " + ", ".join(parts))
    else:
        print("hot_cache_ttl: (default table / env)")
    _keep_display = launch_config.get("summary_keep")
    if _keep_display:
        _keep_kind, _keep_value = _keep_display
        if _keep_kind == "tokens" and _keep_value == 0:
            print("summary_keep: 0 (no tail — summary only)")
        elif _keep_kind == "tokens":
            print(f"summary_keep: {_keep_value:,} tokens (ceiling)")
        elif _keep_kind == "fraction":
            print(f"summary_keep: {_keep_value:.0%} of the window")
        else:
            print(f"summary_keep: last {_keep_value} messages")
    else:
        print("summary_keep: (default — 10% of context_limit)")
    print("flags:")
    for flag_name, flag_value in flags.items():
        print(f"  {flag_name}: {str(flag_value).lower()}")
    print(f"cli_args: {cli_args}")
    print(f"status: {status}")
    print(f"last_activity: {last_activity}")
    return 0


def _normalize_status_value(status: object) -> str:
    status_value = getattr(status, "value", status)
    return str(status_value).lower()


def _print_reindex_stats(stats: object) -> None:
    print(
        f"Reindexed {getattr(stats, 'files_processed')} file(s): "
        f"{getattr(stats, 'lines_read')} lines -> "
        f"{getattr(stats, 'chunks_indexed')} chunks "
        f"({getattr(stats, 'errors')} error(s))."
    )


async def _reindex_running_agent(agent_name: str, status: object) -> int:
    normalized_status = _normalize_status_value(status)
    if normalized_status != "idle":
        print(
            f"Error: agent '{agent_name}' is busy (status: {normalized_status}); "
            "stop it or wait until idle before reindexing.",
            file=sys.stderr,
        )
        return 1

    client = AgentClient(agent_name)
    try:
        await client.connect()
        from rich.console import Console

        console = Console()
        with console.status(f"Reindexing history for '{agent_name}'..."):
            result = await client.reindex_history()
    except Exception as exc:
        print(
            f"Error: failed to reach running agent '{agent_name}': {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    if not result.get("success"):
        print(
            f"Error: failed to rebuild history index for '{agent_name}': "
            f"{result.get('error', 'unknown error')}",
            file=sys.stderr,
        )
        return 1

    stats = SimpleNamespace(
        files_processed=result.get("files", 0),
        lines_read=result.get("lines", 0),
        chunks_indexed=result.get("chunks", 0),
        errors=result.get("errors", 0),
    )
    _print_reindex_stats(stats)
    return 1 if stats.errors else 0


async def reindex_history_command(agent_name: str) -> int:
    process_info = ProcessRegistry().lookup(agent_name)
    if process_info is not None:
        client = AgentClient(agent_name)
        try:
            await client.connect()
            session_info = await client.get_status()
            status = getattr(session_info, "status", "unknown")
        except Exception as exc:
            print(
                f"Error: failed to reach running agent '{agent_name}': {exc}",
                file=sys.stderr,
            )
            return 1
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass
        return await _reindex_running_agent(agent_name, status)

    from deep_worker_cli.summaries import get_summaries_dir

    summaries_dir = get_summaries_dir(agent_name)
    if not summaries_dir.exists():
        print(
            f"Error: no summaries directory for agent '{agent_name}' ({summaries_dir})",
            file=sys.stderr,
        )
        return 1

    jsonl_files = sorted(summaries_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(
            f"Agent '{agent_name}' has no conversation history "
            f"(no *.jsonl in {summaries_dir}); nothing to reindex."
        )
        return 0

    try:
        from deep_worker_cli.semantic_memory.client import SemanticMemoryClient

        reply = await SemanticMemoryClient().index_history(
            summaries_dir, clear_first=True
        )
    except Exception as exc:
        print(f"Error: failed to rebuild history index: {exc}", file=sys.stderr)
        return 1

    stats = SimpleNamespace(
        files_processed=reply.files_indexed,
        lines_read=reply.lines_read,
        chunks_indexed=reply.chunks_added,
        errors=reply.errors,
    )
    _print_reindex_stats(stats)
    return 1 if stats.errors else 0


def _build_subcommand_registry() -> dict[str, tuple[Any, Any]]:
    """``{name: (setup_parser, handler)}`` for the generic subcommands.

    Imports are local and lazy: the registry is only built once the first
    token turns out to be a subcommand, so a normal agent launch does not
    pay for loading these modules.

    ``skills`` is NOT part of this: its ``setup_skills_parser`` requires a
    keyword-only ``make_help_action``, and ``execute_skills_command``
    returns ``None`` instead of a return code. It is handled separately
    in :func:`_dispatch_skills_subcommand`.
    """
    from deep_worker_cli.cli.lsp_command import (
        handle_lsp_command,
        setup_lsp_parser,
    )
    from deep_worker_cli.cli.mcp_command import (
        handle_mcp_command,
        setup_mcp_parser,
    )
    from deep_worker_cli.cli.migrate_command import (
        handle_migrate_agent_dbs_command,
        setup_migrate_agent_dbs_parser,
    )

    return {
        "lsp": (setup_lsp_parser, handle_lsp_command),
        "mcp": (setup_mcp_parser, handle_mcp_command),
        "migrate-agent-dbs": (
            setup_migrate_agent_dbs_parser,
            handle_migrate_agent_dbs_command,
        ),
    }


def _dispatch_skills_subcommand(argv: list[str]) -> int | None:
    """``deep-worker skills …``.

    Kept separate from the generic registry because of its different signature (see above):
    its parser needs ``make_help_action``, and its handler does not return a code.
    """
    if not argv or argv[0] != "skills":
        return None

    from deep_worker_cli.skills import execute_skills_command, setup_skills_parser

    def _make_help_action(show: Callable[[], None]) -> type[argparse.Action]:
        class _HelpAction(argparse.Action):
            def __init__(self, option_strings, dest, **kwargs):  # type: ignore[no-untyped-def]
                super().__init__(option_strings, dest, nargs=0, **kwargs)

            def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[no-untyped-def]
                show()
                parser.exit()

        return _HelpAction

    parser = argparse.ArgumentParser(
        prog="deep-worker", description="Start a deep-worker agent process"
    )
    subparsers = parser.add_subparsers(dest="command")
    setup_skills_parser(subparsers, make_help_action=_make_help_action)
    execute_skills_command(parser.parse_args(argv))
    return 0


def _dispatch_subcommand(argv: list[str]) -> int | None:
    """Intercept ``deep-worker <subcommand> …`` BEFORE the main parser.

    Why not ``add_subparsers`` on the main parser: it has a positional
    ``name`` with ``nargs='?'``. Adding subparsers breaks ALL normal
    launches — ``deep-worker myagent`` starts failing with
    ``invalid choice: 'myagent'``. So routing is done here, and the main
    parser is left untouched.

    The comparison is by FULL match of the first token: an agent name
    like ``lspagent`` / ``mcpagent`` will not be intercepted.

    Returns:
        The return code if the command was handled; ``None`` if this is
        not a subcommand and the normal agent-launch path should continue.

    See lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md §4.6.
    """
    if not argv or argv[0].startswith("-"):
        return None

    registry = _build_subcommand_registry()
    entry = registry.get(argv[0])
    if entry is None:
        return None

    setup_parser, handler = entry

    parser = argparse.ArgumentParser(
        prog="deep-worker", description="Start a deep-worker agent process"
    )
    subparsers = parser.add_subparsers(dest="command")
    setup_parser(subparsers)
    return handler(parser.parse_args(argv))


def _dispatch_threads_subcommand(argv: list[str]) -> int | None:
    """``deep-worker threads list|delete`` — a thin wrapper over the already
    existing thread commands of the main parser.

    Kept separate because ``threads`` has no dedicated ``setup_*_parser``:
    its subcommands are described inline (see main.py, dead entry point)
    and map onto the existing handlers ``list_threads_command`` /
    ``remove_thread_command``.
    """
    if len(argv) < 1 or argv[0] != "threads":
        return None

    parser = argparse.ArgumentParser(prog="deep-worker threads")
    sub = parser.add_subparsers(dest="threads_command")

    list_p = sub.add_parser("list", aliases=["ls"], help="List conversation threads")
    list_p.add_argument("--agent", default=None, help="Filter by agent name")
    list_p.add_argument("--format", default="text", choices=["text", "json"])
    list_p.add_argument("--with-counts", action="store_true")
    list_p.add_argument("--sort", default="updated",
                        choices=["thread_id", "agent", "updated"])
    list_p.add_argument("--reverse", action="store_true")

    del_p = sub.add_parser("delete", aliases=["rm"], help="Delete a thread")
    del_p.add_argument("thread_id")
    del_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv[1:])

    if args.threads_command in {"list", "ls"}:
        args.list_threads = True
        args.name = None
        return asyncio.run(list_threads_command(args))
    if args.threads_command in {"delete", "rm"}:
        return asyncio.run(remove_thread_command(args.thread_id, args.dry_run))

    parser.print_help()
    return 2


def main() -> int:
    """Main entry point for deep-worker command.

    Returns:
        Exit code
    """
    # CLI subcommands are routed BEFORE create_parser() — see
    # _dispatch_subcommand (the main parser uses a positional ``name``,
    # so subparsers cannot be added to it).
    _argv = sys.argv[1:]
    for _dispatch in (
        _dispatch_subcommand,
        _dispatch_skills_subcommand,
        _dispatch_threads_subcommand,
    ):
        _sub_rc = _dispatch(_argv)
        if _sub_rc is not None:
            return _sub_rc

    parser = create_parser()
    args = parser.parse_args()

    # Heal a legacy poisoned ``--model unknown`` before any launch path
    # reads it. See ``normalize_model_arg`` — covers direct launch,
    # --restart, and web-spawned subprocesses alike.
    normalize_model_arg(args)

    apply_preferred_provider_override(args)

    if args.migrate_storage:
        return asyncio.run(migrate_storage_command(args))

    # Handle --list option (mutually exclusive with running agent)
    if args.list_threads:
        if args.name:
            print("Error: Cannot specify agent name with --list option", file=sys.stderr)
            return 2
        return asyncio.run(list_threads_command(args))

    # Handle -rm/--remove-thread option
    if args.remove_thread:
        return asyncio.run(remove_thread_command(args.remove_thread, args.dry_run))

    # Handle --history option
    if args.show_history:
        return asyncio.run(show_history_command(args.show_history, args.format))

    # Handle --remove-before option
    if args.remove_before:
        return asyncio.run(
            remove_before_command(
                args.remove_before,
                args.dry_run,
                force=getattr(args, "force", False),
            )
        )

    # Handle --list-summaries option (mutually exclusive with running agent)
    if getattr(args, "list_summaries", False):
        if args.name:
            print("Error: Cannot specify agent name with --list-summaries (use --agent NAME instead)", file=sys.stderr)
            return 2
        return asyncio.run(
            list_summaries_command(agent_name=args.agent, fmt=args.format)
        )

    if getattr(args, "reindex_history", False):
        if not args.name:
            print("Error: --reindex-history requires an agent name.", file=sys.stderr)
            return 2
        return asyncio.run(reindex_history_command(args.name))

    if getattr(args, "list_failover_chains", False):
        if args.name:
            print("Error: --list-failover-chains does not take an agent name.", file=sys.stderr)
            return 2
        return show_failover_chains()

    if getattr(args, "limits_info", False):
        if args.name:
            print("Error: --limits-info does not take an agent name; use --provider NAME.", file=sys.stderr)
            return 2
        if "provider" not in _explicit_cli_flags():
            print("Error: --limits-info requires explicit --provider NAME.", file=sys.stderr)
            return 2
        return show_provider_limits_info(args.provider)

    # Require agent name for all other operations
    if not args.name:
        print("Error: Agent name is required (or use --list / --info)", file=sys.stderr)
        parser.print_help()
        return 2

    # Handle --info option (read-only, offline — no agent start needed)
    if getattr(args, "info", False):
        return show_agent_info(args.name)

    # Validate agent name: prevent ':tm' suffix (reserved for TaskManager)
    if ":tm" in args.name:
        print(
            f"Error: Agent name '{args.name}' cannot contain ':tm' suffix.\n"
            f"This suffix is reserved for TaskManager threads.\n"
            f"Please choose a different agent name.",
            file=sys.stderr
        )
        return 1

    # ──────────────────────────────────────────────────────────────────────────
    # --fork <source_name> [--from <message_id>]
    # ──────────────────────────────────────────────────────────────────────────
    fork_source = getattr(args, "fork", None)

    if fork_source:
        if getattr(args, "clean", False):
            print(
                "Error: --clean and --fork are incompatible. "
                "A fresh fork is already clean; use --fork without --clean.",
                file=sys.stderr,
            )
            return 2

        return asyncio.run(do_fork(args))

    # --- --import-claude / --import-only (DESIGN_IMPORT_CLAUDE_SESSION.md) ----
    # Combination guards first (cheap, terminal-visible).
    if getattr(args, "import_only", False) and not getattr(args, "import_claude", None):
        print(
            "Error: --import-only requires --import-claude <uuid|path>.",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "import_claude", None) and getattr(args, "restart", False):
        print(
            "Error: --import-claude cannot be combined with --restart "
            "(restart inherits a prior history; importing into it is "
            "ambiguous). Run the import on a fresh start instead.",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "import_claude", None):
        # One-shot, pre-daemonize: resolve + parse + write to the agent's
        # SQLite DB. Errors are visible on the terminal. Refuse if the agent
        # is currently running (we'd be writing its DB underneath it).
        from deep_worker_cli.process.registry import ProcessRegistry as _Reg
        if _Reg().lookup(args.name):
            print(
                f"Error: Agent '{args.name}' is running — stop it before "
                f"importing history into it.",
                file=sys.stderr,
            )
            return 1
        import_rc = asyncio.run(handle_import_claude(args))
        if import_rc != 0:
            return import_rc
        if getattr(args, "import_only", False):
            # Import-and-exit: do not start the agent.
            return 0
        # Otherwise fall through to the normal launch below so the agent
        # starts with the freshly-imported history. ``--clean`` already ran
        # inside the import (replace), so clear it to avoid wiping what we
        # just wrote during the normal launch path.
        args.clean = False

    # Handle --restart option
    if getattr(args, "restart", False):
        # Pre-flight checks BEFORE daemonize (so errors are visible in terminal)
        saved = load_launch_config(args.name)
        if saved is None:
            print(
                f"Error: Cannot restart agent '{args.name}' — no previous launch configuration found.\n"
                f"Start the agent normally first:  deep-worker {args.name} --provider ... --model ...",
                file=sys.stderr,
            )
            return 1

        # Check if running agent can be stopped
        from deep_worker_cli.process.registry import ProcessRegistry
        from deep_worker_cli.process.agent_stop import _read_agent_status

        registry = ProcessRegistry()
        existing = registry.lookup(args.name)
        if existing:
            data_dir = _get_agent_data_dir(args.name)
            status = _read_agent_status(data_dir)
            if status == "running":
                print(
                    f"Error: Cannot restart agent '{args.name}' — it is actively running (status: running).\n"
                    f"Wait for it to finish or interrupt it first.",
                    file=sys.stderr,
                )
                return 1

        # Validate registry writable
        validation = registry.validate_writable()
        if not validation["writable"]:
            print(f"ERROR: Cannot start agent - registry file is not writable", file=sys.stderr)
            print(f"  {validation['error']}", file=sys.stderr)
            return 1
        if validation.get("fixed"):
            print(f"✓ Fixed registry file ownership")

        # Save CWD before daemonize changes it
        effective = merge_restart_config(saved, args)
        _emit_tool_selection_preflight(
            effective.provider, _preflight_spec(getattr(effective, "tools", None))
        )
        if effective.dir:
            os.environ["DEEP_WORKER_ORIGINAL_CWD"] = str(Path(effective.dir).resolve())
        elif "DEEP_WORKER_ORIGINAL_CWD" not in os.environ:
            os.environ["DEEP_WORKER_ORIGINAL_CWD"] = os.getcwd()

        daemonize(args.name)

        # After daemonize: stop running agent (if any), then launch
        return asyncio.run(restart_agent(args))

    # ========================================================================
    # PRE-DAEMONIZE: ONLY FAST CHECKS (must not block terminal for > ~0.5s)
    # ========================================================================
    # The bash user expects:
    #     $ deep-worker foo -asm --dir /tmp --clean
    #     Agent 'foo' starting in background (...)
    #     $                                           ← prompt back in ≤ 1s
    #
    # So every heavy thing (SQLite WAL wait for --clean, loading langgraph,
    # building the LLM, etc.) MUST happen AFTER daemonize(), in the detached
    # grandchild process. Errors from that phase go to the per-agent log file
    # that daemonize() prints the path to.
    #
    # Only checks that are both (a) fast and (b) produce an error message the
    # user MUST see on their terminal belong here.
    # ========================================================================
    # Validate + resolve one-shot startup queries PRE-DAEMONIZE so the user
    # sees file-read / empty-text errors (exit 2) on their terminal, not in a
    # log file. The resolved values are recomputed inside run_agent from the
    # same args (idempotent) — here we only need the validation side effect and
    # the values for the running-agent send-through path below.
    startup_sum_query, startup_query = resolve_startup_queries(args)
    # Stash so run_agent() (in the daemonized child) reuses these instead of
    # re-reading the file a second time (NIT-2 / TOCTOU).
    args._resolved_startup_queries = (startup_sum_query, startup_query)
    summarize = bool(getattr(args, "summarize", False))
    has_startup_query = startup_sum_query is not None or (
        startup_query is not None and not summarize
    )

    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    # (a) Reject if an agent with this name is already running. This is a
    #     pure registry lookup (file + fcntl) — sub-millisecond.
    existing = registry.lookup(args.name)
    if existing:
        if summarize:
            return asyncio.run(
                _send_summarize_to_running_agent(args.name, startup_query)
            )
        # Idle send-through (§4.5): a running agent + a startup query means
        # "send this query to the live agent" rather than an error. Dispatch
        # via the existing RPyC invoke path (no new endpoint) and return.
        if has_startup_query:
            return asyncio.run(
                _send_startup_query_to_running_agent(
                    args.name, startup_sum_query, startup_query
                )
            )
        print(
            f"Error: Agent '{args.name}' is already running (PID {existing.pid})",
            file=sys.stderr,
        )
        print(
            f"Stop it first with: pkill -f 'deep-worker {args.name}'",
            file=sys.stderr,
        )
        return 1

    # (b) Validate registry file is writable. Cheap file-stat check.
    validation = registry.validate_writable()
    if not validation["writable"]:
        print(f"ERROR: Cannot start agent - registry file is not writable", file=sys.stderr)
        print(f"  {validation['error']}", file=sys.stderr)
        return 1
    if validation.get("fixed"):
        print(f"✓ Fixed registry file ownership")

    # Preserve the original CWD so the daemonized child can still resolve
    # paths the user typed.
    if args.dir:
        os.environ["DEEP_WORKER_ORIGINAL_CWD"] = str(args.dir.resolve())
    elif "DEEP_WORKER_ORIGINAL_CWD" not in os.environ:
        os.environ["DEEP_WORKER_ORIGINAL_CWD"] = os.getcwd()

    # Hint for the user if --clean was requested: it used to run before
    # daemonize and could block the terminal for 20+ seconds waiting on
    # SQLite WAL locks held by other running agents. We moved the SLOW
    # part (DB cleanup) into the background. The FAST parts (agent data
    # dir + log files) still run pre-daemonize so they don't race with
    # daemonize() re-opening the log files under our feet.
    if args.clean:
        import shutil
        # Remove per-agent data directory (contains session_metadata.json).
        # Fast (a few unlinks); no DB lock involved.
        data_dir = agent_root(args.name, create=False)
        if data_dir.exists():
            try:
                shutil.rmtree(data_dir)
            except OSError as e:
                print(
                    f"Warning: could not remove {data_dir}: {e}", file=sys.stderr
                )
        # Remove stale log files BEFORE daemonize re-opens them for
        # append. If we did this post-daemonize we'd unlink our own
        # active log-fd from the directory entry and silently lose all
        # future agent output.
        log_dir = deep_worker_home() / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob(f"{args.name}.*"):
                try:
                    log_file.unlink()
                except OSError:
                    pass  # best-effort; not worth blocking startup
        print(f"  [--clean: DB cleanup will run in background]")

    # Tool-selection disclosure must reach the terminal, not the log file.
    _emit_effective_tool_selection_preflight(args)

    # Daemonize IMMEDIATELY. Everything below here executes in the
    # detached grandchild; parent already returned to the shell.
    daemonize(args.name)

    # --- Post-daemonize (grandchild only) -------------------------------
    # Output goes to ~/.deep_worker/logs/<name>.{out,err}.log from here on.

    if args.clean:
        # Database cleanup is the slow part (can wait up to 30s on SQLite
        # WAL locks held by other running agents). We do it here, in the
        # daemonized grandchild, so the terminal is already free.
        try:
            from deep_worker_cli.sessions import clean_agent_history_db_only
            asyncio.run(clean_agent_history_db_only(args.name))
        except Exception as e:
            # Log but do NOT abort — a clean failure should not prevent
            # the agent from starting with whatever history is there.
            print(f"[startup] Warning: --clean DB cleanup failed: {e}", file=sys.stderr)

    # Run agent main loop.
    return asyncio.run(run_agent(args))


async def list_threads_command(args: argparse.Namespace) -> int:
    """Handle --list command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from deep_worker_cli import sessions

    try:
        # Get all threads
        threads = await sessions.list_all_threads(
            agent_name=args.agent,
            sort_by=args.sort,
            reverse=args.reverse,
            include_message_count=args.with_counts,
        )

        # Format output
        if args.format == "json":
            output = format_threads_json(threads, args.sort, args.agent)
        else:  # text
            output = format_threads_text(threads, args.with_counts)

        print(output)
        return 0

    except Exception as e:
        print(f"Error listing threads: {e}", file=sys.stderr)
        return 1


def format_threads_text(threads: list[dict], with_counts: bool) -> str:
    """Format threads as aligned text table.

    Args:
        threads: List of thread dicts with thread_id, agent_name, updated_at, optional message_count
        with_counts: Include message count column

    Returns:
        Formatted text string
    """
    if not threads:
        return "\nNo threads found in database.\n"

    # Column widths
    thread_id_width = 24
    agent_name_width = 20
    messages_width = 10
    updated_width = 20

    # Build header
    if with_counts:
        header = f"{'Thread ID':<{thread_id_width}} {'Agent Name':<{agent_name_width}} {'Messages':>{messages_width}} {'Last Updated':<{updated_width}}"
        separator = "=" * (thread_id_width + agent_name_width + messages_width + updated_width + 3)
    else:
        header = f"{'Thread ID':<{thread_id_width}} {'Agent Name':<{agent_name_width}} {'Last Updated':<{updated_width}}"
        separator = "=" * (thread_id_width + agent_name_width + updated_width + 2)

    lines = ["\n" + header, separator]

    # Format each thread
    for thread in threads:
        thread_id = thread["thread_id"][:thread_id_width]  # Truncate if too long
        agent_name = thread["agent_name"] if thread["agent_name"] else "(self)"
        agent_name = agent_name[:agent_name_width]  # Truncate if too long

        # Format timestamp
        updated_at = thread.get("updated_at", "")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at).astimezone()
                updated_str = dt.strftime("%b %d, %-I:%M%p").lower()
                # Capitalize month
                parts = updated_str.split(" ")
                if parts:
                    parts[0] = parts[0].capitalize()
                    updated_str = " ".join(parts)
            except (ValueError, TypeError):
                updated_str = updated_at[:updated_width]
        else:
            updated_str = ""

        if with_counts:
            message_count = thread.get("message_count", 0)
            line = f"{thread_id:<{thread_id_width}} {agent_name:<{agent_name_width}} {message_count:>{messages_width}} {updated_str:<{updated_width}}"
        else:
            line = f"{thread_id:<{thread_id_width}} {agent_name:<{agent_name_width}} {updated_str:<{updated_width}}"

        lines.append(line)

    lines.append("")  # Empty line at end
    return "\n".join(lines)


def format_threads_json(threads: list[dict], sort_by: str, filter_agent: str | None) -> str:
    """Format threads as JSON.

    Args:
        threads: List of thread dicts
        sort_by: Sort field used
        filter_agent: Agent filter applied (or None)

    Returns:
        JSON string with pretty printing
    """
    output = {
        "threads": threads,
        "total": len(threads),
        "sort_by": sort_by,
        "filter_agent": filter_agent,
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
