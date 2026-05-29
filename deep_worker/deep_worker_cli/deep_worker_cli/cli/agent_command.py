"""CLI command to start a deep-worker agent process."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from deep_worker_cli.agent_config_pkg.agent_config import AgentConfig
from deep_worker_cli.process.agent_process import AgentProcess

# Claude-session import (DESIGN_IMPORT_CLAUDE_SESSION.md). The provider lives at
# <repo-root>/claude_code_provider/ and is NOT importable until the repo root is
# on sys.path — exactly the bootstrap ``tools.py`` does for web_search. A plain
# top-level ``from claude_code_provider...`` import here breaks EVERY invocation
# of ``deep-worker`` (even ``--help``), because the CLI module is imported
# before any sys.path setup runs. Add the repo root first, THEN import, so the
# names exist as module globals (tests patch
# ``agent_command.parse_claude_transcript`` / ``resolve_transcript_path``).
_repo_root = Path(__file__).resolve().parents[4]  # .../<repo>/  (cli→cli_pkg→pkg→deep_worker→repo)
if (_repo_root / "claude_code_provider").exists() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from claude_code_provider.session_import import (  # noqa: E402
    parse_claude_transcript,
    resolve_transcript_path,
)


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
    "auto_approve_send_message": False,
    "dev_tree": False,
}


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


def _get_agent_data_dir(name: str) -> Path:
    """Return the data directory for an agent."""
    return Path("~/.deep_worker").expanduser() / name


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
    saved provider would silently win (incident observed 2026-05-29).

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
    detector.add_argument("--use-sqlite", dest="use_sqlite",
                          action="store_true", default=argparse.SUPPRESS)
    detector.add_argument("--dev", dest="dev_tree",
                          action="store_true", default=argparse.SUPPRESS)

    # parse_known_args: ignore every other flag the real parser owns.
    ns, _ = detector.parse_known_args(argv)
    return set(vars(ns).keys()) - {"name"}


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
    (incident observed 2026-05-29).

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
    # the literal "unknown" (incident observed 2026-05-20). Treat that
    # as "no model specified" so the provider default kicks back in on
    # restart instead of forwarding the bad string to the API.
    if effective.model == "unknown":
        effective.model = None
    effective.dir = Path(saved["working_dir"]) if saved.get("working_dir") else _PARSER_DEFAULTS["dir"]
    effective.skip_permissions = saved.get("skip_permissions", False)
    effective.task_manager = saved.get("task_manager", False)
    effective.memory_only = saved.get("memory_only", False)
    effective.use_sqlite = saved.get("use_sqlite", False)
    effective.auto_approve_send_message = saved.get(
        "auto_approve_send_message", False
    )
    effective.dev_tree = saved.get("dev_tree", False)
    effective.clean = False  # NEVER inherit --clean

    # Override with explicitly provided CLI args
    provider_overridden = _is_override(
        "provider", args.provider, _PARSER_DEFAULTS["provider"]
    )
    if provider_overridden:
        effective.provider = args.provider

    model_explicit = _is_override("model", args.model, _PARSER_DEFAULTS["model"])
    if model_explicit:
        effective.model = args.model
    elif provider_overridden and effective.provider != saved.get("provider"):
        # Provider changed but no --model given: do NOT carry the old
        # provider's model forward. Reset so build_launch_config() resolves
        # the new provider's default (incident observed 2026-05-29).
        effective.model = None
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

    return effective


def build_launch_config(args: argparse.Namespace) -> dict:
    """Build a launch_config dict from effective argparse namespace.

    Args:
        args: Namespace with effective (merged) values.

    Returns:
        Dict suitable for storage in session_metadata.json.
    """
    from deep_worker_cli.config import get_default_model_name

    model = args.model
    if model is None:
        # Resolve the provider default. For providers with no pinned
        # default (codex), this stays None and MUST be persisted as
        # None — the codex CLI then picks its own current default. Do
        # NOT coerce to "unknown": that string would be read back on
        # restart and forwarded to CodexChatModel, which the API
        # rejects with HTTP 400 (incident observed 2026-05-20).
        model = get_default_model_name(args.provider)

    working_dir = args.dir
    if working_dir is None:
        original_cwd = os.environ.get("DEEP_WORKER_ORIGINAL_CWD")
        working_dir = Path(original_cwd) if original_cwd else Path.cwd()

    # dev_tree may come from either the parsed flag or, after the bash
    # launcher's re-exec (which strips --dev from argv), from the
    # DEEP_WORKER_DEV_TREE=1 env var. Honor either source. Only the
    # exact string "1" counts as truthy — '0', 'false', '' do not.
    dev_tree = getattr(args, "dev_tree", False) or (
        os.environ.get("DEEP_WORKER_DEV_TREE") == "1"
    )

    return {
        "provider": args.provider,
        "model": model,
        "working_dir": str(working_dir),
        "skip_permissions": args.skip_permissions,
        "task_manager": args.task_manager,
        "memory_only": args.memory_only,
        "use_sqlite": args.use_sqlite,
        "clean": args.clean,
        "auto_approve_send_message": getattr(
            args, "auto_approve_send_message", False
        ),
        "dev_tree": dev_tree,
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

    # 3. Stop running agent if needed
    registry = ProcessRegistry()
    existing = registry.lookup(name)
    if existing:
        try:
            data_dir = _get_agent_data_dir(name)
            await stop_agent(name, registry, data_dir=data_dir)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except TimeoutError as e:
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

    # 5. Apply effective args and delegate to run_agent
    # Copy effective values back into args for run_agent()
    args.provider = effective.provider
    args.model = effective.model
    args.dir = effective.dir
    args.skip_permissions = effective.skip_permissions
    args.task_manager = effective.task_manager
    args.memory_only = effective.memory_only
    args.use_sqlite = effective.use_sqlite
    args.auto_approve_send_message = effective.auto_approve_send_message
    args.clean = False  # Already handled above if needed

    return await run_agent(args)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for deep-worker command.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="deep-worker",
        description="Start a deep-worker agent process",
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
        choices=["deepseek", "gemini", "openai", "claude-code", "codex"],
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
        "-tm",
        "--task-manager",
        action="store_true",
        dest="task_manager",
        help="Enable TaskManager for AI-powered task evaluation and correction",
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
            "Run the agent using the development codebase instead of the "
            "production codebase. The flag is normally consumed by the "
            "bash launcher (deep-worker) BEFORE Python starts; it is also "
            "declared here so argparse does not reject it inside the dev "
            "tree after the re-exec, and so it can be persisted to "
            "launch_config for --restart auto-detection. See "
            "docs/DESIGN_DEV_FLAG.md."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart agent with parameters from previous launch. "
             "Additional options override saved values. "
             "--clean is NOT inherited; pass it explicitly if needed.",
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
            "sections under the agent's data directory (summaries/) across ALL "
            "agents in the current $HOME (destructive and global — NEVER "
            "run in ad-hoc integration tests without an isolated HOME). "
            "If DATE resolves to a future timestamp, the command refuses "
            "by default and requires --force (a future cutoff means "
            "\"delete everything\"). Always pair with --dry-run first."
        ),
    )

    parser.add_argument(
        "--list-summaries",
        action="store_true",
        dest="list_summaries",
        help=(
            "List offloaded conversation-history files from the agent's "
            "data directory (summaries/). Use --agent NAME to filter, "
            "--format json for machine-readable output."
        ),
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

    return parser


# ⚠️ DO NOT re-add a "seed the claude session_uuid hint" step here.
#
# An earlier version seeded ``claude_session_state.json`` with the imported
# transcript's uuid so the claude-code provider would RESUME that real file.
# That broke tool availability (observed with a live agent on 2026-05-24): the imported transcript is
# an ORDINARY ``claude`` session created OUTSIDE deep_worker. Resuming it makes
# the CLI restore THAT session's tool/permission configuration, which has no
# ``mcp__deepworker__*`` tools. deep_worker only sets ``disallowed_tools`` (it
# blocks the native Bash/Read/Edit) and never sets ``allowed_tools`` — it relies
# on a FRESH session granting the deepworker MCP tools by default. On a resumed
# external session that default does not apply, so the agent ends up with ~1
# tool instead of its full set.
#
# Correct behaviour: import writes ONLY the SQLite history (provider-agnostic).
# Under claude-code, with no hint, the provider starts a fresh session (full MCP
# tools) and carries the imported memory the normal way — synthesizing OUR own
# transcript from the LangGraph history (session_synthesis), exactly like a
# cross-provider continuation. See DESIGN_IMPORT_CLAUDE_SESSION.md §6.


async def handle_import_claude(args: argparse.Namespace) -> int:
    """Import a Claude CLI session transcript into the agent's SQLite history.

    Resolve ``--import-claude`` → parse → write to the agent's per-agent DB.
    This is provider-agnostic: it writes ONLY the message history (the agent's
    memory). It deliberately does NOT touch claude-code session state — see the
    note above on why seeding a resume hint loses the agent's tools.
    Returns 0 on success, non-zero on a user-facing error. Errors are printed
    to stderr (this runs pre-daemonize so the user sees them).
    """
    from deep_worker_cli import sessions  # noqa: PLC0415

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
    # side effect — provider-agnostic. We do NOT seed any claude-code session
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
            )

        # Build launch_config for persistent storage
        # This dict is saved to the agent's session metadata file
        # so that launch parameters survive process termination.
        launch_config = {
            "provider": config.provider,
            "model": config.resolved_model,
            "working_dir": str(config.working_dir),
            "skip_permissions": args.skip_permissions,
            "task_manager": args.task_manager,
            "memory_only": args.memory_only,
            "use_sqlite": args.use_sqlite,
            "clean": getattr(args, "clean", False),
            "auto_approve_send_message": getattr(
                args, "auto_approve_send_message", False
            ),
            "started_at": datetime.now().isoformat(),
            "pid": os.getpid(),
            "cli_args": " ".join(sys.argv[1:]),
        }

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
    """Move the calling process out of any ``session-*.scope`` /
    ``tmux-spawn-*.scope`` / ``app-*.scope`` it currently lives in,
    into a freshly-created leaf cgroup under ``user@UID.service``.

    Why this is needed in addition to ``setsid()`` + double-fork:
    the classic Unix daemonization recipe gives the daemon ``PPid=1``
    and a fresh session/process-group, but it does NOT change the
    process's cgroup membership. On systemd hosts the launching
    shell lives in a session-scope (or tmux-/app-scope), and that
    scope is what gets torn down on SSH logout / tmux pane close /
    terminal exit. When the kernel tears down the scope it kills
    every process inside it — including a fully-daemonised one,
    because cgroup membership was inherited across ``fork`` and
    ``setsid`` does not touch it.

    Production incident this guards against: 2026-05-08, an agent
    died at the moment systemd deactivated the SSH
    session-scope of the bash that originally launched it, even
    though ``loginctl enable-linger`` was set on the user. ``linger``
    keeps ``user@UID.service`` running after SSH logout, but it
    does not make session-scopes immortal — and the daemon was in
    a session-scope, not in a delegated user-manager subgroup. See
    a related internal incident analysis (not included here).

    Mechanism (cgroup v2, the unified hierarchy used by every
    systemd v240+ default):

    1. ``user@UID.service`` itself has ``cgroup.subtree_control``
       enabled (typically ``cpu memory pids``). cgroup v2's
       "no internal processes" rule then forbids writing PIDs
       directly into ``user@UID.service/cgroup.procs`` — that
       returns ``EBUSY``. So we MUST create a leaf cgroup first.
    2. ``mkdir`` a fresh ``user@UID.service/dwagent-<name>-<pid>``
       directory. The user owns ``user@UID.service`` (systemd
       grants ``Delegate=yes`` by default) so this needs no root.
    3. Write our PID into ``<that>/cgroup.procs`` to migrate.

    The leaf cgroup is intentionally per-process (PID in the name)
    so that two agents launched concurrently never collide on the
    cgroup directory. systemd's user-manager will reap the empty
    cgroup automatically when the daemon exits and the directory
    becomes empty — no explicit cleanup is needed in the happy
    path.

    Failure handling: if any step fails (no cgroup v2, no
    delegation, sysfs read-only, container without a user manager,
    EPERM on mkdir, …) the function returns silently. The daemon
    keeps running in its current cgroup; the parent-shell-death
    risk remains on those platforms but we don't have a better
    primitive without taking on a hard dbus / systemd-run runtime
    dependency.

    Idempotent: if the process is already outside the unsafe scope
    set, it is a no-op.

    Args:
        agent_name: optional name to embed in the new cgroup
            directory name, for ``systemd-cgls`` readability. The
            PID is always appended, so collisions are impossible.
    """
    try:
        with open("/proc/self/cgroup") as f:
            line = f.read().strip()
    except OSError:
        return  # /proc not available — nothing we can do
    # cgroup v2 single-line entry: ``0::/path``. cgroup v1 has
    # multiple lines with ``N:controller:/path``; we don't support
    # migration there.
    if not line.startswith("0::"):
        return
    cur = line.split("::", 1)[1]
    user_service = (
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service"
    )
    safe_name = "".join(
        c if (c.isalnum() or c in "-_") else "_"
        for c in (agent_name or "agent")
    )[:64]
    leaf_basename = f"dwagent-{safe_name}-{os.getpid()}"
    leaf_dir = f"{user_service}/{leaf_basename}"
    # Skip migration in two cases:
    # 1. We are already in our OWN leaf cgroup (re-entry, e.g. a
    #    daemonize() called twice somehow). Idempotency.
    # 2. We are in a cgroup that is genuinely safe AND owned by us
    #    (i.e. another daemon's leaf is NOT considered safe — if
    #    that daemon dies and its leaf is reaped, our process will
    #    be migrated up by the kernel, but more importantly the
    #    parent daemon's lifetime should not own ours).
    # The cgroups that DO get killed when the parent shell dies:
    #   /user.slice/user-UID.slice/session-N.scope    SSH/login
    #   /user.slice/user-UID.slice/user@UID.service/tmux-spawn-X.scope
    #   /user.slice/user-UID.slice/user@UID.service/app-foo.scope
    # AND: any /dwagent-OTHER-PID leaf belonging to a DIFFERENT
    # process. If we inherited some other daemon's leaf via fork,
    # we should still split off into our own.
    if cur.endswith(f"/{leaf_basename}"):
        return  # already in our own leaf
    # If cur is some non-shell, non-dwagent cgroup (e.g. bare
    # user@UID.service path on a host without subgroups) we still
    # try to migrate — the cost of an unnecessary migration is one
    # mkdir, the cost of NOT migrating when we should is the bug.
    try:
        # exist_ok=True: in the extremely unlikely case the dir is
        # already there (re-execed daemon, debugger restart, …) we
        # just reuse it.
        os.makedirs(leaf_dir, exist_ok=True)
        with open(f"{leaf_dir}/cgroup.procs", "w") as f:
            f.write(f"{os.getpid()}\n")
    except (OSError, PermissionError):
        # Delegation not available, sysfs read-only, container, or
        # cgroup v1. Daemonization continues; the caller is on a
        # platform where the parent-shell-death-kills-daemon problem
        # may still apply, but we have no other primitive to use
        # without taking on a hard dbus / systemd-run dependency.
        pass


def _reap_empty_cgroup_leaves() -> None:
    """rmdir every EMPTY ``dwagent-*`` leaf cgroup under
    ``user@UID.service``.

    ``_detach_from_session_scope`` creates a ``dwagent-<name>-<pid>`` leaf per
    agent and migrates the agent into it. Nothing removed those leaves when
    the agent died: the kernel does NOT auto-reap an empty cgroup directory,
    and systemd's user-manager does not rmdir delegated leaves either. They
    accumulated without bound (272 stale leaves observed 2026-05-23,
    ``nr_dying_descendants`` climbing), which made cgroup operations slower /
    flakier under pressure — the root of the intermittent
    ``test_daemonize_cgroup_detach`` failures.

    A process on cgroup v2 cannot rmdir its OWN leaf while it lives in it (it
    can't move back into ``user@UID.service`` — the no-internal-processes rule
    forbids it once subtree_control is set). So we "reap the corpses": each
    agent removes every OTHER empty dwagent leaf at startup and shutdown. Its
    own leaf is reaped by the next agent's sweep after it dies.

    Safety: only EMPTY leaves (``cgroup.procs`` has no PIDs) are removed — a
    leaf with a live process is never touched. Best-effort: never raises;
    individual rmdir failures (race with another reaper, momentary
    non-emptiness) are ignored.
    """
    user_service = (
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service"
    )
    try:
        entries = os.listdir(user_service)
    except OSError:
        return  # no cgroup v2 / no delegation / not present — nothing to do
    for name in entries:
        if not name.startswith("dwagent-"):
            continue
        leaf = f"{user_service}/{name}"
        procs_file = f"{leaf}/cgroup.procs"
        try:
            # A leaf is reapable only if it has NO processes. Reading
            # cgroup.procs is the authoritative check (a non-empty file ==
            # live members). os.rmdir refuses a non-empty cgroup anyway, so
            # this is belt-and-suspenders against killing a live agent's leaf.
            with open(procs_file) as f:
                if f.read().strip():
                    continue  # still has procs — leave it alone
            os.rmdir(leaf)
        except OSError:
            # Not a dir, already gone, raced with another reaper, or briefly
            # non-empty — all benign. Skip and move on.
            continue


def daemonize(name: str, log_dir: Path | None = None) -> None:
    """Daemonize the current process (Unix double-fork + cgroup detach).

    Args:
        name: Agent name for log files
        log_dir: Directory for log files (default: the agent's log directory
            under its data directory)
    """
    # Set up log directory
    if log_dir is None:
        log_dir = Path("~/.deep_worker").expanduser() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file paths
    stdout_log = log_dir / f"{name}.out.log"
    stderr_log = log_dir / f"{name}.err.log"

    # CRITICAL: Flush Python's stdio buffers BEFORE forking. Otherwise any
    # text that a caller printed before calling daemonize() still sits in
    # the Python-level buffer and gets duplicated into every forked process
    # (parent + child #1 + grandchild = same line printed 3 times).
    sys.stdout.flush()
    sys.stderr.flush()

    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - print info and exit
            print(f"Agent '{name}' starting in background (PID will be written to /tmp/deep_worker_{name}.pid)")
            print(f"Logs: {stdout_log}")
            sys.stdout.flush()
            sys.exit(0)
    except OSError as e:
        print(f"Fork #1 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Reap leftover empty dwagent-* leaf cgroups from agents that have since
    # died. Nothing else removes them (kernel doesn't auto-rmdir empty cgroup
    # dirs; systemd doesn't reap delegated leaves), so without this they leak
    # without bound and eventually make cgroup ops flaky (2026-05-23). A
    # process can't rmdir its own live leaf, so each agent reaps the corpses
    # left by previous agents — including, on the next launch, our own.
    _reap_empty_cgroup_leaves()

    # Detach from the launching shell's systemd cgroup (session-scope,
    # tmux-spawn-scope, app-scope, …) so that scope teardown does not
    # take the daemon down with it. See _detach_from_session_scope's
    # docstring for the why and the failure-mode rationale. Done after
    # setsid() but before the second fork, so the migrated cgroup is
    # what the final daemon process inherits.
    _detach_from_session_scope(agent_name=name)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # First child exits
            sys.exit(0)
    except OSError as e:
        print(f"Fork #2 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # IMPORTANT (fork+thread ordering): os.fork() copies ONLY the calling
    # thread, so the writer threads MUST be created HERE — after the final
    # fork, in the surviving daemon. Creating them before the fork would leave
    # a dead, un-copied thread in the daemon and a full pipe would then block
    # the agent's first write() forever.
    from deep_worker_cli.process.log_rotation import start_pipe_writer

    # Point stdout/stderr at the WRITE end of a pipe instead of straight at the
    # log file. A per-stream daemon thread drains the pipe and appends through a
    # RotatingLogWriter that enforces a hard 512MB-active + 1-backup cap, so the
    # on-disk total per stream NEVER exceeds 1 GiB — enforced in-process, with no
    # logrotate/cron needed. Captures print, logging, AND C-level writes to fd
    # 1/2 alike. See log_rotation.py and a related internal incident analysis.
    stdout_write_fd, _stdout_log_thread = start_pipe_writer(str(stdout_log))
    stderr_write_fd, _stderr_log_thread = start_pipe_writer(str(stderr_log))

    # Redirect stdin to /dev/null
    stdin_fd = os.open(os.devnull, os.O_RDONLY)

    # Duplicate file descriptors. fd 1/2 now hold a dup of each pipe's write
    # end, which keeps that write end open until the process exits — so the
    # writer threads see EOF only at teardown.
    os.dup2(stdin_fd, sys.stdin.fileno())
    os.dup2(stdout_write_fd, sys.stdout.fileno())
    os.dup2(stderr_write_fd, sys.stderr.fileno())

    # Close the redundant fds. The dup'd copies on fd 0/1/2 keep stdin/stdout/
    # stderr alive; closing these extra write-end copies still leaves fd 1/2
    # holding a write end, so the pipes are NOT prematurely EOF'd.
    os.close(stdin_fd)
    os.close(stdout_write_fd)
    os.close(stderr_write_fd)


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
    summary sections under the agent's data directory (``summaries/``) across
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
            "delete ALL data under the agents' data directory and "
            "per-agent summaries/ subdirectories."
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

    Walks the agent's data directory (``summaries/*.md``) and prints one row per
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
    console.print("[dim]Files live under the agent's data directory (summaries/).[/dim]")
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


def main() -> int:
    """Main entry point for deep-worker command.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    # Heal a legacy poisoned ``--model unknown`` before any launch path
    # reads it. See ``normalize_model_arg`` — covers direct launch,
    # --restart, and web-spawned subprocesses alike.
    normalize_model_arg(args)

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

    # Require agent name for all other operations
    if not args.name:
        print("Error: Agent name is required (or use --list)", file=sys.stderr)
        parser.print_help()
        return 2

    # Validate agent name: prevent ':tm' suffix (reserved for TaskManager)
    if ":tm" in args.name:
        print(
            f"Error: Agent name '{args.name}' cannot contain ':tm' suffix.\n"
            f"This suffix is reserved for TaskManager threads.\n"
            f"Please choose a different agent name.",
            file=sys.stderr
        )
        return 1

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
        from deep_worker_cli.process.agent_stop import stop_agent as _stop_agent, _read_agent_status

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
    from deep_worker_cli.process.registry import ProcessRegistry
    registry = ProcessRegistry()

    # (a) Reject if an agent with this name is already running. This is a
    #     pure registry lookup (file + fcntl) — sub-millisecond.
    existing = registry.lookup(args.name)
    if existing:
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
        data_dir = Path("~/.deep_worker").expanduser() / args.name
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
        log_dir = Path("~/.deep_worker").expanduser() / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob(f"{args.name}.*"):
                try:
                    log_file.unlink()
                except OSError:
                    pass  # best-effort; not worth blocking startup
        print(f"  [--clean: DB cleanup will run in background]")

    # Daemonize IMMEDIATELY. Everything below here executes in the
    # detached grandchild; parent already returned to the shell.
    daemonize(args.name)

    # --- Post-daemonize (grandchild only) -------------------------------
    # Output goes to the agent's log files under its data directory from here on.

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
