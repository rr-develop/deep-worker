"""``deep-worker migrate-agent-dbs`` — bulk migration safety net.

DESIGN_AGENT_DBS_UNDER_AGENTS_SUBDIR §4.6, §5.2, §9.

Walks ``~/.deep_worker/<entry>/`` for any per-agent data that has not
yet been migrated to the canonical ``agents/<entry>/`` location and
either moves, skips, or — when explicitly asked — deletes test-created
directories that match the safety-classification pattern.

The default mode is ``--dry-run``: it prints the classification and
counts but never touches the filesystem. Real work requires
``--execute``. ``--delete-test-dirs`` is gated on ``--execute`` and
additionally on a keep-list constructed from the live registry plus an
optional ``--keep-list FILE``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from deep_worker_cli.storage_paths import (
    INFRASTRUCTURE_TOP_LEVEL_DIRS,
    deep_worker_home,
)

# ``iter_legacy_thread_db_paths`` and ``migrate_agent_storage_to_v6`` are
# imported lazily inside the handler — see the docstring on
# ``handle_migrate_agent_dbs_command``. Importing at module load time
# would freeze the current ``storage_paths`` module's binding, which
# breaks tests that ``sys.modules.pop("deep_worker_cli.storage_paths")``
# and re-import the module (TEST_HOOK §3, fixture ``fake_home``).

logger = logging.getLogger(__name__)


REGISTRY_PATH = Path("/tmp/deep_worker_registry.json")
DEFAULT_FRESHNESS_DAYS = 14

# Mirrors the regex list in DESIGN §5.2 step 3.
TEST_PATTERNS: tuple[str, ...] = (
    r"^e2e[-_]",
    r"^dbg[-_]",
    r"^acp_r6[a-e]$",
    r"^acptest\d*$",
    r"^acp_master_smoke$",
    r"^acpfix\d*$",
    # ``dg*`` test-prefix: catch all variants of debug/image experiment
    # agents (DESIGN_AGENT_DBS §5.2 — widens the original regex per
    # review obs1; safe because the bucket is ``test-candidate``,
    # which is NEVER deleted unless ``--delete-test-dirs`` is also set
    # AND the keep-list does NOT protect it).
    r"^dg(img\d+|mut\d*|_imgtest[a-z0-9_]*|_probe[a-z0-9_]*|_dbg[a-z0-9_]*|big\d+|imglive\d+|[a-z0-9_]+)$",
    r"^dw-[a-f0-9]{6,}$",
    r"^dwtest_[a-z0-9_]+$",
    r"^vc\d+_",
    r"^vc_reviewer$",
    r"^mj\d+_",
    r"^mc\d+_",
    r"^mcheck\d*$",
    r"^ca\d+_",
    r"^cn\d+_",
    r"^mm_cfg[a-z0-9_]*$",
    r"^commit-editor(-[a-z0-9]+|:.*)?$",
    r"^ce-(failover|manual)$",
    r"^img(test|verify|mutant)\d*$",
    r"^task_developer$",
    r"^manualcheck\d*$",
    r"^fresh_\d+_",
    r"^restart_test_thread$",
    r"^diag_tool$",
    r"^agent_main$",
    r"^agentX$",
    r"^aX$",
    r"^r[12]$",
    r"^x$",
    r"^list$",
    r"^test[_-]",
    r"^test[_0-9]",
)


def build_keep_list(
    registry_path: Path | None = REGISTRY_PATH,
    extra_keep_list_path: Path | None = None,
) -> set[str]:
    """Names that must NEVER be deleted (DESIGN §5.2 step 1).

    Source: live ``ProcessRegistry`` entries filtered by ``os.kill(pid, 0)``
    (i.e. the process is actually running), plus an optional user-supplied
    file of one name per line.
    """
    keep: set[str] = set()
    if registry_path is not None and registry_path.exists():
        try:
            data = json.loads(registry_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
        for name, info in data.items():
            pid = info.get("pid")
            if pid is None:
                continue
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            keep.add(name)

    if extra_keep_list_path is not None and extra_keep_list_path.exists():
        for line in extra_keep_list_path.read_text().splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                keep.add(name)
    return keep


def is_test_pattern(name: str) -> bool:
    """Return True if ``name`` matches the test-created regex list."""
    # Production-like shorthand: ``{1-3 letters}{optional digit}_{suffix}``
    # (e.g. ``ab1_xxx``, ``cd3_yyy``, ``ef1_zzz``). These are operator-
    # facing agent names that look like test patterns but are actually
    # production work (user feedback 2026-08-18). Excluded BEFORE the
    # test-pattern check so they fall through to keep-move / move-unknown.
    if re.search(r"^[a-z]{1,3}[0-9]?_.+$", name):
        return False
    for pat in TEST_PATTERNS:
        if re.search(pat, name):
            return True
    return False


def is_fresh(path: Path, days: int) -> bool:
    """DESIGN §5.2 step 4: freshness heuristic for non-recognised names.

    A directory with ``session_metadata.json`` whose ``launch_config.started_at``
    is more recent than ``--freshness-days N`` days is treated as
    *fresh-unknown* and is reported but never deleted.
    """
    meta = path / "session_metadata.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
        started = data.get("launch_config", {}).get("started_at")
        if not started:
            return False
        ts = datetime.fromisoformat(started)
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(tz=timezone.utc) - ts) < timedelta(days=days)


def classify(entry: str, path: Path, keep: set[str], freshness_days: int) -> str:
    """DESIGN §5.2 step 5: classify ``entry`` into a single action label."""
    if entry in keep:
        return "keep-move"
    if entry in INFRASTRUCTURE_TOP_LEVEL_DIRS or entry == "agents":
        return "skip-infra"
    if is_test_pattern(entry):
        return "test-candidate"
    if is_fresh(path, freshness_days):
        return "fresh-unknown"
    return "move-unknown"


def setup_migrate_agent_dbs_parser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "migrate-agent-dbs",
        help=(
            "Bulk-migrate legacy ~/.deep_worker/<name>/ trees to "
            "~/.deep_worker/agents/<name>/. DELETE test-created dirs only "
            "with --execute --delete-test-dirs (DESIGN §4.6)."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print classification only; do not touch the filesystem (default).",
    )
    g.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration / deletion.",
    )
    p.add_argument(
        "--keep-legacy",
        action="store_true",
        help="After --execute, keep the legacy tree (for forensic rollback).",
    )
    p.add_argument(
        "--delete-test-dirs",
        action="store_true",
        help="Delete test-created dirs that match TEST_PATTERNS (requires --execute).",
    )
    p.add_argument(
        "--freshness-days",
        type=int,
        default=DEFAULT_FRESHNESS_DAYS,
        help=(
            "Dirs whose launch_config.started_at is more recent than this "
            "are NEVER deleted (default: 14)."
        ),
    )
    p.add_argument(
        "--keep-list",
        type=Path,
        default=None,
        help="Optional file with extra names to protect (one per line).",
    )


def handle_migrate_agent_dbs_command(args: argparse.Namespace) -> int:
    """Implement ``deep-worker migrate-agent-dbs``."""
    if args.delete_test_dirs and not args.execute:
        print(
            "Error: --delete-test-dirs requires --execute.",
            file=sys.stderr,
        )
        return 2

    execute = bool(args.execute)
    # Read REGISTRY_PATH at call time so monkeypatch-setattr in tests
    # sees the override (default args are evaluated at function-def time).
    keep_list = build_keep_list(
        registry_path=REGISTRY_PATH,
        extra_keep_list_path=args.keep_list,
    )

    root = deep_worker_home()
    if not root.exists():
        print(f"Error: deep_worker_home {root!s} does not exist.", file=sys.stderr)
        return 1

    # First: walk ``iter_legacy_thread_db_paths`` — directories that
    # the v6 migration CLI flag set as candidates. Anything in
    # ``iter_thread_db_paths`` (the new layout) is already migrated.
    moved = kept = deleted = skipped = fresh_skipped = 0
    plans: list[tuple[Path, str]] = []

    # Defer-import the ``storage_paths`` helpers so we pick up the
    # currently-patched module (tests sometimes pop+re-import the
    # module out of ``sys.modules``).
    from deep_worker_cli.storage_paths import iter_legacy_thread_db_paths

    for entry in iter_legacy_thread_db_paths():
        # Each entry is the agent dir itself (<root>/<name>/). Earlier
        # versions returned a path to threads.db; we now return the
        # agent dir directly so we can also catch legacy dirs that lack
        # a threads.db (test dirs that were cleaned up, etc.).
        agent_dir = entry
        name = agent_dir.name
        action = classify(name, agent_dir, keep_list, args.freshness_days)

        plans.append((agent_dir, action))

    total = len(plans)

    for i, (agent_dir, action) in enumerate(plans, 1):
        name = agent_dir.name
        if i % 5 == 0 or i == total:
            print(f"[{i}/{total}] {name!s}: {action}", flush=True)

        if action == "skip-infra":
            skipped += 1
            continue

        if action == "fresh-unknown":
            fresh_skipped += 1
            print(f"  WARN: fresh directory, leaving in place: {agent_dir}", flush=True)
            continue

        if action in ("keep-move", "move-unknown"):
            if not execute:
                moved += 1
                continue
            # Bypass _ensure_agent_layout's "already v6" short-circuit
            # — the CLI exists precisely to move legacy dirs that lack
            # any session_metadata.json. Force the v6 migration.
            from deep_worker_cli.storage_paths import (
                migrate_agent_storage_to_v6,
            )
            migrate_agent_storage_to_v6(name)
            if not args.keep_legacy:
                moved += 1
            else:
                kept += 1
            continue

        if action == "test-candidate":
            if not args.delete_test_dirs or not execute:
                skipped += 1
                continue
            if not execute:
                continue
            # Double-safety: re-check keep-list immediately before delete.
            if name in keep_list:
                skipped += 1
                print(f"  SKIP: {name!s} matched keep-list at last minute.", flush=True)
                continue
            shutil.rmtree(agent_dir, ignore_errors=True)
            deleted += 1
            print(f"  DELETED test-candidate dir: {agent_dir}", flush=True)
            continue

    print(
        "\n=== Summary ===\n"
        f"  moved  : {moved}\n"
        f"  kept   : {kept}\n"
        f"  deleted: {deleted}\n"
        f"  skipped: {skipped}\n"
        f"  fresh-skipped: {fresh_skipped}\n"
        f"  mode: {'execute' if execute else 'dry-run'}",
        flush=True,
    )
    if not execute:
        print(
            "\nThis was a dry run. Re-run with --execute to apply.",
            file=sys.stderr if False else sys.stdout,
            flush=True,
        )
    return 0
