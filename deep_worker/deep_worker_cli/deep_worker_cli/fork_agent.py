"""Fork an agent's data directory.

Public API:
    fork_agent_data(source_name, new_name, *, message_id=None, home=None)

Resolves agent names to ``~/.deep_worker/<name>/`` paths and copies
all data relevant to the agent: session_metadata.json, threads.db (via
SQLite backup API — safe for a running source), summaries/, and data/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from deep_worker_cli.storage_paths import agent_root, deep_worker_home, legacy_agent_root

logger = logging.getLogger(__name__)

# ── Public API ──────────────────────────────────────────────────────────────────


def fork_agent_data(
    source_name: str,
    new_name: str,
    *,
    message_id: str | None = None,
    home: Path | None = None,
) -> None:
    """Clone *source_name* agent data into a new agent *new_name*.

    Args:
        source_name: Existing agent to fork.
        new_name: New agent name (must not exist yet).
        message_id: Optional — truncate copied history at this message ID.
        home: Override for ``Path.home()`` (tests).

    Raises:
        FileNotFoundError: source agent directory does not exist.
        FileExistsError: new agent already exists.
        ValueError: *message_id* not found in any summary file.
    """
    home = home or Path.home()
    src_dir = legacy_agent_root(source_name, home=home)
    if not src_dir.is_dir():
        # Source may live under the canonical layout (post-v6). Fallback to
        # the new location so a fork works regardless of which migration
        # state the source agent is in (DESIGN_AGENT_DBS §3.2, read fallback).
        src_dir = agent_root(source_name, home=home, create=False)
        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"Source agent '{source_name}' not found at "
                f"{legacy_agent_root(source_name, home=home)} or "
                f"{agent_root(source_name, home=home, create=False)}"
            )

    dst_dir = agent_root(new_name, home=home, create=False)

    if dst_dir.exists():
        raise FileExistsError(f"Agent '{new_name}' already exists at {dst_dir}")

    # ``agent_root(create=True)`` creates both parent (agents/) and leaf with
    # mode 0o700 (DESIGN §3.3 step 1, §5.3 privacy contract).
    agent_root(new_name, home=home, create=True)

    # ── data/ ──────────────────────────────────────────────────────────────
    src_data = src_dir / "data"
    if src_data.is_dir():
        _copy_directory(src_data, dst_dir / "data")

    # ── summaries/ ─────────────────────────────────────────────────────────
    src_summaries = src_dir / "summaries"
    if message_id:
        if not src_summaries.is_dir():
            raise ValueError(
                f"Source agent '{source_name}' has no summaries — "
                f"message_id '{message_id}' cannot be found"
            )
        _copy_summaries_truncated(src_summaries, dst_dir / "summaries", message_id)
    elif src_summaries.is_dir():
        _copy_directory(src_summaries, dst_dir / "summaries")

    # ── session_metadata.json ──────────────────────────────────────────────
    src_meta = src_dir / "session_metadata.json"
    if src_meta.exists():
        _copy_session_metadata(src_meta, dst_dir / "session_metadata.json", new_name)

    # ── threads.db ─────────────────────────────────────────────────────────
    src_db = src_dir / "threads.db"
    if src_db.exists():
        dst_db = dst_dir / "threads.db"
        _copy_threads_db(src_db, dst_db)


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _ensure_private_tree(path: Path) -> None:
    """Walk *path* and set permissions: dirs 0o700, files 0o600."""
    if not path.exists():
        return
    if path.is_dir():
        os.chmod(path, 0o700)
        for child in path.rglob("*"):
            if child.is_dir():
                os.chmod(child, 0o700)
            else:
                os.chmod(child, 0o600)
    else:
        os.chmod(path, 0o600)


def _copy_directory(src: Path, dst: Path) -> None:
    """Copy a directory tree and fix permissions. No-op if *src* does not exist."""
    if not src.is_dir():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    _ensure_private_tree(dst)


def _copy_session_metadata(src: Path, dst: Path, new_name: str) -> dict:
    """Copy session_metadata.json, update pid/started_at/cli_args."""
    data = json.loads(src.read_text())
    launch = data.get("launch_config", {})
    now = datetime.now(UTC)

    launch["pid"] = os.getpid()
    launch["started_at"] = now.isoformat()
    launch["cli_args"] = f"{new_name} --fork ..."

    data["status"] = "idle"
    data["error_message"] = None
    data["tasks_completed"] = 0
    data["message_count"] = 0
    data["last_activity"] = now.isoformat()

    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.chmod(dst, 0o600)
    return data


def _copy_threads_db(src_db: Path, dst_db: Path) -> None:
    """Copy threads.db via SQLite backup API (safe for running source).

    Also copies WAL/SHM companion files.
    """
    if not src_db.exists():
        logger.warning("Source has no threads.db — skipping")
        return

    dst_db.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(src_db))
    try:
        dst_conn = sqlite3.connect(str(dst_db))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()

    os.chmod(dst_db, 0o600)

    for ext in (".db-wal", ".db-shm"):
        f = src_db.with_suffix(ext)
        if f.exists():
            dst = dst_db.with_suffix(ext)
            shutil.copy2(f, dst)
            os.chmod(dst, 0o600)


# ── Summary-truncation helper (used when --from <message_id> is given) ─────────


def _copy_summaries_truncated(
    src_dir: Path, dst_dir: Path, message_id: str
) -> None:
    """Copy summary files but keep only entries up to and including *message_id*.

    Raises ValueError if *message_id* is not found in any JSONL file.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(dst_dir, 0o700)

    found = False
    for entry in src_dir.iterdir():
        if entry.suffix == ".jsonl":
            if _truncate_jsonl_file(entry, dst_dir / entry.name, message_id):
                found = True
        elif entry.suffix == ".md":
            # Markdown format has no message_ids — copy whole file
            shutil.copy2(entry, dst_dir / entry.name)
            os.chmod(dst_dir / entry.name, 0o600)

    if not found:
        raise ValueError(
            f"message_id '{message_id}' not found in any summary file under {src_dir}"
        )


def _truncate_jsonl_file(src: Path, dst: Path, message_id: str) -> bool:
    """Filter a JSONL summary file, keeping only entries up to *message_id*.

    Returns True if *message_id* was found in this file.
    """
    if not src.exists():
        return False

    found = False
    keep_lines: list[str] = []

    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        keep_lines.append(line)
        if entry.get("message_id") == message_id:
            found = True
            break

    if found:
        dst.write_text("\n".join(keep_lines) + "\n")
        os.chmod(dst, 0o600)
        return True

    # message_id not in this file — copy entire file
    shutil.copy2(src, dst)
    os.chmod(dst, 0o600)
    return False


def _count_checkpoints(db_path: Path) -> int:
    """Count checkpoints in a threads.db. Returns 0 if not found."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0
    finally:
        conn.close()


_DELTA_VERSION = 2
