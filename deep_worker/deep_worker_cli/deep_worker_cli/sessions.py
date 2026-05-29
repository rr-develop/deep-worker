"""Thread management using LangGraph's built-in checkpoint persistence."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.table import Table

from deep_worker_cli.config import COLORS, console
from deep_worker_cli.storage_paths import (
    agent_name_from_thread_id,
    ensure_private_file,
    iter_thread_db_paths,
    threads_db_path,
)

# Patch aiosqlite.Connection to add is_alive() method required by
# langgraph-checkpoint>=2.1.0
# See: https://github.com/langchain-ai/langgraph/issues/6583
if not hasattr(aiosqlite.Connection, "is_alive"):

    def _is_alive(self: aiosqlite.Connection) -> bool:
        """Check if the connection is still alive.

        Returns:
            True if connection is alive, False otherwise.
        """
        return self._connection is not None

    # Dynamically adding a method to aiosqlite.Connection at runtime.
    # Type checkers can't understand this monkey-patch, so we suppress the
    # unresolved-attribute error that would otherwise be raised.
    aiosqlite.Connection.is_alive = _is_alive  # ty: ignore[unresolved-attribute]


def _format_timestamp(iso_timestamp: str | None) -> str:
    """Format ISO timestamp for display (e.g., 'Dec 30, 6:10pm').

    Returns:
        Formatted timestamp string or empty string if invalid.
    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
        return (
            dt.strftime("%b %d, %-I:%M%p")
            .lower()
            .replace("am", "am")
            .replace("pm", "pm")
        )
    except (ValueError, TypeError):
        return ""


def get_db_path(agent_name: str | None = None) -> Path:
    """Get path to SQLite database file.

    When ``agent_name`` is provided, returns a per-agent database file
    (``sessions_{agent_name}.db``). This avoids SQLite "database is locked"
    errors when multiple agents write concurrently.

    When ``agent_name`` is ``None``, returns the legacy global
    ``sessions.db`` for backward compatibility (e.g. ``--list`` command
    that queries all agents).

    Args:
        agent_name: Agent name for per-agent isolation. ``None`` for global DB.

    Returns:
        Path to the SQLite database file.
    """
    return threads_db_path(agent_name)


def _agent_name_from_thread_id(thread_id: str) -> str:
    """Derive an agent name from a thread_id.

    Convention used throughout the codebase:

    - Main agent thread:      thread_id == agent_name
    - TaskManager thread:     thread_id == f"{agent_name}:tm"

    See ``_migrate_from_global_db`` (same mapping).
    """
    return agent_name_from_thread_id(thread_id)


def iter_all_db_paths() -> list[Path]:
    """Return all session DB files that may contain checkpoint data.

    Includes every ``sessions_<agent>.db`` file in the same directory as
    the legacy global ``sessions.db`` plus the legacy global itself (if it
    exists).  Used by offline commands that operate across all agents
    (``--list``, ``--remove-before``, ...).

    The directory is derived from ``get_db_path(None)`` so tests that
    monkey-patch ``get_db_path`` to point into ``tmp_path`` transparently
    restrict discovery to that directory.
    """
    legacy = get_db_path(None)
    default_legacy = Path.home() / ".deepagents" / "sessions.db"
    if legacy != default_legacy:
        db_dir = legacy.parent
        if not db_dir.exists():
            return []
        paths: list[Path] = sorted(db_dir.glob("sessions_*.db"))
        if legacy.exists() and legacy not in paths:
            paths.append(legacy)
        return paths
    return iter_thread_db_paths()


def _resolve_db_path_for_thread(thread_id: str) -> Path | None:
    """Pick the DB file most likely to hold ``thread_id``.

    1. ``sessions_<agent>.db`` where agent is derived from thread_id.
    2. Legacy global ``sessions.db`` (only if the per-agent file does
       not exist – keeps pre-migration installs working).
    3. ``None`` if neither exists.
    """
    agent = _agent_name_from_thread_id(thread_id)
    per_agent = get_db_path(agent)
    if per_agent.exists():
        return per_agent
    legacy = get_db_path(None)
    if legacy.exists():
        return legacy
    return None


def generate_thread_id() -> str:
    """Generate a new 8-char hex thread ID.

    Returns:
        8-character hexadecimal string.
    """
    return uuid.uuid4().hex[:8]


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    """Check if a table exists in the database.

    Returns:
        True if table exists, False otherwise.
    """
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    async with conn.execute(query, (table,)) as cursor:
        return await cursor.fetchone() is not None


async def list_threads(
    agent_name: str | None = None,
    limit: int = 20,
    include_message_count: bool = False,
) -> list[dict]:
    """List threads from checkpoints table.

    Iterates across every ``sessions_*.db`` file in the agent data directory
    plus the legacy global ``sessions.db`` and merges results, so offline
    threads of all agents are visible (not only those in the legacy
    global DB).

    Args:
        agent_name: Optional filter by agent name.  When provided, only
            the matching ``sessions_<agent>.db`` is read (+ the legacy
            global as a fallback for pre-migration data).
        limit: Maximum number of threads to return (applied after merge).
        include_message_count: Whether to include message counts.

    Returns:
        List of thread dicts with `thread_id`, `agent_name`, `updated_at`,
            and optionally `message_count`, sorted by ``updated_at`` desc.
    """
    # Decide which DB files to read.
    if agent_name is not None:
        db_paths: list[Path] = []
        per_agent = get_db_path(agent_name)
        if per_agent.exists():
            db_paths.append(per_agent)
        legacy = get_db_path(None)
        if legacy.exists():
            db_paths.append(legacy)
    else:
        db_paths = iter_all_db_paths()

    threads: list[dict] = []
    seen_thread_ids: set[str] = set()

    for db_path in db_paths:
        try:
            async with aiosqlite.connect(str(db_path), timeout=30.0) as conn:
                if not await _table_exists(conn, "checkpoints"):
                    continue

                if agent_name:
                    query = """
                        SELECT thread_id,
                               json_extract(metadata, '$.agent_name') as agent_name,
                               MAX(json_extract(metadata, '$.updated_at')) as updated_at
                        FROM checkpoints
                        WHERE json_extract(metadata, '$.agent_name') = ?
                        GROUP BY thread_id
                    """
                    params: tuple = (agent_name,)
                else:
                    query = """
                        SELECT thread_id,
                               json_extract(metadata, '$.agent_name') as agent_name,
                               MAX(json_extract(metadata, '$.updated_at')) as updated_at
                        FROM checkpoints
                        GROUP BY thread_id
                    """
                    params = ()

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                file_threads: list[dict] = []
                for r in rows:
                    tid = r[0]
                    if tid in seen_thread_ids:
                        # Same thread_id may appear in both per-agent and
                        # legacy global DB right after migration — keep the
                        # first (per-agent, sorted first).
                        continue
                    file_threads.append(
                        {
                            "thread_id": tid,
                            "agent_name": r[1],
                            "updated_at": r[2],
                        }
                    )
                    seen_thread_ids.add(tid)

                if include_message_count and file_threads:
                    serde = JsonPlusSerializer()
                    for thread in file_threads:
                        thread["message_count"] = (
                            await _count_messages_from_checkpoint(
                                conn, thread["thread_id"], serde
                            )
                        )

                threads.extend(file_threads)
        except aiosqlite.OperationalError:
            # Skip corrupt / locked files — best-effort listing.
            continue

    # Sort by updated_at DESC (newest first), then apply limit.
    threads.sort(key=lambda t: t.get("updated_at") or "", reverse=True)
    return threads[:limit]


async def list_all_threads(
    agent_name: str | None = None,
    sort_by: str = "thread_id",
    reverse: bool = False,
    include_message_count: bool = False,
) -> list[dict]:
    """List ALL threads without pagination.

    Iterates across per-agent ``sessions_<name>.db`` files and the legacy
    global ``sessions.db``, merging results so nothing is missed after the
    per-agent storage migration.

    Args:
        agent_name: Optional filter by agent name.
        sort_by: Sort field: 'thread_id' (default), 'agent', 'updated'.
        reverse: Reverse sort order (default: False).
        include_message_count: Include message counts (slow, default: False).

    Returns:
        List of thread dicts with:
        - thread_id: str
        - agent_name: str | None
        - updated_at: str (ISO timestamp)
        - message_count: int (only if include_message_count=True)
    """
    if agent_name is not None:
        db_paths: list[Path] = []
        per_agent = get_db_path(agent_name)
        if per_agent.exists():
            db_paths.append(per_agent)
        legacy = get_db_path(None)
        if legacy.exists():
            db_paths.append(legacy)
    else:
        db_paths = iter_all_db_paths()

    threads: list[dict] = []
    seen: set[str] = set()

    for db_path in db_paths:
        try:
            async with aiosqlite.connect(str(db_path), timeout=30.0) as conn:
                if not await _table_exists(conn, "checkpoints"):
                    continue

                if agent_name:
                    query = """
                        SELECT thread_id,
                               json_extract(metadata, '$.agent_name') as agent_name,
                               MAX(json_extract(metadata, '$.updated_at')) as updated_at
                        FROM checkpoints
                        WHERE json_extract(metadata, '$.agent_name') = ?
                        GROUP BY thread_id
                    """
                    params: tuple = (agent_name,)
                else:
                    query = """
                        SELECT thread_id,
                               json_extract(metadata, '$.agent_name') as agent_name,
                               MAX(json_extract(metadata, '$.updated_at')) as updated_at
                        FROM checkpoints
                        GROUP BY thread_id
                    """
                    params = ()

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                file_threads: list[dict] = []
                for r in rows:
                    tid = r[0]
                    if tid in seen:
                        continue
                    file_threads.append(
                        {
                            "thread_id": tid,
                            "agent_name": r[1],
                            "updated_at": r[2],
                        }
                    )
                    seen.add(tid)

                if include_message_count and file_threads:
                    serde = JsonPlusSerializer()
                    for thread in file_threads:
                        thread["message_count"] = (
                            await _count_messages_from_checkpoint(
                                conn, thread["thread_id"], serde
                            )
                        )

                threads.extend(file_threads)
        except aiosqlite.OperationalError:
            continue

    # Sort threads in Python (allows complex sort logic)
    if sort_by == "thread_id":
        threads.sort(key=lambda t: t["thread_id"].lower(), reverse=reverse)
    elif sort_by == "agent":
        # NULL agent_names first, then alphabetical with secondary sort by thread_id.
        threads.sort(
            key=lambda t: ((t["agent_name"] or "").lower(), t["thread_id"].lower()),
            reverse=reverse,
        )
    elif sort_by == "updated":
        # Most recent first by default (descending)
        threads.sort(key=lambda t: t["updated_at"] or "", reverse=not reverse)

    return threads


async def _count_messages_from_checkpoint(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> int:
    """Count messages from the most recent checkpoint blob.

    Messages are stored in the checkpoint blob (channel_values).
    This function deserializes the checkpoint and counts them.

    Args:
        conn: Database connection.
        thread_id: The thread ID to count messages for.
        serde: Serializer for decoding checkpoint data.

    Returns:
        Number of messages in the checkpoint, or 0 if not found.
    """
    query = """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id DESC
        LIMIT 1
    """
    async with conn.execute(query, (thread_id,)) as cursor:
        row = await cursor.fetchone()
        if not row or not row[0] or not row[1]:
            return 0

        type_str, checkpoint_blob = row
        try:
            data = serde.loads_typed((type_str, checkpoint_blob))
            channel_values = data.get("channel_values", {})
            messages = channel_values.get("messages", [])
            return len(messages)
        except (ValueError, TypeError, KeyError):
            # If deserialization fails, fall back to 0
            return 0


async def get_most_recent(agent_name: str | None = None) -> str | None:
    """Get most recent thread_id, optionally filtered by agent.

    Returns:
        Most recent thread_id or None if no threads exist.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        if agent_name:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params: tuple = (agent_name,)
        else:
            query = (
                "SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            )
            params = ()

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_thread_agent(thread_id: str) -> str | None:
    """Get agent_name for a thread.

    Returns:
        Agent name associated with the thread, or None if not found.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        query = """
            SELECT json_extract(metadata, '$.agent_name')
            FROM checkpoints
            WHERE thread_id = ?
            LIMIT 1
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def thread_exists(thread_id: str) -> bool:
    """Check if a thread exists in checkpoints.

    Returns:
        True if thread exists, False otherwise.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        query = "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1"
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row is not None


async def find_similar_threads(thread_id: str, limit: int = 3) -> list[str]:
    """Find threads whose IDs start with the given prefix.

    Args:
        thread_id: Prefix to match against thread IDs.
        limit: Maximum number of matching threads to return.

    Returns:
        List of thread IDs that begin with the given prefix.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        query = """
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id LIKE ?
            ORDER BY thread_id
            LIMIT ?
        """
        prefix = thread_id + "%"
        async with conn.execute(query, (prefix, limit)) as cursor:
            rows = await cursor.fetchall()
            return [r[0] for r in rows]


async def delete_thread(thread_id: str) -> bool:
    """Delete thread checkpoints.

    Returns:
        True if thread was deleted, False if not found.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        cursor = await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        deleted = cursor.rowcount > 0
        if await _table_exists(conn, "writes"):
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        return deleted


def _migrate_from_global_db(agent_name: str, agent_db_path: Path) -> None:
    """Migrate agent data from legacy global sessions.db to per-agent DB.

    Called once when a per-agent DB is first created. Copies checkpoints and
    writes for the agent (including :tm thread) from the global file.

    This is a one-time migration — subsequent runs skip it because the
    per-agent DB already has data.

    Args:
        agent_name: Agent name to migrate.
        agent_db_path: Path to the per-agent DB file.
    """
    import sqlite3

    global_path = get_db_path()  # Legacy sessions.db
    if not global_path.exists():
        return

    # Skip if per-agent DB already has data (migration already done)
    if agent_db_path.exists() and agent_db_path.stat().st_size > 8192:
        return

    try:
        src = sqlite3.connect(str(global_path))
        dst = sqlite3.connect(str(agent_db_path))

        # Create tables from source schema
        for row in src.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall():
            if row[0]:
                dst.execute(row[0].replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS"))

        tids = [agent_name, f"{agent_name}:tm"]
        migrated_any = False

        for table in ["checkpoints", "writes"]:
            try:
                rows = src.execute(
                    f"SELECT * FROM {table} WHERE thread_id IN (?,?)", tids
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            if rows:
                ph = ",".join(["?"] * len(rows[0]))
                dst.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({ph})", rows)
                migrated_any = True

        if migrated_any:
            dst.commit()

        dst.close()
        src.close()
    except Exception:
        # Migration is best-effort — don't crash the agent
        pass


@asynccontextmanager
async def get_checkpointer(agent_name: str | None = None) -> AsyncIterator["DeltaSqliteCheckpointer"]:
    """Get DeltaSqliteCheckpointer for a database.

    Uses delta-based checkpointing: stores only changed channel values
    in each checkpoint, not full state copies. Backward compatible with
    existing v1 (full snapshot) data from AsyncSqliteSaver.

    When ``agent_name`` is provided, uses a per-agent database file to avoid
    SQLite lock contention between concurrent agents. On first use, migrates
    existing data from the legacy global ``sessions.db``.

    When ``None``, uses the legacy global ``sessions.db``.

    Args:
        agent_name: Agent name for per-agent DB isolation.

    Yields:
        DeltaSqliteCheckpointer instance for checkpoint persistence.
    """
    from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer

    db_path = get_db_path(agent_name)

    # One-time migration from global DB to per-agent DB
    if agent_name is not None:
        _migrate_from_global_db(agent_name, db_path)

    async with DeltaSqliteCheckpointer.from_conn_string(str(db_path)) as checkpointer:
        ensure_private_file(db_path)
        yield checkpointer


async def list_threads_command(
    agent_name: str | None = None,
    limit: int = 20,
) -> None:
    """CLI handler for `deepagents threads list`.

    Fetches and displays a table of recent conversation threads, optionally
    filtered by agent name.

    Args:
        agent_name: Only show threads belonging to this agent.

            When `None`, threads for all agents are shown.
        limit: Maximum number of threads to display.
    """
    threads = await list_threads(agent_name, limit=limit, include_message_count=True)

    if not threads:
        if agent_name:
            console.print(
                f"[yellow]No threads found for agent '{agent_name}'.[/yellow]"
            )
        else:
            console.print("[yellow]No threads found.[/yellow]")
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title = (
        f"Recent threads for '{agent_name}' (last {limit})"
        if agent_name
        else f"Recent Threads (last {limit})"
    )

    table = Table(
        title=title, show_header=True, header_style=f"bold {COLORS['primary']}"
    )
    table.add_column("Thread ID", style="bold")
    table.add_column("Agent")
    table.add_column("Messages", justify="right")
    table.add_column("Last Used", style="dim")

    for t in threads:
        table.add_row(
            t["thread_id"],
            t["agent_name"] or "unknown",
            str(t.get("message_count", 0)),
            _format_timestamp(t.get("updated_at")),
        )

    console.print()
    console.print(table)
    console.print()


async def delete_thread_command(thread_id: str) -> None:
    """CLI handler for: deepagents threads delete."""
    deleted = await delete_thread(thread_id)

    if deleted:
        console.print(f"[green]Thread '{thread_id}' deleted.[/green]")
    else:
        console.print(f"[red]Thread '{thread_id}' not found.[/red]")


async def clean_agent_history_db_only(
    agent_name: str,
) -> tuple[int, int]:
    """Delete only DB rows (checkpoints + writes) for this agent.

    Does NOT touch agent data directory or logs. Intended to be called
    from the daemonized grandchild during ``deep-worker --clean`` startup.

    After the per-agent DB migration the primary storage is
    ``sessions_<agent>.db``; that file is removed entirely.  For backward
    compatibility we also scrub any stale rows from the legacy global
    ``sessions.db`` (from pre-migration installs).

    Args:
        agent_name: Name of the agent whose DB rows to remove.

    Returns:
        Tuple ``(checkpoints_deleted, writes_deleted)`` – counts from
        both the per-agent file and the legacy global are summed.
    """
    import os

    checkpoints_deleted = 0
    writes_deleted = 0

    per_agent_path = get_db_path(agent_name)
    legacy_path = get_db_path(None)

    # 1. Per-agent DB: count then delete the file entirely (atomic, no lock
    # contention, exactly what the user expects from --clean).  Skip when
    # the per-agent and legacy paths collapse to the same file (happens
    # only in monkey-patched test setups).
    if per_agent_path.exists() and per_agent_path != legacy_path:
        try:
            async with aiosqlite.connect(
                str(per_agent_path), timeout=30.0
            ) as conn:
                if await _table_exists(conn, "checkpoints"):
                    async with conn.execute(
                        "SELECT COUNT(*) FROM checkpoints"
                    ) as cursor:
                        row = await cursor.fetchone()
                        checkpoints_deleted += row[0] if row else 0
                if await _table_exists(conn, "writes"):
                    async with conn.execute(
                        "SELECT COUNT(*) FROM writes"
                    ) as cursor:
                        row = await cursor.fetchone()
                        writes_deleted += row[0] if row else 0
        except aiosqlite.OperationalError:
            # File unreadable – best-effort clean-up, unlink anyway below.
            pass

        try:
            os.unlink(per_agent_path)
        except OSError:
            # Best-effort; continue to legacy scrub.
            pass

    # 2. Legacy global DB: remove any rows still tagged for this agent.
    if legacy_path.exists():
        try:
            thread_ids: list[str] = []
            async with aiosqlite.connect(
                str(legacy_path), timeout=30.0
            ) as conn:
                if await _table_exists(conn, "checkpoints"):
                    async with conn.execute(
                        """
                        SELECT DISTINCT thread_id
                        FROM checkpoints
                        WHERE json_extract(metadata, '$.agent_name') = ?
                        """,
                        (agent_name,),
                    ) as cursor:
                        rows = await cursor.fetchall()
                        thread_ids = [row[0] for row in rows]

            async with aiosqlite.connect(
                str(legacy_path), timeout=30.0
            ) as conn:
                if await _table_exists(conn, "checkpoints"):
                    cursor = await conn.execute(
                        "DELETE FROM checkpoints "
                        "WHERE json_extract(metadata, '$.agent_name') = ?",
                        (agent_name,),
                    )
                    checkpoints_deleted += cursor.rowcount or 0
                if await _table_exists(conn, "writes") and thread_ids:
                    placeholders = ",".join("?" * len(thread_ids))
                    cursor = await conn.execute(
                        f"DELETE FROM writes WHERE thread_id IN ({placeholders})",
                        thread_ids,
                    )
                    writes_deleted += cursor.rowcount or 0
                await conn.commit()
        except aiosqlite.OperationalError:
            pass

    return checkpoints_deleted, writes_deleted


async def clean_agent_history(
    agent_name: str,
    *,
    _home_dir: Path | None = None,  # For testing only
) -> None:
    """Clean all history and state for an agent.

    This removes:
    - All checkpoints/threads from sessions.db for this agent
    - All writes from sessions.db for this agent's threads
    - Agent data directory (under the user's home)
    - Agent log files (under the user's home)

    This does NOT remove:
    - User configuration (AGENTS.md under the agent's data directory)
    - User skills (under the agent's data directory)
    - User subagents (under the agent's data directory)

    Args:
        agent_name: Name of the agent to clean
        _home_dir: Home directory override (for testing only, not public API)

    Note:
        This function is idempotent and handles non-existent data gracefully.
    """
    import os
    import shutil

    home_dir = _home_dir if _home_dir is not None else Path.home()

    checkpoints_deleted = 0
    writes_deleted = 0

    # Resolve DB paths.  In the common runtime case both paths come from
    # ``get_db_path`` and point at ``home_dir/.deepagents/sessions_<agent>.db``
    # and ``home_dir/.deepagents/sessions.db`` respectively.  Two special
    # situations:
    #   - Unit tests that pass ``_home_dir`` without patching ``get_db_path``:
    #     use the home_dir-relative paths so the fake home is honoured.
    #   - Unit tests that patch ``get_db_path`` to return one specific file
    #     regardless of arg: both resolve to the same path, the guard
    #     ``per_agent_path != legacy_path`` below skips the per-agent
    #     unlink step and the legacy scrub cleans the rows.
    if _home_dir is not None:
        # Prefer the monkey-patched ``get_db_path`` if the patch collapses
        # both paths onto a single file outside home_dir (legacy tests).
        patched_per_agent = get_db_path(agent_name)
        patched_legacy = get_db_path(None)
        if patched_per_agent == patched_legacy:
            per_agent_path = patched_per_agent
            legacy_path = patched_legacy
        else:
            per_agent_path = home_dir / ".deepagents" / f"sessions_{agent_name}.db"
            legacy_path = home_dir / ".deepagents" / "sessions.db"
    else:
        per_agent_path = get_db_path(agent_name)
        legacy_path = get_db_path(None)

    # 1. Per-agent DB: count rows then remove the file.  Skip when
    # per-agent and legacy point at the same file (monkey-patched tests).
    if per_agent_path.exists() and per_agent_path != legacy_path:
        try:
            async with aiosqlite.connect(
                str(per_agent_path), timeout=30.0
            ) as conn:
                if await _table_exists(conn, "checkpoints"):
                    async with conn.execute(
                        "SELECT COUNT(*) FROM checkpoints"
                    ) as cursor:
                        row = await cursor.fetchone()
                        checkpoints_deleted += row[0] if row else 0
                if await _table_exists(conn, "writes"):
                    async with conn.execute(
                        "SELECT COUNT(*) FROM writes"
                    ) as cursor:
                        row = await cursor.fetchone()
                        writes_deleted += row[0] if row else 0
        except aiosqlite.OperationalError:
            pass
        try:
            os.unlink(per_agent_path)
        except OSError:
            pass

    # 2. Legacy global DB: scrub any rows tagged with this agent's name
    # (pre-migration data that may still be lying around).
    if legacy_path.exists():
        try:
            thread_ids: list[str] = []
            async with aiosqlite.connect(
                str(legacy_path), timeout=30.0
            ) as conn:
                if await _table_exists(conn, "checkpoints"):
                    query = """
                        SELECT DISTINCT thread_id
                        FROM checkpoints
                        WHERE json_extract(metadata, '$.agent_name') = ?
                    """
                    async with conn.execute(query, (agent_name,)) as cursor:
                        rows = await cursor.fetchall()
                        thread_ids = [row[0] for row in rows]

            async with aiosqlite.connect(
                str(legacy_path), timeout=30.0
            ) as conn:
                if await _table_exists(conn, "checkpoints"):
                    cursor = await conn.execute(
                        "DELETE FROM checkpoints "
                        "WHERE json_extract(metadata, '$.agent_name') = ?",
                        (agent_name,),
                    )
                    checkpoints_deleted += cursor.rowcount or 0

                if await _table_exists(conn, "writes") and thread_ids:
                    placeholders = ",".join("?" * len(thread_ids))
                    query = f"DELETE FROM writes WHERE thread_id IN ({placeholders})"
                    cursor = await conn.execute(query, thread_ids)
                    writes_deleted += cursor.rowcount or 0

                await conn.commit()
        except aiosqlite.OperationalError:
            pass

    # 3. Remove agent data directory
    data_dir = home_dir / Path("~/.deep_worker").name / agent_name
    data_dir_removed = False
    if data_dir.exists():
        shutil.rmtree(data_dir)
        data_dir_removed = True

    # 4. Remove log files
    log_dir = home_dir / Path("~/.deep_worker").name / "logs"
    log_files_removed = 0
    if log_dir.exists():
        for log_file in log_dir.glob(f"{agent_name}.*"):
            log_file.unlink()
            log_files_removed += 1

    # Print summary
    console.print(f"[green]✓ Cleaned history for agent '{agent_name}':[/green]")
    console.print(f"  [dim]- Removed {checkpoints_deleted} checkpoints from database[/dim]")
    if writes_deleted > 0:
        console.print(f"  [dim]- Removed {writes_deleted} writes from database[/dim]")
    if data_dir_removed:
        console.print(f"  [dim]- Removed data directory: {data_dir}[/dim]")
    if log_files_removed > 0:
        console.print(
            f"  [dim]- Removed {log_files_removed} log file(s)[/dim]"
        )
    console.print(
        f"  [dim]- Preserved user config under the agent's data directory[/dim]"
    )


def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object.

    Supports:
    - ISO 8601: "2026-01-01", "2026-01-01T10:30:00", "2026-01-01T10:30:00+00:00"
    - Relative: "7d" (7 days), "2w" (2 weeks), "3m" (3 months), "1y" (1 year)

    Args:
        date_str: Date string to parse

    Returns:
        datetime object (timezone-aware, UTC)

    Raises:
        ValueError: If date format is invalid
    """
    import re
    from datetime import timedelta, UTC

    if not date_str:
        raise ValueError("Invalid date format: empty string")

    # Try relative format first (Nd, Nw, Nm, Ny)
    relative_pattern = r'^(\d+)([dwmy])$'
    match = re.match(relative_pattern, date_str.lower())

    if match:
        amount, unit = match.groups()
        amount = int(amount)
        now = datetime.now(UTC)

        if unit == 'd':  # days
            return now - timedelta(days=amount)
        elif unit == 'w':  # weeks
            return now - timedelta(weeks=amount)
        elif unit == 'm':  # months (approximate: 30 days)
            return now - timedelta(days=amount * 30)
        elif unit == 'y':  # years (approximate: 365 days)
            return now - timedelta(days=amount * 365)

    # Try ISO 8601 format
    try:
        dt = datetime.fromisoformat(date_str)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        pass

    raise ValueError(
        f"Invalid date format: '{date_str}'. "
        f"Use ISO 8601 (2026-01-01) or relative (7d, 2w, 3m, 1y)"
    )


async def remove_thread(
    thread_id: str,
    db_path: str | None = None,
    dry_run: bool = False
) -> dict:
    """Remove a thread from the database.

    DB resolution order (when ``db_path`` is not given) follows the same
    rule as :func:`get_thread_history`:

    1. ``sessions_<agent>.db`` where agent is derived from ``thread_id``
       (``thread_id.split(':')[0]``), if it exists.
    2. Legacy global ``sessions.db`` (pre-migration data).

    Args:
        thread_id: Thread ID to remove
        db_path: Optional explicit database path (bypasses auto-resolution)
        dry_run: If True, only show what would be deleted without deleting

    Returns:
        Dictionary with operation results:
        - success: bool - Whether operation succeeded
        - thread_id: str - Thread ID that was processed
        - dry_run: bool - Whether this was a dry run
        - checkpoints_deleted/checkpoints_found: int - Count of checkpoints
        - writes_deleted/writes_found: int - Count of writes (if any)
        - error: str - Error message (if success=False)
    """
    import aiosqlite

    if db_path is None:
        resolved = _resolve_db_path_for_thread(thread_id)
        if resolved is None:
            return {
                "success": False,
                "thread_id": thread_id,
                "error": f"Thread '{thread_id}' not found in database",
            }
        db_path = str(resolved)

    try:
        async with aiosqlite.connect(db_path) as conn:
            if not await _table_exists(conn, "checkpoints"):
                return {
                    "success": False,
                    "thread_id": thread_id,
                    "error": f"Thread '{thread_id}' not found in database",
                }
            # Check if thread exists
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                checkpoint_count = row[0] if row else 0

            if checkpoint_count == 0:
                return {
                    "success": False,
                    "thread_id": thread_id,
                    "error": f"Thread '{thread_id}' not found in database"
                }

            # Check for writes
            writes_count = 0
            try:
                async with conn.execute(
                    "SELECT COUNT(*) FROM checkpoint_writes WHERE thread_id = ?",
                    (thread_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    writes_count = row[0] if row else 0
            except aiosqlite.OperationalError:
                # checkpoint_writes table doesn't exist
                pass

            if dry_run:
                return {
                    "success": True,
                    "thread_id": thread_id,
                    "dry_run": True,
                    "checkpoints_found": checkpoint_count,
                    "writes_found": writes_count
                }

            # Delete checkpoints
            await conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )

            # Delete writes if they exist
            if writes_count > 0:
                await conn.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = ?",
                    (thread_id,)
                )

            await conn.commit()

            return {
                "success": True,
                "thread_id": thread_id,
                "dry_run": False,
                "checkpoints_deleted": checkpoint_count,
                "writes_deleted": writes_count
            }

    except Exception as e:
        return {
            "success": False,
            "thread_id": thread_id,
            "error": f"Error removing thread: {str(e)}"
        }


async def get_thread_history(
    thread_id: str,
    db_path: str | None = None,
    format: str = "text",
    agent_name: str | None = None,
) -> dict:
    """Get message history for a thread.

    DB resolution order (when ``db_path`` is not given):

    1. ``agent_name`` → ``sessions_<agent_name>.db`` (if exists).
    2. Else derive ``agent`` from ``thread_id`` (``thread_id.split(':')[0]``)
       and use ``sessions_<agent>.db`` if it exists.
    3. Else fall back to the legacy global ``sessions.db`` if it exists.
    4. Else report "not found".

    This matches where checkpoints actually live after the per-agent DB
    migration – fixes ``deep-worker --history <agent>`` failing with
    "no such table: checkpoints" because the legacy global file is empty.

    Args:
        thread_id: Thread ID to retrieve history for.
        db_path: Optional explicit database path (bypasses auto-resolution).
        format: Output format - "text" (default) or "json".
        agent_name: Optional explicit agent name override.  Normally
            derived from ``thread_id``.

    Returns:
        Dictionary with operation results:
        - success: bool - Whether operation succeeded
        - thread_id: str - Thread ID that was queried
        - checkpoints: list - List of checkpoint data
        - messages: list - Extracted messages from checkpoints
        - json_output: str - JSON formatted output (if format="json")
        - error: str - Error message (if success=False)
    """
    import aiosqlite
    import json

    if db_path is None:
        # Priority 1: explicit agent_name.
        resolved: Path | None = None
        if agent_name is not None:
            candidate = get_db_path(agent_name)
            if candidate.exists():
                resolved = candidate
        # Priority 2: derive from thread_id.
        if resolved is None:
            derived_agent = _agent_name_from_thread_id(thread_id)
            candidate = get_db_path(derived_agent)
            if candidate.exists():
                resolved = candidate
        # Priority 3: legacy global.
        if resolved is None:
            legacy = get_db_path(None)
            if legacy.exists():
                resolved = legacy

        if resolved is None:
            return {
                "success": False,
                "thread_id": thread_id,
                "error": (
                    f"Thread '{thread_id}' not found: no session database "
                    f"exists (looked for sessions_<agent>.db and "
                    f"legacy sessions.db under the agent data directory)."
                ),
            }
        db_path = str(resolved)

    try:
        # Use DeltaSqliteCheckpointer for delta-aware state reconstruction.
        # This handles both v1 (full snapshot) and v2 (delta) checkpoints.
        from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer

        async with DeltaSqliteCheckpointer.from_conn_string(db_path) as checkpointer:
            # Get latest checkpoint with full state reconstruction
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp_tuple = await checkpointer.aget_tuple(config)

            if not cp_tuple:
                return {
                    "success": False,
                    "thread_id": thread_id,
                    "error": f"Thread '{thread_id}' not found in database",
                }

            # Extract messages from reconstructed state
            channel_values = cp_tuple.checkpoint.get("channel_values", {})
            all_messages = channel_values.get("messages", [])

            # Add timestamps from checkpoint
            checkpoint_ts = cp_tuple.checkpoint.get("ts")
            for msg in all_messages:
                if isinstance(msg, dict):
                    msg["timestamp"] = checkpoint_ts
                elif hasattr(msg, "__dict__"):
                    msg.timestamp = checkpoint_ts

            # Get checkpoint count
            async with checkpointer.conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
                cp_count = row[0] if row else 0

            result: dict[str, Any] = {
                "success": True,
                "thread_id": thread_id,
                "checkpoints": [{"checkpoint_id": cp_tuple.config["configurable"]["checkpoint_id"]}],
                "messages": all_messages,
            }

            if format == "json":
                result["json_output"] = json.dumps(result, indent=2, default=str)

            return result

    except Exception as e:
        return {
            "success": False,
            "thread_id": thread_id,
            "error": f"Error retrieving thread history: {str(e)}",
        }


async def remove_threads_before(
    date_str: str,
    db_path: str | None = None,
    dry_run: bool = False
) -> dict:
    """Remove all threads older than the specified date.

    When ``db_path`` is not provided, the operation is applied to every
    per-agent ``sessions_<name>.db`` file in the agent data directory **and** the
    legacy global ``sessions.db``.  Counts are summed across all files.

    Args:
        date_str: Date string (ISO 8601 or relative like "30d", "1y")
        db_path: Optional explicit DB path (single file, bypasses
            per-agent iteration; useful for tests and targeted cleanup).
        dry_run: If True, only show what would be deleted without deleting

    Returns:
        Dictionary with operation results:
        - success: bool - Whether operation succeeded
        - dry_run: bool - Whether this was a dry run
        - cutoff_date: str - ISO 8601 cutoff date used
        - threads_deleted/threads_found: int - Count of threads
        - checkpoints_deleted/checkpoints_found: int - Count of checkpoints
        - writes_deleted/writes_found: int - Count of writes (if any)
        - error: str - Error message (if success=False)
    """
    import aiosqlite

    try:
        cutoff_date = parse_date(date_str)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date format: {str(e)}",
        }
    cutoff_iso = cutoff_date.isoformat()

    # Decide which DB files to process.
    if db_path is not None:
        db_paths: list[str] = [db_path]
    else:
        db_paths = [str(p) for p in iter_all_db_paths()]

    threads_total = 0
    checkpoints_total = 0
    writes_total = 0
    any_error: str | None = None

    for current_db in db_paths:
        try:
            async with aiosqlite.connect(current_db) as conn:
                if not await _table_exists(conn, "checkpoints"):
                    continue

                async with conn.execute(
                    """
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE json_extract(metadata, '$.updated_at') < ?
                    """,
                    (cutoff_iso,),
                ) as cursor:
                    thread_ids = [row[0] for row in await cursor.fetchall()]

                if not thread_ids:
                    continue

                placeholders = ",".join("?" * len(thread_ids))
                async with conn.execute(
                    f"SELECT COUNT(*) FROM checkpoints WHERE thread_id IN ({placeholders})",
                    thread_ids,
                ) as cursor:
                    row = await cursor.fetchone()
                    checkpoint_count = row[0] if row else 0

                writes_count = 0
                try:
                    async with conn.execute(
                        f"SELECT COUNT(*) FROM checkpoint_writes "
                        f"WHERE thread_id IN ({placeholders})",
                        thread_ids,
                    ) as cursor:
                        row = await cursor.fetchone()
                        writes_count = row[0] if row else 0
                except aiosqlite.OperationalError:
                    pass

                threads_total += len(thread_ids)
                checkpoints_total += checkpoint_count
                writes_total += writes_count

                if dry_run:
                    continue

                await conn.execute(
                    f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})",
                    thread_ids,
                )
                if writes_count > 0:
                    try:
                        await conn.execute(
                            f"DELETE FROM checkpoint_writes "
                            f"WHERE thread_id IN ({placeholders})",
                            thread_ids,
                        )
                    except aiosqlite.OperationalError:
                        pass

                await conn.commit()
        except Exception as e:  # noqa: BLE001
            any_error = f"Error removing threads from '{current_db}': {str(e)}"
            continue

    if any_error is not None and threads_total == 0:
        return {
            "success": False,
            "error": any_error,
        }

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "cutoff_date": cutoff_iso,
            "threads_found": threads_total,
            "checkpoints_found": checkpoints_total,
            "writes_found": writes_total,
        }

    return {
        "success": True,
        "dry_run": False,
        "cutoff_date": cutoff_iso,
        "threads_deleted": threads_total,
        "checkpoints_deleted": checkpoints_total,
        "writes_deleted": writes_total,
    }


async def import_history(
    agent_name: str,
    messages: list,
    *,
    replace: bool = False,
) -> int:
    """Write an imported message list into the agent's per-agent SQLite DB.

    This is the injection point for ``--import-claude``
    (DESIGN_IMPORT_CLAUDE_SESSION.md §4): the parsed Claude transcript becomes
    the agent's conversation history (its *memory*). The messages land in
    the per-agent session database (``sessions_<agent>.db``) so
    ``deep-worker --history <agent>``
    shows them and the agent sees them on its first turn — independent of which
    provider the agent later runs under.

    We write via a MINIMAL LangGraph graph (just the ``messages`` channel with
    the standard ``add_messages`` reducer) compiled with the per-agent
    checkpointer, then ``aupdate_state``. This produces a checkpoint LangGraph
    reads back correctly (verified against the existing offline read path
    ``get_thread_history``) WITHOUT constructing any provider model — no API
    keys, no network, no inference. The real agent reads only the ``messages``
    channel from this checkpoint; its other state channels default to empty on
    first run.

    Args:
        agent_name: target agent (its thread_id is the agent name).
        messages: the parsed ``list[BaseMessage]`` to import.
        replace: if the agent already has history, ``False`` refuses (so two
            conversations are never silently spliced); ``True`` wipes the
            existing per-agent DB first.

    Returns:
        The number of messages written.

    Raises:
        ValueError: ``messages`` is empty, or history already exists and
            ``replace`` is ``False``.
    """
    from typing import Annotated, TypedDict  # noqa: PLC0415

    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    if not messages:
        raise ValueError("Nothing to import: the message list is empty.")

    # Guard against clobbering an existing conversation.
    existing = await get_thread_history(agent_name, agent_name=agent_name)
    has_history = bool(existing.get("success") and existing.get("messages"))
    if has_history:
        if not replace:
            raise ValueError(
                f"Agent '{agent_name}' already has conversation history. "
                f"Pass --clean to replace it (import refuses to splice two "
                f"conversations together)."
            )
        # replace=True: wipe the per-agent DB first.
        await clean_agent_history_db_only(agent_name)

    class _ImportState(TypedDict):
        messages: Annotated[list, add_messages]

    builder = StateGraph(_ImportState)
    builder.add_node("noop", lambda _state: {})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)

    async with get_checkpointer(agent_name=agent_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {
            "configurable": {"thread_id": agent_name, "checkpoint_ns": ""}
        }
        await graph.aupdate_state(config, {"messages": messages})

    return len(messages)
