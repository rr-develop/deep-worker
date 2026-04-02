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
    # "attr-defined" error that would otherwise be raised.
    aiosqlite.Connection.is_alive = _is_alive  # type: ignore[attr-defined]


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


def get_db_path() -> Path:
    """Get path to global database.

    Returns:
        Path to the SQLite database file.
    """
    db_dir = Path.home() / ".deepagents"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


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

    Args:
        agent_name: Optional filter by agent name.
        limit: Maximum number of threads to return.
        include_message_count: Whether to include message counts.

    Returns:
        List of thread dicts with `thread_id`, `agent_name`, `updated_at`,
            and optionally `message_count`.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        # Return empty if table doesn't exist yet (fresh install)
        if not await _table_exists(conn, "checkpoints"):
            return []

        if agent_name:
            query = """
                SELECT thread_id,
                       json_extract(metadata, '$.agent_name') as agent_name,
                       MAX(json_extract(metadata, '$.updated_at')) as updated_at
                FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                GROUP BY thread_id
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params: tuple = (agent_name, limit)
        else:
            query = """
                SELECT thread_id,
                       json_extract(metadata, '$.agent_name') as agent_name,
                       MAX(json_extract(metadata, '$.updated_at')) as updated_at
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (limit,)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            threads = [
                {"thread_id": r[0], "agent_name": r[1], "updated_at": r[2]}
                for r in rows
            ]

        # Fetch message counts if requested
        if include_message_count and threads:
            serde = JsonPlusSerializer()
            for thread in threads:
                thread["message_count"] = await _count_messages_from_checkpoint(
                    conn, thread["thread_id"], serde
                )

        return threads


async def list_all_threads(
    agent_name: str | None = None,
    sort_by: str = "thread_id",
    reverse: bool = False,
    include_message_count: bool = False,
) -> list[dict]:
    """List ALL threads without pagination.

    Args:
        agent_name: Optional filter by agent name.
        sort_by: Sort field: 'thread_id' (default), 'agent', 'updated'.
        reverse: Reverse sort order (default: False).
        include_message_count: Include message counts (slow operation, default: False).

    Returns:
        List of thread dicts with:
        - thread_id: str
        - agent_name: str | None
        - updated_at: str (ISO timestamp)
        - message_count: int (only if include_message_count=True)
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        # Return empty if table doesn't exist yet (fresh install)
        if not await _table_exists(conn, "checkpoints"):
            return []

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
            threads = [
                {"thread_id": r[0], "agent_name": r[1], "updated_at": r[2]}
                for r in rows
            ]

        # Fetch message counts if requested
        if include_message_count and threads:
            serde = JsonPlusSerializer()
            for thread in threads:
                thread["message_count"] = await _count_messages_from_checkpoint(
                    conn, thread["thread_id"], serde
                )

    # Sort threads in Python (allows complex sort logic)
    if sort_by == "thread_id":
        threads.sort(key=lambda t: t["thread_id"].lower(), reverse=reverse)
    elif sort_by == "agent":
        # NULL agent_names first, then sorted alphabetically with secondary sort by thread_id
        threads.sort(
            key=lambda t: ((t["agent_name"] or "").lower(), t["thread_id"].lower()),
            reverse=reverse
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

    With durability="exit", messages are stored in the checkpoint blob,
    not in the writes table. This function deserializes the checkpoint
    and counts the messages in channel_values.

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


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Get AsyncSqliteSaver for the global database.

    Yields:
        AsyncSqliteSaver instance for checkpoint persistence.
    """
    async with AsyncSqliteSaver.from_conn_string(str(get_db_path())) as checkpointer:
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


async def clean_agent_history(
    agent_name: str,
    *,
    _home_dir: Path | None = None,  # For testing only
) -> None:
    """Clean all history and state for an agent.

    This removes:
    - All checkpoints/threads from sessions.db for this agent
    - All writes from sessions.db for this agent's threads
    - Agent data directory (~/.deep_worker/{agent_name}/)
    - Agent log files (~/.deep_worker/logs/{agent_name}.*)

    This does NOT remove:
    - User configuration (~/.deepagents/{agent_name}/AGENTS.md)
    - User skills (~/.deepagents/{agent_name}/skills/)
    - User subagents (~/.deepagents/{agent_name}/agents/)

    Args:
        agent_name: Name of the agent to clean
        _home_dir: Home directory override (for testing only, not public API)

    Note:
        This function is idempotent and handles non-existent data gracefully.
    """
    import shutil

    # Root the runtime paths under ~/.deep_worker/ (project convention);
    # `_home_dir` is a test hook that redirects that root inside a tmpdir.
    _RUNTIME_SUBDIR = Path("~/.deep_worker").expanduser().name
    if _home_dir is None:
        runtime_root = Path("~/.deep_worker").expanduser()
    else:
        runtime_root = _home_dir / _RUNTIME_SUBDIR

    # 1. Get all thread_ids for this agent before deleting
    # We need this to clean up writes table
    thread_ids: list[str] = []
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if await _table_exists(conn, "checkpoints"):
            query = """
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
            """
            async with conn.execute(query, (agent_name,)) as cursor:
                rows = await cursor.fetchall()
                thread_ids = [row[0] for row in rows]

    # 2. Delete all checkpoints and writes from database
    checkpoints_deleted = 0
    writes_deleted = 0

    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        # Delete checkpoints for this agent
        if await _table_exists(conn, "checkpoints"):
            cursor = await conn.execute(
                "DELETE FROM checkpoints WHERE json_extract(metadata, '$.agent_name') = ?",
                (agent_name,),
            )
            checkpoints_deleted = cursor.rowcount

        # Delete writes for all thread_ids of this agent
        if await _table_exists(conn, "writes") and thread_ids:
            placeholders = ",".join("?" * len(thread_ids))
            query = f"DELETE FROM writes WHERE thread_id IN ({placeholders})"
            cursor = await conn.execute(query, thread_ids)
            writes_deleted = cursor.rowcount

        await conn.commit()

    # 3. Remove agent data directory
    data_dir = runtime_root / agent_name
    data_dir_removed = False
    if data_dir.exists():
        shutil.rmtree(data_dir)
        data_dir_removed = True

    # 4. Remove log files
    log_dir = runtime_root / "logs"
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
        f"  [dim]- Preserved user config: ~/.deepagents/{agent_name}/[/dim]"
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

    Args:
        thread_id: Thread ID to remove
        db_path: Optional custom database path (defaults to global DB)
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
        db_path = str(get_db_path())

    try:
        async with aiosqlite.connect(db_path) as conn:
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
    format: str = "text"
) -> dict:
    """Get message history for a thread.

    Args:
        thread_id: Thread ID to retrieve history for
        db_path: Optional custom database path (defaults to global DB)
        format: Output format - "text" (default) or "json"

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
        db_path = str(get_db_path())

    try:
        async with aiosqlite.connect(db_path) as conn:
            # Check if thread exists
            async with conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0

            if count == 0:
                return {
                    "success": False,
                    "thread_id": thread_id,
                    "error": f"Thread '{thread_id}' not found in database"
                }

            # Retrieve all checkpoints for this thread
            async with conn.execute("""
                SELECT checkpoint_id, type, checkpoint, metadata, parent_checkpoint_id
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY checkpoint_id
            """, (thread_id,)) as cursor:
                rows = await cursor.fetchall()

            # Initialize serializer for decoding checkpoint blobs
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            serde = JsonPlusSerializer()

            checkpoints = []
            all_messages = []
            seen_message_count = 0  # Track number of messages we've already seen

            for row in rows:
                checkpoint_id, type_str, checkpoint_blob, metadata_str, parent_id = row

                checkpoint_data = {}
                if checkpoint_blob and type_str:
                    try:
                        # Use LangGraph serializer to decode checkpoint
                        checkpoint_data = serde.loads_typed((type_str, checkpoint_blob))
                    except (ValueError, TypeError) as e:
                        checkpoint_data = {"error": f"Failed to deserialize: {str(e)}"}
                elif checkpoint_blob:
                    # Fallback: try JSON if no type
                    try:
                        checkpoint_data = json.loads(checkpoint_blob)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        checkpoint_data = {"raw": "binary data"}

                metadata = {}
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        pass

                checkpoint_info = {
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": parent_id,
                    "metadata": metadata,
                    "data": checkpoint_data
                }

                checkpoints.append(checkpoint_info)

                # Extract NEW messages only (not duplicates from accumulation)
                if isinstance(checkpoint_data, dict) and "channel_values" in checkpoint_data:
                    channel_values = checkpoint_data["channel_values"]
                    if "messages" in channel_values:
                        messages = channel_values["messages"]
                        if isinstance(messages, list):
                            # Get checkpoint timestamp
                            checkpoint_ts = checkpoint_data.get("ts")

                            # Only add messages we haven't seen yet
                            new_messages = messages[seen_message_count:]
                            for msg in new_messages:
                                # Add timestamp to message
                                if isinstance(msg, dict):
                                    msg["timestamp"] = checkpoint_ts
                                elif hasattr(msg, "__dict__"):
                                    # For LangChain Message objects, add timestamp attribute
                                    msg.timestamp = checkpoint_ts
                                all_messages.append(msg)

                            # Update seen count
                            seen_message_count = len(messages)

            result = {
                "success": True,
                "thread_id": thread_id,
                "checkpoints": checkpoints,
                "messages": all_messages
            }

            if format == "json":
                result["json_output"] = json.dumps(result, indent=2, default=str)

            return result

    except Exception as e:
        return {
            "success": False,
            "thread_id": thread_id,
            "error": f"Error retrieving thread history: {str(e)}"
        }


async def remove_threads_before(
    date_str: str,
    db_path: str | None = None,
    dry_run: bool = False
) -> dict:
    """Remove all threads older than the specified date.

    Args:
        date_str: Date string (ISO 8601 or relative like "30d", "1y")
        db_path: Optional custom database path (defaults to global DB)
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
    import json

    if db_path is None:
        db_path = str(get_db_path())

    try:
        # Parse the date string
        cutoff_date = parse_date(date_str)
        cutoff_iso = cutoff_date.isoformat()

        async with aiosqlite.connect(db_path) as conn:
            # Find all threads with updated_at before cutoff date
            # Group by thread_id to get unique threads
            async with conn.execute("""
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE json_extract(metadata, '$.updated_at') < ?
            """, (cutoff_iso,)) as cursor:
                thread_ids = [row[0] for row in await cursor.fetchall()]

            if not thread_ids:
                return {
                    "success": True,
                    "dry_run": dry_run,
                    "cutoff_date": cutoff_iso,
                    "threads_deleted": 0,
                    "checkpoints_deleted": 0,
                    "writes_deleted": 0
                }

            # Count checkpoints for these threads
            placeholders = ','.join('?' * len(thread_ids))
            async with conn.execute(
                f"SELECT COUNT(*) FROM checkpoints WHERE thread_id IN ({placeholders})",
                thread_ids
            ) as cursor:
                row = await cursor.fetchone()
                checkpoint_count = row[0] if row else 0

            # Count writes if table exists
            writes_count = 0
            try:
                async with conn.execute(
                    f"SELECT COUNT(*) FROM checkpoint_writes WHERE thread_id IN ({placeholders})",
                    thread_ids
                ) as cursor:
                    row = await cursor.fetchone()
                    writes_count = row[0] if row else 0
            except aiosqlite.OperationalError:
                # checkpoint_writes table doesn't exist
                pass

            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "cutoff_date": cutoff_iso,
                    "threads_found": len(thread_ids),
                    "checkpoints_found": checkpoint_count,
                    "writes_found": writes_count
                }

            # Delete checkpoints for these threads
            await conn.execute(
                f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})",
                thread_ids
            )

            # Delete writes if they exist
            if writes_count > 0:
                await conn.execute(
                    f"DELETE FROM checkpoint_writes WHERE thread_id IN ({placeholders})",
                    thread_ids
                )

            await conn.commit()

            return {
                "success": True,
                "dry_run": False,
                "cutoff_date": cutoff_iso,
                "threads_deleted": len(thread_ids),
                "checkpoints_deleted": checkpoint_count,
                "writes_deleted": writes_count
            }

    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date format: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error removing threads: {str(e)}"
        }
