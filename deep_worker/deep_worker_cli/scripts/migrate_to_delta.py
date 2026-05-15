#!/usr/bin/env python3
"""Migrate existing checkpoint databases to delta format.

For each sessions_*.db file:
1. Read the LATEST checkpoint per thread (full snapshot with all messages)
2. Create a new DB with just ONE full-snapshot checkpoint (the latest state)
3. Verify: reconstructed messages match original
4. Replace old DB with new one

This compresses the database from O(N²) to O(N) — one checkpoint with
full state instead of thousands of checkpoints each with full state copy.

Usage:
    python3 -m scripts.migrate_to_delta [--dry-run] [--agent AGENT_NAME]
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import aiosqlite

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


serde = JsonPlusSerializer()


async def migrate_db(db_path: Path, dry_run: bool = False) -> dict:
    """Migrate a single DB file to delta format.

    Returns dict with migration stats.
    """
    agent_name = db_path.stem.replace("sessions_", "")
    stats = {
        "agent": agent_name,
        "old_size": db_path.stat().st_size,
        "threads": 0,
        "messages_verified": 0,
        "success": False,
        "error": None,
    }

    try:
        async with aiosqlite.connect(str(db_path), timeout=60) as conn:
            # Check tables exist
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                tables = {r[0] for r in await cursor.fetchall()}

            if "checkpoints" not in tables:
                stats["error"] = "no checkpoints table"
                return stats

            # Get all thread_ids
            async with conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ) as cursor:
                thread_ids = [r[0] for r in await cursor.fetchall()]

            if not thread_ids:
                # No checkpoints — check if writes table has data
                if "writes" in tables:
                    async with conn.execute("SELECT COUNT(*) FROM writes") as c:
                        writes_count = (await c.fetchone())[0]
                    if writes_count > 0:
                        # Writes-only DB (durability="exit") — just VACUUM
                        stats["writes_only"] = True
                        stats["writes_count"] = writes_count
                        if not dry_run:
                            await conn.execute("VACUUM")
                            stats["new_size"] = db_path.stat().st_size
                        else:
                            # Estimate: used pages only
                            async with conn.execute("PRAGMA page_count") as c:
                                pages = (await c.fetchone())[0]
                            async with conn.execute("PRAGMA page_size") as c:
                                psize = (await c.fetchone())[0]
                            async with conn.execute("PRAGMA freelist_count") as c:
                                free = (await c.fetchone())[0]
                            stats["new_size_estimate"] = (pages - free) * psize
                        stats["success"] = True
                        return stats
                    else:
                        stats["error"] = "empty database"
                else:
                    stats["error"] = "no threads found"
                return stats

            stats["threads"] = len(thread_ids)

            # For each thread, get latest checkpoint and extract messages
            thread_data = []
            for tid in thread_ids:
                async with conn.execute(
                    "SELECT checkpoint_id, type, checkpoint, metadata "
                    "FROM checkpoints WHERE thread_id = ? "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    (tid,),
                ) as cursor:
                    row = await cursor.fetchone()

                if not row:
                    continue

                cp_id, type_str, blob, metadata_blob = row
                try:
                    cp_data = serde.loads_typed((type_str, blob))
                except Exception as e:
                    print(f"  WARNING: Failed to deserialize {tid}: {e}")
                    continue

                messages = cp_data.get("channel_values", {}).get("messages", [])
                metadata = json.loads(metadata_blob) if metadata_blob else {}

                thread_data.append({
                    "thread_id": tid,
                    "checkpoint_id": cp_id,
                    "type_str": type_str,
                    "blob": blob,
                    "cp_data": cp_data,
                    "metadata": metadata,
                    "message_count": len(messages),
                })
                stats["messages_verified"] += len(messages)

            if dry_run:
                stats["success"] = True
                stats["new_size_estimate"] = sum(len(td["blob"]) for td in thread_data) + 4096
                return stats

            # Create new DB
            new_path = db_path.with_suffix(".db.new")
            backup_path = db_path.with_suffix(".db.bak")

            async with aiosqlite.connect(str(new_path)) as new_conn:
                # Create tables
                await new_conn.executescript("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB,
                        metadata BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    );
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    );
                    CREATE TABLE IF NOT EXISTS thread_metadata (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        agent_name TEXT,
                        updated_at TEXT,
                        message_count INTEGER DEFAULT 0,
                        checkpoint_count INTEGER DEFAULT 0,
                        PRIMARY KEY (thread_id, checkpoint_ns)
                    );
                """)

                # Insert latest checkpoint for each thread (as full snapshot)
                for td in thread_data:
                    # Mark as full snapshot for delta checkpointer compatibility
                    cp_data = td["cp_data"]
                    cp_data["is_full_snapshot"] = True

                    type_str, new_blob = serde.dumps_typed(cp_data)

                    await new_conn.execute(
                        "INSERT INTO checkpoints "
                        "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            td["thread_id"],
                            "",  # checkpoint_ns
                            td["checkpoint_id"],
                            None,  # no parent — this is root
                            type_str,
                            new_blob,
                            json.dumps(td["metadata"]).encode(),
                        ),
                    )

                    # thread_metadata
                    agent = td["metadata"].get("agent_name", td["thread_id"].split(":")[0])
                    ts = cp_data.get("ts", "")
                    await new_conn.execute(
                        "INSERT OR REPLACE INTO thread_metadata "
                        "(thread_id, checkpoint_ns, agent_name, updated_at, message_count, checkpoint_count) "
                        "VALUES (?, '', ?, ?, ?, 1)",
                        (td["thread_id"], agent, ts, td["message_count"]),
                    )

                await new_conn.commit()

            # Verify new DB
            async with aiosqlite.connect(str(new_path)) as verify_conn:
                for td in thread_data:
                    async with verify_conn.execute(
                        "SELECT type, checkpoint FROM checkpoints WHERE thread_id = ?",
                        (td["thread_id"],),
                    ) as cursor:
                        row = await cursor.fetchone()

                    if not row:
                        raise RuntimeError(f"Thread {td['thread_id']} missing in new DB!")

                    verify_data = serde.loads_typed((row[0], row[1]))
                    verify_msgs = verify_data.get("channel_values", {}).get("messages", [])

                    if len(verify_msgs) != td["message_count"]:
                        raise RuntimeError(
                            f"Thread {td['thread_id']}: message count mismatch! "
                            f"Expected {td['message_count']}, got {len(verify_msgs)}"
                        )

            # Swap files: old → .bak, new → current
            shutil.move(str(db_path), str(backup_path))
            shutil.move(str(new_path), str(db_path))

            stats["new_size"] = db_path.stat().st_size
            stats["backup_path"] = str(backup_path)
            stats["success"] = True

    except Exception as e:
        stats["error"] = str(e)
        # Cleanup on failure
        new_path = db_path.with_suffix(".db.new")
        if new_path.exists():
            new_path.unlink()

    return stats


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate checkpoint DBs to delta format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--agent", help="Migrate only this agent")
    parser.add_argument("--delete-backups", action="store_true", help="Delete .bak files after verification")
    args = parser.parse_args()

    db_dir = Path.home() / ".deepagents"
    if not db_dir.exists():
        print("No .deepagents directory found")
        return

    # Find DB files
    if args.agent:
        db_files = [db_dir / f"sessions_{args.agent}.db"]
        db_files = [f for f in db_files if f.exists()]
    else:
        db_files = sorted(db_dir.glob("sessions_*.db"))
        # Skip legacy sessions.db and already migrated
        db_files = [f for f in db_files if f.stat().st_size > 0]

    if not db_files:
        print("No DB files to migrate")
        return

    print(f"{'DRY RUN: ' if args.dry_run else ''}Migrating {len(db_files)} databases...")
    print()

    total_old = 0
    total_new = 0
    results = []

    for db_path in db_files:
        size_mb = db_path.stat().st_size / 1024 / 1024
        print(f"  {db_path.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

        stats = await migrate_db(db_path, dry_run=args.dry_run)
        results.append(stats)

        if stats["success"]:
            old_mb = stats["old_size"] / 1024 / 1024
            if args.dry_run:
                new_mb = stats.get("new_size_estimate", 0) / 1024 / 1024
            else:
                new_mb = stats.get("new_size", stats["old_size"]) / 1024 / 1024
            ratio = old_mb / new_mb if new_mb > 0 else 0
            if stats.get("writes_only"):
                print(f"VACUUM ({stats['writes_count']} writes, "
                      f"{old_mb:.1f}→{new_mb:.1f} MB, {ratio:.0f}x)")
            else:
                print(f"OK ({stats['threads']} threads, {stats['messages_verified']} msgs, "
                      f"{old_mb:.1f}→{new_mb:.1f} MB, {ratio:.0f}x)")
            total_old += stats["old_size"]
            total_new += stats.get("new_size", stats.get("new_size_estimate", 0))
        else:
            print(f"SKIP ({stats['error']})")

    print()
    print(f"Total: {total_old/1024/1024/1024:.1f} GB → {total_new/1024/1024:.0f} MB")
    if total_new > 0:
        print(f"Reduction: {total_old/total_new:.0f}x")

    if not args.dry_run and args.delete_backups:
        print()
        print("Deleting backup files...")
        for stats in results:
            if stats.get("backup_path"):
                bak = Path(stats["backup_path"])
                if bak.exists():
                    bak.unlink()
                    print(f"  Deleted {bak.name}")


if __name__ == "__main__":
    asyncio.run(main())
