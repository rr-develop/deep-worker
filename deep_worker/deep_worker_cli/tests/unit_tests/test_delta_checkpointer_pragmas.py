"""Tests for SQLite PRAGMA settings in DeltaSqliteCheckpointer.setup().

TDD: Verifies that setup() applies journal_mode=WAL and
synchronous=NORMAL for corruption prevention.
"""

import uuid
from pathlib import Path

import pytest

from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    """Return a temp file path for SQLite (WAL mode requires a real file)."""
    return str(tmp_path / f"test_{uuid.uuid4().hex}.db")


class TestCheckpointerPragmaSettings:
    """Verify that PRAGMA settings are applied during setup()."""

    @pytest.mark.asyncio
    async def test_journal_mode_is_wal_after_setup(self, temp_db: str):
        """After setup() on a file DB, journal_mode should be 'wal'."""

        async with DeltaSqliteCheckpointer.from_conn_string(temp_db) as cp:
            await cp.setup()
            async with cp.conn.execute("PRAGMA journal_mode") as cursor:
                row = await cursor.fetchone()
            assert row is not None
            # journal_mode returns 'wal' (lowercase) when set
            assert row[0] == "wal"

    @pytest.mark.asyncio
    async def test_journal_mode_is_wal_persists_across_connections(self, temp_db: str):
        """WAL mode should persist when a new connection opens the same DB."""

        # First connection: setup + PRAGMA
        async with DeltaSqliteCheckpointer.from_conn_string(temp_db) as cp:
            await cp.setup()

        # Second connection: verify WAL is still active
        import aiosqlite
        async with aiosqlite.connect(temp_db) as conn:
            async with conn.execute("PRAGMA journal_mode") as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "wal"

    @pytest.mark.asyncio
    async def test_synchronous_is_normal_after_setup(self, temp_db: str):
        """After setup(), synchronous should be NORMAL (1)."""
        from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer

        async with DeltaSqliteCheckpointer.from_conn_string(temp_db) as cp:
            await cp.setup()
            async with cp.conn.execute("PRAGMA synchronous") as cursor:
                row = await cursor.fetchone()
            assert row is not None
            # synchronous returns integer: 0=OFF, 1=NORMAL, 2=FULL
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_setup_idempotent_pragma_not_overwritten(self, temp_db: str):
        """Calling setup() twice should not change PRAGMA settings."""
        from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer

        async with DeltaSqliteCheckpointer.from_conn_string(temp_db) as cp:
            await cp.setup()
            async with cp.conn.execute("PRAGMA journal_mode") as cursor:
                row1 = await cursor.fetchone()
            await cp.setup()  # second call — no-op
            async with cp.conn.execute("PRAGMA journal_mode") as cursor:
                row2 = await cursor.fetchone()
            assert row1 == row2

    @pytest.mark.asyncio
    async def test_checkpointer_works_normally_with_pragmas(self, temp_db: str):
        """With PRAGMA settings applied, normal checkpoint ops still work."""
        from langchain_core.messages import HumanMessage
        from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer

        async with DeltaSqliteCheckpointer.from_conn_string(temp_db) as cp:
            await cp.setup()

            # Write a checkpoint
            msgs = [HumanMessage(content="hello")]
            cp_data = {
                "v": 1,
                "id": "test123",
                "ts": "2026-05-31T12:00:00",
                "channel_values": {"messages": msgs},
                "channel_versions": {"messages": "00000000000000000000000000000001.0"},
                "versions_seen": {},
                "pending_sends": [],
            }
            config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
            result_config = await cp.aput(config, cp_data, {}, {})

            # Read it back
            loaded = await cp.aget_tuple(result_config)
            assert loaded is not None
            messages = loaded.checkpoint["channel_values"]["messages"]
            assert len(messages) == 1
            assert messages[0].content == "hello"
