"""Delta Checkpointer — stores only changes between checkpoints.

Standard LangGraph AsyncSqliteSaver stores the FULL state (including ALL messages)
in every checkpoint. For a 12-checkpoint exchange with 800+ messages, this means
12 × 800KB = ~10MB per exchange, growing quadratically.

DeltaSqliteCheckpointer stores only changed channel values (delta). Messages use
append-delta (only new messages). Other channels use replace-delta. Unchanged
channels are not stored at all.

Full state is reconstructed by walking the parent chain and accumulating deltas.

Backward compatible: reads v1 (full snapshot) checkpoints from existing databases.
New checkpoints are written as v2 (delta).

Full snapshots are created only for:
  1. Root checkpoint (first in chain — no parent)
  2. Summarization (messages restructured, not just appended)
  3. Optional: --snapshot-interval N (disabled by default)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

# Delta format version
_DELTA_VERSION = 2

# SQL for table creation
_CREATE_TABLES_SQL = """
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
"""


class DeltaSqliteCheckpointer(BaseCheckpointSaver):
    """SQLite checkpointer that stores only deltas between checkpoints.

    Drop-in replacement for AsyncSqliteSaver. Same table schema,
    backward compatible with existing v1 (full snapshot) data.
    """

    serde = JsonPlusSerializer()

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        snapshot_interval: int = 0,
    ) -> None:
        super().__init__(serde=self.serde)
        self.conn = conn
        self._snapshot_interval = snapshot_interval
        self._is_setup = False
        # Cache: (thread_id, checkpoint_ns) → reconstructed channel_values
        self._state_cache: dict[tuple[str, str], dict[str, Any]] = {}

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str, *, snapshot_interval: int = 0
    ) -> AsyncIterator["DeltaSqliteCheckpointer"]:
        async with aiosqlite.connect(conn_string) as conn:
            yield cls(conn, snapshot_interval=snapshot_interval)

    async def setup(self) -> None:
        if self._is_setup:
            return
        async with self.conn.executescript(_CREATE_TABLES_SQL):
            pass
        await self.conn.commit()
        self._is_setup = True

    # ------------------------------------------------------------------
    # Core: aget_tuple — reconstruct full state from delta chain
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id:
            query = (
                "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata "
                "FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
            )
            params: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
        else:
            query = (
                "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata "
                "FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                "ORDER BY checkpoint_id DESC LIMIT 1"
            )
            params = (thread_id, checkpoint_ns)

        async with self.conn.execute(query, params) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        cp_id, parent_id, type_str, blob, metadata_blob = row

        # Reconstruct full state from delta chain
        checkpoint = await self._reconstruct_checkpoint(
            thread_id, checkpoint_ns, cp_id, type_str, blob
        )

        # Deserialize metadata
        metadata: dict[str, Any] = {}
        if metadata_blob:
            metadata = json.loads(metadata_blob)

        # Fetch pending writes
        writes_query = (
            "SELECT task_id, channel, type, value FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
            "ORDER BY task_id, idx"
        )
        pending_writes: list[tuple[str, str, Any]] = []
        async with self.conn.execute(
            writes_query, (thread_id, checkpoint_ns, cp_id)
        ) as cursor:
            async for w_row in cursor:
                task_id, channel, w_type, w_value = w_row
                value = self.serde.loads_typed((w_type, w_value)) if w_value else None
                pending_writes.append((task_id, channel, value))

        # Build configs
        cp_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
            }
        }
        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config=cp_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )

    async def _reconstruct_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        type_str: str,
        blob: bytes,
    ) -> Checkpoint:
        """Reconstruct full checkpoint from delta chain."""
        data = self.serde.loads_typed((type_str, blob))

        # v1 or explicit full snapshot — return as-is
        if data.get("v", 1) != _DELTA_VERSION or data.get("is_full_snapshot", False):
            return data

        # v2 delta — walk parent chain to reconstruct
        chain: list[dict[str, Any]] = [data]
        current_parent = None

        # Find parent_checkpoint_id for current checkpoint
        async with self.conn.execute(
            "SELECT parent_checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
            (thread_id, checkpoint_ns, checkpoint_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                current_parent = row[0]

        # Walk parent chain
        while current_parent:
            async with self.conn.execute(
                "SELECT parent_checkpoint_id, type, checkpoint FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_ns, current_parent),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                break

            parent_parent, p_type, p_blob = row
            parent_data = self.serde.loads_typed((p_type, p_blob))
            chain.append(parent_data)

            if parent_data.get("v", 1) == 1 or parent_data.get("is_full_snapshot", False):
                break  # Found full snapshot base

            current_parent = parent_parent

        # Apply deltas from oldest (base) to newest
        chain.reverse()
        base = chain[0]
        full_channel_values = dict(base.get("channel_values", {}))

        for delta_cp in chain[1:]:
            delta_values = delta_cp.get("delta_channel_values", {})
            delta_meta = delta_cp.get("delta_meta", {})

            for channel, value in delta_values.items():
                channel_delta_type = delta_meta.get(channel, "replace")
                if channel_delta_type == "append" and channel in full_channel_values:
                    # Append-only: extend existing list
                    if isinstance(full_channel_values[channel], list) and isinstance(value, list):
                        full_channel_values[channel] = full_channel_values[channel] + value
                    else:
                        full_channel_values[channel] = value
                else:
                    # Replace
                    full_channel_values[channel] = value

        # Build full checkpoint from newest delta + reconstructed channel_values
        newest = chain[-1] if len(chain) > 1 else base
        result: dict[str, Any] = {
            "v": newest.get("v", 1),
            "id": newest["id"],
            "ts": newest["ts"],
            "channel_values": full_channel_values,
            "channel_versions": newest.get("channel_versions", {}),
            "versions_seen": newest.get("versions_seen", {}),
            "pending_sends": newest.get("pending_sends", []),
        }
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core: aput — compute and store delta
    # ------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        cp_id = checkpoint["id"]

        # Determine if we need full snapshot
        is_root = parent_checkpoint_id is None
        is_summarization = False
        is_interval = False

        prev_state: dict[str, Any] | None = None

        if not is_root:
            # Fetch previous state for delta computation
            prev_state = await self._get_prev_channel_values(
                thread_id, checkpoint_ns, parent_checkpoint_id
            )
            if prev_state is not None:
                is_summarization = self._is_summarization(
                    checkpoint["channel_values"], prev_state
                )
            if self._snapshot_interval > 0:
                depth = await self._count_deltas_since_full(
                    thread_id, checkpoint_ns, parent_checkpoint_id
                )
                is_interval = depth >= self._snapshot_interval

        need_full = is_root or is_summarization or is_interval or prev_state is None

        if need_full:
            # Full snapshot
            to_store = dict(checkpoint)
            to_store["is_full_snapshot"] = True
            type_str, serialized = self.serde.dumps_typed(to_store)
        else:
            # Delta
            delta = self._compute_delta(checkpoint, prev_state)
            type_str, serialized = self.serde.dumps_typed(delta)

        # Serialize metadata
        from langgraph.checkpoint.base import get_checkpoint_metadata
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")

        # Insert checkpoint
        await self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                checkpoint_ns,
                cp_id,
                parent_checkpoint_id,
                type_str,
                serialized,
                serialized_metadata,
            ),
        )

        # Update thread_metadata
        msg_count = len(checkpoint.get("channel_values", {}).get("messages", []))
        agent_name = metadata.get("agent_name") or config["configurable"].get("thread_id")
        ts = checkpoint.get("ts", "")

        await self.conn.execute(
            "INSERT OR REPLACE INTO thread_metadata "
            "(thread_id, checkpoint_ns, agent_name, updated_at, message_count, checkpoint_count) "
            "VALUES (?, ?, ?, ?, ?, "
            "COALESCE((SELECT checkpoint_count FROM thread_metadata "
            "WHERE thread_id = ? AND checkpoint_ns = ?), 0) + 1)",
            (thread_id, checkpoint_ns, agent_name, ts, msg_count, thread_id, checkpoint_ns),
        )

        await self.conn.commit()

        # Update cache
        cache_key = (thread_id, checkpoint_ns)
        self._state_cache[cache_key] = dict(checkpoint.get("channel_values", {}))

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cp_id,
            }
        }

    def _compute_delta(
        self, checkpoint: Checkpoint, prev_channel_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute delta between current checkpoint and previous state."""
        new_cv = checkpoint.get("channel_values", {})
        updated = checkpoint.get("updated_channels") or []

        delta_values: dict[str, Any] = {}
        delta_meta: dict[str, str] = {}

        for channel in updated:
            if channel not in new_cv:
                continue
            new_val = new_cv[channel]
            old_val = prev_channel_values.get(channel)

            if channel == "messages" and isinstance(new_val, list) and isinstance(old_val, list):
                # Append delta for messages
                if len(new_val) >= len(old_val):
                    new_msgs = new_val[len(old_val):]
                    if new_msgs:
                        delta_values[channel] = new_msgs
                        delta_meta[channel] = "append"
                else:
                    # Messages shrunk — summarization (should be full snapshot, but safety)
                    delta_values[channel] = new_val
                    delta_meta[channel] = "replace"
            else:
                # Replace delta for everything else
                delta_values[channel] = new_val
                delta_meta[channel] = "replace"

        result: dict[str, Any] = {
            "v": _DELTA_VERSION,
            "id": checkpoint["id"],
            "ts": checkpoint["ts"],
            "is_full_snapshot": False,
            "delta_channel_values": delta_values,
            "delta_meta": delta_meta,
            "channel_versions": checkpoint.get("channel_versions", {}),
            "versions_seen": checkpoint.get("versions_seen", {}),
            "pending_sends": checkpoint.get("pending_sends", []),
        }
        return result

    def _is_summarization(
        self, new_cv: dict[str, Any], prev_cv: dict[str, Any]
    ) -> bool:
        """Detect summarization: messages were restructured, not just appended."""
        prev_msgs = prev_cv.get("messages", [])
        new_msgs = new_cv.get("messages", [])
        if not prev_msgs:
            return False
        if len(new_msgs) < len(prev_msgs):
            return True
        # First message replaced with summary
        if new_msgs and prev_msgs:
            try:
                if getattr(new_msgs[0], "content", None) != getattr(prev_msgs[0], "content", None):
                    return True
            except (AttributeError, TypeError):
                pass
        return False

    async def _get_prev_channel_values(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> dict[str, Any] | None:
        """Get channel_values from previous checkpoint (cached or reconstructed)."""
        cache_key = (thread_id, checkpoint_ns)
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]

        # Reconstruct from DB
        async with self.conn.execute(
            "SELECT type, checkpoint FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
            (thread_id, checkpoint_ns, checkpoint_id),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        cp = await self._reconstruct_checkpoint(
            thread_id, checkpoint_ns, checkpoint_id, row[0], row[1]
        )
        cv = cp.get("channel_values", {})
        self._state_cache[cache_key] = dict(cv)
        return cv

    async def _count_deltas_since_full(
        self, thread_id: str, checkpoint_ns: str, from_id: str
    ) -> int:
        """Count delta checkpoints since last full snapshot."""
        count = 0
        current_id: str | None = from_id
        while current_id:
            async with self.conn.execute(
                "SELECT parent_checkpoint_id, type, checkpoint FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_ns, current_id),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                break
            data = self.serde.loads_typed((row[1], row[2]))
            if data.get("v", 1) != _DELTA_VERSION or data.get("is_full_snapshot", False):
                break
            count += 1
            current_id = row[0]
        return count

    # ------------------------------------------------------------------
    # aput_writes — same as AsyncSqliteSaver (writes are already efficient)
    # ------------------------------------------------------------------

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            type_str, serialized = self.serde.dumps_typed(value)
            await self.conn.execute(
                "INSERT OR IGNORE INTO writes "
                "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type_str, serialized),
            )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # alist — list checkpoints
    # ------------------------------------------------------------------

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        query = (
            "SELECT checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
        )
        params: list[Any] = [thread_id, checkpoint_ns]

        if before:
            before_id = before["configurable"].get("checkpoint_id")
            if before_id:
                query += "AND checkpoint_id < ? "
                params.append(before_id)

        query += "ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        for (cp_id,) in rows:
            cp_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cp_id,
                }
            }
            result = await self.aget_tuple(cp_config)
            if result:
                yield result

    # ------------------------------------------------------------------
    # get_next_version — monotonic version strings
    # ------------------------------------------------------------------

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        import random

        if current is None:
            current_v = 0
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()  # noqa: S311
        return f"{next_v:032}.{next_h:016}"

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    async def adelete_thread(self, thread_id: str) -> None:
        await self.setup()
        await self.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        await self.conn.execute(
            "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
        )
        await self.conn.execute(
            "DELETE FROM thread_metadata WHERE thread_id = ?", (thread_id,)
        )
        await self.conn.commit()

    # Sync stubs (required by BaseCheckpointSaver but we only use async)
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError("Use aget_tuple")

    def list(self, config: RunnableConfig | None, **kwargs: Any) -> Any:
        raise NotImplementedError("Use alist")

    def put(self, config: RunnableConfig, checkpoint: Checkpoint,
            metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        raise NotImplementedError("Use aput")

    def put_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]],
                   task_id: str, task_path: str = "") -> None:
        raise NotImplementedError("Use aput_writes")
