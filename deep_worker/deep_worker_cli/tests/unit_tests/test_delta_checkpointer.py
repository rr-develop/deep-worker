"""Tests for DeltaSqliteCheckpointer — delta-based checkpoint storage."""

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer


def _make_checkpoint(messages, parent_id=None, updated_channels=None):
    """Create a minimal checkpoint dict."""
    cp_id = uuid.uuid4().hex[:16]
    return {
        "v": 1,
        "id": cp_id,
        "ts": "2026-04-19T12:00:00",
        "channel_values": {"messages": list(messages), "working_dir": "/tmp"},
        "channel_versions": {"messages": f"{len(messages):032}.0"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": updated_channels or ["messages"],
    }, cp_id


@pytest.fixture
async def checkpointer(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with DeltaSqliteCheckpointer.from_conn_string(db_path) as cp:
        yield cp


class TestDeltaBasics:
    """Core delta checkpointing functionality."""

    @pytest.mark.asyncio
    async def test_root_checkpoint_is_full_snapshot(self, checkpointer):
        """First checkpoint (no parent) must be a full snapshot."""
        msgs = [HumanMessage(content="hello")]
        cp, cp_id = _make_checkpoint(msgs)

        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        result_config = await checkpointer.aput(config, cp, {}, {})

        # Retrieve and verify
        loaded = await checkpointer.aget_tuple(result_config)
        assert loaded is not None
        assert len(loaded.checkpoint["channel_values"]["messages"]) == 1
        assert loaded.checkpoint["channel_values"]["messages"][0].content == "hello"

    @pytest.mark.asyncio
    async def test_second_checkpoint_is_delta(self, checkpointer):
        """Second checkpoint should store only the delta."""
        msgs1 = [HumanMessage(content="hello")]
        cp1, cp1_id = _make_checkpoint(msgs1)

        config0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        config1 = await checkpointer.aput(config0, cp1, {}, {})

        # Second checkpoint adds AI response
        msgs2 = [HumanMessage(content="hello"), AIMessage(content="hi there")]
        cp2, cp2_id = _make_checkpoint(msgs2, parent_id=cp1_id, updated_channels=["messages"])

        config2 = await checkpointer.aput(config1, cp2, {}, {})

        # Verify reconstruction
        loaded = await checkpointer.aget_tuple(config2)
        assert loaded is not None
        messages = loaded.checkpoint["channel_values"]["messages"]
        assert len(messages) == 2
        assert messages[0].content == "hello"
        assert messages[1].content == "hi there"

    @pytest.mark.asyncio
    async def test_delta_does_not_duplicate_full_history(self, checkpointer):
        """Delta checkpoint must NOT contain full message history.

        The key benefit: after 10 checkpoints with growing history,
        total DB size should be much less than 10× full snapshot size.
        With full snapshots: size ≈ N × last_snapshot_size (quadratic).
        With deltas: size ≈ first_snapshot + N × small_delta (linear).
        """
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        all_msgs: list = []

        for i in range(10):
            all_msgs.append(HumanMessage(content=f"long message {i} " * 50))
            cp, _ = _make_checkpoint(
                all_msgs,
                parent_id=config["configurable"].get("checkpoint_id"),
                updated_channels=["messages"],
            )
            config = await checkpointer.aput(config, cp, {}, {})

        # Total DB size should be much less than 10 × last checkpoint size
        async with checkpointer.conn.execute(
            "SELECT SUM(LENGTH(checkpoint)) FROM checkpoints WHERE thread_id = 't1'"
        ) as cursor:
            total_size = (await cursor.fetchone())[0]

        # Size of full snapshot with ALL 10 messages (what last cp would be as full)
        full_cp, _ = _make_checkpoint(all_msgs)
        _, full_blob = checkpointer.serde.dumps_typed(full_cp)
        full_snapshot_size = len(full_blob)

        # With full snapshots: total ≈ 10 × full_size (each growing)
        # With deltas: total ≈ full_size + 9 × small_delta
        # We expect at least 3x savings
        full_snapshot_total = full_snapshot_size * 10  # theoretical worst case
        assert total_size < full_snapshot_total / 3, (
            f"Total DB ({total_size:,}) should be much less than "
            f"10× full snapshot ({full_snapshot_total:,})"
        )

    @pytest.mark.asyncio
    async def test_long_chain_reconstruction(self, checkpointer):
        """Reconstruct state from a chain of 10 deltas."""
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        all_msgs = []

        for i in range(10):
            if i % 2 == 0:
                all_msgs.append(HumanMessage(content=f"user msg {i}"))
            else:
                all_msgs.append(AIMessage(content=f"ai msg {i}"))

            cp, cp_id = _make_checkpoint(
                all_msgs,
                parent_id=config["configurable"].get("checkpoint_id"),
                updated_channels=["messages"],
            )
            config = await checkpointer.aput(config, cp, {}, {})

        # Verify full reconstruction
        loaded = await checkpointer.aget_tuple(config)
        assert loaded is not None
        messages = loaded.checkpoint["channel_values"]["messages"]
        assert len(messages) == 10
        assert messages[0].content == "user msg 0"
        assert messages[9].content == "ai msg 9"

    @pytest.mark.asyncio
    async def test_unchanged_channels_not_stored(self, checkpointer):
        """If only non-messages channels update, messages delta should be empty."""
        msgs = [HumanMessage(content="hello")]
        cp1, cp1_id = _make_checkpoint(msgs)
        config0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        config1 = await checkpointer.aput(config0, cp1, {}, {})

        # Second checkpoint: messages unchanged, only working_dir updated
        cp2, cp2_id = _make_checkpoint(msgs, parent_id=cp1_id, updated_channels=["working_dir"])
        cp2["channel_values"]["working_dir"] = "/tmp/new"
        config2 = await checkpointer.aput(config1, cp2, {}, {})

        # Verify messages preserved, working_dir updated
        loaded = await checkpointer.aget_tuple(config2)
        messages = loaded.checkpoint["channel_values"]["messages"]
        assert len(messages) == 1
        assert loaded.checkpoint["channel_values"]["working_dir"] == "/tmp/new"


class TestBackwardCompatibility:
    """Reading existing v1 (full snapshot) checkpoints."""

    @pytest.mark.asyncio
    async def test_reads_v1_checkpoints(self, checkpointer):
        """Must read legacy v1 checkpoints as full snapshots."""
        # Manually insert a v1 checkpoint (like AsyncSqliteSaver would)
        v1_data = {
            "v": 1,
            "id": "legacy001",
            "ts": "2026-01-01T00:00:00",
            "channel_values": {
                "messages": [HumanMessage(content="legacy msg")],
                "working_dir": "/old",
            },
            "channel_versions": {},
            "versions_seen": {},
        }
        type_str, blob = checkpointer.serde.dumps_typed(v1_data)
        await checkpointer.setup()
        await checkpointer.conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy", "", "legacy001", type_str, blob, b"{}"),
        )
        await checkpointer.conn.commit()

        config = {"configurable": {"thread_id": "legacy", "checkpoint_ns": "", "checkpoint_id": "legacy001"}}
        loaded = await checkpointer.aget_tuple(config)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"]["messages"][0].content == "legacy msg"


class TestSummarization:
    """Full snapshot on summarization."""

    @pytest.mark.asyncio
    async def test_summarization_triggers_full_snapshot(self, checkpointer):
        """When messages shrink (summarization), a full snapshot is created."""
        # Build up 5 messages
        msgs = [HumanMessage(content=f"msg {i}") for i in range(5)]
        cp1, cp1_id = _make_checkpoint(msgs)
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        config = await checkpointer.aput(config, cp1, {}, {})

        # Summarize: replace with 1 summary message
        summary_msgs = [HumanMessage(content="Summary of 5 messages")]
        cp2, cp2_id = _make_checkpoint(
            summary_msgs,
            parent_id=config["configurable"]["checkpoint_id"],
            updated_channels=["messages"],
        )
        config2 = await checkpointer.aput(config, cp2, {}, {})

        # Verify: should read correctly (full snapshot, not delta)
        loaded = await checkpointer.aget_tuple(config2)
        assert loaded is not None
        messages = loaded.checkpoint["channel_values"]["messages"]
        assert len(messages) == 1
        assert messages[0].content == "Summary of 5 messages"


class TestThreadMetadata:
    """Fast thread listing via thread_metadata table."""

    @pytest.mark.asyncio
    async def test_metadata_updated_on_put(self, checkpointer):
        """thread_metadata should be updated on every aput."""
        msgs = [HumanMessage(content="hello")]
        cp, _ = _make_checkpoint(msgs)
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        await checkpointer.aput(config, cp, {"agent_name": "test_agent"}, {})

        async with checkpointer.conn.execute(
            "SELECT agent_name, message_count, checkpoint_count FROM thread_metadata WHERE thread_id = 't1'"
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[0] == "test_agent"
        assert row[1] == 1  # 1 message
        assert row[2] == 1  # 1 checkpoint
