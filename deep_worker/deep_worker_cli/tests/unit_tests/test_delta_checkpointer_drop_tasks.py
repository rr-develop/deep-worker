"""TDD regression: ``DeltaSqliteCheckpointer`` MUST NOT persist the
LangGraph-internal ``__pregel_tasks`` channel.

Root cause of the 2026-05-30 incident (see
docs/DESIGN_FIX_DELTA_CHECKPOINTER_DROP_TASKS.md): ``__pregel_tasks`` is an
ephemeral ``Topic[Send](accumulate=False)`` consumed within a single
super-step. Today ``_compute_delta`` special-cases only the ``messages``
channel and stores every other updated channel — including ``__pregel_tasks``
— wholesale on each delta. On a live agent this single channel grew the
checkpoint database to 1.6 GB (92% of weight) and ultimately corrupted the
SQLite B-tree.

These tests pin the invariants:

* the channel never reaches the delta payload;
* it never reaches a full-snapshot payload either;
* it never leaks via ``channel_versions`` / ``versions_seen`` (which are
  written whole-cloth on every delta);
* a fresh ``aput``→``aget_tuple`` round-trip reconstructs the rest of the
  state intact;
* the existing ``messages`` append-delta path is unaffected;
* legacy rows that DO contain ``__pregel_tasks`` are still readable
  (backward compatibility for already-persisted databases).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer

TASKS = "__pregel_tasks"


def _msg_versions(n: int) -> str:
    return f"{n:032}.0"


def _checkpoint(
    messages: list[Any],
    *,
    updated_channels: list[str] | None = None,
    extra_channel_values: dict[str, Any] | None = None,
    extra_channel_versions: dict[str, str] | None = None,
    extra_versions_seen: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], str]:
    cp_id = uuid.uuid4().hex[:16]
    cv = {"messages": list(messages), "working_dir": "/tmp"}
    cv.update(extra_channel_values or {})
    versions = {"messages": _msg_versions(len(messages))}
    versions.update(extra_channel_versions or {})
    seen = dict(extra_versions_seen or {})
    cp = {
        "v": 1,
        "id": cp_id,
        "ts": "2026-05-30T12:00:00",
        "channel_values": cv,
        "channel_versions": versions,
        "versions_seen": seen,
        "pending_sends": [],
        "updated_channels": updated_channels or ["messages"],
    }
    return cp, cp_id


def _serialised_payload(checkpointer: DeltaSqliteCheckpointer, row: tuple[Any, ...]) -> dict[str, Any]:
    """Decode the on-disk checkpoint payload using the checkpointer's own
    serde. Returns the dict that was actually written."""
    type_str, blob = row[0], row[1]
    return checkpointer.serde.loads_typed((type_str, blob))


@pytest.fixture
async def checkpointer(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with DeltaSqliteCheckpointer.from_conn_string(db_path) as cp:
        yield cp


class TestTasksNeverPersisted:
    @pytest.mark.asyncio
    async def test_tasks_excluded_from_delta_payload(self, checkpointer):
        """A delta checkpoint that lists TASKS in ``updated_channels`` MUST
        NOT serialise TASKS into ``delta_channel_values``."""
        msgs1 = [HumanMessage(content="hello")]
        cp1, _cp1_id = _checkpoint(msgs1)
        cfg0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        cfg1 = await checkpointer.aput(cfg0, cp1, {}, {})

        msgs2 = [HumanMessage(content="hello"), AIMessage(content="hi")]
        cp2, cp2_id = _checkpoint(
            msgs2,
            updated_channels=["messages", TASKS],
            extra_channel_values={TASKS: ["fake-send-object-that-would-be-huge"] * 3},
            extra_channel_versions={TASKS: "00000000000000000000000000000001.0"},
        )
        await checkpointer.aput(cfg1, cp2, {}, {})

        async with checkpointer.conn.execute(
            "SELECT type, checkpoint FROM checkpoints "
            "WHERE thread_id='t1' AND checkpoint_id=?",
            (cp2_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        payload = _serialised_payload(checkpointer, row)
        assert payload.get("is_full_snapshot") is False
        assert TASKS not in payload.get("delta_channel_values", {}), (
            "TASKS must NEVER be persisted in delta_channel_values — "
            "it is the LangGraph-internal ephemeral channel that bloated "
            "the checkpoint database to 1.6 GB (the 2026-05-30 incident)."
        )
        assert TASKS not in payload.get("delta_meta", {})

    @pytest.mark.asyncio
    async def test_tasks_excluded_from_full_snapshot_payload(self, checkpointer):
        """The ROOT checkpoint (full snapshot path) must also drop TASKS."""
        msgs = [HumanMessage(content="hello")]
        cp, cp_id = _checkpoint(
            msgs,
            updated_channels=["messages", TASKS],
            extra_channel_values={TASKS: ["fake-send"] * 3},
            extra_channel_versions={TASKS: "00000000000000000000000000000001.0"},
        )
        cfg0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        await checkpointer.aput(cfg0, cp, {}, {})

        async with checkpointer.conn.execute(
            "SELECT type, checkpoint FROM checkpoints "
            "WHERE thread_id='t1' AND checkpoint_id=?",
            (cp_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        payload = _serialised_payload(checkpointer, row)
        assert payload.get("is_full_snapshot") is True
        assert TASKS not in payload.get("channel_values", {}), (
            "TASKS must NEVER be persisted in a full snapshot either."
        )

    @pytest.mark.asyncio
    async def test_tasks_excluded_from_channel_versions_and_versions_seen(self, checkpointer):
        """``channel_versions`` and ``versions_seen`` are written wholesale on
        every delta — they must also exclude the TASKS key, otherwise
        TASKS still leaks (its version map keeps growing per super-step)."""
        msgs1 = [HumanMessage(content="hello")]
        cp1, _cp1_id = _checkpoint(msgs1)
        cfg0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        cfg1 = await checkpointer.aput(cfg0, cp1, {}, {})

        msgs2 = [HumanMessage(content="hello"), AIMessage(content="hi")]
        cp2, cp2_id = _checkpoint(
            msgs2,
            updated_channels=["messages", TASKS],
            extra_channel_values={TASKS: ["fake-send"] * 2},
            extra_channel_versions={TASKS: "00000000000000000000000000000001.0"},
            extra_versions_seen={"node-x": {TASKS: "00000000000000000000000000000001.0"}},
        )
        await checkpointer.aput(cfg1, cp2, {}, {})

        async with checkpointer.conn.execute(
            "SELECT type, checkpoint FROM checkpoints "
            "WHERE thread_id='t1' AND checkpoint_id=?",
            (cp2_id,),
        ) as cursor:
            row = await cursor.fetchone()
        payload = _serialised_payload(checkpointer, row)
        assert TASKS not in payload.get("channel_versions", {}), (
            "TASKS must not leak via channel_versions"
        )
        seen = payload.get("versions_seen", {})
        for node, vmap in seen.items():
            assert TASKS not in vmap, (
                f"TASKS must not leak via versions_seen (node {node!r})"
            )

    @pytest.mark.asyncio
    async def test_round_trip_reconstructs_user_state_without_tasks(self, checkpointer):
        """aput → aget_tuple reconstructs the user state intact; the
        reconstructed checkpoint contains no TASKS."""
        msgs1 = [HumanMessage(content="hello")]
        cp1, _ = _checkpoint(msgs1)
        cfg0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        cfg1 = await checkpointer.aput(cfg0, cp1, {}, {})

        msgs2 = [HumanMessage(content="hello"), AIMessage(content="hi")]
        cp2, _ = _checkpoint(
            msgs2,
            updated_channels=["messages", TASKS],
            extra_channel_values={TASKS: ["fake-send"] * 2},
            extra_channel_versions={TASKS: "00000000000000000000000000000001.0"},
        )
        cfg2 = await checkpointer.aput(cfg1, cp2, {}, {})

        loaded = await checkpointer.aget_tuple(cfg2)
        assert loaded is not None
        cv = loaded.checkpoint["channel_values"]
        assert cv["working_dir"] == "/tmp"
        assert len(cv["messages"]) == 2
        assert TASKS not in cv

    @pytest.mark.asyncio
    async def test_messages_append_delta_still_works(self, checkpointer):
        """Regression guard: the append-delta path for ``messages`` is
        unaffected by the TASKS skip."""
        msgs1 = [HumanMessage(content="hello")]
        cp1, _ = _checkpoint(msgs1)
        cfg0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        cfg1 = await checkpointer.aput(cfg0, cp1, {}, {})

        msgs2 = [HumanMessage(content="hello"), AIMessage(content="hi")]
        cp2, _ = _checkpoint(msgs2)
        cfg2 = await checkpointer.aput(cfg1, cp2, {}, {})

        loaded = await checkpointer.aget_tuple(cfg2)
        assert loaded is not None
        assert [m.content for m in loaded.checkpoint["channel_values"]["messages"]] == [
            "hello", "hi",
        ]

    @pytest.mark.asyncio
    async def test_pre_fix_row_with_tasks_still_reads(self, checkpointer):
        """Backward compat: a hand-crafted legacy row that DOES contain
        ``__pregel_tasks`` in its delta payload must still be readable
        (existing affected databases must not break)."""
        # 1. Real root checkpoint via the public path.
        msgs1 = [HumanMessage(content="hello")]
        cp1, cp1_id = _checkpoint(msgs1)
        cfg0 = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        cfg1 = await checkpointer.aput(cfg0, cp1, {}, {})

        # 2. Hand-write a pre-fix style child row that still has TASKS.
        cp2_id = uuid.uuid4().hex[:16]
        legacy_payload = {
            "v": 2,
            "id": cp2_id,
            "ts": "2026-05-30T12:00:01",
            "is_full_snapshot": False,
            "delta_channel_values": {
                "messages": [AIMessage(content="hi")],
                TASKS: ["legacy-send"],
            },
            "delta_meta": {"messages": "append", TASKS: "replace"},
            "channel_versions": {
                "messages": _msg_versions(2),
                TASKS: "00000000000000000000000000000001.0",
            },
            "versions_seen": {},
            "pending_sends": [],
        }
        type_str, blob = checkpointer.serde.dumps_typed(legacy_payload)
        await checkpointer.conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            " type, checkpoint, metadata) VALUES (?,?,?,?,?,?,?)",
            ("t1", "", cp2_id, cp1_id, type_str, blob, b"{}"),
        )
        await checkpointer.conn.commit()

        # 3. Read it back via the public reader — must succeed.
        cfg2 = {"configurable": {"thread_id": "t1", "checkpoint_ns": "", "checkpoint_id": cp2_id}}
        loaded = await checkpointer.aget_tuple(cfg2)
        assert loaded is not None
        cv = loaded.checkpoint["channel_values"]
        # Backward compat: messages came through, and the legacy TASKS payload
        # is preserved as-is on read (we just stop WRITING new ones).
        assert len(cv["messages"]) == 2


# ════════════════════════════════════════════════════════════════════════════
# The 2026-06-23 incident — second-half of the same fix, this time on the
# ``aput_writes`` ingress point. On a live agent the writes table reached
# 2.43 GB (96 % of threads.db) before this was caught — every super-step
# wrote a full ``__pregel_tasks`` payload via aput_writes, which the
# 2026-05-30 fix did NOT cover (it only touched ``aput`` / delta /
# full-snapshot / channel_versions / versions_seen). See
# a related internal incident analysis (not included here).
# ════════════════════════════════════════════════════════════════════════════


class TestAputWritesDropsTasks:
    """``aput_writes`` must filter ``__pregel_tasks`` the same way ``aput`` does.

    Pins the second half of the ephemeral-channel fix: the SQLite ``writes``
    table is the per-super-step side-channel LangGraph uses to stage tuples
    of ``(channel, value)`` before a checkpoint is committed; on a cold load
    (``aget_tuple``) these pending writes are merged into the reconstructed
    state. If ``__pregel_tasks`` leaks here, it rehydrates a 0.5–4 MB Send
    payload into the in-RAM state on every agent restart — which is exactly
    how the earlier symptom resurfaced on a live agent (2026-06-23).
    """

    @pytest.mark.asyncio
    async def test_tasks_excluded_from_aput_writes(self, checkpointer):
        """A mixed ``(TASKS, …)`` + ``(messages, …)`` writes batch must
        persist only the non-ephemeral entry.

        Reproduces the incident: a live agent had 4 596 ``__pregel_tasks``
        rows in the ``writes`` table totalling 2.43 GB.
        """
        # Arrange — a parent checkpoint so writes have something to attach to.
        cp, cp_id = _checkpoint([HumanMessage(content="hi")])
        cfg = await checkpointer.aput(
            {"configurable": {"thread_id": "t-aw", "checkpoint_ns": ""}},
            cp, {}, {},
        )

        # Act — a write batch carrying TASKS alongside a real channel.
        await checkpointer.aput_writes(
            cfg,
            writes=[
                (TASKS, [{"_send": "fake-send-payload"}]),  # ephemeral
                ("messages", [AIMessage(content="reply")]),  # must persist
            ],
            task_id="task-aw-1",
        )

        # Assert — TASKS absent, messages present.
        async with checkpointer.conn.execute(
            "SELECT channel FROM writes WHERE thread_id = ?", ("t-aw",),
        ) as cur:
            channels = sorted({row[0] async for row in cur})
        assert TASKS not in channels, (
            f"__pregel_tasks leaked into writes table; channels seen: {channels}"
        )
        assert "messages" in channels

    @pytest.mark.asyncio
    async def test_aput_writes_with_only_tasks_writes_nothing(
        self, checkpointer
    ):
        """A writes batch consisting solely of ephemeral entries inserts
        zero rows (not a row with a NULL value, not a placeholder)."""
        cp, _ = _checkpoint([])
        cfg = await checkpointer.aput(
            {"configurable": {"thread_id": "t-aw-solo", "checkpoint_ns": ""}},
            cp, {}, {},
        )

        await checkpointer.aput_writes(
            cfg,
            writes=[(TASKS, [{"_send": "x"}])],
            task_id="task-aw-solo",
        )

        async with checkpointer.conn.execute(
            "SELECT COUNT(*) FROM writes WHERE thread_id = ?", ("t-aw-solo",),
        ) as cur:
            (n,) = await cur.fetchone()
        assert n == 0, (
            f"aput_writes inserted {n} rows for an all-ephemeral batch; "
            "expected 0."
        )

    @pytest.mark.asyncio
    async def test_aput_writes_preserves_non_ephemeral_idx_uniqueness(
        self, checkpointer
    ):
        """Skipping ephemeral entries must NOT cause primary-key collisions
        on the surviving rows.

        Sanity-check: the writes table PK includes ``idx``, and the loop
        increments ``idx`` on the source list — skipped entries create
        gaps in ``idx``, but no collisions.
        """
        cp, _ = _checkpoint([])
        cfg = await checkpointer.aput(
            {"configurable": {"thread_id": "t-aw-mix", "checkpoint_ns": ""}},
            cp, {}, {},
        )

        await checkpointer.aput_writes(
            cfg,
            writes=[
                ("messages", [AIMessage(content="a")]),    # idx 0 → persisted
                (TASKS, [{"_send": "skip-1"}]),            # idx 1 → skipped
                ("working_dir", "/tmp"),                   # idx 2 → persisted
                (TASKS, [{"_send": "skip-2"}]),            # idx 3 → skipped
                ("todos", []),                             # idx 4 → persisted
            ],
            task_id="task-aw-mix",
        )

        async with checkpointer.conn.execute(
            "SELECT channel FROM writes WHERE thread_id = ? ORDER BY idx",
            ("t-aw-mix",),
        ) as cur:
            channels = [row[0] async for row in cur]
        assert channels == ["messages", "working_dir", "todos"], channels


# ════════════════════════════════════════════════════════════════════════════
# The 2026-06-23 incident — second-half of the same fix, this time on the
# ``aput_writes`` ingress point. On a live agent the writes table reached
# 2.43 GB (96 % of threads.db) before this was caught — every super-step
# wrote a full ``__pregel_tasks`` payload via aput_writes, which the
# 2026-05-30 fix did NOT cover (it only touched ``aput`` / delta /
# full-snapshot / channel_versions / versions_seen). See
# a related internal incident analysis (not included here).
# ════════════════════════════════════════════════════════════════════════════


class TestAputWritesDropsTasks:
    """``aput_writes`` must filter ``__pregel_tasks`` the same way ``aput`` does.

    Pins the second half of the ephemeral-channel fix: the SQLite ``writes``
    table is the per-super-step side-channel LangGraph uses to stage tuples
    of ``(channel, value)`` before a checkpoint is committed; on a cold load
    (``aget_tuple``) these pending writes are merged into the reconstructed
    state. If ``__pregel_tasks`` leaks here, it rehydrates a 0.5–4 MB Send
    payload into the in-RAM state on every agent restart — which is exactly
    how the earlier symptom resurfaced on a live agent (2026-06-23).
    """

    @pytest.mark.asyncio
    async def test_tasks_excluded_from_aput_writes(self, checkpointer):
        """A mixed ``(TASKS, …)`` + ``(messages, …)`` writes batch must
        persist only the non-ephemeral entry.

        Reproduces the incident: a live agent had 4 596 ``__pregel_tasks``
        rows in the ``writes`` table totalling 2.43 GB.
        """
        # Arrange — a parent checkpoint so writes have something to attach to.
        cp, cp_id = _checkpoint([HumanMessage(content="hi")])
        cfg = await checkpointer.aput(
            {"configurable": {"thread_id": "t-aw", "checkpoint_ns": ""}},
            cp, {}, {},
        )

        # Act — a write batch carrying TASKS alongside a real channel.
        await checkpointer.aput_writes(
            cfg,
            writes=[
                (TASKS, [{"_send": "fake-send-payload"}]),  # ephemeral
                ("messages", [AIMessage(content="reply")]),  # must persist
            ],
            task_id="task-aw-1",
        )

        # Assert — TASKS absent, messages present.
        async with checkpointer.conn.execute(
            "SELECT channel FROM writes WHERE thread_id = ?", ("t-aw",),
        ) as cur:
            channels = sorted({row[0] async for row in cur})
        assert TASKS not in channels, (
            f"__pregel_tasks leaked into writes table; channels seen: {channels}"
        )
        assert "messages" in channels

    @pytest.mark.asyncio
    async def test_aput_writes_with_only_tasks_writes_nothing(
        self, checkpointer
    ):
        """A writes batch consisting solely of ephemeral entries inserts
        zero rows (not a row with a NULL value, not a placeholder)."""
        cp, _ = _checkpoint([])
        cfg = await checkpointer.aput(
            {"configurable": {"thread_id": "t-aw-solo", "checkpoint_ns": ""}},
            cp, {}, {},
        )

        await checkpointer.aput_writes(
            cfg,
            writes=[(TASKS, [{"_send": "x"}])],
            task_id="task-aw-solo",
        )

        async with checkpointer.conn.execute(
            "SELECT COUNT(*) FROM writes WHERE thread_id = ?", ("t-aw-solo",),
        ) as cur:
            (n,) = await cur.fetchone()
        assert n == 0, (
            f"aput_writes inserted {n} rows for an all-ephemeral batch; "
            "expected 0."
        )

    @pytest.mark.asyncio
    async def test_aput_writes_preserves_non_ephemeral_idx_uniqueness(
        self, checkpointer
    ):
        """Skipping ephemeral entries must NOT cause primary-key collisions
        on the surviving rows.

        Sanity-check: the writes table PK includes ``idx``, and the loop
        increments ``idx`` on the source list — skipped entries create
        gaps in ``idx``, but no collisions.
        """
        cp, _ = _checkpoint([])
        cfg = await checkpointer.aput(
            {"configurable": {"thread_id": "t-aw-mix", "checkpoint_ns": ""}},
            cp, {}, {},
        )

        await checkpointer.aput_writes(
            cfg,
            writes=[
                ("messages", [AIMessage(content="a")]),    # idx 0 → persisted
                (TASKS, [{"_send": "skip-1"}]),            # idx 1 → skipped
                ("working_dir", "/tmp"),                   # idx 2 → persisted
                (TASKS, [{"_send": "skip-2"}]),            # idx 3 → skipped
                ("todos", []),                             # idx 4 → persisted
            ],
            task_id="task-aw-mix",
        )

        async with checkpointer.conn.execute(
            "SELECT channel FROM writes WHERE thread_id = ? ORDER BY idx",
            ("t-aw-mix",),
        ) as cur:
            channels = [row[0] async for row in cur]
        assert channels == ["messages", "working_dir", "todos"], channels
