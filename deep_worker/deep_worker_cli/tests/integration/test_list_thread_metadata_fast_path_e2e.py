"""Integration (no mocks): `--list` reads thread_metadata fast.

Builds a realistic ``$HOME/.deep_worker/agents/`` tree with:

* two DBs written by the REAL ``DeltaSqliteCheckpointer`` (so they carry a
  live-maintained ``thread_metadata`` table), and
* one legacy DB hand-built with ONLY a ``checkpoints`` table (no
  ``thread_metadata``) — exercising the one-time backfill path.

Then drives the REAL ``list_threads_command`` (the exact function
``deep-worker --list`` runs) against that HOME — no patching of the listing
logic, no stubbed DB layer, real SQLite files end to end — and asserts:

1. every thread is listed (legacy DB included), and
2. the legacy DB is backfilled exactly once: a second ``--list`` does not
   rewrite the file (mtime stable → no redundant work).
"""

import argparse
import asyncio
import json
import sqlite3
import uuid
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_worker_cli.cli.agent_command import list_threads_command
from deep_worker_cli.delta_checkpointer import DeltaSqliteCheckpointer


def _checkpoint(messages):
    return {
        "v": 1,
        "id": uuid.uuid4().hex[:16],
        "ts": "2026-04-19T12:00:00",
        "channel_values": {"messages": list(messages), "working_dir": "/tmp"},
        "channel_versions": {"messages": f"{len(messages):032}.0"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["messages"],
    }


async def _write_real_db(db_path: Path, thread_id: str, agent_name: str) -> None:
    async with DeltaSqliteCheckpointer.from_conn_string(str(db_path)) as cp:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        c1 = await cp.aput(
            config, _checkpoint([HumanMessage(content="hi")]),
            {"agent_name": agent_name}, {},
        )
        await cp.aput(
            c1,
            _checkpoint([HumanMessage(content="hi"), AIMessage(content="yo")]),
            {"agent_name": agent_name},
            {},
        )


def _build_legacy_db(db_path: Path, thread_id: str, agent_name: str) -> None:
    """A pre-migration DB: checkpoints table only, NO thread_metadata."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
        """
    )
    meta = json.dumps(
        {"agent_name": agent_name, "updated_at": "2026-04-18T09:00:00+00:00"}
    ).encode()
    conn.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
        (thread_id, "cp1", meta),
    )
    conn.commit()
    conn.close()


def _run_list_json(home: Path) -> dict:
    """Invoke the real --list command with HOME pointed at our temp tree."""
    args = argparse.Namespace(
        list_threads=True,
        format="json",
        with_counts=False,
        agent=None,
        sort="thread_id",
        reverse=False,
    )
    buf = StringIO()
    # Only HOME is overridden — the listing logic, storage-path resolution and
    # SQLite access all run for real.
    with patch("deep_worker_cli.storage_paths.Path.home", return_value=home):
        with patch("sys.stdout", new=buf):
            exit_code = asyncio.run(list_threads_command(args))
    assert exit_code == 0, buf.getvalue()
    out = buf.getvalue()
    start = out.find("{")
    assert start != -1, f"no JSON in output: {out!r}"
    return json.loads(out[start:])


def _agent_db(home: Path, agent_name: str) -> Path:
    """The per-agent DB under the v3 layout: ``agents/<name>/threads.db``."""
    db = home / ".deep_worker" / "agents" / agent_name / "threads.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return db


@pytest.fixture
def home_with_dbs(tmp_path) -> Path:
    asyncio.run(_write_real_db(_agent_db(tmp_path, "alpha"), "alpha", "alpha"))
    asyncio.run(_write_real_db(_agent_db(tmp_path, "beta"), "beta", "beta"))
    _build_legacy_db(_agent_db(tmp_path, "oldagent"), "oldagent", "oldagent")
    return tmp_path


def test_list_includes_all_threads_including_legacy(home_with_dbs):
    data = _run_list_json(home_with_dbs)
    tids = {t["thread_id"] for t in data["threads"]}
    assert {"alpha", "beta", "oldagent"} <= tids


def test_legacy_db_backfilled_once_and_idempotent(home_with_dbs):
    legacy = _agent_db(home_with_dbs, "oldagent")

    # First run backfills the table.
    _run_list_json(home_with_dbs)
    conn = sqlite3.connect(str(legacy))
    has_tm = conn.execute(
        "SELECT count(*) FROM sqlite_master "
        "WHERE type='table' AND name='thread_metadata'"
    ).fetchone()[0]
    rows = conn.execute("SELECT thread_id FROM thread_metadata").fetchall()
    conn.close()
    assert has_tm == 1
    assert [r[0] for r in rows] == ["oldagent"]

    mtime_after_first = legacy.stat().st_mtime_ns

    # Second run must NOT rewrite the file (no gap → no backfill write).
    _run_list_json(home_with_dbs)
    mtime_after_second = legacy.stat().st_mtime_ns
    assert mtime_after_second == mtime_after_first, (
        "legacy DB rewritten on second --list — backfill not idempotent"
    )
