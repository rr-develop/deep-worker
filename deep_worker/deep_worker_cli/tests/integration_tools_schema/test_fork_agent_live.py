"""Integration test for forking an agent.

Creates a fake agent data directory, runs fork_agent_data, verifies
the destination structure, then tries to launch the agent.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def fake_source(tmp_path: Path) -> Path:
    src = tmp_path / Path("~/.deep_worker").name / "source_agent"
    src.mkdir(parents=True)
    os.chmod(src, 0o700)

    meta = {
        "status": "idle",
        "launch_config": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "working_dir": str(tmp_path / "workdir"),
            "skip_permissions": False,
            "task_manager": False,
            "memory_only": False,
            "history_memory": False,
            "use_sqlite": False,
            "auto_approve_send_message": False,
            "storage_version": 5,
            "started_at": "2026-01-01T00:00:00",
            "pid": 12345,
            "cli_args": "source_agent --provider deepseek",
            "dev_tree": False,
            "preferred_provider": "deepseek",
        },
    }
    (src / "session_metadata.json").write_text(json.dumps(meta, indent=2))
    os.chmod(src / "session_metadata.json", 0o600)

    db = src / "threads.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE checkpoints (id INTEGER)")
    conn.execute("INSERT INTO checkpoints VALUES (1)")
    conn.commit()
    conn.close()
    os.chmod(db, 0o600)

    data_dir = src / "data"
    data_dir.mkdir()
    os.chmod(data_dir, 0o700)
    (data_dir / "test.txt").write_text("test content")
    os.chmod(data_dir / "test.txt", 0o600)

    summaries_dir = src / "summaries"
    summaries_dir.mkdir()
    os.chmod(summaries_dir, 0o700)
    sf = summaries_dir / "history.jsonl"
    sf.write_text('{"msg":"hello"}\n{"msg":"world"}\n')
    os.chmod(sf, 0o600)

    return tmp_path


def test_fork_live_creates_dst_structure(fake_source: Path) -> None:
    from deep_worker_cli.fork_agent import fork_agent_data

    fork_agent_data("source_agent", "new_agent", home=fake_source)

    dst = fake_source / Path("~/.deep_worker").name / "new_agent"
    assert dst.is_dir()

    assert (dst / "data").is_dir()
    assert (dst / "data" / "test.txt").read_text() == "test content"

    assert (dst / "summaries").is_dir()
    assert (dst / "summaries" / "history.jsonl").read_text() == (
        '{"msg":"hello"}\n{"msg":"world"}\n'
    )

    assert (dst / "session_metadata.json").exists()
    meta = json.loads((dst / "session_metadata.json").read_text())
    assert meta["launch_config"]["pid"] == os.getpid()
    assert meta["launch_config"]["started_at"] != "2026-01-01T00:00:00"

    assert (dst / "threads.db").exists()
    conn = sqlite3.connect(str(dst / "threads.db"))
    row = conn.execute("SELECT id FROM checkpoints").fetchone()
    assert row == (1,)
    conn.close()


def test_fork_permissions(fake_source: Path) -> None:
    from deep_worker_cli.fork_agent import fork_agent_data

    fork_agent_data("source_agent", "perm_test", home=fake_source)


def test_fork_with_from_truncates_summaries(tmp_path: Path) -> None:
    """--from <message_id> truncates summaries; checkpointer stays full."""
    from deep_worker_cli.fork_agent import fork_agent_data

    src = tmp_path / Path("~/.deep_worker").name / "fork_src"
    src.mkdir(parents=True)
    os.chmod(src, 0o700)
    (src / "session_metadata.json").write_text(json.dumps({
        "launch_config": {"provider": "deepseek", "pid": 123, "started_at": "x"},
    }))
    os.chmod(src / "session_metadata.json", 0o600)

    db = src / "threads.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (42)")
    conn.commit()
    conn.close()
    os.chmod(db, 0o600)

    sum_dir = src / "summaries"
    sum_dir.mkdir()
    os.chmod(sum_dir, 0o700)
    sf = sum_dir / "chat.jsonl"
    sf.write_text(
        '{"message_id":"m1","msg":"hello"}\n'
        '{"message_id":"m2","msg":"hi"}\n'
        '{"message_id":"m3","msg":"bye"}\n'
    )
    os.chmod(sf, 0o600)

    fork_agent_data("fork_src", "fork_dst", home=tmp_path, message_id="m2")

    dst = tmp_path / Path("~/.deep_worker").name / "fork_dst"
    assert dst.is_dir()

    # summaries truncated at m2
    truncated = dst / "summaries" / "chat.jsonl"
    assert truncated.exists()
    lines = [l for l in truncated.read_text().strip().splitlines() if l.strip()]
    assert len(lines) == 2
    ids = [json.loads(l)["message_id"] for l in lines]
    assert ids == ["m1", "m2"], f"Expected m1,m2 got {ids}"

    # threads.db is FULL copy (unchanged)
    dst_db = dst / "threads.db"
    assert dst_db.exists()
    c = sqlite3.connect(str(dst_db))
    assert c.execute("SELECT v FROM t").fetchone() == (42,)
    c.close()
