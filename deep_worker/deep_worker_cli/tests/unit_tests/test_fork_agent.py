"""Unit tests for fork_agent_data()."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from deep_worker_cli.fork_agent import fork_agent_data


from deep_worker_cli.storage_paths import deep_worker_home
def _create_fake_source(src: Path) -> None:
    """Create a minimal fake agent data directory."""
    src.mkdir(parents=True)
    os.chmod(src, 0o700)

    # session_metadata.json
    meta = {
        "status": "running",
        "launch_config": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "working_dir": "/tmp",
            "skip_permissions": True,
            "task_manager": False,
            "pid": 12345,
            "started_at": "2026-01-01T00:00:00",
            "cli_args": "src_agent --provider deepseek",
            "storage_version": 5,
        },
        "error_message": None,
        "tasks_completed": 5,
        "message_count": 42,
        "last_activity": "2026-01-01T01:00:00",
    }
    meta_path = src / "session_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    os.chmod(meta_path, 0o600)

    # threads.db - minimal valid SQLite
    db_path = src / "threads.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.execute("INSERT INTO test VALUES (1)")
    conn.commit()
    conn.close()
    os.chmod(db_path, 0o600)

    # data/ directory with a file
    data_dir = src / "data"
    data_dir.mkdir(parents=True)
    os.chmod(data_dir, 0o700)
    (data_dir / "notes.txt").write_text("working data")
    os.chmod(data_dir / "notes.txt", 0o600)
    sub = data_dir / "subdir"
    sub.mkdir()
    os.chmod(sub, 0o700)
    (sub / "config.ini").write_text("key=value")
    os.chmod(sub / "config.ini", 0o600)

    # summaries/ directory with a jsonl file
    sum_dir = src / "summaries"
    sum_dir.mkdir()
    os.chmod(sum_dir, 0o700)
    sum_file = sum_dir / "thread_abc.jsonl"
    sum_file.write_text(
        '{"message_id":"msg1","message_index":0}\n'
        '{"message_id":"msg2","message_index":1}\n'
    )
    os.chmod(sum_file, 0o600)


class TestForkAgentData:
    def test_full_fork(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _create_fake_source(src)

        home = tmp_path / "home"
        # fork_agent_data resolves ~/.deep_worker/<name>/
        # So source must be at deep_worker_home(home)/src/
        deep_root = deep_worker_home(home)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)
        src_dw = deep_root / "src_agent"
        src.rename(src_dw)

        fork_agent_data("src_agent", "dst_agent", home=home)

        dst_dw = deep_root / "dst_agent"
        assert dst_dw.is_dir()
        assert (dst_dw / "data").is_dir()
        assert (dst_dw / "data" / "notes.txt").read_text() == "working data"
        assert (dst_dw / "summaries").is_dir()
        assert (dst_dw / "session_metadata.json").exists()
        assert (dst_dw / "threads.db").exists()

        meta = json.loads((dst_dw / "session_metadata.json").read_text())
        assert meta["launch_config"]["pid"] == os.getpid()
        assert meta["launch_config"]["started_at"] != "2026-01-01T00:00:00"
        assert meta["status"] == "idle"

        conn = sqlite3.connect(str(dst_dw / "threads.db"))
        result = conn.execute("SELECT id FROM test").fetchone()
        assert result == (1,)
        conn.close()

    def test_source_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            fork_agent_data("nonexistent", "dst_agent", home=tmp_path)

    def test_dst_already_exists(self, tmp_path: Path) -> None:
        deep_root = deep_worker_home(tmp_path)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)

        src_dw = deep_root / "src_agent"
        src_dw.mkdir()
        os.chmod(src_dw, 0o700)
        (src_dw / "session_metadata.json").write_text("{}")
        os.chmod(src_dw / "session_metadata.json", 0o600)

        dst_dw = deep_root / "dst_agent"
        dst_dw.mkdir()
        os.chmod(dst_dw, 0o700)

        with pytest.raises(FileExistsError):
            fork_agent_data("src_agent", "dst_agent", home=tmp_path)

    def test_no_data_dir(self, tmp_path: Path) -> None:
        deep_root = deep_worker_home(tmp_path)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)

        src_dw = deep_root / "src_agent"
        src_dw.mkdir()
        os.chmod(src_dw, 0o700)
        (src_dw / "session_metadata.json").write_text("{}")
        os.chmod(src_dw / "session_metadata.json", 0o600)

        fork_agent_data("src_agent", "dst_agent", home=tmp_path)

        dst_dw = deep_root / "dst_agent"
        assert dst_dw.is_dir()
        assert not (dst_dw / "data").exists()

    def test_no_summaries(self, tmp_path: Path) -> None:
        deep_root = deep_worker_home(tmp_path)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)

        src_dw = deep_root / "src_agent"
        src_dw.mkdir()
        os.chmod(src_dw, 0o700)
        (src_dw / "session_metadata.json").write_text("{}")
        os.chmod(src_dw / "session_metadata.json", 0o600)

        fork_agent_data("src_agent", "dst_agent", home=tmp_path)

        dst_dw = deep_root / "dst_agent"
        assert not (dst_dw / "summaries").exists()

    def test_no_threads_db(self, tmp_path: Path) -> None:
        deep_root = deep_worker_home(tmp_path)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)

        src_dw = deep_root / "src_agent"
        src_dw.mkdir()
        os.chmod(src_dw, 0o700)
        (src_dw / "session_metadata.json").write_text("{}")
        os.chmod(src_dw / "session_metadata.json", 0o600)

        fork_agent_data("src_agent", "dst_agent", home=tmp_path)

        dst_dw = deep_root / "dst_agent"
        assert not (dst_dw / "threads.db").exists()

    def test_no_session_metadata(self, tmp_path: Path) -> None:
        deep_root = deep_worker_home(tmp_path)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)

        src_dw = deep_root / "src_agent"
        src_dw.mkdir()
        os.chmod(src_dw, 0o700)

        fork_agent_data("src_agent", "dst_agent", home=tmp_path)

        dst_dw = deep_root / "dst_agent"
        assert dst_dw.is_dir()
        assert not (dst_dw / "session_metadata.json").exists()

    def test_permissions(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _create_fake_source(src)

        home = tmp_path / "home"
        deep_root = deep_worker_home(home)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)
        src_dw = deep_root / "src_agent"
        src.rename(src_dw)

        fork_agent_data("src_agent", "dst_agent", home=home)
        dst_dw = deep_root / "dst_agent"

        assert oct(os.stat(dst_dw).st_mode)[-3:] == "700"
        assert oct(os.stat(dst_dw / "data").st_mode)[-3:] == "700"
        mode_db = oct(os.stat(dst_dw / "threads.db").st_mode)[-3:]
        assert mode_db == "600", f"Expected 600, got {mode_db}"
        mode_meta = oct(os.stat(dst_dw / "session_metadata.json").st_mode)[-3:]
        assert mode_meta == "600", f"Expected 600, got {mode_meta}"

    # ── --from <message_id> tests ─────────────────────────────────────────

    def test_fork_with_message_id_happy_path(self, tmp_path: Path) -> None:
        """--from truncates summaries at the specified message_id."""
        src = tmp_path / "src"
        _create_fake_source(src)

        home = tmp_path / "home"
        deep_root = deep_worker_home(home)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)
        src.rename(deep_root / "src_agent")

        fork_agent_data("src_agent", "dst_agent", home=home, message_id="msg2")

        dst_dw = deep_root / "dst_agent"
        sum_file = dst_dw / "summaries" / "thread_abc.jsonl"
        assert sum_file.exists()
        lines = [l for l in sum_file.read_text().strip().splitlines() if l.strip()]
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {lines}"
        entries = [json.loads(l)["message_id"] for l in lines]
        assert entries == ["msg1", "msg2"]

    def test_fork_with_message_id_not_found(self, tmp_path: Path) -> None:
        """--from with nonexistent message_id raises ValueError."""
        src = tmp_path / "src"
        _create_fake_source(src)

        home = tmp_path / "home"
        deep_root = deep_worker_home(home)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)
        src.rename(deep_root / "src_agent")

        with pytest.raises(ValueError, match="not found"):
            fork_agent_data("src_agent", "dst_agent", home=home, message_id="nonexistent")

    def test_fork_with_message_id_no_summaries(self, tmp_path: Path) -> None:
        """--from when source has no summaries directory raises ValueError."""
        deep_root = deep_worker_home(tmp_path)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)

        src_dw = deep_root / "src_agent"
        src_dw.mkdir()
        os.chmod(src_dw, 0o700)
        (src_dw / "session_metadata.json").write_text("{}")
        os.chmod(src_dw / "session_metadata.json", 0o600)

        with pytest.raises(ValueError, match="has no summaries"):
            fork_agent_data("src_agent", "dst_agent", home=tmp_path, message_id="msg1")

    def test_fork_with_message_id_empty_summaries(self, tmp_path: Path) -> None:
        """--from when summaries dir is empty raises ValueError."""
        src = tmp_path / "src"
        src.mkdir(parents=True)
        os.chmod(src, 0o700)
        (src / "session_metadata.json").write_text("{}")
        os.chmod(src / "session_metadata.json", 0o600)
        sum_dir = src / "summaries"
        sum_dir.mkdir()
        os.chmod(sum_dir, 0o700)

        home = tmp_path / "home"
        deep_root = deep_worker_home(home)
        deep_root.mkdir(parents=True)
        os.chmod(deep_root, 0o700)
        src.rename(deep_root / "src_agent")

        with pytest.raises(ValueError, match="not found in any summary"):
            fork_agent_data("src_agent", "dst_agent", home=home, message_id="msg1")
