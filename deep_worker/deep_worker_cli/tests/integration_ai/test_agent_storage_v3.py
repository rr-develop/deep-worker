from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from deep_worker_cli.session.session_manager import SessionManager
from deep_worker_cli.storage_paths import STORAGE_VERSION_AGENT_ROOT

from deep_worker_cli.storage_paths import deep_worker_home
def _run_deep_worker(*args: str, home: Path, timeout: int = 20) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    import pwd

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    env["HOME"] = str(home)
    env["PYTHONUSERBASE"] = str(real_home / ".local")
    return subprocess.run(
        ["deep-worker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_session_manager_creates_v3_metadata_with_private_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = deep_worker_home(tmp_path) / "integ_v3_agent"

    SessionManager(root, launch_config={"provider": "codex"})

    metadata_path = root / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["storage_version"] == STORAGE_VERSION_AGENT_ROOT
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert oct(metadata_path.stat().st_mode & 0o777) == "0o600"


def test_deep_worker_list_reads_v3_threads_db_without_tm_root(tmp_path: Path) -> None:
    agent = "integ_v3_list"
    root = deep_worker_home(tmp_path) / agent
    root.mkdir(parents=True, mode=0o700)
    (root / "session_metadata.json").write_text(
        json.dumps({"status": "idle", "storage_version": STORAGE_VERSION_AGENT_ROOT})
    )
    os.chmod(root / "session_metadata.json", 0o600)

    db = root / "threads.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO checkpoints VALUES (?, ?, ?)",
        (
            f"{agent}:tm",
            "1",
            json.dumps({"agent_name": agent, "updated_at": "2026-05-28T00:00:00"}),
        ),
    )
    conn.commit()
    conn.close()
    os.chmod(db, 0o600)

    result = _run_deep_worker("--dev", "--list", "--agent", agent, home=tmp_path)

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"{agent}:tm" in result.stdout
    assert not (deep_worker_home(tmp_path) / f"{agent}:tm").exists()
