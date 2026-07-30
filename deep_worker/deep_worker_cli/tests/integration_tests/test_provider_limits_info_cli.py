from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from pathlib import Path

_CLI_ROOT = Path(__file__).resolve().parents[2]
_DEEP_WORKER_ROOT = _CLI_ROOT.parent
_TREE_ROOT = _DEEP_WORKER_ROOT.parent


def _env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PYTHONPATH"] = ":".join([
        str(_CLI_ROOT),
        str(_DEEP_WORKER_ROOT / "deepagents"),
        str(_TREE_ROOT),
        "/path/to/rpyc_async",
        "/path/to/claude_sdk/src",
        site.getusersitepackages(),
        env.get("PYTHONPATH", ""),
    ])
    return env


def test_limits_info_reads_persisted_provider_availability(tmp_path: Path) -> None:
    data_dir = tmp_path / Path("~/.deep_worker").name
    data_dir.mkdir()
    (data_dir / "provider_availability.json").write_text(json.dumps({
        "grok": {
            "available_after_utc": "2099-07-05T14:44:51+00:00",
            "available_after_source": "fallback",
            "available_after_note": "No provider reset time was parseable; using 24h rough fallback for grok.",
            "last_failure_marker": "Payment Required",
            "last_failure_at_utc": "2026-07-04T14:44:51+00:00",
            "last_failure_agent": "testagent",
        }
    }))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_worker_cli.cli.agent_command",
            "--provider",
            "grok",
            "--limits-info",
        ],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "provider: grok" in proc.stdout
    assert "status: limited" in proc.stdout
    assert "available_after_utc: 2099-07-05T14:44:51+00:00" in proc.stdout
    assert "source: fallback" in proc.stdout
    assert "rough fallback" in proc.stdout


def test_limits_info_empty_tracker_does_not_invent_reset_time(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_worker_cli.cli.agent_command",
            "--provider",
            "codex",
            "--limits-info",
        ],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    assert "provider: codex" in proc.stdout
    assert "status: no_limit_record" in proc.stdout
    assert "available_after_utc:" not in proc.stdout
