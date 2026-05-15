"""Tests for memory_monitor.py — 19 tests covering all paths."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import tools.memory_monitor as mm


# =========================== get_python_processes ===========================


def _fake_proc_dir(entries: dict) -> Path:
    tmp = Path(tempfile.mkdtemp())
    for pid_str, info in entries.items():
        d = tmp / pid_str
        d.mkdir()
        comm = info.get("comm", "python3")
        rss = info.get("rss_kb", 5000)
        extra = info.get("extra_lines", "")
        (d / "status").write_text(f"Name:\t{comm}\nVmRSS:\t{rss} kB\n{extra}")
        cmdline = info.get("cmdline", f"/usr/bin/python3 {pid_str}.py")
        (d / "cmdline").write_bytes(cmdline.encode())
        if info.get("make_unreadable"):
            (d / "status").chmod(0o000)
    return tmp


def test_get_python_processes_finds_python() -> None:
    proc = _fake_proc_dir({"42": {"comm": "python3", "rss_kb": 100000}})
    procs = mm.get_python_processes(str(proc))
    assert len(procs) == 1
    assert procs[0].pid == 42
    assert procs[0].rss_kb == 100000


def test_get_python_processes_skips_non_python() -> None:
    proc = _fake_proc_dir({
        "10": {"comm": "python3", "rss_kb": 100000},
        "20": {"comm": "bash"},
        "30": {"comm": "node"},
    })
    procs = mm.get_python_processes(str(proc))
    assert len(procs) == 1
    assert procs[0].pid == 10


def test_get_python_processes_multiple_returns_sorted() -> None:
    proc = _fake_proc_dir({
        "1": {"comm": "python3", "rss_kb": 10000},
        "2": {"comm": "python3", "rss_kb": 500000},
        "3": {"comm": "python3", "rss_kb": 200000},
    })
    procs = mm.get_python_processes(str(proc))
    assert [p.pid for p in procs] == [2, 3, 1]


def test_get_python_processes_handles_missing_cmdline() -> None:
    proc = _fake_proc_dir({"7": {"comm": "python3", "rss_kb": 30000}})
    (proc / "7" / "cmdline").unlink()
    procs = mm.get_python_processes(str(proc))
    assert len(procs) == 1
    assert "[python3]" in procs[0].cmdline


def test_get_python_processes_handles_permission_error() -> None:
    proc = _fake_proc_dir({"99": {"comm": "python3", "rss_kb": 12345, "make_unreadable": True}})
    procs = mm.get_python_processes(str(proc))
    assert len(procs) == 0


def test_get_python_processes_ignores_self() -> None:
    proc = _fake_proc_dir({
        "100": {"comm": "python3", "rss_kb": 50000, "cmdline": "python3 tools/memory_monitor.py"},
        "200": {"comm": "python3", "rss_kb": 90000, "cmdline": "/usr/bin/python3 other.py"},
    })
    procs = mm.get_python_processes(str(proc), own_pid=100)
    assert len(procs) == 1
    assert procs[0].pid == 200


# ================================ send_alert ================================


def test_send_alert_success() -> None:
    with patch("tools.memory_monitor.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok")
        result = mm.send_alert("test-agent", [mm.PyProcess(pid=1, rss_kb=5000000, cmdline="leaky.py")])
        assert result is True
        assert mock_run.called
        args_str = " ".join(mock_run.call_args[0][0])
        assert "dwcli" in args_str
        assert "test-agent" in args_str


def test_send_alert_dwcli_not_found() -> None:
    with patch("tools.memory_monitor.subprocess.run", side_effect=FileNotFoundError):
        result = mm.send_alert("test-agent", [mm.PyProcess(pid=1, rss_kb=5000000, cmdline="leaky.py")])
        assert result is False


def test_send_alert_timeout() -> None:
    with patch("tools.memory_monitor.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="", timeout=30)):
        result = mm.send_alert("test-agent", [mm.PyProcess(pid=1, rss_kb=5000000, cmdline="leaky.py")])
        assert result is False


def test_send_alert_failure_code() -> None:
    with patch("tools.memory_monitor.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stderr="error")
        result = mm.send_alert("test-agent", [mm.PyProcess(pid=1, rss_kb=5000000, cmdline="leaky.py")])
        assert result is False


def test_send_alert_includes_extra_info() -> None:
    with patch("tools.memory_monitor.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok")
        result = mm.send_alert(
            "test-agent",
            [mm.PyProcess(pid=1, rss_kb=5000000, cmdline="leaky.py")],
            extra_info="Threads: 12\nAnonRW: 4200 MB",
        )
        assert result is True
        args_str = " ".join(mock_run.call_args[0][0])
        assert "Threads: 12" in args_str
        assert "AnonRW: 4200 MB" in args_str


# ============================== investigate_pid ==============================


def test_investigate_pid_returns_info() -> None:
    tmp = Path(tempfile.mkdtemp())
    pid_dir = tmp / "123"
    pid_dir.mkdir()
    (pid_dir / "status").write_text("Name:\tpython3\nVmRSS:\t500000 kB\nThreads:\t8\n")
    maps_lines = [
        "00400000-00500000 r-xp ...\n",
        "00a00000-00b00000 rw-p 00000000 00:00 0\n",
        "00c00000-01000000 rw-p 00000000 00:00 0\n",
        "7f000000-7f010000 rw-p 00000000 00:00 0\n",
    ]
    (pid_dir / "maps").write_text("".join(maps_lines))
    info = mm.investigate_pid(123, proc_root=str(tmp))
    assert "Threads:" in info
    assert "AnonRW:" in info


def test_investigate_pid_handles_missing_proc() -> None:
    info = mm.investigate_pid(99999, proc_root="/nonexistent")
    assert info == ""


# ============================== ensure_agent_alive ===========================


def test_ensure_agent_alive_already_ready() -> None:
    with patch("tools.memory_monitor._agent_pid_from_registry", return_value=12345), \
         patch("tools.memory_monitor._is_pid_alive", return_value=True):
        result = mm.ensure_agent_alive("test-agent")
        assert result is True


def test_ensure_agent_alive_dead() -> None:
    with patch("tools.memory_monitor._agent_pid_from_registry", return_value=None), \
         patch("tools.memory_monitor._is_pid_alive", return_value=False), \
         patch("tools.memory_monitor.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stderr="not ready")
        result = mm.ensure_agent_alive("test-agent")
        assert result is False


def test_ensure_agent_alive_restart_not_found() -> None:
    with patch("tools.memory_monitor._agent_pid_from_registry", return_value=None), \
         patch("tools.memory_monitor._is_pid_alive", return_value=False), \
         patch("tools.memory_monitor.subprocess.run", side_effect=FileNotFoundError):
        result = mm.ensure_agent_alive("test-agent")
        assert result is False


# ================================ main_loop =================================


@patch("tools.memory_monitor.send_alert", return_value=True)
@patch("tools.memory_monitor.ensure_agent_alive", return_value=True)
def test_main_loop_finds_and_alerts(mock_alive: MagicMock, mock_alert: MagicMock) -> None:
    procs = [mm.PyProcess(pid=42, rss_kb=int(3.5 * 1024 * 1024), cmdline="big.py")]
    with patch("tools.memory_monitor.get_python_processes", return_value=procs), \
         patch("tools.memory_monitor.investigate_pid", return_value="Threads: 4"):
        with pytest.raises(SystemExit) as exc:
            mm.main_loop(interval=0, alert_rss_mb=3000, kill_rss_mb=7000, agent_name="test-agent")
        assert exc.value.code == 0
    mock_alert.assert_called_once()


@patch("tools.memory_monitor.send_alert", return_value=True)
@patch("tools.memory_monitor.ensure_agent_alive", return_value=True)
def test_main_loop_kills_over_7gb(mock_alive: MagicMock, mock_alert: MagicMock) -> None:
    procs = [mm.PyProcess(pid=9999, rss_kb=int(8.0 * 1024 * 1024), cmdline="huge.py")]
    killed: list[int] = []
    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)
    with patch("tools.memory_monitor.get_python_processes", return_value=procs), \
         patch("tools.memory_monitor.investigate_pid", return_value="Threads: 4"), \
         patch("tools.memory_monitor.os.kill", side_effect=fake_kill):
        with pytest.raises(SystemExit) as exc:
            mm.main_loop(interval=0, alert_rss_mb=3000, kill_rss_mb=7000, agent_name="test-agent")
        assert exc.value.code == 0
    assert killed == [9999]


def test_empty_process_list() -> None:
    with patch("tools.memory_monitor.get_python_processes", return_value=[]), \
         patch("tools.memory_monitor.ensure_agent_alive", return_value=True), \
         pytest.raises(SystemExit) as exc:
        mm.main_loop(interval=0, alert_rss_mb=3000, kill_rss_mb=7000, agent_name="test")
    assert exc.value.code == 0


@patch("tools.memory_monitor.ensure_agent_alive", return_value=False)
def test_main_loop_agent_dead_exits(mock_alive: MagicMock) -> None:
    with pytest.raises(SystemExit) as exc:
        mm.main_loop(interval=0, alert_rss_mb=3000, kill_rss_mb=7000, agent_name="test-dead")
    assert exc.value.code == 1


# ============================== CLI entry point =============================


@patch("tools.memory_monitor.main_loop")
def test_cli_defaults(mock_loop: MagicMock) -> None:
    with patch.object(sys, "argv", ["memory_monitor.py"]):
        mm.main()
    mock_loop.assert_called_once_with(interval=60, alert_rss_mb=3072, kill_rss_mb=7168, agent_name="lc1_tools")


@patch("tools.memory_monitor.main_loop")
def test_cli_custom(mock_loop: MagicMock) -> None:
    with patch.object(sys, "argv", [
        "memory_monitor.py",
        "--interval", "30",
        "--alert-rss", "2000",
        "--kill-rss", "5000",
        "--agent", "test-agent",
    ]):
        mm.main()
    mock_loop.assert_called_once_with(interval=30, alert_rss_mb=2000, kill_rss_mb=5000, agent_name="test-agent")
