#!/usr/bin/env python3
"""
Memory monitor for Python processes.

Monitors all python processes every 60 seconds.
If any python process exceeds 3 GB RSS, ensures the target agent is running,
sends an alert via dwcli to that agent asking for a memory investigation, then exits.

If the target agent is not running, restarts it via:
    deep-worker --restart <agent_name>

Usage:
    python3 tools/memory_monitor.py [--interval 60] [--alert-rss 3072] [--kill-rss 7168]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


class PyProcess(NamedTuple):
    pid: int
    rss_kb: int
    cmdline: str


# ── Helpers exposed for testing ─────────────────────────────────────────

def _agent_pid_from_registry(agent_name: str) -> int | None:
    """Read agent PID from /tmp/deep_worker_registry.json."""
    registry_path = Path("/tmp/deep_worker_registry.json")
    if not registry_path.exists():
        return None
    try:
        data = json.loads(registry_path.read_text())
        info = data.get(agent_name)
        if info:
            pid = info.get("pid")
            return int(pid) if pid else None
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        pass
    return None


def _is_pid_alive(pid: int) -> bool:
    """Check if PID exists in /proc."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Name:"):
                    return True
    except OSError:
        pass
    return False


# ── Public functions ────────────────────────────────────────────────────

def get_python_processes(proc_root: str = "/proc", own_pid: int | None = None) -> list[PyProcess]:
    """Return all python3 processes with RSS > 0, sorted by RSS descending."""
    results: list[PyProcess] = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        pid_str = entry.name
        try:
            with open(entry / "status") as f:
                status = f.read()
        except OSError:
            continue

        comm = ""
        rss_kb = 0
        for line in status.splitlines():
            if line.startswith("Name:"):
                comm = line.split(":", 1)[1].strip()
            elif line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])

        if "python" not in comm.lower() or rss_kb <= 0:
            continue

        pid_val = int(pid_str)
        if own_pid is not None and pid_val == own_pid:
            continue

        try:
            with open(entry / "cmdline", "rb") as f:
                raw = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            raw = f"[{comm}]"

        results.append(PyProcess(pid=pid_val, rss_kb=rss_kb, cmdline=raw))

    results.sort(key=lambda p: -p.rss_kb)
    return results


def send_alert(
    agent_name: str,
    procs: list[PyProcess],
    extra_info: str = "",
) -> bool:
    """Send an investigation request to the agent via dwcli."""
    lines: list[str] = [
        "MEMORY ALERT: Python process(es) exceeded 3GB RSS.",
        "Please investigate memory usage immediately.",
        "",
        "You have sudo access and all tools available.",
        "If a process grows beyond 7GB or your investigation is complete",
        "while the process keeps growing, kill it.",
        "Write the investigation report to this repo in a docs/ file.",
        "",
        "Affected processes:",
    ]
    for p in procs:
        gb = p.rss_kb / (1024 * 1024)
        lines.append(f"  PID {p.pid}: {gb:.1f} GB — {p.cmdline[:120]}")

    if extra_info:
        lines.append("")
        lines.append("Investigation data:")
        lines.append(extra_info)

    message = "\n".join(lines)
    try:
        result = subprocess.run(
            ["dwcli", agent_name, "-p", message],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_agent_alive(agent_name: str) -> bool:
    """Check if agent is alive via process registry. If not, restart via 'deep-worker --restart'."""
    # 1) Check registry + /proc
    pid = _agent_pid_from_registry(agent_name)
    if pid is not None and _is_pid_alive(pid):
        return True

    # 2) Agent not found — try restart
    print(f"Agent '{agent_name}' not found in registry. Attempting restart via deep-worker --restart ...")
    try:
        restart = subprocess.run(
            ["deep-worker", "--restart", agent_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if restart.returncode != 0:
            err = restart.stderr.strip()
            if "actively running" in err or "is actively running" in err:
                print(f"Agent '{agent_name}' is actively running — treating as alive.")
                return True
            print(f"Restart command failed: {err}")
            return False
    except FileNotFoundError:
        print("deep-worker command not found on PATH")
        return False
    except subprocess.TimeoutExpired:
        print("deep-worker --restart timed out")
        return False

    # 3) Wait for agent to come back (up to ~48s)
    for _ in range(6):
        time.sleep(8)
        pid = _agent_pid_from_registry(agent_name)
        if pid is not None and _is_pid_alive(pid):
            print(f"Agent '{agent_name}' is now alive (PID={pid}).")
            return True

    print(f"Agent '{agent_name}' still dead after restart.")
    return False


def investigate_pid(pid: int, proc_root: str = "/proc") -> str:
    """Gather a brief memory snapshot for a PID — returns human-readable string."""
    parts: list[str] = []

    try:
        with open(f"{proc_root}/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts.append(f"RSS: {line.split()[1]} kB")
                elif line.startswith("Threads:"):
                    parts.append(f"Threads: {line.split()[1]}")
    except OSError:
        return ""

    try:
        large_regions = 0
        total_anon_mb = 0
        with open(f"{proc_root}/{pid}/maps") as f:
            for line in f:
                if "rw-p" not in line:
                    continue
                fields = line.split()
                addrs = fields[0].split("-")
                sz = int(addrs[1], 16) - int(addrs[0], 16)
                path = fields[-1] if len(fields) > 5 else ""
                if path and path not in ("[heap]", "[stack]"):
                    continue
                total_anon_mb += sz
                if sz > 50 * 1048576:
                    large_regions += 1
        total_anon_mb //= 1048576
        parts.append(f"AnonRW: {total_anon_mb} MB")
        if large_regions:
            parts.append(f"LargeAnon(>50MB): {large_regions}")
    except OSError:
        pass

    heavy = ["torch", "tiktok", "huggingface", "lancedb", "pyarrow", "llama_index"]
    try:
        loaded = []
        with open(f"{proc_root}/{pid}/maps") as f:
            data = f.read()
        for h in heavy:
            if h in data:
                loaded.append(h)
        if loaded:
            parts.append(f"HeavyLibs: {', '.join(loaded)}")
    except OSError:
        pass

    return "; ".join(parts)


def main_loop(
    *,
    interval: int = 60,
    alert_rss_mb: int = 3072,
    kill_rss_mb: int = 7168,
    agent_name: str = "lc1_tools",
) -> None:
    """Main monitoring loop."""
    alert_rss_kb = alert_rss_mb * 1024
    kill_rss_kb = kill_rss_mb * 1024
    own_pid = os.getpid()

    if not ensure_agent_alive(agent_name):
        print(f"Cannot reach agent '{agent_name}' and restart failed. Exiting.")
        sys.exit(1)

    iteration = 0
    alerted = False

    print(f"Memory monitor started. Interval={interval}s, Alert at >={alert_rss_mb}MB, "
          f"Kill at >={kill_rss_mb}MB, Agent={agent_name}")

    while True:
        iteration += 1
        procs = get_python_processes(own_pid=own_pid)

        now = time.strftime("%H:%M:%S")
        summary = "; ".join(f"PID={p.pid} RSS={p.rss_kb // 1024}MB" for p in procs[:10])
        print(f"[{now}] Loop {iteration}: {summary}")

        if not procs:
            print("No python processes found. Exiting.")
            sys.exit(0)

        kill_list = sorted([p for p in procs if p.rss_kb > kill_rss_kb], key=lambda p: -p.rss_kb)
        alert_list = sorted(
            [p for p in procs if p.rss_kb > alert_rss_kb and p.rss_kb <= kill_rss_kb],
            key=lambda p: -p.rss_kb,
        )

        for proc in kill_list:
            gb = proc.rss_kb / (1024 * 1024)
            print(f"KILL: PID {proc.pid} at {gb:.1f} GB exceeds {kill_rss_mb} MB limit.")
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass

        if kill_list:
            sys.exit(0)

        if alert_list and not alerted:
            alerted = True
            info = investigate_pid(alert_list[0].pid)
            success = send_alert(agent_name, alert_list, extra_info=info)
            if success:
                print(f"Alert sent to '{agent_name}' for {len(alert_list)} process(es). Exiting.")
                sys.exit(0)
            else:
                print("Failed to send alert. Exiting.")
                sys.exit(1)

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor python process memory and alert agent.")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--alert-rss", type=int, default=3072)
    parser.add_argument("--kill-rss", type=int, default=7168)
    parser.add_argument("--agent", type=str, default="lc1_tools")
    args = parser.parse_args()
    main_loop(
        interval=args.interval,
        alert_rss_mb=args.alert_rss,
        kill_rss_mb=args.kill_rss,
        agent_name=args.agent,
    )


if __name__ == "__main__":
    main()
