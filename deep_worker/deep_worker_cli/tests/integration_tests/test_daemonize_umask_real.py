"""Real-process integration test: ``daemonize()`` MUST leave the daemon
with a normal user umask (``0o022``) so that files/dirs created by the
agent — via ``execute``, ``write_file``, and every implicit-mode
``Path.mkdir(parents=True, exist_ok=True)`` — get sane permissions
(files 644, dirs 755), NOT the world-writable 666/777 you get when the
umask is 0.

Strategy (NO mocks):

* Fork a fresh Python subprocess.
* Inside the child, call the REAL ``deep_worker_cli.cli.agent_command.daemonize``.
* After daemonize returns (i.e. we're the surviving grandchild), the child
  writes its PID to a file, then touches a file and mkdir's a directory
  under a tmp path, then exits.
* The parent (test) reads:
    - the daemon's ``/proc/<pid>/status`` and asserts ``Umask: 0022``;
    - the tmp file / dir modes and asserts 644 / 755;
    - the log directory (auto-created by ``daemonize()`` itself, so it
      exercises the exact real path) mode is 755, not 777.

This test defends the fix documented in
``docs/DESIGN_FIX_DAEMONIZE_UMASK.md`` against regression. It uses only
public, non-mocked code paths and the real filesystem.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


_DW_CLI_ROOT = Path(__file__).resolve().parents[2]


def _read_proc_umask(pid: int) -> str | None:
    """Return the ``Umask:`` line value from ``/proc/<pid>/status`` (e.g.
    ``"0022"``), or ``None`` if the process is gone or the field is
    missing."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _mode_bits(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def _wait_for(condition, *, timeout: float, interval: float = 0.05) -> bool:
    """Block until ``condition()`` returns True or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Isolated per-test tmpdir. Under ``/tmp`` (world-writable parent) so
    that the daemon — which chdir's to ``/`` — can still write here through
    absolute paths."""
    d = tmp_path / "dw_umask_it"
    d.mkdir()
    return d


def _daemon_script(workdir: Path, log_dir: Path, pid_file: Path,
                   file_created: Path, dir_created: Path,
                   sentinel_file: Path) -> str:
    """Python source of the daemonizing helper we spawn as a subprocess."""
    return textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {str(_DW_CLI_ROOT)!r})
        from pathlib import Path
        from deep_worker_cli.cli import agent_command

        # Call the REAL daemonize; it double-forks, so only the surviving
        # grandchild continues past this line.
        agent_command.daemonize("test_umask_agent", log_dir=Path({str(log_dir)!r}))

        # We're now the daemonized grandchild. Prove it by writing our PID,
        # then create a file and a directory WITHOUT explicit mode — so the
        # umask is what determines their permissions.
        Path({str(pid_file)!r}).write_text(str(os.getpid()))
        with open({str(file_created)!r}, "w") as f:
            f.write("x")
        os.mkdir({str(dir_created)!r})
        Path({str(sentinel_file)!r}).write_text("done")

        # Sleep so the parent can inspect /proc/<pid>/status before we exit.
        time.sleep(5)
        """
    ).strip()


@pytest.mark.timeout(30)
def test_daemon_umask_and_created_permissions(workdir: Path) -> None:
    log_dir = workdir / "logs"
    pid_file = workdir / "daemon.pid"
    file_created = workdir / "created_file.txt"
    dir_created = workdir / "created_dir"
    sentinel_file = workdir / "sentinel.done"
    helper_script = workdir / "daemon_helper.py"

    helper_script.write_text(
        _daemon_script(workdir, log_dir, pid_file, file_created, dir_created,
                       sentinel_file)
    )

    # Portability guard: if the calling shell already runs under a strange
    # umask that our test can't undo (e.g. the whole box is misconfigured
    # to 0000 at the systemd-user level), skip — this test needs a real
    # user-shell baseline to be meaningful. We control the daemon umask
    # inside daemonize regardless of caller, so this guard is only about
    # avoiding false-negatives on exotic hosts.
    login_defs = Path("/etc/login.defs")
    if login_defs.exists():
        text = login_defs.read_text()
        want_022 = any(
            ln.strip().startswith("UMASK") and "022" in ln.split()
            for ln in text.splitlines()
        )
        if not want_022:
            pytest.skip("Host /etc/login.defs UMASK != 022; test not meaningful.")

    # Spawn the daemon. We call python3 directly; the launcher is orthogonal
    # to the umask code path. The subprocess double-forks so it returns fast.
    result = subprocess.run(
        [sys.executable, str(helper_script)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"Daemon spawn failed rc={result.returncode}\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )

    daemon_pid: int | None = None
    try:
        # Wait for the daemonized grandchild to write its PID + do the work.
        assert _wait_for(lambda: sentinel_file.exists(), timeout=10), (
            f"Daemon never wrote sentinel {sentinel_file}. Log dir contents: "
            f"{list(log_dir.glob('*')) if log_dir.exists() else 'no log dir'}"
        )
        daemon_pid = int(pid_file.read_text().strip())

        # === PRIMARY GUARD: /proc/<pid>/status Umask MUST be 0022 ===
        umask_str = _read_proc_umask(daemon_pid)
        assert umask_str == "0022", (
            f"/proc/{daemon_pid}/status Umask={umask_str!r}, expected '0022'. "
            f"Regression to os.umask(0) — see docs/DESIGN_FIX_DAEMONIZE_UMASK.md."
        )

        # === Files created without explicit mode → 0o644 ===
        assert file_created.exists()
        fm = _mode_bits(file_created)
        assert fm == 0o644, (
            f"File {file_created} mode = {oct(fm)}, expected 0o644 "
            f"(files must not be world-writable under normal umask)."
        )

        # === Dirs created without explicit mode → 0o755 ===
        assert dir_created.is_dir()
        dm = _mode_bits(dir_created)
        assert dm == 0o755, (
            f"Dir {dir_created} mode = {oct(dm)}, expected 0o755 "
            f"(dirs must not be world-writable under normal umask)."
        )

        # === Log directory auto-created by daemonize() itself → 0o755 ===
        # daemonize() does log_dir.mkdir(parents=True, exist_ok=True) BEFORE
        # the first fork, so it runs under the CALLER's umask (not the
        # 0o022 we set inside the daemon). 0o755 here is guaranteed by the
        # explicit ``log_dir.chmod(0o755)`` that daemonize() calls right
        # after that mkdir — this asserts that explicit chmod stays in
        # place and covers the pre-fork internal-mkdir class (same class
        # as the prod ':sub_*' directories that used to be 0o777).
        assert log_dir.is_dir()
        lm = _mode_bits(log_dir)
        assert lm == 0o755, (
            f"log_dir {log_dir} mode = {oct(lm)}, expected 0o755. "
            f"This covers internal non-private mkdir sites (e.g. ':sub_*') "
            f"that also inherit daemon umask."
        )

    finally:
        # Clean up: SIGKILL the sleeping daemon.
        if daemon_pid is not None:
            try:
                os.kill(daemon_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)
